#!/usr/bin/env python
"""Debug 07 -- run the overfit checkpoint and visualise PREDICTIONS.

Loads a trained checkpoint (default: the mamma_overfit run), runs the model on
the SAME scene/frame it overfit, and for every view produces:
  * GT   : landmarks + per-person mask (from the dataset)
  * PRED : landmarks (denormalised) + per-person mask (sigmoid of the logits)
  * ATTN : the SMPL head's per-person cross-attention over patches (reused from
           visualize_attention.CrossAttnCapture)
plus pred-vs-GT metrics (landmark L2 in px, mask IoU), Hungarian-matched slot->person.

Writes PNGs + a metrics.json under debug_outputs/mamma_pipeline/07_infer_overfit/.

Run:
  CUDA_VISIBLE_DEVICES=1 python debug/debug_07_infer_overfit_viz.py \
      --checkpoint training/logs/mamma_overfit/ckpts/checkpoint_300.pt
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visualize_attention import build_model, CrossAttnCapture, select_attention, overlay_heatmap
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset
from mamma_raw_io import save_rgb

PERSON_COLORS = [(255, 64, 64), (64, 160, 255), (64, 220, 120), (255, 200, 40)]
DEFAULT_SCENE = ("/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/"
                 "interactions_couple_close_1_C_200_00_contact/be_0GcH1mWtRfKu_seq_000050/"
                 "tmp/bedlam_lab_20251031_191434")


def build_sample(scene, num_views, max_people, img_size, patch_size):
    common = SimpleNamespace(
        img_size=img_size, patch_size=patch_size, augs=SimpleNamespace(scales=None),
        rescale=True, rescale_aug=False, landscape_check=False, debug=False, training=False,
        get_nearby=False, inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
    )
    ds = SysSMPLMultiDataset(
        common_conf=common, split="train", SysSMPL_DIR=scene, SysSMPL_ANNOTATION_DIR=scene,
        min_num_images=num_views, max_num_people=max_people,
        emit_landmarks=True, emit_person_mask=True, max_sequences=1, max_frames_per_sequence=1,
    )
    return ds.get_data(seq_index=0, img_per_seq=num_views, aspect_ratio=1.0)


def denorm(xy, W, H):
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] * W          # [0,1] -> pixels
    out[..., 1] = xy[..., 1] * H
    return out


def draw_landmarks(img, xy_px, vis, color):
    for (x, y), v in zip(xy_px, vis):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (int(x), int(y)), 2 if v > 0.5 else 1, color, -1, cv2.LINE_AA)
    return img


def tint_mask(img, mask_hw, color, thr=0.5):
    up = cv2.resize((mask_hw > thr).astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    img[up] = (0.5 * img[up] + 0.5 * np.array(color, np.float32)).astype(np.uint8)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="mamma_overfit")
    p.add_argument("--checkpoint", default=str(REPO / "training/logs/mamma_overfit/ckpts/checkpoint_300.pt"))
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--max-people", type=int, default=2)
    p.add_argument("--img-size", type=int, default=518)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=str(REPO / "debug_outputs/mamma_pipeline/07_infer_overfit"))
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.config, args.checkpoint, args.device)

    batch = build_sample(args.scene, args.num_views, args.max_people, args.img_size, args.patch_size)
    imgs_np = np.stack(batch["images"]).astype(np.float32)              # (S,H,W,3) [0,255]
    S, H, W = imgs_np.shape[:3]
    images = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).div(255.0).to(args.device)  # (S,3,H,W)

    P = int(np.asarray(batch["num_people"]))
    gt_lmk = np.asarray(batch["smpl_landmarks2d"])            # (S,P,512,2) normalised
    gt_vis = np.asarray(batch["smpl_landmarks2d_visibility"]) # (S,P,512)
    gt_mask = np.asarray(batch["person_mask"])               # (S,P,ph,pw)

    decoder_tf = model.smpl_multi_query_trans_rot_head.decoder.transformer.transformer
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with CrossAttnCapture(decoder_tf) as cap:
            pred = model(images)

    pr_lmk = pred["smpl_landmarks2d"][0].float().cpu().numpy()        # (S,P,512,2)
    pr_mask = torch.sigmoid(pred["person_mask_logits"][0].float()).cpu().numpy()  # (S,P,ph,pw)
    Pp = pr_lmk.shape[1]

    # ---- Hungarian match predicted slot -> GT person (mean landmark L2 over visible) ----
    cost = np.zeros((Pp, P), np.float32)
    for i in range(Pp):
        for j in range(P):
            v = gt_vis[:, j] > 0.5
            d = np.linalg.norm(pr_lmk[:, i] - gt_lmk[:, j], axis=-1)   # (S,512)
            cost[i, j] = d[v].mean() if v.any() else 1e3
    row, col = linear_sum_assignment(cost)
    slot_for_gt = {int(j): int(i) for i, j in zip(row, col)}          # gt person j -> pred slot i

    # ---- attention: [heads, people, S*P] -> [people, S, ph, pw] ----
    attn = select_attention(cap.captured, "mean").mean(dim=0)         # [people, S*P]
    ph = H // args.patch_size; pw = W // args.patch_size
    attn_grid = attn.reshape(attn.shape[0], S, ph, pw).numpy()

    # ---- per-view overlays + metrics ----
    metrics = {"per_person": {}, "num_views": S, "num_people": P}
    for s in range(S):
        base = imgs_np[s].astype(np.uint8)
        gt_img, pr_img = base.copy(), base.copy()
        gt_m, pr_m = base.copy(), base.copy()
        attn_tiles = []
        for j in range(P):
            c = PERSON_COLORS[j % len(PERSON_COLORS)]
            i = slot_for_gt[j]  # matched pred slot
            # landmarks
            draw_landmarks(gt_img, denorm(gt_lmk[s, j], W, H), gt_vis[s, j], c)
            # gate predicted landmarks by the same GT visibility for a fair overlay
            # (drawing all 512 incl. back-facing points collapses into strips).
            draw_landmarks(pr_img, denorm(pr_lmk[s, i], W, H), gt_vis[s, j], c)
            # masks
            tint_mask(gt_m, gt_mask[s, j], c)
            tint_mask(pr_m, pr_mask[s, i], c)
            # attention heatmap for the matched pred slot
            heat = attn_grid[i, s]
            heat = heat / (heat.max() + 1e-8)
            attn_tiles.append(overlay_heatmap(base, heat, alpha=0.6))
        save_rgb(out / f"view{s}_gt_landmarks.png", gt_img)
        save_rgb(out / f"view{s}_pred_landmarks.png", pr_img)
        save_rgb(out / f"view{s}_gt_mask.png", gt_m)
        save_rgb(out / f"view{s}_pred_mask.png", pr_m)
        for k, t in enumerate(attn_tiles):
            save_rgb(out / f"view{s}_attn_p{k}.png", t)

    # ---- global metrics (matched) ----
    for j in range(P):
        i = slot_for_gt[j]
        v = gt_vis[:, j] > 0.5
        d_norm = np.linalg.norm(pr_lmk[:, i] - gt_lmk[:, j], axis=-1)  # (S,512), normalised
        l2_px = (d_norm[v].mean()) * W if v.any() else float("nan")
        # mask IoU (binarised at 0.5)
        gm = gt_mask[:, j] > 0.5; pm = pr_mask[:, i] > 0.5
        inter = (gm & pm).sum(); union = (gm | pm).sum()
        iou = float(inter) / float(union) if union > 0 else float("nan")
        metrics["per_person"][f"gt_person_{j}"] = {
            "matched_pred_slot": i,
            "landmark_l2_norm": float(d_norm[v].mean()) if v.any() else None,
            "landmark_l2_px": float(l2_px),
            "mask_iou": iou,
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[debug_07] wrote overlays + metrics to", out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
