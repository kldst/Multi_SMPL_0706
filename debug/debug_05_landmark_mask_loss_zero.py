#!/usr/bin/env python
"""Debug 05 -- GT-as-prediction => dense-landmark & mask losses collapse to ~0.

Builds the exact GT the dataset produces for a real 4-view frame (decode-free:
M @ vertices2d + instance mask), feeds it back in as the *prediction*, and checks
that the landmark GNLL and mask BCE go to ~0. This is the loss-side counterpart
of debug_04 and guards the loss plumbing (shapes, masking, normalisation).

Run:
  python debug/debug_05_landmark_mask_loss_zero.py \
      --seq-name be_HsuS3iLSSWWZ_seq_000001 --frame 0000 --num-views 4
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from mamma_raw_io import add_common_args, choose_frame_and_views, choose_sequence, load_gray
from training.data.landmark_mask_gt import (
    downsample_vertices,
    downsample_visibility,
    load_verts512_matrix,
    rasterize_person_patch_mask,
)
from training.loss import compute_landmark_loss, compute_mask_loss


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    size = args.image_size
    pg = size // args.patch_size
    rng = random.Random(args.seed)
    seq = choose_sequence(args.mamma_root, args.scene_root, args.seq_name, rng)
    frame, views = choose_frame_and_views(seq, args.frame, args.num_views, rng)
    print(f"[debug_05] seq={seq.name} frame={frame} views={views}")

    M = load_verts512_matrix()
    S, P = len(views), None

    lmk_list, vis_list, mask_list = [], [], []
    for view in views:
        pyd = pickle.load(open(seq / view / f"{frame}.data.pyd", "rb"))
        pids = sorted(pyd.keys(), key=lambda x: int(x))
        P = len(pids)
        mask_img = load_gray(seq / view / f"{frame}.mask.jpg")
        orig_h, orig_w = mask_img.shape[:2]
        sx, sy = size / orig_w, size / orig_h

        v_lmk = np.zeros((P, 512, 2), np.float32)
        v_vis = np.zeros((P, 512), np.float32)
        v_mask = np.zeros((P, pg, pg), np.float32)
        for i, pid in enumerate(pids):
            p = pyd[pid]
            lmk_px = downsample_vertices(M, np.asarray(p["vertices2d"], np.float32))
            lmk_px[:, 0] *= sx
            lmk_px[:, 1] *= sy
            # normalise to [0,1] (dataset convention) and clip visibility to in-frame
            v_lmk[i, :, 0] = lmk_px[:, 0] / size
            v_lmk[i, :, 1] = lmk_px[:, 1] / size
            vv = np.asarray(p["vertex_visibility"], np.float32).reshape(-1)
            vis = downsample_visibility(M, vv[None, :])[0]
            inframe = ((lmk_px[:, 0] >= 0) & (lmk_px[:, 0] < size)
                       & (lmk_px[:, 1] >= 0) & (lmk_px[:, 1] < size)).astype(np.float32)
            v_vis[i] = vis * inframe
            pv = int(p.get("person_idx", i)) + 1
            v_mask[i] = rasterize_person_patch_mask(mask_img, pv, pg, pg, image_size=size)
        lmk_list.append(v_lmk); vis_list.append(v_vis); mask_list.append(v_mask)

    # (S,P,...) -> (P,S,...) i.e. the (B*P, S, ...) layout with B=1.
    gt_lmk = torch.from_numpy(np.stack(lmk_list, 0)).permute(1, 0, 2, 3).contiguous()   # (P,S,512,2)
    gt_vis = torch.from_numpy(np.stack(vis_list, 0)).permute(1, 0, 2).contiguous()      # (P,S,512)
    gt_mask = torch.from_numpy(np.stack(mask_list, 0)).permute(1, 0, 2, 3).contiguous() # (P,S,pg,pg)
    has_smpl = torch.ones(P)

    # GT-as-prediction: landmarks == GT, logvar == 0 ; mask logits = (2*gt-1)*10.
    lmk_loss = compute_landmark_loss(
        pred_xy=gt_lmk.clone(), pred_logvar=torch.zeros_like(gt_vis),
        gt_xy=gt_lmk.clone(), visibility=gt_vis, has_smpl=has_smpl, loss_type="gnll",
    )
    # For a SOFT occupancy target, the BCE minimum is the target's binary entropy
    # (not 0), reached at logits = logit(gt). Verify the GT-as-pred loss hits that
    # floor and is far below an uninformative (zero-logit) baseline.
    mask_opt = float(compute_mask_loss(
        pred_logits=torch.logit(gt_mask.clamp(1e-4, 1 - 1e-4)),
        gt_mask=gt_mask, has_smpl=has_smpl)["loss_mask"])
    mask_zero = float(compute_mask_loss(
        pred_logits=torch.zeros_like(gt_mask), gt_mask=gt_mask, has_smpl=has_smpl)["loss_mask"])
    ent = -(gt_mask.clamp(1e-6, 1 - 1e-6) * gt_mask.clamp(1e-6, 1 - 1e-6).log()
            + (1 - gt_mask).clamp(1e-6, 1 - 1e-6) * (1 - gt_mask).clamp(1e-6, 1 - 1e-6).log())
    ent_floor = float((ent.flatten(1).mean(1)).mean())

    print(f"[debug_05] loss_landmark (GNLL, GT==pred) = {float(lmk_loss['loss_landmark']):.3e}  (expect 0)")
    print(f"[debug_05] loss_landmark_l2              = {float(lmk_loss['loss_landmark_l2']):.3e}")
    print(f"[debug_05] loss_mask  optimal={mask_opt:.3e}  entropy_floor={ent_floor:.3e}  zero_baseline={mask_zero:.3e}")

    ok = (float(lmk_loss["loss_landmark"]) < args.tolerance
          and abs(mask_opt - ent_floor) < 1e-3         # reaches the entropy floor
          and mask_opt < 0.5 * mask_zero)              # far below uninformative
    print("PASS" if ok else "FAIL",
          "(landmark~0; mask hits its entropy floor << zero-prediction baseline)")


if __name__ == "__main__":
    main()
