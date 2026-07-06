#!/usr/bin/env python
"""Gradio demo: predicted SMPL (3D + reprojection) + dense landmarks + per-person mask.

Loads a trained checkpoint (default: the mamma_overfit run), runs the model on a raw-Mamma
scene, and shows:
  * an interactive 3D scene (gr.Model3D / .glb) with the predicted SMPL-X meshes and the GT
    meshes in the same world frame (per-person colour; GT translucent grey);
  * per view: predicted & GT SMPL reprojected onto the image (GT world cameras), predicted &
    GT dense landmarks, predicted & GT patch masks, SMPL person-query attention, and dense
    landmark-query attention;
  * per-person metrics: SMPL vertex reprojection L2, landmark L2, mask IoU.

Predicted SMPL is placed with the mesh_translate head in the camera-0 gauge, then inverted
back to world so it can be reprojected with the GT cameras (tight, apples-to-apples overlay).

Launch:  CUDA_VISIBLE_DEVICES=1 ./run_demo_landmark_mask.sh
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import trimesh
import gradio as gr
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent
for p in (str(REPO), str(REPO / "training"), str(REPO / "debug")):
    if p not in sys.path:
        sys.path.insert(0, p)

from visualize_attention import build_model, CrossAttnCapture, select_attention, overlay_heatmap
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset
from training.smpl_body import (
    _decode_smplx_batch, _map_gender_value, _project_points_opencv, _SMPLX_MODEL_PATHS,
)
from training.train_utils.normalization import (
    normalize_camera_extrinsics_points_and_3djoints_batch,
)

PERSON_COLORS = [(255, 64, 64), (64, 160, 255), (64, 220, 120), (255, 200, 40)]
DEFAULT_SCENE = ("/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/"
                 "interactions_couple_close_1_C_200_00_contact/be_0GcH1mWtRfKu_seq_000050/"
                 "tmp/bedlam_lab_20251031_191434")
OUT_DIR = REPO / "debug_outputs/mamma_pipeline/08_demo"
# OpenCV (Y-down, Z-forward) -> upright for the 3D viewer.
_FLIP = np.diag([1.0, -1.0, -1.0])

MODEL = None
ARGS = None
_FACES = None


# ----------------------------------------------------------------------------- SMPL helpers
def smplx_faces():
    global _FACES
    if _FACES is None:
        import pickle
        with open(_SMPLX_MODEL_PATHS["neutral"], "rb") as f:
            _FACES = np.asarray(pickle.load(f, encoding="latin1")["f"], dtype=np.int64)
    return _FACES


def decode_people(pose, beta, trans, gender_ints):
    """(P,72),(P,10),(P,3)|None -> verts (P,V,3), joints (P,24,3) in world (np.float64)."""
    P = int(pose.shape[0])
    if P == 0:
        return np.zeros((0, 10475, 3)), np.zeros((0, 24, 3))
    dev = ARGS.device
    pose_t = torch.as_tensor(pose, dtype=torch.float32, device=dev)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=dev)
    trans_np = np.zeros((P, 3), np.float32) if trans is None else np.asarray(trans, np.float32)
    trans_t = torch.as_tensor(trans_np, dtype=torch.float32, device=dev)
    genders = [_map_gender_value(g) for g in gender_ints]
    joints, verts = _decode_smplx_batch(pose_t, beta_t, trans_t, genders)
    return verts.detach().cpu().numpy().astype(np.float64), joints.detach().cpu().numpy().astype(np.float64)


def avg_scale_from(gt_extr):
    """gt_extr (S,3,4) -> avg_scale float (cam0 gauge, matches training normalization)."""
    t = torch.as_tensor(gt_extr[None], dtype=torch.float32)
    out = normalize_camera_extrinsics_points_and_3djoints_batch(extrinsics=t, scale_by_extrinsics=True)
    return float(np.asarray(out[-1]).reshape(-1)[0])


def place_pred_world(pose, beta, mesh_translate, R0, t0, scale, gender_ints, use_mesh_rot=False):
    """Decode pred people (trans=0), gauge-normalize, re-anchor root to mesh_translate,
    then invert the cam0 gauge back to world so GT cameras can reproject them."""
    verts_raw, joints_raw = decode_people(pose, beta, None, gender_ints)   # canonical world
    P, V = verts_raw.shape[:2]
    gnorm = lambda X: ((X @ R0.T) + t0) / scale
    if use_mesh_rot:
        # In the trans-rot head, smpl_pose[:3] is mesh_rot: the root orientation
        # is already expressed in camera-0 coordinates.  Match the training/main
        # demo placement by applying scale only, not another world->cam0 rotation.
        vn = (verts_raw.reshape(-1, 3) / scale).reshape(P, V, 3)
        rn = joints_raw[:, 0, :] / scale
    else:
        vn = gnorm(verts_raw.reshape(-1, 3)).reshape(P, V, 3)
        rn = gnorm(joints_raw[:, 0, :])                                    # (P,3) root in gauge
    vg = vn + (np.asarray(mesh_translate).reshape(P, 3) - rn)[:, None, :]  # placed in gauge
    vw = ((vg.reshape(-1, 3) * scale) - t0) @ R0                           # gauge -> world
    return vw.reshape(P, V, 3)


def project(verts_world, extr, intr):
    """(V,3),(S,3,4),(S,3,3) -> (S,V,2) pixel coords via OpenCV projection."""
    S = extr.shape[0]
    pw = torch.as_tensor(verts_world[None, None], dtype=torch.float32).expand(1, S, -1, -1)
    E = torch.as_tensor(extr[None], dtype=torch.float32)
    K = torch.as_tensor(intr[None], dtype=torch.float32)
    return _project_points_opencv(pw, E, K)[0].numpy()


def draw_points(img, uv, color):
    u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
    H, W = img.shape[:2]
    for du, dv in ((0, 0), (1, 0), (0, 1)):
        uu, vv = u + du, v + dv
        m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        img[vv[m], uu[m]] = color
    return img


def build_glb(pred_verts, gt_verts, path):
    faces = smplx_faces()
    geoms = []
    for p, v in enumerate(gt_verts):
        vc = np.tile(np.array([205, 205, 205, 110], np.uint8), (v.shape[0], 1))
        geoms.append(trimesh.Trimesh(vertices=v @ _FLIP.T, faces=faces, vertex_colors=vc, process=False))
    for p, v in enumerate(pred_verts):
        c = PERSON_COLORS[p % 4]
        vc = np.tile(np.array([c[0], c[1], c[2], 255], np.uint8), (v.shape[0], 1))
        geoms.append(trimesh.Trimesh(vertices=v @ _FLIP.T, faces=faces, vertex_colors=vc, process=False))
    trimesh.Scene(geoms).export(str(path))
    return str(path)


# ----------------------------------------------------------------------------- data / inference
def _build_sample(scene, num_views, max_people):
    common = SimpleNamespace(
        img_size=ARGS.img_size, patch_size=ARGS.patch_size, augs=SimpleNamespace(scales=None),
        rescale=True, rescale_aug=False, landscape_check=False, debug=False, training=False,
        get_nearby=False, inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
    )
    ds = SysSMPLMultiDataset(
        common_conf=common, split="train", SysSMPL_DIR=scene, SysSMPL_ANNOTATION_DIR=scene,
        min_num_images=num_views, max_num_people=max_people,
        emit_landmarks=True, emit_person_mask=True, max_sequences=1, max_frames_per_sequence=1,
    )
    return ds.get_data(seq_index=0, img_per_seq=num_views, aspect_ratio=1.0)


def _select_attention_all_batches(captured, layer_arg):
    """Return attention without dropping the batch axis.

    ``visualize_attention.select_attention`` is perfect for the SMPL head because
    its batch axis is the real batch and this demo always uses batch item 0.  The
    dense-landmark head flattens (view, person) into its batch axis, so we need to
    preserve all of it: [B, heads, queries, context].
    """
    stack = torch.stack(captured, dim=0)  # [layers, B, heads, queries, context]
    if layer_arg == "mean":
        return stack.mean(dim=0)
    layer_idx = len(captured) - 1 if layer_arg == "last" else int(layer_arg)
    return stack[layer_idx]


@torch.no_grad()
def run(scene, num_views, max_people):
    """Return (glb_path, gallery items, metrics markdown)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        batch = _build_sample(scene, int(num_views), int(max_people))
    except Exception as e:
        return None, [], f"**Could not load scene**\n\n`{e}`"

    imgs = np.stack(batch["images"]).astype(np.float32)        # (S,H,W,3) RGB [0,255]
    S, H, W = imgs.shape[:3]
    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).div(255.0).to(ARGS.device)
    P = int(np.asarray(batch["num_people"]))
    gt_lmk = np.asarray(batch["smpl_landmarks2d"]); gt_vis = np.asarray(batch["smpl_landmarks2d_visibility"])
    gt_mask = np.asarray(batch["person_mask"])
    gt_extr = np.stack(batch["extrinsics"]).astype(np.float64)  # (S,3,4) world->cam
    gt_intr = np.stack(batch["intrinsics"]).astype(np.float64)  # (S,3,3)
    gt_pose = np.asarray(batch["smpl_pose"]); gt_beta = np.asarray(batch["smpl_beta"])
    gt_trans = np.asarray(batch["smpl_trans"]); gt_gender = np.asarray(batch["smpl_gender"])

    decoder_tf = MODEL.smpl_multi_query_trans_rot_head.decoder.transformer.transformer
    landmark_tf = MODEL.smpl_dense_landmark_head.transformer.transformer
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with CrossAttnCapture(decoder_tf) as smpl_cap:
            with CrossAttnCapture(landmark_tf) as lmk_cap:
                pred = MODEL(images)
    pr_lmk = pred["smpl_landmarks2d"][0].float().cpu().numpy()
    pr_mask = torch.sigmoid(pred["person_mask_logits"][0].float()).cpu().numpy()
    pr_pose = pred["smpl_pose"][0].float().cpu().numpy()        # (Pslots,72)
    pr_beta = pred["smpl_beta"][0].float().cpu().numpy()        # (Pslots,10)
    pr_mtr = pred["mesh_translate"][0].float().cpu().numpy()    # (Pslots,3)
    use_mesh_rot = pred.get("mesh_rot", None) is not None
    Pp = pr_lmk.shape[1]

    # ---- Hungarian: pred slot -> GT person (mean visible landmark L2) ----
    cost = np.full((Pp, P), 1e3, np.float32)
    for i in range(Pp):
        for j in range(P):
            v = gt_vis[:, j] > 0.5
            if v.any():
                cost[i, j] = np.linalg.norm(pr_lmk[:, i] - gt_lmk[:, j], axis=-1)[v].mean()
    row, col = linear_sum_assignment(cost)
    slot_for_gt = {int(j): int(i) for i, j in zip(row, col)}    # gt person j -> pred slot i

    # ---- SMPL: decode GT (world) + predicted (gauge->world), person-aligned ----
    scale = avg_scale_from(gt_extr)
    R0, t0 = gt_extr[0, :3, :3], gt_extr[0, :3, 3]
    order_slots = [slot_for_gt[j] for j in range(P)]            # pred slot per GT person
    order_gender = [gt_gender[j] for j in range(P)]
    gt_verts, _ = decode_people(gt_pose[:P], gt_beta[:P], gt_trans[:P], order_gender)
    pred_verts = place_pred_world(pr_pose[order_slots], pr_beta[order_slots], pr_mtr[order_slots],
                                  R0, t0, scale, order_gender, use_mesh_rot=use_mesh_rot)
    glb_path = build_glb(pred_verts, gt_verts, OUT_DIR / "scene.glb")

    # ---- attention grids ----
    # SMPL attention: one query per person slot attends to all view patch tokens.
    attn = select_attention(smpl_cap.captured, "mean").mean(dim=0)
    ph, pw = H // ARGS.patch_size, W // ARGS.patch_size
    attn_grid = attn.reshape(attn.shape[0], S, ph, pw).numpy()

    # Dense-landmark attention: the landmark head runs once per (view, person slot),
    # with 512 landmark queries attending to that view's patch grid. Collapse heads
    # and landmark ids to answer: "where did this person's 512 landmark queries look
    # in this view?"  Shape after capture: [S * Pslots, heads, 512, ph*pw].
    lmk_attn = _select_attention_all_batches(lmk_cap.captured, "mean")
    lmk_attn = lmk_attn.mean(dim=1)           # heads -> [S*Pslots, 512, ph*pw]
    lmk_attn = lmk_attn.mean(dim=1)           # landmarks -> [S*Pslots, ph*pw]
    lmk_attn_grid = lmk_attn.reshape(S, Pp, ph, pw).numpy()

    # ---- per-view overlays ----
    gt_uv = [project(gt_verts[j], gt_extr, gt_intr) for j in range(P)]     # each (S,V,2)
    pr_uv = [project(pred_verts[j], gt_extr, gt_intr) for j in range(P)]
    gallery = []
    for s in range(S):
        base = imgs[s].astype(np.uint8)
        gt_sm, pr_sm, gt_l, pr_l, gt_m, pr_m = (base.copy() for _ in range(6))
        for j in range(P):
            c = PERSON_COLORS[j % 4]; i = slot_for_gt[j]
            draw_points(gt_sm, gt_uv[j][s], c)
            draw_points(pr_sm, pr_uv[j][s], c)
            for (x, y), vv in zip(_denorm(gt_lmk[s, j], W, H), gt_vis[s, j]):
                if vv > 0.5 and 0 <= x < W and 0 <= y < H:
                    cv2.circle(gt_l, (int(x), int(y)), 2, c, -1, cv2.LINE_AA)
            for (x, y), vv in zip(_denorm(pr_lmk[s, i], W, H), gt_vis[s, j]):
                if vv > 0.5 and 0 <= x < W and 0 <= y < H:
                    cv2.circle(pr_l, (int(x), int(y)), 2, c, -1, cv2.LINE_AA)
            _tint(gt_m, gt_mask[s, j], c); _tint(pr_m, pr_mask[s, i], c)
        gallery += [(gt_sm, f"view{s} · GT SMPL reproj"), (pr_sm, f"view{s} · Pred SMPL reproj"),
                    (gt_l, f"view{s} · GT landmarks"), (pr_l, f"view{s} · Pred landmarks"),
                    (gt_m, f"view{s} · GT mask"), (pr_m, f"view{s} · Pred mask")]
        for j in range(P):
            heat = attn_grid[slot_for_gt[j], s]; heat = heat / (heat.max() + 1e-8)
            gallery.append((overlay_heatmap(base, heat, 0.6), f"view{s} · SMPL attn person {j}"))
        for j in range(P):
            heat = lmk_attn_grid[s, slot_for_gt[j]]
            heat = heat / (heat.max() + 1e-8)
            gallery.append((overlay_heatmap(base, heat, 0.6), f"view{s} · landmark attn person {j}"))

    # ---- metrics ----
    rows = ["| person | slot | SMPL reproj L2 (px) | landmark L2 (px) | mask IoU |",
            "|---|---|---|---|---|"]
    for j in range(P):
        i = slot_for_gt[j]; v = gt_vis[:, j] > 0.5
        sm_l2 = float(np.linalg.norm(np.stack(pr_uv[j]) - np.stack(gt_uv[j]), axis=-1).mean())
        lm_l2 = np.linalg.norm(pr_lmk[:, i] - gt_lmk[:, j], axis=-1)[v].mean() * W if v.any() else float("nan")
        gm = gt_mask[:, j] > 0.5; pm = pr_mask[:, i] > 0.5
        iou = (gm & pm).sum() / max((gm | pm).sum(), 1)
        rows.append(f"| {j} | {i} | {sm_l2:.1f} | {lm_l2:.1f} | {iou:.3f} |")
    rot_mode = "mesh_rot(cam0)" if use_mesh_rot else "world root rot"
    md = (f"**scene** `{Path(scene).name}` · **{P} people · {S} views** · "
          f"avg_scale `{scale:.3f}` · rot `{rot_mode}`\n\n"
          + "\n".join(rows)
          + "\n\n_3D: per-person colour = predicted SMPL, translucent grey = GT. "
          "Reprojections use the GT world cameras._")
    return glb_path, gallery, md


def _denorm(xy, W, H):
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] * W          # [0,1] -> pixels
    out[..., 1] = xy[..., 1] * H
    return out


def _tint(img, mask_hw, color, thr=0.5):
    up = cv2.resize((mask_hw > thr).astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    img[up] = (0.5 * img[up] + 0.5 * np.array(color, np.float32)).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------- UI
def build_ui():
    with gr.Blocks(title="SMPL + Landmark / Mask demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Predicted SMPL (3D + reprojection) + dense landmarks + per-person mask\n"
                    f"Checkpoint: `{ARGS.checkpoint}`")
        with gr.Row():
            scene = gr.Textbox(value=ARGS.scene, label="Scene dir (raw Mamma bedlam_lab_* root)", scale=4)
            nv = gr.Slider(1, 8, value=ARGS.num_views, step=1, label="views")
            mp = gr.Slider(1, ARGS.max_people, value=ARGS.max_people, step=1, label="max people")
        btn = gr.Button("Run inference", variant="primary")
        with gr.Row():
            model3d = gr.Model3D(label="3D scene · predicted (colour) vs GT (grey) SMPL", height=520)
            metrics = gr.Markdown()
        gallery = gr.Gallery(label="Per view: SMPL reproj · landmarks · mask · SMPL attention · landmark attention",
                             columns=4, height="auto", object_fit="contain")
        btn.click(run, [scene, nv, mp], [model3d, gallery, metrics])
        demo.load(run, [scene, nv, mp], [model3d, gallery, metrics])
    return demo


def main():
    global MODEL, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mamma_overfit")
    ap.add_argument("--checkpoint", default=str(REPO / "training/logs/mamma_overfit/ckpts/checkpoint_300.pt"))
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--max-people", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--share", action="store_true")
    ARGS = ap.parse_args()

    MODEL = build_model(ARGS.config, ARGS.checkpoint, ARGS.device)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=ARGS.port, share=ARGS.share,
                allowed_paths=[str(OUT_DIR)])


if __name__ == "__main__":
    main()
