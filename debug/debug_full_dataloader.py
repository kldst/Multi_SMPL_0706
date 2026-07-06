#!/usr/bin/env python
"""Debug the mamma_full training dataloader.

What it does
------------
1. Builds the REAL SysSMPLMultiDataset exactly as ``mamma_full.yaml`` configures it
   (same common_config: img_nums / aspects / fixed_view_sampling / min_num_images /
   emit_landmarks / emit_person_mask ...), then pulls one sample via ``get_data`` --
   this is the method the dataloader calls per item, so it exercises the full
   crop/resize + intrinsic-update + projection path.
2. PRINTS every data path in the batch (per view), plus seq_name / ids / people.
3. VERIFIES reprojection is correct: projects the loader's ``smpl_joints3d_world``
   with the loader's (crop/resize-updated) ``extrinsics`` + ``intrinsics`` and
   compares against the loader's ``smpl_joints2d``. If the intrinsic transform is
   consistent with the 2D track transform, the in-frame error is sub-pixel.
4. Saves per-view overlays (GT joints green, reprojected red x, dense landmarks blue)
   so you can eyeball it.

Usage
-----
    conda activate mamma
    # fast smoke test (bounds the dataset BUILD to a few sequences):
    python debug/debug_full_dataloader.py --max-sequences 3 --num-samples 2
    # a specific sample:
    python debug/debug_full_dataloader.py --max-sequences 50 --seq-index 10

``--max-sequences`` / ``--max-frames`` only limit how much of the tree is scanned at
build time (the full build reads a pyd per (frame,view) and is slow); everything else
matches the training config.
"""
import argparse
import inspect
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of debug/)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
os.chdir(REPO)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import cv2  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def project_opencv(points_world, extrinsic, intrinsic, eps=1e-6):
    """points_world (...,3), extrinsic (3,4) cam-from-world, intrinsic (3,3) -> (...,2)."""
    R = np.asarray(extrinsic, np.float64)[:3, :3]
    t = np.asarray(extrinsic, np.float64)[:3, 3]
    K = np.asarray(intrinsic, np.float64)
    Xc = points_world @ R.T + t                      # (...,3) camera coords
    z = np.clip(Xc[..., 2:3], eps, None)
    uv = Xc[..., :2] / z
    u = uv[..., 0] * K[0, 0] + K[0, 2]
    v = uv[..., 1] * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=-1)


def to_uint8_rgb(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8) if img.max() <= 1.0 \
            else img.astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mamma_full")
    ap.add_argument("--max-sequences", type=int, default=3,
                    help="Limit sequences scanned at BUILD time (speed). Full config uses all.")
    ap.add_argument("--max-frames", type=int, default=2,
                    help="Limit frames/sequence scanned at BUILD time (speed).")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seq-index", type=int, default=None,
                    help="Fixed sample index; default = random per sample.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "debug", "out_dataloader"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Start clean so the folder only holds THIS run's overlays (random view sampling
    # otherwise leaves stale files under different IOI names across runs).
    if os.path.isdir(args.out):
        for f in os.listdir(args.out):
            if f.endswith(".png"):
                os.remove(os.path.join(args.out, f))
    os.makedirs(args.out, exist_ok=True)

    # --- resolve the training config, reuse its train.common_config verbatim ---
    with initialize_config_dir(config_dir=os.path.join(REPO, "training", "config"), version_base=None):
        cfg = compose(config_name=args.config)

    train = cfg.data.train
    cc = OmegaConf.create(OmegaConf.to_container(train.common_config, resolve=True))
    # base_dataset needs these knobs present; fill any the config omits.
    cc.setdefault("rescale", True)
    cc.setdefault("debug", False)
    cc.setdefault("get_nearby", False)
    cc.setdefault("allow_duplicate_img", False)
    cc.setdefault("landscape_check", False)
    cc.setdefault("rescale_aug", cc.get("rescale_aug", False))
    if "scales" not in cc.augs:
        cc.augs.scales = None

    ds_cfg = train.dataset.dataset_configs[0]
    print("=" * 78)
    print(f"config={args.config}  img_nums={cc.get('img_nums')}  aspects={cc.augs.get('aspects')}  "
          f"fixed_view_sampling={cc.get('fixed_view_sampling')}")
    print(f"dataset root: {ds_cfg.get('SysSMPL_DIR')}")
    print(f"[build] limiting scan to max_sequences={args.max_sequences}, "
          f"max_frames_per_sequence={args.max_frames} (debug speed only)")

    base = instantiate(
        ds_cfg,
        common_conf=cc,
        max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames,
        _recursive_=False,
    )
    n = base.sequence_list_len
    print(f"[build] dataset samples available (bounded): {n}")
    if n == 0:
        print("!! no samples found -- check the dataset root / min_num_images.")
        sys.exit(1)

    # img_per_seq / aspect exactly as the DynamicBatchSampler would pick.
    img_nums = list(cc.get("img_nums", [4, 4]))
    aspects = list(cc.augs.get("aspects", [1.0, 1.0]))
    img_per_seq = int(img_nums[0])
    aspect = float(aspects[0])

    worst = 0.0
    for si in range(args.num_samples):
        seq_index = args.seq_index if args.seq_index is not None else int(rng.integers(0, n))
        data = base.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect)

        S = len(data["extrinsics"])
        has_smpl = np.asarray(data["has_smpl"]).reshape(-1)
        P = has_smpl.shape[0]
        print("\n" + "-" * 78)
        print(f"[sample {si}] seq_index={seq_index}  seq_name={data['seq_name']}")
        print(f"  views S={S}  people P={P}  has_smpl={has_smpl.astype(int).tolist()}  "
              f"num_people={int(np.asarray(data['num_people']))}")
        print(f"  ids={np.asarray(data['ids']).tolist()}")
        # timestep(frame) = the image file stem, e.g. "0000" from ".../IOI_01/0000.jpg".
        frames = [os.path.splitext(os.path.basename(p))[0] for p in data["image_paths"]]
        same_ts = len(set(frames)) == 1
        print(f"  TIMESTEP(frame): {sorted(set(frames))}  "
              f"[{'OK all views share one timestep' if same_ts else 'MISMATCH!'}]")
        assert same_ts, f"views span multiple timesteps: {frames}"
        frame_tag = frames[0]
        print("  IMAGE PATHS:")
        for s, p in enumerate(data["image_paths"]):
            view = os.path.basename(os.path.dirname(p))
            print(f"    view{s} [{view}] frame={frames[s]}  {p}")

        joints3d = np.asarray(data["smpl_joints3d_world"], np.float64)   # (S,P,24,3)
        joints2d = np.asarray(data["smpl_joints2d"], np.float64)         # (S,P,24,2)
        conf = np.asarray(data["smpl_joints2d_confidence"], np.float64)  # (S,P,24)
        lmk = data.get("smpl_landmarks2d")                              # (S,P,512,2) norm or None
        lmk_vis = data.get("smpl_landmarks2d_visibility")

        # ---- numeric reprojection check (in-frame joints of valid people) ----
        errs = []
        for s in range(S):
            E, K = data["extrinsics"][s], data["intrinsics"][s]
            for p in range(P):
                if has_smpl[p] < 0.5:
                    continue
                proj = project_opencv(joints3d[s, p], E, K)             # (24,2)
                m = conf[s, p] > 0.5
                if m.any():
                    errs.append(np.abs(proj[m] - joints2d[s, p][m]))
        if errs:
            errs = np.concatenate(errs, axis=0)          # (N, 2) = per-coordinate abs error
            per_pt = np.linalg.norm(errs, axis=-1)        # (N,) Euclidean px error per joint
            mean_px = float(per_pt.mean())
            max_px = float(per_pt.max())
            worst = max(worst, max_px)
            status = "OK" if max_px < 1.0 else ("WARN" if max_px < 5.0 else "BAD")
            print(f"  REPROJECTION joints3d_world -> (extr,intr) vs loader joints2d "
                  f"[{per_pt.shape[0]} in-frame joints]: "
                  f"mean={mean_px:.4f}px  max={max_px:.4f}px  [{status}]")
        else:
            print("  REPROJECTION: no in-frame valid joints to check.")

        # ---- visual overlays ----
        for s in range(S):
            img = to_uint8_rgb(data["images"][s]).copy()
            H, W = img.shape[:2]
            E, K = data["extrinsics"][s], data["intrinsics"][s]
            for p in range(P):
                if has_smpl[p] < 0.5:
                    continue
                gt = joints2d[s, p]
                proj = project_opencv(joints3d[s, p], E, K)
                m = conf[s, p] > 0.5
                for j in range(gt.shape[0]):
                    if not m[j]:
                        continue
                    cv2.circle(img, (int(round(gt[j, 0])), int(round(gt[j, 1]))), 4, (0, 255, 0), -1)
                    pu, pv = int(round(proj[j, 0])), int(round(proj[j, 1]))
                    cv2.line(img, (pu - 4, pv - 4), (pu + 4, pv + 4), (255, 0, 0), 1)
                    cv2.line(img, (pu - 4, pv + 4), (pu + 4, pv - 4), (255, 0, 0), 1)
                if lmk is not None:
                    L = np.asarray(lmk)[s, p]                      # (512,2) normalised
                    vis = np.asarray(lmk_vis)[s, p] if lmk_vis is not None else np.ones(L.shape[0])
                    for k in range(L.shape[0]):
                        if vis[k] < 0.5:
                            continue
                        cv2.circle(img, (int(round(L[k, 0] * W)), int(round(L[k, 1] * H))), 1,
                                   (0, 128, 255), -1)
            view = os.path.basename(os.path.dirname(data["image_paths"][s]))
            out_path = os.path.join(args.out, f"sample{si}_frame{frame_tag}_view{s}_{view}.png")
            cv2.imwrite(out_path, img[:, :, ::-1])  # RGB -> BGR
        print(f"  overlays saved -> {args.out}/sample{si}_view*.png")
        print("  legend: green=GT joints2d  red-x=reprojected joints3d  blue=dense landmarks")

    print("\n" + "=" * 78)
    print(f"worst reprojection error across samples: {worst:.4f}px "
          f"({'PASS <1px' if worst < 1.0 else 'CHECK'})")
    sys.exit(0 if worst < 5.0 else 2)


if __name__ == "__main__":
    main()
