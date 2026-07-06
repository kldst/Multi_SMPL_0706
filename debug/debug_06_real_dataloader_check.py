#!/usr/bin/env python
"""Debug 06 -- pull ONE sample from the real SysSMPLMultiDataset and validate it.

Unlike debug_04/05 (which read the raw files directly), this constructs the
*actual* training dataset (`SysSMPLMultiDataset`, raw Mamma_mv_split path) and
calls `get_data`, so it exercises the full loader: `_build_raw_mamma_sequences`,
`process_one_image` (crop/resize + track + extra_maps mask), SMPL-X joint decode,
and the new dense-landmark / person-mask GT.

It (1) prints shapes / value ranges of every batch tensor, (2) overlays the
loader's OUTPUT — landmarks (denormalised from [-1,1]), 24 joints, and the
per-person patch mask — back onto each processed view, so a wrong crop/resize or
normalisation is visible immediately.

Build is bounded via --max-sequences / --max-frames so startup is fast even on a
cold disk. Run:
  python debug/debug_06_real_dataloader_check.py --num-views 4 --max-sequences 1 --max-frames 2
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from mamma_raw_io import DEFAULT_MAMMA_ROOT, DEFAULT_OUTPUT_ROOT, save_rgb
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset

PERSON_COLORS = [(255, 64, 64), (64, 160, 255), (64, 220, 120),
                 (255, 200, 40), (220, 80, 255), (0, 220, 220)]


def _to_uint8_rgb(img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.clip(0, 255).astype(np.uint8)
    return a.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(DEFAULT_MAMMA_ROOT))
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--max-people", type=int, default=5)
    p.add_argument("--max-sequences", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = p.parse_args()

    out_dir = Path(args.output_root) / "06_real_dataloader"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = SimpleNamespace(
        img_size=args.image_size, patch_size=args.patch_size,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=False,
        inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
    )
    print(f"[debug_06] building dataset (max_sequences={args.max_sequences}, "
          f"max_frames={args.max_frames}) — first cold build may take a while...")
    ds = SysSMPLMultiDataset(
        common_conf=common, split="train",
        SysSMPL_DIR=args.root, SysSMPL_ANNOTATION_DIR=args.root,
        min_num_images=args.num_views, max_num_people=args.max_people,
        emit_landmarks=True, emit_person_mask=True,
        max_sequences=args.max_sequences, max_frames_per_sequence=args.max_frames,
    )
    print(f"[debug_06] sequences={ds.sequence_list_len} total_views={ds.total_frame_num}")

    # show which sample/paths get_data pulled (so you can eyeball the source files)
    seq_name = ds.sequence_list[0]
    print(f"[debug_06] sample seq_key = {seq_name}")
    for i, anno in enumerate(ds.data_store[seq_name][: args.num_views]):
        img = anno.get("image_path")
        dp = anno.get("data_path")
        mk = anno.get("mask_path")
        print(f"    view[{i}] image : {img}")
        print(f"            data  : {dp}")
        print(f"            mask  : {mk}")

    batch = ds.get_data(seq_index=0, img_per_seq=args.num_views, aspect_ratio=1.0)
    print(f"[debug_06] batch seq_name = {batch.get('seq_name')}")
    print(f"[debug_06] batch image_paths (what get_data loaded):")
    for i, p in enumerate(batch.get("image_paths", [])):
        print(f"    [{i}] {p}")

    # ---- 1. report every tensor ----
    print("[debug_06] batch tensors:")
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            arr = np.stack(v, 0)
        else:
            continue
        extra = ""
        if np.issubdtype(arr.dtype, np.floating) and arr.size:
            extra = f" range[{arr.min():.3f},{arr.max():.3f}]"
        print(f"    {k:28s} {str(arr.shape):22s} {arr.dtype}{extra}")

    P = int(np.asarray(batch["num_people"]))
    size = args.image_size
    pg = size // args.patch_size
    imgs = batch["images"]
    joints = np.asarray(batch["smpl_joints2d"])                 # (S,P,24,2) pixels
    jconf = np.asarray(batch["smpl_joints2d_confidence"])       # (S,P,24)
    lmk = np.asarray(batch["smpl_landmarks2d"])                 # (S,P,512,2) [0,1]
    vis = np.asarray(batch["smpl_landmarks2d_visibility"])      # (S,P,512)
    pmask = np.asarray(batch["person_mask"])                    # (S,P,pg,pg)
    S = lmk.shape[0]

    # ---- 2. overlay loader output onto each processed view ----
    for s in range(S):
        base = _to_uint8_rgb(imgs[s])
        if base.shape[:2] != (size, size):
            base = cv2.resize(base, (size, size))
        lay = base.copy()
        mask_lay = base.copy()
        for pi in range(P):
            c = PERSON_COLORS[pi % len(PERSON_COLORS)]
            # landmarks: denormalise [0,1] -> pixels
            xy = lmk[s, pi]
            px = xy[:, 0] * size
            py = xy[:, 1] * size
            for (x, y), vv in zip(np.stack([px, py], 1), vis[s, pi]):
                if 0 <= x < size and 0 <= y < size:
                    cv2.circle(lay, (int(x), int(y)), 2 if vv > 0.5 else 1, c, -1, cv2.LINE_AA)
            # 24 joints (already pixels)
            for (jx, jy), cf in zip(joints[s, pi], jconf[s, pi]):
                if cf > 0.5 and 0 <= jx < size and 0 <= jy < size:
                    cv2.drawMarker(lay, (int(jx), int(jy)), (255, 255, 255),
                                   cv2.MARKER_CROSS, 6, 1)
            # patch mask -> upsample + tint
            up = cv2.resize((pmask[s, pi] * 255).astype(np.uint8), (size, size),
                            interpolation=cv2.INTER_NEAREST) > 100
            mask_lay[up] = (0.5 * mask_lay[up] + 0.5 * np.array(c, np.float32)).astype(np.uint8)
        save_rgb(out_dir / f"view{s}_landmarks_joints.png", lay)
        save_rgb(out_dir / f"view{s}_mask.png", mask_lay)

    # ---- 3. quick correctness asserts ----
    ok = True
    ok &= abs(float(lmk.max())) < 3 and abs(float(lmk.min())) < 3   # normalised-ish
    ok &= set(np.unique(vis).tolist()) <= {0.0, 1.0}                # visibility is {0,1}
    ok &= float(pmask.max()) <= 1.0 and float(pmask.min()) >= 0.0   # occupancy [0,1]
    ok &= joints.shape[:2] == lmk.shape[:2] == pmask.shape[:2]      # (S,P) aligned
    print(f"[debug_06] wrote overlays to {out_dir}")
    print("PASS" if ok else "FAIL", "— shapes/ranges consistent; inspect the overlays visually")


if __name__ == "__main__":
    main()
