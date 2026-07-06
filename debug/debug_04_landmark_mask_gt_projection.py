#!/usr/bin/env python
"""Debug 04 -- validate the dense-landmark & per-person-mask GT on real data.

Self-contained and *decode-free*: the raw Mamma_mv_split ``.data.pyd`` already
ships per-person ``vertices2d`` (projected 2D) and ``vertex_visibility``, so the
512-landmark GT is just ``M @ vertices2d`` / ``M @ vertex_visibility`` -- no SMPL
forward pass needed (which is heavy and unnecessary here).

For a chosen 4-view / multi-person frame it:
  1. overlays the 512 dense landmarks on each view (one colour per person),
  2. tints each person's instance mask (``*.mask.jpg``, value == person_idx+1),
  3. renders each person's patch-grid mask occupancy (what the mask head sees),
  4. writes a JSON summary (per-person visible fraction, landmark bbox, ...).

Run:
  python debug/debug_04_landmark_mask_gt_projection.py \
      --seq-name be_HsuS3iLSSWWZ_seq_000001 --frame 0000 --num-views 4
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

from mamma_raw_io import (
    add_common_args,
    choose_frame_and_views,
    choose_sequence,
    load_gray,
    load_rgb_resized,
    save_rgb,
)
from training.data.landmark_mask_gt import (
    downsample_vertices,
    downsample_visibility,
    load_verts512_matrix,
    rasterize_person_patch_mask,
)

PERSON_COLORS = [
    (255, 64, 64), (64, 160, 255), (64, 220, 120), (255, 200, 40),
    (220, 80, 255), (0, 220, 220), (255, 120, 0), (140, 140, 255),
]


def _person_color(idx: int):
    return PERSON_COLORS[idx % len(PERSON_COLORS)]


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()

    out_dir = Path(args.output_root) / "04_landmark_mask_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = args.image_size
    patch_h = patch_w = size // args.patch_size

    rng = random.Random(args.seed)
    seq_dir = choose_sequence(args.mamma_root, args.scene_root, args.seq_name, rng)
    frame, views = choose_frame_and_views(seq_dir, args.frame, args.num_views, rng)
    print(f"[debug_04] seq={seq_dir.name} frame={frame} views={views}")

    matrix = load_verts512_matrix()
    summary = {"seq": seq_dir.name, "frame": frame, "views": views, "per_view": {}}

    for view in views:
        pyd = pickle.load(open(seq_dir / view / f"{frame}.data.pyd", "rb"))
        person_ids = sorted(pyd.keys(), key=lambda x: int(x))
        rgb, (orig_h, orig_w) = load_rgb_resized(seq_dir / view / f"{frame}.jpg", size)
        mask_img = load_gray(seq_dir / view / f"{frame}.mask.jpg")
        scale2d = np.array([size / orig_w, size / orig_h], dtype=np.float32)

        lmk_overlay = rgb.copy()
        mask_overlay = rgb.copy()
        view_stats = {}

        for slot, pid in enumerate(person_ids):
            person = pyd[pid]
            person_idx = int(person.get("person_idx", slot))
            color = _person_color(person_idx)

            # (1) 512 landmark 2D = M @ vertices2d, scaled to the resized image.
            v2d = np.asarray(person["vertices2d"], dtype=np.float32)      # (10475, 2)
            lmk2d = downsample_vertices(matrix, v2d) * scale2d           # (512, 2)
            vv = np.asarray(person["vertex_visibility"], dtype=np.float32).reshape(-1)
            lmk_vis = downsample_visibility(matrix, vv[None, :])[0]      # (512,)
            for (x, y), vis in zip(lmk2d, lmk_vis):
                if 0 <= x < size and 0 <= y < size:
                    r = 2 if vis > 0.5 else 1
                    cv2.circle(lmk_overlay, (int(x), int(y)), r, color, -1, cv2.LINE_AA)

            # (2) instance-mask tint (value == person_idx + 1).
            pv = person_idx + 1
            person_pixels = cv2.resize(
                (mask_img == pv).astype(np.uint8), (size, size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            mask_overlay[person_pixels] = (
                0.5 * mask_overlay[person_pixels] + 0.5 * np.array(color, np.float32)
            ).astype(np.uint8)

            # (3) patch-grid occupancy = what the mask head regresses.
            occ = rasterize_person_patch_mask(mask_img, pv, patch_h, patch_w, image_size=size)
            occ_vis = cv2.resize((occ * 255).astype(np.uint8), (size, size),
                                 interpolation=cv2.INTER_NEAREST)
            save_rgb(out_dir / f"{view}_p{person_idx}_patchmask.png",
                     cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2RGB))

            view_stats[f"p{person_idx}"] = {
                "landmark_visible_fraction": float(lmk_vis.mean()),
                "landmark_x_range": [float(lmk2d[:, 0].min()), float(lmk2d[:, 0].max())],
                "landmark_y_range": [float(lmk2d[:, 1].min()), float(lmk2d[:, 1].max())],
                "mask_patch_occupancy_mean": float(occ.mean()),
            }

        save_rgb(out_dir / f"{view}_landmarks.png", lmk_overlay)
        save_rgb(out_dir / f"{view}_masks.png", mask_overlay)
        summary["per_view"][view] = view_stats

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[debug_04] wrote overlays + summary to {out_dir}")
    for view, stats in summary["per_view"].items():
        print(f"  {view}: {stats}")


if __name__ == "__main__":
    main()
