#!/usr/bin/env python
"""Debug 08 -- visually verify the dense-landmark VISIBILITY GT from the dataset.

Visibility target chain (raw Mamma_mv_split path):
  pyd["vertex_visibility"] (10475,)   # per-vertex occlusion, ships in the pyd
    -> downsample_visibility(M @ vv, thr)  -> (512,) in {0,1}
    -> * in-frame(after crop/resize)       -> final smpl_landmarks2d_visibility

This pulls ONE real sample from SysSMPLMultiDataset (the actual training loader)
and, per view/person, overlays the 512 landmarks colour-coded:
    GREEN  = visible   (vis > 0.5)
    RED    = occluded / out-of-frame (vis == 0)
so you can eyeball whether "visible" points really sit on the *front-facing,
un-occluded* body surface (back-of-body / behind-other-person / off-image points
should be red). Also prints the per-person, per-view visible fraction.

Run (bounded build so cold start is fast):
  python debug/debug_08_landmark_visibility_check.py --num-views 4 --max-sequences 1 --max-frames 1
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from mamma_raw_io import DEFAULT_MAMMA_ROOT, DEFAULT_OUTPUT_ROOT, save_rgb
from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset

VIS_COLOR = (40, 220, 80)     # green  = visible
OCC_COLOR = (240, 60, 60)     # red    = occluded / out-of-frame


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
    p.add_argument("--max-frames", type=int, default=1)
    p.add_argument("--seq-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--vis-threshold", type=float, default=0.5)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = p.parse_args()

    out_dir = Path(args.output_root) / "08_landmark_visibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = SimpleNamespace(
        img_size=args.image_size, patch_size=args.patch_size,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=False,
        inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
    )
    print(f"[debug_08] building dataset (max_sequences={args.max_sequences}, "
          f"max_frames={args.max_frames}) — first cold build may take a while...")
    ds = SysSMPLMultiDataset(
        common_conf=common, split="train",
        SysSMPL_DIR=args.root, SysSMPL_ANNOTATION_DIR=args.root,
        min_num_images=args.num_views, max_num_people=args.max_people,
        emit_landmarks=True, emit_person_mask=False,
        landmark_visibility_threshold=args.vis_threshold,
        max_sequences=args.max_sequences, max_frames_per_sequence=args.max_frames,
    )
    print(f"[debug_08] sequences={ds.sequence_list_len} total_views={ds.total_frame_num}")

    seq_name = ds.sequence_list[args.seq_index]
    print(f"[debug_08] sample seq_key = {seq_name}")

    batch = ds.get_data(seq_index=args.seq_index, img_per_seq=args.num_views, aspect_ratio=1.0)
    print(f"[debug_08] batch seq_name = {batch.get('seq_name')}")
    for i, pth in enumerate(batch.get("image_paths", [])):
        print(f"    view[{i}] {pth}")

    P = int(np.asarray(batch["num_people"]))
    size = args.image_size
    imgs = batch["images"]
    lmk = np.asarray(batch["smpl_landmarks2d"])            # (S,P,512,2) in [0,1]
    vis = np.asarray(batch["smpl_landmarks2d_visibility"]) # (S,P,512) in {0,1}
    S = lmk.shape[0]

    print(f"\n[debug_08] visible fraction per (view, person)  [512 landmarks each]:")
    for s in range(S):
        base = _to_uint8_rgb(imgs[s])
        if base.shape[:2] != (size, size):
            base = cv2.resize(base, (size, size))
        lay = base.copy()
        rates = []
        for pi in range(P):
            xy = lmk[s, pi]
            px = xy[:, 0] * size
            py = xy[:, 1] * size
            vv = vis[s, pi]
            rates.append(float(vv.mean()))
            for (x, y), v in zip(np.stack([px, py], 1), vv):
                if 0 <= x < size and 0 <= y < size:
                    col = VIS_COLOR if v > 0.5 else OCC_COLOR
                    r = 2 if v > 0.5 else 1
                    cv2.circle(lay, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
        # small legend bar
        cv2.rectangle(lay, (6, 6), (150, 44), (0, 0, 0), -1)
        cv2.circle(lay, (16, 18), 4, VIS_COLOR, -1); cv2.putText(lay, "visible", (26, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(lay, (16, 34), 4, OCC_COLOR, -1); cv2.putText(lay, "occluded", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        save_rgb(out_dir / f"view{s}_visibility.png", lay)
        print(f"    view {s}: " + "  ".join(f"P{pi}={r*100:5.1f}%" for pi, r in enumerate(rates)))

    print(f"\n[debug_08] wrote overlays to {out_dir}")
    print("[debug_08] EYEBALL CHECK: green dots should lie on the front-facing, "
          "un-occluded body surface; back-of-body / behind-other-person / off-frame "
          "landmarks should be red.")


if __name__ == "__main__":
    main()
