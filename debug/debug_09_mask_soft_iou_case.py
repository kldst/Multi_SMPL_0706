#!/usr/bin/env python
"""Debug 09 -- one dataset example for the person-mask GT + soft-IoU metric.

Pulls ONE real sample from SysSMPLMultiDataset (emit_person_mask=True), then for
each view/person:
  (1) overlays the patch-grid GT mask on the processed image (so you see what the
      PersonMaskHead is supervised against), and
  (2) computes `mask_soft_iou` (the exact metric from loss_mask.compute_mask_loss)
      for a few illustrative "predictions" so the numbers are concrete:
        - perfect  : pred == GT              -> soft IoU should be ~1.0 (soft, so a
                     hair under 1 when GT has soft/fractional patches)
        - shift1   : GT shifted by 1 patch   -> partial overlap
        - blurred  : GT box-blurred (3x3)    -> slight drop
        - empty    : pred all-zero           -> 0
        - random   : pred ~U(0,1)            -> low

Run:
  python debug/debug_09_mask_soft_iou_case.py --num-views 4 --max-sequences 1 --max-frames 1
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


def soft_iou(prob, gt):
    """Exact port of loss_mask.compute_mask_loss's soft-IoU (single map)."""
    prob = prob.astype(np.float64).ravel()
    gt = gt.astype(np.float64).ravel()
    inter = (prob * gt).sum()
    union = (prob + gt).sum() - inter
    return float(inter / max(union, 1e-6))


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
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = p.parse_args()

    out_dir = Path(args.output_root) / "09_mask_soft_iou"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = SimpleNamespace(
        img_size=args.image_size, patch_size=args.patch_size,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=False,
        inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
    )
    print(f"[debug_09] building dataset (max_sequences={args.max_sequences}, "
          f"max_frames={args.max_frames})...")
    ds = SysSMPLMultiDataset(
        common_conf=common, split="train",
        SysSMPL_DIR=args.root, SysSMPL_ANNOTATION_DIR=args.root,
        min_num_images=args.num_views, max_num_people=args.max_people,
        emit_landmarks=False, emit_person_mask=True,
        max_sequences=args.max_sequences, max_frames_per_sequence=args.max_frames,
    )
    seq_name = ds.sequence_list[args.seq_index]
    print(f"[debug_09] sample seq_key = {seq_name}")
    batch = ds.get_data(seq_index=args.seq_index, img_per_seq=args.num_views, aspect_ratio=1.0)

    P = int(np.asarray(batch["num_people"]))
    size = args.image_size
    imgs = batch["images"]
    pmask = np.asarray(batch["person_mask"])          # (S,P,pg,pg) soft occupancy [0,1]
    S, _, pg, _ = pmask.shape
    print(f"[debug_09] person_mask shape={pmask.shape} range[{pmask.min():.3f},{pmask.max():.3f}] "
          f"(patch grid {pg}x{pg})\n")

    rng = np.random.default_rng(0)
    print("[debug_09] mask_soft_iou for illustrative predictions "
          "(sigmoid(logits) modelled directly as the prob map):")
    print(f"    {'view/person':14s} {'perfect':>8s} {'shift1':>8s} {'blur3':>8s} {'empty':>8s} {'random':>8s} {'gt_occ%':>8s}")

    for s in range(S):
        base = _to_uint8_rgb(imgs[s])
        if base.shape[:2] != (size, size):
            base = cv2.resize(base, (size, size))
        overlay = base.copy()
        for pi in range(P):
            gt = pmask[s, pi].astype(np.float32)                       # (pg,pg) in [0,1]
            if gt.sum() < 1e-6:
                continue  # empty slot (person not in this view)
            # --- concrete soft-IoU cases ---
            perfect = gt.copy()
            shift1 = np.roll(gt, shift=1, axis=1)
            blur3 = cv2.blur(gt, (3, 3))
            empty = np.zeros_like(gt)
            rand = rng.random(gt.shape).astype(np.float32)
            print(f"    v{s} p{pi:<10d} "
                  f"{soft_iou(perfect, gt):8.3f} {soft_iou(shift1, gt):8.3f} "
                  f"{soft_iou(blur3, gt):8.3f} {soft_iou(empty, gt):8.3f} "
                  f"{soft_iou(rand, gt):8.3f} {100*float((gt>0.5).mean()):7.1f}%")
            # --- overlay GT mask on image ---
            up = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
            m = up > 0.5
            c = np.array(PERSON_COLORS[pi % len(PERSON_COLORS)], np.float32)
            overlay[m] = (0.5 * overlay[m] + 0.5 * c).astype(np.uint8)
        save_rgb(out_dir / f"view{s}_gt_mask.png", overlay)

    print(f"\n[debug_09] wrote GT-mask overlays to {out_dir}")
    print("[debug_09] note: 'perfect' is soft IoU of GT-as-prob vs GT; it is < 1.0 "
          "exactly where GT has fractional (soft) patches, which is expected.")


if __name__ == "__main__":
    main()
