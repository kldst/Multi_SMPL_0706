#!/usr/bin/env python
"""Visualize the MAMMA-style contact GT emitted by SysSMPLMultiDataset.

For one real sample it overlays, per view & per person, the 512 dense landmarks
coloured by their contact GT:
  * BLUE  = floor_contact   (smpl_floor_contact == 1)
  * RED   = person-person contact (smpl_contact == 1)
  * faint = neither
so you can eyeball that floor-contact lands on feet and person-person contact
lands where the two bodies touch.

Run (fast smoke build):
    conda activate mamma
    python debug/debug_contact_gt.py --max-sequences 3 --num-samples 1
"""
import argparse
import inspect
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
os.chdir(REPO)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import cv2  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def to_uint8_rgb(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).round().astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mamma_overfit")
    ap.add_argument("--max-sequences", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-root", default=None,
                    help="Override SysSMPL_DIR/ANNOTATION_DIR (a dir containing .../png/<seq>).")
    ap.add_argument("--out", default=os.path.join(REPO, "debug", "out_contact"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if os.path.isdir(args.out):
        for f in os.listdir(args.out):
            if f.endswith(".png"):
                os.remove(os.path.join(args.out, f))
    os.makedirs(args.out, exist_ok=True)

    with initialize_config_dir(config_dir=os.path.join(REPO, "training", "config"), version_base=None):
        cfg = compose(config_name=args.config)
    train = cfg.data.train
    cc = OmegaConf.create(OmegaConf.to_container(train.common_config, resolve=True))
    for k, v in dict(rescale=True, debug=False, get_nearby=False, allow_duplicate_img=False,
                     landscape_check=False, rescale_aug=False, fixed_view_sampling=True,
                     inside_random=False, img_nums=[4, 4]).items():
        if k not in cc:
            cc[k] = v
    if "scales" not in cc.augs:
        cc.augs.scales = None
    ds_cfg = OmegaConf.create(OmegaConf.to_container(train.dataset.dataset_configs[0], resolve=True))
    if args.scene_root:
        ds_cfg["SysSMPL_DIR"] = args.scene_root
        ds_cfg["SysSMPL_ANNOTATION_DIR"] = args.scene_root

    base = instantiate(ds_cfg, common_conf=cc, max_sequences=args.max_sequences,
                       max_frames_per_sequence=args.max_frames, _recursive_=False)
    print(f"[build] emit_contact={base.emit_contact} contact_threshold={base.contact_threshold} "
          f"samples={base.sequence_list_len}")
    assert base.emit_contact, "config has emit_contact disabled; enable it to visualize contact GT"

    for si in range(args.num_samples):
        seq_index = int(rng.integers(0, base.sequence_list_len))
        data = base.get_data(seq_index=seq_index, img_per_seq=4, aspect_ratio=1.0)
        assert "smpl_contact" in data, "dataset did not emit smpl_contact"
        S = len(data["extrinsics"])
        has = np.asarray(data["has_smpl"]).reshape(-1)
        lmk = np.asarray(data["smpl_landmarks2d"])       # (S,P,512,2) normalised [0,1]
        contact = np.asarray(data["smpl_contact"])       # (S,P,512)
        floor = np.asarray(data["smpl_floor_contact"])   # (S,P,512)
        vis = np.asarray(data["smpl_landmarks2d_visibility"])
        P = has.shape[0]
        print(f"\n[sample {si}] seq_index={seq_index}  views={S} people={int(has.sum())}")
        for p in range(P):
            if has[p] < 0.5:
                continue
            print(f"  person {p}: contact={int(contact[:, p].sum())}  "
                  f"floor={int(floor[:, p].sum())}  (summed over {S} views, of {S*512} pts)")

        for s in range(S):
            img = to_uint8_rgb(data["images"][s]).copy()
            H, W = img.shape[:2]
            for p in range(P):
                if has[p] < 0.5:
                    continue
                for k in range(512):
                    x = int(round(lmk[s, p, k, 0] * W))
                    y = int(round(lmk[s, p, k, 1] * H))
                    if not (0 <= x < W and 0 <= y < H):
                        continue
                    if floor[s, p, k] > 0.5:
                        cv2.circle(img, (x, y), 2, (0, 100, 255), -1)   # floor = blue
                    elif contact[s, p, k] > 0.5:
                        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)     # contact = red
                    else:
                        cv2.circle(img, (x, y), 1, (0, 200, 0), -1)     # neither = faint green
            view = os.path.basename(os.path.dirname(data["image_paths"][s]))
            out = os.path.join(args.out, f"sample{si}_view{s}_{view}_contact.png")
            cv2.imwrite(out, img[:, :, ::-1])
        print(f"  overlays -> {args.out}/sample{si}_view*_contact.png "
              f"(blue=floor, red=person-person, green=none)")

    print("\nDONE")


if __name__ == "__main__":
    main()
