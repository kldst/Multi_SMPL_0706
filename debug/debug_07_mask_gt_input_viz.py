"""debug_07_mask_gt_input_viz.py

Show EXACTLY what the mamma_mask_dpt pipeline feeds the model and what the mask
head is supervised against:

  * INPUT  = batch["images"]      (S views, 3xHxW, values in [0,1])
  * TARGET = batch["person_mask"] (S views, P person-slots, hxw soft occupancy in
             [0,1]); slot p is a real person iff batch["has_smpl"][p] == 1.
             With person_mask_stride=1 the mask grid == the image grid (518x518),
             so it overlays 1:1 on the input.

The mask GT for (view s, person p) is 1 where the instance-mask JPG pixel value
== person_idx+1 (occlusion-aware), 0 elsewhere; it rides through the SAME
crop/resize as the image, so pixels line up.

Run (no GPU / no checkpoint needed):
  cd training
  DATA_ROOT=/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma \
  conda run -n mamma python ../debug/debug_07_mask_gt_input_viz.py
Outputs PNGs to $OUT (default: scratchpad).
"""
import os, sys

REPO = "/mnt/train-data-4-hdd/yian/vggt_multi_0621_mamma_demo_eval_bundle"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
os.chdir(os.path.join(REPO, "training"))

# --- single-process distributed group (the dynamic dataloader's sampler needs it) ---
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29777")

import numpy as np
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma")
OUT = os.environ.get("OUT", "/tmp/claude-1010/-mnt-train-data-4-hdd-yian/"
                            "6ab6daf1-c2b4-4d0a-8d82-f985d4ebb5b9/scratchpad")
os.makedirs(OUT, exist_ok=True)

if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "training/config")):
    cfg = compose(config_name="mamma_mask_dpt")

dc = cfg.data.train.dataset.dataset_configs[0]
dc.SysSMPL_DIR = DATA_ROOT
dc.SysSMPL_ANNOTATION_DIR = DATA_ROOT
dc.emit_person_mask = True
dc.emit_landmarks = False
dc.person_mask_stride = 1          # pixel-level GT (518x518), 1:1 with the image
dc.max_sequences = 4               # bound the cold pyd read
cfg.data.train.common_config.img_nums = [4, 4]
cfg.max_img_per_gpu = 4            # -> 1 sample (4 views) per batch
cfg.num_workers = 0

# build ONLY the dataset (no model / no checkpoint)
train_ds = instantiate(cfg.data.train, _recursive_=False)
train_ds.seed = 42
loader = train_ds.get_loader(epoch=0)

batch = next(iter(loader))
images = batch["images"][0]            # (S, 3, H, W) in [0,1]
pmask = batch["person_mask"][0]        # (S, P, h, w) in [0,1]
has_smpl = batch["has_smpl"][0]        # (P,)
seq = batch.get("seq_name", ["?"])[0]

S, _, H, W = images.shape
P = pmask.shape[1]
present = [p for p in range(P) if float(has_smpl[p]) > 0.5]

print("=" * 74)
print("seq          :", seq)
print("images       :", tuple(images.shape), " range [%.2f, %.2f]" % (images.min(), images.max()))
print("person_mask  :", tuple(pmask.shape), " range [%.2f, %.2f]" % (pmask.min(), pmask.max()))
print("has_smpl     :", has_smpl.tolist())
print("present slots:", present, f"({len(present)} real people of {P} slots)")
for p in present:
    cov = [f"{float(pmask[s, p].mean())*100:.1f}%" for s in range(S)]
    print(f"  person slot {p}: per-view mask coverage = {cov}")
print("=" * 74)

img_np = images.permute(0, 2, 3, 1).clamp(0, 1).numpy()     # (S,H,W,3)
colors = [cm.tab10(i % 10)[:3] for i in range(P)]

# ---- Figure 1: per view, INPUT | GT-mask OVERLAY (all persons colored) ----
fig, axes = plt.subplots(S, 2, figsize=(8, 4 * S))
if S == 1:
    axes = axes[None, :]
for s in range(S):
    axes[s, 0].imshow(img_np[s]); axes[s, 0].set_title(f"view {s}: INPUT image"); axes[s, 0].axis("off")
    ov = img_np[s].copy()
    for p in present:
        m = pmask[s, p].numpy()
        if m.shape != (H, W):   # only needed if stride!=1
            m = np.asarray(plt.matplotlib.image.imread) if False else m
        a = (m[..., None] > 0.5).astype(np.float32) * 0.55
        ov = ov * (1 - a) + np.array(colors[p])[None, None, :] * a
    axes[s, 1].imshow(ov)
    axes[s, 1].set_title(f"view {s}: + GT person_mask (color=slot)")
    axes[s, 1].axis("off")
f1 = os.path.join(OUT, "mask_gt_overview.png")
plt.tight_layout(); plt.savefig(f1, dpi=90); plt.close(fig)

# ---- Figure 2: view 0, one column per present person (raw GT mask) ----
ncol = 1 + len(present)
fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
axes[0].imshow(img_np[0]); axes[0].set_title("view 0 INPUT"); axes[0].axis("off")
for j, p in enumerate(present):
    axes[j + 1].imshow(img_np[0])
    axes[j + 1].imshow(pmask[0, p].numpy(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[j + 1].set_title(f"GT mask slot {p}"); axes[j + 1].axis("off")
f2 = os.path.join(OUT, "mask_gt_view0_perperson.png")
plt.tight_layout(); plt.savefig(f2, dpi=90); plt.close(fig)

print("wrote:\n ", f1, "\n ", f2)
dist.destroy_process_group()
