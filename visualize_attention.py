#!/usr/bin/env python
"""Visualize the SMPL head cross-attention for a MAMMA test scene.

Each of the ``num_people`` person queries in
``smpl_multi_query_trans_rot_head.decoder.transformer`` cross-attends to the
flattened aggregator patch tokens (context = [B, S*P, C]).  The attention
weight therefore has shape [B, heads, num_people, S*P], which reshapes to
[num_people, S, patch_h, patch_w] -- a per-person spatial heatmap over every
camera view.  This script captures that weight, overlays it back onto the
input images, and writes PNGs (per person x view) plus a per-person grid.

Run inside the same conda env as the demo (`conda activate mamma`), e.g.:

    python visualize_attention.py \
        --checkpoint training/logs/0621_mamma/ckpts/checkpoint_step_10000.pt \
        --scene runs_00000 --num-views 8 --out-dir attn_out
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from einops import rearrange

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_REPO_DIR)
sys.path.append(os.path.join(_REPO_DIR, "training"))

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="0621_mamma", help="hydra config name under training/config")
    p.add_argument(
        "--checkpoint",
        default=os.path.join(_REPO_DIR, "training/logs/0621_mamma/ckpts/checkpoint_step_10000.pt"),
        help="Finetuned checkpoint with the SMPL head weights.",
    )
    p.add_argument(
        "--dataset-root",
        default="/mnt/train-data-4-hdd/yian/MAMMA_eval_dance",
        help="Eval dataset root (expects <root>/<split>/out_image/<scene>/*.jpg).",
    )
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--scene", default="", help="Scene name (substring match). Default: first sorted scene.")
    p.add_argument("--images-dir", default="", help="Bypass dataset layout and read images directly from this folder.")
    p.add_argument(
        "--mv-root",
        default="",
        help=(
            "Multi-view dataset root (Mamma_mv_split style): a tree whose leaf 'scene' folders "
            "contain per-camera subfolders (IOI_*) of per-frame jpgs. One frame is gathered across "
            "all cameras as the multi-view input. e.g. /mnt/train-data-4-hdd/yian/Mamma_mv_split/train"
        ),
    )
    p.add_argument("--seq-dir", default="", help="Point directly at one scene folder containing per-camera (IOI_*) subfolders.")
    p.add_argument("--frame", default="", help="Frame stem (e.g. 0000) for --mv-root/--seq-dir. Default: first common frame.")
    p.add_argument("--num-views", type=int, default=8, help="Max number of camera views (images) to feed in.")
    p.add_argument(
        "--layer",
        default="last",
        help="Which decoder cross-attn layer to show: an int index, 'last', or 'mean' (average all layers).",
    )
    p.add_argument("--presence-threshold", type=float, default=0.5, help="Sigmoid(presence) cutoff for 'present' people.")
    p.add_argument("--all-people", action="store_true", help="Visualize all query slots, not just present ones.")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap blend weight over the base image.")
    p.add_argument("--out-dir", default=os.path.join(_REPO_DIR, "attn_out"))
    return p.parse_args()


def build_model(config_name: str, checkpoint: str, device: str):
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    # We only need the aggregator + SMPL head for cross-attention; skip point head.
    cfg.model.enable_point = False
    cfg.model.enable_depth = False

    model = instantiate(cfg.model, _recursive_=False)
    if model.smpl_multi_query_trans_rot_head is None:
        raise RuntimeError(
            "config does not enable smpl_multi_query_trans_rot head -- this script "
            "visualizes that head's cross-attention."
        )

    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_info = model.load_state_dict(state_dict, strict=False)
    print(f"[CKPT] {checkpoint}\n       missing={len(load_info.missing_keys)} unexpected={len(load_info.unexpected_keys)}")
    model.eval().to(device)
    return model


_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _is_view_image(path: str) -> bool:
    """A real camera image, not a mask (….mask.jpg) or annotation."""
    name = os.path.basename(path).lower()
    if name.endswith(".mask.jpg") or name.endswith(".mask.png"):
        return False
    return os.path.splitext(name)[1] in _IMG_EXTS


def _looks_like_scene_dir(d: str) -> bool:
    """A multi-view 'scene' folder: has >=2 subdirs that each contain view jpgs."""
    subdirs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    cam_dirs = [s for s in subdirs if any(_is_view_image(f) for f in glob.glob(os.path.join(s, "*")))]
    return len(cam_dirs) >= 2


def _find_scene_dirs(root: str) -> list[str]:
    """Walk a Mamma_mv_split-style tree and collect leaf scene folders."""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if _looks_like_scene_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []  # don't descend below a scene
    return sorted(found)


def _gather_multiview_frame(scene_dir: str, frame: str, num_views: int) -> list[str]:
    """Gather one frame stem across all camera (IOI_*) subfolders of a scene."""
    cam_dirs = sorted(
        s for s in (os.path.join(scene_dir, x) for x in os.listdir(scene_dir))
        if os.path.isdir(s) and any(_is_view_image(f) for f in glob.glob(os.path.join(s, "*")))
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camera subfolders with images under {scene_dir}")

    if not frame:
        stems = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(cam_dirs[0], "*")) if _is_view_image(f)
        )
        frame = stems[0]
        print(f"[frame] auto-selected first frame: {frame}")

    paths = []
    for cam in cam_dirs:
        matches = [f for f in glob.glob(os.path.join(cam, f"{frame}.*")) if _is_view_image(f)]
        if matches:
            paths.append(sorted(matches)[0])
        else:
            print(f"[frame] warn: frame '{frame}' missing in {os.path.basename(cam)}, skipping this view")
    if not paths:
        raise FileNotFoundError(f"Frame '{frame}' not found in any camera of {scene_dir}")
    if num_views > 0:
        paths = paths[:num_views]
    return paths


def resolve_image_paths(args) -> list[str]:
    # 1) Direct flat folder of images.
    if args.images_dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.images_dir, "*")) if _is_view_image(p))
        if not paths:
            raise FileNotFoundError(f"No images found in {args.images_dir}")
        if args.num_views > 0:
            paths = paths[: args.num_views]
        print(f"[views] using {len(paths)} views from {args.images_dir}")
        return paths

    # 2) Multi-view layout: one scene folder with per-camera subfolders, one frame across cameras.
    scene_dir = args.seq_dir
    if not scene_dir and args.mv_root:
        scenes = _find_scene_dirs(args.mv_root)
        if not scenes:
            raise FileNotFoundError(f"No multi-view scene folders found under {args.mv_root}")
        if args.scene:
            matches = [s for s in scenes if args.scene in s]
            if not matches:
                raise FileNotFoundError(f"No scene matching '{args.scene}' under {args.mv_root}")
            scene_dir = matches[0]
        else:
            scene_dir = scenes[0]
            print(f"[scene] auto-selected first of {len(scenes)} scenes")
    if scene_dir:
        print(f"[scene] {scene_dir}")
        paths = _gather_multiview_frame(scene_dir, args.frame, args.num_views)
        print(f"[views] using {len(paths)} camera views @ frame")
        return paths

    # 3) Eval layout: <root>/<split>/out_image/<scene>/*.jpg
    scene = args.scene
    image_root = os.path.join(args.dataset_root, args.dataset_split, "out_image")
    if not scene:
        scenes = sorted(d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d)))
        if not scenes:
            raise FileNotFoundError(f"No scene folders under {image_root}")
        scene = scenes[0]
        print(f"[scene] auto-selected first scene: {scene}")
    image_dir = os.path.join(image_root, scene)
    paths = sorted(p for p in glob.glob(os.path.join(image_dir, "*")) if _is_view_image(p))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if args.num_views > 0:
        paths = paths[: args.num_views]
    print(f"[views] using {len(paths)} views from {image_dir}")
    return paths


class CrossAttnCapture:
    """Temporarily replace each CrossAttention.forward to stash its softmax weights."""

    def __init__(self, decoder_transformer):
        # transformer.layers = ModuleList([ [PreNorm(self_attn), PreNorm(cross_attn), PreNorm(ff)] , ... ])
        self.cross_attns = [layer[1].fn for layer in decoder_transformer.layers]
        self.captured: list[torch.Tensor] = [None] * len(self.cross_attns)
        self._originals = []

    def __enter__(self):
        for idx, ca in enumerate(self.cross_attns):
            self._originals.append(ca.forward)
            ca.forward = self._make_forward(idx, ca)
        return self

    def __exit__(self, *exc):
        for ca, orig in zip(self.cross_attns, self._originals):
            ca.forward = orig
        self._originals = []
        return False

    def _make_forward(self, idx, module):
        def forward(x, context=None):
            ctx = context if context is not None else x
            k, v = module.to_kv(ctx).chunk(2, dim=-1)
            q = module.to_q(x)
            q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=module.heads) for t in (q, k, v))
            dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
            attn = dots.softmax(dim=-1)
            # [B, heads, num_people, S*P] -> keep on CPU float for later reshaping
            self.captured[idx] = attn.detach().float().cpu()
            attn_d = module.dropout(attn)
            out = torch.matmul(attn_d, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            return module.to_out(out)

        return forward


def select_attention(captured: list[torch.Tensor], layer_arg: str) -> torch.Tensor:
    """Return a single [heads, num_people, S*P] tensor for batch item 0."""
    stack = torch.stack(captured, dim=0)  # [L, B, heads, people, S*P]
    stack = stack[:, 0]  # batch 0 -> [L, heads, people, S*P]
    if layer_arg == "mean":
        return stack.mean(dim=0)
    layer_idx = len(captured) - 1 if layer_arg == "last" else int(layer_arg)
    return stack[layer_idx]


def overlay_heatmap(base_rgb: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    """base_rgb: HxWx3 uint8. heat: hxw float (any range). Returns HxWx3 uint8 RGB."""
    h, w = base_rgb.shape[:2]
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_u8 = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(color, alpha, base_rgb, 1 - alpha, 0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args.config, args.checkpoint, device)
    image_paths = resolve_image_paths(args)

    images = load_and_preprocess_images(image_paths).to(device)  # [S, 3, H, W] in [0,1]
    S, _, H, W = images.shape
    patch_size = model.aggregator.patch_size
    patch_h, patch_w = H // patch_size, W // patch_size
    print(f"[shape] S={S} H={H} W={W} -> patch grid {patch_h}x{patch_w} = {patch_h * patch_w} per view")

    # head.decoder (SMPLRotTransformerDecoderHead)
    #   .transformer          -> TransformerDecoder
    #     .transformer        -> TransformerCrossAttn (has .layers with the cross-attn)
    decoder_transformer = model.smpl_multi_query_trans_rot_head.decoder.transformer.transformer
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    with CrossAttnCapture(decoder_transformer) as cap:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = model(images)

    if any(c is None for c in cap.captured):
        raise RuntimeError("Cross-attention was not captured -- did the SMPL head run?")

    attn = select_attention(cap.captured, args.layer)  # [heads, people, S*P]
    heads, num_people, SP = attn.shape
    assert SP == S * patch_h * patch_w, f"S*P mismatch: {SP} vs {S * patch_h * patch_w}"
    attn = attn.mean(dim=0)  # average heads -> [people, S*P]
    attn_grid = attn.reshape(num_people, S, patch_h, patch_w).numpy()

    # Decide which people to draw.
    presence_logits = predictions.get("smpl_presence_logits")
    if presence_logits is not None:
        presence_prob = torch.sigmoid(presence_logits.float())[0].cpu().numpy()  # [people]
    else:
        presence_prob = np.ones(num_people, dtype=np.float32)

    if args.all_people:
        people = list(range(num_people))
    else:
        people = [i for i in range(num_people) if presence_prob[i] >= args.presence_threshold]
        if not people:
            top = int(np.argmax(presence_prob))
            people = [top]
            print(f"[presence] none >= {args.presence_threshold}; falling back to top slot {top}")
    print(f"[people] visualizing {len(people)} person slots: {people}")

    # Base images as uint8 RGB HWC.
    base_imgs = [
        (images[s].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for s in range(S)
    ]

    layer_tag = args.layer
    for p in people:
        prob = presence_prob[p]
        # Shared normalization across this person's views so brightness is comparable.
        pmax = attn_grid[p].max()
        row_tiles = []
        for s in range(S):
            heat = attn_grid[p, s]
            if pmax > 0:
                heat = heat / pmax  # normalize per-person across views
            overlay = overlay_heatmap(base_imgs[s], heat, args.alpha)
            label = f"view{s}"
            cv2.putText(overlay, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            out_path = os.path.join(args.out_dir, f"person{p:02d}_layer-{layer_tag}_view{s:02d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            row_tiles.append(overlay)

        # Per-person grid across all views.
        grid = make_grid(row_tiles)
        title = f"person {p:02d}  presence={prob:.2f}  layer={layer_tag}"
        grid = add_title_bar(grid, title)
        grid_path = os.path.join(args.out_dir, f"person{p:02d}_layer-{layer_tag}_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[write] {grid_path}")

    print(f"\nDone. Outputs in: {args.out_dir}")


def make_grid(tiles: list[np.ndarray], max_cols: int = 4, pad: int = 4) -> np.ndarray:
    n = len(tiles)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = t
    return canvas


def add_title_bar(img: np.ndarray, title: str, bar_h: int = 34) -> np.ndarray:
    bar = np.full((bar_h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


if __name__ == "__main__":
    main()
