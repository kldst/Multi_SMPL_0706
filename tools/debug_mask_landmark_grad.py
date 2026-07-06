#!/usr/bin/env python3
"""Debug dense-landmark mask conditioning on one configured batch.

Example:
  python tools/debug_mask_landmark_grad.py \
    --config training/config/mamma_overfit.yaml \
    --device cuda \
    --checkpoint training/logs/mamma_overfit_newlandmark/ckpts/checkpoint_step_1500.pt

By default the script does not load the config's base checkpoint, so graph/shape
checks stay quick. Pass --checkpoint when you want to inspect real weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "training"))

from training.train_utils.freeze import freeze_modules  # noqa: E402
from training.train_utils.general import copy_data_to_device  # noqa: E402
from training.train_utils.normalization import (  # noqa: E402
    normalize_camera_extrinsics_points_and_3djoints_batch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(REPO_DIR / "training/config/mamma_overfit.yaml"),
        help="Path to a Hydra yaml config.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint to load. Empty means do not load cfg.checkpoint.resume_checkpoint_path.",
    )
    p.add_argument("--load-config-checkpoint", action="store_true")
    p.add_argument("--no-freeze", action="store_true", help="Do not apply cfg.optim.frozen_module_names.")
    return p.parse_args()


def load_cfg(config_path: str):
    path = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        return compose(config_name=path.stem)


def load_checkpoint_if_requested(model: torch.nn.Module, cfg, checkpoint: str, load_config_checkpoint: bool) -> None:
    ckpt_path = checkpoint.strip()
    if not ckpt_path and load_config_checkpoint:
        ckpt_path = str(OmegaConf.select(cfg, "checkpoint.resume_checkpoint_path") or "")
    if not ckpt_path:
        print("[ckpt] skipped")
        return

    ckpt_path = str(Path(ckpt_path).expanduser())
    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[ckpt] loaded missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:8]:
        print("[ckpt] missing sample:", missing[:8])
    if unexpected[:8]:
        print("[ckpt] unexpected sample:", unexpected[:8])


def first_batch(cfg):
    dataset = instantiate(cfg.data.train, _recursive_=False)
    loader = dataset.get_loader(epoch=0)
    return next(iter(loader))


def process_batch_like_trainer(batch, cfg):
    """Mirror Trainer._process_batch for a single debug batch.

    The SMPL loss expects raw_extrinsics/avg_scale when normalize_cam=True. The
    regular trainer adds those fields before moving tensors to CUDA, so the
    debug script must do the same.
    """
    if bool(OmegaConf.select(cfg, "data.train.common_config.repeat_batch") or False):
        for key in ("images", "depths", "extrinsics", "intrinsics", "cam_points", "world_points", "point_masks"):
            if key in batch:
                batch[key] = torch.concatenate([batch[key], torch.flip(batch[key], dims=[1])], dim=0)
        if "seq_name" in batch:
            batch["seq_name"] = batch["seq_name"] * 2

    normalize_cam = bool(OmegaConf.select(cfg, "loss.smpl.normalize_cam") if OmegaConf.select(cfg, "loss.smpl.normalize_cam") is not None else True)
    scale_by_extrinsics = bool(OmegaConf.select(cfg, "scale_by_extrinsics") if OmegaConf.select(cfg, "scale_by_extrinsics") is not None else True)

    if normalize_cam:
        (
            normalized_extrinsics,
            normalized_cam_points,
            normalized_world_points,
            normalized_joints3d_world,
            normalized_depths,
            avg_scale,
        ) = normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch.get("cam_points"),
            world_points=batch.get("world_points"),
            joints3d_world=batch.get("smpl_joints3d_world"),
            depths=batch.get("depths"),
            scale_by_extrinsics=scale_by_extrinsics,
            point_masks=batch.get("point_masks"),
        )
        batch["avg_scale"] = avg_scale
        batch["raw_extrinsics"] = batch["extrinsics"].clone()
        batch["extrinsics"] = normalized_extrinsics
        if normalized_cam_points is not None:
            batch["cam_points"] = normalized_cam_points
        if normalized_world_points is not None:
            batch["world_points"] = normalized_world_points
        if normalized_joints3d_world is not None:
            batch["smpl_joints3d_world"] = normalized_joints3d_world
        if normalized_depths is not None:
            batch["depths"] = normalized_depths
    else:
        B = batch["extrinsics"].shape[0]
        batch["avg_scale"] = torch.ones(B, device=batch["extrinsics"].device)
        batch["raw_extrinsics"] = batch["extrinsics"].clone()

    return batch


def tensor_summary(name: str, value) -> None:
    if torch.is_tensor(value):
        with torch.no_grad():
            msg = f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
            if value.is_floating_point():
                msg += f" min={value.min().item():.4g} max={value.max().item():.4g} mean={value.mean().item():.4g}"
            print(msg)


def grad_norm_for_prefixes(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "aggregator": 0.0,
        "smpl": 0.0,
        "landmark": 0.0,
        "person_mask_head": 0.0,
        "camera": 0.0,
        "other": 0.0,
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm_sq = float(p.grad.detach().float().norm().item() ** 2)
        if name.startswith("aggregator."):
            groups["aggregator"] += norm_sq
        elif name.startswith("smpl_dense_landmark_head."):
            groups["landmark"] += norm_sq
        elif name.startswith("person_mask_head."):
            groups["person_mask_head"] += norm_sq
        elif name.startswith("camera_head."):
            groups["camera"] += norm_sq
        elif name.startswith("smpl_"):
            groups["smpl"] += norm_sq
        else:
            groups["other"] += norm_sq
    return {k: v ** 0.5 for k, v in groups.items()}


def backward_one(model, loss, loss_name: str, scale: float = 1.0) -> None:
    model.zero_grad(set_to_none=True)
    if not torch.is_tensor(loss):
        print(f"[grad] {loss_name}: not a tensor, skipped")
        return
    objective = loss * float(scale)
    if not objective.requires_grad:
        print(f"[grad] {loss_name}: requires_grad=False, skipped")
        return
    objective.backward(retain_graph=True)
    norms = grad_norm_for_prefixes(model)
    print(f"[grad] {loss_name} * {scale:g}: " + ", ".join(f"{k}={v:.4g}" for k, v in norms.items()))


def main() -> None:
    args = parse_args()
    os.chdir(REPO_DIR)
    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    print("[cfg] model landmark_use_mask_embedding =", OmegaConf.select(cfg, "model.landmark_use_mask_embedding"))
    print("[cfg] model landmark_detach_mask_context =", OmegaConf.select(cfg, "model.landmark_detach_mask_context"))
    print("[cfg] loss weight_landmark =", OmegaConf.select(cfg, "loss.smpl.weight_landmark"))
    print("[cfg] loss weight_mask =", OmegaConf.select(cfg, "loss.smpl.weight_mask"))

    model = instantiate(cfg.model, _recursive_=False).to(device)
    if not args.no_freeze and getattr(cfg.optim, "frozen_module_names", None):
        model = freeze_modules(model, patterns=cfg.optim.frozen_module_names)
    load_checkpoint_if_requested(model, cfg, args.checkpoint, args.load_config_checkpoint)
    model.train()

    loss_fn = instantiate(cfg.loss, _recursive_=False).to(device)

    batch = process_batch_like_trainer(first_batch(cfg), cfg)
    batch = copy_data_to_device(batch, device)
    print("\n[batch]")
    for key in (
        "images",
        "extrinsics",
        "raw_extrinsics",
        "avg_scale",
        "smpl_landmarks2d",
        "smpl_landmarks2d_visibility",
        "person_mask",
        "has_smpl",
        "smpl_pose",
        "smpl_beta",
        "smpl_joints3d_world",
    ):
        if key in batch:
            tensor_summary(key, batch[key])

    smpl_inputs = {
        key: batch[key]
        for key in ("views_per_frame", "temporal_num_frames", "frame_ids", "view_ids")
        if key in batch
    }
    preds = model(images=batch["images"], smpl_inputs=smpl_inputs)

    print("\n[predictions]")
    for key in (
        "person_tokens",
        "person_mask_logits",
        "smpl_landmarks2d",
        "smpl_landmarks_logvar",
        "smpl_landmarks_visibility_logits",
        "smpl_pose",
        "smpl_beta",
        "mesh_translate",
    ):
        if key in preds:
            tensor_summary(key, preds[key])

    losses = loss_fn(preds, batch)
    print("\n[losses]")
    for key in (
        "loss_objective",
        "loss_smpl",
        "loss_smpl_losses",
        "loss_landmark",
        "loss_landmark_l2",
        "loss_landmark_vis",
        "landmark_px",
        "loss_mask",
        "mask_soft_iou",
    ):
        if key in losses:
            value = losses[key]
            print(f"{key}: {value.detach().item():.6g}" if torch.is_tensor(value) else f"{key}: {value}")

    print("\n[grad norms]")
    smpl_w = float(OmegaConf.select(cfg, "loss.smpl.weight") or 1.0)
    backward_one(model, losses.get("loss_smpl_losses"), "raw_smpl_terms", smpl_w)
    backward_one(
        model,
        losses.get("loss_landmark"),
        "landmark_only",
        smpl_w * float(OmegaConf.select(cfg, "loss.smpl.weight_landmark") or 0.0),
    )
    backward_one(
        model,
        losses.get("loss_landmark_vis"),
        "landmark_vis_only",
        smpl_w * float(OmegaConf.select(cfg, "loss.smpl.weight_landmark_vis") or 0.0),
    )
    backward_one(
        model,
        losses.get("loss_mask"),
        "mask_only",
        smpl_w * float(OmegaConf.select(cfg, "loss.smpl.weight_mask") or 0.0),
    )
    backward_one(model, losses.get("loss_objective"), "full_objective", 1.0)


if __name__ == "__main__":
    main()
