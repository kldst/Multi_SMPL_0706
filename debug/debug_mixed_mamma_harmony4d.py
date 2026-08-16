#!/usr/bin/env python
"""End-to-end smoke test for the mixed MAMMA + Harmony4D DataLoader.

The test deliberately creates one concat batch containing one sample from each
dataset.  It checks the collated schema, reprojects loader 3-D joints onto the
images, evaluates the configured loss with GT-as-prediction, and (by default)
runs the configured VGGT model once and verifies that all prediction losses are
finite.  Artifacts are written to ``debug/debug_dataset/MIX``.

The smoke test uses the production eight-view geometry and mask settings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO / "training"
for path in (str(REPO), str(TRAINING_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO)

# Compatibility for the legacy SMPL-X pickle dependencies in this environment.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if _name not in np.__dict__:
        setattr(np, _name, _value)
if not hasattr(np, "unicode"):
    np.unicode = str

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from debug.debug_harmony4d_dataset import (  # noqa: E402
    init_single_process_dist,
    loader_reprojection_metrics,
    make_gt_as_prediction,
    move_to_device,
    process_batch_like_trainer,
    scalar_metrics,
)
from training.smpl_body import set_smplx_model_root  # noqa: E402


DEFAULT_OUTPUT = REPO / "debug" / "debug_dataset" / "MIX"
DEFAULT_MAMMA_SEQUENCE = (
    REPO
    / "mamma"
    / "mamma"
    / "harmony4d_train_1_NC_200_00_contact"
    / "be_HsuS3iLSSWWZ_seq_000000"
)
DEFAULT_HARMONY_ROOT = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)


def source_name(seq_name: str) -> str:
    if seq_name.startswith("syssmpl_multi_"):
        return "mamma"
    if seq_name.startswith("harmony4d_"):
        return "harmony4d"
    return "unknown"


def tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def per_source_reprojection(
    batch: dict[str, Any], reprojection: torch.Tensor
) -> dict[str, dict[str, float | int | None]]:
    target = batch["smpl_joints2d"].float()
    valid = batch["smpl_joints2d_confidence"] > 0.5
    error = torch.linalg.vector_norm(reprojection - target, dim=-1)
    metrics = {}
    for batch_idx, seq_name in enumerate(batch["seq_name"]):
        values = error[batch_idx][valid[batch_idx]]
        metrics[source_name(seq_name)] = {
            "valid_joint_count": int(values.numel()),
            "mean_px": float(values.mean()) if values.numel() else None,
            "median_px": float(values.median()) if values.numel() else None,
            "max_px": float(values.max()) if values.numel() else None,
        }
    return metrics


def save_projection_overlays(
    batch: dict[str, Any], reprojection: torch.Tensor, output: Path
) -> list[str]:
    colors = ((0, 180, 255), (255, 80, 180), (80, 220, 80))
    saved = []
    for batch_idx, seq_name in enumerate(batch["seq_name"]):
        source = source_name(seq_name)
        source_dir = output / source
        source_dir.mkdir(parents=True, exist_ok=True)
        images = (
            batch["images"][batch_idx]
            .detach()
            .cpu()
            .permute(0, 2, 3, 1)
            .clamp(0, 1)
            .numpy()
        )
        images = (images * 255.0).round().astype(np.uint8)
        target = batch["smpl_joints2d"][batch_idx].cpu().numpy()
        predicted = reprojection[batch_idx].cpu().numpy()
        confidence = batch["smpl_joints2d_confidence"][batch_idx].cpu().numpy()
        masks = batch.get("person_mask")
        masks_np = masks[batch_idx].cpu().numpy() if masks is not None else None

        canvases = []
        for view_idx, rgb in enumerate(images):
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if masks_np is not None:
                for person_idx in range(masks_np.shape[1]):
                    foreground = masks_np[view_idx, person_idx] >= 0.5
                    if not np.any(foreground):
                        continue
                    color = np.asarray(colors[person_idx % len(colors)])
                    canvas[foreground] = (
                        0.72 * canvas[foreground] + 0.28 * color
                    ).astype(np.uint8)
            for person_idx in range(target.shape[1]):
                color = colors[person_idx % len(colors)]
                for joint_idx in range(target.shape[2]):
                    if confidence[view_idx, person_idx, joint_idx] <= 0.5:
                        continue
                    gt_xy = tuple(
                        np.rint(target[view_idx, person_idx, joint_idx]).astype(int)
                    )
                    projected_xy = tuple(
                        np.rint(predicted[view_idx, person_idx, joint_idx]).astype(int)
                    )
                    cv2.circle(canvas, gt_xy, 3, color, -1, cv2.LINE_AA)
                    cv2.drawMarker(
                        canvas,
                        projected_xy,
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        7,
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                f"{source} view={view_idx}: dots=GT, green-x=3D reprojection",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = source_dir / f"view_{view_idx:02d}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise RuntimeError(f"Failed to write {path}")
            saved.append(str(path))
            canvases.append(canvas)
        if canvases:
            cv2.imwrite(str(source_dir / "contact_sheet.jpg"), np.hstack(canvases))
    return saved


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "path": str(checkpoint_path),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
    }


@torch.no_grad()
def run_model_forward_and_loss(
    cfg: Any,
    batch: dict[str, Any],
    loss_module: torch.nn.Module,
    checkpoint: Path | None,
) -> dict[str, Any]:
    model = instantiate(cfg.model, _recursive_=False).to(batch["images"].device).eval()
    checkpoint_info = None
    if checkpoint is not None:
        checkpoint_info = load_checkpoint(model, checkpoint)
    device = batch["images"].device
    amp_enabled = device.type == "cuda" and bool(cfg.optim.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if str(cfg.optim.amp.amp_dtype).lower() == "bfloat16"
        else torch.float16
    )
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        predictions = model(batch["images"], smpl_inputs={})
        loss_dict = loss_module(predictions, batch)
    losses = scalar_metrics(loss_dict)
    return {
        "checkpoint": checkpoint_info,
        "prediction_shapes": {
            key: list(value.shape)
            for key, value in predictions.items()
            if torch.is_tensor(value)
        },
        "losses": losses,
        "all_losses_finite": bool(
            losses and all(np.isfinite(value) for value in losses.values())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mamma-sequence", type=Path, default=DEFAULT_MAMMA_SEQUENCE)
    parser.add_argument("--harmony-root", type=Path, default=DEFAULT_HARMONY_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model-forward", action="store_true")
    parser.add_argument("--reprojection-tolerance-px", type=float, default=2.0)
    parser.add_argument("--gt-loss-tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    mamma_sequence = args.mamma_sequence.expanduser().resolve()
    harmony_root = args.harmony_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    for path in (mamma_sequence, harmony_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    created_dist = init_single_process_dist()
    try:
        with initialize_config_dir(
            config_dir=str(TRAINING_DIR / "config"), version_base=None
        ):
            cfg = compose(config_name=args.config)

        with open_dict(cfg):
            # Two 8-view samples fit in one batch: concat index 0 is MAMMA and
            # index 1 is Harmony4D because shuffle is disabled for this test.
            cfg.num_workers = 0
            cfg.max_img_per_gpu = 16
            cfg.data.train.num_workers = 0
            cfg.data.train.max_img_per_gpu = 16
            cfg.data.train.shuffle = False
            cfg.data.train.pin_memory = False
            cfg.data.train.persistent_workers = False
            cfg.data.train.common_config.training = False
            cfg.data.train.common_config.fixed_view_sampling = True
            cfg.data.train.common_config.img_nums = [8, 8]
            cfg.data.train.common_config.include_metadata = False

            mamma_cfg, harmony_cfg = cfg.data.train.dataset.dataset_configs
            mamma_cfg.SysSMPL_DIR = str(mamma_sequence)
            mamma_cfg.SysSMPL_ANNOTATION_DIR = str(mamma_sequence)
            mamma_cfg.min_num_images = 8
            mamma_cfg.max_num_people = 6
            mamma_cfg.max_sequences = 1
            mamma_cfg.max_frames_per_sequence = 1

            # Keep the dataset root (rather than a narrowed sequence path) so
            # record.name remains ``activity/sequence`` and resolves the offline
            # mask bundle hierarchy correctly. max_sequences=1 selects hugging.
            harmony_cfg.Harmony4D_DIR = str(harmony_root)
            harmony_cfg.min_num_images = 8
            harmony_cfg.max_num_people = 6
            harmony_cfg.val_sequence_fraction = 0.0
            harmony_cfg.max_sequences = 1
            harmony_cfg.max_frames_per_sequence = 1

            cfg.loss.smplx_model_dir = str(REPO / "mamma" / "smplx_models")

        set_smplx_model_root(str(REPO / "mamma" / "smplx_models"))
        OmegaConf.save(cfg, output / "config_resolved.yaml")

        dynamic_dataset = instantiate(cfg.data.train, _recursive_=False)
        dynamic_dataset.seed = args.seed
        raw_batch = next(iter(dynamic_dataset.get_loader(epoch=0)))
        seq_names = list(raw_batch["seq_name"])
        sources = [source_name(name) for name in seq_names]
        expected_sources = {"mamma", "harmony4d"}
        source_pass = set(sources) == expected_sources and len(sources) == 2
        schema_pass = (
            raw_batch["images"].shape[:2] == (2, 8)
            and raw_batch["smpl_pose"].shape[:2] == (2, 6)
            and raw_batch["person_mask"].shape[:3] == (2, 8, 6)
            and "image_filenames" not in raw_batch
            and "cam_ids" not in raw_batch
        )

        overall_reprojection, reprojection = loader_reprojection_metrics(raw_batch)
        reprojection_by_source = per_source_reprojection(raw_batch, reprojection)
        reprojection_pass = all(
            metrics["max_px"] is not None
            and metrics["max_px"] <= args.reprojection_tolerance_px
            for metrics in reprojection_by_source.values()
        )
        overlays = save_projection_overlays(raw_batch, reprojection, output)

        processed = process_batch_like_trainer(
            raw_batch, scale_by_extrinsics=bool(cfg.scale_by_extrinsics)
        )
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        processed = move_to_device(processed, device)
        loss_module = instantiate(cfg.loss, _recursive_=False).to(device).eval()
        gt_prediction = make_gt_as_prediction(
            processed,
            model_people=int(cfg.model.smpl_num_people),
            use_mamma=bool(cfg.loss.smpl.use_mamma),
        )
        with torch.no_grad():
            gt_losses = scalar_metrics(loss_module(gt_prediction, processed))
        zero_loss_keys = (
            "loss_camera",
            "loss_T",
            "loss_R",
            "loss_FL",
            "loss_mesh_translate",
            "loss_smpl_joints3d",
            "loss_smpl_vertices",
            "loss_mask",
        )
        max_gt_core_loss = max(abs(gt_losses.get(key, 0.0)) for key in zero_loss_keys)
        gt_loss_pass = bool(
            gt_losses
            and all(np.isfinite(value) for value in gt_losses.values())
            and max_gt_core_loss <= args.gt_loss_tolerance
        )

        model_forward = {"skipped": True, "all_losses_finite": True}
        if not args.skip_model_forward:
            model_forward = run_model_forward_and_loss(
                cfg, processed, loss_module, checkpoint
            )
            model_forward["skipped"] = False

        passed = bool(
            source_pass
            and schema_pass
            and reprojection_pass
            and gt_loss_pass
            and model_forward["all_losses_finite"]
        )
        result = {
            "passed": passed,
            "config": args.config,
            "debug_override": {
                "views_per_sample": 8,
                "samples_per_batch": 2,
                "reason": "Production-equivalent eight-view mixed batch.",
            },
            "seq_names": seq_names,
            "sources": sources,
            "source_pass": source_pass,
            "schema_pass": schema_pass,
            "batch_shapes": tensor_shapes(raw_batch),
            "metadata_keys_present": {
                key: key in raw_batch for key in ("image_filenames", "cam_ids")
            },
            "reprojection": {
                "overall": overall_reprojection,
                "by_source": reprojection_by_source,
                "tolerance_px": args.reprojection_tolerance_px,
                "passed": reprojection_pass,
            },
            "gt_as_prediction": {
                "losses": gt_losses,
                "zero_loss_keys": list(zero_loss_keys),
                "max_core_loss": max_gt_core_loss,
                "tolerance": args.gt_loss_tolerance,
                "passed": gt_loss_pass,
            },
            "model_forward": model_forward,
            "overlays": overlays,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, default=str))
        print(f"[output] {output}")
        print(f"[result] {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 2
    finally:
        # Keep the process group alive while Python formats an active exception;
        # PyTorch's distributed exception hook queries the current rank.
        if (
            created_dist
            and torch.distributed.is_initialized()
            and sys.exc_info()[0] is None
        ):
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
