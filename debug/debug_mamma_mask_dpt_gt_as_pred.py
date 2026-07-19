#!/usr/bin/env python
"""One-batch integration test for ``mamma_mask_dpt.yaml``.

The test deliberately uses GT as the SMPL/camera/mask prediction.  It exercises
the real configured SysSMPLMultiDataset and DynamicTorchDataset, the same cam0
normalization used by Trainer._process_batch, the configured MultitaskLoss, and
optionally one actual VGGT forward pass (random weights unless --checkpoint is
given).  It also projects the GT-as-pred 24 joints and SMPL-X vertices back onto
the four processed training images.

Default input:
  mamma/mamma/harmony4d_train_1_NC_200_00_contact/
  be_HsuS3iLSSWWZ_seq_000000

Examples:
  conda activate mamma
  python debug/debug_mamma_mask_dpt_gt_as_pred.py
  python debug/debug_mamma_mask_dpt_gt_as_pred.py --scale-by-extrinsics false
  python debug/debug_mamma_mask_dpt_gt_as_pred.py --checkpoint model/checkpoint_49.pt
  python debug/debug_mamma_mask_dpt_gt_as_pred.py --skip-model-forward

Outputs are written under ``debug_outputs/mamma_mask_dpt_gt_as_pred``.
Green circles are loader GT joints2d, red crosses are GT-as-pred reprojections,
and cyan dots are a sparse projection of the reconstructed vertices.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist


REPO = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO / "training"
for path in (str(REPO), str(TRAINING_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO)

# chumpy compatibility used by the SMPL-X pickle loader.
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

from training import smpl_body as smpl_body_module  # noqa: E402
from training.smpl_body import (  # noqa: E402
    _decode_smpl_batch,
    _project_points_opencv,
    compute_gt_mesh_rot,
    compute_gt_mesh_translate,
    scale_joints_to_batch_gauge,
    set_smplx_model_root,
)
from training.train_utils.normalization import (  # noqa: E402
    normalize_camera_extrinsics_points_and_3djoints_batch,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


DEFAULT_SEQUENCE = (
    REPO
    / "mamma"
    / "mamma"
    / "harmony4d_train_1_NC_200_00_contact"
    / "be_HsuS3iLSSWWZ_seq_000000"
)
DEFAULT_OUT = REPO / "debug_outputs" / "mamma_mask_dpt_gt_as_pred"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_single_process_dist() -> bool:
    """DynamicTorchDataset's sampler expects torch.distributed to exist."""
    if dist.is_initialized():
        return False
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return True


def parse_bool_mode(value: str, config_value: bool) -> bool:
    value = value.strip().lower()
    if value == "config":
        return bool(config_value)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid --scale-by-extrinsics value: {value}")


def configure_smplx_models(configured_root: str | None) -> dict:
    """Use configured per-gender PKLs, or the repo's neutral NPZ fallback.

    The requested Harmony4D sample is neutral.  The fallback is intentionally
    limited to neutral instead of silently using a neutral model for male/female.
    """
    if configured_root:
        root = Path(configured_root).expanduser()
        expected = root / "neutral" / "model.pkl"
        if expected.is_file():
            set_smplx_model_root(root)
            return {"kind": "per-gender-pkl", "root": str(root), "neutral": str(expected)}

    neutral_npz = REPO / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not neutral_npz.is_file():
        raise FileNotFoundError(
            "No usable MAMMA SMPL-X model. Expected either "
            f"{configured_root}/neutral/model.pkl or {neutral_npz}"
        )
    smpl_body_module._SMPLX_MODEL_PATHS["neutral"] = str(neutral_npz)
    smpl_body_module._SMPLX_MODEL_CACHE.clear()
    return {"kind": "neutral-npz-fallback", "neutral": str(neutral_npz)}


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def process_batch_like_trainer(batch: dict, scale_by_extrinsics: bool) -> dict:
    """Mirror Trainer._process_batch without constructing a DDP Trainer."""
    out = dict(batch)
    raw_extrinsics = out["extrinsics"].clone()
    (
        norm_extrinsics,
        norm_cam_points,
        norm_world_points,
        norm_joints,
        norm_depths,
        avg_scale,
    ) = normalize_camera_extrinsics_points_and_3djoints_batch(
        extrinsics=out["extrinsics"],
        cam_points=out.get("cam_points"),
        world_points=out.get("world_points"),
        joints3d_world=out.get("smpl_joints3d_world"),
        depths=out.get("depths"),
        scale_by_extrinsics=scale_by_extrinsics,
        point_masks=out.get("point_masks"),
    )
    out["raw_extrinsics"] = raw_extrinsics
    out["extrinsics"] = norm_extrinsics
    out["avg_scale"] = avg_scale
    if norm_cam_points is not None:
        out["cam_points"] = norm_cam_points
    if norm_world_points is not None:
        out["world_points"] = norm_world_points
    if norm_joints is not None:
        out["smpl_joints3d_world"] = norm_joints
    if norm_depths is not None:
        out["depths"] = norm_depths
    return out


def pad_people(tensor: torch.Tensor, target_people: int, people_axis: int, fill: float = 0.0) -> torch.Tensor:
    current = tensor.shape[people_axis]
    if current >= target_people:
        return tensor
    shape = list(tensor.shape)
    shape[people_axis] = target_people - current
    pad = torch.full(shape, fill, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=people_axis)


@torch.no_grad()
def make_gt_as_prediction(batch: dict, cfg) -> dict:
    """Build exact targets in the output convention of the trans+rot head."""
    B, S = batch["images"].shape[:2]
    p_model = int(cfg.model.smpl_num_people)

    mesh_translate = compute_gt_mesh_translate(batch, normalize_cam=True, use_mamma=True)
    mesh_rot = compute_gt_mesh_rot(batch)
    pose = batch["smpl_pose"].clone()
    pose[..., :3] = mesh_rot.to(dtype=pose.dtype)

    pose = pad_people(pose, p_model, 1)
    beta = pad_people(batch["smpl_beta"].clone(), p_model, 1)
    mesh_translate = pad_people(mesh_translate, p_model, 1)
    has_smpl = pad_people(batch["has_smpl"].clone(), p_model, 1)

    gt_pose_encoding = extri_intri_to_pose_encoding(
        batch["extrinsics"],
        batch["intrinsics"],
        batch["images"].shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )

    # Exact GT masks as probabilities, converted to finite logits. Padded slots
    # are empty. This also exercises the configured Hungarian mask cost.
    gt_mask = batch.get("person_mask")
    mask_logits = None
    if gt_mask is not None:
        gt_mask = pad_people(gt_mask, p_model, 2)
        eps = 1e-6
        mask_logits = torch.logit(gt_mask.float().clamp(eps, 1.0 - eps))

    presence_logits = torch.where(
        has_smpl > 0.5,
        torch.full_like(has_smpl, 20.0),
        torch.full_like(has_smpl, -20.0),
    )
    predictions = {
        "pose_enc_list": [gt_pose_encoding],
        "pose_enc": gt_pose_encoding,
        "smpl_pose": pose,
        "pred_pose_0": pose.clone(),
        "smpl_beta": beta,
        "mesh_translate": mesh_translate,
        "mesh_rot": pose[..., :3],
        "smpl_presence_logits": presence_logits,
    }
    if mask_logits is not None:
        predictions["person_mask_logits"] = mask_logits
    return predictions


def scalar_metrics(losses: dict) -> dict[str, float]:
    result = {}
    for key, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().cpu().item())
    return result


@torch.no_grad()
def reconstruct_gt_as_pred_geometry(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce loss_smpl.py's mesh_rot + mesh_translate geometry path."""
    B, P = batch["smpl_pose"].shape[:2]
    mesh_rot = compute_gt_mesh_rot(batch)
    pose = batch["smpl_pose"].clone()
    pose[..., :3] = mesh_rot.to(pose.dtype)
    mesh_translate = compute_gt_mesh_translate(batch, normalize_cam=True, use_mamma=True)

    genders_raw = batch["smpl_gender"].detach().cpu().reshape(-1).tolist()
    gender_names = [{0: "male", 1: "female", 2: "neutral"}.get(int(g), "neutral") for g in genders_raw]
    joints_flat, verts_flat = _decode_smpl_batch(
        pose_aa=pose.reshape(B * P, 72).float(),
        betas=batch["smpl_beta"].reshape(B * P, 10).float(),
        trans=torch.zeros((B * P, 3), device=pose.device, dtype=torch.float32),
        genders=gender_names,
        use_mamma=True,
    )
    joints = joints_flat.reshape(B, P, joints_flat.shape[1], 3)
    verts = verts_flat.reshape(B, P, verts_flat.shape[1], 3)
    joints = scale_joints_to_batch_gauge(joints, batch["avg_scale"])
    verts = scale_joints_to_batch_gauge(verts, batch["avg_scale"])
    offset = mesh_translate.float() - joints[..., 0, :]
    joints = joints + offset.unsqueeze(-2)
    verts = verts + offset.unsqueeze(-2)
    return joints, verts


def draw_cross(image: np.ndarray, xy: np.ndarray, color: tuple[int, int, int], size: int = 5) -> None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 2, cv2.LINE_AA)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, 2, cv2.LINE_AA)


@torch.no_grad()
def save_projection_overlays(batch: dict, out_dir: Path) -> dict[str, float]:
    joints, vertices = reconstruct_gt_as_pred_geometry(batch)
    B, S = batch["images"].shape[:2]
    assert B == 1, f"Expected one sequence batch, got B={B}"
    P = batch["smpl_pose"].shape[1]
    has = batch["has_smpl"][0] > 0.5
    confidence = batch["smpl_joints2d_confidence"][0]
    gt_joints2d = batch["smpl_joints2d"][0]

    points = joints[:, None].expand(B, S, P, joints.shape[-2], 3)
    pred_joints2d = _project_points_opencv(
        points.reshape(B, S, P * joints.shape[-2], 3),
        batch["extrinsics"].float(),
        batch["intrinsics"].float(),
    ).reshape(B, S, P, joints.shape[-2], 2)[0]

    vert_stride = max(1, vertices.shape[-2] // 700)
    sparse_vertices = vertices[..., ::vert_stride, :]
    vpoints = sparse_vertices[:, None].expand(B, S, P, sparse_vertices.shape[-2], 3)
    pred_vertices2d = _project_points_opencv(
        vpoints.reshape(B, S, P * sparse_vertices.shape[-2], 3),
        batch["extrinsics"].float(),
        batch["intrinsics"].float(),
    ).reshape(B, S, P, sparse_vertices.shape[-2], 2)[0]

    valid = (confidence > 0.5) & has[None, :, None]
    errors = torch.linalg.vector_norm(pred_joints2d - gt_joints2d, dim=-1)
    valid_errors = errors[valid]
    metrics = {
        "valid_joint_count": int(valid_errors.numel()),
        "joint_reprojection_mean_px": float(valid_errors.mean().cpu()) if valid_errors.numel() else float("nan"),
        "joint_reprojection_max_px": float(valid_errors.max().cpu()) if valid_errors.numel() else float("nan"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    images = batch["images"][0].detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    images = (images * 255.0).round().astype(np.uint8)
    palette = [(255, 255, 0), (255, 128, 0), (160, 255, 0), (255, 0, 255)]
    for s in range(S):
        canvas = cv2.cvtColor(images[s], cv2.COLOR_RGB2BGR)
        H, W = canvas.shape[:2]
        for p in range(P):
            if not bool(has[p]):
                continue
            vertex_color = palette[p % len(palette)]
            for xy in pred_vertices2d[s, p].detach().cpu().numpy():
                x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(canvas, (x, y), 1, vertex_color, -1, cv2.LINE_AA)
            for j in range(min(24, gt_joints2d.shape[-2])):
                if not bool(confidence[s, p, j] > 0.5):
                    continue
                gt_xy = gt_joints2d[s, p, j].detach().cpu().numpy()
                pred_xy = pred_joints2d[s, p, j].detach().cpu().numpy()
                cv2.circle(canvas, tuple(np.rint(gt_xy).astype(int)), 4, (0, 255, 0), -1, cv2.LINE_AA)
                draw_cross(canvas, pred_xy, (0, 0, 255))
        cv2.putText(
            canvas,
            "green=GT joints2d, red-x=GT-as-pred, cyan/yellow=vertices",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"projection_view_{s:02d}.png"), canvas)
    return metrics


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"path": str(checkpoint_path), "missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def run_configured_model_forward(cfg, batch: dict, device: torch.device, checkpoint: Path | None) -> dict:
    model = instantiate(cfg.model, _recursive_=False).to(device).eval()
    checkpoint_info = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_info = load_checkpoint(model, checkpoint)
    amp_enabled = device.type == "cuda" and bool(cfg.optim.amp.enabled)
    amp_dtype = torch.bfloat16 if str(cfg.optim.amp.amp_dtype).lower() == "bfloat16" else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(batch["images"], smpl_inputs={})
    shapes = {
        key: list(value.shape)
        for key, value in predictions.items()
        if torch.is_tensor(value)
    }
    return {
        "checkpoint": checkpoint_info,
        "random_weights": checkpoint is None,
        "output_shapes": shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_mask_dpt")
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=1, help="Frames indexed from the target sequence.")
    parser.add_argument(
        "--scale-by-extrinsics",
        choices=("config", "true", "false"),
        default="config",
        help="Use YAML value (default True when absent), or explicitly test with/without avg_scale.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-model-forward", action="store_true")
    parser.add_argument("--loss-tolerance", type=float, default=2e-4)
    parser.add_argument("--projection-tolerance-px", type=float, default=2.0)
    args = parser.parse_args()

    args.sequence = args.sequence.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.sequence.is_dir():
        raise FileNotFoundError(args.sequence)
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    created_dist = init_single_process_dist()

    try:
        with initialize_config_dir(config_dir=str(TRAINING_DIR / "config"), version_base=None):
            cfg = compose(config_name=args.config)

        config_scale = bool(OmegaConf.select(cfg, "scale_by_extrinsics", default=True))
        scale_by_extrinsics = parse_bool_mode(args.scale_by_extrinsics, config_scale)

        # Keep the configured train pipeline, changing only the requested one-batch
        # limits and pointing its first dataset at this exact sequence directory.
        with open_dict(cfg):
            cfg.max_img_per_gpu = 4
            cfg.num_workers = 0
            cfg.data.train.max_img_per_gpu = 4
            cfg.data.train.num_workers = 0
            cfg.data.train.common_config.img_nums = [4, 4]
            cfg.data.train.common_config.fixed_view_sampling = True
            cfg.data.train.dataset.dataset_configs[0].SysSMPL_DIR = str(args.sequence)
            cfg.data.train.dataset.dataset_configs[0].SysSMPL_ANNOTATION_DIR = str(args.sequence)
            cfg.data.train.dataset.dataset_configs[0].max_sequences = 1
            cfg.data.train.dataset.dataset_configs[0].max_frames_per_sequence = int(args.max_frames)
            # Configure explicitly below. Keeping this None prevents
            # MultitaskLoss from overwriting the NPZ fallback with a missing
            # machine-specific /dataset/.../model.pkl path.
            configured_smplx_root = OmegaConf.select(cfg, "loss.smplx_model_dir", default=None)
            cfg.loss.smplx_model_dir = None

        smplx_model_info = configure_smplx_models(configured_smplx_root)

        print(f"[config] {args.config}; max_img_per_gpu=4; img_nums=[4,4]")
        print(f"[data] sequence={args.sequence}")
        print(f"[gauge] scale_by_extrinsics={scale_by_extrinsics}")
        print(f"[smplx] {smplx_model_info}")

        train_dataset = instantiate(cfg.data.train, _recursive_=False)
        train_dataset.seed = args.seed
        loader = train_dataset.get_loader(epoch=0)
        raw_batch = next(iter(loader))
        batch = process_batch_like_trainer(raw_batch, scale_by_extrinsics=scale_by_extrinsics)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        batch = move_to_device(batch, device)

        gt_predictions = make_gt_as_prediction(batch, cfg)
        loss_module = instantiate(cfg.loss, _recursive_=False).to(device).eval()
        with torch.no_grad():
            loss_dict = loss_module(gt_predictions, batch)
        losses = scalar_metrics(loss_dict)

        projection_metrics = save_projection_overlays(batch, args.output)
        # Pixel-space GT goes through integer crop/resize bookkeeping, so a
        # sub-pixel difference is expected and is judged by the explicit pixel
        # reprojection threshold below. These core 3D/parameter losses should be
        # numerically zero for GT-as-pred.
        zero_loss_keys = (
            "loss_camera",
            "loss_T",
            "loss_R",
            "loss_FL",
            "loss_smpl_losses",
            "loss_mesh_translate",
            "loss_smpl_joints3d",
            "loss_smpl_vertices",
        )
        max_core_loss = max(abs(losses.get(key, 0.0)) for key in zero_loss_keys)
        loss_pass = max_core_loss <= args.loss_tolerance
        reproj_max = projection_metrics["joint_reprojection_max_px"]
        projection_pass = np.isfinite(reproj_max) and reproj_max <= args.projection_tolerance_px

        model_forward = {"skipped": True}
        if not args.skip_model_forward:
            model_forward = run_configured_model_forward(cfg, batch, device, args.checkpoint)
            model_forward["skipped"] = False

        result = {
            "passed": bool(loss_pass and projection_pass),
            "config": args.config,
            "sequence": str(args.sequence),
            "seq_name": raw_batch.get("seq_name"),
            "image_paths": raw_batch.get("image_paths"),
            "batch_images_shape": list(batch["images"].shape),
            "max_img_per_gpu": 4,
            "scale_by_extrinsics": scale_by_extrinsics,
            "smplx_model": smplx_model_info,
            "avg_scale": batch["avg_scale"].detach().cpu().tolist(),
            "loss_tolerance": args.loss_tolerance,
            "max_core_loss": max_core_loss,
            "loss_pass": loss_pass,
            "losses": losses,
            "projection_tolerance_px": args.projection_tolerance_px,
            "projection_pass": bool(projection_pass),
            "projection": projection_metrics,
            "model_forward": model_forward,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

        print(json.dumps(result, indent=2, default=str))
        print(f"[output] {args.output}")
        print(f"[result] {'PASS' if result['passed'] else 'FAIL'}")
        return 0 if result["passed"] else 2
    finally:
        if created_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
