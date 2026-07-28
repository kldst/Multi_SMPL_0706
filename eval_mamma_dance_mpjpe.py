#!/usr/bin/env python3
"""Standalone frame-by-frame MPJPE evaluation for organized MAMMA datasets.

Expected layout:
    <dataset-root>/<split>/out_image/runs_XXXXX/IOI_YY.jpg
    <dataset-root>/<split>/out_data/runs_XXXXX.npz

This script does not import or launch the Gradio demo. It loads the model once,
runs every ``runs_*`` frame, presence-filters predicted person slots, matches
them to GT people with a root-aligned joint cost, and reports MPJPE/PA-MPJPE.
"""

import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

REPO_DIR = Path(__file__).resolve().parent
TRAINING_DIR = REPO_DIR / "training"
for path in (REPO_DIR, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Compatibility for old SMPL/chumpy model files.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for name, value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if name not in np.__dict__:
        setattr(np, name, value)
if not hasattr(np, "unicode"):
    np.unicode = str

from training.loss import _decode_smpl_batch  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every organized MAMMA frame with root-aligned MPJPE and PA-MPJPE."
        )
    )
    parser.add_argument("--config", default="mamma_mask_direct_cam")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_DIR / "model/direct_cam/checkpoint_17.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_DIR / "MAMMA_eval_dance"),
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--image-ids",
        default="0 1 2 3",
        help="Zero-based indices into each frame's sorted image list.",
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection-mode",
        choices=("threshold", "topk_gt"),
        default="threshold",
        help=(
            "threshold: keep slots above --presence-threshold. "
            "topk_gt: rank slots by presence probability and keep exactly the "
            "number of GT people in that frame (e.g. top 3 for a three-person scene)."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_DIR
            / "eval/eval_results/direct_cam_checkpoint_17_mamma_dance_summary.json"
        ),
        help="Summary JSON path. The per-frame CSV uses the same stem.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug-only frame limit. Omit to evaluate every runs_* frame.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first bad frame instead of recording it and continuing.",
    )
    return parser.parse_args()


def run_sort_key(path: Path) -> tuple[int, object]:
    match = re.fullmatch(r"runs_(\d+)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def parse_image_ids(value: str) -> list[int]:
    ids = [int(token) for token in re.split(r"[\s,]+", value.strip()) if token]
    if not ids or any(index < 0 for index in ids):
        raise ValueError("--image-ids must contain one or more non-negative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("--image-ids contains duplicate indices")
    return ids


def discover_frames(dataset_root: Path, split: str) -> list[Path]:
    image_root = dataset_root / split / "out_image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    frames = sorted(
        (path for path in image_root.iterdir() if path.is_dir() and path.name.startswith("runs_")),
        key=run_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No runs_* frame directories found: {image_root}")
    return frames


def select_frame_images(frame_dir: Path, image_ids: list[int]) -> list[str]:
    images = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max(image_ids) >= len(images):
        raise IndexError(
            f"{frame_dir.name} has {len(images)} images but image index {max(image_ids)} was requested"
        )
    return [str(images[index]) for index in image_ids]


def normalize_gender(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else "neutral"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().lower()
    if text.startswith("m") or text == "0":
        return "male"
    if text.startswith("f") or text == "1":
        return "female"
    return "neutral"


def orthonormalize_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float64).reshape(3, 4).copy()
    rotation = extrinsic[:, :3]
    determinant = float(np.linalg.det(rotation))
    scale = np.sign(determinant) * abs(determinant) ** (1.0 / 3.0)
    if abs(scale) < 1e-9:
        scale = 1.0
    extrinsic[:, :3] /= scale
    extrinsic[:, 3] /= scale
    return extrinsic


def load_frame_gt(npz_path: Path, first_view_stem: str) -> dict:
    if not npz_path.is_file():
        raise FileNotFoundError(f"GT archive not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        keys = set(archive.files)
        pose_keys = sorted(
            key
            for key in keys
            if key.startswith("out_param/") and key.endswith("/smpl_params/poses")
        )
        if not pose_keys:
            raise KeyError(f"No out_param/*/smpl_params/poses in {npz_path}")

        poses, betas, trans, genders, people = [], [], [], [], []
        for pose_key in pose_keys:
            person_key = pose_key[: -len("/poses")]
            poses.append(np.asarray(archive[pose_key], dtype=np.float32).reshape(-1)[:72])
            betas.append(
                np.asarray(archive[f"{person_key}/betas"], dtype=np.float32).reshape(-1)[:10]
            )
            trans_key = f"{person_key}/trans"
            trans.append(
                np.asarray(archive[trans_key], dtype=np.float32).reshape(-1)[:3]
                if trans_key in keys
                else np.zeros(3, dtype=np.float32)
            )
            gender_key = f"{person_key}/gender"
            genders.append(
                normalize_gender(archive[gender_key] if gender_key in keys else "neutral")
            )
            people.append(person_key[len("out_param/") : -len("/smpl_params")])

        camera_views = sorted(
            {key.split("/", 2)[1] for key in keys if key.startswith("cam_param_min/")}
        )
        view = first_view_stem if first_view_stem in camera_views else None
        if view is None:
            matches = [candidate for candidate in camera_views if first_view_stem.endswith(candidate)]
            view = max(matches, key=len) if matches else None
        if view is None:
            raise KeyError(f"Camera for {first_view_stem} not found in {npz_path}")
        extrinsic_key = f"cam_param_min/{view}/extrinsics.worldToCamera12"
        if extrinsic_key not in keys:
            raise KeyError(f"Missing {extrinsic_key} in {npz_path}")
        extrinsic = np.asarray(archive[extrinsic_key], dtype=np.float64)
        extrinsic = extrinsic[:3, :4] if extrinsic.shape == (4, 4) else extrinsic.reshape(3, 4)

    return {
        "pose": np.stack(poses),
        "beta": np.stack(betas),
        "trans": np.stack(trans),
        "genders": genders,
        "people": people,
        "camera0_extrinsic": orthonormalize_extrinsic(extrinsic),
    }


def load_model(config_name: str, checkpoint_path: Path, device: torch.device):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(TRAINING_DIR / "config"),
    ):
        cfg = compose(config_name=config_name)
    model = instantiate(cfg.model, _recursive_=False)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)

    active_smpl_heads = [
        name
        for name in (
            "smpl_head",
            "smpl_multi_query_head",
            "smpl_multi_query_trans_head",
            "smpl_multi_query_trans_rot_head",
        )
        if getattr(model, name, None) is not None
    ]
    if not active_smpl_heads:
        raise RuntimeError(f"Config {config_name} does not enable an SMPL prediction head")
    critical_missing = [
        key
        for key in incompatible.missing_keys
        if any(key.startswith(f"{head}.") for head in active_smpl_heads)
    ]
    if critical_missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(critical_missing)} active SMPL-head parameters "
            f"(first: {critical_missing[0]})"
        )

    # These heads do not affect SMPL pose/beta/person queries. Disabling them avoids
    # dense maps, camera iterations, and full-resolution mask tensors during eval.
    for head_name in (
        "camera_head",
        "depth_head",
        "point_head",
        "track_head",
        "person_mask_head",
        "smpl_dense_landmark_head",
    ):
        if hasattr(model, head_name):
            setattr(model, head_name, None)

    model.eval().to(device)
    return model, cfg, incompatible


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def decode_joints(
    poses: np.ndarray,
    betas: np.ndarray,
    genders: list[str],
    device: torch.device,
) -> np.ndarray:
    count = int(poses.shape[0])
    if count == 0:
        return np.empty((0, 24, 3), dtype=np.float64)
    pose_tensor = torch.as_tensor(poses, dtype=torch.float32, device=device)
    beta_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
    zero_trans = torch.zeros((count, 3), dtype=torch.float32, device=device)
    joints, _ = _decode_smpl_batch(
        pose_aa=pose_tensor,
        betas=beta_tensor,
        trans=zero_trans,
        genders=genders,
        use_mamma=False,
    )
    if joints is None:
        raise RuntimeError("SMPL decoder returned no joints")
    return joints[:, :24].detach().cpu().numpy().astype(np.float64)


def root_align(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    return joints - joints[..., :1, :]


def similarity_align(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pred_mean = pred.mean(axis=0, keepdims=True)
    gt_mean = gt.mean(axis=0, keepdims=True)
    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean
    u, singular_values, vh = np.linalg.svd(pred_centered.T @ gt_centered)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    variance = max(float(np.sum(pred_centered**2)), eps)
    scale = float(np.sum(singular_values)) / variance
    return scale * (pred_centered @ rotation) + gt_mean


def match_and_measure(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_aligned = root_align(pred_joints)
    gt_aligned = root_align(gt_joints)
    pairwise_cost = np.linalg.norm(
        pred_aligned[:, None, :, :] - gt_aligned[None, :, :, :],
        axis=-1,
    ).mean(axis=-1)
    pred_indices, gt_indices = linear_sum_assignment(pairwise_cost)

    matched_pred = pred_aligned[pred_indices]
    matched_gt = gt_aligned[gt_indices]
    mpjpe_mm = np.linalg.norm(matched_pred - matched_gt, axis=-1) * 1000.0
    pa_mpjpe_mm = np.stack(
        [
            np.linalg.norm(similarity_align(pred, gt) - gt, axis=-1) * 1000.0
            for pred, gt in zip(matched_pred, matched_gt)
        ]
    )
    return pred_indices, gt_indices, mpjpe_mm, pa_mpjpe_mm


def evaluate_frame(
    model,
    frame_dir: Path,
    data_root: Path,
    image_ids: list[int],
    threshold: float,
    selection_mode: str,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    image_paths = select_frame_images(frame_dir, image_ids)
    gt = load_frame_gt(data_root / f"{frame_dir.name}.npz", Path(image_paths[0]).stem)
    images = load_and_preprocess_images(image_paths).to(device)

    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            predictions = model(images)

    poses = predictions["smpl_pose"][0].float().cpu().numpy()
    betas = predictions["smpl_beta"][0].float().cpu().numpy()
    logits_tensor = predictions.get("smpl_presence_logits")
    probabilities = (
        stable_sigmoid(logits_tensor[0].float().cpu().numpy())
        if logits_tensor is not None
        else np.ones((poses.shape[0],), dtype=np.float64)
    )
    gt_count = int(gt["pose"].shape[0])
    if selection_mode == "topk_gt":
        topk = min(gt_count, int(probabilities.size))
        selected = np.argsort(-probabilities, kind="stable")[:topk]
    else:
        selected = np.flatnonzero(probabilities >= threshold)
    selected_count = int(selected.size)

    row = {
        "frame": frame_dir.name,
        "status": "ok",
        "views": len(image_paths),
        "gt_people": gt_count,
        "selected_people": selected_count,
        "matched_people": 0,
        "miss_count": gt_count,
        "fp_count": selected_count,
        "max_presence_probability": float(probabilities.max()) if probabilities.size else float("nan"),
        "selection_mode": selection_mode,
        "mpjpe_mm": float("nan"),
        "pa_mpjpe_mm": float("nan"),
        "error": "",
    }
    if selected_count == 0 or gt_count == 0:
        return row, np.empty(0), np.empty(0)

    pred_joints = decode_joints(
        poses[selected],
        betas[selected],
        ["neutral"] * selected_count,
        device,
    )
    gt_joints = decode_joints(gt["pose"], gt["beta"], gt["genders"], device)

    rotation0 = gt["camera0_extrinsic"][:, :3]
    if predictions.get("mesh_rot") is not None:
        # The trans-rot head's predicted root orientation is already in camera0.
        gt_joints = root_align(gt_joints) @ rotation0.T
    else:
        # Other heads decode both prediction and GT in the world orientation.
        pred_joints = root_align(pred_joints) @ rotation0.T
        gt_joints = root_align(gt_joints) @ rotation0.T

    pred_match, gt_match, mpjpe_values, pa_values = match_and_measure(
        pred_joints,
        gt_joints,
    )
    matched = int(len(pred_match))
    row.update(
        matched_people=matched,
        miss_count=gt_count - matched,
        fp_count=selected_count - matched,
        mpjpe_mm=float(mpjpe_values.mean()),
        pa_mpjpe_mm=float(pa_values.mean()),
        matched_pred_slots=" ".join(str(int(selected[index])) for index in pred_match),
        matched_gt_people=" ".join(str(int(index)) for index in gt_match),
    )
    return row, mpjpe_values.reshape(-1), pa_values.reshape(-1)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    image_ids = parse_image_ids(args.image_ids)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    data_root = dataset_root / args.dataset_split / "out_data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"GT data root not found: {data_root}")

    frames = discover_frames(dataset_root, args.dataset_split)
    total_discovered_frames = len(frames)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        frames = frames[: args.max_frames]

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, cfg, incompatible = load_model(args.config, checkpoint_path, device)
    output_json = Path(args.output).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_json.with_suffix(".csv")

    print(f"[EVAL] config={args.config}")
    print(f"[EVAL] checkpoint={checkpoint_path}")
    print(f"[EVAL] dataset={dataset_root / args.dataset_split}")
    print(
        f"[EVAL] frames={len(frames)}/{total_discovered_frames} "
        f"views={image_ids} selection={args.selection_mode} threshold={args.presence_threshold}"
    )
    print(
        f"[EVAL] checkpoint missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )

    fieldnames = [
        "frame",
        "status",
        "views",
        "gt_people",
        "selected_people",
        "matched_people",
        "miss_count",
        "fp_count",
        "max_presence_probability",
        "selection_mode",
        "mpjpe_mm",
        "pa_mpjpe_mm",
        "matched_pred_slots",
        "matched_gt_people",
        "error",
    ]
    all_mpjpe, all_pa = [], []
    totals = {"gt": 0, "selected": 0, "matched": 0, "miss": 0, "fp": 0, "failed": 0}
    started = time.time()

    # Write and flush each row immediately, so a long all-frame run leaves useful
    # partial results even if interrupted.
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()

        for frame_index, frame_dir in enumerate(frames, 1):
            try:
                row, mpjpe_values, pa_values = evaluate_frame(
                    model,
                    frame_dir,
                    data_root,
                    image_ids,
                    args.presence_threshold,
                    args.selection_mode,
                    device,
                )
                all_mpjpe.extend(mpjpe_values.tolist())
                all_pa.extend(pa_values.tolist())
                totals["gt"] += int(row["gt_people"])
                totals["selected"] += int(row["selected_people"])
                totals["matched"] += int(row["matched_people"])
                totals["miss"] += int(row["miss_count"])
                totals["fp"] += int(row["fp_count"])
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "frame": frame_dir.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                totals["failed"] += 1
                print(f"[EVAL][WARN] {frame_dir.name}: {row['error']}")

            writer.writerow(row)
            csv_file.flush()
            if args.log_every > 0 and (
                frame_index == 1
                or frame_index % args.log_every == 0
                or frame_index == len(frames)
            ):
                elapsed = time.time() - started
                rate = frame_index / max(elapsed, 1e-6)
                eta = (len(frames) - frame_index) / max(rate, 1e-6)
                print(
                    f"[EVAL] {frame_index}/{len(frames)} {frame_dir.name} "
                    f"status={row['status']} MPJPE={row.get('mpjpe_mm', float('nan'))} "
                    f"PA={row.get('pa_mpjpe_mm', float('nan'))} "
                    f"rate={rate:.3f} frame/s ETA={eta / 60.0:.1f} min"
                )

    precision = totals["matched"] / totals["selected"] if totals["selected"] else 0.0
    recall = totals["matched"] / totals["gt"] if totals["gt"] else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_root / args.dataset_split),
        "image_ids": image_ids,
        "presence_threshold": args.presence_threshold,
        "selection_mode": args.selection_mode,
        "all_frames_requested": args.max_frames is None,
        "discovered_frames": total_discovered_frames,
        "evaluated_frames": len(frames),
        "successful_frames": len(frames) - totals["failed"],
        "failed_frames": totals["failed"],
        "gt_people": totals["gt"],
        "selected_predicted_people": totals["selected"],
        "matched_people": totals["matched"],
        "miss_count": totals["miss"],
        "fp_count": totals["fp"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mpjpe_root_aligned_mm": float(np.mean(all_mpjpe)) if all_mpjpe else None,
        "pa_mpjpe_mm": float(np.mean(all_pa)) if all_pa else None,
        "joint_count": len(all_mpjpe),
        "elapsed_seconds": time.time() - started,
        "per_frame_csv": str(output_csv),
        "coordinate_note": (
            "First 24 SMPL joints; joint 0 root alignment. GT root orientation is "
            "rotated into camera0 for mesh_rot checkpoints. Units are millimetres."
        ),
        "config_scale_by_extrinsics": bool(
            OmegaConf.select(cfg, "scale_by_extrinsics", default=True)
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[RESULT] MPJPE: {summary['mpjpe_root_aligned_mm']} mm")
    print(f"[RESULT] PA-MPJPE: {summary['pa_mpjpe_mm']} mm")
    print(f"[RESULT] precision={precision:.6f} recall={recall:.6f} F1={f1:.6f}")
    print(f"[RESULT] summary={output_json}")
    print(f"[RESULT] per_frame={output_csv}")
    if not all_mpjpe:
        raise RuntimeError(
            "No matched joint errors were produced. Check failed frames and presence threshold."
        )


if __name__ == "__main__":
    main()
