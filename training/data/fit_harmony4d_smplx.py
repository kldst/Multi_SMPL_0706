#!/usr/bin/env python
"""Offline Harmony4D SMPL -> body-only SMPL-X fitting.

The deformation-transfer matrix first maps each stored 6890-vertex SMPL mesh
onto the 10475-vertex SMPL-X topology.  We then optimize the exact differentiable
SMPL-X implementation used by this repository.  Results are cached outside the
source dataset and can be loaded by :class:`Harmony4DDataset`.

This is deliberately an offline command, not work performed in DataLoader
workers.  Fitting inside ``__getitem__`` would repeat the optimization every
epoch and make cache writes race between workers.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Collection, Optional

import numpy as np
import torch

from training.smpl_body import _get_smplx_model, set_smplx_model_root


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)
DEFAULT_OUTPUT = REPO / "mamma" / "harmony4d_smplx_fit"
DEFAULT_TRANSFER = REPO / "mamma" / "model_transfer"
DEFAULT_MODELS = REPO / "mamma" / "smplx_models"
PEOPLE = ("aria01", "aria02")


def load_transfer(transfer_root: Path) -> tuple[object, np.ndarray]:
    matrix_path = transfer_root / "smpl2smplx_deftrafo_setup.pkl"
    mask_path = transfer_root / "smplx_mask_ids.npy"
    with matrix_path.open("rb") as stream:
        setup = pickle.load(stream, encoding="latin1")
    if "mtx" not in setup:
        raise KeyError(f"mtx missing from {matrix_path}")
    matrix = setup["mtx"].tocsr()
    if matrix.shape != (10475, 2 * 6890):
        raise ValueError(f"unexpected transfer shape: {matrix.shape}")
    matrix = matrix[:, :6890]
    mask = np.load(mask_path, allow_pickle=False).astype(np.int64)
    if mask.ndim != 1 or not ((mask >= 0) & (mask < 10475)).all():
        raise ValueError(f"invalid SMPL-X mask: {mask.shape}")
    return matrix, mask


def surface_metrics_mm(pred: torch.Tensor, target: torch.Tensor) -> dict:
    errors = torch.linalg.vector_norm(pred - target, dim=-1).detach() * 1000.0
    return {
        "mean_mm": float(errors.mean().cpu()),
        "median_mm": float(errors.median().cpu()),
        "p95_mm": float(torch.quantile(errors, 0.95).cpu()),
        "max_mm": float(errors.max().cpu()),
    }


def fit_person(
    person: dict,
    transfer_matrix,
    mask_np: np.ndarray,
    model,
    device: torch.device,
    steps: int,
    metric_steps: Optional[Collection[int]] = None,
) -> tuple[dict, dict]:
    source_vertices = np.asarray(person["vertices"], dtype=np.float64)
    if source_vertices.shape != (6890, 3):
        raise ValueError(f"expected SMPL vertices (6890,3), got {source_vertices.shape}")
    target_np = np.asarray(transfer_matrix @ source_vertices, dtype=np.float32)
    target = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.long)

    global_orient = torch.nn.Parameter(
        torch.as_tensor(person["global_orient"], dtype=torch.float32, device=device)
        .reshape(1, 3)
        .clone()
    )
    source_body = (
        torch.as_tensor(person["body_pose"], dtype=torch.float32, device=device)
        .reshape(1, 69)[:, :63]
        .clone()
    )
    body_pose = torch.nn.Parameter(source_body.clone())
    betas = torch.nn.Parameter(
        torch.as_tensor(person["betas"], dtype=torch.float32, device=device)
        .reshape(1, -1)[:, :10]
        .clone()
    )
    transl = torch.nn.Parameter(
        torch.as_tensor(person["transl"], dtype=torch.float32, device=device)
        .reshape(1, 3)
        .clone()
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [global_orient, body_pose], "lr": 3e-3},
            {"params": [betas, transl], "lr": 1e-2},
        ]
    )

    requested_steps = {int(step) for step in (metric_steps or ())}
    requested_steps.add(int(steps))
    history = {}
    start_time = time.perf_counter()
    for step in range(max(1, int(steps)) + 1):
        optimizer.zero_grad(set_to_none=True)
        # SMPL-X has 21 body joints (63 dims). The final six slots in this
        # repository's 72-dim head are intentionally zero: copying SMPL's two
        # hand joints there would rotate SMPL-X jaw/eye joints.
        pose72 = torch.cat(
            [
                global_orient,
                body_pose,
                torch.zeros(1, 6, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )
        joints, vertices = model(pose72, betas, transl, with_vertices=True)
        pred_body = vertices[:, mask]
        target_body = target[:, mask]
        if step == 0 or step in requested_steps:
            checkpoint_metrics = surface_metrics_mm(pred_body, target_body)
            checkpoint_metrics["elapsed_seconds"] = time.perf_counter() - start_time
            history[str(step)] = checkpoint_metrics
        distance = torch.sqrt(
            (pred_body - target_body).square().sum(dim=-1) + 1e-8
        ).mean()
        pose_prior = (body_pose - source_body).square().mean()
        shape_prior = betas.square().mean()
        loss = distance + 1e-5 * pose_prior + 1e-5 * shape_prior
        if step < int(steps):
            loss.backward()
            optimizer.step()

    final_metrics = dict(history[str(int(steps))])
    final_metrics["initial_mean_mm"] = history["0"]["mean_mm"]
    final_metrics["steps"] = int(steps)
    final_metrics["history"] = history
    body_pose69 = torch.cat(
        [body_pose.detach(), torch.zeros(1, 6, device=device)], dim=-1
    )
    annotation = {
        "global_orient": global_orient.detach().cpu().numpy().reshape(3),
        "body_pose": body_pose69.cpu().numpy().reshape(69),
        "betas": betas.detach().cpu().numpy().reshape(10),
        "transl": transl.detach().cpu().numpy().reshape(3),
        "joints": joints.detach().cpu().numpy().reshape(24, 3),
        "fit_metrics": final_metrics,
    }
    return annotation, final_metrics


def atomic_save_npy(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npy", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.save(stream, value, allow_pickle=True)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sequence", default="01_hugging/001_hugging")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--frames", nargs="*")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset_root = args.dataset_root.expanduser().resolve()
    sequence = dataset_root / args.sequence
    source_dir = sequence / "processed_data" / "smpl"
    if args.frames:
        frames = [str(frame) for frame in args.frames]
    else:
        frames = [path.stem for path in sorted(source_dir.glob("*.npy"))]
        frames = frames[: max(0, int(args.max_frames))]
    if not frames:
        raise ValueError(f"no source annotations in {source_dir}")

    transfer_matrix, mask = load_transfer(args.transfer_root.expanduser().resolve())
    set_smplx_model_root(args.model_root.expanduser().resolve())
    model = _get_smplx_model(device, "neutral")
    reports = []
    for frame in frames:
        source_path = source_dir / f"{frame}.npy"
        output_path = args.output_root / args.sequence / f"{frame}.npy"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path}")
            continue
        source = np.load(source_path, allow_pickle=True).item()
        output = {}
        frame_report = {"frame": frame, "source": str(source_path), "people": {}}
        for key in PEOPLE:
            if key not in source:
                raise KeyError(f"{key} missing from {source_path}")
            annotation, metrics = fit_person(
                source[key], transfer_matrix, mask, model, device, args.steps
            )
            output[key] = annotation
            frame_report["people"][key] = metrics
        output["_meta"] = {
            "model_type": "smplx_body_only",
            "gender": "neutral",
            "source": str(source_path),
            "pose_layout": "global_orient(3)+body_pose(63)+zeros(6)",
        }
        atomic_save_npy(output_path, output)
        reports.append(frame_report)
        print(json.dumps(frame_report, indent=2))

    report_path = args.output_root / args.sequence / "fit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"[output] {args.output_root / args.sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
