#!/usr/bin/env python3
"""Convert Harmony4D neutral-SMPL ground truth to body-only SMPL-X.

The stored Harmony4D SMPL mesh is first mapped to SMPL-X topology with the
official deformation-transfer matrix in ``mamma/model_transfer``.  SMPL-X
parameters are then optimized against the transferred body surface using the
same differentiable SMPL-X implementation as the MAMMA training path.

The output mirrors ``<activity>/<sequence>/<frame>.npy`` and contains both the
Harmony4D loader fields (global_orient/body_pose/betas/transl/joints) and the
MAMMA names (pose_world/shape/trans_world).  Fingers, jaw and eyes are zero,
because classic SMPL does not provide those rotations.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # Let --dry-run and --help work outside the training env.
    torch = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET_ROOT = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)
DEFAULT_TRANSFER_ROOT = REPO_ROOT / "mamma" / "model_transfer"
DEFAULT_MODEL_ROOT = REPO_ROOT / "mamma" / "smplx_models"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"
PEOPLE = ("aria01", "aria02")


def load_transfer_assets(transfer_root: Path):
    """Load and validate the SMPL->SMPL-X barycentric transfer operator."""
    setup_path = transfer_root / "smpl2smplx_deftrafo_setup.pkl"
    mask_path = transfer_root / "smplx_mask_ids.npy"
    with setup_path.open("rb") as stream:
        setup = pickle.load(stream, encoding="latin1")
    if not isinstance(setup, dict) or "mtx" not in setup:
        raise ValueError(f"No 'mtx' entry in {setup_path}")

    full_matrix = setup["mtx"].tocsr()
    if full_matrix.shape != (10475, 13780):
        raise ValueError(
            f"Expected transfer matrix (10475, 13780), got {full_matrix.shape}"
        )
    # This setup reserves two 6890-vertex blocks.  Its second block is empty;
    # keeping only the active block makes the intended (10475 x 6890) map clear.
    if full_matrix[:, 6890:].nnz:
        raise ValueError("Unexpected non-zero second block in transfer matrix")
    matrix = full_matrix[:, :6890].tocsr().astype(np.float32)

    mask = np.load(mask_path, allow_pickle=False).astype(np.int64)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError(f"Expected a non-empty 1-D mask, got {mask.shape}")
    if np.unique(mask).size != mask.size or np.any((mask < 0) | (mask >= 10475)):
        raise ValueError(f"Invalid SMPL-X body mask in {mask_path}")
    return matrix, mask


def _metrics_mm(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(pred - target, dim=-1).detach() * 1000.0
    return {
        "mean_mm": float(error.mean().cpu()),
        "median_mm": float(error.median().cpu()),
        "p95_mm": float(torch.quantile(error, 0.95).cpu()),
        "max_mm": float(error.max().cpu()),
    }


def fit_person(
    source: dict,
    transfer_matrix,
    body_mask_np: np.ndarray,
    model,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit one SMPL-X body and return annotation, target mesh, predicted mesh."""
    source_vertices = np.asarray(source["vertices"], dtype=np.float32)
    if source_vertices.shape != (6890, 3):
        raise ValueError(
            f"Expected source vertices (6890, 3), got {source_vertices.shape}"
        )
    target_np = np.asarray(transfer_matrix @ source_vertices, dtype=np.float32)
    target = torch.from_numpy(target_np).unsqueeze(0).to(device)
    body_mask = torch.from_numpy(body_mask_np).to(device=device, dtype=torch.long)

    global_orient = torch.nn.Parameter(
        torch.as_tensor(source["global_orient"], dtype=torch.float32, device=device)
        .reshape(1, 3)
        .clone()
    )
    source_body = (
        torch.as_tensor(source["body_pose"], dtype=torch.float32, device=device)
        .reshape(1, 69)[:, :63]
        .clone()
    )
    body_pose = torch.nn.Parameter(source_body.clone())
    source_betas = (
        torch.as_tensor(source["betas"], dtype=torch.float32, device=device)
        .reshape(1, -1)[:, :10]
        .clone()
    )
    betas = torch.nn.Parameter(source_betas.clone())
    transl = torch.nn.Parameter(
        torch.as_tensor(source["transl"], dtype=torch.float32, device=device)
        .reshape(1, 3)
        .clone()
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [global_orient, body_pose], "lr": learning_rate * 0.3},
            {"params": [betas, transl], "lr": learning_rate},
        ]
    )
    initial_metrics = None
    started = time.perf_counter()
    steps = max(0, int(steps))
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # The repository's MAMMA-compatible model accepts a 72-D head.  The
        # final 6 values are kept at zero rather than copying SMPL hand joints:
        # in SMPL-X joint order those slots would rotate jaw and eyes.
        pose72 = torch.cat(
            [
                global_orient,
                body_pose,
                torch.zeros(1, 6, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )
        joints, vertices = model(pose72, betas, transl, with_vertices=True)
        pred_body = vertices[:, body_mask]
        target_body = target[:, body_mask]
        if step == 0:
            initial_metrics = _metrics_mm(pred_body, target_body)
        if step == steps:
            break

        surface_loss = torch.sqrt(
            (pred_body - target_body).square().sum(dim=-1) + 1e-10
        ).mean()
        pose_prior = (body_pose - source_body).square().mean()
        # SMPL and SMPL-X shape bases differ, but the SMPL beta is still a much
        # better initialization/weak prior than an unconstrained framewise fit.
        shape_prior = (betas - source_betas).square().mean()
        loss = surface_loss + 1e-5 * pose_prior + 1e-6 * shape_prior
        loss.backward()
        optimizer.step()

    assert initial_metrics is not None
    final_metrics = _metrics_mm(pred_body, target_body)
    final_metrics.update(
        {
            "initial_mean_mm": initial_metrics["mean_mm"],
            "improvement_mm": initial_metrics["mean_mm"] - final_metrics["mean_mm"],
            "steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "evaluated_vertices": int(body_mask_np.size),
        }
    )

    global_np = global_orient.detach().cpu().numpy().reshape(3)
    body63_np = body_pose.detach().cpu().numpy().reshape(63)
    body69_np = np.zeros(69, dtype=np.float32)
    body69_np[:63] = body63_np
    pose165_np = np.zeros(165, dtype=np.float32)
    pose165_np[:3] = global_np
    pose165_np[3:66] = body63_np
    betas_np = betas.detach().cpu().numpy().reshape(10)
    shape16_np = np.zeros(16, dtype=np.float32)
    shape16_np[:10] = betas_np
    transl_np = transl.detach().cpu().numpy().reshape(3)
    joints_np = joints.detach().cpu().numpy().reshape(24, 3)
    vertices_np = vertices.detach().cpu().numpy().reshape(10475, 3)

    annotation = {
        # Harmony4DDataset-compatible fields.
        "global_orient": global_np,
        "body_pose": body69_np,
        "betas": betas_np,
        "transl": transl_np,
        "joints": joints_np,
        # MAMMA-compatible parameter names and dimensions.
        "pose_world": pose165_np,
        "shape": shape16_np,
        "trans_world": transl_np.copy(),
        "gender": "neutral",
        "fit_metrics": final_metrics,
    }
    return annotation, target_np, vertices_np


def atomic_save_npy(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npy", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.save(stream, value, allow_pickle=True)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def save_validation_plot(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    title: str,
    max_points: int = 4500,
) -> None:
    """Save three orthographic overlays and a correspondence-error histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = mask
    if selected.size > max_points:
        selected = selected[np.linspace(0, selected.size - 1, max_points).astype(int)]
    target_body = target[selected]
    pred_body = prediction[selected]
    error_mm = np.linalg.norm(prediction[mask] - target[mask], axis=1) * 1000.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, dims, labels in zip(
        axes[:3], ((0, 1), (0, 2), (1, 2)), (("x", "y"), ("x", "z"), ("y", "z"))
    ):
        ax.scatter(
            target_body[:, dims[0]],
            target_body[:, dims[1]],
            s=1.2,
            c="#e69f00",
            alpha=0.35,
            label="transferred SMPL target",
            rasterized=True,
        )
        ax.scatter(
            pred_body[:, dims[0]],
            pred_body[:, dims[1]],
            s=1.2,
            c="#0072b2",
            alpha=0.35,
            label="fitted SMPL-X",
            rasterized=True,
        )
        ax.set(xlabel=f"{labels[0]} (m)", ylabel=f"{labels[1]} (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", markerscale=5, fontsize=8)
    axes[3].hist(error_mm, bins=60, color="#009e73", alpha=0.85)
    axes[3].axvline(error_mm.mean(), color="black", linestyle="--", linewidth=1)
    axes[3].set(
        xlabel="body-surface correspondence error (mm)",
        ylabel="vertex count",
        title=f"mean={error_mm.mean():.2f} mm, p95={np.quantile(error_mm, .95):.2f} mm",
    )
    axes[3].grid(alpha=0.2)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def discover_sequences(dataset_root: Path) -> list[Path]:
    return sorted(
        smpl_dir.parent.parent
        for smpl_dir in dataset_root.glob("*/*/processed_data/smpl")
        if smpl_dir.is_dir()
    )


def resolve_sequences(
    dataset_root: Path, requested: Iterable[str] | None, process_all: bool
) -> list[Path]:
    if process_all:
        sequences = discover_sequences(dataset_root)
    else:
        sequences = []
        for relative in requested or ():
            sequence = dataset_root / relative
            if not (sequence / "processed_data" / "smpl").is_dir():
                raise FileNotFoundError(
                    f"No processed_data/smpl directory below {sequence}"
                )
            sequences.append(sequence)
    if not sequences:
        raise ValueError("No sequences selected; pass --sequence ... or --all")
    return sequences


def frame_paths(sequence: Path, frames: list[str] | None, limit: int | None):
    source_dir = sequence / "processed_data" / "smpl"
    if frames:
        paths = [source_dir / f"{Path(frame).stem}.npy" for frame in frames]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing frame(s): {missing}")
    else:
        paths = sorted(source_dir.glob("*.npy"))
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--sequence",
        action="append",
        help="Relative sequence, e.g. 01_hugging/001_hugging; repeatable",
    )
    selection.add_argument("--all", action="store_true", help="Process all sequences")
    parser.add_argument("--frames", nargs="*", help="Only these frame stems")
    parser.add_argument("--max-frames-per-sequence", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch is not None and torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        metavar="N",
        help="Save plots for the first N converted frames per sequence",
    )
    parser.add_argument("--visualization-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-store-vertices",
        action="store_true",
        help="Do not store the 10475 fitted vertices in each person record",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    visualization_root = (
        args.visualization_root.expanduser().resolve()
        if args.visualization_root is not None
        else output_root / "_validation"
    )
    sequences = resolve_sequences(dataset_root, args.sequence, args.all)
    work = [
        (sequence, path)
        for sequence in sequences
        for path in frame_paths(sequence, args.frames, args.max_frames_per_sequence)
    ]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    work = work[args.shard_index :: args.num_shards]
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "sequences": len(sequences),
                "frames": len(work),
                "people_per_frame": len(PEOPLE),
                "shard": f"{args.shard_index}/{args.num_shards}",
                "device": args.device,
                "steps": args.steps,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if not work:
        raise ValueError("Selected sequences contain no SMPL .npy files")

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for fitting. Activate the repository's 'mamma' "
            "conda environment, or use --dry-run to inspect the selected scope."
        )
    # This import pulls in the training stack and is intentionally delayed so
    # --help/--dry-run remain usable from a lightweight shell environment.
    from training.smpl_body import _get_smplx_model, set_smplx_model_root

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    transfer_matrix, body_mask = load_transfer_assets(args.transfer_root.resolve())
    set_smplx_model_root(args.model_root.expanduser().resolve())
    model = _get_smplx_model(device, "neutral")

    reports = []
    errors = []
    skipped = 0
    visualized_by_sequence: dict[str, int] = {}
    for index, (sequence, source_path) in enumerate(work, 1):
        relative_sequence = sequence.relative_to(dataset_root)
        output_path = output_root / relative_sequence / source_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(work)}] skip existing {output_path}")
            continue
        try:
            source_frame = np.load(source_path, allow_pickle=True).item()
            output_frame = {}
            frame_report = {
                "sequence": str(relative_sequence),
                "frame": source_path.stem,
                "source": str(source_path),
                "output": str(output_path),
                "people": {},
            }
            validation_payload = []
            for person_key in PEOPLE:
                if person_key not in source_frame:
                    raise KeyError(f"{person_key} missing from {source_path}")
                annotation, target_mesh, predicted_mesh = fit_person(
                    source_frame[person_key],
                    transfer_matrix,
                    body_mask,
                    model,
                    device,
                    args.steps,
                    args.learning_rate,
                )
                if not args.no_store_vertices:
                    annotation["vertices"] = predicted_mesh
                output_frame[person_key] = annotation
                frame_report["people"][person_key] = annotation["fit_metrics"]
                validation_payload.append((person_key, target_mesh, predicted_mesh))

            output_frame["_meta"] = {
                "model_type": "smplx",
                "gender": "neutral",
                "source_model_type": "smpl",
                "source": str(source_path),
                "coordinate_frame": "Harmony4D world (unchanged)",
                "pose_world_layout": "55 joints x axis-angle = 165",
                "unobserved_smplx_joints": "hands, jaw and eyes set to zero",
                "transfer_mask_vertices": int(body_mask.size),
            }
            atomic_save_npy(output_path, output_frame)
            reports.append(frame_report)

            seq_key = str(relative_sequence)
            visualized = visualized_by_sequence.get(seq_key, 0)
            if visualized < max(0, args.visualize):
                for person_key, target_mesh, predicted_mesh in validation_payload:
                    plot_path = (
                        visualization_root
                        / relative_sequence
                        / f"{source_path.stem}_{person_key}.png"
                    )
                    save_validation_plot(
                        plot_path,
                        target_mesh,
                        predicted_mesh,
                        body_mask,
                        f"{relative_sequence}/{source_path.stem} — {person_key}",
                    )
                visualized_by_sequence[seq_key] = visualized + 1
            means = [p["mean_mm"] for p in frame_report["people"].values()]
            print(
                f"[{index}/{len(work)}] {relative_sequence}/{source_path.stem} "
                f"mean={np.mean(means):.2f} mm -> {output_path}"
            )
        except Exception as exc:
            error = {
                "sequence": str(relative_sequence),
                "frame": source_path.stem,
                "source": str(source_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"[error] {json.dumps(error)}", file=sys.stderr)
            if args.fail_fast:
                raise

    report_name = (
        "conversion_report.json"
        if args.num_shards == 1
        else f"conversion_report.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    )
    report_path = output_root / report_name
    previous_frames = []
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            previous_frames = previous.get("frames", [])
        except (OSError, ValueError, TypeError):
            previous_frames = []
    # Keep one latest successful record per sequence/frame. This makes resume
    # runs useful without erasing metrics for outputs that were skipped.
    merged_frames = {
        (frame.get("sequence"), frame.get("frame")): frame
        for frame in previous_frames
        if isinstance(frame, dict)
    }
    for frame in reports:
        merged_frames[(frame["sequence"], frame["frame"])] = frame
    merged_frames_list = sorted(
        merged_frames.values(), key=lambda frame: (frame["sequence"], frame["frame"])
    )
    all_means = [
        person["mean_mm"]
        for frame in merged_frames_list
        for person in frame["people"].values()
    ]
    summary = {
        "converted_frames_total": len(merged_frames_list),
        "mean_surface_error_mm": float(np.mean(all_means)) if all_means else None,
        "max_person_mean_surface_error_mm": float(np.max(all_means)) if all_means else None,
        "last_run": {
            "requested_frames": len(work),
            "converted_frames": len(reports),
            "failed_frames": len(errors),
            "skipped_existing_frames": skipped,
        },
        "frames": merged_frames_list,
        "last_run_errors": errors,
    }
    atomic_write_json(report_path, summary)
    print(f"[report] {report_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
