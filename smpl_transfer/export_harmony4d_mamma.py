#!/usr/bin/env python3
"""Export converted Harmony4D as the minimal raw-MAMMA format used for training.

Only fields consumed by ``mamma_mask_dpt.yaml`` are written: rectified RGB,
an instance-label mask, SMPL-X pose/shape/translation, gender/person ID, and
pinhole camera intrinsics/extrinsics. Dense vertices, 2D landmarks, visibility,
SDF/contact, normals, and other large MAMMA fields are intentionally omitted.

Output layout::

    <output>/png/h4d_<activity>_<sequence>/<camera>/<frame>.jpg
                                                    <frame>.mask.png
                                                    <frame>.data.pyd

This directory is directly discoverable by ``SysSMPLMultiDataset``. The input
RGB, mask, and pinhole intrinsic must come from ``generate_harmony4d_masks.py``,
which rectifies the source image before mask generation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # Let --help work in the base shell.
    cv2 = None  # type: ignore[assignment]

from generate_harmony4d_masks import Camera, load_cameras


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)
DEFAULT_SMPLX_ROOT = SCRIPT_DIR / "output"
DEFAULT_MASK_ROOT = SCRIPT_DIR / "masks"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "mamma_harmony4d"
PEOPLE = ("aria01", "aria02")


def minimal_person_record(person: dict, intrinsic: np.ndarray, extrinsic: np.ndarray, person_idx: int):
    pose = np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)
    shape = np.asarray(person["shape"], dtype=np.float32).reshape(-1)
    translation = np.asarray(person["trans_world"], dtype=np.float32).reshape(3)
    if pose.size != 165:
        raise ValueError(f"Expected pose_world(165), got {pose.shape}")
    if shape.size != 16:
        raise ValueError(f"Expected shape(16), got {shape.shape}")
    if not (np.isfinite(pose).all() and np.isfinite(shape).all() and np.isfinite(translation).all()):
        raise ValueError("Non-finite SMPL-X parameters")
    return {
        "pose_world": pose,
        "shape": shape,
        "trans_world": translation,
        "gender": str(person.get("gender", "neutral")),
        "person_idx": int(person_idx),
        "cam_int": np.asarray(intrinsic, dtype=np.float32),
        "cam_ext": np.asarray(extrinsic, dtype=np.float32),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    os.replace(temporary, path)


def atomic_imwrite(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    ok, encoded = cv2.imencode(path.suffix, image, params or [])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    atomic_bytes(path, encoded.tobytes())


def atomic_pickle(path: Path, value) -> None:
    atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode("utf-8"))


def write_validation_panel(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    """Write rectified RGB next to its MAMMA instance-mask overlay."""
    colours = np.asarray(
        [[0, 0, 0], [48, 92, 255], [54, 210, 92]], dtype=np.uint8
    )
    colour_mask = colours[mask]
    foreground = mask > 0
    overlay = image.copy()
    overlay[foreground] = (
        0.55 * image[foreground] + 0.45 * colour_mask[foreground]
    ).astype(np.uint8)
    boundary = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    overlay[boundary] = (255, 255, 255)
    panel = np.concatenate([image, overlay], axis=1)
    atomic_imwrite(path, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])


def discover_sequences(root: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in root.glob("*/*/processed_data/smpl")
        if path.is_dir()
    )


def resolve_sequences(root: Path, requested: Iterable[str] | None, process_all: bool):
    sequences = discover_sequences(root) if process_all else [root / item for item in requested or ()]
    for sequence in sequences:
        if not (sequence / "processed_data" / "smpl").is_dir():
            raise FileNotFoundError(f"Invalid Harmony4D sequence: {sequence}")
    if not sequences:
        raise ValueError("No sequences selected")
    return sequences


def exported_sequence_name(relative: Path) -> str:
    return "h4d_" + "_".join(relative.parts)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence", action="append")
    selection.add_argument("--all", action="store_true")
    result.add_argument("--frames", nargs="*")
    result.add_argument("--camera", action="append")
    result.add_argument("--max-frames-per-sequence", type=int)
    result.add_argument("--max-cameras", type=int)
    result.add_argument("--min-views", type=int, default=8)
    result.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    result.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--jpeg-quality", type=int, default=95)
    result.add_argument(
        "--visualize",
        type=int,
        default=0,
        metavar="N",
        help="write rectified RGB/mask panels for the first N exported camera frames",
    )
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--fail-fast", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if cv2 is None and not args.dry_run:
        raise RuntimeError("OpenCV is required. Activate the mamma or posegam conda environment.")
    if args.min_views < 1:
        raise ValueError("--min-views must be positive")
    if args.visualize < 0:
        raise ValueError("--visualize must be non-negative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards>=1 and 0<=shard-index<num-shards")

    dataset_root = args.dataset_root.expanduser().resolve()
    smplx_root = args.smplx_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sequences = resolve_sequences(dataset_root, args.sequence, args.all)

    jobs = []
    sequence_summaries = {}
    for sequence in sequences:
        relative = sequence.relative_to(dataset_root)
        smplx_dir = smplx_root / relative
        if args.frames:
            frames = [Path(frame).stem for frame in args.frames]
        else:
            frames = [path.stem for path in sorted(smplx_dir.glob("*.npy"))]
        if args.max_frames_per_sequence is not None:
            frames = frames[: max(0, args.max_frames_per_sequence)]
        if not frames:
            continue

        cameras = load_cameras(sequence)
        if args.camera:
            requested = set(args.camera)
            cameras = [camera for camera in cameras if camera.name in requested]
            missing = requested - {camera.name for camera in cameras}
            if missing:
                raise ValueError(f"Unknown cameras in {relative}: {sorted(missing)}")
        if args.max_cameras is not None:
            cameras = cameras[: max(0, args.max_cameras)]

        valid_frames = []
        for frame in frames:
            annotation_path = smplx_dir / f"{frame}.npy"
            available = [
                camera
                for camera in cameras
                if (camera.image_dir / f"{frame}.jpg").is_file()
                and (mask_root / relative / camera.name / f"{frame}.mask.png").is_file()
                and (mask_root / relative / camera.name / "rectified" / f"{frame}.jpg").is_file()
                and (mask_root / relative / camera.name / "rectified" / f"{frame}.camera.npz").is_file()
            ]
            if len(available) < args.min_views:
                continue
            valid_frames.append(frame)
            for camera in available:
                jobs.append(
                    (
                        relative,
                        exported_sequence_name(relative),
                        frame,
                        annotation_path,
                        camera,
                        mask_root / relative / camera.name / "rectified" / f"{frame}.jpg",
                        mask_root / relative / camera.name / f"{frame}.mask.png",
                        mask_root / relative / camera.name / "rectified" / f"{frame}.camera.npz",
                    )
                )
        if valid_frames:
            sequence_summaries[str(relative)] = {
                "exported_name": exported_sequence_name(relative),
                "frames": len(valid_frames),
            }

    jobs = jobs[args.shard_index :: args.num_shards]
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "smplx_root": str(smplx_root),
                "mask_root": str(mask_root),
                "output_root": str(output_root),
                "sequences_with_complete_frames": len(sequence_summaries),
                "camera_frames": len(jobs),
                "min_views": args.min_views,
                "shard": f"{args.shard_index}/{args.num_shards}",
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if not jobs:
        raise ValueError("No complete SMPL-X + mask camera/frame jobs selected")

    annotation_cache = {}
    reports, errors, skipped = [], [], 0
    for index, (relative, export_name, frame, annotation_path, camera, image_path, mask_path, camera_path) in enumerate(jobs, 1):
        view_dir = output_root / "png" / export_name / camera.name
        output_image = view_dir / f"{frame}.jpg"
        output_mask = view_dir / f"{frame}.mask.png"
        output_data = view_dir / f"{frame}.data.pyd"
        if all(path.is_file() for path in (output_image, output_mask, output_data)) and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(jobs)}] skip {view_dir}/{frame}")
            continue
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise RuntimeError("Could not read source image or mask")
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} vs {mask.shape}")
            invalid_values = np.setdiff1d(np.unique(mask), np.asarray([0, 1, 2], dtype=np.uint8))
            if invalid_values.size:
                raise ValueError(f"Mask contains invalid labels: {invalid_values.tolist()}")

            with np.load(camera_path, allow_pickle=False) as camera_data:
                intrinsic = np.asarray(camera_data["intrinsic"], dtype=np.float32)
                stored_extrinsics = np.asarray(camera_data["extrinsics"], dtype=np.float32)
                rectified_shape = tuple(
                    np.asarray(camera_data["rectified_shape"], dtype=np.int32).tolist()
                )
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError(f"Invalid rectified intrinsic in {camera_path}")
            if stored_extrinsics.shape != (3, 4) or not np.allclose(
                stored_extrinsics, camera.extrinsics, atol=1e-5
            ):
                raise ValueError(f"Camera extrinsics mismatch in {camera_path}")
            if rectified_shape != image.shape[:2]:
                raise ValueError(
                    f"Rectified metadata/image mismatch: {rectified_shape} vs {image.shape[:2]}"
                )

            if annotation_path not in annotation_cache:
                annotation_cache[annotation_path] = np.load(annotation_path, allow_pickle=True).item()
            annotation = annotation_cache[annotation_path]
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3] = camera.extrinsics
            pyd = {
                person_idx: minimal_person_record(
                    annotation[person_key], intrinsic, extrinsic, person_idx
                )
                for person_idx, person_key in enumerate(PEOPLE)
            }
            atomic_imwrite(
                output_image,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
            )
            atomic_imwrite(output_mask, mask)
            atomic_pickle(output_data, pyd)
            validation_path = None
            if len(reports) < args.visualize:
                validation_path = (
                    output_root / "_validation" / export_name / camera.name
                    / f"{frame}_overlay.jpg"
                )
                write_validation_panel(validation_path, image, mask)
            reports.append(
                {
                    "sequence": str(relative),
                    "exported_sequence": export_name,
                    "camera": camera.name,
                    "frame": frame,
                    "image_shape": list(image.shape),
                    "mask_values": np.unique(mask).astype(int).tolist(),
                    "person_pixels": {
                        person_key: int((mask == person_idx + 1).sum())
                        for person_idx, person_key in enumerate(PEOPLE)
                    },
                    "validation_panel": str(validation_path) if validation_path else None,
                }
            )
            print(f"[{index}/{len(jobs)}] {export_name}/{camera.name}/{frame}")
        except Exception as exc:
            error = {
                "sequence": str(relative), "camera": camera.name, "frame": frame,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"[error] {json.dumps(error)}")
            if args.fail_fast:
                raise

    report_name = (
        "export_report.json"
        if args.num_shards == 1
        else f"export_report.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    )
    atomic_json(
        output_root / report_name,
        {
            "format": "minimal_raw_mamma_v1",
            "required_by": "training/config/mamma_mask_dpt.yaml",
            "omitted_fields": [
                "vertices2d", "vertices3d", "joints2d", "joints3d",
                "vertex_visibility", "sdf_vertices", "floor_contact_mask", "camera_normals",
            ],
            "last_run": {
                "requested_camera_frames": len(jobs),
                "exported_camera_frames": len(reports),
                "skipped_camera_frames": skipped,
                "failed_camera_frames": len(errors),
            },
            "sequences": sequence_summaries,
            "camera_frames": reports,
            "errors": errors,
        },
    )
    exported_names = sorted(item["exported_name"] for item in sequence_summaries.values())
    atomic_bytes(output_root / "train_data.txt", ("\n".join(exported_names) + "\n").encode("utf-8"))
    print(f"[output] {output_root}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
