#!/usr/bin/env python3
"""Generate MAMMA-style person masks for Harmony4D.

The source fisheye image is rectified first. Identity then comes from SMPL-X
ground truth projected with the resulting pinhole intrinsic, while an optional SAM2 image
predictor expands the naked-body mesh prompt to the visible clothing/hair
silhouette, following MAMMA-off's real-image segmentation approach.  Person
overlaps are made mutually exclusive with mesh depth and nearest-prompt
distance, so the instance map contains only background=0, aria01=1, aria02=2.

Run this with the PoseGAM conda environment when ``--refiner sam2`` is used.
The deterministic ``--refiner none`` mode needs only NumPy and OpenCV.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # Keep --help/--dry-run available in the base shell.
    cv2 = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)
DEFAULT_SMPLX_ROOT = SCRIPT_DIR / "output"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "masks"
DEFAULT_FACES = REPO_ROOT / "mamma_off" / "visualization" / "assets" / "smplx_faces.npy"
DEFAULT_SAM2_CHECKPOINT = Path(
    "/train-data-3-hdd/yian/PoseGAM/pretrained/sam2/sam2.1_hiera_tiny.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
PEOPLE = ("aria01", "aria02")
COLORS_BGR = ((54, 116, 240), (232, 160, 45))


@dataclass(frozen=True)
class Camera:
    name: str
    image_dir: Path
    width: int
    height: int
    model: str
    params: np.ndarray
    extrinsics: np.ndarray


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def orthonormalize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Match Harmony4DDataset's removal of COLMAP similarity scale."""
    extr = np.asarray(extrinsics, dtype=np.float64).copy()
    det = float(np.linalg.det(extr[:3, :3]))
    scale = np.sign(det) * abs(det) ** (1.0 / 3.0)
    if abs(scale) < 1e-9:
        raise ValueError("Degenerate camera rotation")
    extr[:3, :4] /= scale
    u, _, vt = np.linalg.svd(extr[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    extr[:3, :3] = rotation
    return extr[:3].astype(np.float32)


def read_colmap_cameras(path: Path) -> dict[int, tuple[str, int, int, np.ndarray]]:
    cameras = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            cameras[int(fields[0])] = (
                fields[1], int(fields[2]), int(fields[3]),
                np.asarray(fields[4:], dtype=np.float64),
            )
    return cameras


def read_colmap_exo_poses(path: Path):
    candidates = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 10 or fields[0].startswith("#"):
                continue
            parts = Path(fields[9]).parts
            if not parts or not parts[0].startswith("cam"):
                continue
            camera_name = parts[0]
            candidates.setdefault(camera_name, []).append(
                (
                    Path(fields[9]).stem,
                    np.asarray(fields[1:5], dtype=np.float64),
                    np.asarray(fields[5:8], dtype=np.float64),
                    int(fields[8]),
                )
            )
    return {
        camera: min(values, key=lambda value: int(value[0]))
        for camera, values in candidates.items()
    }


def world_from_colmap(workplace: Path) -> np.ndarray:
    scale_path = workplace / "scale.npy"
    if scale_path.is_file():
        transform = np.load(scale_path)
    else:
        transform_path = workplace / "aria_from_colmap_transforms.pkl"
        with transform_path.open("rb") as stream:
            transforms = pickle.load(stream)
        transform = transforms["aria01"]
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Invalid world-from-COLMAP transform: {transform.shape}")
    return transform


def load_cameras(sequence: Path) -> list[Camera]:
    workplace = sequence / "colmap" / "workplace"
    camera_table = read_colmap_cameras(workplace / "cameras.txt")
    exo_poses = read_colmap_exo_poses(workplace / "images.txt")
    colmap_from_world = np.linalg.inv(world_from_colmap(workplace))
    result = []
    for name in sorted(exo_poses):
        image_dir = sequence / "exo" / name / "images"
        if not image_dir.is_dir():
            continue
        _, qvec, tvec, camera_id = exo_poses[name]
        if camera_id not in camera_table:
            continue
        model, width, height, params = camera_table[camera_id]
        if model not in {"OPENCV_FISHEYE", "PINHOLE", "SIMPLE_PINHOLE"}:
            continue
        colmap_extr = np.eye(4, dtype=np.float64)
        colmap_extr[:3, :3] = quaternion_to_rotation(qvec)
        colmap_extr[:3, 3] = tvec
        world_extr = orthonormalize_extrinsics(colmap_extr @ colmap_from_world)
        result.append(Camera(name, image_dir, width, height, model, params, world_extr))
    return result


def camera_intrinsic_and_distortion(camera: Camera):
    p = camera.params
    if camera.model == "OPENCV_FISHEYE":
        if p.size != 8:
            raise ValueError(f"Invalid OPENCV_FISHEYE params: {p.shape}")
        fx, fy, cx, cy = p[:4]
        distortion = p[4:8].reshape(4, 1)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        distortion = None
    else:
        fx, cx, cy = p[:3]
        fy = fx
        distortion = None
    intrinsic = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return intrinsic, distortion


def rectify_image(
    image: np.ndarray,
    camera: Camera,
    max_side: int | None,
    balance: float,
):
    """Resize/undistort the RGB first and return its matching pinhole K."""
    intrinsic, distortion = camera_intrinsic_and_distortion(camera)
    height, width = image.shape[:2]
    if max_side is not None and max(height, width) > max_side:
        scale = float(max_side) / max(height, width)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        intrinsic[0, :] *= new_size[0] / width
        intrinsic[1, :] *= new_size[1] / height

    if distortion is None:
        return image, intrinsic.astype(np.float32)

    height, width = image.shape[:2]
    new_intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsic,
        distortion,
        (width, height),
        np.eye(3),
        balance=float(balance),
        new_size=(width, height),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3),
        new_intrinsic,
        (width, height),
        cv2.CV_16SC2,
    )
    rectified = cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified, np.asarray(new_intrinsic, dtype=np.float32)


def project_vertices(
    vertices_world: np.ndarray,
    camera: Camera,
    intrinsic: np.ndarray,
):
    """Project world vertices into the already-rectified pinhole image."""
    vertices = np.asarray(vertices_world, dtype=np.float64)
    camera_xyz = vertices @ camera.extrinsics[:, :3].T + camera.extrinsics[:, 3]
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    pixels = np.empty((camera_xyz.shape[0], 2), dtype=np.float64)
    safe_z = np.where(np.abs(camera_xyz[:, 2]) < 1e-9, 1e-9, camera_xyz[:, 2])
    pixels[:, 0] = intrinsic[0, 0] * camera_xyz[:, 0] / safe_z + intrinsic[0, 2]
    pixels[:, 1] = intrinsic[1, 1] * camera_xyz[:, 1] / safe_z + intrinsic[1, 2]
    return pixels.astype(np.float32), camera_xyz[:, 2].astype(np.float32)


def project_mask_to_original(
    rectified_mask: np.ndarray,
    camera: Camera,
    rectified_intrinsic: np.ndarray,
    source_shape: tuple[int, int],
) -> np.ndarray:
    """Sample a rectified mask back onto the original distorted image grid.

    ``cv2.remap`` needs, for every destination pixel in the distorted image,
    the corresponding source coordinate in the rectified mask.  Fisheye
    ``undistortPoints`` supplies exactly that distorted->rectified mapping.
    Rows are processed in chunks so a typical 3840x2160 Harmony4D inverse map
    remains exact at source resolution without materializing all 8M points.
    """
    source_height, source_width = (int(source_shape[0]), int(source_shape[1]))
    intrinsic, distortion = camera_intrinsic_and_distortion(camera)
    intrinsic[0, :] *= source_width / camera.width
    intrinsic[1, :] *= source_height / camera.height

    if distortion is None:
        return cv2.resize(
            rectified_mask,
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        )

    original_mask = np.zeros((source_height, source_width), dtype=rectified_mask.dtype)
    x_coordinates = np.arange(source_width, dtype=np.float32)
    for row_start in range(0, source_height, 256):
        row_end = min(source_height, row_start + 256)
        grid_x, grid_y = np.meshgrid(
            x_coordinates,
            np.arange(row_start, row_end, dtype=np.float32),
        )
        distorted_pixels = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
        rectified_pixels = cv2.fisheye.undistortPoints(
            distorted_pixels,
            intrinsic,
            distortion,
            R=np.eye(3),
            P=np.asarray(rectified_intrinsic, dtype=np.float64),
        ).reshape(row_end - row_start, source_width, 2)
        original_mask[row_start:row_end] = cv2.remap(
            rectified_mask,
            rectified_pixels[..., 0].astype(np.float32),
            rectified_pixels[..., 1].astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return original_mask


def _scaled_shape(height: int, width: int, long_side: int) -> tuple[int, int, float]:
    scale = min(1.0, float(long_side) / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale)), scale


def rasterize_people(
    projected: list[np.ndarray],
    depths: list[np.ndarray],
    faces: np.ndarray,
    height: int,
    width: int,
    long_side: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Painter-rasterize dense SMPL-X triangles with inter-person depth order."""
    rh, rw, scale = _scaled_shape(height, width, long_side)
    projected_scaled = [points * scale for points in projected]
    individual = []
    face_batches = []
    for person_idx, (points, z) in enumerate(zip(projected_scaled, depths), 1):
        tri_z = z[faces]
        valid = np.all(tri_z > 1e-3, axis=1)
        face_ids = np.flatnonzero(valid)
        person_mask = np.zeros((rh, rw), dtype=np.uint8)
        # Far-to-near painter order is sufficient for the dense, small triangles.
        order = face_ids[np.argsort(tri_z[valid].mean(axis=1))[::-1]]
        for face_id in order:
            polygon = np.rint(points[faces[face_id]]).astype(np.int32)
            if (
                polygon[:, 0].max() < 0 or polygon[:, 0].min() >= rw
                or polygon[:, 1].max() < 0 or polygon[:, 1].min() >= rh
            ):
                continue
            cv2.fillConvexPoly(person_mask, polygon, 1, lineType=cv2.LINE_8)
        individual.append(person_mask.astype(bool))
        face_batches.extend(
            (float(tri_z[face_id].mean()), person_idx, points[faces[face_id]])
            for face_id in face_ids
        )

    instance = np.zeros((rh, rw), dtype=np.uint8)
    for _, person_idx, polygon_float in sorted(face_batches, key=lambda item: item[0], reverse=True):
        polygon = np.rint(polygon_float).astype(np.int32)
        if (
            polygon[:, 0].max() < 0 or polygon[:, 0].min() >= rw
            or polygon[:, 1].max() < 0 or polygon[:, 1].min() >= rh
        ):
            continue
        cv2.fillConvexPoly(instance, polygon, person_idx, lineType=cv2.LINE_8)

    if (rh, rw) != (height, width):
        instance = cv2.resize(instance, (width, height), interpolation=cv2.INTER_NEAREST)
        individual = [
            cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            for mask in individual
        ]
    return instance, individual


def mask_box(mask: np.ndarray, padding: float = 0.12) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot prompt SAM with an empty projected mesh")
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x = max(4, round((x2 - x1 + 1) * padding))
    pad_y = max(4, round((y2 - y1 + 1) * padding))
    h, w = mask.shape
    return np.asarray(
        [max(0, x1 - pad_x), max(0, y1 - pad_y), min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)],
        dtype=np.float32,
    )


def prompt_points(positive: np.ndarray, negative: np.ndarray, count: int = 4):
    points, labels = [], []
    work = positive.astype(np.uint8)
    for _ in range(count):
        distance = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        _, maximum, _, location = cv2.minMaxLoc(distance)
        if maximum <= 0:
            break
        points.append(location)
        labels.append(1)
        cv2.circle(work, location, max(4, int(maximum * 0.75)), 0, -1)
    negative_only = negative & ~positive
    if np.any(negative_only):
        distance = cv2.distanceTransform(negative_only.astype(np.uint8), cv2.DIST_L2, 5)
        _, maximum, _, location = cv2.minMaxLoc(distance)
        if maximum > 0:
            points.append(location)
            labels.append(0)
    return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def keep_component_overlapping_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    scores = [np.logical_and(labels == label, seed).sum() for label in range(1, count)]
    best = int(np.argmax(scores)) + 1
    return labels == best


def refine_with_sam(predictor, image_bgr: np.ndarray, full_mesh: list[np.ndarray]):
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    refined, details = [], []
    for person_idx, seed in enumerate(full_mesh):
        other = np.logical_or.reduce(
            [mask for idx, mask in enumerate(full_mesh) if idx != person_idx]
        )
        box = mask_box(seed)
        points, labels = prompt_points(seed, other)
        masks, qualities, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=True,
        )
        x1, y1, x2, y2 = box.astype(int)
        allowed = np.zeros_like(seed, dtype=bool)
        allowed[y1 : y2 + 1, x1 : x2 + 1] = True
        candidates = []
        for mask, quality in zip(masks.astype(bool), qualities):
            mask &= allowed
            intersection = np.logical_and(mask, seed).sum()
            recall = float(intersection / max(1, seed.sum()))
            iou = mask_iou(mask, seed)
            score = 0.50 * recall + 0.30 * iou + 0.20 * float(quality)
            candidates.append((score, recall, iou, float(quality), mask))
        score, recall, iou, quality, best = max(candidates, key=lambda item: item[0])
        best = keep_component_overlapping_seed(best, seed)
        best |= seed
        refined.append(best)
        details.append(
            {
                "sam_score": score,
                "sam_predicted_iou": quality,
                "mesh_recall_before_union": recall,
                "sam_mesh_iou": iou,
                "prompt_box_xyxy": box.astype(int).tolist(),
                "positive_points": int((labels == 1).sum()),
                "negative_points": int((labels == 0).sum()),
            }
        )
    return refined, details


def resolve_instances(masks: list[np.ndarray], mesh_instance: np.ndarray) -> np.ndarray:
    stack = np.stack(masks, axis=0).astype(bool)
    instance = np.zeros(stack.shape[1:], dtype=np.uint8)
    count = stack.sum(axis=0)
    for person_idx in range(stack.shape[0]):
        instance[stack[person_idx] & (count == 1)] = person_idx + 1
    overlap = count > 1
    if not np.any(overlap):
        return instance

    known = overlap & (mesh_instance > 0)
    instance[known] = mesh_instance[known]
    unknown = overlap & ~known
    if np.any(unknown):
        distances = np.stack(
            [cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) for mask in masks]
        )
        instance[unknown] = np.argmin(distances[:, unknown], axis=0).astype(np.uint8) + 1
    return instance


def atomic_imwrite(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded.tobytes())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.stem}.", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def color_overlay(image: np.ndarray, instance: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    result = image.astype(np.float32).copy()
    for person_idx, color in enumerate(COLORS_BGR, 1):
        selected = instance == person_idx
        result[selected] = result[selected] * (1.0 - alpha) + np.asarray(color) * alpha
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 3)
    return np.clip(result, 0, 255).astype(np.uint8)


def validation_panel(image: np.ndarray, mesh_instance: np.ndarray, final_instance: np.ndarray) -> np.ndarray:
    max_side = 1500
    scale = min(1.0, max_side / max(image.shape[:2]))
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    original = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mesh = cv2.resize(mesh_instance, size, interpolation=cv2.INTER_NEAREST)
    final = cv2.resize(final_instance, size, interpolation=cv2.INTER_NEAREST)
    panels = [original, color_overlay(original, mesh), color_overlay(original, final)]
    titles = ["rectified RGB", "SMPL-X pinhole prompt", "final mutually-exclusive mask"]
    for panel, title in zip(panels, titles):
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 580), 54), (0, 0, 0), -1)
        cv2.putText(panel, title, (15, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def build_sam2_predictor(config: str, checkpoint: Path, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is unavailable. Activate the PoseGAM conda environment or use --refiner none."
        ) from exc
    model = build_sam2(config, str(checkpoint), device=device, apply_postprocessing=True)
    return SAM2ImagePredictor(model)


def discover_sequences(dataset_root: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in dataset_root.glob("*/*/processed_data/smpl")
        if path.is_dir()
    )


def resolve_sequences(root: Path, requested: Iterable[str] | None, all_sequences: bool):
    sequences = discover_sequences(root) if all_sequences else [root / item for item in requested or ()]
    for sequence in sequences:
        if not (sequence / "processed_data" / "smpl").is_dir():
            raise FileNotFoundError(f"Invalid Harmony4D sequence: {sequence}")
    if not sequences:
        raise ValueError("No sequences selected")
    return sequences


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence", action="append")
    selection.add_argument("--all", action="store_true")
    result.add_argument("--frames", nargs="*")
    result.add_argument("--camera", action="append", help="Camera name, e.g. cam01; repeatable")
    result.add_argument("--max-frames-per-sequence", type=int)
    result.add_argument("--max-cameras", type=int)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--faces", type=Path, default=DEFAULT_FACES)
    result.add_argument("--refiner", choices=("sam2", "none"), default="sam2")
    result.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    result.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--raster-long-side", type=int, default=1280)
    result.add_argument("--undistort-max-side", type=int, default=1280)
    result.add_argument("--undistort-balance", type=float, default=0.0)
    result.add_argument("--rectified-jpeg-quality", type=int, default=95)
    result.add_argument(
        "--export-original-mask",
        action="store_true",
        help="also inverse-project masks to the original distorted RGB resolution",
    )
    result.add_argument("--visualize", type=int, default=0, metavar="N")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--fail-fast", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    smplx_root = args.smplx_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sequences = resolve_sequences(dataset_root, args.sequence, args.all)
    faces = np.load(args.faces.expanduser().resolve(), allow_pickle=False).astype(np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= 10475:
        raise ValueError(f"Invalid SMPL-X faces: {faces.shape}")

    jobs = []
    for sequence in sequences:
        relative = sequence.relative_to(dataset_root)
        annotation_dir = smplx_root / relative
        if args.frames:
            frames = [Path(frame).stem for frame in args.frames]
        else:
            frames = [path.stem for path in sorted(annotation_dir.glob("*.npy"))]
        if args.max_frames_per_sequence is not None:
            frames = frames[: max(0, args.max_frames_per_sequence)]
        if not frames:
            continue
        cameras = load_cameras(sequence)
        if args.camera:
            names = set(args.camera)
            cameras = [camera for camera in cameras if camera.name in names]
            missing = names - {camera.name for camera in cameras}
            if missing:
                raise ValueError(f"Unknown camera(s) in {relative}: {sorted(missing)}")
        if args.max_cameras is not None:
            cameras = cameras[: max(0, args.max_cameras)]
        for frame in frames:
            annotation = annotation_dir / f"{frame}.npy"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            for camera in cameras:
                image_path = camera.image_dir / f"{frame}.jpg"
                if image_path.is_file():
                    jobs.append((relative, frame, annotation, camera, image_path))

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    jobs = jobs[args.shard_index :: args.num_shards]

    print(json.dumps({
        "dataset_root": str(dataset_root), "smplx_root": str(smplx_root),
        "output_root": str(output_root), "sequences": len(sequences),
        "camera_frames": len(jobs), "refiner": args.refiner,
        "shard": f"{args.shard_index}/{args.num_shards}",
    }, indent=2))
    if args.dry_run:
        return 0
    if not jobs:
        raise ValueError("No camera/frame jobs selected")
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for mask generation. Activate the PoseGAM conda environment."
        )
    predictor = None
    if args.refiner == "sam2":
        predictor = build_sam2_predictor(
            args.sam2_config, args.sam2_checkpoint.expanduser().resolve(), args.device
        )

    reports, errors, visualized, skipped = [], [], 0, 0
    for job_idx, (relative, frame, annotation_path, camera, image_path) in enumerate(jobs, 1):
        camera_dir = output_root / relative / camera.name
        instance_path = camera_dir / f"{frame}.mask.png"
        rectified_dir = camera_dir / "rectified"
        rectified_path = rectified_dir / f"{frame}.jpg"
        camera_path = rectified_dir / f"{frame}.camera.npz"
        required_outputs = [instance_path, rectified_path, camera_path]
        if args.export_original_mask:
            required_outputs.append(camera_dir / "original_fisheye" / f"{frame}.mask.png")
        if all(path.is_file() for path in required_outputs) and not args.overwrite:
            skipped += 1
            print(f"[{job_idx}/{len(jobs)}] skip {instance_path}")
            continue
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            source_shape = np.asarray(image.shape[:2], dtype=np.int32)
            image, intrinsic = rectify_image(
                image, camera, args.undistort_max_side, args.undistort_balance
            )
            annotation = np.load(annotation_path, allow_pickle=True).item()
            vertices = [np.asarray(annotation[key]["vertices"], np.float32) for key in PEOPLE]
            projected_and_depth = [project_vertices(item, camera, intrinsic) for item in vertices]
            projected = [item[0] for item in projected_and_depth]
            depths = [item[1] for item in projected_and_depth]
            mesh_instance, full_mesh = rasterize_people(
                projected, depths, faces, image.shape[0], image.shape[1], args.raster_long_side
            )
            if any(not np.any(mask) for mask in full_mesh):
                raise RuntimeError("At least one projected person has an empty mesh mask")

            if predictor is None:
                refined = full_mesh
                details = [{"sam_used": False} for _ in PEOPLE]
            else:
                refined, details = refine_with_sam(predictor, image, full_mesh)
            final_instance = resolve_instances(refined, mesh_instance)
            overlap_before = int(np.logical_and(refined[0], refined[1]).sum())

            original_instance = None
            original_instance_path = None
            if args.export_original_mask:
                original_instance = project_mask_to_original(
                    final_instance,
                    camera,
                    intrinsic,
                    tuple(source_shape.tolist()),
                )
                invalid_original = np.setdiff1d(
                    np.unique(original_instance), np.asarray([0, 1, 2], dtype=np.uint8)
                )
                if invalid_original.size:
                    raise RuntimeError(
                        f"Inverse projection changed mask labels: {invalid_original.tolist()}"
                    )
                original_dir = camera_dir / "original_fisheye"
                original_instance_path = original_dir / f"{frame}.mask.png"
                atomic_imwrite(original_instance_path, original_instance)
                for person_idx in range(1, len(PEOPLE) + 1):
                    binary_original = (
                        (original_instance == person_idx) * 255
                    ).astype(np.uint8)
                    atomic_imwrite(
                        original_dir / "masks" / f"mask_{frame}_{person_idx:02d}.png",
                        binary_original,
                    )

            atomic_imwrite(
                rectified_path,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.rectified_jpeg_quality)],
            )
            atomic_save_npz(
                camera_path,
                intrinsic=np.asarray(intrinsic, dtype=np.float32),
                extrinsics=np.asarray(camera.extrinsics, dtype=np.float32),
                source_shape=source_shape,
                rectified_shape=np.asarray(image.shape[:2], dtype=np.int32),
                source_camera_model=np.asarray(camera.model),
                undistort_balance=np.asarray(args.undistort_balance, dtype=np.float32),
            )
            atomic_imwrite(instance_path, final_instance)
            for person_idx, person_key in enumerate(PEOPLE, 1):
                binary = ((final_instance == person_idx) * 255).astype(np.uint8)
                atomic_imwrite(camera_dir / "masks" / f"mask_{frame}_{person_idx:02d}.png", binary)

            person_report = {}
            for person_idx, person_key in enumerate(PEOPLE, 1):
                final_mask = final_instance == person_idx
                person_report[person_key] = {
                    **details[person_idx - 1],
                    "mesh_pixels": int(full_mesh[person_idx - 1].sum()),
                    "final_pixels": int(final_mask.sum()),
                    "final_mesh_iou": mask_iou(final_mask, full_mesh[person_idx - 1]),
                }
            report = {
                "sequence": str(relative), "camera": camera.name, "frame": frame,
                "source_image": str(image_path), "rectified_image": str(rectified_path),
                "camera_metadata": str(camera_path), "instance_mask": str(instance_path),
                "original_fisheye_instance_mask": (
                    str(original_instance_path) if original_instance_path else None
                ),
                "overlap_pixels_resolved": overlap_before, "people": person_report,
                "instance_values": np.unique(final_instance).astype(int).tolist(),
            }
            reports.append(report)
            if visualized < max(0, args.visualize):
                panel = validation_panel(image, mesh_instance, final_instance)
                atomic_imwrite(
                    output_root / "_validation" / relative / camera.name / f"{frame}_overlay.jpg",
                    panel, [cv2.IMWRITE_JPEG_QUALITY, 94],
                )
                if original_instance is not None:
                    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if source_image is None:
                        raise RuntimeError(f"Could not reread source image {image_path}")
                    original_overlay = color_overlay(source_image, original_instance)
                    max_side = 1500
                    scale = min(1.0, max_side / max(source_image.shape[:2]))
                    display_size = (
                        round(source_image.shape[1] * scale),
                        round(source_image.shape[0] * scale),
                    )
                    original_panel = np.concatenate(
                        [
                            cv2.resize(source_image, display_size, interpolation=cv2.INTER_AREA),
                            cv2.resize(original_overlay, display_size, interpolation=cv2.INTER_AREA),
                        ],
                        axis=1,
                    )
                    atomic_imwrite(
                        output_root / "_validation_original_fisheye" / relative
                        / camera.name / f"{frame}_overlay.jpg",
                        original_panel,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                visualized += 1
            print(
                f"[{job_idx}/{len(jobs)}] {relative}/{camera.name}/{frame} "
                f"areas={[int((final_instance == i).sum()) for i in (1, 2)]} -> {instance_path}"
            )
        except Exception as exc:
            error = {
                "sequence": str(relative), "camera": camera.name, "frame": frame,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"[error] {json.dumps(error)}", file=sys.stderr)
            if args.fail_fast:
                raise

    report_name = (
        "mask_report.json"
        if args.num_shards == 1
        else f"mask_report.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    )
    report_path = output_root / report_name
    previous_frames = []
    if report_path.is_file():
        try:
            previous_frames = json.loads(report_path.read_text(encoding="utf-8")).get("frames", [])
        except (OSError, ValueError, TypeError):
            previous_frames = []
    merged = {
        (item.get("sequence"), item.get("camera"), item.get("frame")): item
        for item in previous_frames if isinstance(item, dict)
    }
    for item in reports:
        merged[(item["sequence"], item["camera"], item["frame"])] = item
    merged_frames = sorted(
        merged.values(), key=lambda item: (item["sequence"], item["camera"], item["frame"])
    )
    summary = {
        "converted_camera_frames_total": len(merged_frames),
        "refiner": args.refiner,
        "last_run": {
            "requested_camera_frames": len(jobs),
            "converted_camera_frames": len(reports),
            "failed_camera_frames": len(errors),
            "skipped_existing_camera_frames": skipped,
        },
        "frames": merged_frames,
        "last_run_errors": errors,
    }
    atomic_write_json(report_path, summary)
    print(f"[report] {report_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
