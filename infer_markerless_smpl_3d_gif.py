#!/usr/bin/env python3
"""Render predicted multi-person SMPL as a continuous, fixed-view 3D GIF.

This is a pure prediction visualizer: it never reads GT SMPL parameters, GT
meshes, or GT cameras.  The model consumes four image views.  Its top-K SMPL
predictions are decoded into 3D meshes, temporally associated by pelvis
position, then rendered by a fixed virtual camera looking diagonally downward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment

import infer_markerless_smpl_gif as common
from training.smpl_body import _decode_smpl_batch, _get_smpl_model
from training.train_utils.normalization import (
    normalize_camera_extrinsics_points_and_3djoints_batch,
)
from vggt.utils.load_fn import load_and_preprocess_images


# RGB base colors: three distinguishable shades in a blue/cyan/violet family.
PERSON_COLORS_RGB = np.asarray(
    [
        (40, 132, 255),
        (0, 205, 220),
        (135, 105, 255),
        (45, 180, 125),
        (245, 150, 55),
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a pure-3D, oblique top-down SMPL mesh GIF."
    )
    parser.add_argument("--config", default="mamma_mask_dpt")
    parser.add_argument(
        "--checkpoint",
        default=str(common.REPO_DIR / "model/no_avg/checkpoint_47.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(common.REPO_DIR / "MAMMA_markerless_multiple_people"),
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--num-input-views", type=int, default=4)
    parser.add_argument(
        "--input-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit zero-based camera indices. When provided, these replace "
            "the seed-based random selection and must not contain duplicates."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        default=35.0,
        help="Horizontal direction of the fixed virtual camera.",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=28.0,
        help="Camera elevation above the scene; positive values look downward.",
    )
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--output-dir",
        default=str(
            common.REPO_DIR
            / "outputs/no_avg_checkpoint47_markerless_first100_3d_gif"
        ),
    )
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), eps)


def checkpoint_output_tag(checkpoint_path: Path) -> str:
    """Build an unambiguous output tag for checkpoints with repeated filenames."""
    parent_name = checkpoint_path.parent.name
    parts = (
        [parent_name, checkpoint_path.stem]
        if parent_name not in {"model", ""}
        else [checkpoint_path.stem]
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parts)).strip("_")


def orthonormalize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Remove a uniform scale baked into a world-to-camera rotation block."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64).copy()
    rotation = extrinsics[..., :3, :3]
    determinant = np.linalg.det(rotation)
    scale = np.sign(determinant) * np.abs(determinant) ** (1.0 / 3.0)
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    extrinsics[..., :3, :3] = rotation / scale[..., None, None]
    extrinsics[..., :3, 3] /= scale[..., None]
    return extrinsics


def load_frame_avg_scale(
    dataset_root: Path,
    split: str,
    frame_dir: Path,
    image_paths: list[str],
    scale_by_extrinsics: bool,
) -> float:
    """Compute the training-time camera-baseline scale for one organized frame.

    Only ``cam_param_min/*/extrinsics.worldToCamera12`` is read. GT people,
    poses, shapes, translations, and meshes are never loaded.
    """
    if not scale_by_extrinsics:
        return 1.0
    archive_path = dataset_root / split / "out_data" / f"{frame_dir.name}.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Camera calibration archive not found: {archive_path}"
        )
    extrinsics = []
    with np.load(archive_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        for image_path in image_paths:
            camera_name = Path(image_path).stem
            key = (
                f"cam_param_min/{camera_name}/"
                "extrinsics.worldToCamera12"
            )
            if key not in keys:
                raise KeyError(f"Missing camera extrinsic {key} in {archive_path}")
            extrinsic = np.asarray(archive[key], dtype=np.float64)
            extrinsics.append(
                extrinsic[:3, :4]
                if extrinsic.shape == (4, 4)
                else extrinsic.reshape(3, 4)
            )
    extrinsics_np = orthonormalize_extrinsics(np.stack(extrinsics, axis=0))
    extrinsics_tensor = torch.as_tensor(
        extrinsics_np,
        dtype=torch.float32,
        device="cpu",
    ).unsqueeze(0)
    _, _, _, _, _, avg_scale = (
        normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=extrinsics_tensor,
            scale_by_extrinsics=True,
        )
    )
    return float(avg_scale.reshape(-1)[0].item())


def decode_people(
    predictions: dict,
    slots: np.ndarray,
    device: torch.device,
    avg_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    poses = predictions["smpl_pose"][0, slots].to(device=device, dtype=torch.float32)
    betas = predictions["smpl_beta"][0, slots].to(
        device=device,
        dtype=torch.float32,
    )
    translations = predictions.get("mesh_translate")
    if translations is None:
        translations = predictions.get("smpl_trans")
    if translations is None:
        raise KeyError("Model produced neither mesh_translate nor smpl_trans")
    translations = translations[0, slots].to(
        device=device,
        dtype=torch.float32,
    ).reshape(len(slots), 3)
    zero_trans = torch.zeros((len(slots), 3), device=device)

    joints, vertices = _decode_smpl_batch(
        pose_aa=poses,
        betas=betas,
        trans=zero_trans,
        genders=["neutral"] * len(slots),
        use_mamma=False,
    )
    # mesh_translate is expressed in the camera0 gauge used during training.
    # For avg-scale configs both body geometry and its decoded root must be
    # divided by the per-frame mean camera baseline before re-anchoring.
    scale = max(float(avg_scale), 1e-6)
    vertices_gauge = vertices / scale
    joints_gauge = joints / scale
    offsets = translations - joints_gauge[:, 0, :]
    placed_vertices = vertices_gauge + offsets[:, None, :]
    placed_joints = joints_gauge + offsets[:, None, :]
    return (
        placed_vertices.detach().cpu().numpy().astype(np.float32),
        placed_joints.detach().cpu().numpy().astype(np.float32),
    )


def associate_tracks(
    all_joints: list[np.ndarray],
) -> list[np.ndarray]:
    """Assign stable color IDs with frame-to-frame pelvis matching."""
    assignments: list[np.ndarray] = []
    previous_centers = None
    previous_track_ids = None
    next_track_id = 0
    for joints in all_joints:
        centers = np.asarray(joints[:, 0, :], dtype=np.float64)
        count = centers.shape[0]
        if previous_centers is None:
            # Deterministic initial ordering; later frames follow 3D proximity.
            order = np.lexsort((centers[:, 2], centers[:, 0]))
            track_ids = np.empty(count, dtype=np.int64)
            track_ids[order] = np.arange(count, dtype=np.int64)
            next_track_id = count
        else:
            track_ids = np.full(count, -1, dtype=np.int64)
            cost = np.linalg.norm(
                previous_centers[:, None, :] - centers[None, :, :],
                axis=-1,
            )
            previous_indices, current_indices = linear_sum_assignment(cost)
            for previous_index, current_index in zip(
                previous_indices,
                current_indices,
            ):
                track_ids[current_index] = previous_track_ids[previous_index]
            for current_index in np.flatnonzero(track_ids < 0):
                track_ids[current_index] = next_track_id
                next_track_id += 1
        assignments.append(track_ids)
        previous_centers = centers
        previous_track_ids = track_ids
    return assignments


def estimate_up_direction(all_joints: list[np.ndarray]) -> np.ndarray:
    """Estimate scene-up from pelvis-to-head vectors, independent of camera gauge."""
    directions = []
    reference = None
    for joints in all_joints:
        # SMPL joint 15 is the head and joint 0 is the pelvis.
        frame_directions = joints[:, 15, :] - joints[:, 0, :]
        for direction in frame_directions:
            direction = normalize(direction)
            if reference is None:
                reference = direction
            if float(np.dot(direction, reference)) < 0:
                direction = -direction
            directions.append(direction)
    return normalize(np.median(np.stack(directions, axis=0), axis=0))


def make_horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    first = max(
        candidates,
        key=lambda candidate: np.linalg.norm(
            candidate - np.dot(candidate, up) * up
        ),
    )
    horizontal_x = normalize(first - np.dot(first, up) * up)
    horizontal_z = normalize(np.cross(up, horizontal_x))
    return horizontal_x, horizontal_z


def compute_virtual_camera(
    all_vertices: list[np.ndarray],
    all_joints: list[np.ndarray],
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float,
) -> dict:
    up = estimate_up_direction(all_joints)
    horizontal_x, horizontal_z = make_horizontal_basis(up)
    roots = np.concatenate([joints[:, 0, :] for joints in all_joints], axis=0)
    center = np.median(roots, axis=0).astype(np.float64)

    # Robust global bounds keep the camera fixed while ignoring rare wild verts.
    sampled = np.concatenate(
        [vertices[:, ::8, :].reshape(-1, 3) for vertices in all_vertices],
        axis=0,
    ).astype(np.float64)
    relative = sampled - center
    coordinates = np.stack(
        [
            relative @ horizontal_x,
            relative @ horizontal_z,
            relative @ up,
        ],
        axis=1,
    )
    low = np.percentile(coordinates, 1.0, axis=0)
    high = np.percentile(coordinates, 99.0, axis=0)
    scene_center_offset = (
        0.5 * (low[0] + high[0]) * horizontal_x
        + 0.5 * (low[1] + high[1]) * horizontal_z
        + 0.5 * (low[2] + high[2]) * up
    )
    target = center + scene_center_offset
    ranges = np.maximum(high - low, 0.5)
    radius = 0.5 * float(np.linalg.norm(ranges))
    fov_rad = math.radians(fov_deg)
    distance = 1.18 * radius / max(math.sin(fov_rad / 2.0), 1e-3)

    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    ground_direction = normalize(
        math.cos(azimuth) * horizontal_x
        + math.sin(azimuth) * horizontal_z
    )
    eye_direction = normalize(
        math.cos(elevation) * ground_direction
        + math.sin(elevation) * up
    )
    eye = target + distance * eye_direction

    # Floor uses the lower robust bound along the inferred vertical axis.
    floor_point = center + low[2] * up
    grid_extent = 0.72 * max(ranges[0], ranges[1], 1.0)
    return {
        "eye": eye,
        "target": target,
        "up": up,
        "horizontal_x": horizontal_x,
        "horizontal_z": horizontal_z,
        "floor_point": floor_point,
        "grid_extent": grid_extent,
        "fov_deg": fov_deg,
    }


def camera_basis(camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(camera["target"] - camera["eye"])
    right = normalize(np.cross(forward, camera["up"]))
    screen_up = normalize(np.cross(right, forward))
    return right, screen_up, forward


def project_virtual(
    points: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    right, screen_up, forward = camera_basis(camera)
    relative = np.asarray(points, dtype=np.float64) - camera["eye"]
    cam_x = relative @ right
    cam_y = relative @ screen_up
    depth = relative @ forward
    focal = 0.5 * min(width, height) / math.tan(
        math.radians(camera["fov_deg"]) / 2.0
    )
    safe_depth = np.maximum(depth, 1e-6)
    pixel_x = focal * cam_x / safe_depth + width / 2.0
    pixel_y = -focal * cam_y / safe_depth + height / 2.0
    return np.stack([pixel_x, pixel_y], axis=-1), depth


def draw_floor_grid(
    canvas: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> None:
    extent = float(camera["grid_extent"])
    grid_values = np.linspace(-extent, extent, 11)
    line_color = (67, 69, 75)
    axis_color = (88, 91, 100)
    for axis, value in ((0, value) for value in grid_values):
        del axis
        points = np.stack(
            [
                camera["floor_point"]
                + value * camera["horizontal_x"]
                - extent * camera["horizontal_z"],
                camera["floor_point"]
                + value * camera["horizontal_x"]
                + extent * camera["horizontal_z"],
            ]
        )
        pixels, depth = project_virtual(points, camera, width, height)
        if np.all(depth > 0):
            color = axis_color if abs(value) < 1e-8 else line_color
            cv2.line(
                canvas,
                tuple(np.rint(pixels[0]).astype(int)),
                tuple(np.rint(pixels[1]).astype(int)),
                color,
                1,
                cv2.LINE_AA,
            )
    for value in grid_values:
        points = np.stack(
            [
                camera["floor_point"]
                - extent * camera["horizontal_x"]
                + value * camera["horizontal_z"],
                camera["floor_point"]
                + extent * camera["horizontal_x"]
                + value * camera["horizontal_z"],
            ]
        )
        pixels, depth = project_virtual(points, camera, width, height)
        if np.all(depth > 0):
            color = axis_color if abs(value) < 1e-8 else line_color
            cv2.line(
                canvas,
                tuple(np.rint(pixels[0]).astype(int)),
                tuple(np.rint(pixels[1]).astype(int)),
                color,
                1,
                cv2.LINE_AA,
            )


def render_mesh_frame(
    people_vertices: np.ndarray,
    faces: np.ndarray,
    track_ids: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (43, 44, 48), dtype=np.uint8)
    draw_floor_grid(canvas, camera, width, height)

    person_count, vertices_per_person = people_vertices.shape[:2]
    vertices = people_vertices.reshape(-1, 3)
    combined_faces = np.concatenate(
        [faces + index * vertices_per_person for index in range(person_count)],
        axis=0,
    )
    face_track_ids = np.concatenate(
        [
            np.full(len(faces), track_ids[index], dtype=np.int64)
            for index in range(person_count)
        ]
    )
    pixels, depth = project_virtual(vertices, camera, width, height)
    face_depth = depth[combined_faces].mean(axis=1)
    valid = np.all(depth[combined_faces] > 1e-4, axis=1)

    triangles_world = vertices[combined_faces]
    normals = np.cross(
        triangles_world[:, 1] - triangles_world[:, 0],
        triangles_world[:, 2] - triangles_world[:, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-8)
    light_direction = normalize(
        camera["eye"] - camera["target"] + 0.8 * camera["up"]
    )
    diffuse = np.abs(normals @ light_direction)
    intensity = np.clip(0.34 + 0.66 * diffuse, 0.0, 1.0)

    valid_indices = np.flatnonzero(valid)
    draw_order = valid_indices[np.argsort(-face_depth[valid_indices])]
    triangle_pixels = pixels[combined_faces[draw_order]].astype(np.int32)
    for draw_position, face_index in enumerate(draw_order):
        track_id = int(face_track_ids[face_index])
        rgb = PERSON_COLORS_RGB[track_id % len(PERSON_COLORS_RGB)]
        shaded_rgb = np.clip(rgb * intensity[face_index], 0, 255).astype(np.uint8)
        bgr = tuple(int(value) for value in shaded_rgb[::-1])
        cv2.fillConvexPoly(
            canvas,
            triangle_pixels[draw_position],
            bgr,
            lineType=cv2.LINE_AA,
        )
    return canvas


def save_gif(frame_paths: list[Path], gif_path: Path, fps: float) -> None:
    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs = common.discover_frames(
        dataset_root,
        args.dataset_split,
        args.max_frames,
    )
    first_images = common.list_frame_images(frame_dirs[0])
    if args.input_indices is not None:
        input_indices = sorted(args.input_indices)
        if not input_indices:
            raise ValueError("--input-indices must contain at least one index")
        if len(set(input_indices)) != len(input_indices):
            raise ValueError("--input-indices must not contain duplicates")
        if input_indices[0] < 0 or input_indices[-1] >= len(first_images):
            raise ValueError(
                f"--input-indices must be within [0, {len(first_images) - 1}]"
            )
    else:
        if not 1 <= args.num_input_views <= len(first_images):
            raise ValueError(
                f"--num-input-views must be within [1, {len(first_images)}]"
            )
        rng = random.Random(args.seed)
        input_indices = sorted(
            rng.sample(range(len(first_images)), args.num_input_views)
        )

    model, cfg, incompatible = common.load_model(
        args.config,
        checkpoint_path,
        device,
    )
    scale_by_extrinsics = bool(
        OmegaConf.select(cfg, "scale_by_extrinsics", default=True)
    )
    print(f"[3D-GIF] checkpoint={checkpoint_path}")
    print(f"[3D-GIF] dataset={dataset_root / args.dataset_split}")
    print(
        f"[3D-GIF] frames={len(frame_dirs)} input_indices={input_indices} "
        f"top_k={args.top_k}"
    )
    print(
        f"[3D-GIF] virtual_camera azimuth={args.azimuth_deg} "
        f"elevation={args.elevation_deg} fov={args.fov_deg}"
    )
    print(f"[3D-GIF] scale_by_extrinsics={scale_by_extrinsics}")
    print(
        f"[3D-GIF] checkpoint missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )

    all_vertices: list[np.ndarray] = []
    all_joints: list[np.ndarray] = []
    frame_records = []
    started = time.time()
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if autocast_enabled and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )

    for frame_index, frame_dir in enumerate(frame_dirs):
        frame_images = common.list_frame_images(frame_dir)
        image_paths = [str(frame_images[index]) for index in input_indices]
        avg_scale = load_frame_avg_scale(
            dataset_root,
            args.dataset_split,
            frame_dir,
            image_paths,
            scale_by_extrinsics,
        )
        images = load_and_preprocess_images(image_paths).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                predictions = model(images)

        logits = predictions.get("smpl_presence_logits")
        probabilities = (
            common.stable_sigmoid(logits[0].float().cpu().numpy())
            if logits is not None
            else np.ones(predictions["smpl_pose"].shape[1], dtype=np.float64)
        )
        slots = common.choose_slots(
            probabilities,
            args.top_k,
            args.presence_threshold,
        )
        vertices, joints = decode_people(
            predictions,
            slots,
            device,
            avg_scale,
        )
        all_vertices.append(vertices)
        all_joints.append(joints)
        frame_records.append(
            {
                "frame_index": frame_index,
                "run": frame_dir.name,
                "selected_slots": slots.tolist(),
                "selected_probabilities": [
                    float(probabilities[index]) for index in slots
                ],
                "avg_scale": avg_scale,
            }
        )
        del predictions, images
        torch.cuda.empty_cache()

        if (frame_index + 1) % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            rate = (frame_index + 1) / max(elapsed, 1e-6)
            eta = (len(frame_dirs) - frame_index - 1) / max(rate, 1e-6)
            print(
                f"[3D-GIF] inference {frame_index + 1}/{len(frame_dirs)} "
                f"{frame_dir.name} rate={rate:.3f} frame/s "
                f"ETA={eta / 60.0:.1f} min"
            )

    track_assignments = associate_tracks(all_joints)
    camera = compute_virtual_camera(
        all_vertices,
        all_joints,
        args.azimuth_deg,
        args.elevation_deg,
        args.fov_deg,
    )
    faces = np.asarray(_get_smpl_model(device, "neutral").faces, dtype=np.int64)
    rendered_paths = []
    for frame_index, (vertices, track_ids) in enumerate(
        zip(all_vertices, track_assignments)
    ):
        image = render_mesh_frame(
            vertices,
            faces,
            track_ids,
            camera,
            args.width,
            args.height,
        )
        frame_path = (
            frames_dir
            / f"{frame_index:04d}_{frame_dirs[frame_index].name}.png"
        )
        cv2.imwrite(str(frame_path), image)
        rendered_paths.append(frame_path)
        frame_records[frame_index]["track_color_ids"] = track_ids.tolist()
        frame_records[frame_index]["rendered_png"] = str(frame_path)
        if (frame_index + 1) % max(1, args.log_every) == 0:
            print(
                f"[3D-GIF] render {frame_index + 1}/{len(frame_dirs)} "
                f"{frame_dirs[frame_index].name}"
            )

    checkpoint_tag = checkpoint_output_tag(checkpoint_path)
    gif_path = output_dir / (
        f"{checkpoint_tag}_first{len(rendered_paths)}"
        f"_top{args.top_k}_pred_smpl_3d.gif"
    )
    save_gif(rendered_paths, gif_path, args.fps)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tag": checkpoint_tag,
        "config": args.config,
        "dataset": str(dataset_root / args.dataset_split),
        "uses_gt_mesh": False,
        "uses_gt_smpl": False,
        "uses_gt_camera_for_rendering": False,
        "uses_dataset_camera_calibration_for_scale": scale_by_extrinsics,
        "scale_by_extrinsics": scale_by_extrinsics,
        "render_mode": "fixed virtual camera, oblique top-down",
        "input_camera_indices": input_indices,
        "input_camera_names": [
            first_images[index].stem for index in input_indices
        ],
        "virtual_camera": {
            "azimuth_deg": args.azimuth_deg,
            "elevation_deg": args.elevation_deg,
            "fov_deg": args.fov_deg,
            "eye": camera["eye"].tolist(),
            "target": camera["target"].tolist(),
            "estimated_up": camera["up"].tolist(),
        },
        "identity_colors": PERSON_COLORS_RGB.astype(int).tolist(),
        "identity_tracking": "frame-to-frame Hungarian matching on predicted pelvis positions",
        "top_k": args.top_k,
        "fps": args.fps,
        "frame_count": len(rendered_paths),
        "resolution": [args.width, args.height],
        "gif": str(gif_path),
        "elapsed_seconds": time.time() - started,
        "frames": frame_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] GIF={gif_path}")
    print(f"[RESULT] frames={frames_dir}")
    print(f"[RESULT] manifest={manifest_path}")


if __name__ == "__main__":
    main()
