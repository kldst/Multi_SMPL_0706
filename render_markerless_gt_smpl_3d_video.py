#!/usr/bin/env python3
"""Render markerless GT SMPL meshes from NPZ archives as a fixed-view 3D video."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import infer_markerless_smpl_gif as common
from infer_markerless_smpl_3d_gif import (
    PERSON_COLORS_RGB,
    compute_virtual_camera,
    render_mesh_frame,
)
from training.smpl_body import _decode_smpl_batch, _get_smpl_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT multi-person SMPL meshes in a pure 3D scene."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(common.REPO_DIR / "MAMMA_markerless_multiple_people"),
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=28.0)
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=str(
            common.REPO_DIR
            / "outputs/gt_markerless_first500_smpl_3d_video"
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def normalize_gender(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.reshape(-1)[0]
    token = str(value).strip().lower()
    if token.startswith("m"):
        return "male"
    if token.startswith("f"):
        return "female"
    return "neutral"


def load_gt_smpl(archive_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(archive_path, allow_pickle=True) as archive:
        keys = set(archive.files)
        pose_keys = sorted(
            key
            for key in keys
            if key.startswith("out_param/")
            and key.endswith("/smpl_params/poses")
        )
        if not pose_keys:
            raise KeyError(f"No GT SMPL parameters found in {archive_path}")

        person_ids: list[str] = []
        poses, betas, translations, genders = [], [], [], []
        for pose_key in pose_keys:
            person_id = pose_key.split("/")[1]
            prefix = f"out_param/{person_id}/frame_{archive_path.stem[-5:]}/smpl_params/"
            if prefix + "poses" not in keys:
                # Do not depend on the frame directory spelling: derive the exact
                # prefix from the pose key stored in the archive.
                prefix = pose_key[: -len("poses")]
            person_ids.append(person_id)
            poses.append(
                np.asarray(archive[prefix + "poses"], dtype=np.float32)
                .reshape(-1)[:72]
            )
            betas.append(
                np.asarray(archive[prefix + "betas"], dtype=np.float32)
                .reshape(-1)[:10]
            )
            translations.append(
                np.asarray(archive[prefix + "trans"], dtype=np.float32)
                .reshape(-1)[:3]
            )
            genders.append(
                normalize_gender(
                    archive[prefix + "gender"]
                    if prefix + "gender" in keys
                    else "neutral"
                )
            )
    return (
        person_ids,
        np.stack(poses),
        np.stack(betas),
        np.stack(translations),
        genders,
    )


def encode_mp4(frames_dir: Path, output_path: Path, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / "%04d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_dirs = common.discover_frames(
        dataset_root, args.dataset_split, args.max_frames
    )

    all_vertices: list[np.ndarray] = []
    all_joints: list[np.ndarray] = []
    all_person_ids: list[list[str]] = []
    canonical_person_ids: list[str] | None = None
    records = []
    started = time.time()

    print(f"[GT-3D] dataset={dataset_root / args.dataset_split}")
    print(f"[GT-3D] frames={len(frame_dirs)}")
    for frame_index, frame_dir in enumerate(frame_dirs):
        archive_path = (
            dataset_root
            / args.dataset_split
            / "out_data"
            / f"{frame_dir.name}.npz"
        )
        person_ids, poses, betas, translations, genders = load_gt_smpl(
            archive_path
        )
        if canonical_person_ids is None:
            canonical_person_ids = sorted(person_ids)
        unknown = set(person_ids).difference(canonical_person_ids)
        if unknown:
            canonical_person_ids.extend(sorted(unknown))

        with torch.inference_mode():
            joints, vertices = _decode_smpl_batch(
                pose_aa=torch.as_tensor(
                    poses, dtype=torch.float32, device=device
                ),
                betas=torch.as_tensor(
                    betas, dtype=torch.float32, device=device
                ),
                trans=torch.as_tensor(
                    translations, dtype=torch.float32, device=device
                ),
                genders=genders,
                use_mamma=False,
            )
        all_vertices.append(
            vertices.detach().cpu().numpy().astype(np.float32)
        )
        all_joints.append(joints.detach().cpu().numpy().astype(np.float32))
        all_person_ids.append(person_ids)
        records.append(
            {
                "frame_index": frame_index,
                "run": frame_dir.name,
                "gt_archive": str(archive_path),
                "person_ids": person_ids,
            }
        )
        if (frame_index + 1) % max(1, args.log_every) == 0:
            print(
                f"[GT-3D] decode {frame_index + 1}/{len(frame_dirs)} "
                f"{frame_dir.name}"
            )

    assert canonical_person_ids is not None
    color_id_by_person = {
        person_id: index for index, person_id in enumerate(canonical_person_ids)
    }
    camera = compute_virtual_camera(
        all_vertices,
        all_joints,
        args.azimuth_deg,
        args.elevation_deg,
        args.fov_deg,
    )
    faces = np.asarray(_get_smpl_model(device, "neutral").faces, dtype=np.int64)

    for frame_index, (vertices, person_ids) in enumerate(
        zip(all_vertices, all_person_ids)
    ):
        track_ids = np.asarray(
            [color_id_by_person[person_id] for person_id in person_ids],
            dtype=np.int64,
        )
        image = render_mesh_frame(
            vertices,
            faces,
            track_ids,
            camera,
            args.width,
            args.height,
        )
        frame_path = frames_dir / f"{frame_index:04d}.png"
        if not cv2.imwrite(str(frame_path), image):
            raise IOError(f"Failed to write {frame_path}")
        records[frame_index]["rendered_png"] = str(frame_path)
        records[frame_index]["color_ids"] = track_ids.tolist()
        if (frame_index + 1) % max(1, args.log_every) == 0:
            print(
                f"[GT-3D] render {frame_index + 1}/{len(frame_dirs)} "
                f"{frame_dirs[frame_index].name}"
            )

    mp4_path = output_dir / f"gt_first{len(frame_dirs)}_smpl_3d.mp4"
    encode_mp4(frames_dir, mp4_path, args.fps)
    manifest = {
        "dataset": str(dataset_root / args.dataset_split),
        "source": "GT out_param/*/smpl_params in per-frame NPZ archives",
        "uses_prediction": False,
        "uses_gt_smpl": True,
        "render_mode": "fixed virtual camera, oblique top-down",
        "person_color_ids": color_id_by_person,
        "identity_colors": PERSON_COLORS_RGB.astype(int).tolist(),
        "fps": args.fps,
        "frame_count": len(frame_dirs),
        "resolution": [args.width, args.height],
        "mp4": str(mp4_path),
        "virtual_camera": {
            "azimuth_deg": args.azimuth_deg,
            "elevation_deg": args.elevation_deg,
            "fov_deg": args.fov_deg,
            "eye": camera["eye"].tolist(),
            "target": camera["target"].tolist(),
            "estimated_up": camera["up"].tolist(),
        },
        "elapsed_seconds": time.time() - started,
        "frames": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[RESULT] MP4={mp4_path}")
    print(f"[RESULT] manifest={manifest_path}")


if __name__ == "__main__":
    main()
