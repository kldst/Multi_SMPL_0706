#!/usr/bin/env python3
"""Render converted Harmony4D SMPL-X annotations as a fixed-camera MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_markerless_smpl_3d_gif as renderer  # noqa: E402
from training import smpl_body  # noqa: E402


PEOPLE = ("aria01", "aria02")
smpl_body.set_smplx_model_root(REPO_ROOT / "mamma" / "smplx_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory containing frame .npy files")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--elevation-deg", type=float, default=28.0)
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--title", default="GT")
    parser.add_argument("--subtitle", default="Harmony4D converted SMPL-X GT")
    return parser.parse_args()


def load_faces() -> np.ndarray:
    model_path = Path(smpl_body._SMPLX_MODEL_PATHS["neutral"])
    if model_path.suffix == ".npz":
        with np.load(model_path, allow_pickle=False) as archive:
            return np.asarray(archive["f"], dtype=np.int64)
    import pickle

    with model_path.open("rb") as stream:
        model = pickle.load(stream, encoding="latin1")
    return np.asarray(model["f"], dtype=np.int64)


def label_frame(image: np.ndarray, title: str, subtitle: str, frame: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 52), (28, 29, 33), -1)
    cv2.putText(
        image, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
        (120, 255, 140), 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, subtitle, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (185, 185, 185), 1, cv2.LINE_AA,
    )
    cv2.line(image, (0, 52), (image.shape[1], 52), (70, 72, 78), 1)
    cv2.putText(
        image, frame, (image.shape[1] - 105, image.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / "GT_3d.mp4"
    )
    frame_paths = sorted(input_dir.glob("*.npy"), key=lambda path: int(path.stem))
    if not frame_paths:
        raise FileNotFoundError(f"No .npy frames found under {input_dir}")

    vertices_by_frame: list[np.ndarray] = []
    joints_by_frame: list[np.ndarray] = []
    for path in frame_paths:
        annotation = np.load(path, allow_pickle=True).item()
        missing = [person for person in PEOPLE if person not in annotation]
        if missing:
            raise KeyError(f"{path}: missing people {missing}")
        vertices = np.stack(
            [np.asarray(annotation[p]["vertices"], dtype=np.float32) for p in PEOPLE]
        )
        joints = np.stack(
            [np.asarray(annotation[p]["joints"], dtype=np.float32) for p in PEOPLE]
        )
        if vertices.shape != (2, 10475, 3) or joints.shape != (2, 24, 3):
            raise ValueError(
                f"{path}: expected vertices (2,10475,3) and joints (2,24,3), "
                f"got {vertices.shape} and {joints.shape}"
            )
        if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            raise ValueError(f"{path}: non-finite mesh data")
        vertices_by_frame.append(vertices)
        joints_by_frame.append(joints)

    camera = renderer.compute_virtual_camera(
        vertices_by_frame,
        joints_by_frame,
        args.azimuth_deg,
        args.elevation_deg,
        args.fov_deg,
    )
    faces = load_faces()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{args.size}x{args.size}",
        "-framerate", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, (path, vertices) in enumerate(zip(frame_paths, vertices_by_frame), 1):
            image = renderer.render_mesh_frame(
                vertices, faces, np.arange(len(PEOPLE), dtype=np.int64),
                camera, args.size, args.size,
            )
            label_frame(image, args.title, args.subtitle, f"{path.stem}  {index}/{len(frame_paths)}")
            process.stdin.write(image.tobytes())
            if index % 25 == 0 or index == len(frame_paths):
                print(f"rendered {index}/{len(frame_paths)}", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    manifest = {
        "input": str(input_dir),
        "output": str(output),
        "frames": len(frame_paths),
        "first_frame": frame_paths[0].stem,
        "last_frame": frame_paths[-1].stem,
        "people": list(PEOPLE),
        "fps": args.fps,
        "resolution": [args.size, args.size],
        "camera": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in camera.items()
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
