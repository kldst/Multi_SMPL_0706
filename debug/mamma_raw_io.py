"""Lightweight raw Mamma_mv_split IO / sampling helpers.

Deliberately imports only argparse / pathlib / random / cv2 / numpy -- NO torch,
trimesh, or ``training.*`` -- so debug scripts that only need to *read* the raw
data (images, .data.pyd, .mask.jpg) start instantly. The heavier
``mamma_debug_common`` (SMPL decode, trimesh scenes) takes ~90s just to import;
scripts that don't decode SMPL should use this module instead.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
# Make repo-root packages (``training.*``) importable when this file is imported
# from a script whose sys.path[0] is the ``debug/`` folder.
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
DEFAULT_MAMMA_ROOT = Path("/mnt/train-data-4-hdd/yian/Mamma_mv_split/train")
DEFAULT_SCENE_ROOT = Path("tmp/bedlam_lab_20251031_191436/harmony4d_train_1_NC_200_00")
DEFAULT_OUTPUT_ROOT = REPO_DIR / "debug_outputs" / "mamma_pipeline"
IMAGE_SIZE = 518


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mamma-root", type=Path, default=DEFAULT_MAMMA_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--frame", default="")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    return parser


def scene_search_root(mamma_root: Path, scene_root: Path) -> Path:
    scene_root = Path(scene_root)
    return scene_root if scene_root.is_absolute() else Path(mamma_root) / scene_root


def is_sequence_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        v.is_dir() and any(v.glob("*.data.pyd")) for v in path.iterdir()
    )


def list_sequences(mamma_root: Path, scene_root: Path) -> List[Path]:
    root = scene_search_root(mamma_root, scene_root)
    png_root = root / "png" if (root / "png").is_dir() else root
    if is_sequence_dir(png_root):
        return [png_root]
    seqs = sorted(p for p in png_root.iterdir() if is_sequence_dir(p))
    if not seqs:
        raise FileNotFoundError(f"No raw Mamma sequence folders under {png_root}")
    return seqs


def choose_sequence(mamma_root: Path, scene_root: Path, seq_name: str, rng: random.Random) -> Path:
    seqs = list_sequences(mamma_root, scene_root)
    if seq_name:
        matches = [p for p in seqs if p.name == seq_name or str(p).endswith(seq_name)]
        if not matches:
            raise FileNotFoundError(f"Sequence {seq_name!r} not found")
        return matches[0]
    return rng.choice(seqs)


def group_frames(seq_dir: Path) -> Dict[str, List[str]]:
    frame_to_views: Dict[str, List[str]] = {}
    for view_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
        for data_path in view_dir.glob("*.data.pyd"):
            frame = data_path.name.split(".")[0]
            if (view_dir / f"{frame}.jpg").is_file() or (view_dir / f"{frame}.png").is_file():
                frame_to_views.setdefault(frame, []).append(view_dir.name)
    return {k: sorted(v) for k, v in frame_to_views.items()}


def choose_frame_and_views(seq_dir: Path, frame: str, num_views: int, rng: random.Random) -> Tuple[str, List[str]]:
    grouped = group_frames(seq_dir)
    candidates = [(f, views) for f, views in grouped.items() if len(views) >= num_views]
    if not candidates:
        raise RuntimeError(f"No frames with >= {num_views} views in {seq_dir}")
    if frame:
        frame = str(frame).zfill(4)
        if frame not in grouped or len(grouped[frame]) < num_views:
            raise RuntimeError(f"Frame {frame} lacks {num_views} views in {seq_dir}")
        views = grouped[frame]
    else:
        frame, views = rng.choice(candidates)
    return frame, sorted(rng.sample(views, num_views))


def load_rgb_resized(path: Path, size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR), (h, w)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
