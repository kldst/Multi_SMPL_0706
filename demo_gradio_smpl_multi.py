# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import atexit
import json
import pickle
import socket
import subprocess

# Work around a PyTorch nightly backend autoload hang during `import torch`.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import cv2
import torch
import numpy as np
import gradio as gr
import trimesh
import sys
import shutil
import json
import struct
from datetime import datetime
import glob
import gc
import time
import re
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
from scipy.optimize import linear_sum_assignment

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_REPO_DIR)
sys.path.append(os.path.join(_REPO_DIR, "training"))
sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, project_world_points_to_cam
from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_image_cv2
from training.loss import axis_angle_to_rotmat
from training.train_utils.normalization import normalize_camera_extrinsics_points_and_3djoints_batch
import argparse
from hydra import initialize, compose
from typing import Any, Dict
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as _np
from PIL import Image
from contextlib import nullcontext
# Reuse the standalone attention-viz helpers (no side effects on import).
from visualize_attention import (
    CrossAttnCapture,
    select_attention,
    overlay_heatmap,
    make_grid,
    add_title_bar,
)

PYTORCH3D_PROJECTION_PYTHON = os.environ.get(
    "PYTORCH3D_PROJECTION_PYTHON",
    "/home/clchen/miniconda3/envs/vggt/bin/python",
)
PYTORCH3D_PROJECTION_WORKER_SCRIPT = os.path.join(_REPO_DIR, "render_mesh_projection_worker.py")
_projection_worker = None

# --- chumpy / 舊 SMPL 相容性修正（一定要在 import smplx 之前） ---
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "complex"):
    _np.complex = complex
if not hasattr(_np, "object"):
    _np.object = object
if not hasattr(_np, "str"):
    _np.str = str
if not hasattr(np, "unicode"):
    np.unicode = str
# chumpy 0.70 still calls inspect.getargspec, removed in Python 3.10+.
import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec

from smplx import SMPL

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")


def find_available_port(start_port: int = 6815, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts - 1}"
    )

_LOCAL_EXAMPLES = os.path.join(_REPO_DIR, "examples")
allowed_paths = [_LOCAL_EXAMPLES]
os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(allowed_paths)
os.environ["GRADIO_TEMP_DIR"] = "./.gradio_cache"
RUNTIME_OUTPUT_ROOT = os.path.join(_REPO_DIR, ".gradio_cache", "runtime_outputs")

parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
parser.add_argument(
    "--config",
    type=str,
    default="test",
    help="Name of the config file (without .yaml extension, default: default)"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/mnt/train-data-4-hdd/clchen/vggt_multi/training/logs/0604_multi_100k_trans_resume/ckpts/checkpoint.pt",
    help="Path to the model checkpoint (.pt) to load",
)
parser.add_argument(
    "--dataset-root",
    type=str,
    default="/mnt/train-data-5-hdd/clchen/SMPL_multi_dataset_eval/MAMMA_eval_dance",
    help="Root directory of the evaluation dataset",
)
parser.add_argument(
    "--dataset-split",
    type=str,
    default="test",
    help="Dataset split to use (e.g. train / test)",
)
parser.add_argument(
    "--demo-seq-dir",
    type=str,
    default="",
    help="Run a raw Mamma sequence directory directly instead of launching Gradio.",
)
parser.add_argument(
    "--demo-view",
    type=str,
    default="",
    help="Camera/view folder to use inside --demo-seq-dir. Defaults to the first sorted view.",
)
parser.add_argument(
    "--demo-sequence",
    type=str,
    default="",
    help="Sequence folder name to use when --demo-seq-dir points at a raw Mamma dataset root.",
)
parser.add_argument(
    "--demo-max-frames",
    type=int,
    default=10,
    help="Number of frames to copy from the selected view for CLI demo.",
)
parser.add_argument(
    "--demo-fps",
    type=float,
    default=2.0,
    help="FPS for the projected-mesh GIF written by CLI demo.",
)
parser.add_argument(
    "--postprocess-target-dir",
    type=str,
    default="",
    help="Build missing SMPL mesh/projection/GIF from an existing target_dir with predictions.npz.",
)
args = parser.parse_args()

with initialize(version_base=None, config_path="training/config"):
    cfg = compose(config_name=args.config)

OmegaConf.set_struct(cfg, False)
cfg.model.enable_point = True

# SMPL multi-person head flags.
#   enable_smpl_multi_query        -> head predicts `smpl_trans` (translation in world space)
#   enable_smpl_multi_query_trans  -> head predicts `mesh_translate` (SMPL root directly in the
#                                     first-camera normalized coordinate frame)
# Both may be enabled at once: the model then produces BOTH `smpl_trans` and `mesh_translate`.
# In that case we prefer `mesh_translate` (the placement target the trans head is trained for).
ENABLE_SMPL_MULTI_QUERY_TRANS = bool(
    OmegaConf.select(cfg, "model.enable_smpl_multi_query_trans", default=False)
)
ENABLE_SMPL_MULTI_QUERY = bool(
    OmegaConf.select(cfg, "model.enable_smpl_multi_query", default=False)
)
print(
    f"[CFG] config={args.config} "
    f"enable_smpl_multi_query={ENABLE_SMPL_MULTI_QUERY} "
    f"enable_smpl_multi_query_trans={ENABLE_SMPL_MULTI_QUERY_TRANS}"
)


def resolve_smpl_hungarian_cost_weights(config) -> dict[str, float]:
    return {
        "pose": float(OmegaConf.select(config, "loss.smpl.hungarian_cost_pose_weight", default=1.0)),
        "beta": float(OmegaConf.select(config, "loss.smpl.hungarian_cost_beta_weight", default=0.1)),
        "trans": float(OmegaConf.select(config, "loss.smpl.hungarian_cost_trans_weight", default=0.0)),
        "mesh_trans": float(OmegaConf.select(config, "loss.smpl.hungarian_cost_mesh_trans_weight", default=0.0)),
        "presence": float(OmegaConf.select(config, "loss.smpl.hungarian_cost_presence_weight", default=0.0)),
    }


SMPL_HUNGARIAN_COST_WEIGHTS = resolve_smpl_hungarian_cost_weights(cfg)
print(f"[SMPL] Hungarian cost weights from config: {SMPL_HUNGARIAN_COST_WEIGHTS}")

DEFAULT_DATASET_ROOT = str(args.dataset_root)
DEFAULT_DATASET_SPLIT = str(args.dataset_split)


def _run_sort_key(run_name: str) -> tuple[int, object]:
    match = re.fullmatch(r"runs_(\d+)", str(run_name))
    if match:
        return (0, int(match.group(1)))
    return (1, str(run_name))


def discover_image_subdir(dataset_root: str, split: str) -> str:
    """Locate the folder that directly contains the run_* dirs, relative to <root>/<split>.

    Two dataset layouts are supported by auto-detection (keeping the fixed 'out_image'
    component that infer_run_npz_path relies on):
      A) runs sit directly under out_image            -> 'out_image'        (e.g. MAMMA_eval_dance)
      B) a category folder holds the runs             -> 'out_image/<cat>'  (e.g. out_image/CLOTH3D)
    """
    out_image = Path(dataset_root) / str(split) / "out_image"
    if not out_image.is_dir():
        return "out_image/CLOTH3D"

    subdirs = sorted(entry.name for entry in os.scandir(out_image) if entry.is_dir(follow_symlinks=False))
    # Layout A: out_image itself holds the runs.
    if any(name.startswith("runs_") for name in subdirs):
        return "out_image"
    # Layout B: prefer a category folder that actually contains runs.
    for name in subdirs:
        try:
            if any(child.name.startswith("runs_") for child in os.scandir(out_image / name)):
                return f"out_image/{name}"
        except OSError:
            continue
    if subdirs:
        return f"out_image/{subdirs[0]}"
    return "out_image/CLOTH3D"


def discover_default_run(dataset_root: str, split: str, image_subdir: str) -> str | None:
    """Pick the first available run under the image subdir, or None if there are none."""
    run_root = Path(dataset_root) / str(split) / image_subdir
    if not run_root.is_dir():
        return None
    runs = [
        entry.name
        for entry in os.scandir(run_root)
        if entry.name.startswith("runs_") and entry.is_dir(follow_symlinks=False)
    ]
    runs.sort(key=_run_sort_key)
    return runs[0] if runs else None


def _looks_like_raw_mamma_sequence(path: Path) -> bool:
    if not path.is_dir():
        return False
    for view_dir in sorted(p for p in path.iterdir() if p.is_dir())[:4]:
        try:
            has_data = any(view_dir.glob("*.data.pyd"))
            has_image = any(
                p.is_file()
                and p.suffix.lower() in {".png", ".jpg"}
                and ".mask" not in p.name
                for p in view_dir.iterdir()
            )
        except OSError:
            continue
        if has_data and has_image:
            return True
    return False


def _raw_mamma_search_base(dataset_root: str, split: str) -> Path:
    root = Path(str(dataset_root or "").strip())
    split_root = root / str(split)
    return split_root if split_root.is_dir() else root


def _find_raw_mamma_sequences(dataset_root: str, split: str) -> list[Path]:
    base = _raw_mamma_search_base(dataset_root, split)
    if _looks_like_raw_mamma_sequence(base):
        return [base]
    # 'png' scene roots can sit at varying depth, e.g. Mamma_mv_split nests them under
    # train/tmp/<batch>/<dataset>/png/<seq>. Search a few levels deep (not full rglob, to
    # stay fast on large image trees).
    roots, seen = [], set()
    if (base / "png").is_dir():
        roots.append(base / "png")
        seen.add(base / "png")
    for depth in range(1, 6):
        pattern = "/".join(["*"] * depth + ["png"])
        for p in base.glob(pattern):
            if p.is_dir() and p not in seen:
                seen.add(p)
                roots.append(p)
    sequences = []
    for png_root in roots:
        try:
            sequences.extend(p for p in png_root.iterdir() if _looks_like_raw_mamma_sequence(p))
        except OSError:
            continue
    return sorted(sequences)


def _raw_mamma_run_name(dataset_root: str, split: str, seq_dir: Path) -> str:
    base = _raw_mamma_search_base(dataset_root, split)
    try:
        return str(seq_dir.relative_to(base))
    except ValueError:
        return str(seq_dir)


def is_raw_mamma_dataset_root(dataset_root: str, split: str) -> bool:
    return len(_find_raw_mamma_sequences(dataset_root, split)) > 0


DATASET_IMAGE_SUBDIR = discover_image_subdir(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT)
_DEFAULT_RAW_SEQUENCES = _find_raw_mamma_sequences(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT)
_DEFAULT_RAW_RUNS = [
    _raw_mamma_run_name(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT, seq_dir)
    for seq_dir in _DEFAULT_RAW_SEQUENCES
]
DEFAULT_DATASET_RUN = (
    _DEFAULT_RAW_RUNS[0]
    if _DEFAULT_RAW_RUNS
    else discover_default_run(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT, DATASET_IMAGE_SUBDIR)
)
print(
    f"[DATASET] root={DEFAULT_DATASET_ROOT} split={DEFAULT_DATASET_SPLIT} "
    f"image_subdir={DATASET_IMAGE_SUBDIR} default_run={DEFAULT_DATASET_RUN}"
)
DEFAULT_INPUT_IMAGE_DIR = (
    str(_DEFAULT_RAW_SEQUENCES[0])
    if _DEFAULT_RAW_SEQUENCES
    else os.path.join(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT, DATASET_IMAGE_SUBDIR, DEFAULT_DATASET_RUN or "")
    if DEFAULT_DATASET_RUN
    else ""
)
allowed_paths.extend([
    DEFAULT_DATASET_ROOT,
    "/mnt/train-data-4-hdd/yian/Mamma_mv_split",
])
os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(allowed_paths)
DEFAULT_IMAGE_IDS = os.environ.get("DEMO_IMAGE_IDS", "0 1 2 3 4 5 6 7")
SUPPORTED_IMAGE_EXTS = {".png", ".jpg"}
SMPL_PRESENCE_THRESHOLD = 0.5
GLB_VIS_VERSION = "v3_single_model_smallcam"
SMPL_MESH_PALETTE = [
    ("red", (230, 57, 70, 255)),
    ("blue", (29, 126, 214, 255)),
    ("green", (42, 157, 143, 255)),
    ("orange", (244, 162, 97, 255)),
    ("purple", (131, 56, 236, 255)),
    ("yellow", (233, 196, 106, 255)),
    ("cyan", (0, 188, 212, 255)),
    ("pink", (255, 99, 164, 255)),
]

# Uniform colors used when overlaying the GT mesh on top of the predicted mesh in the same
# scene, so the two sources are easy to tell apart (prediction = blue, GT = orange).
PRED_OVERLAY_MESH_COLOR = (29, 126, 214, 255)   # blue
GT_OVERLAY_MESH_COLOR = (244, 162, 97, 255)     # orange


class Demo:
    def __init__(self, model: Dict[str, Any], **kwargs,):
        self.model_conf = model
        self.model = instantiate(self.model_conf, _recursive_=False)

    def get_model(self):
        return self.model


demo = Demo(**cfg)
model = demo.get_model()

# 透過 --checkpoint 指定要載入的 checkpoint 路徑（trans head 請改成對應的 trans checkpoint）
MY_MODEL_PATH = str(args.checkpoint)


def load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


finetune_load = model.load_state_dict(load_checkpoint_state_dict(MY_MODEL_PATH), strict=False)
print(
    f"[CKPT] Loaded model from {MY_MODEL_PATH} "
    f"(missing={len(finetune_load.missing_keys)}, unexpected={len(finetune_load.unexpected_keys)})"
)


model.eval()
model = model.to(device)


def get_smpl_mesh_color(mesh_idx: int) -> tuple[str, tuple[int, int, int, int]]:
    color_idx = int(mesh_idx) % len(SMPL_MESH_PALETTE)
    return SMPL_MESH_PALETTE[color_idx]


def rgba_to_hex(rgba: tuple[int, int, int, int]) -> str:
    return f"#{int(rgba[0]):02x}{int(rgba[1]):02x}{int(rgba[2]):02x}"


def build_smpl_mesh_color_names(people_count: int) -> np.ndarray:
    names = []
    for mesh_idx in range(int(people_count)):
        name, rgba = get_smpl_mesh_color(mesh_idx)
        names.append(f"{name} ({rgba_to_hex(rgba)})")
    return np.asarray(names, dtype="<U32")


def build_smpl_vertex_colors(vertices: np.ndarray, people_count: int) -> np.ndarray:
    vertices = np.asarray(vertices)
    people_count = int(people_count)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or people_count <= 0:
        return np.tile(np.array([[150, 150, 150, 255]], dtype=np.uint8), (vertices.reshape(-1, 3).shape[0], 1))

    vertex_count = vertices.shape[0]
    colors = np.tile(np.array([[150, 150, 150, 255]], dtype=np.uint8), (vertex_count, 1))
    verts_per_person = vertex_count // people_count
    if verts_per_person <= 0:
        return colors

    for mesh_idx in range(people_count):
        start = mesh_idx * verts_per_person
        end = vertex_count if mesh_idx == people_count - 1 else min((mesh_idx + 1) * verts_per_person, vertex_count)
        _, rgba = get_smpl_mesh_color(mesh_idx)
        colors[start:end] = np.asarray(rgba, dtype=np.uint8)
    return colors

GLB_JSON_CHUNK = 0x4E4F534A
GLB_LIGHTS_EXTENSION = "KHR_lights_punctual"


def safe_filename_component(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def _quat_from_negative_z(direction: np.ndarray) -> list[float]:
    source = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    target = np.asarray(direction, dtype=np.float64)
    target = target / (np.linalg.norm(target) + 1e-12)

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]

    axis = np.cross(source, target)
    scale = np.sqrt((1.0 + dot) * 2.0)
    quat = np.concatenate([axis / scale, [scale * 0.5]])
    return [float(x) for x in quat]


def _read_glb_chunks(glb_path: str) -> list[tuple[int, bytes]]:
    with open(glb_path, "rb") as f:
        data = f.read()

    if len(data) < 12:
        raise ValueError("GLB file is too small")

    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError("Only binary glTF 2.0 GLB files are supported")
    if total_length != len(data):
        raise ValueError("GLB header length does not match file size")

    chunks = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Invalid GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        if len(chunk_data) != chunk_length:
            raise ValueError("Invalid GLB chunk length")
        chunks.append((chunk_type, chunk_data))
        offset += chunk_length
    return chunks


def _write_glb_chunks(glb_path: str, chunks: list[tuple[int, bytes]]) -> None:
    payload = bytearray()
    for chunk_type, chunk_data in chunks:
        payload.extend(struct.pack("<II", len(chunk_data), chunk_type))
        payload.extend(chunk_data)

    header = struct.pack("<III", 0x46546C67, 2, 12 + len(payload))
    with open(glb_path, "wb") as f:
        f.write(header)
        f.write(payload)


def add_lights_to_glb(glb_path: str) -> None:
    """
    Add punctual lights directly to an exported GLB so Gradio's Babylon Model3D
    viewer renders shaded meshes with stronger scene lighting.
    """
    if not glb_path or not str(glb_path).lower().endswith(".glb") or not os.path.exists(glb_path):
        return

    try:
        chunks = _read_glb_chunks(glb_path)
        json_idx = next(i for i, (chunk_type, _) in enumerate(chunks) if chunk_type == GLB_JSON_CHUNK)
        gltf = json.loads(chunks[json_idx][1].rstrip(b" \t\r\n\0").decode("utf-8"))

        extensions = gltf.setdefault("extensions", {})
        light_ext = extensions.setdefault(GLB_LIGHTS_EXTENSION, {})
        lights = light_ext.setdefault("lights", [])
        if any(str(light.get("name", "")).startswith("gradio_scene_") for light in lights):
            return

        extension_used = gltf.setdefault("extensionsUsed", [])
        if GLB_LIGHTS_EXTENSION not in extension_used:
            extension_used.append(GLB_LIGHTS_EXTENSION)

        light_specs = [
            {
                "name": "gradio_scene_key_light",
                "type": "directional",
                "color": [1.0, 0.96, 0.90],
                "intensity": 4.0,
                "direction": [0.35, -0.45, -0.82],
            },
            {
                "name": "gradio_scene_fill_light",
                "type": "directional",
                "color": [0.65, 0.78, 1.0],
                "intensity": 1.6,
                "direction": [-0.65, 0.15, -0.55],
            },
            {
                "name": "gradio_scene_rim_light",
                "type": "directional",
                "color": [1.0, 1.0, 1.0],
                "intensity": 2.4,
                "direction": [0.2, 0.7, 0.45],
            },
        ]

        nodes = gltf.setdefault("nodes", [])
        scenes = gltf.setdefault("scenes", [{"nodes": []}])
        scene_idx = int(gltf.get("scene", 0))
        if scene_idx >= len(scenes):
            scene_idx = 0
            gltf["scene"] = 0
        scene_nodes = scenes[scene_idx].setdefault("nodes", [])

        for spec in light_specs:
            light_idx = len(lights)
            lights.append(
                {
                    "name": spec["name"],
                    "type": spec["type"],
                    "color": spec["color"],
                    "intensity": spec["intensity"],
                }
            )
            node_idx = len(nodes)
            nodes.append(
                {
                    "name": f"{spec['name']}_node",
                    "rotation": _quat_from_negative_z(np.asarray(spec["direction"], dtype=np.float64)),
                    "extensions": {GLB_LIGHTS_EXTENSION: {"light": light_idx}},
                }
            )
            scene_nodes.append(node_idx)

        json_data = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_data += b" " * ((4 - len(json_data) % 4) % 4)
        chunks[json_idx] = (GLB_JSON_CHUNK, json_data)
        _write_glb_chunks(glb_path, chunks)
        print(f"[GLB] Added Gradio scene lights to: {glb_path}")
    except Exception as e:
        print(f"[WARN] Failed to add GLB lights: {e}")


def build_smpl_scene_mesh(predictions: dict) -> trimesh.Trimesh | None:
    if "smpl_vertices" not in predictions or "smpl_faces" not in predictions:
        return None

    vertices = np.asarray(predictions["smpl_vertices"])
    faces = np.asarray(predictions["smpl_faces"])
    if vertices.size == 0 or faces.size == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=vertices.reshape(-1, 3).astype(np.float64),
        faces=faces.reshape(-1, 3).astype(np.int64),
        process=False,
    )
    vertex_colors = predictions.get("smpl_vertex_colors", None)
    if vertex_colors is None:
        visible_indices = np.asarray(predictions.get("smpl_visible_indices", []))
        people_count = int(visible_indices.shape[0]) if visible_indices.size else 1
        vertex_colors = build_smpl_vertex_colors(mesh.vertices, people_count)
    vertex_colors = np.asarray(vertex_colors, dtype=np.uint8).reshape(-1, 4)
    if vertex_colors.shape[0] != len(mesh.vertices):
        vertex_colors = np.tile(np.array([[150, 150, 150, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = vertex_colors
    return mesh


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def select_present_smpl_slots(
    predictions: dict,
    presence_threshold: float = SMPL_PRESENCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    poses = ensure_people_array(predictions["smpl_pose"], 72, "smpl_pose")
    betas = ensure_people_array(predictions["smpl_beta"], 10, "smpl_beta")
    trans = None
    if predictions.get("smpl_trans", None) is not None:
        trans = ensure_people_array(predictions["smpl_trans"], 3, "smpl_trans")

    people_count = min(poses.shape[0], betas.shape[0], trans.shape[0] if trans is not None else poses.shape[0])
    poses = poses[:people_count]
    betas = betas[:people_count]
    if trans is not None:
        trans = trans[:people_count]

    presence_logits = predictions.get("smpl_presence_logits", None)
    if presence_logits is None:
        presence_prob = np.ones((people_count,), dtype=np.float64)
        presence_mask = np.ones((people_count,), dtype=bool)
    else:
        raw_prob = _sigmoid_np(np.asarray(presence_logits).reshape(-1))
        presence_prob = np.zeros((people_count,), dtype=np.float64)
        valid_count = min(people_count, raw_prob.shape[0])
        presence_prob[:valid_count] = raw_prob[:valid_count]
        presence_mask = presence_prob >= float(presence_threshold)

    selected_indices = np.flatnonzero(presence_mask)
    predictions["smpl_presence_prob"] = presence_prob.astype(np.float32)
    predictions["smpl_presence_mask"] = presence_mask.astype(np.bool_)
    predictions["smpl_visible_indices"] = selected_indices.astype(np.int64)
    print(
        "[SMPL] presence prob:",
        np.array2string(presence_prob, precision=3),
        f"threshold={presence_threshold}",
        f"selected={selected_indices.tolist()}",
    )

    selected_trans = trans[presence_mask] if trans is not None else None
    return poses[presence_mask], betas[presence_mask], selected_trans, presence_prob, selected_indices


def format_smpl_presence_output(
    predictions: dict,
    use_gt: bool = False,
    presence_threshold: float = SMPL_PRESENCE_THRESHOLD,
) -> str:
    if use_gt:
        return "SMPL presence:\n- Use GT enabled: no model presence prediction was produced."
    presence_logits = predictions.get("smpl_presence_logits", None)
    if presence_logits is None:
        return "SMPL presence:\n- N/A"

    logits = np.asarray(presence_logits, dtype=np.float64).reshape(-1)
    probs = predictions.get("smpl_presence_prob", None)
    if probs is None:
        probs = _sigmoid_np(logits)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    selected = predictions.get("smpl_visible_indices", np.flatnonzero(probs >= float(presence_threshold)))
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    selected_set = {int(idx) for idx in selected.tolist()}
    mesh_selection = predictions.get("smpl_mesh_selection_mode", "presence_threshold")

    lines = [
        "SMPL presence probabilities:",
        "",
        f"- threshold: {float(presence_threshold):.4f}",
        f"- mesh selection: {mesh_selection}",
        f"- selected slots: {selected.tolist()}",
        "",
        "| slot | probability | selected |",
        "| ---: | ---: | :--- |",
    ]
    for slot_idx, prob in enumerate(probs):
        selected_text = "yes" if slot_idx in selected_set else ""
        lines.append(f"| {slot_idx} | {prob:.6f} | {selected_text} |")
    return "\n".join(lines)


def select_smpl_slots_for_mesh(
    predictions: dict,
    presence_threshold: float,
    use_hungarian_mesh_selection: bool,
    matching_metrics: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    threshold_pose, threshold_beta, threshold_trans, presence_prob, threshold_indices = select_present_smpl_slots(
        predictions,
        presence_threshold=presence_threshold,
    )

    if not use_hungarian_mesh_selection:
        predictions["smpl_mesh_selection_mode"] = "presence_threshold"
        return threshold_pose, threshold_beta, threshold_trans, presence_prob, threshold_indices

    matching_metrics = matching_metrics or {}
    matched_indices = np.asarray(matching_metrics.get("matched_pred_indices", []), dtype=np.int64).reshape(-1)
    if matched_indices.size == 0:
        predictions["smpl_mesh_selection_mode"] = "presence_threshold_fallback_no_hungarian_match"
        print("[SMPL] Hungarian mesh selection requested but no matched_pred_indices found; using presence threshold.")
        return threshold_pose, threshold_beta, threshold_trans, presence_prob, threshold_indices

    poses = ensure_people_array(predictions["smpl_pose"], 72, "smpl_pose")
    betas = ensure_people_array(predictions["smpl_beta"], 10, "smpl_beta")
    trans = None
    if predictions.get("smpl_trans", None) is not None:
        trans = ensure_people_array(predictions["smpl_trans"], 3, "smpl_trans")
    people_count = min(poses.shape[0], betas.shape[0], trans.shape[0] if trans is not None else poses.shape[0])

    selected_indices = []
    for pred_idx in matched_indices.tolist():
        pred_idx = int(pred_idx)
        if 0 <= pred_idx < people_count and pred_idx not in selected_indices:
            selected_indices.append(pred_idx)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices.size == 0:
        predictions["smpl_mesh_selection_mode"] = "presence_threshold_fallback_invalid_hungarian_indices"
        print("[SMPL] Hungarian matched indices were outside valid SMPL slots; using presence threshold.")
        return threshold_pose, threshold_beta, threshold_trans, presence_prob, threshold_indices

    predictions["smpl_visible_indices"] = selected_indices
    predictions["smpl_mesh_selection_mode"] = "hungarian_matched_pred_indices"
    print(f"[SMPL] Mesh selection uses Hungarian matched pred slots {selected_indices.tolist()}.")
    selected_trans = trans[selected_indices] if trans is not None else None
    return poses[selected_indices], betas[selected_indices], selected_trans, presence_prob, selected_indices


def list_existing_target_images(target_dir: object) -> list[str]:
    image_dir = Path(str(target_dir or "")) / "images"
    if not image_dir.is_dir():
        return []
    return sorted(
        str(path)
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


# --------------------------------------------------------
# SMPL model cache
# --------------------------------------------------------
SMPL_MODEL_PATHS = {
    "female": os.path.join(_REPO_DIR, "smpl_models", "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
    "male": os.path.join(_REPO_DIR, "smpl_models", "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    "neutral": os.path.join(_REPO_DIR, "smpl_models", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
}
_SMPL_MODELS: dict[str, SMPL] = {}


def normalize_gender(gender) -> str:
    if isinstance(gender, np.ndarray):
        gender = gender.reshape(-1)[0] if gender.size else "neutral"
    if isinstance(gender, np.generic):
        gender = gender.item()
    if isinstance(gender, (bytes, bytearray)):
        gender = gender.decode("utf-8", errors="ignore")
    text = str(gender).strip().lower()
    if text.startswith("m") or text == "0":
        return "male"
    if text.startswith("f") or text == "1":
        return "female"
    return "neutral"


def get_smpl_model_by_gender(gender) -> SMPL:
    gender = normalize_gender(gender)
    if gender not in _SMPL_MODELS:
        _SMPL_MODELS[gender] = SMPL(
            model_path=SMPL_MODEL_PATHS[gender],
            gender=gender,
            batch_size=1,
        ).to(device)
        _SMPL_MODELS[gender].eval()
    return _SMPL_MODELS[gender]


def smpl_to_mesh_and_obj(
    smpl_pose: np.ndarray,
    smpl_beta: np.ndarray,
    smpl_trans: np.ndarray | None,
    out_obj_path: str,
    genders: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = smpl_to_mesh_joints_and_obj(
        smpl_pose,
        smpl_beta,
        smpl_trans,
        out_obj_path,
        genders=genders,
    )
    return verts, faces


def smpl_to_mesh_joints_and_obj(
    smpl_pose: np.ndarray,
    smpl_beta: np.ndarray,
    smpl_trans: np.ndarray | None,
    out_obj_path: str,
    genders: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    將多人的 smpl_pose / smpl_beta / smpl_trans 丟進 SMPL model，產生合併 mesh，
    並寫出 OBJ 檔案。

    smpl_pose: (P, 72) / (1, P, 72) / (72,)
    smpl_beta: (P, 10) / (1, P, 10) / (10,)
    smpl_trans: (P, 3) / (1, P, 3) / (3,)
    """

    print(f"[SMPL] Generating SMPL mesh...")

    poses = ensure_people_array(smpl_pose, 72, "smpl_pose")
    betas = ensure_people_array(smpl_beta, 10, "smpl_beta")
    trans = ensure_people_array(
        np.zeros((poses.shape[0], 3), dtype=np.float32) if smpl_trans is None else smpl_trans,
        3,
        "smpl_trans",
    )
    people_count = min(poses.shape[0], betas.shape[0], trans.shape[0])
    if genders is None:
        genders = ["neutral"] * people_count

    all_verts = []
    all_faces = []
    all_joints = []
    person_faces = []
    offset = 0
    for person_idx in range(people_count):
        verts, faces, joints = smpl_forward_with_joints(
            poses[person_idx],
            betas[person_idx],
            trans[person_idx],
            genders[person_idx] if person_idx < len(genders) else "neutral",
        )
        all_verts.append(verts)
        all_joints.append(joints)
        person_faces.append(faces.astype(np.int64))
        all_faces.append(faces.astype(np.int64) + offset)
        offset += verts.shape[0]

    if not all_verts:
        raise ValueError("No SMPL people to export.")

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    joints = np.stack(all_joints, axis=0)

    # ---- 寫成 OBJ ----
    write_obj(verts, faces, out_obj_path)
    per_person_dir = os.path.join(os.path.dirname(out_obj_path), "smpl_people")
    per_person_paths = write_per_person_objs(all_verts, person_faces, per_person_dir)

    print(f"[SMPL] ✔ OBJ saved: {out_obj_path}")
    print(f"[SMPL] ✔ Per-person OBJs saved: {per_person_dir} ({len(per_person_paths)} files)")
    print(f"[SMPL] ✔ #verts={verts.shape}, #faces={faces.shape}")

    return verts, faces, joints


def ensure_people_array(value: np.ndarray, last_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim > 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[-1] < last_dim:
        raise ValueError(f"Expected {name} shape (..., {last_dim}), got {arr.shape}")
    return arr[:, :last_dim]


def smpl_forward_with_joints(
    smpl_pose: np.ndarray,
    smpl_beta: np.ndarray,
    smpl_trans: np.ndarray | None,
    gender="neutral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smpl_pose = np.asarray(smpl_pose, dtype=np.float32).reshape(-1)[:72]
    smpl_beta = np.asarray(smpl_beta, dtype=np.float32).reshape(-1)[:10]
    smpl_trans = (
        np.zeros(3, dtype=np.float32)
        if smpl_trans is None
        else np.asarray(smpl_trans, dtype=np.float32).reshape(-1)[:3]
    )

    pose_t = torch.from_numpy(smpl_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
    beta_t = torch.from_numpy(smpl_beta).to(device=device, dtype=torch.float32).unsqueeze(0)
    trans_t = torch.from_numpy(smpl_trans).to(device=device, dtype=torch.float32).unsqueeze(0)
    smpl_model_local = get_smpl_model_by_gender(gender)

    with torch.no_grad():
        smpl_out = smpl_model_local(
            global_orient=pose_t[:, :3],
            body_pose=pose_t[:, 3:72],
            betas=beta_t,
            transl=trans_t,
        )

    verts = smpl_out.vertices[0].detach().cpu().numpy().astype(np.float64)
    joints = smpl_out.joints[0, :24].detach().cpu().numpy().astype(np.float64)
    faces = smpl_model_local.faces.copy().astype(np.int64)
    return verts, faces, joints


def orthonormalize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Strip any uniform scale baked into the rotation block of world->cam extrinsics.

    Some datasets (e.g. Harmony4D) store `worldToCamera` with a scalar `s` folded into R
    (so det(R)=s^3 != 1). A uniform scale on camera-space coordinates cancels under perspective
    division, so [R/s | t/s] is a proper SE(3) extrinsic that projects to the SAME pixels while
    being compatible with the SE(3)-based camera0-gauge normalization. For already-orthonormal
    extrinsics (s=1) this is a no-op.
    """
    extr = np.asarray(extrinsics, dtype=np.float64).copy()
    R = extr[..., :3, :3]
    det = np.linalg.det(R)
    s = np.sign(det) * np.abs(det) ** (1.0 / 3.0)
    s = np.where(np.abs(s) < 1e-9, 1.0, s)
    extr[..., :3, :3] = R / s[..., None, None]
    extr[..., :3, 3] = extr[..., :3, 3] / s[..., None]
    return extr


def normalize_vertices_to_camera0_gauge(
    vertices_world: np.ndarray,
    raw_extrinsics: np.ndarray,
    avg_scale: float,
) -> np.ndarray:
    raw_extrinsics = np.asarray(raw_extrinsics, dtype=np.float64)
    vertices_world = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    R0 = raw_extrinsics[0, :3, :3]
    t0 = raw_extrinsics[0, :3, 3]
    scale = max(float(avg_scale), 1e-6)
    return ((vertices_world @ R0.T) + t0.reshape(1, 3)) / scale


def scale_vertices_to_camera0_gauge(vertices: np.ndarray, avg_scale: float) -> np.ndarray:
    """Scale-only gauge (no cam0 rotation), for mesh_rot mode.

    In mesh_rot mode the SMPL body is decoded with global_orient = mesh_rot, which is
    already expressed in the cam0 frame, so only the avg_scale division remains. This
    mirrors training.loss.scale_joints_to_batch_gauge.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    scale = max(float(avg_scale), 1e-6)
    return vertices / scale


def predictions_use_mesh_rot(predictions: dict) -> bool:
    """True when the head predicts mesh_rot (cam0-frame root rotation in smpl_pose[:3])."""
    return predictions.get("mesh_rot", None) is not None


def _aa_to_rotmat_np(aa: np.ndarray) -> np.ndarray:
    """axis-angle (...,3) -> rotation matrix (...,3,3) via Rodrigues."""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.clip(theta, 1e-8, None)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = np.zeros_like(kx)
    K = np.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1).reshape(-1, 3, 3)
    th = theta.reshape(-1, 1, 1)
    eye = np.eye(3, dtype=np.float64)[None]
    R = eye + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
    return R


def _rotmat_to_aa_np(R: np.ndarray) -> np.ndarray:
    """rotation matrix (...,3,3) -> axis-angle (...,3), stable near 0 and pi."""
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos)
    out = np.zeros((R.shape[0], 3), dtype=np.float64)
    small = angle < 1e-6
    near_pi = angle > (np.pi - 1e-4)
    normal = ~small & ~near_pi
    if normal.any():
        idx = np.where(normal)[0]
        s = (2.0 * np.sin(angle[idx]))
        ax = np.stack([R[idx, 2, 1] - R[idx, 1, 2],
                       R[idx, 0, 2] - R[idx, 2, 0],
                       R[idx, 1, 0] - R[idx, 0, 1]], axis=-1) / s[:, None]
        out[idx] = ax * angle[idx][:, None]
    if near_pi.any():
        idx = np.where(near_pi)[0]
        A = (R[idx] + np.eye(3)[None]) * 0.5  # ~ outer(axis, axis)
        ax = np.sqrt(np.clip(np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]], axis=-1), 0.0, None))
        sgn = np.sign(np.stack([A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]], axis=-1))
        ax[:, 1] *= np.where(sgn[:, 0] != 0, np.sign(A[:, 0, 1]), 1.0)
        ax[:, 2] *= np.where(sgn[:, 1] != 0, np.sign(A[:, 0, 2]), 1.0)
        out[idx] = ax * angle[idx][:, None]
    return out


def compute_gt_mesh_rot_np(gt_pose: np.ndarray, raw_extrinsics: np.ndarray) -> np.ndarray:
    """GT mesh_rot (P,3) axis-angle: cam0-frame root = R0 @ R_global_orient_world.

    Mirrors training.loss.compute_gt_mesh_rot for the demo metric path.
    """
    gt_pose = np.asarray(gt_pose, dtype=np.float64).reshape(-1, 72)
    R0 = np.asarray(raw_extrinsics, dtype=np.float64)[0, :3, :3]
    R_go = _aa_to_rotmat_np(gt_pose[:, :3])           # (P,3,3)
    R_mesh = R0[None] @ R_go                           # (P,3,3)
    return _rotmat_to_aa_np(R_mesh)                    # (P,3)


def normalize_gt_cameras_joints_and_mesh(
    raw_extrinsics: np.ndarray,
    raw_joints3d: np.ndarray,
    raw_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    extrinsics_t = torch.as_tensor(np.asarray(raw_extrinsics, dtype=np.float32)).unsqueeze(0)
    joints_t = torch.as_tensor(np.asarray(raw_joints3d, dtype=np.float32)).unsqueeze(0)
    (
        norm_extrinsics_t,
        _,
        _,
        norm_joints_t,
        _,
        avg_scale_t,
    ) = normalize_camera_extrinsics_points_and_3djoints_batch(
        extrinsics=extrinsics_t,
        joints3d_world=joints_t,
        scale_by_extrinsics=True,
    )
    avg_scale = float(avg_scale_t.reshape(-1)[0].item())
    norm_vertices = normalize_vertices_to_camera0_gauge(raw_vertices, raw_extrinsics, avg_scale)

    return (
        norm_extrinsics_t.squeeze(0).detach().cpu().numpy().astype(np.float64),
        norm_joints_t.squeeze(0).detach().cpu().numpy().astype(np.float64),
        norm_vertices.astype(np.float64),
        avg_scale,
    )


def write_obj(verts: np.ndarray, faces: np.ndarray, out_obj_path: str) -> None:
    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    with open(out_obj_path, "w") as f:
        for v in np.asarray(verts):
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in np.asarray(faces, dtype=np.int64):
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_per_person_objs(
    person_vertices: list[np.ndarray],
    person_faces: list[np.ndarray],
    out_dir: str,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    obj_paths = []
    for person_idx, (verts, faces) in enumerate(zip(person_vertices, person_faces)):
        obj_path = os.path.join(out_dir, f"person_{person_idx:03d}.obj")
        write_obj(verts, faces, obj_path)
        obj_paths.append(obj_path)
    return obj_paths


ATTENTION_LAYER = "last"          # decoder cross-attn layer to visualize: int / "last" / "mean"
ATTENTION_ALPHA = 0.55            # heatmap blend weight over the base image
ATTENTION_MAX_PEOPLE = 8          # cap number of person overlays written


def write_attention_overlays(
    attn_cap,
    images: torch.Tensor,
    presence_logits,
    patch_size: int,
    out_dir: str,
    *,
    layer: str = ATTENTION_LAYER,
    presence_threshold: float = SMPL_PRESENCE_THRESHOLD,
    alpha: float = ATTENTION_ALPHA,
    max_people: int = ATTENTION_MAX_PEOPLE,
) -> list[str]:
    """Overlay the captured SMPL cross-attention onto each view and save per-person grids.

    Returns the list of written grid PNG paths (also leaves per-view PNGs in out_dir).
    """
    imgs = images if images.dim() == 4 else images[0]  # -> [S, 3, H, W]
    S, _, H, W = imgs.shape
    patch_h, patch_w = H // patch_size, W // patch_size

    attn = select_attention(attn_cap.captured, layer)          # [heads, people, S*P]
    heads, num_people, SP = attn.shape
    if SP != S * patch_h * patch_w:
        raise ValueError(f"attn S*P mismatch: {SP} vs {S * patch_h * patch_w}")
    attn = attn.mean(dim=0)                                     # [people, S*P]
    attn_grid = attn.reshape(num_people, S, patch_h, patch_w).numpy()

    if presence_logits is not None:
        probs = torch.sigmoid(torch.as_tensor(presence_logits).float()).reshape(-1).cpu().numpy()
    else:
        probs = _np.ones(num_people, dtype="float32")

    people = [i for i in range(num_people) if probs[i] >= presence_threshold]
    if not people:
        people = [int(_np.argmax(probs))]
    people = people[:max_people]

    base_imgs = [
        (imgs[s].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8") for s in range(S)
    ]

    os.makedirs(out_dir, exist_ok=True)
    # Clear stale overlays from a previous run so the gallery only shows this scene.
    for stale in glob.glob(os.path.join(out_dir, "*.png")):
        try:
            os.remove(stale)
        except OSError:
            pass

    # Persist the full attention grid (all query slots) so the Attention-vs-SMPL step
    # can index it by query index after predictions.npz / the mesh have been built.
    _np.save(os.path.join(out_dir, "attn_grid.npy"), attn_grid.astype("float32"))

    grid_paths = []
    for p in people:
        pmax = attn_grid[p].max()
        tiles = []
        for s in range(S):
            heat = attn_grid[p, s]
            if pmax > 0:
                heat = heat / pmax  # per-person normalization so views are comparable
            overlay = overlay_heatmap(base_imgs[s], heat, alpha)
            cv2.putText(overlay, f"view{s}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(overlay)
        grid = add_title_bar(make_grid(tiles), f"person {p:02d}  presence={probs[p]:.2f}  layer={layer}")
        grid_path = os.path.join(out_dir, f"person{p:02d}_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        grid_paths.append(grid_path)
    print(f"[Attention] wrote {len(grid_paths)} person overlay(s) to {out_dir}")
    return grid_paths


def run_attention_gallery(target_dir: str) -> list[str]:
    """Collect the per-person attention grid PNGs written by run_model()."""
    if not target_dir:
        return []
    attn_dir = os.path.join(str(target_dir), "attention")
    if not os.path.isdir(attn_dir):
        return []
    return sorted(glob.glob(os.path.join(attn_dir, "person*_grid.png")))


def _attention_peak_centroid(heat: np.ndarray) -> tuple[float, float]:
    """Weighted centroid (x, y) of a heatmap, in the heatmap's own pixel coords."""
    h, w = heat.shape
    total = heat.sum()
    if total <= 0:
        return w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    return float((xs * heat).sum() / total), float((ys * heat).sum() / total)


def run_attention_smpl_gallery(target_dir: str) -> list[str]:
    """Overlay each query's predicted SMPL projection on top of its attention heatmap.

    For every present query slot we draw, per view:
      - its attention heatmap (where the query looked),
      - the convex-hull outline of its reprojected SMPL mesh (where the query placed the
        body), in that person's palette color, reusing the stored gauge-space vertices +
        project_world_points_to_cam so it matches the projection gallery exactly,
      - centroid crosses for both, plus the attention-vs-mesh centroid pixel distance.
    Also writes attention_smpl_summary.txt. Returns the per-person grid PNG paths.
    """
    if not target_dir:
        return []
    attn_dir = os.path.join(str(target_dir), "attention")
    grid_npy = os.path.join(attn_dir, "attn_grid.npy")
    preds_path = os.path.join(str(target_dir), "predictions.npz")
    processed_dir = os.path.join(str(target_dir), "processed_images")
    if not (os.path.isfile(grid_npy) and os.path.isfile(preds_path) and os.path.isdir(processed_dir)):
        return []

    data = np.load(preds_path, allow_pickle=True)
    if "smpl_vertices" not in data.files or "extrinsic" not in data.files:
        print("[Attn-SMPL] no mesh in predictions.npz (camera-only / no confident slots); skipping.")
        return []

    attn_grid = np.load(grid_npy)                                   # [people, S, ph, pw]
    vertices = np.asarray(data["smpl_vertices"], dtype=np.float32)  # [P_people*V, 3], gauge space
    extrinsic = np.asarray(data["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
    visible_indices = np.asarray(data.get("smpl_visible_indices", np.arange(0))).reshape(-1).astype(int)
    presence_prob = np.asarray(data.get("smpl_presence_prob", np.zeros(attn_grid.shape[0]))).reshape(-1)

    num_mesh_people = int(visible_indices.shape[0])
    if num_mesh_people == 0 or vertices.shape[0] == 0:
        return []
    verts_per_person = vertices.shape[0] // num_mesh_people

    base_paths = sorted(
        p for p in glob.glob(os.path.join(processed_dir, "*"))
        if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg"}
    )
    S = min(len(base_paths), attn_grid.shape[1], extrinsic.shape[0], intrinsic.shape[0])
    base_bgr = [cv2.imread(base_paths[s], cv2.IMREAD_COLOR) for s in range(S)]

    summary_lines = ["# Attention vs predicted-SMPL agreement (attention peak vs mesh-projection centroid)"]
    grid_paths = []
    for j in range(num_mesh_people):
        p = int(visible_indices[j])
        if p >= attn_grid.shape[0]:
            continue
        _, rgba = get_smpl_mesh_color(j)
        color_bgr = (int(rgba[2]), int(rgba[1]), int(rgba[0]))

        verts_p = vertices[j * verts_per_person:(j + 1) * verts_per_person]
        verts_t = torch.from_numpy(verts_p).float()
        with torch.no_grad():
            img_pts, cam_pts = project_world_points_to_cam(
                verts_t, torch.from_numpy(extrinsic[:S]).float(), torch.from_numpy(intrinsic[:S]).float()
            )
        img_pts = img_pts.cpu().numpy()      # [S, V, 2]
        depth = cam_pts[:, 2, :].cpu().numpy()  # [S, V]

        pmax = attn_grid[p, :S].max()
        tiles, dists = [], []
        for s in range(S):
            base = base_bgr[s]
            if base is None:
                continue
            H, W = base.shape[:2]

            heat = attn_grid[p, s]
            if pmax > 0:
                heat = heat / pmax
            overlay = overlay_heatmap(cv2.cvtColor(base, cv2.COLOR_BGR2RGB), heat, ATTENTION_ALPHA)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # Reprojected mesh. Robust centroid = median of in-front vertices (always
            # meaningful). The convex-hull outline uses a tight IQR (25-75 pct) core and is
            # only drawn when the mesh actually projects as a compact body -- in the
            # predicted-camera gauge (no GT-camera calibration) absolute scale is unknown
            # and the mesh can fill the frame, so we suppress the misleading outline there.
            pts = img_pts[s]
            infront = depth[s] > 1e-3
            mesh_cx = mesh_cy = None
            if infront.sum() >= 3:
                fp = pts[infront]
                mesh_cx, mesh_cy = float(np.median(fp[:, 0])), float(np.median(fp[:, 1]))
                xlo, xhi = np.percentile(fp[:, 0], [25, 75])
                ylo, yhi = np.percentile(fp[:, 1], [25, 75])
                inlier = (fp[:, 0] >= xlo) & (fp[:, 0] <= xhi) & (fp[:, 1] >= ylo) & (fp[:, 1] <= yhi)
                ip = fp[inlier]
                compact = (xhi - xlo) < 0.5 * W and (yhi - ylo) < 0.6 * H
                if ip.shape[0] >= 3 and compact:
                    ip[:, 0] = np.clip(ip[:, 0], 0, W - 1)
                    ip[:, 1] = np.clip(ip[:, 1], 0, H - 1)
                    hull = cv2.convexHull(ip.astype(np.int32))
                    cv2.polylines(overlay, [hull], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Attention peak centroid (in heat grid coords -> scale to image).
            hx, hy = _attention_peak_centroid(heat)
            ph, pw = heat.shape
            ax, ay = hx / pw * W, hy / ph * H
            cv2.drawMarker(overlay, (int(ax), int(ay)), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

            dist_txt = "mesh:offscreen"
            if mesh_cx is not None:
                cv2.drawMarker(overlay, (int(mesh_cx), int(mesh_cy)), color_bgr, cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.line(overlay, (int(ax), int(ay)), (int(mesh_cx), int(mesh_cy)), (255, 255, 255), 1, cv2.LINE_AA)
                d = float(np.hypot(ax - mesh_cx, ay - mesh_cy))
                diag = float(np.hypot(H, W))
                dists.append(d)
                dist_txt = f"gap={d:.0f}px ({100 * d / diag:.1f}%)"

            cv2.putText(overlay, f"view{s} {dist_txt}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(overlay)

        if not tiles:
            continue
        mean_gap = float(np.mean(dists)) if dists else float("nan")
        tiles_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in tiles]
        title = f"person {p:02d}  presence={presence_prob[p] if p < len(presence_prob) else float('nan'):.2f}  mean gap={mean_gap:.0f}px"
        grid = add_title_bar(make_grid(tiles_rgb), title)
        grid_path = os.path.join(attn_dir, f"person{p:02d}_smpl_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        grid_paths.append(grid_path)
        summary_lines.append(
            f"person {p:02d}: presence={presence_prob[p] if p < len(presence_prob) else float('nan'):.3f} "
            f"mean_gap_px={mean_gap:.1f} per_view_px={[round(x, 1) for x in dists]}"
        )

    with open(os.path.join(attn_dir, "attention_smpl_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Attn-SMPL] wrote {len(grid_paths)} attention-vs-SMPL overlay(s) to {attn_dir}")
    return sorted(grid_paths)


def _as_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(v) for v in value]


def _landmark_points_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.size == 0:
        return pts.reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=-1)
    if finite.any():
        max_abs = float(np.nanmax(np.abs(pts[finite])))
        if max_abs <= 2.0:
            pts[:, 0] *= float(width)
            pts[:, 1] *= float(height)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
    return pts


def _select_people_for_landmark_mask(data, num_people: int, threshold: float = SMPL_PRESENCE_THRESHOLD) -> list[int]:
    visible = np.asarray(data.get("smpl_visible_indices", np.array([], dtype=np.int64))).reshape(-1)
    if visible.size > 0:
        people = [int(p) for p in visible if 0 <= int(p) < num_people]
        if people:
            return people

    probs = np.asarray(data.get("smpl_presence_prob", np.array([], dtype=np.float32))).reshape(-1)
    if probs.size >= num_people:
        people = [i for i in range(num_people) if float(probs[i]) >= threshold]
        if people:
            return people
        return [int(np.argmax(probs[:num_people]))]
    return list(range(num_people))


def _draw_landmark_overlay(base_bgr: np.ndarray, points: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.copy()
    H, W = out.shape[:2]
    pts = _landmark_points_to_pixels(points, W, H)
    for x, y in pts:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color_bgr, -1, cv2.LINE_AA)
    cv2.putText(out, f"{len(pts)} landmarks", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_landmark_points(
    image_bgr: np.ndarray,
    points: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    radius: int = 1,
    visibility: np.ndarray | None = None,
) -> int:
    H, W = image_bgr.shape[:2]
    pts = _landmark_points_to_pixels(points, W, H)
    vis = None
    if visibility is not None:
        vis = np.asarray(visibility).reshape(-1) > 0.5
    count = 0
    for i, (x, y) in enumerate(pts):
        if vis is not None and (i >= len(vis) or not vis[i]):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(image_bgr, (int(round(x)), int(round(y))), radius, color_bgr, -1, cv2.LINE_AA)
        count += 1
    return count


def _draw_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    H, W = base_bgr.shape[:2]
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -30.0, 30.0)))
    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_resized = np.clip(mask_resized, 0.0, 1.0)

    color = np.zeros_like(base_bgr, dtype=np.uint8)
    color[:, :] = color_bgr
    alpha = (0.62 * mask_resized)[..., None]
    out = (base_bgr.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)

    contour_mask = (mask_resized >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, 2, cv2.LINE_AA)
    cv2.putText(out, f"mask mean={float(mask_resized.mean()):.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _transform_points_like_vggt_preprocess(image_paths: list[str], points: np.ndarray) -> np.ndarray:
    """Map original image pixel coords through load_and_preprocess_images(mode='crop')."""
    target_size = 518
    points = np.asarray(points, dtype=np.float32).copy()
    if points.ndim != 4:
        raise ValueError(f"expected points [S,P,L,2], got {points.shape}")

    shapes: list[tuple[int, int]] = []
    for s, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        scale_x = new_width / float(width)
        scale_y = new_height / float(height)

        points[s, ..., 0] *= scale_x
        points[s, ..., 1] *= scale_y

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            points[s, ..., 1] -= float(start_y)
            out_h = target_size
        else:
            out_h = new_height
        out_w = new_width
        shapes.append((out_h, out_w))

    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    for s, (h, w) in enumerate(shapes):
        pad_top = (max_h - h) // 2
        pad_left = (max_w - w) // 2
        points[s, ..., 0] += float(pad_left)
        points[s, ..., 1] += float(pad_top)

    points[..., 0] /= float(max_w)
    points[..., 1] /= float(max_h)
    return points.astype(np.float32)


def _build_training_preprocess_dataset() -> BaseDataset:
    common = SimpleNamespace(
        img_size=int(OmegaConf.select(cfg, "img_size", default=518)),
        patch_size=int(OmegaConf.select(cfg, "patch_size", default=14)),
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
    )
    ds = BaseDataset(common)
    ds.training = False
    return ds


def _load_raw_mamma_camera(seq_dir: Path, image_path: str) -> tuple[np.ndarray, np.ndarray]:
    view_name, frame = _raw_mamma_view_frame_from_image_path(seq_dir, image_path)
    data_path = seq_dir / view_name / f"{frame}.data.pyd"
    frame_people = _load_pickle(data_path)
    if not isinstance(frame_people, dict) or not frame_people:
        raise ValueError(f"No people/camera records found in raw Mamma file: {data_path}")
    cam_person = next(iter(frame_people.values()))
    K = np.asarray(cam_person["cam_int"], dtype=np.float32).reshape(3, 3)
    E = np.asarray(cam_person["cam_ext"], dtype=np.float32)
    E = E[:3, :4] if E.shape == (4, 4) else E.reshape(3, 4)
    return E, K


def load_raw_mamma_images_with_training_preprocess(
    input_image_dir: str,
    image_paths: list[str],
    target_dir: str,
) -> tuple[torch.Tensor, list[str]]:
    """Preprocess raw MAMMA views exactly like SysSMPLMultiDataset training."""
    seq_dir = Path(str(input_image_dir or "").strip())
    if not _looks_like_raw_mamma_sequence(seq_dir):
        raise ValueError(f"Not a raw Mamma sequence directory: {input_image_dir}")
    if not image_paths:
        raise ValueError("No raw Mamma image paths provided.")

    ds = _build_training_preprocess_dataset()
    target_shape = ds.get_target_shape(aspect_ratio=1.0)
    processed = []
    for image_path in image_paths:
        image = read_image_cv2(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read raw Mamma image: {image_path}")
        depth_map = np.zeros(image.shape[:2], dtype=np.float32)
        extri_opencv, intri_opencv = _load_raw_mamma_camera(seq_dir, str(image_path))
        image, _, _, _, _, _, _, _, _ = ds.process_one_image(
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            np.array(image.shape[:2]),
            target_shape,
            track=None,
            filepath=str(image_path),
        )
        processed.append(torch.from_numpy(image).permute(2, 0, 1).float().div(255.0))

    images = torch.stack(processed, dim=0)
    processed_dir = os.path.join(target_dir, "processed_images")
    processed_image_paths = save_processed_images(image_paths, images, processed_dir)
    return images, processed_image_paths


def _transform_points_like_training_preprocess(
    image_paths: list[str],
    points: np.ndarray,
    input_image_dir: str,
) -> np.ndarray:
    """Map original raw MAMMA pixel coords through the training dataset preprocess."""
    seq_dir = Path(str(input_image_dir or "").strip())
    if not _looks_like_raw_mamma_sequence(seq_dir):
        raise ValueError(f"Not a raw Mamma sequence directory: {input_image_dir}")
    points = np.asarray(points, dtype=np.float32).copy()
    if points.ndim != 4:
        raise ValueError(f"expected points [S,P,L,2], got {points.shape}")

    ds = _build_training_preprocess_dataset()
    target_shape = ds.get_target_shape(aspect_ratio=1.0)
    transformed = []
    for s, image_path in enumerate(image_paths):
        image = read_image_cv2(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read raw Mamma image: {image_path}")
        depth_map = np.zeros(image.shape[:2], dtype=np.float32)
        extri_opencv, intri_opencv = _load_raw_mamma_camera(seq_dir, str(image_path))
        flat_points = points[s].reshape(-1, 2)
        image, _, _, _, _, _, _, track_new, confidence = ds.process_one_image(
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            np.array(image.shape[:2]),
            target_shape,
            track=flat_points,
            filepath=str(image_path),
        )
        H_final, W_final = image.shape[:2]
        lmk_px = track_new.reshape(points.shape[1], points.shape[2], 2)
        lmk_norm = np.empty_like(lmk_px)
        lmk_norm[..., 0] = lmk_px[..., 0] / float(W_final)
        lmk_norm[..., 1] = lmk_px[..., 1] / float(H_final)
        if confidence is not None:
            in_frame = confidence.reshape(points.shape[1], points.shape[2]) > 0.5
            lmk_norm[~in_frame] = -1.0
        transformed.append(lmk_norm)
    return np.stack(transformed, axis=0).astype(np.float32)


def load_raw_mamma_gt_landmarks_for_demo(input_image_dir: str, image_paths: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    seq_dir = Path(str(input_image_dir or "").strip())
    if not _looks_like_raw_mamma_sequence(seq_dir) or not image_paths:
        return None

    from training.data.landmark_mask_gt import (
        downsample_vertices,
        downsample_visibility,
        load_verts512_matrix,
    )

    verts512 = load_verts512_matrix(None)
    all_landmarks = []
    all_visibility = []
    for image_path in image_paths:
        view_name, frame = _raw_mamma_view_frame_from_image_path(seq_dir, image_path)
        data_path = seq_dir / view_name / f"{frame}.data.pyd"
        people = _load_pickle(data_path)
        if not isinstance(people, dict) or not people:
            raise ValueError(f"No people found in raw Mamma GT file: {data_path}")

        frame_landmarks = []
        frame_visibility = []
        for person_id in sorted(people.keys(), key=lambda item: int(item)):
            person = people[person_id]
            vertices2d = np.asarray(person["vertices2d"], dtype=np.float32)
            vertex_visibility = np.asarray(person["vertex_visibility"], dtype=np.float32).reshape(-1)
            frame_landmarks.append(downsample_vertices(verts512, vertices2d))
            frame_visibility.append(
                downsample_visibility(verts512, vertex_visibility[None, :], threshold=0.5)[0]
            )
        all_landmarks.append(np.stack(frame_landmarks, axis=0))
        all_visibility.append(np.stack(frame_visibility, axis=0))

    gt_px = np.stack(all_landmarks, axis=0).astype(np.float32)
    gt_vis = np.stack(all_visibility, axis=0).astype(np.float32)
    gt_norm = _transform_points_like_training_preprocess(image_paths, gt_px, input_image_dir)
    inframe = (
        (gt_norm[..., 0] >= 0.0)
        & (gt_norm[..., 0] <= 1.0)
        & (gt_norm[..., 1] >= 0.0)
        & (gt_norm[..., 1] <= 1.0)
    )
    gt_vis = gt_vis * inframe.astype(np.float32)
    return gt_norm, gt_vis


def run_landmark_mask_gallery(target_dir: str) -> list[str]:
    """Write per-view/person 512-landmark and mask overlays from predictions.npz."""
    if not target_dir:
        return []
    preds_path = os.path.join(str(target_dir), "predictions.npz")
    if not os.path.isfile(preds_path):
        return []

    data = np.load(preds_path, allow_pickle=True)
    has_landmarks = "smpl_landmarks2d" in data.files
    has_gt_landmarks = "gt_smpl_landmarks2d" in data.files
    has_masks = "person_mask_logits" in data.files
    if not (has_landmarks or has_gt_landmarks or has_masks):
        print("[Landmark/Mask] predictions.npz has no smpl_landmarks2d/person_mask_logits; skipping.")
        return []

    base_paths = _as_path_list(data.get("processed_image_paths", None))
    if not base_paths:
        processed_dir = os.path.join(str(target_dir), "processed_images")
        base_paths = sorted(
            p for p in glob.glob(os.path.join(processed_dir, "*"))
            if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg"}
        )
    if not base_paths:
        base_paths = _as_path_list(data.get("source_image_paths", None))
    if not base_paths:
        return []

    landmarks = np.asarray(data["smpl_landmarks2d"], dtype=np.float32) if has_landmarks else None
    gt_landmarks = np.asarray(data["gt_smpl_landmarks2d"], dtype=np.float32) if has_gt_landmarks else None
    gt_visibility = np.asarray(data["gt_smpl_landmarks2d_visibility"], dtype=np.float32) if "gt_smpl_landmarks2d_visibility" in data.files else None
    masks = np.asarray(data["person_mask_logits"], dtype=np.float32) if has_masks else None

    if landmarks is not None:
        if landmarks.ndim == 5 and landmarks.shape[0] == 1:
            landmarks = landmarks[0]
        if landmarks.ndim != 4:
            print(f"[Landmark/Mask] unexpected smpl_landmarks2d shape: {landmarks.shape}")
            landmarks = None
    if gt_landmarks is not None:
        if gt_landmarks.ndim == 5 and gt_landmarks.shape[0] == 1:
            gt_landmarks = gt_landmarks[0]
        if gt_landmarks.ndim != 4:
            print(f"[Landmark/Mask] unexpected gt_smpl_landmarks2d shape: {gt_landmarks.shape}")
            gt_landmarks = None
    if gt_visibility is not None:
        if gt_visibility.ndim == 4 and gt_visibility.shape[0] == 1:
            gt_visibility = gt_visibility[0]
        if gt_visibility.ndim != 3:
            print(f"[Landmark/Mask] unexpected gt_smpl_landmarks2d_visibility shape: {gt_visibility.shape}")
            gt_visibility = None
    if masks is not None:
        if masks.ndim == 5 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim != 4:
            print(f"[Landmark/Mask] unexpected person_mask_logits shape: {masks.shape}")
            masks = None

    if landmarks is None and gt_landmarks is None and masks is None:
        return []

    shapes_for_s = [arr.shape[0] for arr in (landmarks, gt_landmarks, masks) if arr is not None]
    S = min([len(base_paths)] + shapes_for_s)
    num_people = next(arr.shape[1] for arr in (landmarks, gt_landmarks, masks) if arr is not None)
    people = _select_people_for_landmark_mask(data, num_people)

    out_dir = os.path.join(str(target_dir), "landmark_mask")
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "*.png")):
        try:
            os.remove(stale)
        except OSError:
            pass

    written = []
    for p in people:
        _, rgba = get_smpl_mesh_color(p)
        color_bgr = (int(rgba[2]), int(rgba[1]), int(rgba[0]))
        for s in range(S):
            base = cv2.imread(base_paths[s], cv2.IMREAD_COLOR)
            if base is None:
                continue
            if landmarks is not None:
                overlay = _draw_landmark_overlay(base, landmarks[s, p], color_bgr)
                cv2.putText(overlay, f"view{s} person{p}", (8, base.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_bgr, 2, cv2.LINE_AA)
                path = os.path.join(out_dir, f"view{s:02d}_person{p:02d}_pred_landmarks512.png")
                cv2.imwrite(path, overlay)
                written.append(path)
            if gt_landmarks is not None and p < gt_landmarks.shape[1]:
                gt_overlay = base.copy()
                gt_count = _draw_landmark_points(
                    gt_overlay,
                    gt_landmarks[s, p],
                    (80, 255, 80),
                    radius=1,
                    visibility=gt_visibility[s, p] if gt_visibility is not None and p < gt_visibility.shape[1] else None,
                )
                cv2.putText(gt_overlay, f"GT {gt_count}/512 landmarks", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(gt_overlay, f"view{s} person{p}", (8, base.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 2, cv2.LINE_AA)
                path = os.path.join(out_dir, f"view{s:02d}_person{p:02d}_gt_landmarks512.png")
                cv2.imwrite(path, gt_overlay)
                written.append(path)
            if landmarks is not None and gt_landmarks is not None and p < gt_landmarks.shape[1]:
                both = base.copy()
                pred_count = _draw_landmark_points(both, landmarks[s, p], color_bgr, radius=1)
                gt_count = _draw_landmark_points(
                    both,
                    gt_landmarks[s, p],
                    (80, 255, 80),
                    radius=1,
                    visibility=gt_visibility[s, p] if gt_visibility is not None and p < gt_visibility.shape[1] else None,
                )
                cv2.putText(both, f"pred={pred_count}  gt-visible={gt_count}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(both, "pred=color  gt=green", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(both, f"view{s} person{p}", (8, base.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_bgr, 2, cv2.LINE_AA)
                path = os.path.join(out_dir, f"view{s:02d}_person{p:02d}_pred_vs_gt_landmarks512.png")
                cv2.imwrite(path, both)
                written.append(path)
            if masks is not None:
                overlay = _draw_mask_overlay(base, masks[s, p], color_bgr)
                cv2.putText(overlay, f"view{s} person{p}", (8, base.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_bgr, 2, cv2.LINE_AA)
                path = os.path.join(out_dir, f"view{s:02d}_person{p:02d}_mask.png")
                cv2.imwrite(path, overlay)
                written.append(path)

    print(f"[Landmark/Mask] wrote {len(written)} overlay(s) to {out_dir}")
    return sorted(written)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(
    target_dir,
    model,
    *,
    compute_world_points_from_depth: bool = True,
    camera_only: bool = False,
    disabled_heads: tuple[str, ...] = (),
    input_image_dir: str = "",
    raw_mamma_image_paths: list[str] | None = None,
) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images. Raw MAMMA scenes use the same
    # principal-point crop/resize path as SysSMPLMultiDataset training.
    raw_scene = Path(str(input_image_dir or "").strip())
    use_raw_training_preprocess = (
        bool(str(input_image_dir or "").strip())
        and _looks_like_raw_mamma_sequence(raw_scene)
        and bool(raw_mamma_image_paths)
    )
    if use_raw_training_preprocess:
        image_names = [str(p) for p in raw_mamma_image_paths]
    else:
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    if use_raw_training_preprocess:
        print("[Preprocess] raw MAMMA: using SysSMPLMultiDataset training crop/resize.")
        images, processed_image_paths = load_raw_mamma_images_with_training_preprocess(
            input_image_dir,
            image_names,
            target_dir,
        )
        images = images.to(device)
    else:
        images = load_and_preprocess_images(image_names).to(device)
        processed_dir = os.path.join(target_dir, "processed_images")
        processed_image_paths = save_processed_images(image_names, images, processed_dir)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    original_heads = {}
    if camera_only:
        disabled_heads = tuple(
            dict.fromkeys(disabled_heads + ("depth_head", "point_head", "smpl_head", "smpl_multi_head"))
        )

    for head_name in disabled_heads:
        if hasattr(model, head_name):
            original_heads[head_name] = getattr(model, head_name)
            setattr(model, head_name, None)

    # Capture the SMPL head cross-attention (person query -> image patches) during the
    # actual forward, so we can overlay it back onto the input views afterwards. Only
    # possible when the SMPL multi-query trans-rot head is present and not disabled.
    smpl_rot_head = getattr(model, "smpl_multi_query_trans_rot_head", None)
    capture_cm = (
        CrossAttnCapture(smpl_rot_head.decoder.transformer.transformer)
        if smpl_rot_head is not None
        else nullcontext()
    )
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                with capture_cm as attn_cap:
                    # predictions = model(images)
                    predictions = model(
                        images,
                        return_encoder_feature_map=True,
                        return_aggregator_feature_map=True,
                        return_camera_tokens=True,
                        return_depth_feature_map=False,
                    )
    finally:
        for head_name, head in original_heads.items():
            setattr(model, head_name, head)

    # Write per-person attention overlays before predictions get converted to numpy.
    if smpl_rot_head is not None and attn_cap is not None and all(c is not None for c in attn_cap.captured):
        try:
            write_attention_overlays(
                attn_cap,
                images,
                predictions.get("smpl_presence_logits"),
                model.aggregator.patch_size,
                os.path.join(target_dir, "attention"),
            )
        except Exception as exc:  # never let visualization break inference
            print(f"[Attention] skipped: {exc}")
    if "encoder_feature_l2_map" in predictions:
        print(
            "Encoder feature map ready:",
            predictions["encoder_feature_l2_map"].shape,
            predictions["encoder_feature_l2_map"].dtype,
        )
    if "aggregator_feature_l2_map" in predictions:
        print(
            "Aggregator feature map ready:",
            predictions["aggregator_feature_l2_map"].shape,
            predictions["aggregator_feature_l2_map"].dtype,
        )

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions["source_image_paths"] = np.asarray(image_names)
    predictions["processed_image_paths"] = np.asarray(processed_image_paths)
    # remove pose_enc_list (avoid saving None into npz)
    if "pose_enc_list" in predictions:
        predictions.pop("pose_enc_list")

    # Generate world points from depth map (optional)
    if compute_world_points_from_depth and ("depth" in predictions):
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
    else:
        # Keep the npz smaller and allow camera-only visualization.
        predictions.pop("depth", None)
        predictions.pop("depth_conf", None)
        predictions.pop("world_points_from_depth", None)

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def save_processed_images(image_paths: list[str], images: torch.Tensor, processed_dir: str) -> list[str]:
    os.makedirs(processed_dir, exist_ok=True)
    processed_image_paths = []
    images_cpu = images.detach().cpu().clamp(0, 1)
    for i, image_path in enumerate(image_paths):
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = (img * 255.0).round().astype(np.uint8)
        img_bgr = img[:, :, ::-1]
        out_path = os.path.join(processed_dir, f"{i:06}_{Path(image_path).stem}.png")
        cv2.imwrite(out_path, img_bgr)
        processed_image_paths.append(out_path)
    return processed_image_paths


def parse_image_ids(image_ids: str) -> list[int] | None:
    text = str(image_ids or "").strip()
    if not text:
        return None
    ids = []
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Image id must be an integer, got: {token}") from exc
        if value < 0:
            raise ValueError(f"Image id must be >= 0, got: {value}")
        ids.append(value)
    return ids or None


def camera_stem_from_id(camera_id: int) -> str:
    return "Main_Camera" if int(camera_id) == 0 else f"Main_Camera_({int(camera_id)})"


def resolve_camera_image_path(folder_path: Path, camera_id: int, all_images: list[str] | None = None) -> str:
    """Resolve an image id to a file path.

    Prefer the CLOTH3D 'Main_Camera(_(N))' naming; otherwise fall back to treating the id
    as an index into the sorted image list (e.g. MAMMA_eval_dance uses IOI_NN.jpg names).
    """
    stem = camera_stem_from_id(camera_id)
    for suffix in SUPPORTED_IMAGE_EXTS:
        candidate = folder_path / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate)
    if all_images is not None and 0 <= int(camera_id) < len(all_images):
        return all_images[int(camera_id)]
    raise FileNotFoundError(f"Image for id {camera_id} not found under {folder_path} ({stem}.jpg/.png)")


# Alias to the early run-sort helper defined alongside dataset auto-discovery.
dataset_run_sort_key = _run_sort_key


@lru_cache(maxsize=64)
def _list_dataset_runs_cached(root: str, split: str, run_root_mtime_ns: int) -> tuple[str, ...]:
    image_subdir = discover_image_subdir(root, split)
    run_root = Path(root) / str(split) / image_subdir
    runs = []
    with os.scandir(run_root) as entries:
        for entry in entries:
            if entry.name.startswith("runs_") and entry.is_dir(follow_symlinks=False):
                runs.append(entry.name)
    return tuple(sorted(
        runs,
        key=dataset_run_sort_key,
    ))


def list_dataset_runs(dataset_root: str, split: str) -> list[str]:
    root = str(dataset_root or DEFAULT_DATASET_ROOT).strip() or DEFAULT_DATASET_ROOT
    raw_sequences = _find_raw_mamma_sequences(root, split)
    if raw_sequences:
        return [_raw_mamma_run_name(root, split, seq_dir) for seq_dir in raw_sequences]

    image_subdir = discover_image_subdir(root, split)
    run_root = Path(root) / str(split) / image_subdir
    try:
        run_root_mtime_ns = run_root.stat().st_mtime_ns
    except FileNotFoundError:
        return []
    if not run_root.is_dir():
        return []
    return list(_list_dataset_runs_cached(root, str(split), run_root_mtime_ns))


def dataset_image_dir(dataset_root: str, split: str, run_name: str) -> str:
    root = str(dataset_root or DEFAULT_DATASET_ROOT).strip() or DEFAULT_DATASET_ROOT
    raw_base = _raw_mamma_search_base(root, split)
    raw_seq = raw_base / str(run_name)
    if _looks_like_raw_mamma_sequence(raw_seq):
        return str(raw_seq)

    image_subdir = discover_image_subdir(root, split)
    return str(Path(root) / str(split) / image_subdir / str(run_name))


def list_scene_frames(input_image_dir: str) -> list[str]:
    """Frame (time-step) stems available in a raw Mamma sequence, e.g. ['0000','0011',...].

    Returns [] for non-raw datasets (eval scenes are a single time-step of multiple cameras).
    """
    folder = Path(str(input_image_dir or "").strip())
    if not folder.is_dir() or not _looks_like_raw_mamma_sequence(folder):
        return []
    view_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if not view_dirs:
        return []
    stems = {
        p.stem for p in view_dirs[0].iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS and ".mask" not in p.name
    }
    return sorted(stems)


def list_images_from_folder(input_image_dir: str, image_ids: str = "", frame: str | None = None) -> list[str]:
    folder = str(input_image_dir or "").strip()
    if not folder:
        return []
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    if _looks_like_raw_mamma_sequence(folder_path):
        view_dirs = sorted(p for p in folder_path.iterdir() if p.is_dir())
        selected_ids = parse_image_ids(image_ids)
        if selected_ids is None:
            selected_views = view_dirs
        else:
            selected_views = []
            for view_idx in selected_ids:
                if view_idx >= len(view_dirs):
                    raise FileNotFoundError(
                        f"View index {view_idx} not found under {folder_path}; only {len(view_dirs)} views available."
                    )
                selected_views.append(view_dirs[view_idx])
        frame = str(frame).strip() if frame not in (None, "") else None
        image_paths = []
        for view_dir in selected_views:
            frames = sorted(
                p for p in view_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS and ".mask" not in p.name
            )
            if not frames:
                continue
            chosen = None
            if frame is not None:
                chosen = next((p for p in frames if p.stem == frame), None)
            image_paths.append(str(chosen if chosen is not None else frames[0]))
        if not image_paths:
            raise ValueError(f"No images found in raw Mamma sequence: {folder}")
        return image_paths

    all_images = sorted(
        str(p) for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    selected_ids = parse_image_ids(image_ids)
    if selected_ids is None:
        image_paths = all_images
    else:
        image_paths = [
            resolve_camera_image_path(folder_path, camera_id, all_images) for camera_id in selected_ids
        ]
    if not image_paths:
        raise ValueError(f"No images found in folder: {folder}")
    return image_paths


def infer_run_npz_path(input_image_dir: str) -> Path | None:
    """Map an image folder (.../<split>/out_image/CLOTH3D/runs_X) to its merged GT
    archive (.../<split>/out_data/CLOTH3D/runs_X.npz).

    The dataset stores all GT for a run in a single npz:
        out_param/<subject>/<frame>/smpl_params/{poses,betas,trans,gender}  (per person)
        cam_param_min/<CAM>/{intrinsics.K_flat9, extrinsics.worldToCamera12}  (per camera)
        reprojection_data/<CAM>/people                                        (per camera joints)
    """
    folder = Path(str(input_image_dir or "").strip()).resolve()
    parts = list(folder.parts)
    if "out_image" not in parts:
        return None
    out_image_idx = parts.index("out_image")
    prefix = Path(*parts[:out_image_idx])
    suffix = parts[out_image_idx + 1 :]  # e.g. ["CLOTH3D", "runs_X"]
    if not suffix:
        return None
    run_name = suffix[-1]
    rel = Path(*suffix[:-1]) if len(suffix) > 1 else Path()
    return prefix / "out_data" / rel / f"{run_name}.npz"


def has_eval_gt_archive(input_image_dir: str) -> bool:
    npz_path = infer_run_npz_path(input_image_dir)
    return bool(npz_path is not None and npz_path.is_file())


def has_gt_camera_metadata(input_image_dir: str) -> bool:
    folder = Path(str(input_image_dir or "").strip())
    return has_eval_gt_archive(input_image_dir) or _looks_like_raw_mamma_sequence(folder)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _raw_mamma_view_frame_from_image_path(seq_dir: Path, image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    if path.parent.parent.resolve() == seq_dir.resolve():
        return path.parent.name, path.stem

    stem = path.stem
    match = re.fullmatch(r"\d+_(.+)_(\d+)", stem)
    if match:
        return match.group(1), match.group(2)

    view_names = sorted((p.name for p in seq_dir.iterdir() if p.is_dir()), key=len, reverse=True)
    for view_name in view_names:
        marker = f"_{view_name}_"
        if marker in stem:
            return view_name, stem.rsplit(marker, 1)[1]
    raise ValueError(f"Cannot infer raw Mamma view/frame from image path: {image_path}")


def load_raw_mamma_gt_from_sequence(input_image_dir: str, image_paths: list[str]) -> dict:
    seq_dir = Path(str(input_image_dir or "").strip())
    if not _looks_like_raw_mamma_sequence(seq_dir):
        raise ValueError(f"Not a raw Mamma sequence directory: {input_image_dir}")
    if not image_paths:
        raise ValueError("No image paths provided for raw Mamma GT loading.")

    view_frames = [_raw_mamma_view_frame_from_image_path(seq_dir, image_path) for image_path in image_paths]
    first_view, first_frame = view_frames[0]
    first_data_path = seq_dir / first_view / f"{first_frame}.data.pyd"
    people = _load_pickle(first_data_path)
    if not isinstance(people, dict) or not people:
        raise ValueError(f"No people found in raw Mamma GT file: {first_data_path}")

    poses, betas, trans, genders, person_keys = [], [], [], [], []
    for person_id in sorted(people.keys(), key=lambda item: int(item)):
        person = people[person_id]
        poses.append(np.asarray(person["pose_world"], dtype=np.float64).reshape(-1)[:72])
        betas.append(np.asarray(person["shape"], dtype=np.float64).reshape(-1)[:10])
        trans.append(np.asarray(person["trans_world"], dtype=np.float64).reshape(-1)[:3])
        genders.append(normalize_gender(person.get("gender", "neutral")))
        person_keys.append(f"person_{int(person_id):02d}")

    intrinsics, extrinsics = [], []
    for view_name, frame in view_frames:
        data_path = seq_dir / view_name / f"{frame}.data.pyd"
        frame_people = _load_pickle(data_path)
        if not isinstance(frame_people, dict) or not frame_people:
            raise ValueError(f"No people found in raw Mamma camera file: {data_path}")
        cam_person = next(iter(frame_people.values()))
        K = np.asarray(cam_person["cam_int"], dtype=np.float64).reshape(3, 3)
        E = np.asarray(cam_person["cam_ext"], dtype=np.float64)
        E = E[:3, :4] if E.shape == (4, 4) else E.reshape(3, 4)
        intrinsics.append(K)
        extrinsics.append(E)

    return {
        "param_dir": str(first_data_path),
        "cam_dir": str(seq_dir),
        "smpl_files": np.asarray(person_keys),
        "person_keys": person_keys,
        "smpl_pose": np.stack(poses, axis=0),
        "smpl_beta": np.stack(betas, axis=0),
        "smpl_trans": np.stack(trans, axis=0),
        "genders": genders,
        "intrinsic_raw": np.stack(intrinsics, axis=0),
        "extrinsic": orthonormalize_extrinsics(np.stack(extrinsics, axis=0)),
    }


def load_gt_multi_from_image_dir(input_image_dir: str, image_paths: list[str]) -> dict:
    if _looks_like_raw_mamma_sequence(Path(str(input_image_dir or "").strip())):
        return load_raw_mamma_gt_from_sequence(input_image_dir, image_paths)

    npz_path = infer_run_npz_path(input_image_dir)
    if npz_path is None:
        raise ValueError("Cannot infer GT archive. Input folder path must contain an out_image component.")
    if not npz_path.is_file():
        raise FileNotFoundError(f"GT archive not found: {npz_path}")

    archive = np.load(npz_path, allow_pickle=True)
    keys = set(archive.files)

    # --- per-person SMPL params: out_param/<subject>/<frame>/smpl_params/<field> ---
    poses, betas, trans, genders, person_keys = [], [], [], [], []
    for key in sorted(keys):
        if not (key.startswith("out_param/") and key.endswith("/smpl_params/poses")):
            continue
        person_key = key[len("out_param/") : -len("/smpl_params/poses")]
        prefix = f"out_param/{person_key}/smpl_params/"
        poses.append(np.asarray(archive[prefix + "poses"], dtype=np.float64).reshape(-1)[:72])
        beta = np.asarray(archive[prefix + "betas"], dtype=np.float64)
        betas.append(beta.reshape(-1)[:10])
        if prefix + "trans" in keys:
            trans.append(np.asarray(archive[prefix + "trans"], dtype=np.float64).reshape(-1)[:3])
        else:
            trans.append(np.zeros(3, dtype=np.float64))
        gender = archive[prefix + "gender"] if prefix + "gender" in keys else "neutral"
        genders.append(normalize_gender(gender))
        person_keys.append(person_key)

    if not poses:
        raise FileNotFoundError(f"No out_param/*/smpl_params found in GT archive: {npz_path}")

    # --- per-image cameras: cam_param_min/<view>/{intrinsics.K_flat9, extrinsics.worldToCamera12} ---
    # The archive keys cameras by the bare view id (e.g. "IOI_01"), but images copied into
    # target_dir/images are renamed to "<frame>_<run>_<view>" (e.g. "0000_runs_00000_IOI_01"),
    # so match the image stem's trailing view id back to an available camera key.
    cam_views = sorted({k.split("/", 2)[1] for k in keys if k.startswith("cam_param_min/")})

    def _resolve_cam_view(stem: str) -> str | None:
        if stem in cam_views:
            return stem
        cands = [v for v in cam_views if stem.endswith("_" + v)]
        return max(cands, key=len) if cands else None

    intrinsics, extrinsics = [], []
    for image_path in image_paths:
        stem = Path(image_path).stem
        view = _resolve_cam_view(stem)
        if view is None:
            raise FileNotFoundError(
                f"GT camera params for '{stem}' not found in {npz_path} "
                f"(available views: {cam_views[:8]}{'...' if len(cam_views) > 8 else ''}). "
                "This run may not include camera annotations (cam_param_min)."
            )
        cam_prefix = f"cam_param_min/{view}/"
        k_key = cam_prefix + "intrinsics.K_flat9"
        e_key = cam_prefix + "extrinsics.worldToCamera12"
        if k_key not in keys or e_key not in keys:
            raise FileNotFoundError(
                f"GT camera params for view '{view}' (from image '{stem}') incomplete in {npz_path}."
            )
        K = np.asarray(archive[k_key], dtype=np.float64).reshape(3, 3)
        E = np.asarray(archive[e_key], dtype=np.float64)
        if E.shape == (4, 4):
            E = E[:3]
        else:
            E = E.reshape(3, 4)
        intrinsics.append(K)
        extrinsics.append(E)

    return {
        "param_dir": str(npz_path),
        "cam_dir": str(npz_path),
        "smpl_files": np.asarray(person_keys),
        "person_keys": person_keys,
        "smpl_pose": np.stack(poses, axis=0),
        "smpl_beta": np.stack(betas, axis=0),
        "smpl_trans": np.stack(trans, axis=0),
        "genders": genders,
        "intrinsic_raw": np.stack(intrinsics, axis=0),
        # Strip any baked-in uniform scale so the SE(3) camera0-gauge normalization is valid
        # (no-op for clean extrinsics; fixes datasets like Harmony4D with det(R) != 1).
        "extrinsic": orthonormalize_extrinsics(np.stack(extrinsics, axis=0)),
    }


def normalize_mesh_vertices_from_gt_cameras(
    input_image_dir: str,
    image_paths: list[str],
    vertices_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    gt = load_gt_multi_from_image_dir(input_image_dir, image_paths)
    raw_extrinsics = gt["extrinsic"]
    extrinsics_t = torch.as_tensor(np.asarray(raw_extrinsics, dtype=np.float32)).unsqueeze(0)
    norm_extrinsics_t, _, _, _, _, avg_scale_t = normalize_camera_extrinsics_points_and_3djoints_batch(
        extrinsics=extrinsics_t,
        scale_by_extrinsics=True,
    )
    avg_scale = float(avg_scale_t.reshape(-1)[0].item())
    norm_vertices = normalize_vertices_to_camera0_gauge(vertices_world, raw_extrinsics, avg_scale)
    return (
        norm_vertices.astype(np.float64),
        norm_extrinsics_t.squeeze(0).detach().cpu().numpy().astype(np.float64),
        avg_scale,
        gt,
    )


def adjust_intrinsics_for_preprocess(
    image_paths: list[str],
    intrinsics: np.ndarray,
    input_image_dir: str = "",
    mode: str = "crop",
    target_size: int = 518,
    align: int = 14,
) -> np.ndarray:
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None, ...], len(image_paths), axis=0)

    seq_dir = Path(str(input_image_dir or "").strip())
    if _looks_like_raw_mamma_sequence(seq_dir):
        ds = _build_training_preprocess_dataset()
        target_shape = ds.get_target_shape(aspect_ratio=1.0)
        adjusted_intrinsics = []
        for image_path, K in zip(image_paths, intrinsics):
            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read raw Mamma image: {image_path}")
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            extri_opencv, _ = _load_raw_mamma_camera(seq_dir, str(image_path))
            _, _, _, K_adj, _, _, _, _, _ = ds.process_one_image(
                image,
                depth_map,
                extri_opencv,
                np.asarray(K, dtype=np.float32),
                np.array(image.shape[:2]),
                target_shape,
                track=None,
                filepath=str(image_path),
            )
            adjusted_intrinsics.append(np.asarray(K_adj, dtype=np.float64))
        return np.stack(adjusted_intrinsics, axis=0)

    processed_sizes = []
    adjusted_intrinsics = []
    for image_path, K in zip(image_paths, intrinsics):
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if mode != "crop":
            raise NotImplementedError("Only crop mode implemented for intrinsics adjustment.")

        new_w = target_size
        new_h = round(h * (new_w / w) / align) * align
        scale_x = new_w / w
        scale_y = new_h / h

        K_adj = K.copy()
        K_adj[0, 0] *= scale_x
        K_adj[1, 1] *= scale_y
        K_adj[0, 2] *= scale_x
        K_adj[1, 2] *= scale_y

        final_h = new_h
        final_w = new_w
        if new_h > target_size:
            crop_y = (new_h - target_size) // 2
            K_adj[1, 2] -= crop_y
            final_h = target_size

        processed_sizes.append((final_h, final_w))
        adjusted_intrinsics.append(K_adj)

    max_h = max(h for h, _ in processed_sizes)
    max_w = max(w for _, w in processed_sizes)
    if any((h != max_h or w != max_w) for h, w in processed_sizes):
        padded_intrinsics = []
        for (h, w), K_adj in zip(processed_sizes, adjusted_intrinsics):
            pad_top = (max_h - h) // 2
            pad_left = (max_w - w) // 2
            K_pad = K_adj.copy()
            K_pad[0, 2] += pad_left
            K_pad[1, 2] += pad_top
            padded_intrinsics.append(K_pad)
        adjusted_intrinsics = padded_intrinsics

    return np.stack(adjusted_intrinsics, axis=0)


def project_points_world(
    points_world: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    points_t = torch.from_numpy(np.asarray(points_world, dtype=np.float32)).to(device=device)
    extr_t = torch.from_numpy(np.asarray(extrinsics, dtype=np.float32)).to(device=device)
    intr_t = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).to(device=device)
    with torch.no_grad():
        img_points, _ = project_world_points_to_cam(points_t, extr_t, intr_t)
    return img_points.detach().cpu().numpy().astype(np.float64)


def mean_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64))))


def mean_rotation_angle_from_matrices(pred_rot: np.ndarray, gt_rot: np.ndarray, degrees: bool = True) -> float:
    pred_rot = np.asarray(pred_rot, dtype=np.float64)
    gt_rot = np.asarray(gt_rot, dtype=np.float64)
    if pred_rot.shape != gt_rot.shape or pred_rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matching rotation matrix shapes (..., 3, 3), got {pred_rot.shape} and {gt_rot.shape}")

    rel_rot = pred_rot @ np.swapaxes(gt_rot, -1, -2)
    trace = np.trace(rel_rot, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    if degrees:
        angles = np.degrees(angles)
    return float(np.mean(np.abs(angles)))


def mean_pose_geodesic_deg(pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    pred_t = torch.from_numpy(np.asarray(pred_pose, dtype=np.float32).reshape(-1, 72))
    gt_t = torch.from_numpy(np.asarray(gt_pose, dtype=np.float32).reshape(-1, 72))
    with torch.no_grad():
        pred_rot = axis_angle_to_rotmat(pred_t)
        gt_rot = axis_angle_to_rotmat(gt_t)
        rel = pred_rot.transpose(-1, -2) @ gt_rot
        trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(torch.rad2deg(theta).mean().item())


def binary_cross_entropy_with_logits_np(logit: np.ndarray | float, target: float = 1.0) -> np.ndarray:
    logit = np.asarray(logit, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    return np.maximum(logit, 0.0) - logit * target_arr + np.log1p(np.exp(-np.abs(logit)))


def get_presence_costs_for_matching(predictions: dict, people_count: int) -> np.ndarray:
    presence_logits = predictions.get("smpl_presence_logits", None)
    if presence_logits is None:
        return np.zeros((int(people_count),), dtype=np.float64)
    logits = np.asarray(presence_logits, dtype=np.float64).reshape(-1)
    costs = np.zeros((int(people_count),), dtype=np.float64)
    valid_count = min(costs.shape[0], logits.shape[0])
    if valid_count > 0:
        costs[:valid_count] = binary_cross_entropy_with_logits_np(logits[:valid_count], target=1.0)
    return costs


def compute_matching_loss_value(
    pred_pose,
    gt_pose,
    pred_beta,
    gt_beta,
    pred_trans,
    gt_trans,
    weights: dict[str, float],
    presence_cost: float = 0.0,
) -> float:
    return float(
        weights["pose"] * mean_l1(pred_pose, gt_pose)
        + weights["beta"] * mean_l1(pred_beta, gt_beta)
        + weights["trans"] * mean_l1(pred_trans, gt_trans)
        + weights["presence"] * float(presence_cost)
    )


def has_smpl_param_outputs(predictions: dict) -> bool:
    """True when predictions carry enough SMPL params to compute errors / build meshes.

    Supports both the `smpl_trans` head and the `mesh_translate` head: the latter
    predicts the SMPL root position directly in the first-camera normalized frame.
    """
    if not {"extrinsic", "intrinsic", "smpl_pose", "smpl_beta"}.issubset(predictions.keys()):
        return False
    return ("smpl_trans" in predictions) or ("mesh_translate" in predictions)


def predictions_use_mesh_translate(predictions: dict) -> bool:
    """Decide whether to place/score the mesh via `mesh_translate` rather than `smpl_trans`.

    Use `mesh_translate` when it is available AND it is the active placement source:
    either the trans head is enabled (covers the both-heads-on case, where the model
    emits both `smpl_trans` and `mesh_translate` and we prefer `mesh_translate`), or no
    `smpl_trans` was produced at all.
    """
    if predictions.get("mesh_translate", None) is None:
        return False
    return ENABLE_SMPL_MULTI_QUERY_TRANS or predictions.get("smpl_trans", None) is None


def compute_gt_mesh_translate_np(
    gt_pose: np.ndarray,
    gt_beta: np.ndarray,
    gt_trans: np.ndarray,
    genders: list[str],
    raw_extrinsics: np.ndarray,
    avg_scale: float,
) -> np.ndarray:
    """Mirror training.loss.compute_gt_mesh_translate in numpy.

    GT mesh_translate = normalized( zero_trans_root_world + gt_trans ), i.e. the SMPL
    root joint position in the first-camera normalized coordinate frame.
    """
    gt_pose = ensure_people_array(gt_pose, 72, "gt smpl_pose")
    gt_beta = ensure_people_array(gt_beta, 10, "gt smpl_beta")
    gt_trans = ensure_people_array(gt_trans, 3, "gt smpl_trans")
    people_count = gt_pose.shape[0]
    zero_roots = []
    for person_idx in range(people_count):
        gender = genders[person_idx] if person_idx < len(genders) else "neutral"
        _, _, joints = smpl_forward_with_joints(gt_pose[person_idx], gt_beta[person_idx], None, gender)
        zero_roots.append(joints[0])
    zero_root_world = np.stack(zero_roots, axis=0)
    target_root_world = zero_root_world + gt_trans
    return normalize_vertices_to_camera0_gauge(target_root_world, raw_extrinsics, avg_scale).reshape(people_count, 3)


def place_pred_smpl_with_mesh_translate(
    visible_pose: np.ndarray,
    visible_beta: np.ndarray,
    mesh_translate_visible: np.ndarray,
    raw_extrinsics: np.ndarray,
    avg_scale: float,
    out_raw_obj_path: str,
    use_mesh_rot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode visible SMPL people (trans=0), gauge-normalize them, then re-anchor each
    person's root joint to the predicted mesh_translate (first-camera normalized frame).

    This reproduces the forward placement used in training.loss.compute_smpl_loss for the
    mesh_translate head, so the mesh lands in the same normalized scene as the cameras.

    Returns (placed_vertices, faces, raw_trans0_vertices).
    """
    verts_raw, faces, joints_raw = smpl_to_mesh_joints_and_obj(
        visible_pose,
        visible_beta,
        None,  # trans=0: mesh_translate re-anchors the root afterwards
        out_raw_obj_path,
    )
    people_count = int(joints_raw.shape[0])
    verts_per_person = verts_raw.shape[0] // people_count
    # mesh_rot mode: smpl_pose[:3] is the cam0-frame mesh_rot, so the decoded body is
    # already oriented in cam0 -> scale-only gauge (no R0 rotation). Otherwise rotate
    # world->cam0 then scale.
    if use_mesh_rot:
        verts_norm = scale_vertices_to_camera0_gauge(verts_raw, avg_scale).reshape(
            people_count, verts_per_person, 3
        )
        root_norm = scale_vertices_to_camera0_gauge(joints_raw[:, 0, :], avg_scale).reshape(
            people_count, 3
        )
    else:
        verts_norm = normalize_vertices_to_camera0_gauge(verts_raw, raw_extrinsics, avg_scale).reshape(
            people_count, verts_per_person, 3
        )
        root_norm = normalize_vertices_to_camera0_gauge(joints_raw[:, 0, :], raw_extrinsics, avg_scale).reshape(
            people_count, 3
        )
    mesh_translate_visible = np.asarray(mesh_translate_visible, dtype=np.float64).reshape(people_count, 3)
    offset = mesh_translate_visible - root_norm
    verts_placed = (verts_norm + offset[:, None, :]).reshape(-1, 3)
    return verts_placed, faces, verts_raw


def compute_multi_prediction_errors(
    predictions: dict,
    image_paths: list[str],
    input_image_dir: str,
) -> tuple[dict, dict]:
    if not str(input_image_dir or "").strip():
        return {}, {}

    gt = load_gt_multi_from_image_dir(input_image_dir, image_paths)
    gt_intrinsics = adjust_intrinsics_for_preprocess(image_paths, gt["intrinsic_raw"], input_image_dir)
    gt_extrinsics_raw = gt["extrinsic"]
    if "gt_extrinsic_normalized" in predictions and "avg_scale" in predictions:
        gt_extrinsics = np.asarray(predictions["gt_extrinsic_normalized"], dtype=np.float64)
        avg_scale = float(np.asarray(predictions["avg_scale"], dtype=np.float64).reshape(-1)[0])
    else:
        gt_extrinsics_t = torch.as_tensor(np.asarray(gt_extrinsics_raw, dtype=np.float32)).unsqueeze(0)
        norm_gt_extrinsics_t, _, _, _, _, avg_scale_t = normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=gt_extrinsics_t,
            scale_by_extrinsics=True,
        )
        gt_extrinsics = norm_gt_extrinsics_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        avg_scale = float(avg_scale_t.reshape(-1)[0].item())

    use_mesh_translate = predictions_use_mesh_translate(predictions)
    use_mesh_rot = predictions_use_mesh_rot(predictions)
    pred_pose = ensure_people_array(predictions["smpl_pose"], 72, "pred smpl_pose")
    pred_beta = ensure_people_array(predictions["smpl_beta"], 10, "pred smpl_beta")
    gt_pose = ensure_people_array(gt["smpl_pose"], 72, "gt smpl_pose")
    gt_beta = ensure_people_array(gt["smpl_beta"], 10, "gt smpl_beta")
    gt_trans = ensure_people_array(gt["smpl_trans"], 3, "gt smpl_trans")
    # mesh_rot mode: pred smpl_pose[:3] is the cam0-frame mesh_rot. Build a GT pose whose
    # root is the GT mesh_rot (R0 @ R_global_orient_world) so Hungarian matching and the
    # SMPL pose-error geodesic compare like-for-like. gt_pose itself stays world-frame for
    # the SMPL joint decode below (which is then rotated world->cam0 via the R0 gauge).
    gt_pose_cmp = gt_pose
    if use_mesh_rot:
        gt_pose_cmp = np.array(gt_pose, dtype=np.float64, copy=True)
        gt_pose_cmp[:, :3] = compute_gt_mesh_rot_np(gt_pose, gt_extrinsics_raw)
    if use_mesh_translate:
        # mesh_translate is already the SMPL root in the first-camera normalized frame,
        # so it is comparable directly to the normalized GT root (no extra gauge transform).
        pred_mesh_translate = ensure_people_array(predictions["mesh_translate"], 3, "pred mesh_translate")
        gt_mesh_translate = compute_gt_mesh_translate_np(
            gt_pose, gt_beta, gt_trans, gt["genders"], gt_extrinsics_raw, avg_scale
        )
        pred_trans = pred_mesh_translate
        pred_trans_for_matching = pred_mesh_translate
        gt_trans_for_matching = gt_mesh_translate
    else:
        pred_trans = ensure_people_array(predictions["smpl_trans"], 3, "pred smpl_trans")
        pred_mesh_translate = None
        gt_mesh_translate = None
        pred_trans_for_matching = normalize_vertices_to_camera0_gauge(pred_trans, gt_extrinsics_raw, avg_scale)
        gt_trans_for_matching = normalize_vertices_to_camera0_gauge(gt_trans, gt_extrinsics_raw, avg_scale)
    matching_weights = dict(SMPL_HUNGARIAN_COST_WEIGHTS)
    if use_mesh_translate:
        # The mesh_translate head uses a dedicated Hungarian cost weight for the root term.
        matching_weights["trans"] = SMPL_HUNGARIAN_COST_WEIGHTS.get(
            "mesh_trans", matching_weights["trans"]
        )
    pred_presence_cost = get_presence_costs_for_matching(predictions, pred_pose.shape[0])

    cost = np.zeros((pred_pose.shape[0], gt_pose.shape[0]), dtype=np.float64)
    for i in range(pred_pose.shape[0]):
        for j in range(gt_pose.shape[0]):
            cost[i, j] = (
                matching_weights["pose"] * mean_l1(pred_pose[i], gt_pose_cmp[j])
                + matching_weights["beta"] * mean_l1(pred_beta[i], gt_beta[j])
                + matching_weights["trans"] * mean_l1(pred_trans_for_matching[i], gt_trans_for_matching[j])
                + matching_weights["presence"] * pred_presence_cost[i]
            )
    row_ind, col_ind = linear_sum_assignment(cost)
    match_count = min(len(row_ind), len(col_ind))
    row_ind = row_ind[:match_count]
    col_ind = col_ind[:match_count]

    pred_joints3d, gt_joints3d = [], []
    pred_joints2d, gt_joints2d = [], []
    pred_extrinsics = np.asarray(predictions["extrinsic"], dtype=np.float64)
    pred_intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float64)
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        gender = gt["genders"][gt_idx] if gt_idx < len(gt["genders"]) else "neutral"
        _, _, pred_joints = smpl_forward_with_joints(
            pred_pose[pred_idx],
            pred_beta[pred_idx],
            None if use_mesh_translate else pred_trans[pred_idx],
            gender,
        )
        _, _, gt_joints = smpl_forward_with_joints(
            gt_pose[gt_idx],
            gt_beta[gt_idx],
            gt_trans[gt_idx],
            gender,
        )
        # mesh_rot mode: pred body is decoded with cam0-frame mesh_rot -> scale-only gauge
        # (no R0). GT body is still world-frame -> rotate world->cam0 via R0.
        if use_mesh_rot:
            pred_joints = scale_vertices_to_camera0_gauge(pred_joints, avg_scale)
        else:
            pred_joints = normalize_vertices_to_camera0_gauge(pred_joints, gt_extrinsics_raw, avg_scale)
        gt_joints = normalize_vertices_to_camera0_gauge(gt_joints, gt_extrinsics_raw, avg_scale)
        if use_mesh_translate:
            # Re-anchor the gauge-normalized root to the predicted mesh_translate.
            pred_joints = pred_joints - pred_joints[0:1, :] + pred_mesh_translate[pred_idx].reshape(1, 3)
        pred_joints3d.append(pred_joints)
        gt_joints3d.append(gt_joints)
        pred_joints2d.append(project_points_world(pred_joints, pred_extrinsics, pred_intrinsics))
        gt_joints2d.append(project_points_world(gt_joints, gt_extrinsics, gt_intrinsics))

    if match_count == 0:
        return {}, gt

    pred_joints3d = np.stack(pred_joints3d, axis=0)
    gt_joints3d = np.stack(gt_joints3d, axis=0)
    pred_joints2d = np.stack(pred_joints2d, axis=0)
    gt_joints2d = np.stack(gt_joints2d, axis=0)

    matched_pred_pose = pred_pose[row_ind]
    matched_gt_pose = gt_pose_cmp[col_ind]
    matched_pred_beta = pred_beta[row_ind]
    matched_gt_beta = gt_beta[col_ind]
    matched_pred_trans = pred_trans[row_ind]
    matched_gt_trans = gt_trans[col_ind]
    if use_mesh_translate:
        # Both already live in the first-camera normalized frame.
        matched_pred_trans_for_error = pred_mesh_translate[row_ind]
        matched_gt_trans_for_error = gt_mesh_translate[col_ind]
    else:
        matched_pred_trans_for_error = normalize_vertices_to_camera0_gauge(
            matched_pred_trans,
            gt_extrinsics_raw,
            avg_scale,
        )
        matched_gt_trans_for_error = normalize_vertices_to_camera0_gauge(
            matched_gt_trans,
            gt_extrinsics_raw,
            avg_scale,
        )

    visible_indices = np.asarray(predictions.get("smpl_visible_indices", []), dtype=np.int64).reshape(-1)
    mesh_index_by_pred_slot = {int(pred_idx): int(mesh_idx) for mesh_idx, pred_idx in enumerate(visible_indices)}
    mesh_color_names = predictions.get("smpl_mesh_color_names", None)
    if mesh_color_names is None:
        mesh_color_names = build_smpl_mesh_color_names(len(visible_indices))
    mesh_color_names = np.asarray(mesh_color_names, dtype=object).reshape(-1)

    per_person_losses = []
    for person_idx, (pred_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        pred_slot = int(pred_idx)
        gt_person = int(gt_idx)
        mesh_idx = mesh_index_by_pred_slot.get(pred_slot)
        mesh_color = (
            str(mesh_color_names[mesh_idx])
            if mesh_idx is not None and mesh_idx < mesh_color_names.shape[0]
            else "N/A"
        )
        per_person_losses.append(
            {
                "person": int(person_idx),
                "pred_slot": pred_slot,
                "gt_person": gt_person,
                "mesh": f"person_{mesh_idx:03d}.obj" if mesh_idx is not None else "N/A",
                "mesh_color": mesh_color,
                "matching_loss": compute_matching_loss_value(
                    pred_pose[pred_idx],
                    gt_pose_cmp[gt_idx],
                    pred_beta[pred_idx],
                    gt_beta[gt_idx],
                    matched_pred_trans_for_error[person_idx],
                    matched_gt_trans_for_error[person_idx],
                    matching_weights,
                    presence_cost=pred_presence_cost[pred_idx],
                ),
                "pose_error_deg": mean_pose_geodesic_deg(
                    pred_pose[pred_idx : pred_idx + 1],
                    gt_pose_cmp[gt_idx : gt_idx + 1],
                ),
                "beta_error_l1": mean_l1(pred_beta[pred_idx], gt_beta[gt_idx]),
                "translate_error_l1": mean_l1(
                    matched_pred_trans_for_error[person_idx],
                    matched_gt_trans_for_error[person_idx],
                ),
                "joint3d_error_l1": mean_l1(pred_joints3d[person_idx], gt_joints3d[person_idx]),
                "joint2d_error_l1_px": mean_l1(pred_joints2d[person_idx], gt_joints2d[person_idx]),
            }
        )

    metrics = {
        "matched_people": int(match_count),
        "pred_people": int(pred_pose.shape[0]),
        "gt_people": int(gt_pose.shape[0]),
        "camera_rotation_error_l1": mean_rotation_angle_from_matrices(
            pred_extrinsics[..., :3, :3],
            gt_extrinsics[..., :3, :3],
            degrees=True,
        ),
        "camera_translation_error_l1": mean_l1(pred_extrinsics[..., :3, 3], gt_extrinsics[..., :3, 3]),
        "camera_focal_length_error_l1": mean_l1(
            np.stack([pred_intrinsics[:, 0, 0], pred_intrinsics[:, 1, 1]], axis=-1),
            np.stack([gt_intrinsics[:, 0, 0], gt_intrinsics[:, 1, 1]], axis=-1),
        ),
        "smpl_pose_error_deg": mean_pose_geodesic_deg(matched_pred_pose, matched_gt_pose),
        "smpl_beta_error_l1": mean_l1(matched_pred_beta, matched_gt_beta),
        "smpl_translate_error_l1": mean_l1(matched_pred_trans_for_error, matched_gt_trans_for_error),
        "smpl_3d_joint_error_l1": mean_l1(pred_joints3d, gt_joints3d),
        "smpl_2d_joint_error_l1_px": mean_l1(pred_joints2d, gt_joints2d),
        "per_person_losses": per_person_losses,
        "error_gt_param_dir": gt["param_dir"],
        "error_gt_cam_dir": gt["cam_dir"],
        "error_coordinate_space": "normalized_camera0",
        "error_avg_scale": avg_scale,
        "error_translate_kind": "mesh_translate" if use_mesh_translate else "smpl_trans",
    }

    metrics["matched_pred_indices"] = row_ind
    metrics["matched_gt_indices"] = col_ind
    metrics["pred_smpl_joints3d"] = pred_joints3d
    metrics["gt_smpl_joints3d"] = gt_joints3d
    metrics["pred_smpl_joints2d"] = pred_joints2d
    metrics["gt_smpl_joints2d"] = gt_joints2d
    metrics["gt_intrinsic_adjusted"] = gt_intrinsics
    return metrics, gt


def out_mesh_dir_for_image_dir(input_image_dir: str) -> Path | None:
    """Map an image folder (.../<split>/out_image/.../runs_X) to its GT mesh folder
    (.../<split>/out_mesh/.../runs_X), mirroring the path but swapping out_image -> out_mesh.
    """
    folder = Path(str(input_image_dir or "").strip()).resolve()
    parts = list(folder.parts)
    if "out_image" not in parts:
        return None
    idx = parts.index("out_image")
    return Path(*parts[:idx], "out_mesh", *parts[idx + 1 :])


def load_out_mesh_people(input_image_dir: str, person_keys: list[str]) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load per-person GT meshes from out_mesh/<...>/runs_X/<subject>.npz (verts/faces in world coords).

    Returns one (verts, faces) pair per person, in the same order as `person_keys`, or None when
    the out_mesh folder or any expected per-subject file is missing (caller falls back to SMPL mesh).
    """
    mesh_dir = out_mesh_dir_for_image_dir(input_image_dir)
    if mesh_dir is None or not mesh_dir.is_dir():
        return None
    people = []
    for person_key in person_keys:
        subject = str(person_key).split("/")[0]
        cand = mesh_dir / f"{subject}.npz"
        if not cand.is_file():
            print(f"[GT] out_mesh file missing for subject '{subject}' ({cand}); falling back to SMPL mesh.")
            return None
        arr = np.load(str(cand))
        if "verts" not in arr or "faces" not in arr:
            print(f"[GT] out_mesh file {cand} lacks verts/faces; falling back to SMPL mesh.")
            return None
        people.append(
            (np.asarray(arr["verts"], dtype=np.float64), np.asarray(arr["faces"], dtype=np.int64))
        )
    return people


def combine_out_mesh_people_to_camera0_gauge(
    out_mesh_people: list[tuple[np.ndarray, np.ndarray]],
    raw_extrinsics: np.ndarray,
    avg_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each person's world-space GT mesh to the camera0 gauge and merge into one mesh.

    Returns (vertices, faces, vertex_colors). Vertex colors follow the same per-person palette as
    the SMPL meshes, assigned by each person's own vertex count (meshes may differ in size).
    """
    combined_verts, combined_faces, colors = [], [], []
    offset = 0
    for person_idx, (verts_world, faces) in enumerate(out_mesh_people):
        norm_verts = normalize_vertices_to_camera0_gauge(verts_world, raw_extrinsics, avg_scale)
        combined_verts.append(norm_verts)
        combined_faces.append(faces.astype(np.int64) + offset)
        offset += norm_verts.shape[0]
        _, rgba = get_smpl_mesh_color(person_idx)
        colors.append(np.tile(np.asarray(rgba, dtype=np.uint8), (norm_verts.shape[0], 1)))
    return (
        np.concatenate(combined_verts, axis=0),
        np.concatenate(combined_faces, axis=0),
        np.concatenate(colors, axis=0),
    )


def build_gt_multi_predictions(
    target_dir: str,
    image_paths: list[str],
    input_image_dir: str,
) -> tuple[dict, np.ndarray, np.ndarray, str, dict]:
    gt = load_gt_multi_from_image_dir(input_image_dir, image_paths)
    gt_intrinsics = adjust_intrinsics_for_preprocess(image_paths, gt["intrinsic_raw"], input_image_dir)
    if _looks_like_raw_mamma_sequence(Path(str(input_image_dir or "").strip())):
        images, processed_image_paths = load_raw_mamma_images_with_training_preprocess(
            input_image_dir,
            image_paths,
            target_dir,
        )
    else:
        images = load_and_preprocess_images(image_paths)
        processed_dir = os.path.join(target_dir, "processed_images")
        processed_image_paths = save_processed_images(image_paths, images, processed_dir)

    smpl_raw_obj_path = os.path.join(target_dir, "smpl_mesh_gt_raw.obj")
    smpl_verts_raw, smpl_faces, smpl_joints3d_raw = smpl_to_mesh_joints_and_obj(
        gt["smpl_pose"],
        gt["smpl_beta"],
        gt["smpl_trans"],
        smpl_raw_obj_path,
        genders=gt["genders"],
    )
    norm_extrinsics, norm_joints3d, smpl_verts, avg_scale = normalize_gt_cameras_joints_and_mesh(
        gt["extrinsic"],
        smpl_joints3d_raw,
        smpl_verts_raw,
    )
    people_count = ensure_people_array(gt["smpl_pose"], 72, "gt smpl_pose").shape[0]

    # Prefer the real GT meshes from out_mesh/ for the displayed/projected mesh when available;
    # fall back to the SMPL mesh reconstructed from params otherwise. Error metrics keep using
    # the param-derived SMPL joints regardless, so they are unaffected.
    smpl_vertex_colors = build_smpl_vertex_colors(smpl_verts, people_count)
    out_mesh_people = load_out_mesh_people(input_image_dir, gt["person_keys"])
    if out_mesh_people:
        smpl_verts, smpl_faces, smpl_vertex_colors = combine_out_mesh_people_to_camera0_gauge(
            out_mesh_people,
            gt["extrinsic"],
            avg_scale,
        )
        print(
            f"[GT] Using out_mesh for projection/visualization "
            f"({len(out_mesh_people)} people, verts={smpl_verts.shape[0]})."
        )

    smpl_obj_path = os.path.join(target_dir, "smpl_mesh_gt.obj")
    write_obj(smpl_verts, smpl_faces, smpl_obj_path)
    smpl_obj_path_compat = os.path.join(target_dir, "smpl_mesh.obj")
    write_obj(smpl_verts, smpl_faces, smpl_obj_path_compat)

    predictions = {
        "extrinsic": norm_extrinsics,
        "intrinsic": gt_intrinsics,
        "smpl_pose": gt["smpl_pose"],
        "smpl_beta": gt["smpl_beta"],
        "smpl_trans": gt["smpl_trans"],
        "smpl_vertices": smpl_verts,
        "smpl_faces": smpl_faces,
        "smpl_vertex_colors": smpl_vertex_colors,
        "smpl_visible_indices": np.arange(people_count, dtype=np.int64),
        "smpl_mesh_color_names": build_smpl_mesh_color_names(people_count),
        "smpl_obj_path": smpl_obj_path,
        "smpl_raw_obj_path": smpl_raw_obj_path,
        "smpl_vertices_raw": smpl_verts_raw,
        "smpl_joints3d_world": norm_joints3d,
        "smpl_joints3d_world_raw": smpl_joints3d_raw,
        "avg_scale": np.asarray(avg_scale, dtype=np.float64),
        "source_image_paths": np.asarray(image_paths),
        "processed_image_paths": np.asarray(processed_image_paths),
        "gt_smpl_pose": gt["smpl_pose"],
        "gt_smpl_beta": gt["smpl_beta"],
        "gt_smpl_trans": gt["smpl_trans"],
        "gt_extrinsic": gt["extrinsic"],
        "gt_extrinsic_normalized": norm_extrinsics,
        "gt_intrinsic_raw": gt["intrinsic_raw"],
        "gt_intrinsic_adjusted": gt_intrinsics,
    }
    return predictions, smpl_verts, smpl_faces, smpl_obj_path, gt


def build_gt_scene_mesh_for_prediction(
    input_image_dir: str,
    image_paths: list[str],
    target_dir: str,
    avg_scale: float | None = None,
    color: tuple[int, int, int, int] = GT_OVERLAY_MESH_COLOR,
) -> trimesh.Trimesh | None:
    """Build the GT mesh as a single trimesh in the camera0-normalized gauge.

    Uses the same GT cameras + avg_scale gauge that the predicted mesh is placed in, so the
    returned GT mesh overlays the predicted mesh in the same scene. Prefers the real GT meshes
    from out_mesh/, falling back to SMPL meshes decoded from the GT params. The whole mesh is
    painted a single uniform `color` so it is visually distinct from the predicted mesh.

    Returns None when GT cannot be loaded.
    """
    if not str(input_image_dir or "").strip():
        return None
    try:
        gt = load_gt_multi_from_image_dir(input_image_dir, image_paths)
    except Exception as e:
        print(f"[GT-OVERLAY] Could not load GT for overlay: {e}")
        return None

    raw_extrinsics = gt["extrinsic"]
    if avg_scale is None:
        extrinsics_t = torch.as_tensor(np.asarray(raw_extrinsics, dtype=np.float32)).unsqueeze(0)
        _, _, _, _, _, avg_scale_t = normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=extrinsics_t,
            scale_by_extrinsics=True,
        )
        avg_scale = float(avg_scale_t.reshape(-1)[0].item())

    out_mesh_people = load_out_mesh_people(input_image_dir, gt["person_keys"])
    if out_mesh_people:
        verts, faces, _ = combine_out_mesh_people_to_camera0_gauge(
            out_mesh_people, raw_extrinsics, avg_scale
        )
    else:
        # Write the fallback SMPL mesh into a dedicated subdir so it does not clobber the
        # predicted mesh's smpl_mesh.obj / smpl_people/ outputs under target_dir.
        gt_overlay_obj = os.path.join(target_dir, "gt_overlay", "smpl_mesh_gt_overlay.obj")
        verts_raw, faces = smpl_to_mesh_and_obj(
            gt["smpl_pose"],
            gt["smpl_beta"],
            gt["smpl_trans"],
            gt_overlay_obj,
            genders=gt["genders"],
        )
        verts = normalize_vertices_to_camera0_gauge(verts_raw, raw_extrinsics, avg_scale)

    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if verts.size == 0 or faces.size == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(
        np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1)
    )
    return mesh


def format_error_metrics(metrics: dict) -> str:
    if not metrics:
        return ""

    lines = ["Error metrics (prediction vs GT):"]
    is_mesh_translate = metrics.get("error_translate_kind") == "mesh_translate"
    translate_label = (
        "Mesh translate error (L1, normalized cam0)"
        if is_mesh_translate
        else "SMPL translate error (L1)"
    )
    labels = [
        ("matched_people", "matched people"),
        ("camera_rotation_error_l1", "Camera rotation error (deg)"),
        ("camera_translation_error_l1", "Camera translation error (L1)"),
        ("camera_focal_length_error_l1", "Camera focal length error (L1 px)"),
        ("smpl_pose_error_deg", "SMPL pose error (24 joints, deg)"),
        ("smpl_beta_error_l1", "SMPL beta error (L1)"),
        ("smpl_translate_error_l1", translate_label),
        ("smpl_3d_joint_error_l1", "3D joint error (L1)"),
        ("smpl_2d_joint_error_l1_px", "2D joint error (L1 px)"),
    ]
    for key, label in labels:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, (int, np.integer)):
            lines.append(f"- {label}: {int(value)} / pred {metrics.get('pred_people')} / gt {metrics.get('gt_people')}")
        else:
            lines.append(f"- {label}: {float(value):.6f}")
    lines.append(f"- GT SMPL: {metrics.get('error_gt_param_dir', '')}")
    lines.append(f"- GT Camera: {metrics.get('error_gt_cam_dir', '')}")
    return "\n".join(lines)


def format_per_person_losses(metrics: dict) -> str:
    rows = metrics.get("per_person_losses") if metrics else None
    if not rows:
        return ""

    lines = [
        "Per-person loss (prediction vs GT):",
        "",
        "| Person | Pred slot | GT person | Mesh | Mesh color | Matching loss | Pose deg | Beta L1 | Trans L1 | 3D joint L1 | 2D joint L1 px |",
        "|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row['person'])} | "
            f"{int(row.get('pred_slot', -1))} | "
            f"{int(row.get('gt_person', -1))} | "
            f"{row.get('mesh', 'N/A')} | "
            f"{row.get('mesh_color', 'N/A')} | "
            f"{float(row['matching_loss']):.6f} | "
            f"{float(row['pose_error_deg']):.6f} | "
            f"{float(row['beta_error_l1']):.6f} | "
            f"{float(row['translate_error_l1']):.6f} | "
            f"{float(row['joint3d_error_l1']):.6f} | "
            f"{float(row['joint2d_error_l1_px']):.6f} |"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def infer_target_dir_prefix(input_video=None, input_images=None, input_image_dir=None, use_gt: bool = False) -> str:
    if use_gt:
        return "gt_only"
    if input_video is not None:
        return "input_video"
    if input_images is not None:
        return "uploaded_images"
    if str(input_image_dir or "").strip():
        return "folder_images"
    return "input_images"


def handle_uploads(
    input_video,
    input_images,
    input_image_dir=None,
    image_ids="",
    target_prefix: str | None = None,
    save_outputs: bool = True,
    frame: str | None = None,
):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_prefix = target_prefix or infer_target_dir_prefix(input_video, input_images, input_image_dir)
    target_name = f"{target_prefix}_{timestamp}"
    target_dir = target_name if save_outputs else os.path.join(RUNTIME_OUTPUT_ROOT, target_name)
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle image folder path ---
    folder_images = list_images_from_folder(input_image_dir, image_ids, frame) if str(input_image_dir or "").strip() else []
    for file_path in folder_images:
        src_path = Path(file_path)
        parent_name = safe_filename_component(src_path.parent.name)
        dst_name = f"{len(image_paths):04d}_{parent_name}_{src_path.name}"
        dst_path = os.path.join(target_dir_images, dst_name)
        shutil.copy(file_path, dst_path)
        image_paths.append(dst_path)

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_image_dir, image_ids, save_outputs=False, frame=None):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return:
        reconstruction_output, target_dir, image_paths, log_msg, smpl_obj_state, error_output, presence_output, per_person_loss_output
    """
    if not str(input_image_dir or "").strip():
        # 沒有上傳任何東西：清空場景 + log
        return None, "None", None, [], [], [], "No input uploaded.", "None", "", "", ""

    try:
        target_dir, image_paths = handle_uploads(
            None,
            None,
            input_image_dir,
            image_ids,
            save_outputs=bool(save_outputs),
            frame=frame,
        )
    except Exception as e:
        return None, "None", None, [], [], [], f"Input error: {e}", "None", "", "", ""
    # 新上傳：清空 viewer，更新 target_dir 與 gallery
    return (
        None,                              # reconstruction_output
        target_dir,                        # target_dir_output
        image_paths,                       # image_gallery
        [],                                # projection_gallery
        [],                                # attention_smpl_gallery
        [],                                # landmark_mask_gallery
        "Input ready. Click 'Reconstruct' to begin 3D processing.",  # log_output
        "None",                            # smpl_obj_state
        "",                                # error_output
        "",                                # smpl_presence_output
        "",                                # per_person_loss_output
    )


def update_gallery_for_dataset_selection(dataset_root, split, run_name, image_ids, save_outputs=False, frame=None):
    if not str(split or "").strip() or not str(run_name or "").strip():
        return "", None, "None", None, [], [], [], "No run selected.", "None", "", "", ""
    input_image_dir = dataset_image_dir(dataset_root, split, run_name)
    return (input_image_dir, *update_gallery_on_upload(input_image_dir, image_ids, save_outputs, frame))


def populate_frame_dropdown(dataset_root, split, run_name):
    """Refresh the Frame dropdown for the selected scene (raw Mamma only)."""
    if not str(split or "").strip() or not str(run_name or "").strip():
        return gr.update(choices=[], value=None)
    frames = list_scene_frames(dataset_image_dir(dataset_root, split, run_name))
    return gr.update(choices=frames, value=(frames[0] if frames else None))


def update_dataset_split(dataset_root, split, image_ids, save_outputs=False):
    root = str(dataset_root or DEFAULT_DATASET_ROOT).strip() or DEFAULT_DATASET_ROOT
    runs = list_dataset_runs(root, split)
    if not runs:
        image_subdir = discover_image_subdir(root, split)
        return (
            gr.update(choices=[], value=None),
            "",
            None,
            "None",
            None,
            [],
            [],
            [],
            f"No eval runs or raw Mamma sequences found under {Path(root) / str(split) / image_subdir} or {root}",
            "None",
            "",
            "",
            "",
        )

    run_name = DEFAULT_DATASET_RUN if str(split) == DEFAULT_DATASET_SPLIT and DEFAULT_DATASET_RUN in runs else runs[0]
    input_image_dir, *gallery_update = update_gallery_for_dataset_selection(
        root,
        split,
        run_name,
        image_ids,
        save_outputs,
    )
    return (gr.update(choices=runs, value=run_name), input_image_dir, *gallery_update)


def _shutdown_projection_worker() -> None:
    global _projection_worker
    worker = _projection_worker
    _projection_worker = None
    if worker is None:
        return
    try:
        if worker.poll() is None and worker.stdin is not None:
            worker.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            worker.stdin.flush()
    except Exception:
        pass
    try:
        worker.terminate()
    except Exception:
        pass


atexit.register(_shutdown_projection_worker)


def _get_projection_worker() -> subprocess.Popen:
    global _projection_worker
    if _projection_worker is not None and _projection_worker.poll() is None:
        return _projection_worker
    if not os.path.exists(PYTORCH3D_PROJECTION_PYTHON):
        raise FileNotFoundError(f"Projection Python not found: {PYTORCH3D_PROJECTION_PYTHON}")
    _projection_worker = subprocess.Popen(
        [PYTORCH3D_PROJECTION_PYTHON, "-u", PYTORCH3D_PROJECTION_WORKER_SCRIPT],
        cwd=_REPO_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    return _projection_worker


def run_pytorch3d_projection(target_dir: str) -> tuple[list[str], str]:
    if not target_dir or not os.path.isdir(target_dir):
        return [], "Projection skipped: target directory does not exist."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    processed_dir = os.path.join(target_dir, "processed_images")
    if not os.path.exists(predictions_path):
        return [], f"Projection skipped: missing {predictions_path}"
    if not os.path.isdir(processed_dir):
        return [], f"Projection skipped: missing {processed_dir}"

    try:
        worker = _get_projection_worker()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("Projection worker pipes are unavailable.")
        print(f"[Projection] Running persistent CPU worker: {target_dir}")
        worker.stdin.write(json.dumps({"cmd": "project", "base_dir": target_dir, "workers": 2}) + "\n")
        worker.stdin.flush()
        response_line = worker.stdout.readline()
        if not response_line:
            raise RuntimeError("Projection worker exited without a response.")
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Projection worker failed."))
        projection_paths = response.get("paths", [])
    except Exception as e:
        msg = f"Projection failed: {e}"
        print(f"[WARN] {msg}")
        return [], msg

    projection_dir = os.path.join(target_dir, "projections_processed")
    projection_paths = projection_paths or sorted(glob.glob(os.path.join(projection_dir, "*.png")))
    if not projection_paths:
        return [], f"Projection finished but no PNG files found in {projection_dir}"
    return projection_paths, f"Projection images: {projection_dir}"


def run_projection_gallery(target_dir: str) -> list[str]:
    projection_paths, projection_msg = run_pytorch3d_projection(target_dir)
    if projection_msg:
        print(f"[Projection] {projection_msg}")
    return projection_paths


def _save_gif_from_paths(image_paths: list[str], gif_path: str, fps: float = 2.0) -> str:
    if not image_paths:
        raise ValueError("No images provided for GIF.")
    frames = []
    for path in image_paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return gif_path


def _is_mamma_sequence_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    view_dirs = [p for p in path.iterdir() if p.is_dir()]
    if not view_dirs:
        return False
    for view_dir in view_dirs[:4]:
        if any(view_dir.glob("*.data.pyd")) and any(
            p.suffix.lower() in SUPPORTED_IMAGE_EXTS and ".mask" not in p.name
            for p in view_dir.iterdir()
            if p.is_file()
        ):
            return True
    return False


def _resolve_mamma_sequence_dir(path: str, sequence_name: str = "") -> Path:
    root = Path(path)
    if _is_mamma_sequence_dir(root):
        return root
    candidates_root = root / "png" if (root / "png").is_dir() else root
    if sequence_name:
        selected = candidates_root / sequence_name
        if not _is_mamma_sequence_dir(selected):
            raise FileNotFoundError(f"Sequence '{sequence_name}' not found under: {candidates_root}")
        return selected
    candidates = sorted(p for p in candidates_root.iterdir() if _is_mamma_sequence_dir(p)) if candidates_root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw Mamma sequence folders under: {root}. "
            "Expected either <seq>/<view>/*.data.pyd or <root>/png/<seq>/<view>/*.data.pyd."
        )
    return candidates[0]


def _copy_mamma_sequence_frames(seq_dir: str, view_name: str, max_frames: int) -> tuple[str, list[str]]:
    seq_path = _resolve_mamma_sequence_dir(seq_dir, args.demo_sequence)
    if not seq_path.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    view_dirs = sorted(p for p in seq_path.iterdir() if p.is_dir())
    if not view_dirs:
        raise FileNotFoundError(f"No view folders found under: {seq_dir}")
    if view_name:
        selected_view = seq_path / view_name
        if not selected_view.is_dir():
            raise FileNotFoundError(f"Requested view '{view_name}' not found under: {seq_dir}")
    else:
        selected_view = view_dirs[0]

    frame_paths = sorted(
        p for p in selected_view.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS and ".mask" not in p.name
    )
    if max_frames > 0:
        frame_paths = frame_paths[: int(max_frames)]
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found under: {selected_view}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_dir = Path(RUNTIME_OUTPUT_ROOT) / f"mamma_seq_{seq_path.name}_{selected_view.name}_{timestamp}" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, src in enumerate(frame_paths):
        dst = input_dir / f"{idx:04d}_{src.name}"
        shutil.copyfile(src, dst)
        copied.append(str(dst))
    print(f"[CLI-DEMO] Copied {len(copied)} frames from {selected_view} to {input_dir}")
    return str(input_dir), copied


def run_cli_mamma_sequence_demo(seq_dir: str, view_name: str, max_frames: int, fps: float) -> str:
    input_dir, _ = _copy_mamma_sequence_frames(seq_dir, view_name, max_frames)
    result = gradio_demo(
        target_dir="None",
        conf_thres=50,
        mask_black_bg=False,
        mask_white_bg=False,
        mask_green_bg=False,
        show_cam=True,
        mask_sky=False,
        prediction_mode="SMPL Only (Pose/Beta)",
        input_image_dir=input_dir,
        image_ids="",
        use_gt=False,
        smpl_presence_threshold=SMPL_PRESENCE_THRESHOLD,
        use_hungarian_smpl_mesh=False,
        save_outputs=True,
        overlay_gt_mesh=False,
    )
    (
        glb_path,
        _log_msg,
        smpl_obj_path,
        target_dir,
        _preview_paths,
        _projection_paths,
        error_text,
        presence_text,
        _per_person_text,
    ) = result
    if not target_dir or target_dir == "None":
        raise RuntimeError(f"CLI demo failed: {error_text}")
    print(f"[CLI-DEMO] target_dir={target_dir}")
    print(f"[CLI-DEMO] glb={glb_path}")
    print(f"[CLI-DEMO] smpl_obj={smpl_obj_path}")
    if presence_text:
        print(f"[CLI-DEMO] presence:\n{presence_text}")
    projection_paths = run_projection_gallery(target_dir)
    gif_path = os.path.join(target_dir, "projected_smpl_10frames.gif")
    _save_gif_from_paths(projection_paths, gif_path, fps=fps)
    print(f"[CLI-DEMO] projection_frames={len(projection_paths)}")
    print(f"[CLI-DEMO] gif={gif_path}")
    return gif_path


def postprocess_existing_target_dir(target_dir: str, fps: float = 2.0) -> str:
    target_dir = str(target_dir)
    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(predictions_path)
    predictions = dict(np.load(predictions_path, allow_pickle=True))
    if "smpl_vertices" not in predictions or "smpl_faces" not in predictions:
        if "smpl_pose" not in predictions or "smpl_beta" not in predictions:
            raise RuntimeError(f"No SMPL pose/beta in {predictions_path}")
        visible_pose, visible_beta, visible_trans, _, visible_indices = select_smpl_slots_for_mesh(
            predictions,
            presence_threshold=SMPL_PRESENCE_THRESHOLD,
            use_hungarian_mesh_selection=False,
            matching_metrics={},
        )
        if visible_pose.shape[0] == 0:
            raise RuntimeError("No SMPL slots passed the presence threshold.")
        smpl_obj_path = os.path.join(target_dir, "smpl_mesh.obj")
        if predictions_use_mesh_translate(predictions):
            print("[POSTPROCESS] Building mesh from mesh_translate in predicted camera gauge.")
            mesh_translate_all = ensure_people_array(predictions["mesh_translate"], 3, "mesh_translate")
            mesh_translate_visible = mesh_translate_all[visible_indices]
            image_count = len(predictions.get("processed_image_paths", []))
            identity_extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, :3, :], max(1, image_count), axis=0)
            smpl_verts, smpl_faces, smpl_verts_raw = place_pred_smpl_with_mesh_translate(
                visible_pose,
                visible_beta,
                mesh_translate_visible,
                identity_extrinsics,
                1.0,
                smpl_obj_path,
                use_mesh_rot=predictions_use_mesh_rot(predictions),
            )
            write_obj(smpl_verts, smpl_faces, smpl_obj_path)
            predictions["smpl_vertices_raw"] = smpl_verts_raw
            predictions["mesh_translate_visible"] = mesh_translate_visible
        else:
            print("[POSTPROCESS] Building mesh from smpl_trans.")
            smpl_verts, smpl_faces = smpl_to_mesh_and_obj(
                visible_pose,
                visible_beta,
                visible_trans,
                smpl_obj_path,
            )
        predictions["smpl_vertices"] = smpl_verts
        predictions["smpl_faces"] = smpl_faces
        predictions["smpl_vertex_colors"] = build_smpl_vertex_colors(smpl_verts, visible_pose.shape[0])
        predictions["smpl_mesh_color_names"] = build_smpl_mesh_color_names(visible_pose.shape[0])
        predictions["smpl_obj_path"] = smpl_obj_path
        predictions["smpl_visible_indices"] = visible_indices
        np.savez(predictions_path, **predictions)
        print(f"[POSTPROCESS] Added smpl_vertices/smpl_faces to {predictions_path}")
    else:
        print("[POSTPROCESS] predictions.npz already has smpl_vertices/smpl_faces.")

    projection_paths = run_projection_gallery(target_dir)
    gif_path = os.path.join(target_dir, "projected_smpl_mesh.gif")
    _save_gif_from_paths(projection_paths, gif_path, fps=fps)
    print(f"[POSTPROCESS] projection_frames={len(projection_paths)}")
    print(f"[POSTPROCESS] gif={gif_path}")
    return gif_path


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    mask_black_bg=False,
    mask_white_bg=False,
    mask_green_bg=True,
    show_cam=True,
    mask_sky=False,
    prediction_mode="SMPL Only (Pose/Beta)",
    input_image_dir="",
    image_ids=DEFAULT_IMAGE_IDS,
    use_gt=False,
    smpl_presence_threshold=SMPL_PRESENCE_THRESHOLD,
    use_hungarian_smpl_mesh=False,
    save_outputs=False,
    overlay_gt_mesh=True,
    frame=None,
):
    """
    Perform reconstruction in a fresh output directory for every button click.
    """
    use_gt = bool(use_gt)
    use_hungarian_smpl_mesh = bool(use_hungarian_smpl_mesh)
    save_outputs = bool(save_outputs)
    overlay_gt_mesh = bool(overlay_gt_mesh)
    try:
        smpl_presence_threshold = float(smpl_presence_threshold)
    except (TypeError, ValueError):
        smpl_presence_threshold = float(SMPL_PRESENCE_THRESHOLD)
    presence_output_text = ""
    raw_mamma_image_paths = None
    if use_gt:
        if not str(input_image_dir or "").strip():
            return None, "Use GT requires a selected dataset run.", "None", "None", [], [], "", "", ""
        try:
            target_dir, _ = handle_uploads(
                None,
                None,
                input_image_dir,
                image_ids,
                target_prefix="gt_only",
                save_outputs=save_outputs,
                frame=frame,
            )
        except Exception as e:
            return None, f"Input folder error: {e}", "None", "None", [], [], "", "", ""
    elif str(input_image_dir or "").strip():
        try:
            if _looks_like_raw_mamma_sequence(Path(str(input_image_dir).strip())):
                raw_mamma_image_paths = list_images_from_folder(input_image_dir, image_ids, frame)
            target_dir, _ = handle_uploads(
                None,
                None,
                input_image_dir,
                image_ids,
                target_prefix="folder_images",
                save_outputs=save_outputs,
                frame=frame,
            )
        except Exception as e:
            return None, f"Input folder error: {e}", "None", "None", [], [], "", "", ""
    else:
        existing_images = list_existing_target_images(target_dir)
        if not existing_images:
            return None, "No valid input images found. Please select a dataset run first.", "None", "None", [], [], "", "", ""
        try:
            target_dir, _ = handle_uploads(
                None,
                existing_images,
                None,
                "",
                target_prefix="uploaded_images",
                save_outputs=save_outputs,
            )
        except Exception as e:
            return None, f"Input copy error: {e}", "None", "None", [], [], "", "", ""

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    mode_str = str(prediction_mode)
    # "Camera Only" / "SMPL Only" modes skip depth unprojection + depth storage
    camera_only = "Camera Only" in mode_str
    compute_world_points_from_depth = (not camera_only) and ("SMPL Only" not in mode_str)

    smpl_verts = None
    smpl_faces = None
    smpl_obj_path = None
    if use_gt:
        eval_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
        print("[GT] Using GT intrinsics/extrinsics and SMPL from inferred multi-person folders")
        predictions, smpl_verts, smpl_faces, smpl_obj_path, gt_pack = build_gt_multi_predictions(
            target_dir,
            eval_image_paths,
            input_image_dir,
        )
    else:
        print("Running model...")
        with torch.no_grad():
            predictions = run_model(
                target_dir,
                model,
                compute_world_points_from_depth=compute_world_points_from_depth,
                camera_only=camera_only,
                input_image_dir=input_image_dir,
                raw_mamma_image_paths=raw_mamma_image_paths,
            )

    pre_mesh_error_metrics = {}
    if (
        (not use_gt)
        and use_hungarian_smpl_mesh
        and str(input_image_dir or "").strip()
        and has_smpl_param_outputs(predictions)
    ):
        try:
            eval_image_paths = predictions.get("source_image_paths", None)
            if isinstance(eval_image_paths, np.ndarray):
                eval_image_paths = eval_image_paths.tolist()
            if eval_image_paths is None or len(eval_image_paths) == 0:
                eval_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
            pre_mesh_error_metrics, _ = compute_multi_prediction_errors(
                predictions,
                eval_image_paths,
                input_image_dir=input_image_dir,
            )
        except Exception as e:
            print(f"[WARN] Failed to compute pre-mesh Hungarian matching: {e}")

    # --------------------------------------------------------
    # NEW: 用 SMPL 輸出產生 mesh + OBJ，並存進 predictions
    # --------------------------------------------------------
    if (not use_gt) and "smpl_pose" in predictions and "smpl_beta" in predictions:
        try:
            print(f"[CGV LOG] SMPL POSE :", predictions["smpl_pose"])
            visible_pose, visible_beta, visible_trans, _, visible_indices = select_smpl_slots_for_mesh(
                predictions,
                presence_threshold=smpl_presence_threshold,
                use_hungarian_mesh_selection=use_hungarian_smpl_mesh,
                matching_metrics=pre_mesh_error_metrics,
            )

            if visible_pose.shape[0] == 0:
                print("[SMPL] No slots passed presence threshold; skipping SMPL mesh export.")
            else:
                use_mesh_translate = predictions_use_mesh_translate(predictions)
                has_gt_camera_for_mesh = has_gt_camera_metadata(input_image_dir)
                smpl_obj_path = os.path.join(target_dir, "smpl_mesh.obj")
                smpl_raw_obj_path = (
                    os.path.join(target_dir, "smpl_mesh_raw.obj")
                    if has_gt_camera_for_mesh
                    else smpl_obj_path
                )

                mesh_image_paths = predictions.get("source_image_paths", None)
                if isinstance(mesh_image_paths, np.ndarray):
                    mesh_image_paths = mesh_image_paths.tolist()
                if mesh_image_paths is None or len(mesh_image_paths) == 0:
                    mesh_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))

                if use_mesh_translate and has_gt_camera_for_mesh:
                    # mesh_translate head: place the SMPL mesh by anchoring each person's
                    # root to the predicted mesh_translate in the GT camera0 normalized frame.
                    _, norm_gt_extrinsics, avg_scale, gt_camera_pack = normalize_mesh_vertices_from_gt_cameras(
                        input_image_dir,
                        mesh_image_paths,
                        np.zeros((1, 3), dtype=np.float64),
                    )
                    raw_extrinsics = gt_camera_pack["extrinsic"]
                    mesh_translate_all = ensure_people_array(predictions["mesh_translate"], 3, "mesh_translate")
                    mesh_translate_visible = mesh_translate_all[visible_indices]
                    smpl_verts, smpl_faces, smpl_verts_raw = place_pred_smpl_with_mesh_translate(
                        visible_pose,
                        visible_beta,
                        mesh_translate_visible,
                        raw_extrinsics,
                        avg_scale,
                        smpl_raw_obj_path,
                        use_mesh_rot=predictions_use_mesh_rot(predictions),
                    )
                    write_obj(smpl_verts, smpl_faces, smpl_obj_path)
                    predictions["smpl_vertices_raw"] = smpl_verts_raw
                    predictions["smpl_raw_obj_path"] = smpl_raw_obj_path
                    predictions["gt_extrinsic_normalized"] = norm_gt_extrinsics
                    predictions["gt_extrinsic"] = gt_camera_pack["extrinsic"]
                    predictions["avg_scale"] = np.asarray(avg_scale, dtype=np.float64)
                    predictions["mesh_translate_visible"] = mesh_translate_visible
                    print(
                        "[SMPL] Placed mesh_translate-anchored mesh in normalized cam0 gauge "
                        f"(avg_scale={avg_scale:.6f})."
                    )
                else:
                    if use_mesh_translate and not has_gt_camera_for_mesh:
                        print(
                            "[SMPL] No eval GT camera archive found; placing mesh_translate output "
                            "directly in the model's predicted camera gauge."
                        )
                        mesh_translate_all = ensure_people_array(predictions["mesh_translate"], 3, "mesh_translate")
                        mesh_translate_visible = mesh_translate_all[visible_indices]
                        identity_extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, :3, :], len(mesh_image_paths), axis=0)
                        smpl_verts, smpl_faces, smpl_verts_raw = place_pred_smpl_with_mesh_translate(
                            visible_pose,
                            visible_beta,
                            mesh_translate_visible,
                            identity_extrinsics,
                            1.0,
                            smpl_raw_obj_path,
                            use_mesh_rot=predictions_use_mesh_rot(predictions),
                        )
                        write_obj(smpl_verts, smpl_faces, smpl_obj_path)
                        predictions["smpl_vertices_raw"] = smpl_verts_raw
                        predictions["smpl_raw_obj_path"] = smpl_raw_obj_path
                        predictions["avg_scale"] = np.asarray(1.0, dtype=np.float64)
                        predictions["mesh_translate_visible"] = mesh_translate_visible
                    else:
                        smpl_verts_raw, smpl_faces = smpl_to_mesh_and_obj(
                            visible_pose,
                            visible_beta,
                            visible_trans,
                            smpl_raw_obj_path,
                        )
                        smpl_verts = smpl_verts_raw

                    if has_gt_camera_for_mesh:
                        smpl_verts, norm_gt_extrinsics, avg_scale, gt_camera_pack = normalize_mesh_vertices_from_gt_cameras(
                            input_image_dir,
                            mesh_image_paths,
                            smpl_verts_raw,
                        )
                        write_obj(smpl_verts, smpl_faces, smpl_obj_path)
                        predictions["smpl_vertices_raw"] = smpl_verts_raw
                        predictions["smpl_raw_obj_path"] = smpl_raw_obj_path
                        predictions["gt_extrinsic_normalized"] = norm_gt_extrinsics
                        predictions["gt_extrinsic"] = gt_camera_pack["extrinsic"]
                        predictions["avg_scale"] = np.asarray(avg_scale, dtype=np.float64)
                        print(
                            "[SMPL] Normalized predicted mesh to GT camera0 gauge "
                            f"(avg_scale={avg_scale:.6f})."
                        )

                predictions["smpl_vertices"] = smpl_verts
                predictions["smpl_faces"] = smpl_faces
                predictions["smpl_vertex_colors"] = build_smpl_vertex_colors(smpl_verts, visible_pose.shape[0])
                predictions["smpl_mesh_color_names"] = build_smpl_mesh_color_names(visible_pose.shape[0])
                predictions["smpl_obj_path"] = smpl_obj_path

                print(f"[SMPL] Mesh generated from slots {visible_indices.tolist()} and stored at: {smpl_obj_path}")
        except Exception as e:
            print(f"[WARN] SMPL mesh generation failed: {e}")
            smpl_verts, smpl_faces, smpl_obj_path = None, None, None

    presence_output_text = format_smpl_presence_output(
        predictions,
        use_gt=use_gt,
        presence_threshold=smpl_presence_threshold,
    )
    smpl_mesh = build_smpl_scene_mesh(predictions)

    # Overlay the GT mesh in the same scene so the predicted mesh (one uniform color) and the
    # GT mesh (another uniform color) can be compared directly. Only applies when visualizing
    # predictions (not the GT-only mode) and a GT dataset run is selected; the GT mesh is built
    # in the same camera0-normalized gauge the predicted mesh was placed in, so they align.
    if (not use_gt) and overlay_gt_mesh and smpl_mesh is not None and str(input_image_dir or "").strip():
        try:
            overlay_image_paths = predictions.get("source_image_paths", None)
            if isinstance(overlay_image_paths, np.ndarray):
                overlay_image_paths = overlay_image_paths.tolist()
            if not overlay_image_paths:
                overlay_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))

            overlay_avg_scale = predictions.get("avg_scale", None)
            if overlay_avg_scale is not None:
                overlay_avg_scale = float(np.asarray(overlay_avg_scale, dtype=np.float64).reshape(-1)[0])

            gt_overlay_mesh = build_gt_scene_mesh_for_prediction(
                input_image_dir,
                overlay_image_paths,
                target_dir,
                avg_scale=overlay_avg_scale,
                color=GT_OVERLAY_MESH_COLOR,
            )
            if gt_overlay_mesh is not None:
                # Recolor the predicted mesh to a single uniform color (overrides the per-person
                # palette) so prediction vs GT is distinguishable, then merge GT into the scene mesh.
                smpl_mesh.visual.vertex_colors = np.tile(
                    np.asarray(PRED_OVERLAY_MESH_COLOR, dtype=np.uint8), (len(smpl_mesh.vertices), 1)
                )
                smpl_mesh = trimesh.util.concatenate([smpl_mesh, gt_overlay_mesh])
                print(
                    f"[GT-OVERLAY] Added GT mesh overlay "
                    f"(pred={PRED_OVERLAY_MESH_COLOR}, gt={GT_OVERLAY_MESH_COLOR}, "
                    f"gt_verts={len(gt_overlay_mesh.vertices)})."
                )
        except Exception as e:
            print(f"[WARN] Failed to build GT mesh overlay: {e}")

    if str(input_image_dir or "").strip():
        try:
            gt_lmk_image_paths = predictions.get("source_image_paths", None)
            if isinstance(gt_lmk_image_paths, np.ndarray):
                gt_lmk_image_paths = gt_lmk_image_paths.tolist()
            if gt_lmk_image_paths is None or len(gt_lmk_image_paths) == 0:
                gt_lmk_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
            gt_landmark_pack = load_raw_mamma_gt_landmarks_for_demo(
                input_image_dir,
                gt_lmk_image_paths,
            )
            if gt_landmark_pack is not None:
                gt_landmarks, gt_landmark_visibility = gt_landmark_pack
                predictions["gt_smpl_landmarks2d"] = gt_landmarks
                predictions["gt_smpl_landmarks2d_visibility"] = gt_landmark_visibility
                print(
                    "[GT-Landmark] Added GT 512-landmark overlays "
                    f"shape={gt_landmarks.shape}, visible={int(gt_landmark_visibility.sum())}."
                )
        except Exception as e:
            print(f"[WARN] Failed to add GT dense landmarks: {e}")

    error_metrics = {}
    if has_smpl_param_outputs(predictions):
        try:
            eval_image_paths = predictions.get("source_image_paths", None)
            if isinstance(eval_image_paths, np.ndarray):
                eval_image_paths = eval_image_paths.tolist()
            if eval_image_paths is None or len(eval_image_paths) == 0:
                eval_image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
            error_metrics, gt_pack = compute_multi_prediction_errors(
                predictions,
                eval_image_paths,
                input_image_dir=input_image_dir,
            )
            predictions.update(error_metrics)
            if gt_pack:
                predictions["gt_smpl_pose"] = gt_pack["smpl_pose"]
                predictions["gt_smpl_beta"] = gt_pack["smpl_beta"]
                predictions["gt_smpl_trans"] = gt_pack["smpl_trans"]
                predictions["gt_extrinsic"] = gt_pack["extrinsic"]
                predictions["gt_intrinsic_raw"] = gt_pack["intrinsic_raw"]
        except Exception as e:
            print(f"[WARN] Failed to compute prediction errors: {e}")

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Always export pose/beta as a lightweight artifact for downstream use
    if ("smpl_pose" in predictions) or ("smpl_beta" in predictions):
        smpl_save_path = os.path.join(target_dir, "smpl_pose_beta.npz")
        np.savez(
            smpl_save_path,
            smpl_pose=predictions.get("smpl_pose", None),
            smpl_beta=predictions.get("smpl_beta", None),
            smpl_trans=predictions.get("smpl_trans", None),
            mesh_translate=predictions.get("mesh_translate", None),
            smpl_presence_logits=predictions.get("smpl_presence_logits", None),
            smpl_presence_prob=predictions.get("smpl_presence_prob", None),
            smpl_presence_mask=predictions.get("smpl_presence_mask", None),
            smpl_visible_indices=predictions.get("smpl_visible_indices", None),
            smpl_presence_threshold=np.asarray(smpl_presence_threshold, dtype=np.float32),
            smpl_mesh_selection_mode=np.asarray(str(predictions.get("smpl_mesh_selection_mode", ""))),
        )

    frame_filter = "All"
    frame_filter_name = safe_filename_component(frame_filter)
    prediction_mode_name = safe_filename_component(prediction_mode)

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter_name}"
        f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}"
        f"_pred{prediction_mode_name}_{GLB_VIS_VERSION}.glb",
    )

    # 不用刪 key，交給 predictions_to_glb 依 show_cam 處理
    viz_pred = predictions

    # Convert predictions to GLB, including the SMPL mesh when available.
    glbscene = predictions_to_glb(
        viz_pred,
        conf_thres=conf_thres,
        scene_scale=1.0,
        filter_by_frames="All",
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_green_bg=mask_green_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        smpl_mesh=smpl_mesh,
    )

    glbscene.export(file_obj=glbfile)
    add_lights_to_glb(glbfile)

    preview_paths = predictions.get("processed_image_paths", None)
    if preview_paths is None:
        preview_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    elif isinstance(preview_paths, np.ndarray):
        preview_paths = preview_paths.tolist()

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = ""
    formatted_metrics = format_error_metrics(error_metrics)
    formatted_per_person_losses = format_per_person_losses(error_metrics)
    if formatted_metrics:
        print(f"[ERROR_METRICS]\n{formatted_metrics}")
    if formatted_per_person_losses:
        print(f"[PER_PERSON_LOSSES]\n{formatted_per_person_losses}")

    smpl_state_path = smpl_obj_path if smpl_obj_path is not None else "None"

    # 回傳：場景 GLB、log、SMPL OBJ state
    return (
        glbfile,
        log_msg,
        smpl_state_path,
        target_dir,
        preview_paths,
        [],
        formatted_metrics,
        presence_output_text,
        formatted_per_person_losses,
    )


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    清空主場景。
    """
    return None, "", "", "", [], [], []


def update_log():
    """
    Display a quick log message while waiting.
    """
    return ""


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

if str(args.demo_seq_dir or "").strip():
    run_cli_mamma_sequence_demo(
        seq_dir=args.demo_seq_dir,
        view_name=args.demo_view,
        max_frames=args.demo_max_frames,
        fps=args.demo_fps,
    )
    raise SystemExit(0)

if str(args.postprocess_target_dir or "").strip():
    postprocess_existing_target_dir(
        target_dir=args.postprocess_target_dir,
        fps=args.demo_fps,
    )
    raise SystemExit(0)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }

    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }

    .error-log * {
        font-size: 16px !important;
        line-height: 1.35 !important;
    }

    .error-log p,
    .error-log li,
    .error-log pre,
    .error-log code {
        font-size: 12px !important;
    }

    .smpl-presence-log table,
    .smpl-presence-log table * {
        font-size: 11px !important;
        line-height: 1.15 !important;
    }

    .smpl-presence-log th,
    .smpl-presence-log td {
        padding: 3px 6px !important;
    }

    .per-person-loss-log table,
    .per-person-loss-log table * {
        font-size: 12px !important;
        line-height: 1.2 !important;
    }

    .per-person-loss-log th,
    .per-person-loss-log td {
        padding: 4px 6px !important;
    }

    footer,
    .footer,
    #footer,
    .built-with,
    .api-docs,
    button[aria-label="Settings"],
    button[title="Settings"] {
        display: none !important;
    }

    """,
) as demo:
    # 這裡不需要 frame_filter 了
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    # 用來存 SMPL OBJ 路徑的 hidden state
    smpl_obj_state = gr.Textbox(label="SMPL OBJ Path", visible=False, value="None")
    initial_runs = list_dataset_runs(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT)
    initial_run = DEFAULT_DATASET_RUN if DEFAULT_DATASET_RUN in initial_runs else (initial_runs[0] if initial_runs else None)
    initial_input_image_dir = (
        dataset_image_dir(DEFAULT_DATASET_ROOT, DEFAULT_DATASET_SPLIT, initial_run)
        if initial_run is not None
        else DEFAULT_INPUT_IMAGE_DIR
    )
    initial_frames = list_scene_frames(initial_input_image_dir)
    try:
        initial_preview_images = list_images_from_folder(
            initial_input_image_dir, DEFAULT_IMAGE_IDS, initial_frames[0] if initial_frames else None
        )
    except Exception as e:
        print(f"[WARN] Failed to load initial preview images: {e}")
        initial_preview_images = []
    input_image_dir = gr.State(initial_input_image_dir)

    # Fixed visualization settings; no UI controls.
    mask_black_bg = gr.State(False)
    mask_white_bg = gr.State(False)
    mask_green_bg = gr.State(False)
    show_cam = gr.State(True)
    mask_sky = gr.State(False)
    conf_thres = gr.State(50)
    prediction_mode = gr.State("SMPL Only (Pose/Beta)")

    with gr.Row():
        with gr.Column(scale=2):
            dataset_root = gr.Textbox(
                label="Dataset root",
                value=DEFAULT_DATASET_ROOT,
                lines=1,
                interactive=True,
            )
            dataset_split = gr.Dropdown(
                choices=["train", "test"],
                value=DEFAULT_DATASET_SPLIT,
                label="Dataset Split",
                interactive=True,
            )
            dataset_run = gr.Dropdown(
                choices=initial_runs,
                value=initial_run,
                label="Run Folder",
                interactive=True,
            )
            image_ids = gr.Textbox(
                label="Image IDs",
                value=DEFAULT_IMAGE_IDS,
                placeholder="0 1 2 3 4 5 6 7",
                lines=1,
            )
            frame_dropdown = gr.Dropdown(
                choices=initial_frames,
                value=(initial_frames[0] if initial_frames else None),
                label="Frame / time step (raw Mamma sequences)",
                interactive=True,
            )

            smpl_presence_threshold = gr.Number(
                label="SMPL Presence Threshold",
                value=SMPL_PRESENCE_THRESHOLD,
                precision=4,
                interactive=True,
            )
            use_gt = gr.Checkbox(label="Use GT", value=False)
            overlay_gt_mesh = gr.Checkbox(
                label="Overlay GT mesh (pred=blue, GT=orange)",
                value=True,
                interactive=True,
            )
            use_hungarian_smpl_mesh = gr.Checkbox(
                label="Use Hungarian matched SMPL slots for mesh",
                value=False,
                interactive=True,
            )
            save_outputs = gr.Checkbox(
                label="Save outputs to folder",
                value=False,
                interactive=True,
            )

            image_gallery = gr.Gallery(
                label="Preview",
                value=initial_preview_images,
                columns=4,
                height="300px",
                buttons=["download"],
                object_fit="contain",
                preview=True,
            )
            error_output = gr.Markdown("", elem_classes=["error-log"])
            smpl_presence_output = gr.Markdown("", elem_classes=["error-log", "smpl-presence-log"])

        with gr.Column(scale=4):
            log_output = gr.Markdown("", elem_classes=["custom-log"], visible=False)
            per_person_loss_output = gr.Markdown("", elem_classes=["error-log", "per-person-loss-log"])
            with gr.Tabs():
                with gr.Tab("🔧 Reconstruction"):
                    reconstruction_output = gr.Model3D(
                        label="Scene Reconstruction",
                        height=600,
                        zoom_speed=0.5,
                        pan_speed=0.5,
                    )
                with gr.Tab("🔥 Attention Analysis"):
                    gr.Markdown(
                        "Per-person query across all views. **Projected Mesh**: where the predicted "
                        "SMPL body lands on each processed image. **Attention vs SMPL**: attention "
                        "heatmap + reprojected mesh outline + centroid gap (agreement between where the "
                        "query looks and where it places the body)."
                    )
                    projection_gallery = gr.Gallery(
                        label="Projected Mesh on Processed Images",
                        columns=4,
                        height=680,
                        buttons=["download"],
                        object_fit="contain",
                        preview=True,
                    )
                    attention_smpl_gallery = gr.Gallery(
                        label="Attention vs Predicted SMPL (heatmap + reprojected mesh outline + centroid gap)",
                        columns=1,
                        height=680,
                        buttons=["download"],
                        object_fit="contain",
                        preview=True,
                    )
                with gr.Tab("📍 Landmark / Mask"):
                    landmark_mask_gallery = gr.Gallery(
                        label="512 Landmarks and Predicted Person Masks",
                        columns=4,
                        height=680,
                        buttons=["download"],
                        object_fit="contain",
                        preview=True,
                    )

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [
                        image_ids,
                        reconstruction_output,
                        log_output,
                        error_output,
                        smpl_presence_output,
                        per_person_loss_output,
                        target_dir_output,
                        smpl_obj_state,
                        image_gallery,
                        projection_gallery,
                        attention_smpl_gallery,
                        landmark_mask_gallery,
                    ],
                    scale=1,
                )

    # ---------------------- Reconstruct button ----------------------
    submit_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[
            reconstruction_output,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
        ],
    ).then(
        fn=update_log,
        inputs=[],
        outputs=[log_output],
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            mask_green_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            input_image_dir,
            image_ids,
            use_gt,
            smpl_presence_threshold,
            use_hungarian_smpl_mesh,
            save_outputs,
            overlay_gt_mesh,
            frame_dropdown,
        ],
        outputs=[
            reconstruction_output,
            log_output,
            smpl_obj_state,
            target_dir_output,
            image_gallery,
            projection_gallery,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    ).then(
        fn=run_projection_gallery,
        inputs=[target_dir_output],
        outputs=[projection_gallery],
    ).then(
        fn=run_attention_smpl_gallery,
        inputs=[target_dir_output],
        outputs=[attention_smpl_gallery],
    ).then(
        fn=run_landmark_mask_gallery,
        inputs=[target_dir_output],
        outputs=[landmark_mask_gallery],
    )

    # ---------------------- Dataset selection 更新時更新 gallery + 清空 SMPL ----------------------
    dataset_root.change(
        fn=update_dataset_split,
        inputs=[dataset_root, dataset_split, image_ids, save_outputs],
        outputs=[
            dataset_run,
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )
    dataset_split.change(
        fn=update_dataset_split,
        inputs=[dataset_root, dataset_split, image_ids, save_outputs],
        outputs=[
            dataset_run,
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )
    dataset_run.change(
        fn=update_gallery_for_dataset_selection,
        inputs=[dataset_root, dataset_split, dataset_run, image_ids, save_outputs],
        outputs=[
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )
    image_ids.change(
        fn=update_gallery_for_dataset_selection,
        inputs=[dataset_root, dataset_split, dataset_run, image_ids, save_outputs],
        outputs=[
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )
    save_outputs.change(
        fn=update_gallery_for_dataset_selection,
        inputs=[dataset_root, dataset_split, dataset_run, image_ids, save_outputs],
        outputs=[
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )

    # Repopulate the Frame dropdown when the selected run changes (also fires when a
    # dataset-root/split change updates the run value, so its frames stay in sync).
    dataset_run.change(
        fn=populate_frame_dropdown,
        inputs=[dataset_root, dataset_split, dataset_run],
        outputs=[frame_dropdown],
    )
    # Refresh the preview when the chosen frame changes.
    frame_dropdown.change(
        fn=update_gallery_for_dataset_selection,
        inputs=[dataset_root, dataset_split, dataset_run, image_ids, save_outputs, frame_dropdown],
        outputs=[
            input_image_dir,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            projection_gallery,
            attention_smpl_gallery,
            landmark_mask_gallery,
            log_output,
            smpl_obj_state,
            error_output,
            smpl_presence_output,
            per_person_loss_output,
        ],
    )

    server_port = find_available_port(6815)
    print(f"Launching Gradio on available port {server_port}")
    demo.queue(max_size=20).launch(show_error=True, share=True, server_port=server_port)
