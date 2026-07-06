import argparse
import inspect
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import trimesh

for _name, _value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if not hasattr(np, "unicode"):
    np.unicode = str
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from training.train_utils.normalization import normalize_camera_extrinsics_points_and_3djoints_batch
from training.loss import _decode_smpl_batch, _normalize_gender_string


DEFAULT_MAMMA_ROOT = Path("/mnt/train-data-4-hdd/yian/Mamma_mv_split/train")
DEFAULT_SCENE_ROOT = Path(
    "tmp/bedlam_lab_20251031_191436/harmony4d_train_1_NC_200_00"
)
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
    parser.add_argument("--require-visible-joints", action="store_true")
    parser.add_argument("--min-visible-joints", type=int, default=8)
    return parser


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


def gender_id(gender) -> int:
    text = normalize_gender(gender)
    return {"male": 0, "female": 1, "neutral": 2}[text]


def scene_search_root(mamma_root: Path, scene_root: Path) -> Path:
    scene_root = Path(scene_root)
    return scene_root if scene_root.is_absolute() else Path(mamma_root) / scene_root


def is_sequence_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for view_dir in path.iterdir():
        if view_dir.is_dir() and any(view_dir.glob("*.data.pyd")):
            return True
    return False


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
            raise FileNotFoundError(f"Sequence {seq_name!r} not found under {scene_search_root(mamma_root, scene_root)}")
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
        raise RuntimeError(f"No frames with at least {num_views} views in {seq_dir}")
    if frame:
        frame = str(frame).zfill(4)
        if frame not in grouped:
            raise FileNotFoundError(f"Frame {frame} not found in {seq_dir}")
        views = grouped[frame]
        if len(views) < num_views:
            raise RuntimeError(f"Frame {frame} has {len(views)} views, need {num_views}")
    else:
        frame, views = rng.choice(candidates)
    return frame, sorted(rng.sample(views, num_views))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image_and_intrinsics(image: np.ndarray, K: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    out = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    K_new = np.asarray(K, dtype=np.float32).copy()
    K_new[0, :] *= float(size) / float(w)
    K_new[1, :] *= float(size) / float(h)
    scale = np.array([float(size) / float(w), float(size) / float(h)], dtype=np.float32)
    return out, K_new, scale


def project_points(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    E = np.asarray(extrinsic, dtype=np.float64).reshape(3, 4)
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    cam = pts @ E[:3, :3].T + E[:3, 3]
    z = cam[:, 2].copy()
    z_safe = np.where(np.abs(z) < eps, eps, z)
    pix_h = cam @ K.T
    pix = pix_h[:, :2] / z_safe[:, None]
    return pix.astype(np.float32), z.astype(np.float32)


def decode_mamma_smplx(
    pose: np.ndarray,
    beta: np.ndarray,
    trans: np.ndarray,
    gender: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pose_t = torch.as_tensor(np.asarray(pose, dtype=np.float32).reshape(1, -1)[:, :72])
    beta_t = torch.as_tensor(np.asarray(beta, dtype=np.float32).reshape(1, -1)[:, :10])
    trans_t = torch.as_tensor(np.asarray(trans, dtype=np.float32).reshape(1, 3))
    gender_key = _normalize_gender_string(gender)
    with torch.no_grad():
        joints, vertices = _decode_smpl_batch(pose_t, beta_t, trans_t, [gender_key], use_mamma=True)
    return (
        joints[0, :24].detach().cpu().numpy().astype(np.float32),
        vertices[0].detach().cpu().numpy().astype(np.float32),
    )


def orthonormalize_extrinsics(extrinsics: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    extr = np.asarray(extrinsics, dtype=np.float64).copy()
    flat = extr.reshape(-1, 3, 4)
    for E in flat:
        R = E[:3, :3]
        norms = np.linalg.norm(R, axis=0)
        scale = float(np.mean(norms[norms > eps])) if np.any(norms > eps) else 1.0
        E[:3, :3] = R / scale
        E[:3, 3] = E[:3, 3] / scale
    return flat.reshape(extr.shape).astype(np.float32)


def draw_points(image: np.ndarray, points: np.ndarray, color=(255, 40, 40), radius: int = 3) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for x, y in np.asarray(points).reshape(-1, 2):
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_projected_vertices(image: np.ndarray, vertices: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, color=(40, 180, 255)) -> np.ndarray:
    pts, depth = project_points(vertices, extrinsic, intrinsic)
    valid = depth > 1e-4
    if valid.sum() > 1200:
        valid_idx = np.flatnonzero(valid)
        valid_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, 1200).astype(np.int64)]
        mask = np.zeros_like(valid)
        mask[valid_idx] = True
        valid = mask
    return draw_points(image, pts[valid], color=color, radius=1)


def draw_points_on_diagnostic_canvas(
    point_sets: List[Tuple[str, np.ndarray, Tuple[int, int, int]]],
    size: int = 700,
    margin: int = 40,
) -> np.ndarray:
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    valid_sets = []
    all_points = []
    for label, points, color in point_sets:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        finite = np.isfinite(pts).all(axis=-1)
        pts = pts[finite]
        if pts.size:
            valid_sets.append((label, pts, color))
            all_points.append(pts)
    if not all_points:
        cv2.putText(canvas, "no finite 2D points", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        return canvas

    all_points = np.concatenate(all_points, axis=0)
    lo = np.nanmin(all_points, axis=0)
    hi = np.nanmax(all_points, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = float(size - 2 * margin) / float(np.max(span))
    offset = np.array([margin, margin], dtype=np.float32) - lo * scale
    for label, pts, color in valid_sets:
        pts_canvas = pts * scale + offset
        for x, y in pts_canvas:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"diagnostic fit: x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}]",
        (16, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def joint_visibility_mask(
    joints2d: np.ndarray,
    joints3d_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
) -> np.ndarray:
    joints2d = np.asarray(joints2d, dtype=np.float32)
    finite = np.isfinite(joints2d).all(axis=-1)
    in_frame = (
        (joints2d[..., 0] >= 0)
        & (joints2d[..., 0] < image_size)
        & (joints2d[..., 1] >= 0)
        & (joints2d[..., 1] < image_size)
    )
    return (finite & in_frame).astype(np.float32)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def load_smplx_faces(gender: str = "neutral") -> np.ndarray:
    model_path = REPO_DIR / "smplx_models" / normalize_gender(gender) / "model.pkl"
    data = load_pickle(model_path)
    faces = data.get("f", data.get("faces"))
    if faces is None:
        raise KeyError(f"No faces/f key in {model_path}")
    return np.asarray(faces, dtype=np.int64)


def make_scene(vertices: np.ndarray, faces: np.ndarray, extrinsics: np.ndarray, out_path: Path) -> None:
    scene = trimesh.Scene()
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices).reshape(-1, 3), faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(np.array([60, 180, 255, 210], dtype=np.uint8), (len(mesh.vertices), 1))
    scene.add_geometry(mesh, node_name="smplx_gt")
    for i, E in enumerate(np.asarray(extrinsics).reshape(-1, 3, 4)):
        T = np.eye(4)
        T[:3, :4] = E
        cam_to_world = np.linalg.inv(T)
        axis = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
        axis.apply_transform(cam_to_world)
        scene.add_geometry(axis, node_name=f"camera_{i:02d}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))


class RawMammaMultiViewDataset:
    def __init__(
        self,
        mamma_root: Path = DEFAULT_MAMMA_ROOT,
        scene_root: Path = DEFAULT_SCENE_ROOT,
        num_views: int = 4,
        image_size: int = IMAGE_SIZE,
        seed: int = 7,
        seq_name: str = "",
        frame: str = "",
        require_visible_joints: bool = False,
        min_visible_joints: int = 8,
    ):
        self.mamma_root = Path(mamma_root)
        self.scene_root = Path(scene_root)
        self.num_views = int(num_views)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.seq_name = seq_name
        self.frame = frame
        self.require_visible_joints = bool(require_visible_joints)
        self.min_visible_joints = int(min_visible_joints)

    def sample(self) -> Dict:
        rng = random.Random(self.seed)
        last_batch = None
        for _ in range(200):
            seq_dir = choose_sequence(self.mamma_root, self.scene_root, self.seq_name, rng)
            frame, views = choose_frame_and_views(seq_dir, self.frame, self.num_views, rng)
            batch = self.load(seq_dir, frame, views)
            last_batch = batch
            visible_count = int(batch["smpl_joints2d_confidence"].sum().item())
            if not self.require_visible_joints or visible_count >= self.min_visible_joints:
                return batch
            if self.seq_name and self.frame:
                break
        if last_batch is None:
            raise RuntimeError("No raw Mamma batch could be sampled")
        return last_batch

    def load(self, seq_dir: Path, frame: str, views: List[str]) -> Dict:
        first_people = load_pickle(seq_dir / views[0] / f"{frame}.data.pyd")
        person_ids = sorted(first_people.keys(), key=lambda x: int(x))
        people = [first_people[pid] for pid in person_ids]
        P = len(people)

        pose = np.stack([np.asarray(p["pose_world"], dtype=np.float32).reshape(-1)[:72] for p in people], axis=0)
        beta = np.stack([np.asarray(p["shape"], dtype=np.float32).reshape(-1)[:10] for p in people], axis=0)
        trans = np.stack([np.asarray(p["trans_world"], dtype=np.float32).reshape(-1)[:3] for p in people], axis=0)
        genders = np.asarray([gender_id(p.get("gender", "neutral")) for p in people], dtype=np.int64)
        has_smpl = np.ones((P,), dtype=np.float32)
        decoded_joints_world, decoded_vertices_world = [], []
        for person in people:
            joints_w, verts_w = decode_mamma_smplx(
                np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)[:72],
                np.asarray(person["shape"], dtype=np.float32).reshape(-1)[:10],
                np.asarray(person["trans_world"], dtype=np.float32).reshape(-1)[:3],
                normalize_gender(person.get("gender", "neutral")),
            )
            decoded_joints_world.append(joints_w)
            decoded_vertices_world.append(verts_w)

        images, extrinsics, intrinsics, joints2d, joints3d, joint_conf, vertices = [], [], [], [], [], [], []
        image_paths, original_sizes = [], []
        for view in views:
            data_path = seq_dir / view / f"{frame}.data.pyd"
            frame_people = load_pickle(data_path)
            cam_person = next(iter(frame_people.values()))
            image_path = seq_dir / view / f"{frame}.jpg"
            if not image_path.is_file():
                image_path = seq_dir / view / f"{frame}.png"
            image = read_rgb(image_path)
            original_sizes.append(np.asarray(image.shape[:2], dtype=np.int64))
            image_paths.append(str(image_path))
            K = np.asarray(cam_person["cam_int"], dtype=np.float32).reshape(3, 3)
            E = np.asarray(cam_person["cam_ext"], dtype=np.float32)
            E = E[:3, :4] if E.shape == (4, 4) else E.reshape(3, 4)
            E = orthonormalize_extrinsics(E)
            image_resized, K_resized, scale = preprocess_image_and_intrinsics(image, K, self.image_size)

            view_j2d, view_j3d, view_conf, view_verts = [], [], [], []
            for pid in person_ids:
                p = frame_people[pid]
                person_idx = person_ids.index(pid)
                j3 = decoded_joints_world[person_idx]
                verts = decoded_vertices_world[person_idx]
                j2, _ = project_points(j3, E, K_resized)
                conf = joint_visibility_mask(j2, j3, E, K_resized, self.image_size)
                view_j2d.append(j2)
                view_j3d.append(j3)
                view_conf.append(conf)
                view_verts.append(verts)

            images.append(image_resized)
            extrinsics.append(E)
            intrinsics.append(K_resized)
            joints2d.append(np.stack(view_j2d, axis=0))
            joints3d.append(np.stack(view_j3d, axis=0))
            joint_conf.append(np.stack(view_conf, axis=0))
            vertices.append(view_verts)

        images_np = np.stack(images, axis=0)
        batch = {
            "seq_name": f"mamma_raw_{seq_dir.name}_frame_{frame}",
            "seq_dir": str(seq_dir),
            "frame": frame,
            "views": views,
            "image_paths": image_paths,
            "original_sizes": np.stack(original_sizes, axis=0),
            "images_np": images_np,
            "images": torch.from_numpy(images_np).permute(0, 3, 1, 2).float().div(255.0).unsqueeze(0),
            "extrinsics": torch.from_numpy(np.stack(extrinsics, axis=0)).float().unsqueeze(0),
            "intrinsics": torch.from_numpy(np.stack(intrinsics, axis=0)).float().unsqueeze(0),
            "smpl_pose": torch.from_numpy(pose).float().unsqueeze(0),
            "smpl_beta": torch.from_numpy(beta).float().unsqueeze(0),
            "smpl_trans": torch.from_numpy(trans).float().unsqueeze(0),
            "smpl_joints2d": torch.from_numpy(np.stack(joints2d, axis=0)).float().unsqueeze(0),
            "smpl_joints3d_world": torch.from_numpy(np.stack(joints3d, axis=0)).float().unsqueeze(0),
            "smpl_gender": torch.from_numpy(genders).long().unsqueeze(0),
            "has_smpl": torch.from_numpy(has_smpl).float().unsqueeze(0),
            "num_people": torch.asarray([P], dtype=torch.long),
            "person_keys": [f"person_{int(pid):02d}" for pid in person_ids],
            "smpl_joints2d_confidence": torch.from_numpy(np.stack(joint_conf, axis=0)).float().unsqueeze(0),
            "gt_vertices_world": vertices,
        }
        return batch


def process_batch_like_trainer(batch: Dict, scale_by_extrinsics: bool = True) -> Dict:
    out = dict(batch)
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_joints3d_world, normalized_depths, avg_scale = (
        normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=out["extrinsics"],
            cam_points=out.get("cam_points"),
            world_points=out.get("world_points"),
            joints3d_world=out.get("smpl_joints3d_world"),
            depths=out.get("depths"),
            scale_by_extrinsics=scale_by_extrinsics,
            point_masks=out.get("point_masks"),
        )
    )
    out["raw_extrinsics"] = out["extrinsics"].clone()
    out["extrinsics"] = normalized_extrinsics
    out["avg_scale"] = avg_scale
    if normalized_joints3d_world is not None:
        out["smpl_joints3d_world"] = normalized_joints3d_world
    if normalized_cam_points is not None:
        out["cam_points"] = normalized_cam_points
    if normalized_world_points is not None:
        out["world_points"] = normalized_world_points
    if normalized_depths is not None:
        out["depths"] = normalized_depths
    return out


def tensor_summary(batch: Dict) -> Dict:
    summary = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            t = value.detach().cpu()
            item = {"shape": list(t.shape), "dtype": str(t.dtype)}
            if t.numel() and t.is_floating_point():
                finite = torch.isfinite(t)
                finite_values = t[finite]
                item.update(
                    min=float(finite_values.min()) if finite_values.numel() else None,
                    max=float(finite_values.max()) if finite_values.numel() else None,
                    finite=float(finite.float().mean()),
                )
            summary[key] = item
        elif isinstance(value, np.ndarray):
            item = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.size and np.issubdtype(value.dtype, np.number):
                item.update(
                    min=float(np.nanmin(value)),
                    max=float(np.nanmax(value)),
                    finite=float(np.isfinite(value).mean()),
                )
            summary[key] = item
        elif key in {"seq_name", "seq_dir", "frame", "views", "image_paths", "person_keys"}:
            summary[key] = value
    return summary


def write_summary(path: Path, batch: Dict, extra: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensor_summary(batch)
    if extra:
        data["extra"] = extra
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
