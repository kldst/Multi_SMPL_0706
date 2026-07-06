# Replacement for the (missing-from-bundle) pytorch3d-based projection worker.
#
# Renders the combined SMPL mesh from predictions.npz onto each processed_images
# frame using a plain CPU triangle rasterizer (numpy + OpenCV), reusing
# vggt.utils.geometry.project_world_points_to_cam so the projection matches the
# exact camera convention used everywhere else in this repo. No GPU/pytorch3d
# rendering dependency required.
import glob
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from vggt.utils.geometry import project_world_points_to_cam

NEAR_CLIP = 1e-3
OVERLAY_ALPHA = 0.6
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


def _list_processed_images(processed_dir: str) -> list[str]:
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(processed_dir, f"*{ext}")))
    return sorted(paths)


def _project_mesh_to_frames(
    vertices: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image_points[S,V,2], depth[S,V]) for the given world/gauge-space vertices."""
    device = torch.device("cpu")
    verts_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).to(device)
    extr_t = torch.from_numpy(np.asarray(extrinsic, dtype=np.float32)).to(device)
    intr_t = torch.from_numpy(np.asarray(intrinsic, dtype=np.float32)).to(device)
    with torch.no_grad():
        image_points, cam_points = project_world_points_to_cam(verts_t, extr_t, intr_t)
    depth = cam_points[:, 2, :]  # (S, V)
    return image_points.cpu().numpy(), depth.cpu().numpy()


def _render_one_frame(
    image_path: str,
    out_path: str,
    pts2d: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    face_colors_bgr: np.ndarray,
) -> str:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    face_depth = depth[faces].mean(axis=1)  # (F,)
    valid = (depth[faces] > NEAR_CLIP).all(axis=1)
    order = np.argsort(-face_depth[valid])  # far -> near (painter's algorithm)
    valid_face_idx = np.flatnonzero(valid)[order]

    if valid_face_idx.size > 0:
        overlay = image.copy()
        tri_pts = pts2d[faces[valid_face_idx]].astype(np.int32)  # (F,3,2)
        colors = face_colors_bgr[valid_face_idx]
        for tri, color in zip(tri_pts, colors):
            cv2.fillConvexPoly(overlay, tri, color.tolist(), lineType=cv2.LINE_AA)
        image = cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1.0 - OVERLAY_ALPHA, 0)

    cv2.imwrite(out_path, image)
    return out_path


def project_mesh_from_base_dir(base_dir: str, workers: int = 1) -> list[str]:
    base_dir = str(base_dir)
    predictions_path = os.path.join(base_dir, "predictions.npz")
    processed_dir = os.path.join(base_dir, "processed_images")
    out_dir = os.path.join(base_dir, "projections_processed")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(predictions_path)
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(processed_dir)

    image_paths = _list_processed_images(processed_dir)
    if not image_paths:
        raise RuntimeError(f"No processed images found under {processed_dir}")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(predictions_path, allow_pickle=True)
    has_mesh = "smpl_vertices" in data.files and "smpl_faces" in data.files
    num_frames = len(image_paths)

    if not has_mesh:
        # Nothing to project (e.g. camera-only mode / no confident SMPL slots) -
        # fall back to passing the plain frames through so the gallery still shows something.
        print(f"[projection-cpu] No SMPL mesh in {predictions_path}; passing frames through unmodified.")
        out_paths = []
        for image_path in image_paths:
            out_path = os.path.join(out_dir, os.path.basename(image_path))
            shutil.copyfile(image_path, out_path)
            out_paths.append(out_path)
        return out_paths

    vertices = np.asarray(data["smpl_vertices"], dtype=np.float32)
    faces = np.asarray(data["smpl_faces"], dtype=np.int64)
    if "smpl_vertex_colors" in data.files and data["smpl_vertex_colors"] is not None:
        vertex_colors_rgba = np.asarray(data["smpl_vertex_colors"], dtype=np.uint8)
    else:
        vertex_colors_rgba = np.tile(np.array([[150, 150, 150, 255]], dtype=np.uint8), (vertices.shape[0], 1))
    # RGBA -> BGR for OpenCV, one color per face (all 3 verts of a face share the same
    # person's uniform color in this codebase, so the first vertex's color is exact).
    face_colors_bgr = vertex_colors_rgba[faces[:, 0]][:, [2, 1, 0]]

    extrinsic = np.asarray(data["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
    num_frames = min(num_frames, extrinsic.shape[0], intrinsic.shape[0])
    image_paths = image_paths[:num_frames]

    image_points, depth = _project_mesh_to_frames(vertices, extrinsic[:num_frames], intrinsic[:num_frames])

    def _task(i: int) -> str:
        image_path = image_paths[i]
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        return _render_one_frame(image_path, out_path, image_points[i], depth[i], faces, face_colors_bgr)

    max_workers = max(1, int(workers))
    if max_workers == 1 or num_frames <= 1:
        out_paths = [_task(i) for i in range(num_frames)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            out_paths = list(pool.map(_task, range(num_frames)))

    return sorted(out_paths)
