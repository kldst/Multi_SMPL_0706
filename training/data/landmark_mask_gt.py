"""Ground-truth derivation for the dense-landmark and person-mask heads.

Both new heads are supervised with targets that are *derived* from data the raw
Mamma_mv_split format already ships -- nothing new needs to be annotated:

* **512 dense landmarks** come from the MAMMA ``verts_512.pkl`` down-sampling
  matrix ``M`` (shape ``(512, 10475)``).  Each of the 512 landmarks is a fixed
  linear combination of the 10475 SMPL-X vertices, so a landmark's 3D / 2D
  position and visibility are just ``M @ vertices`` / ``M @ visibility``.
* **Per-person masks** come from the per-view instance mask ``*.mask.jpg``,
  whose pixel value equals ``person_idx + 1`` (0 = background).

These helpers are numpy-based and live on the data side; the loss never needs
``M`` because the dataset hands it pre-computed GT tensors.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Bundled location of the MAMMA 512-vertex down-sampling matrix. The repo-local
# copy (training/data/assets/verts_512.pkl) is tried FIRST so training is
# self-contained; the external MAMMA paths remain as fallbacks.
DEFAULT_VERTS512_PATHS = (
    str(Path(__file__).resolve().parent / "assets" / "verts_512.pkl"),
    "/mnt/train-data-4-hdd/yian/mamma/data/body_models/downsampled_verts/verts_512.pkl",
    "/mnt/train-data-4-hdd/yian/yian_vggt_smpl/data/body_models/downsampled_verts/verts_512.pkl",
)

NUM_LANDMARKS = 512
NUM_SMPLX_VERTS = 10475


@lru_cache(maxsize=4)
def load_verts512_matrix(path: str | None = None) -> np.ndarray:
    """Load the ``(512, 10475)`` down-sampling matrix as float32 (cached).

    The pickle stores a torch tensor, so we import torch lazily only to unpickle
    it and immediately drop to numpy -- the rest of the data pipeline is numpy.
    """
    candidates = [path] if path else list(DEFAULT_VERTS512_PATHS)
    for cand in candidates:
        if cand and Path(cand).is_file():
            with open(cand, "rb") as f:
                mat = pickle.load(f)
            if hasattr(mat, "detach"):  # torch.Tensor
                mat = mat.detach().cpu().numpy()
            mat = np.asarray(mat, dtype=np.float32)
            if mat.shape != (NUM_LANDMARKS, NUM_SMPLX_VERTS):
                raise ValueError(
                    f"verts_512 matrix has shape {mat.shape}, expected "
                    f"{(NUM_LANDMARKS, NUM_SMPLX_VERTS)} (from {cand})"
                )
            return mat
    raise FileNotFoundError(
        f"verts_512.pkl not found in any of: {candidates}"
    )


def downsample_vertices(matrix: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Reduce ``(..., 10475, D)`` vertices to ``(..., 512, D)`` via ``M @ v``."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape[-2] != NUM_SMPLX_VERTS:
        raise ValueError(
            f"expected vertices with {NUM_SMPLX_VERTS} rows, got {vertices.shape}"
        )
    # matrix: (512, 10475); vertices: (..., 10475, D) -> (..., 512, D)
    return np.einsum("lv,...vd->...ld", matrix, vertices).astype(np.float32)


def downsample_visibility(
    matrix: np.ndarray, visibility: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Reduce per-vertex visibility ``(..., 10475)`` to a ``(..., 512)`` mask.

    ``M`` rows are (approximately) partitions of unity, so ``M @ vis`` is the
    coverage-weighted visible fraction of each landmark's supporting vertices;
    we threshold it back to a {0,1} landmark-visibility target.
    """
    visibility = np.asarray(visibility, dtype=np.float32).reshape(*visibility.shape)
    if visibility.shape[-1] != NUM_SMPLX_VERTS:
        raise ValueError(
            f"expected visibility with {NUM_SMPLX_VERTS} cols, got {visibility.shape}"
        )
    soft = np.einsum("lv,...v->...l", matrix, visibility)
    row_sum = matrix.sum(axis=1).clip(min=1e-6)  # (512,)
    soft = soft / row_sum
    return (soft >= threshold).astype(np.float32)


def rasterize_person_patch_mask(
    mask_image: np.ndarray,
    person_value: int,
    patch_h: int,
    patch_w: int,
    image_size: int | None = None,
) -> np.ndarray:
    """Turn a per-view instance mask into a per-person patch-grid occupancy map.

    Args:
        mask_image: ``(H, W)`` instance mask, pixel value == person_idx + 1.
        person_value: the instance value for this person (``person_idx + 1``).
        patch_h, patch_w: output patch-grid resolution (e.g. 518//14 = 37).
        image_size: if given, the mask is first resized (nearest) to a square of
            this size so it matches the network's resized RGB input.

    Returns:
        ``(patch_h, patch_w)`` float32 occupancy in [0, 1] -- the fraction of
        pixels inside each patch cell that belong to this person.
    """
    mask = np.asarray(mask_image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if image_size is not None:
        mask = cv2.resize(
            mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
        )
    binary = (mask == int(person_value)).astype(np.float32)
    # Area-average pool to the patch grid -> soft occupancy per patch.
    occupancy = cv2.resize(
        binary, (patch_w, patch_h), interpolation=cv2.INTER_AREA
    )
    return occupancy.astype(np.float32)


def landmarks3d_world_from_vertices(
    matrix: np.ndarray, vertices_world: np.ndarray
) -> np.ndarray:
    """Convenience: ``(P, 10475, 3)`` world verts -> ``(P, 512, 3)`` landmarks."""
    return downsample_vertices(matrix, vertices_world)
