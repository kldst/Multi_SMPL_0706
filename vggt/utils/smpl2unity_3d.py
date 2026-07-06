from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch


JOINT_COUNT = 24

# SMPL space -> Unity space
#
# This is a fixed axis convention change:
#   x' =  x
#   y' =  z
#   z' = -y
#
# For points:  p_unity = R_X90 @ p_smpl + t
# For a rotation matrix (e.g. root orientation): R_unity = R_X90 @ R_smpl
R_X90 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)

# A default pelvis target (Unity/world) used as fallback.
# Prefer to override this per-frame/per-sequence when possible.
T_UNITY = np.array(
    [-0.00288206385448575, 1.0212889909744263, 0.23515743017196655],
    dtype=np.float32,
)


def _as_numpy_3vec(x: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (3,):
        raise ValueError(f"pelvis_target must be shape (3,), got {x.shape}")
    return x


def smpl_to_unity_points_np(
    points_smpl: np.ndarray,
    *,
    pelvis_target: Optional[np.ndarray] = None,
    pelvis_index: int = 0,
    return_offset: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Convert SMPL-space 3D points to Unity-space.

    Supports shapes:
    - (J, 3)
    - (B, J, 3)
    - (..., 3)

    The translation is chosen so that the pelvis point (by pelvis_index) maps
    to pelvis_target after applying the axis conversion rotation.
    """
    pts = np.asarray(points_smpl, dtype=np.float32)
    if pts.shape[-1] != 3:
        raise ValueError(f"points_smpl must end with dim 3, got {pts.shape}")

    target = _as_numpy_3vec(pelvis_target) if pelvis_target is not None else T_UNITY

    # pelvis in SMPL space
    pelvis = pts.reshape(-1, 3)[pelvis_index]
    pelvis_rot = R_X90 @ pelvis
    offset = (target - pelvis_rot).astype(np.float32)

    if pts.ndim == 2:
        pts_unity = (R_X90 @ pts.T).T + offset
    else:
        # batched / arbitrary leading dims
        pts_unity = np.einsum("ij,...j->...i", R_X90, pts) + offset

    if return_offset:
        return pts_unity, offset
    return pts_unity


def smpl_to_unity_points_torch(
    points_smpl: torch.Tensor,
    pelvis_target: Optional[torch.Tensor] = None,
    pelvis_index: int = 0,
    return_offset: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Torch version of SMPL->Unity conversion for 3D points.

    points_smpl: (..., 3)
    pelvis_target:
      - None: uses T_UNITY constant
      - (3,) or (B,3) (if you pass a batch; broadcasting is supported)
    """
    if points_smpl.shape[-1] != 3:
        raise ValueError(f"points_smpl must end with dim 3, got {tuple(points_smpl.shape)}")

    R = torch.tensor(R_X90, device=points_smpl.device, dtype=points_smpl.dtype)

    if pelvis_target is None:
        target = torch.tensor(T_UNITY, device=points_smpl.device, dtype=points_smpl.dtype)
    else:
        target = pelvis_target.to(device=points_smpl.device, dtype=points_smpl.dtype)
        if target.shape not in [(3,), (points_smpl.shape[0], 3)]:
            # allow (3,) or (B,3) for common (B,J,3) inputs
            raise ValueError(
                "pelvis_target must be shape (3,) or (B,3); "
                f"got {tuple(target.shape)} for points {tuple(points_smpl.shape)}"
            )

    pelvis = points_smpl.reshape(-1, 3)[pelvis_index]
    pelvis_rot = R @ pelvis
    offset = target - pelvis_rot

    # pts_unity = torch.einsum("ij,...j->...i", R, points_smpl) + offset

    fixed_shift = torch.tensor([0.0, 1.0, 0.0], device=points_smpl.device, dtype=points_smpl.dtype)
    pts_unity = torch.einsum("ij,...j->...i", R, points_smpl) + fixed_shift
    
    if return_offset:
        return pts_unity, offset
    return pts_unity


def smpl_to_unity_root_rotmat_torch(root_rotmat_smpl: torch.Tensor) -> torch.Tensor:
    """Convert a SMPL root rotation matrix to Unity frame.

    root_rotmat_smpl: (..., 3, 3)
    Returns: (..., 3, 3)
    """
    if root_rotmat_smpl.shape[-2:] != (3, 3):
        raise ValueError(
            f"root_rotmat_smpl must have shape (...,3,3), got {tuple(root_rotmat_smpl.shape)}"
        )
    R = torch.tensor(R_X90, device=root_rotmat_smpl.device, dtype=root_rotmat_smpl.dtype)
    return R @ root_rotmat_smpl


__all__ = [
    "JOINT_COUNT",
    "R_X90",
    "T_UNITY",
    "smpl_to_unity_points_np",
    "smpl_to_unity_points_torch",
    "smpl_to_unity_root_rotmat_torch",
]
