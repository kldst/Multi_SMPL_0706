# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from .general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_extrinsics: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_extrinsics:
        new_cam_points = cam_points.clone() if cam_points is not None else None
        new_depths = depths.clone() if depths is not None else None

        # Estimate scale from camera positions derived from extrinsics
        R_rel = new_extrinsics[:, :, :3, :3]
        t_rel = new_extrinsics[:, :, :3, 3]
        cam_centers = -torch.matmul(R_rel.transpose(-1, -2), t_rel.unsqueeze(-1)).squeeze(-1)

        if cam_centers.shape[1] > 1:
            center_norm = cam_centers[:, 1:].norm(dim=-1)
            avg_scale = center_norm.mean(dim=1)
        else:
            avg_scale = torch.ones(B, device=device)

        avg_scale = avg_scale.clamp(min=1e-6, max=1e6)

        if new_world_points is not None:
            new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if new_depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if new_cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    return new_extrinsics, new_cam_points, new_world_points, new_depths


def normalize_camera_extrinsics_points_and_3djoints_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    joints3d_world: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_extrinsics: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Normalize camera extrinsics and all associated 3D data, including joints.

    Args:
        extrinsics: (B, S, 3, 4) 相機外參 (world->cam, OpenCV)
        cam_points: (B, S, H, W, 3) 或其他 shape 的相機座標點
        world_points: (B, S, H, W, 3) 或其他 shape 的世界座標點
        joints3d_world: (B, S, J, 3) 世界座標的 3D joints
        depths: (B, S, H, W) 深度圖
        scale_by_extrinsics: 是否依據相機中心距離做尺度正規化
        point_masks: 目前未使用，保留介面以後擴充

    Returns:
        new_extrinsics: (B, S, 3, 4) normalized 外參
        new_cam_points: 同輸入 shape 的 normalized cam_points
        new_world_points: 同輸入 shape 的 normalized world_points
        new_joints3d_world: (B, S, J, 3) normalized joints
        new_depths: 同輸入 shape 的 normalized depths
    """

    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(joints3d_world, "joints3d_world")
    check_valid_tensor(depths, "depths")

    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")

    # 轉成齊次座標 (B, S, 4, 4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # 第一個相機外參的逆 (cam->world)，用來把所有相機變到 first-cam frame
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B, S, 4, 4)

    # 將 world_points 轉到第一個相機座標系
    if world_points is not None:
        R = extrinsics[:, 0, :3, :3]  # (B, 3, 3)
        t = extrinsics[:, 0, :3, 3]   # (B, 3)
        new_world_points = (
            world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
        ) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    # 將 joints3d_world 也轉到第一個相機座標系
    if joints3d_world is not None:
        R = extrinsics[:, 0, :3, :3]  # (B, 3, 3)
        t = extrinsics[:, 0, :3, 3]   # (B, 3)
        # joints3d_world can be (B, S, J, 3), (B, P, J, 3), or
        # (B, S, P, J, 3). Keep all middle axes and transform only xyz.
        new_joints3d_world = torch.einsum(
            "b...c,bdc->b...d",
            joints3d_world,
            R,
        )
        t_view = (B,) + (1,) * (new_joints3d_world.dim() - 2) + (3,)
        new_joints3d_world = new_joints3d_world + t.view(*t_view)
    else:
        new_joints3d_world = None

    if scale_by_extrinsics:
        new_cam_points = cam_points.clone() if cam_points is not None else None
        new_depths = depths.clone() if depths is not None else None

        # 根據 normalized extrinsics 推出相機中心，再估計平均尺度
        R_rel = new_extrinsics[:, :, :3, :3]
        t_rel = new_extrinsics[:, :, :3, 3]
        cam_centers = -torch.matmul(R_rel.transpose(-1, -2), t_rel.unsqueeze(-1)).squeeze(-1)

        if cam_centers.shape[1] > 1:
            center_norm = cam_centers[:, 1:].norm(dim=-1)
            avg_scale = center_norm.mean(dim=1)
        else:
            avg_scale = torch.ones(B, device=device)

        avg_scale = avg_scale.clamp(min=1e-6, max=1e6)

        if new_world_points is not None:
            new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        if new_joints3d_world is not None:
            joints_scale_view = (-1,) + (1,) * (new_joints3d_world.dim() - 1)
            new_joints3d_world = new_joints3d_world / avg_scale.view(*joints_scale_view)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if new_depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if new_cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        avg_scale = torch.ones(B, device=device)
        new_extrinsics = new_extrinsics[:, :, :3]
        new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
        new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
        new_joints3d_world = check_and_fix_inf_nan(new_joints3d_world, "new_joints3d_world", hard_max=None)
        return new_extrinsics, cam_points, new_world_points, new_joints3d_world, depths, avg_scale

    new_extrinsics = new_extrinsics[:, :, :3]
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_joints3d_world = check_and_fix_inf_nan(new_joints3d_world, "new_joints3d_world", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_joints3d_world, new_depths, avg_scale

#* gt:      smpl 3d joint -> unity 3d joint -> ex, in -> 2d joint
#* pred:    normalize ex, in 
#* pred:    pose, beta -> smpl 3d joint -> unity 3d joint -> normalzie -> ex, in -> 2d joint
