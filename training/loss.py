# Copyright (c) Meta Platforms, Inc. and affiliates.
# Multi-task loss aggregator. The heavy loss code now lives in three modules:
#   * loss_camera.py  -- camera pose loss
#   * loss_smpl.py    -- SMPL pose/shape/joints/vertices + dense-landmark loss
#   * loss_mask.py    -- per-person mask loss
# This file keeps ``MultitaskLoss`` and re-exports the public names so existing
# ``from training.loss import ...`` call sites keep working unchanged.

from dataclasses import dataclass

import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

from training.loss_camera import compute_camera_loss, camera_loss_single
from training.loss_smpl import *  # noqa: F401,F403
from training.loss_smpl import (
    _decode_smpl_batch,
    _decode_smplx_batch,
    _normalize_gender_string,
    _resolve_batch_genders,
    _project_points_opencv,
    _TorchSMPLX,
    apply_hungarian_matching,
    compute_smpl_loss,
    compute_landmark_loss,
    compute_gt_mesh_translate,
    compute_gt_mesh_rot,
    normalize_joints_world_to_batch_gauge,
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
)
from training.loss_mask import compute_mask_loss


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - SMPL loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, smpl=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.smpl = smpl

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # for k, v in batch.items():
        #     if torch.is_tensor(v):
        #         print(f"  {k}: shape={tuple(v)}, dtype={v}, device={v.device}")
        #     else:
        #         print(f"  {k}: type={type(v)}, value={v}")
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # SMPL pose / shape / joints loss
        if self.smpl is not None and "smpl_pose" in predictions:
            smpl_loss_dict = compute_smpl_loss(predictions, batch, **self.smpl)
            smpl_loss = smpl_loss_dict["loss_smpl"] * self.smpl.get("weight", 1.0)
            total_loss = total_loss + smpl_loss
            loss_dict.update(smpl_loss_dict)

        loss_dict["objective"] = total_loss
        loss_dict["loss_objective"] = total_loss

        return loss_dict


