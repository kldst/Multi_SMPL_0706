"""Lightweight tensor-layout helpers for framewise temporal training."""

from typing import Mapping

import torch


def flatten_temporal_batch_for_framewise_model(batch: Mapping) -> Mapping:
    """Fold a time-major ``[B,T*V,...]`` clip into ``[B*T,V,...]`` frames."""
    if "temporal_num_frames" not in batch or "views_per_frame" not in batch:
        return batch

    temporal_values = batch["temporal_num_frames"].reshape(-1)
    view_values = batch["views_per_frame"].reshape(-1)
    if temporal_values.numel() == 0 or int(temporal_values[0].item()) <= 1:
        return batch
    if not torch.all(temporal_values == temporal_values[0]):
        raise ValueError("All clips in a batch must have the same temporal length")
    if not torch.all(view_values == view_values[0]):
        raise ValueError("All clips in a batch must have the same views_per_frame")

    B = int(batch["images"].shape[0])
    T = int(temporal_values[0].item())
    V = int(view_values[0].item())
    if int(batch["images"].shape[1]) != T * V:
        raise ValueError(
            f"Temporal images have S={batch['images'].shape[1]}, expected T*V={T*V}"
        )

    out = dict(batch)
    view_keys = {
        "images", "depths", "extrinsics", "intrinsics", "raw_extrinsics",
        "cam_points", "world_points", "point_masks", "original_sizes",
        "smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence",
        "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask",
        "smpl_contact", "smpl_floor_contact", "ids",
    }
    person_keys = {
        "smpl_pose", "smpl_beta", "smpl_trans", "smpl_gender", "has_smpl",
        "num_people", "frame_ids",
    }

    for key in view_keys:
        value = out.get(key)
        if torch.is_tensor(value) and value.shape[0] == B and value.shape[1] == T * V:
            out[key] = value.reshape(B, T, V, *value.shape[2:]).reshape(
                B * T, V, *value.shape[2:]
            )

    for key in person_keys:
        value = out.get(key)
        if torch.is_tensor(value) and value.shape[0] == B and value.shape[1] == T:
            out[key] = value.reshape(B * T, *value.shape[2:])

    avg_scale = out.get("avg_scale")
    if torch.is_tensor(avg_scale) and avg_scale.shape[0] == B:
        out["avg_scale"] = avg_scale.repeat_interleave(T, dim=0)

    for key in ("seq_name", "person_keys"):
        value = out.get(key)
        if isinstance(value, (list, tuple)) and len(value) == B:
            out[key] = [item for item in value for _ in range(T)]

    device = batch["images"].device
    out["temporal_shape"] = torch.tensor([B, T], device=device, dtype=torch.long)
    out["temporal_num_frames"] = torch.full(
        (B * T,), T, device=device, dtype=torch.long
    )
    out["views_per_frame"] = torch.full(
        (B * T,), V, device=device, dtype=torch.long
    )
    return out


__all__ = ["flatten_temporal_batch_for_framewise_model"]
