"""Hungarian matching for multi-person SMPL predictions (split out of loss_smpl.py)."""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from training.smpl_body import compute_gt_mesh_translate


def _binary_cross_entropy_prob(
    probs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    probs = probs.float().clamp(1e-6, 1.0 - 1e-6)
    targets = targets.float()
    loss = -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log())
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported BCE reduction: {reduction}")


def apply_hungarian_matching(
    predictions,
    batch,
    cost_pose_weight: float = 1.0,
    cost_beta_weight: float = 0.1,
    cost_trans_weight: float = 0.0,
    cost_mesh_trans_weight: float = 0.0,
    cost_presence_weight: float = 0.0,
    return_cost_metrics: bool = False,
    use_mamma: bool = False,
):
    """
    Reorder multi-person SMPL predictions to match GT person order.

    Input:
      predictions["smpl_pose"]:  (B, P, 72)
      predictions["smpl_beta"]:  (B, P, 10)
      predictions["smpl_trans"]: (B, P, 3), optional
      predictions["pred_pose_0"]: (B, P, 72), optional

      batch["smpl_pose"]:        (B, P, 72)
      batch["smpl_beta"]:        (B, P, 10)
      batch["smpl_trans"]:       (B, P, 3), optional

    Output:
      predictions are reordered so:
        predictions[:, gt_index] matches batch[:, gt_index]
    """

    pred_pose = predictions["smpl_pose"]
    pred_beta = predictions["smpl_beta"]
    gt_pose = batch["smpl_pose"]
    gt_beta = batch["smpl_beta"]

    if pred_pose.dim() != 3:
        if return_cost_metrics:
            zero = pred_pose.new_zeros(())
            return predictions, batch, {
                "presence_cost": zero,
            }
        return predictions, batch

    B, P_pred = pred_pose.shape[:2]
    P_gt = gt_pose.shape[1]
    P_match = min(P_pred, P_gt)

    pred_trans = predictions.get("smpl_trans", None)
    gt_trans = batch.get("smpl_trans", None)
    pred_mesh_translate = predictions.get("mesh_translate", None)
    gt_mesh_translate = batch.get("mesh_translate", None)
    if pred_mesh_translate is not None and gt_mesh_translate is None:
        gt_mesh_translate = compute_gt_mesh_translate(
            batch,
            normalize_cam=True,
            use_mamma=use_mamma,
        )
        batch = dict(batch)
        batch["mesh_translate"] = gt_mesh_translate
    if pred_trans is None and pred_mesh_translate is not None:
        pred_trans = pred_mesh_translate
        gt_trans = gt_mesh_translate
    pred_presence_logits = predictions.get("smpl_presence_logits", None)
    pred_confidence = predictions.get("smpl_confidence", None)
    has_smpl = batch.get("has_smpl", None)

    predictions = dict(predictions)
    cost_metric_sums = {
        "presence_cost": pred_pose.new_zeros(()),
    }
    cost_metric_count = pred_pose.new_zeros(())

    matched_indices_all = []

    for b in range(B):
        if has_smpl is not None and has_smpl.dim() >= 2:
            valid_gt_indices = torch.where(has_smpl[b, :P_gt].to(device=pred_pose.device) > 0.5)[0]
        else:
            valid_gt_indices = torch.arange(P_gt, device=pred_pose.device)

        if valid_gt_indices.numel() == 0:
            matched_indices_all.append(
                torch.arange(P_pred, device=pred_pose.device, dtype=torch.long)
            )
            continue

        cost = torch.zeros(
            P_pred,
            valid_gt_indices.numel(),
            device=pred_pose.device,
            dtype=pred_pose.dtype,
        )
        presence_cost_mat = torch.zeros_like(cost)

        for i in range(P_pred):
            for col, gt_j in enumerate(valid_gt_indices.tolist()):
                pose_loss = (pred_pose[b, i, :72] - gt_pose[b, gt_j, :72]).abs().mean()
                beta_loss = (pred_beta[b, i] - gt_beta[b, gt_j]).abs().mean()

                total_cost = (
                    cost_pose_weight * pose_loss
                    + cost_beta_weight * beta_loss
                )

                if (
                    cost_trans_weight > 0.0
                    and pred_trans is not None
                    and gt_trans is not None
                ):
                    trans_loss = (pred_trans[b, i] - gt_trans[b, gt_j]).abs().mean()
                    total_cost = total_cost + cost_trans_weight * trans_loss

                if (
                    cost_mesh_trans_weight > 0.0
                    and pred_mesh_translate is not None
                    and gt_mesh_translate is not None
                ):
                    mesh_trans_loss = (
                        pred_mesh_translate[b, i] - gt_mesh_translate[b, gt_j]
                    ).abs().mean()
                    total_cost = total_cost + cost_mesh_trans_weight * mesh_trans_loss

                if (
                    cost_presence_weight > 0.0
                    and pred_presence_logits is not None
                    and pred_presence_logits.dim() == 2
                    and pred_presence_logits.shape[:2] == (B, P_pred)
                ):
                    presence_logit = pred_presence_logits[b, i].to(dtype=pred_pose.dtype)
                    presence_target = torch.ones_like(presence_logit)
                    presence_cost = F.binary_cross_entropy_with_logits(
                        presence_logit,
                        presence_target,
                        reduction="none",
                    )
                    presence_cost_mat[i, col] = presence_cost
                    total_cost = total_cost + cost_presence_weight * presence_cost
                elif (
                    cost_presence_weight > 0.0
                    and pred_confidence is not None
                    and pred_confidence.dim() == 2
                    and pred_confidence.shape[:2] == (B, P_pred)
                ):
                    presence_score = pred_confidence[b, i].to(dtype=pred_pose.dtype).clamp(1e-6, 1.0 - 1e-6)
                    presence_target = torch.ones_like(presence_score)
                    presence_cost = _binary_cross_entropy_prob(
                        presence_score,
                        presence_target,
                        reduction="none",
                    )
                    presence_cost_mat[i, col] = presence_cost
                    total_cost = total_cost + cost_presence_weight * presence_cost

                cost[i, col] = total_cost

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        if len(row_ind) > 0:
            row_t = torch.as_tensor(row_ind, device=pred_pose.device, dtype=torch.long)
            col_t = torch.as_tensor(col_ind, device=pred_pose.device, dtype=torch.long)
            cost_metric_sums["presence_cost"] = cost_metric_sums["presence_cost"] + presence_cost_mat[row_t, col_t].sum()
            cost_metric_count = cost_metric_count + row_t.numel()

        matched_pred_indices = torch.full(
            (P_pred,),
            -1,
            device=pred_pose.device,
            dtype=torch.long,
        )

        used_pred_indices = set()
        for pred_i, gt_col in zip(row_ind, col_ind):
            gt_j = int(valid_gt_indices[int(gt_col)].item())
            if gt_j < P_pred:
                matched_pred_indices[gt_j] = int(pred_i)
                used_pred_indices.add(int(pred_i))

        remaining_pred_indices = [
            idx for idx in range(P_pred) if idx not in used_pred_indices
        ]
        for slot_idx in range(P_pred):
            if matched_pred_indices[slot_idx] < 0:
                matched_pred_indices[slot_idx] = remaining_pred_indices.pop(0)

        matched_indices_all.append(matched_pred_indices)

    matched_indices = torch.stack(matched_indices_all, dim=0)  # (B, P_pred)

    def _gather_people(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 2 and tensor.shape == (B, P_pred):
            return torch.gather(tensor, dim=1, index=matched_indices)
        if tensor.dim() >= 3 and tensor.shape[0] == B and tensor.shape[1] == P_pred:
            index_shape = [B, P_pred] + [1] * (tensor.dim() - 2)
            gather_index = matched_indices.view(*index_shape).expand_as(tensor)
            return torch.gather(tensor, dim=1, index=gather_index)
        return tensor

    def _gather_people_view(tensor):
        if tensor is None:
            return None
        if tensor.dim() >= 3 and tensor.shape[0] == B and tensor.shape[2] == P_pred:
            index_shape = [B, 1, P_pred] + [1] * (tensor.dim() - 3)
            gather_index = matched_indices[:, None].view(*index_shape).expand_as(tensor)
            return torch.gather(tensor, dim=2, index=gather_index)
        return tensor

    for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate", "mesh_rot", "smpl_presence_logits", "pred_pose_0", "smpl_pose_0", "smpl_pose_init"):
        if key in predictions:
            predictions[key] = _gather_people(predictions[key])
    for key in (
        "smpl_anchor_heatmap",
        "smpl_anchor_heatmap_probs",
        "smpl_anchor_2d_patch",
        "smpl_anchor_2d",
        "smpl_view_visibility_logits",
        "smpl_view_query_tokens",
        # dense-landmark / per-person-mask head outputs, laid out (B,S,P,...)
        "smpl_landmarks2d",
        "smpl_landmarks_logvar",
        "smpl_landmarks_visibility_logits",
        "person_mask_logits",
    ):
        if key in predictions:
            predictions[key] = _gather_people_view(predictions[key])

    if return_cost_metrics:
        denom = cost_metric_count.clamp(min=1.0)
        return predictions, batch, {
            key: (value / denom).detach()
            for key, value in cost_metric_sums.items()
        }
    return predictions, batch

