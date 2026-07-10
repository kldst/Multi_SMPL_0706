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
    cost_mask_weight: float = 0.0,
    mask_cost_grid: int = 32,
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

    Mask cost (``cost_mask_weight > 0``): matches on 2D image-space evidence.
    ``predictions["person_mask_logits"] (B,S,P,H,W)`` vs ``batch["person_mask"]
    (B,S,P,h,w)`` are both adaptive-avg-pooled to ``mask_cost_grid`` squared and
    compared per (pred, gt) pair with soft BCE.  This is what keeps the matching
    stable when two people are in contact: their pose / mesh_translate costs
    become near-identical, but their image masks stay distinct.

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
                "mask_cost": zero,
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

    # 2D mask evidence for the matching cost (optional).
    pred_mask_logits = predictions.get("person_mask_logits", None)
    gt_person_mask = batch.get("person_mask", None)
    use_mask_cost = (
        cost_mask_weight > 0.0
        and pred_mask_logits is not None
        and gt_person_mask is not None
        and pred_mask_logits.dim() == 5          # (B, S, P, H, W)
        and gt_person_mask.dim() == 5
        and pred_mask_logits.shape[2] == P_pred
    )

    predictions = dict(predictions)
    cost_metric_sums = {
        "presence_cost": pred_pose.new_zeros(()),
        "mask_cost": pred_pose.new_zeros(()),
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

        # The assignment is a hard (non-differentiable) decision; build the whole
        # cost matrix (P_pred, G) in one shot under no_grad instead of a Python
        # double loop -- pairwise L1 via broadcasting.
        with torch.no_grad():
            valid = valid_gt_indices
            pose_cost = (
                pred_pose[b, :, None, :72].float() - gt_pose[b, None, valid, :72].float()
            ).abs().mean(-1)                                       # (P_pred, G)
            beta_cost = (
                pred_beta[b, :, None].float() - gt_beta[b, None, valid].float()
            ).abs().mean(-1)

            cost = cost_pose_weight * pose_cost + cost_beta_weight * beta_cost
            presence_cost_mat = torch.zeros_like(cost)
            mask_cost_mat = torch.zeros_like(cost)

            if cost_trans_weight > 0.0 and pred_trans is not None and gt_trans is not None:
                cost = cost + cost_trans_weight * (
                    pred_trans[b, :, None].float() - gt_trans[b, None, valid].float()
                ).abs().mean(-1)

            if (
                cost_mesh_trans_weight > 0.0
                and pred_mesh_translate is not None
                and gt_mesh_translate is not None
            ):
                cost = cost + cost_mesh_trans_weight * (
                    pred_mesh_translate[b, :, None].float()
                    - gt_mesh_translate[b, None, valid].float()
                ).abs().mean(-1)

            if (
                cost_presence_weight > 0.0
                and pred_presence_logits is not None
                and pred_presence_logits.dim() == 2
                and pred_presence_logits.shape[:2] == (B, P_pred)
            ):
                presence_col = F.binary_cross_entropy_with_logits(
                    pred_presence_logits[b].float(),
                    torch.ones_like(pred_presence_logits[b].float()),
                    reduction="none",
                )                                                  # (P_pred,)
                presence_cost_mat = presence_col[:, None].expand_as(cost).clone()
                cost = cost + cost_presence_weight * presence_cost_mat
            elif (
                cost_presence_weight > 0.0
                and pred_confidence is not None
                and pred_confidence.dim() == 2
                and pred_confidence.shape[:2] == (B, P_pred)
            ):
                presence_col = _binary_cross_entropy_prob(
                    pred_confidence[b].float(),
                    torch.ones_like(pred_confidence[b].float()),
                    reduction="none",
                )
                presence_cost_mat = presence_col[:, None].expand_as(cost).clone()
                cost = cost + cost_presence_weight * presence_cost_mat

            if use_mask_cost:
                grid = int(mask_cost_grid)
                # pred: (S, P, H, W) -> probs (P, S, g, g); gt: (S, P, h, w) -> (G, S, g, g).
                # Pool BOTH to the same coarse grid so pred/GT resolutions need not match.
                probs = torch.sigmoid(pred_mask_logits[b].float()).permute(1, 0, 2, 3)
                probs = F.adaptive_avg_pool2d(probs, (grid, grid))
                gm = gt_person_mask[b].float().permute(1, 0, 2, 3)[valid]
                gm = F.adaptive_avg_pool2d(gm, (grid, grid)).clamp(0.0, 1.0)
                p = probs.reshape(P_pred, -1).clamp(1e-6, 1.0 - 1e-6)  # (P_pred, N)
                g = gm.reshape(valid.numel(), -1)                      # (G, N)
                # pairwise soft BCE via two matmuls: (P_pred, G), mean over N pixels.
                mask_cost_mat = -(
                    p.log() @ g.t() + (1.0 - p).log() @ (1.0 - g).t()
                ) / p.shape[1]
                cost = cost + cost_mask_weight * mask_cost_mat

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        if len(row_ind) > 0:
            row_t = torch.as_tensor(row_ind, device=pred_pose.device, dtype=torch.long)
            col_t = torch.as_tensor(col_ind, device=pred_pose.device, dtype=torch.long)
            cost_metric_sums["presence_cost"] = cost_metric_sums["presence_cost"] + presence_cost_mat[row_t, col_t].sum()
            cost_metric_sums["mask_cost"] = cost_metric_sums["mask_cost"] + mask_cost_mat[row_t, col_t].sum()
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

