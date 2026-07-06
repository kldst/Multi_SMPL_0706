"""Per-person mask loss (split out of loss.py).

Supervises the PersonMaskHead's patch-grid occupancy logits against the
down-sampled instance-mask GT. Called from ``compute_smpl_loss`` AFTER Hungarian
matching + people-flattening, so tensors arrive as ``(B*P, S, H, W)`` and the
per-person ``has_smpl`` mask silences padded / unmatched slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_mask_loss(
    pred_logits: torch.Tensor,     # (B*P, S, H, W)
    gt_mask: torch.Tensor,         # (B*P, S, H, W) soft occupancy in [0,1]
    has_smpl: torch.Tensor | None, # (B*P,)
    loss_type: str = "bce",
) -> dict:
    """BCE between predicted mask logits and GT occupancy, masked by has_smpl."""
    pred_logits = pred_logits.float()
    gt_mask = gt_mask.to(device=pred_logits.device, dtype=pred_logits.dtype).clamp(0.0, 1.0)

    if pred_logits.shape != gt_mask.shape:
        raise ValueError(
            f"person mask shape mismatch: pred {tuple(pred_logits.shape)} vs gt {tuple(gt_mask.shape)}"
        )

    # per-element BCE, then reduce over the spatial + view dims.
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction="none")
    per_person = bce.flatten(1).mean(dim=1)                       # (B*P,)

    if has_smpl is not None:
        w = has_smpl.to(device=per_person.device, dtype=per_person.dtype).reshape(-1)
        loss_mask = (per_person * w).sum() / w.sum().clamp(min=1.0)
    else:
        loss_mask = per_person.mean()

    # soft IoU as a readable diagnostic (not back-propped weight).
    with torch.no_grad():
        prob = torch.sigmoid(pred_logits)
        inter = (prob * gt_mask).flatten(1).sum(1)
        union = (prob + gt_mask).flatten(1).sum(1) - inter
        iou = (inter / union.clamp(min=1e-6))
        if has_smpl is not None:
            w = has_smpl.reshape(-1).to(iou.dtype)
            iou = (iou * w).sum() / w.sum().clamp(min=1.0)
        else:
            iou = iou.mean()

    return {"loss_mask": loss_mask, "mask_soft_iou": iou.detach()}


__all__ = ["compute_mask_loss"]
