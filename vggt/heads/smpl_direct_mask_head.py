"""Token -> mask-image decoders (ablations of the DPT / dot mask heads).

Both existing mask heads (:class:`PersonMaskHead`, :class:`PersonMaskDPTHead`)
predict the per-person mask as a *dot product* between a projected person token
and per-view image features (patch tokens / a DPT per-pixel embedding map).  The
per-view differences therefore come entirely from the image side.

This module holds two ablation heads that instead *decode* the mask with a
ConvTranspose upsampling stack and **no image patch features**:

* :class:`SMPLDirectMaskHead` -- pure token decoder.  Consumes only the
  view-agnostic person token ``(B, P, C)``, so it produces ONE mask per person
  broadcast to all views; it cannot express per-view differences by construction.
  Measures how much of the mask the token alone carries.
* :class:`SMPLDirectMaskCamHead` -- token + per-view camera conditioning.  Fuses
  the person token with each view's camera token into a per-view feature
  ``t[b, s, p]`` before decoding, so it CAN produce different masks per view --
  using geometry, still without image patch features.  Measures the extra lift
  from knowing "which camera" on top of the token.

Output is ``(B, S, P, out_size, out_size)`` logits (default 518x518, full image
resolution).  Down-sampling for the Hungarian mask cost happens later in
``smpl_matching`` (adaptive-pool to ``hungarian_mask_cost_grid``); the pixel loss
runs at full resolution against the stride-1 GT.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpBlock(nn.Module):
    """ConvTranspose x2 upsample + GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        # GroupNorm needs num_groups | out_ch; fall back to the largest divisor <= groups.
        ng = next(g for g in range(min(groups, out_ch), 0, -1) if out_ch % g == 0)
        self.norm = nn.GroupNorm(ng, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


def _channel_schedule(seed_size: int, out_size: int, base_channels: int) -> List[int]:
    """Channels from the seed grid up to the largest x2 grid <= out_size.

    We stop just below out_size and bilinearly interpolate the last step up
    (e.g. 512 -> 518), rather than overshoot to 1024 and downsample -- same
    result, ~4x less high-res compute/memory."""
    n_up = 0
    cur = seed_size
    while cur * 2 <= out_size:
        cur *= 2
        n_up += 1
    sched = [base_channels]
    for i in range(n_up):
        nxt = max(16, sched[-1] // 2 if i < 2 else int(sched[-1] * 0.75))
        sched.append(nxt)
    return sched


def _build_decoder(
    seed_size: int, out_size: int, base_channels: int
) -> Tuple[nn.ModuleList, nn.Conv2d, int]:
    """Build the shared ConvTranspose decoder. Returns (blocks, to_logit, decoded_size)."""
    chans = _channel_schedule(seed_size, out_size, base_channels)
    blocks = nn.ModuleList()
    cur = seed_size
    for i in range(len(chans) - 1):
        blocks.append(_UpBlock(chans[i], chans[i + 1]))
        cur *= 2
    to_logit = nn.Conv2d(chans[-1], 1, kernel_size=3, padding=1)
    return blocks, to_logit, cur


def _decode(
    seed: torch.Tensor, blocks: nn.ModuleList, to_logit: nn.Conv2d,
    n_items: int, base_channels: int, seed_size: int, out_size: int,
) -> torch.Tensor:
    """Run the seed grid through the decoder -> (n_items, out_size, out_size)."""
    x = seed.reshape(n_items, base_channels, seed_size, seed_size)
    for blk in blocks:
        x = blk(x)
    logits = to_logit(x)                                          # (n_items, 1, D, D)
    if logits.shape[-1] != out_size:
        logits = F.interpolate(
            logits, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
    return logits.reshape(n_items, out_size, out_size)


class SMPLDirectMaskHead(nn.Module):
    """Pure token decoder: decode each person token into a full-res mask image.

    View-agnostic by construction (person tokens have no view dim), so the single
    per-person mask is broadcast to all S views.

    Args:
        query_dim: person-token channel dim (SMPL head query dim).
        out_size: final square mask resolution (518 = full image res).
        seed_size: spatial size of the decoded seed grid (token -> seed_size^2).
        base_channels: channel count at the seed grid.
    """

    def __init__(
        self,
        *,
        query_dim: int = 1024,
        out_size: int = 518,
        seed_size: int = 8,
        base_channels: int = 256,
    ):
        super().__init__()
        self.out_size = int(out_size)
        self.seed_size = int(seed_size)
        self.base_channels = int(base_channels)

        self.token_norm = nn.LayerNorm(query_dim)
        self.to_seed = nn.Linear(query_dim, self.base_channels * self.seed_size * self.seed_size)
        self.blocks, self.to_logit, self.decoded_size = _build_decoder(
            self.seed_size, self.out_size, self.base_channels
        )

    def _channel_schedule(self):  # kept for introspection / tests
        return _channel_schedule(self.seed_size, self.out_size, self.base_channels)

    def forward(
        self,
        person_tokens: torch.Tensor,   # (B, P, C_q)
        images: torch.Tensor,          # (B, S, 3, H, W) -- used only for S / device
    ) -> torch.Tensor:
        B, P, _ = person_tokens.shape
        S = images.shape[1]

        seed = self.to_seed(self.token_norm(person_tokens))      # (B, P, base*s*s)
        logits = _decode(
            seed, self.blocks, self.to_logit,
            B * P, self.base_channels, self.seed_size, self.out_size,
        )
        logits = logits.reshape(B, P, self.out_size, self.out_size)
        # View-agnostic: broadcast the single per-person mask to all S views.
        return logits.unsqueeze(1).expand(B, S, P, self.out_size, self.out_size)


class SMPLDirectMaskCamHead(nn.Module):
    """Token + per-view camera conditioning -> per-view mask image (no image feats).

    Fuses the person token with each view's camera token into a per-view seed
    ``t[b, s, p]``, then decodes each independently -- so masks genuinely differ
    per view, driven by camera geometry rather than image patch content.

    Args:
        query_dim: person-token channel dim.
        cam_dim: camera-token channel dim (2*embed_dim).
        out_size / seed_size / base_channels: as in SMPLDirectMaskHead.
        q_embed / c_embed: projection dims for the token / camera before fusion.
    """

    def __init__(
        self,
        *,
        query_dim: int = 1024,
        cam_dim: int = 2048,
        out_size: int = 518,
        seed_size: int = 8,
        base_channels: int = 256,
        q_embed: int = 512,
        c_embed: int = 256,
    ):
        super().__init__()
        self.out_size = int(out_size)
        self.seed_size = int(seed_size)
        self.base_channels = int(base_channels)

        self.token_norm = nn.LayerNorm(query_dim)
        self.cam_norm = nn.LayerNorm(cam_dim)
        self.q_proj = nn.Linear(query_dim, q_embed)
        self.c_proj = nn.Linear(cam_dim, c_embed)
        self.to_seed = nn.Linear(q_embed + c_embed, self.base_channels * self.seed_size * self.seed_size)
        self.blocks, self.to_logit, self.decoded_size = _build_decoder(
            self.seed_size, self.out_size, self.base_channels
        )

    def _channel_schedule(self):  # kept for introspection / tests
        return _channel_schedule(self.seed_size, self.out_size, self.base_channels)

    def forward(
        self,
        person_tokens: torch.Tensor,   # (B, P, C_q)
        cam_tokens: torch.Tensor,      # (B, S, C_cam)
    ) -> torch.Tensor:
        B, P, _ = person_tokens.shape
        S = cam_tokens.shape[1]

        q = self.q_proj(self.token_norm(person_tokens))          # (B, P, Dq)
        c = self.c_proj(self.cam_norm(cam_tokens.to(person_tokens.dtype)))  # (B, S, Dc)

        q_exp = q[:, None, :, :].expand(B, S, P, q.shape[-1])
        c_exp = c[:, :, None, :].expand(B, S, P, c.shape[-1])
        fused = torch.cat([q_exp, c_exp], dim=-1)                 # (B, S, P, Dq+Dc)

        seed = self.to_seed(fused)                               # (B, S, P, base*s*s)
        logits = _decode(
            seed, self.blocks, self.to_logit,
            B * S * P, self.base_channels, self.seed_size, self.out_size,
        )
        return logits.reshape(B, S, P, self.out_size, self.out_size)


__all__ = ["SMPLDirectMaskHead", "SMPLDirectMaskCamHead"]
