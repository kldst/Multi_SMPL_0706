"""Per-person, per-view segmentation-mask head (MaskFormer-style).

Each person query already localises "its" person for the SMPL head; this head
makes that localisation *explicit and dense* by predicting, for every person and
every view, a soft occupancy mask over the aggregator patch grid.  Supervised by
the instance mask shipped with the raw data (``*.mask.jpg``, pixel value ==
person_idx + 1, so it is occlusion-aware), it pushes each person token to attend
to exactly one body's pixels -- the same identity-disentangling pressure as the
dense-landmark head, but expressed directly in image space.

Mechanism
---------
The mask for (person p, view s) is the scaled dot product between a projected
person token and every patch token of that view:

    logits[b, s, p, n] = <W_q · person_token[b, p], W_k · patch_token[b, s, n]> / sqrt(D)

which is one attention map per (person, view) -- cheap, and it reuses the very
tokens the SMPL head binds to.  Output is patch-resolution logits; the loss
compares them to the down-sampled instance mask.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PersonMaskHead(nn.Module):
    """Predict per-person per-view patch-grid mask logits.

    Args:
        query_dim: person-token channel dim (SMPL head query dim).
        context_dim: aggregator patch-token channel dim (2*embed_dim).
        embed_dim: shared dot-product embedding dim.
    """

    def __init__(self, *, query_dim: int = 1024, context_dim: int = 2048, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, embed_dim),
        )
        self.k_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, embed_dim),
        )
        # Learnable temperature (log-scale) on the dot product.
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        person_tokens: torch.Tensor,     # (B, P, C_q)
        patch_tokens: torch.Tensor,      # (B, S, N_patch, C_ctx)
        patch_hw: Optional[tuple] = None,
    ) -> torch.Tensor:
        B, P, _ = person_tokens.shape
        _, S, N, _ = patch_tokens.shape

        q = self.q_proj(person_tokens)                       # (B, P, D)
        k = self.k_proj(patch_tokens)                        # (B, S, N, D)

        # logits[b, s, p, n] = q[b, p] . k[b, s, n]
        logits = torch.einsum("bpd,bsnd->bspn", q, k)
        logits = logits * (self.embed_dim ** -0.5) * self.log_scale.exp()

        if patch_hw is not None:
            ph, pw = patch_hw
            if ph * pw != N:
                raise ValueError(
                    f"patch_hw {patch_hw} does not match {N} patch tokens"
                )
            logits = logits.reshape(B, S, P, ph, pw)
        return logits                                        # (B,S,P,N) or (B,S,P,H,W)


__all__ = ["PersonMaskHead"]
