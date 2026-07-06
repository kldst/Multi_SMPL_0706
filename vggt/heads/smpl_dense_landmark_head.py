"""Dense 512-landmark head for the multi-person VGGT SMPL model.

Motivation
----------
The multi-person SMPL head binds each of the ``P`` person queries to one person
via a single token, supervised only by global per-person params (pose/beta/
trans).  In crowded / interacting scenes that signal is too weak to stop a query
from mixing two people (limbs of A + torso of B).  This head adds a *dense,
spatially-grounded* auxiliary target: for every person, in every view, it
predicts 512 surface landmarks (the MAMMA ``verts_512`` down-sampling of the
SMPL-X mesh) as 2D image coordinates -- exactly like MAMMA's single-view
detector, but multi-person and multi-view.  A query can only lower this loss by
attending to a single coherent body, which is the identity-disentangling
pressure we want.

Design (direct 2D, per view)
----------------------------
* It reuses the **same** person tokens produced by the SMPL head
  (``person_tokens``, shape ``(B, P, C_q)``) so the dense gradient flows back
  into the very tokens that decode SMPL -- no separate person queries, no extra
  matching.
* 512 learnable vertex embeddings (one per fixed landmark) are added to the
  broadcast person token to form the decoder queries; each query means "person
  p's landmark i".
* Because a person token is fused across views, per-view 2D cannot come from one
  token alone -- so the queries cross-attend **each view's own patch tokens
  separately** (mamma's decoder, run once per view).  The view-specific context
  is what makes the same query produce that view's 2D.
* Output is ``sigmoid`` -> normalised 2D in ``[0, 1]`` image coordinates
  ``(B, S, P, 512, 2)`` plus a per-landmark log-variance for the Gaussian NLL
  and a visibility logit -- no camera or gauge dependency, so the loss stays
  decoupled from the camera head.
  The ``[0, 1]`` sigmoid bound matches MAMMA's DenseLandmarkHead and keeps the
  Gaussian-NLL well-posed (an unbounded coord output collapses under GNLL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from vggt.heads.pose_transformer import TransformerDecoder

NUM_LANDMARKS = 512


@dataclass(frozen=True)
class DenseLandmarkHeadConfig:
    num_landmarks: int = NUM_LANDMARKS
    transformer_depth: int = 3
    transformer_heads: int = 8
    transformer_mlp_dim: int = 1024
    transformer_dim_head: int = 64
    transformer_dropout: float = 0.0
    transformer_dim: int = 1024          # must match the SMPL head query dim
    # Subsample each view's patch context (0 = use all patch tokens of the view).
    max_context_tokens: int = 0
    use_mask_embedding: bool = False
    mask_embed_dim: int = 256
    detach_mask_context: bool = True
    # MAMMA-style per-landmark contact heads: person-person contact + floor contact.
    predict_contact: bool = False


class SMPLDenseLandmarkHead(nn.Module):
    """Predict sigmoid ``[0,1]`` 2D landmarks, log-var, and visibility logits.

    Args:
        context_dim: channel dim of the aggregator patch tokens (2*embed_dim).
        query_dim: channel dim of the person tokens from the SMPL head.
        cfg: head hyper-parameters.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        query_dim: int = 1024,
        cfg: Optional[DenseLandmarkHeadConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or DenseLandmarkHeadConfig()
        self.num_landmarks = int(self.cfg.num_landmarks)
        self.query_dim = int(query_dim)

        # One embedding per fixed landmark id -> "which vertex am I". Init at std~1
        # (like nn.Embedding) NOT 0.02: the person_token added below has norm ~50,
        # so a tiny landmark id would be drowned out (74x) -> all 512 queries become
        # near-identical -> the 2D predictions collapse to a line. A strong per-vertex
        # identity is what lets the head resolve fine (esp. horizontal) structure.
        self.landmark_embed = nn.Parameter(
            torch.randn(1, self.num_landmarks, self.query_dim)
        )
        # Normalise the person token before adding, so its (large, variable) magnitude
        # can't swamp the per-landmark identity. LN preserves the identity DIRECTION,
        # so person disentangling is unaffected -- only the scale is tamed.
        self.person_ln = nn.LayerNorm(self.query_dim)
        self.use_mask_embedding = bool(self.cfg.use_mask_embedding)
        self.detach_mask_context = bool(self.cfg.detach_mask_context)
        if self.use_mask_embedding:
            self.mask_embedding = nn.Sequential(
                nn.Conv2d(1, int(self.cfg.mask_embed_dim), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(int(self.cfg.mask_embed_dim), context_dim, kernel_size=1),
            )
            # Preserve the previous head at init. The mask prompt starts as a no-op
            # and learns to bias per-person patch evidence during overfit/training.
            nn.init.zeros_(self.mask_embedding[-1].weight)
            nn.init.zeros_(self.mask_embedding[-1].bias)
        else:
            self.mask_embedding = None
        self.transformer = TransformerDecoder(
            num_tokens=self.num_landmarks,
            token_dim=self.query_dim,
            dim=self.query_dim,
            depth=self.cfg.transformer_depth,
            heads=self.cfg.transformer_heads,
            mlp_dim=self.cfg.transformer_mlp_dim,
            dim_head=self.cfg.transformer_dim_head,
            dropout=self.cfg.transformer_dropout,
            emb_dropout=0.0,
            norm="layer",
            context_dim=context_dim,
            skip_token_embedding=True,
        )
        self.dec_xy = nn.Linear(self.query_dim, 2)
        self.dec_logvar = nn.Linear(self.query_dim, 1)
        self.dec_vis = nn.Linear(self.query_dim, 1)
        # MAMMA-style per-landmark contact logits (person-person + floor). Optional.
        self.predict_contact = bool(self.cfg.predict_contact)
        if self.predict_contact:
            self.dec_contact = nn.Linear(self.query_dim, 1)
            self.dec_floor_contact = nn.Linear(self.query_dim, 1)
        else:
            self.dec_contact = None
            self.dec_floor_contact = None
        # Small init so pre-sigmoid ~0 -> predictions start at image centre (0.5,0.5).
        nn.init.xavier_uniform_(self.dec_xy.weight, gain=0.01)
        nn.init.zeros_(self.dec_xy.bias)
        nn.init.zeros_(self.dec_logvar.weight)
        nn.init.zeros_(self.dec_logvar.bias)
        nn.init.zeros_(self.dec_vis.weight)
        nn.init.zeros_(self.dec_vis.bias)
        if self.predict_contact:
            for layer in (self.dec_contact, self.dec_floor_contact):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _maybe_subsample_context(self, context: torch.Tensor) -> torch.Tensor:
        cap = int(self.cfg.max_context_tokens)
        if cap <= 0 or context.shape[-2] <= cap:
            return context
        idx = torch.randperm(context.shape[-2], device=context.device)[:cap]
        return context.index_select(-2, idx)

    def _add_mask_context(
        self,
        context: torch.Tensor,
        person_mask_logits: Optional[torch.Tensor],
        *,
        B: int,
        S: int,
        P: int,
        Np: int,
        Cctx: int,
    ) -> torch.Tensor:
        if self.mask_embedding is None or person_mask_logits is None:
            return context

        mask = person_mask_logits
        if mask.dim() != 5:
            raise ValueError(
                f"person_mask_logits must have shape (B,S,P,H,W), got {tuple(mask.shape)}"
            )
        if mask.shape[:3] != (B, S, P):
            raise ValueError(
                f"person_mask_logits shape {tuple(mask.shape)} does not match B,S,P={(B, S, P)}"
            )
        if self.detach_mask_context:
            mask = mask.detach()
        mask = torch.sigmoid(mask.float()).reshape(B * S * P, 1, mask.shape[-2], mask.shape[-1])
        mask_feat = self.mask_embedding(mask).flatten(2).transpose(1, 2)  # (B*S*P, Hm*Wm, Cctx)
        if mask_feat.shape[1] != Np:
            # Keep this robust if a future mask head uses a different grid.
            ph = pw = None
            side = int(Np ** 0.5)
            if side * side == Np:
                ph = pw = side
            if ph is None:
                raise ValueError(
                    f"Cannot resize mask prompt with {mask_feat.shape[1]} tokens to non-square context {Np}"
                )
            mask_feat = mask_feat.transpose(1, 2).reshape(B * S * P, Cctx, mask.shape[-2], mask.shape[-1])
            mask_feat = torch.nn.functional.interpolate(
                mask_feat, size=(ph, pw), mode="bilinear", align_corners=False
            ).flatten(2).transpose(1, 2)
        return context + mask_feat.to(dtype=context.dtype)

    def forward(
        self,
        person_tokens: torch.Tensor,       # (B, P, C_q)
        patch_tokens: torch.Tensor,        # (B, S, N_patch, C_ctx)  per-view
        person_mask_logits: Optional[torch.Tensor] = None,  # (B, S, P, H, W)
    ) -> Dict[str, torch.Tensor]:
        B, P, Cq = person_tokens.shape
        assert Cq == self.query_dim, f"person token dim {Cq} != {self.query_dim}"
        _, S, N, Cctx = patch_tokens.shape
        L = self.num_landmarks

        # queries[b, s, p, i] = landmark_embed[i] + LN(person_token[b, p])
        # broadcast over views -> (B, S, P, L, Cq). The LN keeps the person token
        # from swamping the per-landmark identity (see __init__).
        person = self.person_ln(person_tokens)                         # (B, P, Cq)
        queries = (
            self.landmark_embed.to(person_tokens.dtype).view(1, 1, 1, L, Cq)
            + person.view(B, 1, P, 1, Cq)
        ).expand(B, S, P, L, Cq).reshape(B * S * P, L, Cq)

        # each (view, person) group attends that view's own patch tokens.
        context = self._maybe_subsample_context(patch_tokens)          # (B,S,N',Cctx)
        Np = context.shape[-2]
        context = (
            context.view(B, S, 1, Np, Cctx)
            .expand(B, S, P, Np, Cctx)
            .reshape(B * S * P, Np, Cctx)
        )
        context = self._add_mask_context(
            context,
            person_mask_logits,
            B=B,
            S=S,
            P=P,
            Np=Np,
            Cctx=Cctx,
        )

        tokens = self.transformer(queries, context=context)           # (B*S*P, L, Cq)
        # sigmoid -> [0,1] image coords: bounds the output and keeps the GNLL
        # well-posed (matches MAMMA's DenseLandmarkHead; unbounded coords collapse).
        xy = torch.sigmoid(self.dec_xy(tokens)).reshape(B, S, P, L, 2)
        logvar = self.dec_logvar(tokens).reshape(B, S, P, L)
        vis_logits = self.dec_vis(tokens).reshape(B, S, P, L)

        out = {
            # normalised [0,1] image 2D, per view -- compared to M @ vertices2d.
            "smpl_landmarks2d": xy,
            "smpl_landmarks_logvar": logvar,
            "smpl_landmarks_visibility_logits": vis_logits,
        }
        if self.predict_contact:
            # per-landmark contact logits (MAMMA-style), BCE/focal-supervised.
            out["smpl_contact_logits"] = self.dec_contact(tokens).reshape(B, S, P, L)
            out["smpl_floor_contact_logits"] = self.dec_floor_contact(tokens).reshape(B, S, P, L)
        return out


__all__ = ["SMPLDenseLandmarkHead", "DenseLandmarkHeadConfig", "NUM_LANDMARKS"]
