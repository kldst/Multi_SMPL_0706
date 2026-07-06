from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from vggt.heads.pose_transformer import TransformerDecoder


@dataclass(frozen=True)
class SMPLRotHeadConfig:
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_dim: int = 1024
    transformer_dim_head: int = 64
    transformer_dropout: float = 0.0
    transformer_emb_dropout: float = 0.0
    transformer_norm: str = "layer"
    transformer_dim: int = 1024
    ief_iters: int = 1
    mean_params_path: Optional[str] = None


def _default_mean_params_path() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "tram" / "data" / "smpl" / "smpl_mean_params.npz"
    return str(candidate) if candidate.is_file() else None


def _load_mean_params(mean_params_path: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    if mean_params_path is None:
        mean_params_path = _default_mean_params_path()
    if mean_params_path is None:
        return np.zeros((72,), dtype=np.float32), np.zeros((10,), dtype=np.float32)

    data = np.load(mean_params_path)
    pose = data.get("pose")
    shape = data.get("shape")
    if pose is None or shape is None:
        return np.zeros((72,), dtype=np.float32), np.zeros((10,), dtype=np.float32)
    return pose.astype(np.float32).reshape(-1)[:72], shape.astype(np.float32).reshape(-1)[:10]


class SMPLRotTransformerDecoderHead(nn.Module):
    """SMPL multi-query decoder predicting pose / shape / mesh_translate / presence.

    The first 3 dims of the predicted pose are interpreted downstream as mesh_rot --
    the SMPL root rotation in the first-camera (cam0) frame -- rather than the
    world-frame global_orient.
    """

    def __init__(self, *, context_dim: int, cfg: Optional[SMPLRotHeadConfig] = None, num_people: int = 1):
        super().__init__()
        self.cfg = cfg or SMPLRotHeadConfig()
        self.num_people = int(num_people)
        self.query_dim = int(self.cfg.transformer_dim)

        init_pose, init_betas = _load_mean_params(self.cfg.mean_params_path)
        self.register_buffer("init_pose", torch.from_numpy(init_pose).view(1, 1, 72))
        self.register_buffer("init_betas", torch.from_numpy(init_betas).view(1, 1, 10))

        self.person_queries = nn.Parameter(torch.randn(1, self.num_people, self.query_dim) * 0.02)
        self.transformer = TransformerDecoder(
            num_tokens=self.num_people,
            token_dim=self.query_dim,
            dim=self.query_dim,
            depth=self.cfg.transformer_depth,
            heads=self.cfg.transformer_heads,
            mlp_dim=self.cfg.transformer_mlp_dim,
            dim_head=self.cfg.transformer_dim_head,
            dropout=self.cfg.transformer_dropout,
            emb_dropout=self.cfg.transformer_emb_dropout,
            norm=self.cfg.transformer_norm,
            context_dim=context_dim,
            skip_token_embedding=True,
        )

        self.decpose = nn.Linear(self.query_dim, 72)
        self.decshape = nn.Linear(self.query_dim, 10)
        self.dectrans = nn.Linear(self.query_dim, 3)
        self.decpresence = nn.Linear(self.query_dim, 1)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dectrans.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decpresence.weight, gain=0.01)

    def forward(
        self, context_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = context_tokens.shape[0]
        pred_pose = self.init_pose.expand(B, self.num_people, -1)
        pred_beta = self.init_betas.expand(B, self.num_people, -1)
        pred_translate = torch.zeros(
            (B, self.num_people, 3),
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        presence_logits = torch.zeros(
            (B, self.num_people),
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        )
        pred_pose_0 = pred_pose

        queries = self.person_queries.expand(B, -1, -1).to(
            device=context_tokens.device,
            dtype=context_tokens.dtype,
        ).clone()
        token_out = queries
        for iter_idx in range(int(self.cfg.ief_iters)):
            token_out = self.transformer(queries, context=context_tokens)
            pred_pose = self.decpose(token_out) + pred_pose
            pred_beta = self.decshape(token_out) + pred_beta
            pred_translate = self.dectrans(token_out) + pred_translate
            presence_logits = self.decpresence(token_out).squeeze(-1)
            if iter_idx == 0:
                pred_pose_0 = pred_pose

        # token_out (B, P, query_dim) is surfaced so auxiliary heads (dense
        # landmarks, per-person mask) can condition on the SAME person tokens the
        # SMPL params are decoded from -- this is what keeps every head's slot p
        # bound to one identity without any extra matching.
        return pred_pose, pred_beta, pred_translate, presence_logits, pred_pose_0, token_out


class SMPLMultiQueryTransRotHead(nn.Module):
    """SMPL multi-query head that predicts mesh_translate AND mesh_rot.

    The first 3 dims of the predicted ``smpl_pose`` are reinterpreted as ``mesh_rot``
    -- the SMPL mesh root rotation expressed in the FIRST camera (cam0) coordinate
    frame -- instead of the world-frame global_orient.

    Predicting the orientation in the cam0 gauge (the same gauge the camera head is
    normalized into) makes the target invariant to the dataset's world up-axis
    convention (Y-up vs Z-up), which otherwise leaks into global_orient and makes
    the body tip over by ~90 deg on out-of-distribution datasets.

    ``mesh_rot`` (a view onto ``smpl_pose[..., :3]``) is surfaced so the loss / eval /
    demo can detect this mode the same way ``mesh_translate`` flags translation mode.
    """

    def __init__(
        self,
        *,
        dim_in: int,
        smpl_cfg: Optional[SMPLRotHeadConfig] = None,
        context_pool: str = "flatten",
        num_people: int = 1,
    ):
        super().__init__()
        self.context_pool = context_pool
        self.decoder = SMPLRotTransformerDecoderHead(
            context_dim=dim_in,
            cfg=smpl_cfg,
            num_people=num_people,
        )

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        patch_tokens = tokens[:, :, patch_start_idx:, :]

        if self.context_pool == "mean":
            context_tokens = patch_tokens.mean(dim=2)
        elif self.context_pool == "flatten":
            B, S, P, C = patch_tokens.shape
            context_tokens = patch_tokens.reshape(B, S * P, C)
        else:
            raise ValueError(f"Unknown context_pool: {self.context_pool}")

        smpl_pose, smpl_beta, mesh_translate, smpl_presence_logits, pred_pose_0, person_tokens = self.decoder(context_tokens)

        return {
            "smpl_pose": smpl_pose,
            "smpl_beta": smpl_beta,
            "mesh_translate": mesh_translate,
            # mesh_rot == smpl_pose[..., :3], surfaced so downstream code can detect
            # that the root rotation is in cam0 frame (mesh_rot mode).
            "mesh_rot": smpl_pose[..., :3],
            "smpl_presence_logits": smpl_presence_logits,
            "pred_pose_0": pred_pose_0,
            # (B, P, query_dim) person tokens for auxiliary heads (dense landmark,
            # person mask). Not a loss target; consumed inside VGGT.forward.
            "person_tokens": person_tokens,
        }


__all__ = [
    "SMPLMultiQueryTransRotHead",
    "SMPLRotTransformerDecoderHead",
    "SMPLRotHeadConfig",
]
