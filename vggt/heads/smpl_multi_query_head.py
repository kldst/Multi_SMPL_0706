
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from vggt.heads.pose_transformer import TransformerDecoder


@dataclass(frozen=True)
class SMPLHeadConfig:
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
	# Prefer not to hard-depend on other submodules; this is best-effort.
	# If not found, we fall back to zeros.
	repo_root = Path(__file__).resolve().parents[2]
	candidate = repo_root / "tram" / "data" / "smpl" / "smpl_mean_params.npz"
	return str(candidate) if candidate.is_file() else None


def _load_mean_params(mean_params_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (init_pose_aa72, init_betas10) as numpy arrays."""
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


class SMPLTransformerDecoderHead(nn.Module):
	"""A VGGT-local SMPL transformer-decoder head.

	Predicts:
	  - SMPL pose in axis-angle (72D)
	  - SMPL betas (10D)
	  - SMPL translation (3D)
	  - per-query/person presence logits

	`num_people` is the number of learned person queries. Each query attends to
	the context tokens and produces one set of SMPL parameters.
	"""

	def __init__(self, *, context_dim: int, cfg: Optional[SMPLHeadConfig] = None, num_people: int = 1):
		super().__init__()
		self.cfg = cfg or SMPLHeadConfig()
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
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""context_tokens: (B, N, C_context) -> (smpl_pose_aa72, smpl_beta10, smpl_trans, presence_logits, pred_pose_0).

		`pred_pose_0` is the pose after the first refinement step (or equals final if `ief_iters==1`).
		"""
		B = context_tokens.shape[0]
		pred_pose = self.init_pose.expand(B, self.num_people, -1)
		pred_beta = self.init_betas.expand(B, self.num_people, -1)
		pred_trans = torch.zeros(
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
		for iter_idx in range(int(self.cfg.ief_iters)):
			token_out = self.transformer(queries, context=context_tokens)
			pred_pose = self.decpose(token_out) + pred_pose
			pred_beta = self.decshape(token_out) + pred_beta
			pred_trans = self.dectrans(token_out) + pred_trans
			presence_logits = self.decpresence(token_out).squeeze(-1)
			if iter_idx == 0:
				pred_pose_0 = pred_pose

		return pred_pose, pred_beta, pred_trans, presence_logits, pred_pose_0


class SMPLMultiQueryHead(nn.Module):
	"""SMPL head for VGGT.

	Inputs:
	  - aggregated_tokens_list: list of tokens (take last)
	  - patch_start_idx: where patch tokens start

	Outputs (always):
	  - smpl_pose: (B, num_people, 72) axis-angle, one prediction per query
	  - smpl_beta: (B, num_people, 10)
	  - smpl_trans: (B, num_people, 3)
	  - smpl_presence_logits: (B, num_people)
	"""

	def __init__(
		self,
		*,
		dim_in: int,
		smpl_cfg: Optional[SMPLHeadConfig] = None,
		context_pool: str = "flatten",  # mean over patch tokens per view
		num_people: int = 1,
	):
		super().__init__()
		self.context_pool = context_pool
		self.decoder = SMPLTransformerDecoderHead(
			context_dim=dim_in,
			cfg=smpl_cfg,
			num_people=num_people,
		)

	def forward(
		self,
		aggregated_tokens_list,
		patch_start_idx: int,
	) -> Dict[str, torch.Tensor]:
		tokens = aggregated_tokens_list[-1]  # (B,S,N,C)
		patch_tokens = tokens[:, :, patch_start_idx:, :]  # (B,S,P,C)

		if self.context_pool == "mean":
			context = patch_tokens.mean(dim=2)  # (B,S,C)
		elif self.context_pool == "flatten":
			B, S, P, C = patch_tokens.shape
			context = patch_tokens.reshape(B, S * P, C)
		else:
			raise ValueError(f"Unknown context_pool: {self.context_pool}")

		# Ensure context is (B,N,C)
		if context.dim() == 3 and context.shape[1] == tokens.shape[1]:
			context_tokens = context
		else:
			context_tokens = context

		smpl_pose, smpl_beta, smpl_trans, smpl_presence_logits, pred_pose_0 = self.decoder(context_tokens)

		outputs: Dict[str, torch.Tensor] = {
			"smpl_pose": smpl_pose,
			"smpl_beta": smpl_beta,
			"smpl_trans": smpl_trans,
			"smpl_presence_logits": smpl_presence_logits,
			"pred_pose_0": pred_pose_0,
		}
		return outputs


__all__ = ["SMPLMultiQueryHead", "SMPLTransformerDecoderHead", "SMPLHeadConfig"]