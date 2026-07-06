"""SMPL pose/shape/joints/vertices + dense-landmark losses (split out of loss.py)."""
from training.loss_mask import compute_mask_loss  # noqa: E402  (used in compute_smpl_loss)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import logging

from dataclasses import dataclass
from pathlib import Path
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from training.train_utils.general import check_and_fix_inf_nan
import numpy as np
from scipy.optimize import linear_sum_assignment

# Compatibility for chumpy/smpl pkl loading on newer NumPy.
# chumpy does: `from numpy import bool, int, float, complex, object, unicode, str, ...`
for _name, _value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if not hasattr(np, "unicode"):
    np.unicode = str

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SMPL_MODEL_PATHS = {
    "female": str(PROJECT_ROOT / "smpl_models" / "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
    "male": str(PROJECT_ROOT / "smpl_models" / "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    "neutral": str(PROJECT_ROOT / "smpl_models" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
}
_SMPL_MODEL_CACHE = {}

# SMPL-X models used for MAMMA-style supervision (use_mamma=True).  These are the
# same per-gender NPZ files (J_regressor/shapedirs/posedirs/weights/v_template/f)
# consumed by Scripts/process_mamma_syn.py's SMPLModel; we reimplement that exact
# forward pass in torch (see _TorchSMPLX) so predicted joints/vertices follow the
# identical convention to the MAMMA ground truth.
_SMPLX_MODEL_PATHS = {
    "female": str(PROJECT_ROOT / "smplx_models" / "female" / "model.pkl"),
    "male": str(PROJECT_ROOT / "smplx_models" / "male" / "model.pkl"),
    "neutral": str(PROJECT_ROOT / "smplx_models" / "neutral" / "model.pkl"),
}
_SMPLX_MODEL_CACHE = {}



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


def _normalize_gender_string(gender: str) -> str:
    token = str(gender).strip().lower()
    if token.startswith("m"):
        return "male"
    if token.startswith("f"):
        return "female"
    return "neutral"


def _map_gender_value(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size:
            value = value.reshape(-1)[0]
        else:
            return "neutral"
    try:
        v = int(value)
    except Exception:
        return "neutral"
    if v == 0:
        return "male"
    if v == 1:
        return "female"
    return "neutral"


def _resolve_batch_genders(batch, batch_size: int) -> list[str]:
    gender = batch.get("smpl_gender", None)
    if gender is None:
        gender = batch.get("smpl_genders", None)
    if gender is None:
        return ["neutral"] * batch_size

    if torch.is_tensor(gender):
        gender = gender.detach().cpu().numpy()

    if isinstance(gender, (list, tuple)):
        if len(gender) == 1:
            return [_map_gender_value(gender[0])] * batch_size
        if len(gender) == batch_size:
            return [_map_gender_value(g) for g in gender]

    if isinstance(gender, np.ndarray):
        if gender.ndim == 0:
            return [_map_gender_value(gender)] * batch_size
        if gender.size == 1:
            return [_map_gender_value(gender.reshape(-1)[0])] * batch_size
        if gender.size == batch_size:
            return [_map_gender_value(g) for g in gender.reshape(-1)]

    return [_map_gender_value(gender)] * batch_size


def _get_smpl_model(device: torch.device, gender: str):
    """Lazily load and cache a SMPL model on the given device/gender."""
    gender_key = _normalize_gender_string(gender)
    key = f"{device}:{gender_key}"
    if key in _SMPL_MODEL_CACHE:
        return _SMPL_MODEL_CACHE[key]

    model_path = Path(_SMPL_MODEL_PATHS[gender_key])
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    try:
        from smplx import SMPL  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "smplx is required for SMPL joints3d loss. Install smplx or set weight_joints3d=0."
        ) from exc

    smpl_model = SMPL(model_path=str(model_path), gender=gender_key, batch_size=1).to(device)

    smpl_model.eval()
    _SMPL_MODEL_CACHE[key] = smpl_model
    return smpl_model


class _TorchSMPLX:
    """Differentiable SMPL-X forward, a torch port of process_mamma_syn.py SMPLModel.

    Loads the per-gender NPZ (J_regressor / shapedirs / posedirs / weights /
    v_template / kintree_table) and reproduces SMPLModel.joints_world +
    SMPLModel.lbs_vertices so that predicted joints/vertices match the MAMMA
    ground truth exactly:

      * input pose is the 72-dim SMPL pose, zero-padded to SMPL-X's 165-dim
        (hand/jaw/eye pose stays zero, identical to the data-generation script);
      * first 10 betas drive the shaped template;
      * 24 SMPL-convention joints are returned (0-21 body, 22<-20 left wrist,
        23<-21 right wrist) alongside 10475 world-space vertices.
    """

    NUM_BETAS = 10

    @staticmethod
    def _to_np(v) -> np.ndarray:
        """Convert chumpy / scipy-sparse / plain array to float64 ndarray.

        Mirrors SMPLModel._to_np in Scripts/process_mamma_syn.py so the .pkl
        models (chumpy-backed) load identically to the .npz ones.
        """
        if hasattr(v, "r"):          # chumpy Ch object
            return np.array(v.r, dtype=np.float64)
        if hasattr(v, "todense"):    # scipy sparse matrix
            return np.array(v.todense(), dtype=np.float64)
        return np.asarray(v, dtype=np.float64)

    def __init__(self, model_path: str, device: torch.device):
        if model_path.endswith(".npz"):
            p = np.load(model_path, allow_pickle=False)
            get = lambda k: np.asarray(p[k], dtype=np.float64)
        else:
            import pickle
            with open(model_path, "rb") as f:
                p = pickle.load(f, encoding="latin1")
            get = lambda k: self._to_np(p[k])

        self.device = device
        f32 = torch.float32

        def _t(arr):
            return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device, dtype=f32)

        self.J_regressor = _t(get("J_regressor"))                     # (55, V)
        self.v_template = _t(get("v_template"))                       # (V, 3)
        self.weights = _t(get("weights"))                             # (V, 55)
        self.posedirs = _t(get("posedirs"))                           # (V, 3, 486)
        shapedirs = _t(get("shapedirs"))                              # (V, 3, 400)
        self.shapedirs = shapedirs[:, :, : self.NUM_BETAS].contiguous()  # (V, 3, 10)
        kt = get("kintree_table").astype(int)                         # (2, 55)
        self.n_joints = kt.shape[1]
        id2col = {kt[1, i]: i for i in range(kt.shape[1])}
        # parents[i] is the column index of joint i's parent (parents[0] unused).
        self.parents = [0] + [id2col[kt[0, i]] for i in range(1, kt.shape[1])]

    @staticmethod
    def _make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Pack rotation (B,3,3) + translation (B,3) into a homogeneous (B,4,4)."""
        B = R.shape[0]
        top = torch.cat([R, t.unsqueeze(-1)], dim=-1)                 # (B,3,4)
        bottom = torch.zeros(B, 1, 4, device=R.device, dtype=R.dtype)
        bottom[..., 3] = 1.0
        return torch.cat([top, bottom], dim=1)                        # (B,4,4)

    def __call__(self, pose72: torch.Tensor, betas: torch.Tensor,
                 trans: torch.Tensor, with_vertices: bool = True):
        device, dtype = self.device, torch.float32
        pose72 = pose72.to(device=device, dtype=dtype)
        betas = betas[:, : self.NUM_BETAS].to(device=device, dtype=dtype)
        trans = trans.to(device=device, dtype=dtype)
        B, n = pose72.shape[0], self.n_joints

        # Zero-pad the 72-dim SMPL pose up to SMPL-X's 165 dims (matches the
        # data-generation script: hand/jaw/eye pose remain zero).
        npose = min(72, pose72.shape[1])
        pose165 = torch.zeros(B, n * 3, device=device, dtype=dtype)
        pose165[:, :npose] = pose72[:, :npose]

        v_shaped = self.v_template.unsqueeze(0) + torch.einsum(
            "vck,bk->bvc", self.shapedirs, betas
        )                                                             # (B,V,3)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)   # (B,55,3)
        R = axis_angle_to_rotmat(pose165)                             # (B,55,3,3)

        # Forward kinematics: G[i] = G[parent] @ local(R[i], J[i]-J[parent]).
        G = [self._make_transform(R[:, 0], J[:, 0])]
        for i in range(1, n):
            loc = self._make_transform(R[:, i], J[:, i] - J[:, self.parents[i]])
            G.append(torch.bmm(G[self.parents[i]], loc))
        G = torch.stack(G, dim=1)                                     # (B,55,4,4)

        all_joints = G[:, :, :3, 3] + trans.unsqueeze(1)             # (B,55,3)
        joints24 = torch.cat(
            [all_joints[:, :22], all_joints[:, 20:21], all_joints[:, 21:22]],
            dim=1,
        )                                                             # (B,24,3)

        if not with_vertices:
            return joints24, None

        eye3 = torch.eye(3, device=device, dtype=dtype)
        lrotmin = (R[:, 1:] - eye3).reshape(B, -1)                   # (B,486)
        v_posed = v_shaped + torch.einsum("vck,bk->bvc", self.posedirs, lrotmin)

        J_h = torch.cat([J, torch.zeros(B, n, 1, device=device, dtype=dtype)], dim=-1)
        G_J = torch.einsum("bnij,bnj->bni", G, J_h)                  # (B,55,4)
        zero_col = torch.zeros_like(G_J)
        G_pack = torch.stack([zero_col, zero_col, zero_col, G_J], dim=-1)  # (B,55,4,4)
        G_lbs = G - G_pack
        T = torch.einsum("vj,bjmk->bvmk", self.weights, G_lbs)       # (B,V,4,4)
        v_h = torch.cat(
            [v_posed, torch.ones(B, v_posed.shape[1], 1, device=device, dtype=dtype)],
            dim=-1,
        )
        v_out = torch.einsum("bvmk,bvk->bvm", T, v_h)[..., :3]       # (B,V,3)
        verts = v_out + trans.unsqueeze(1)
        return joints24, verts


def _get_smplx_model(device: torch.device, gender: str) -> "_TorchSMPLX":
    """Lazily load and cache a torch SMPL-X model on the given device/gender."""
    gender_key = _normalize_gender_string(gender)
    key = f"{device}:{gender_key}"
    if key in _SMPLX_MODEL_CACHE:
        return _SMPLX_MODEL_CACHE[key]

    model_path = Path(_SMPLX_MODEL_PATHS[gender_key])
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model file not found: {model_path}")

    model = _TorchSMPLX(str(model_path), device)
    _SMPLX_MODEL_CACHE[key] = model
    return model


def _decode_smplx_batch(
    pose_aa: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    genders: list[str],
):
    """SMPL-X variant of _decode_smpl_batch (used when use_mamma=True).

    Returns (joints_world (B,24,3), vertices (B,10475,3)) using the torch SMPL-X
    forward that mirrors the MAMMA data-generation script.
    """
    B = pose_aa.shape[0]
    if B == 0:
        return None, None

    pred_joints_world = None
    pred_vertices = None

    for gender_key in ("male", "female", "neutral"):
        idx_list = [i for i, g in enumerate(genders) if g == gender_key]
        if not idx_list:
            continue
        idx = torch.tensor(idx_list, device=pose_aa.device, dtype=torch.long)

        model = _get_smplx_model(pose_aa.device, gender_key)
        joints_g, verts_g = model(
            pose_aa[idx], betas[idx], trans[idx], with_vertices=True
        )
        joints_g = joints_g.to(dtype=pose_aa.dtype)
        verts_g = verts_g.to(dtype=pose_aa.dtype)

        if pred_joints_world is None:
            pred_joints_world = torch.zeros(
                (B, joints_g.shape[1], 3), device=pose_aa.device, dtype=pose_aa.dtype
            )
            pred_vertices = torch.zeros(
                (B, verts_g.shape[1], 3), device=pose_aa.device, dtype=pose_aa.dtype
            )

        pred_joints_world[idx] = joints_g
        pred_vertices[idx] = verts_g

    return pred_joints_world, pred_vertices


def _decode_smpl_batch(
    pose_aa: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor | None,
    genders: list[str],
    use_mamma: bool = False,
):
    if use_mamma:
        return _decode_smplx_batch(pose_aa, betas, trans, genders)

    B = pose_aa.shape[0]
    if B == 0:
        return None, None

    pred_joints_world = None
    pred_vertices_smpl = None

    for gender_key in ("male", "female", "neutral"):
        idx_list = [i for i, g in enumerate(genders) if g == gender_key]
        if not idx_list:
            continue
        idx = torch.tensor(idx_list, device=pose_aa.device, dtype=torch.long)

        smpl_model = _get_smpl_model(
            pose_aa.device,
            gender_key,
        )
        gender_pose = pose_aa[idx]
        gender_beta = betas[idx]
        gender_trans = trans[idx]
        gender_body_pose = gender_pose[:, 3:72]

        smpl_kwargs = {
            "global_orient": gender_pose[:, :3].to(dtype=torch.float32),
            "body_pose": gender_body_pose.to(dtype=torch.float32),
            "betas": gender_beta.to(dtype=torch.float32),
            "transl": gender_trans.to(dtype=torch.float32),
        }
        smpl_out = smpl_model(**smpl_kwargs)
        # logging.info(f"_decode_smpl_batch global_orient: {gender_pose[:, :3].detach().cpu().numpy()}")
        # logging.info(f"_decode_smpl_batch body_pose: {gender_pose[:, 3:].detach().cpu().numpy()}")
        # logging.info(f"_decode_smpl_batch betas: {gender_beta.detach().cpu().numpy()}")
        # logging.info(f"_decode_smpl_batch transl: {gender_trans.detach().cpu().numpy()}")

        joints_smpl = smpl_out.joints[:, :24, :].to(dtype=pose_aa.dtype)
        vertices_smpl = smpl_out.vertices.to(dtype=pose_aa.dtype)
        # logging.info(f"_decode_smpl_batch joints_smpl: {joints_smpl.detach().cpu().numpy()}")

        if pred_joints_world is None:
            pred_joints_world = torch.zeros(
                (B, joints_smpl.shape[1], 3),
                device=pose_aa.device,
                dtype=pose_aa.dtype,
            )
            pred_vertices_smpl = torch.zeros(
                (B, vertices_smpl.shape[1], 3),
                device=pose_aa.device,
                dtype=pose_aa.dtype,
            )

        pred_joints_world[idx] = joints_smpl
        pred_vertices_smpl[idx] = vertices_smpl

    return pred_joints_world, pred_vertices_smpl



def axis_angle_to_rotmat(pose_aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    將 axis-angle pose 轉成旋轉矩陣。

    pose_aa: (..., J * 3)，最後一維是 J 個 axis-angle 關節
    回傳:    (..., J, 3, 3)
    """
    if pose_aa.shape[-1] % 3 != 0:
        raise ValueError(
            f"axis_angle_to_rotmat expects the last dim to be a multiple of 3, got shape {tuple(pose_aa.shape)}"
        )

    num_joints = pose_aa.shape[-1] // 3

    # orig_shape 不含最後 J*3 那一維，例如 (B, S)
    orig_shape = pose_aa.shape[:-1]
    device = pose_aa.device
    dtype = pose_aa.dtype

    # 攤平成 [..., J, 3]
    pose_aa = pose_aa.reshape(*orig_shape, num_joints, 3)   # (..., J, 3)

    # 角度: (..., J)
    angle = torch.norm(pose_aa, dim=-1)          # 不要 keepdim
    # 單位軸: (..., J, 3)
    axis = pose_aa / (angle.unsqueeze(-1) + eps)

    # cos / sin / (1 - cos): (..., J)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C  = 1.0 - ca

    # x, y, z: (..., J)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    # 建 R: (..., J, 3, 3)
    R = torch.empty(*orig_shape, num_joints, 3, 3, device=device, dtype=dtype)

    R[..., 0, 0] = ca + x * x * C
    R[..., 0, 1] = x * y * C - z * sa
    R[..., 0, 2] = x * z * C + y * sa

    R[..., 1, 0] = y * x * C + z * sa
    R[..., 1, 1] = ca + y * y * C
    R[..., 1, 2] = y * z * C - x * sa

    R[..., 2, 0] = z * x * C - y * sa
    R[..., 2, 1] = z * y * C + x * sa
    R[..., 2, 2] = ca + z * z * C

    return R


def rotmat_to_axis_angle(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse of axis_angle_to_rotmat for a single rotation per element.

    R: (..., 3, 3) -> axis-angle (..., 3). Uses Shepperd's method (rotmat ->
    quaternion -> axis-angle) which is numerically stable for all angles,
    including near pi.
    """
    orig_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    m21, m12 = R[:, 2, 1], R[:, 1, 2]
    m02, m20 = R[:, 0, 2], R[:, 2, 0]
    m10, m01 = R[:, 1, 0], R[:, 0, 1]
    trace = m00 + m11 + m22

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)  # (w, x, y, z)
    cond0 = trace > 0
    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    cond2 = (~cond0) & (~cond1) & (m11 >= m22)
    cond3 = (~cond0) & (~cond1) & (~cond2)

    # branch 0: trace > 0
    if cond0.any():
        s = torch.sqrt((trace[cond0] + 1.0).clamp(min=eps)) * 2.0
        q[cond0, 0] = 0.25 * s
        q[cond0, 1] = (m21[cond0] - m12[cond0]) / s
        q[cond0, 2] = (m02[cond0] - m20[cond0]) / s
        q[cond0, 3] = (m10[cond0] - m01[cond0]) / s
    if cond1.any():
        s = torch.sqrt((1.0 + m00[cond1] - m11[cond1] - m22[cond1]).clamp(min=eps)) * 2.0
        q[cond1, 0] = (m21[cond1] - m12[cond1]) / s
        q[cond1, 1] = 0.25 * s
        q[cond1, 2] = (m01[cond1] + m10[cond1]) / s
        q[cond1, 3] = (m02[cond1] + m20[cond1]) / s
    if cond2.any():
        s = torch.sqrt((1.0 + m11[cond2] - m00[cond2] - m22[cond2]).clamp(min=eps)) * 2.0
        q[cond2, 0] = (m02[cond2] - m20[cond2]) / s
        q[cond2, 1] = (m01[cond2] + m10[cond2]) / s
        q[cond2, 2] = 0.25 * s
        q[cond2, 3] = (m12[cond2] + m21[cond2]) / s
    if cond3.any():
        s = torch.sqrt((1.0 + m22[cond3] - m00[cond3] - m11[cond3]).clamp(min=eps)) * 2.0
        q[cond3, 0] = (m10[cond3] - m01[cond3]) / s
        q[cond3, 1] = (m02[cond3] + m20[cond3]) / s
        q[cond3, 2] = (m12[cond3] + m21[cond3]) / s
        q[cond3, 3] = 0.25 * s

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    w = q[:, 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)                      # (N,)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=0.0))
    small = sin_half < 1e-6
    axis = q[:, 1:] / torch.where(small, torch.ones_like(sin_half), sin_half).unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)
    aa = torch.where(small.unsqueeze(-1), torch.zeros_like(aa), aa)
    return aa.reshape(*orig_shape, 3)


def scale_joints_to_batch_gauge(joints: torch.Tensor, avg_scale: torch.Tensor) -> torch.Tensor:
    """Scale-only variant of normalize_joints_world_to_batch_gauge (no cam0 rotation).

    Used in mesh_rot mode, where the SMPL body is decoded with global_orient set to
    the predicted/GT mesh_rot and is therefore already oriented in the cam0 frame;
    only the avg_scale gauge division remains to be applied.
    """
    view = (-1,) + (1,) * (joints.dim() - 1)
    return joints / avg_scale.view(*view).clamp(min=1e-6)


def compute_gt_mesh_rot(batch) -> torch.Tensor:
    """GT mesh_rot (axis-angle, B,P,3): SMPL root rotation in the cam0 frame.

    mesh_rot = R0 @ R_global_orient_world, where R0 = raw_extrinsics[:,0,:3,:3]
    (world->cam0 rotation) and R_global_orient_world = the world-frame SMPL root.
    """
    R0 = batch["raw_extrinsics"][:, 0, :3, :3].float()      # (B,3,3)
    go = batch["smpl_pose"][..., :3].float()                # (B,P,3) world global_orient
    B, P = go.shape[:2]
    R_go = axis_angle_to_rotmat(go.reshape(B * P, 3)).reshape(B, P, 3, 3)  # (B,P,3,3)
    R_mesh = R0[:, None] @ R_go                             # (B,P,3,3)
    return rotmat_to_axis_angle(R_mesh.reshape(B * P, 3, 3)).reshape(B, P, 3)


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss for presence/objectness logits.

    Normalizes by the number of positive slots, matching dense detection losses
    more closely than averaging over all positive and negative slots.
    `alpha` can optionally be used as the positive-class weight.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = (1.0 - p_t).clamp(min=0.0).pow(gamma)

    if alpha is not None and alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = focal_weight * alpha_t

    loss = focal_weight * bce
    num_pos = (targets > 0.5).to(dtype=loss.dtype).sum()
    if num_pos > 0:
        return loss.sum() / num_pos.clamp(min=1.0)
    return loss.sum()


def smpl_losses_plus_from_axis_angle(
    pred_pose_aa: torch.Tensor,
    pred_beta: torch.Tensor,
    gt_pose_aa: torch.Tensor,
    gt_beta: torch.Tensor,
    pred_pose_aa_0: torch.Tensor | None = None,
    pose_weight: float = 1.0,
    beta_weight: float = 1.0,
    init_w: float = 1.0,
    loss_type: str = "l1",
    has_smpl: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """TRAM-style SMPL+ loss but with axis-angle inputs.

    - GT only needs pose (axis-angle) + beta.
    - If pred_pose_aa_0 is provided, adds an "init" pose loss weighted by init_w.

    Shapes supported:
      pose: (B,72) or (B,S,72)
      beta: (B,10) or (B,S,10)
      has_smpl: (B,) or (B,S) boolean/0-1
    """

    def _ensure_BS(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    pred_pose_aa = _ensure_BS(pred_pose_aa)
    pred_beta = _ensure_BS(pred_beta)
    gt_pose_aa = _ensure_BS(gt_pose_aa)
    gt_beta = _ensure_BS(gt_beta)

    if pred_pose_aa_0 is None:
        pred_pose_aa_0 = pred_pose_aa
    else:
        pred_pose_aa_0 = _ensure_BS(pred_pose_aa_0)

    if has_smpl is None:
        has_smpl = torch.ones(pred_pose_aa.shape[:2], device=pred_pose_aa.device, dtype=pred_pose_aa.dtype)
    else:
        if has_smpl.dim() == 1:
            has_smpl = has_smpl[:, None]
        has_smpl = has_smpl.to(device=pred_pose_aa.device, dtype=pred_pose_aa.dtype)

    # rotmat regression (Frobenius) on 24 joints
    R_pred = axis_angle_to_rotmat(pred_pose_aa)
    R_gt = axis_angle_to_rotmat(gt_pose_aa)
    R_pred0 = axis_angle_to_rotmat(pred_pose_aa_0)

    pose_loss_final = ((R_pred - R_gt) ** 2).sum(dim=(-1, -2)).mean(dim=-1)  # (B,S)
    pose_loss_init = ((R_pred0 - R_gt) ** 2).sum(dim=(-1, -2)).mean(dim=-1)  # (B,S)

    if loss_type == "l1":
        beta_loss = (pred_beta - gt_beta).abs().mean(dim=-1)  # (B,S)
    elif loss_type == "l2":
        beta_loss = ((pred_beta - gt_beta) ** 2).mean(dim=-1)
    else:
        raise ValueError(f"Unknown SMPL loss_type: {loss_type}")

    # apply has_smpl mask
    denom = has_smpl.sum().clamp(min=1.0)
    loss_pose_final = (pose_loss_final * has_smpl).sum() / denom
    loss_pose_init = (pose_loss_init * has_smpl).sum() / denom
    loss_beta = (beta_loss * has_smpl).sum() / denom

    total = pose_weight * loss_pose_final + pose_weight * init_w * loss_pose_init + beta_weight * loss_beta
    return total, {
        "loss_smpl_plus": total,
        "loss_smpl_plus_pose": loss_pose_final,
        "loss_smpl_plus_pose_init": loss_pose_init,
        "loss_smpl_plus_beta": loss_beta,
    }


def compute_gt_mesh_translate(
    batch,
    normalize_cam: bool = True,
    use_mamma: bool = False,
) -> torch.Tensor:
    gt_pose = batch["smpl_pose"]
    gt_beta = batch["smpl_beta"]
    gt_trans = batch["smpl_trans"]

    original_shape = gt_pose.shape[:-1]
    pose_flat = gt_pose.reshape(-1, gt_pose.shape[-1])
    beta_flat = gt_beta.reshape(-1, gt_beta.shape[-1])
    trans_flat = gt_trans.reshape(-1, gt_trans.shape[-1])
    zero_trans_flat = torch.zeros_like(trans_flat)

    genders = _resolve_batch_genders(batch, pose_flat.shape[0])
    if len(genders) != pose_flat.shape[0]:
        genders = ["neutral"] * pose_flat.shape[0]

    with torch.no_grad():
        zero_joints_flat, _ = _decode_smpl_batch(
            pose_aa=pose_flat,
            betas=beta_flat,
            trans=zero_trans_flat,
            genders=genders,
            use_mamma=use_mamma,
        )
    if zero_joints_flat is None:
        raise ValueError("Failed to decode zero-translation SMPL joints for mesh_translate target")

    zero_root_world = zero_joints_flat[:, 0, :].reshape(*original_shape, 3)
    target_root_world = zero_root_world + gt_trans.reshape(*original_shape, 3)
    if normalize_cam:
        if "raw_extrinsics" not in batch or "avg_scale" not in batch:
            raise KeyError("raw_extrinsics and avg_scale are required for normalized mesh_translate GT")
        return normalize_joints_world_to_batch_gauge(
            target_root_world,
            batch["raw_extrinsics"],
            batch["avg_scale"],
        )
    return target_root_world


def compute_smpl_loss(
    predictions,
    batch,
    loss_type: str = "l1",
    loss_type_joints2d: str | None = None,
    loss_type_joints3d: str | None = None,
    weight_pose: float = 1.0,
    weight_beta: float = 0.1,
    weight_trans: float = 0.0,
    weight_mesh_translate: float = 0.0,
    weight_presence: float = 1.0,
    weight_joints2d: float = 0.0,
    weight_joints3d: float = 0.0,
    weight_vertices: float = 0.0,
    use_gt: bool = False,
    normalize_cam: bool = True,
    use_hungarian: bool = False,
    hungarian_cost_pose_weight: float = 1.0,
    hungarian_cost_beta_weight: float = 0.1,
    hungarian_cost_trans_weight: float = 0.0,
    hungarian_cost_mesh_trans_weight: float = 0.0,
    hungarian_cost_presence_weight: float = 0.0,
    use_mamma: bool = False,
    **kwargs,
):
    """
    SMPL supervision:
    - predictions["smpl_pose"]: (B, 72), axis-angle
    - predictions["smpl_beta"]: (B, 10)
    - predictions["smpl_trans"]: (B, S, 1, 3)
    - predictions["smpl_presence_logits"]: optional multi-person slot logits (B, P)
    - batch["smpl_pose"]:      (B, 72)
    - batch["smpl_beta"]:      (B, 10)
    - batch["smpl_trans"]: (B, S, 1, 3)

    joint confidence masks (only joints with confidence==1 contribute):
    - batch["smpl_joints2d_confidence"]: (B, S, J) / (B, J) / (S, J)

    """
    # Multi-person samples use shape [B, P, ...] for SMPL params and
    # [B, S, P, ...] for per-view joints.  Flatten people into the batch axis so
    # the existing single-person loss path can supervise every person jointly.
    matching_cost_metrics = {
        "presence_cost": predictions["smpl_pose"].new_zeros(()),
    }
    if "mesh_translate" in predictions and "mesh_translate" not in batch:
        batch = dict(batch)
        # fp32: this derives the GT root (mesh_translate) via SMPL decode + gauge
        # normalize; under bf16 autocast its value is rounded, which then shifts every
        # projected joint and inflates loss_smpl_joints2d. Keep it fp32.
        with torch.cuda.amp.autocast(enabled=False):
            batch["mesh_translate"] = compute_gt_mesh_translate(
                batch,
                normalize_cam=normalize_cam,
                use_mamma=use_mamma,
            )

    # mesh_rot mode: the head predicts smpl_pose[:3] as the cam0-frame mesh root
    # rotation. Replace the GT global_orient (world frame) with the GT mesh_rot so
    # that Hungarian matching AND the pose loss compare like-for-like (cam0 frame).
    # Must run BEFORE matching and BEFORE overwriting any pose value.
    use_mesh_rot = predictions.get("mesh_rot", None) is not None
    if use_mesh_rot:
        if "raw_extrinsics" not in batch:
            raise KeyError("mesh_rot mode requires batch['raw_extrinsics'] (set normalize_cam=True)")
        batch = dict(batch)
        with torch.cuda.amp.autocast(enabled=False):
            gt_mesh_rot = compute_gt_mesh_rot(batch)  # (B,P,3) axis-angle, cam0 frame
        gt_pose_new = batch["smpl_pose"].clone()
        gt_pose_new[..., :3] = gt_mesh_rot.to(gt_pose_new.dtype)
        batch["smpl_pose"] = gt_pose_new
        batch["mesh_rot"] = gt_mesh_rot

    has_people_axis = "num_people" in batch or any(
        key in batch and batch[key].dim() >= 5
        for key in ("smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence")
    )
    if predictions["smpl_pose"].dim() == 3 and has_people_axis:
        if use_hungarian:
            predictions, batch, matching_cost_metrics = apply_hungarian_matching(
                predictions,
                batch,
                cost_pose_weight=hungarian_cost_pose_weight,
                cost_beta_weight=hungarian_cost_beta_weight,
                cost_trans_weight=hungarian_cost_trans_weight,
                cost_mesh_trans_weight=hungarian_cost_mesh_trans_weight,
                cost_presence_weight=hungarian_cost_presence_weight,
                return_cost_metrics=True,
                use_mamma=use_mamma,
            )

        B_people, P_people = predictions["smpl_pose"].shape[:2]

        def _pad_people_param(tensor, fill_value: float = 0.0):
            if tensor is None:
                return None
            if tensor.dim() >= 3 and tensor.shape[0] == B_people and tensor.shape[1] < P_people:
                pad_shape = list(tensor.shape)
                pad_shape[1] = P_people - tensor.shape[1]
                pad = torch.full(
                    pad_shape,
                    fill_value,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                return torch.cat([tensor, pad], dim=1)
            return tensor

        def _pad_people_view_tensor(tensor, fill_value: float = 0.0):
            if tensor is None:
                return None
            if tensor.dim() >= 4 and tensor.shape[0] == B_people and tensor.shape[2] < P_people:
                pad_shape = list(tensor.shape)
                pad_shape[2] = P_people - tensor.shape[2]
                pad = torch.full(
                    pad_shape,
                    fill_value,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                return torch.cat([tensor, pad], dim=2)
            return tensor

        batch = dict(batch)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate"):
            if key in batch:
                batch[key] = _pad_people_param(batch[key])
        for key in (
            "smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence",
            "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask",
        ):
            if key in batch:
                batch[key] = _pad_people_view_tensor(batch[key])
        if "smpl_gender" in batch and batch["smpl_gender"].dim() >= 2 and batch["smpl_gender"].shape[1] < P_people:
            gender = batch["smpl_gender"]
            pad = torch.full(
                (B_people, P_people - gender.shape[1]),
                2,
                device=gender.device,
                dtype=gender.dtype,
            )
            batch["smpl_gender"] = torch.cat([gender, pad], dim=1)
        if "has_smpl" in batch and batch["has_smpl"].dim() >= 2 and batch["has_smpl"].shape[1] < P_people:
            has_smpl = batch["has_smpl"]
            pad = torch.zeros(
                (B_people, P_people - has_smpl.shape[1]),
                device=has_smpl.device,
                dtype=has_smpl.dtype,
            )
            batch["has_smpl"] = torch.cat([has_smpl, pad], dim=1)

        def _flatten_people_param(tensor):
            if tensor is None:
                return None
            if tensor.dim() == 3 and tensor.shape[:2] == (B_people, P_people):
                return tensor.reshape(B_people * P_people, *tensor.shape[2:])
            return tensor

        def _flatten_people_view_tensor(tensor):
            if tensor is None:
                return None
            if tensor.dim() >= 4 and tensor.shape[0] == B_people and tensor.shape[2] == P_people:
                permute_order = [0, 2, 1] + list(range(3, tensor.dim()))
                tensor = tensor.permute(*permute_order).contiguous()
                return tensor.reshape(B_people * P_people, *tensor.shape[2:])
            return tensor

        def _repeat_per_person_view_tensor(tensor):
            if tensor is None:
                return None
            if tensor.shape[0] != B_people:
                return tensor
            expanded = tensor[:, None].expand(
                B_people,
                P_people,
                *tensor.shape[1:],
            )
            return expanded.reshape(B_people * P_people, *tensor.shape[1:])

        predictions = dict(predictions)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate", "pred_pose_0", "smpl_pose_0", "smpl_pose_init"):
            if key in predictions:
                predictions[key] = _flatten_people_param(predictions[key])
        if "smpl_presence_logits" in predictions and predictions["smpl_presence_logits"] is not None:
            presence_logits = predictions["smpl_presence_logits"]
            if presence_logits.dim() == 2 and presence_logits.shape == (B_people, P_people):
                predictions["smpl_presence_logits"] = presence_logits.reshape(B_people * P_people)
        if "pose_enc_list" in predictions:
            predictions["pose_enc_list"] = [
                _repeat_per_person_view_tensor(pose_enc)
                for pose_enc in predictions["pose_enc_list"]
            ]
        # dense-landmark / mask head outputs -> (B*P, S, ...)
        for key in ("smpl_landmarks2d", "smpl_landmarks_logvar", "person_mask_logits"):
            if key in predictions and predictions[key] is not None:
                predictions[key] = _flatten_people_view_tensor(predictions[key])

        batch = dict(batch)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate"):
            if key in batch:
                batch[key] = _flatten_people_param(batch[key])
        for key in ("smpl_joints2d", "smpl_joints3d_world",
                    "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask"):
            if key in batch:
                batch[key] = _flatten_people_view_tensor(batch[key])
        if "smpl_joints2d_confidence" in batch:
            batch["smpl_joints2d_confidence"] = _flatten_people_view_tensor(
                batch["smpl_joints2d_confidence"]
            )
        for key in ("images", "extrinsics", "intrinsics", "raw_extrinsics", "point_masks"):
            if key in batch:
                batch[key] = _repeat_per_person_view_tensor(batch[key])
        if "avg_scale" in batch and batch["avg_scale"].shape[0] == B_people:
            batch["avg_scale"] = batch["avg_scale"][:, None].expand(B_people, P_people).reshape(-1)
        if "smpl_gender" in batch and batch["smpl_gender"].dim() >= 2:
            batch["smpl_gender"] = batch["smpl_gender"].reshape(B_people * P_people)
        if "has_smpl" in batch and batch["has_smpl"].dim() >= 2:
            batch["has_smpl"] = batch["has_smpl"].reshape(B_people * P_people)

    pred_pose = predictions["smpl_pose"]   # [B, 72]
    pred_beta = predictions["smpl_beta"]   # [B, 10]

    gt_pose = batch["smpl_pose"]           # [B, 72]
    gt_beta = batch["smpl_beta"]           # [B, 10]

    # SMPL translation losses
    pred_trans = predictions.get("smpl_trans", None)
    gt_trans = batch.get("smpl_trans", None)
    pred_mesh_translate = predictions.get("mesh_translate", None)
    gt_mesh_translate = batch.get("mesh_translate", None)

    if use_gt:
        pred_pose = gt_pose
        pred_beta = gt_beta
        pred_trans = gt_trans
        pred_mesh_translate = gt_mesh_translate

    # ---- 保證都有 time 維度 S ----
    if pred_pose.dim() == 2:   # [B,72] -> [B,1,72]
        pred_pose = pred_pose.unsqueeze(1)
    if pred_beta.dim() == 2:   # [B,10] -> [B,1,10]
        pred_beta = pred_beta.unsqueeze(1)

    if gt_pose.dim() == 2:     # [B,72] -> [B,1,72]
        gt_pose = gt_pose.unsqueeze(1)
    if gt_beta.dim() == 2:     # [B,10] -> [B,1,10]
        gt_beta = gt_beta.unsqueeze(1)

    def _align_trans_to_pose(trans, pose, name: str):
        if trans is None:
            raise ValueError(f"{name} is required")

        # Accept [B,3], [B,T,3], [B,T,1,3], and occasional loader/model variants.
        if trans.dim() == 2:
            trans = trans.unsqueeze(1)
        elif trans.dim() == 4:
            if trans.shape[-2] == 1:
                trans = trans.squeeze(-2)
            else:
                trans = trans[:, :, 0, :]
        elif trans.dim() != 3:
            raise ValueError(f"Unsupported {name} shape: {tuple(trans.shape)}")

        if trans.shape[-1] != 3:
            raise ValueError(f"{name} last dimension must be 3, got {tuple(trans.shape)}")

        target_t = pose.shape[1]
        if trans.shape[1] == target_t:
            return trans
        if trans.shape[1] == 1:
            return trans.expand(-1, target_t, -1)
        if target_t == 1:
            return trans[:, :1, :]

        raise ValueError(
            f"Cannot align {name} shape {tuple(trans.shape)} to pose shape {tuple(pose.shape)}"
        )

    if pred_trans is not None and gt_trans is not None:
        pred_trans = _align_trans_to_pose(pred_trans, pred_pose, "pred_trans")
        gt_trans = _align_trans_to_pose(gt_trans, gt_pose, "gt_trans")
    if pred_mesh_translate is not None and gt_mesh_translate is not None:
        pred_mesh_translate = _align_trans_to_pose(pred_mesh_translate, pred_pose, "pred_mesh_translate")
        gt_mesh_translate = _align_trans_to_pose(gt_mesh_translate, gt_pose, "gt_mesh_translate")

    smpl_decode_uses_mesh_translate = pred_mesh_translate is not None

    # ---- Keep ONLY: keypoint_loss (2D), keypoint_3d_loss (3D), smpl_losses (pose/beta), vertice_loss ----
    loss_joints2d = (pred_pose * 0.0).mean()
    loss_joints3d = (pred_pose * 0.0).mean()
    loss_vertices = (pred_pose * 0.0).mean()
    loss_smpl_losses = (pred_pose * 0.0).mean()
    loss_trans = (pred_pose * 0.0).mean()
    loss_mesh_translate = (pred_pose * 0.0).mean()
    loss_presence = (pred_pose * 0.0).mean()
    smpl_presence_positive_prob_mean = (pred_pose * 0.0).mean()
    smpl_presence_empty_prob_mean = (pred_pose * 0.0).mean()
    # 投影爆掉偵測:in-frame 的 normalized 2D 應在 [-1,1];若 mesh_translate 把人放到太靠近
    # 相機(深度 z→0),投影 Jacobian ~ f/z 爆大,座標 >> 1 → 這個值會飆高。
    smpl_joints2d_max_abs = (pred_pose * 0.0).mean()

    # NOTE: don't use Python `or` with tensors; it triggers boolean evaluation.
    pred_pose_0 = None
    if "pred_pose_0" in predictions and predictions["pred_pose_0"] is not None:
        pred_pose_0 = predictions["pred_pose_0"]
    elif "smpl_pose_0" in predictions and predictions["smpl_pose_0"] is not None:
        pred_pose_0 = predictions["smpl_pose_0"]
    elif "smpl_pose_init" in predictions and predictions["smpl_pose_init"] is not None:
        pred_pose_0 = predictions["smpl_pose_init"]
    init_w = float(kwargs.get("init_w", 1.0))
    has_smpl = batch.get("has_smpl", None)
    presence_loss_type = str(kwargs.get("presence_loss_type", "bce")).lower()
    presence_focal_alpha = kwargs.get("presence_focal_alpha", None)
    if presence_focal_alpha is not None:
        presence_focal_alpha = float(presence_focal_alpha)
    presence_focal_gamma = float(kwargs.get("presence_focal_gamma", 2.0))

    presence_logits = predictions.get("smpl_presence_logits", None)
    if weight_presence > 0.0 and presence_logits is not None and has_smpl is not None:
        presence_logits = presence_logits.to(dtype=pred_pose.dtype)
        presence_target = has_smpl.to(device=presence_logits.device, dtype=presence_logits.dtype)
        if presence_target.shape != presence_logits.shape:
            if presence_target.numel() == presence_logits.numel():
                presence_target = presence_target.reshape_as(presence_logits)
            elif presence_logits.dim() == 2 and presence_logits.shape[-1] == 1 and presence_target.shape == presence_logits.shape[:1]:
                presence_target = presence_target[:, None]
            else:
                raise ValueError(
                    f"Cannot align has_smpl shape {tuple(has_smpl.shape)} to smpl_presence_logits shape {tuple(presence_logits.shape)}"
                )
        presence_probs = torch.sigmoid(presence_logits)
        positive_mask = presence_target > 0.5
        empty_mask = ~positive_mask
        if positive_mask.any():
            smpl_presence_positive_prob_mean = presence_probs[positive_mask].mean()
        if empty_mask.any():
            smpl_presence_empty_prob_mean = presence_probs[empty_mask].mean()
        if presence_loss_type == "bce":
            loss_presence = F.binary_cross_entropy_with_logits(
                presence_logits,
                presence_target,
            )
        elif presence_loss_type == "focal":
            loss_presence = binary_focal_loss_with_logits(
                presence_logits,
                presence_target,
                alpha=presence_focal_alpha,
                gamma=presence_focal_gamma,
            )
        else:
            raise ValueError(f"Unknown presence_loss_type: {presence_loss_type}")

    #* SMPL losses (pose/beta) with optional init pose (pred_pose_0)
    if weight_pose > 0.0 or weight_beta > 0.0:
        pred_pose_for_smpl = pred_pose[..., :72]
        gt_pose_for_smpl = gt_pose[..., :72]
        pred_pose_0_for_smpl = pred_pose_0[..., :72] if pred_pose_0 is not None else None

        smpl_plus_total, _smpl_plus_dict = smpl_losses_plus_from_axis_angle(
            pred_pose_aa=pred_pose_for_smpl,
            pred_beta=pred_beta,
            gt_pose_aa=gt_pose_for_smpl,
            gt_beta=gt_beta,
            pred_pose_aa_0=pred_pose_0_for_smpl,
            pose_weight=weight_pose,
            beta_weight=weight_beta,
            init_w=init_w,
            loss_type=loss_type,
            has_smpl=has_smpl,
        )
        loss_smpl_losses = smpl_plus_total

    # Shared SMPL decode (only when needed)
    need_smpl_decode = (
        (weight_joints3d > 0.0 and "smpl_joints3d_world" in batch)
        or (weight_joints2d > 0.0 and "smpl_joints2d" in batch)
        or (weight_vertices > 0.0)
    )
    pred_joints_world = None
    pred_vertices_smpl = None
    if need_smpl_decode:
        B, T = pred_pose.shape[:2]
        pred_pose_flat = pred_pose.reshape(B * T, -1)
        pred_beta_flat = pred_beta.reshape(B * T, -1)
        if smpl_decode_uses_mesh_translate:
            pred_trans_flat = torch.zeros(B * T, 3, device=pred_pose.device, dtype=pred_pose.dtype)
        else:
            pred_trans_flat = pred_trans.reshape(B * T, -1)

        genders = _resolve_batch_genders(batch, B)
        if len(genders) == B:
            genders = [g for g in genders for _ in range(T)]

        # Decode SMPL in fp32: under bf16 autocast the LBS/joint regression matmuls
        # round to bf16, and that error (scaled by scene magnitude at projection time)
        # is what inflates loss_smpl_joints2d. Keep the whole geometry chain fp32.
        with torch.cuda.amp.autocast(enabled=False):
            decode_out = _decode_smpl_batch(
                pose_aa=pred_pose_flat.float(),
                betas=pred_beta_flat.float(),
                trans=pred_trans_flat.float(),
                genders=genders,
                use_mamma=use_mamma,
            )
        pred_joints_world_flat, pred_vertices_smpl_flat = decode_out
        if pred_joints_world_flat is not None:
            pred_joints_world = pred_joints_world_flat.reshape(B, T, pred_joints_world_flat.shape[1], 3)
        if pred_vertices_smpl_flat is not None:
            pred_vertices_smpl = pred_vertices_smpl_flat.reshape(
                B, T, pred_vertices_smpl_flat.shape[1], 3
            )
        if smpl_decode_uses_mesh_translate:
            if not normalize_cam:
                raise ValueError("mesh_translate predictions require normalize_cam=True")
            with torch.cuda.amp.autocast(enabled=False):
                pred_mesh_translate_aligned = pred_mesh_translate.to(device=pred_pose.device).float()
                raw_extr_f = batch["raw_extrinsics"].float()
                avg_scale_f = batch["avg_scale"].float()
                # mesh_rot mode: smpl_pose[:3] is the cam0-frame mesh_rot, so the
                # decoded body is already oriented in cam0 -> scale-only gauge (no R0
                # rotation). Otherwise rotate world->cam0 then scale.
                if pred_joints_world is not None:
                    if use_mesh_rot:
                        pred_joints_world = scale_joints_to_batch_gauge(pred_joints_world.float(), avg_scale_f)
                    else:
                        pred_joints_world = normalize_joints_world_to_batch_gauge(
                            pred_joints_world.float(),
                            raw_extr_f,
                            avg_scale_f,
                        )
                    # detach root anchor:切掉「所有關節的梯度疊加到 decode 出的 root」這條
                    # 放大路徑(dL/d(jw_root) 會含 -Σ_i dL/d(final_i)),它讓 Grad/smpl 隨訓練
                    # 爆增、灌爆共享 aggregator 而拖垮相機。forward 值不變(root 仍 anchor 到
                    # mesh_translate),只移除這條梯度放大。
                    joint_offset = pred_mesh_translate_aligned - pred_joints_world[..., 0, :].detach()
                    pred_joints_world = pred_joints_world + joint_offset.unsqueeze(-2)
                if pred_vertices_smpl is not None:
                    if use_mesh_rot:
                        pred_vertices_smpl = scale_joints_to_batch_gauge(pred_vertices_smpl.float(), avg_scale_f)
                    else:
                        pred_vertices_smpl = normalize_joints_world_to_batch_gauge(
                            pred_vertices_smpl.float(),
                            raw_extr_f,
                            avg_scale_f,
                        )
                    pred_vertices_smpl = pred_vertices_smpl + joint_offset.unsqueeze(-2)
    
    if weight_trans > 0.0 and pred_trans is not None and gt_trans is not None:
        if loss_type == "l1":
            diff = (pred_trans - gt_trans).abs().mean(dim=-1)
        elif loss_type == "l2":
            diff = ((pred_trans - gt_trans) ** 2).mean(dim=-1)
        else:
            raise ValueError(loss_type)

        if has_smpl is not None:
            trans_mask = has_smpl.to(device=diff.device, dtype=diff.dtype)
            if trans_mask.dim() == 1:
                trans_mask = trans_mask[:, None].expand_as(diff)
            loss_trans = (diff * trans_mask).sum() / trans_mask.sum().clamp(min=1.0)
        else:
            loss_trans = diff.mean()

    if weight_mesh_translate > 0.0 and pred_mesh_translate is not None and gt_mesh_translate is not None:
        if loss_type == "l1":
            diff = (pred_mesh_translate - gt_mesh_translate).abs().mean(dim=-1)
        elif loss_type == "l2":
            diff = ((pred_mesh_translate - gt_mesh_translate) ** 2).mean(dim=-1)
        else:
            raise ValueError(loss_type)

        if has_smpl is not None:
            mesh_translate_mask = has_smpl.to(device=diff.device, dtype=diff.dtype)
            if mesh_translate_mask.dim() == 1:
                mesh_translate_mask = mesh_translate_mask[:, None].expand_as(diff)
            loss_mesh_translate = (
                diff * mesh_translate_mask
            ).sum() / mesh_translate_mask.sum().clamp(min=1.0)
        else:
            loss_mesh_translate = diff.mean()
    
    #* pred 3d joints -> normalize by gt -> 3d joints loss (GT joints are already normalized in _process_batch)
    if weight_joints3d > 0.0 and "smpl_joints3d_world" in batch:
        gt_joints3d_world = batch["smpl_joints3d_world"]
        joints3d_loss_type = loss_type_joints3d or loss_type
        loss_joints3d = compute_smpl_3d_joint_loss(
            pred_joints_world[..., :24, :],
            gt_joints3d_world[..., :24, :],
            batch=batch,
            loss_type=joints3d_loss_type,
            pelvis_center=True,
            pelvis_ids=(1, 2),
            normalize_cam=False if smpl_decode_uses_mesh_translate else normalize_cam,
        )
    
    #* 2D keypoint reprojection loss
    # Requirement: use Unity-space AND gauge-normalized 3D joints, projected with camera-head predicted extrinsics/intrinsics.
    if weight_joints2d > 0.0 and "smpl_joints2d" in batch:
        if "pose_enc_list" not in predictions:
            raise KeyError("pose_enc_list not found in predictions; enable camera head to use joints2d loss")

        # NOTE: the reprojection below (gauge normalize -> FoV pose-encoding
        # round-trip -> projection) is run in fp32 even under bf16 autocast. In bf16
        # these matmuls accumulate rounding error that scales with the scene's
        # coordinate magnitude, which inflates loss_smpl_joints2d and makes it
        # unstable / not comparable across datasets. fp32 keeps the geometry exact.
        if normalize_cam and not smpl_decode_uses_mesh_translate:
            # Normalize pred joints to the same batch gauge as camera normalization.
            with torch.cuda.amp.autocast(enabled=False):
                pred_joints_norm = normalize_joints_world_to_batch_gauge(
                    pred_joints_world.float(),
                    batch["raw_extrinsics"].float(),
                    batch["avg_scale"].float(),
                )
        else:
            pred_joints_norm = pred_joints_world.float()

        # print(f"[CGV LOG] pred_joints_norm shape : {pred_joints_norm.shape}")

        pred_pose_enc = predictions["pose_enc_list"][-1]  # (B,S,9)
        # Optional: stop gradients from joints2d reprojection loss to camera head.
        # This keeps joints2d supervising SMPL (pose/beta) while not updating pose_enc.

        pred_pose_enc = pred_pose_enc.detach()
        image_hw = batch["images"].shape[-2:]
        pose_encoding_type = kwargs.get("pose_encoding_type", "absT_quaR_FoV")

        with torch.cuda.amp.autocast(enabled=False):
            if use_gt:
                gt_extrinsics = batch['extrinsics'].float()
                gt_intrinsics = batch['intrinsics'].float()
                gt_pose_encoding = extri_intri_to_pose_encoding(
                    gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
                )
                pred_pose_enc = gt_pose_encoding

            pred_extr, pred_intr = pose_encoding_to_extri_intri(
                pred_pose_enc.float(),
                image_size_hw=image_hw,
                pose_encoding_type=pose_encoding_type,
                build_intrinsics=True,
            )

        B, S = pred_pose_enc.shape[:2]
        if pred_joints_norm.dim() == 4:
            temporal_frames = pred_joints_norm.shape[1]
            views_per_frame = int(batch.get("views_per_frame", torch.tensor([S], device=pred_pose_enc.device))[0].item())
            if temporal_frames > 1 and S == temporal_frames * views_per_frame:
                points_world = pred_joints_norm[:, :, None, :, :].expand(
                    B, temporal_frames, views_per_frame, pred_joints_norm.shape[2], 3
                )
                points_world = points_world.reshape(B, S, pred_joints_norm.shape[2], 3)
            else:
                points_world = pred_joints_norm[:, :1, :, :].expand(B, S, pred_joints_norm.shape[2], 3)
        else:
            points_world = pred_joints_norm[:, None, :, :].expand(B, S, pred_joints_norm.shape[1], 3)
        # Depth clamp for projection: in the batch gauge the scene is ~O(1) (divided by
        # avg_scale), so a valid in-frame person has depth z~O(1). When mesh_translate puts
        # someone at z→0, u=fx*X/z explodes (smpl_joints2d_max_abs → 1e7) and floods the
        # shared aggregator. Clamping z to a sane positive min bounds the projection.
        joints2d_depth_min = float(kwargs.get("joints2d_depth_min", 1e-6))
        with torch.cuda.amp.autocast(enabled=False):
            pred_joints2d = _project_points_opencv(
                points_world.float(), pred_extr.float(), pred_intr.float(), eps=joints2d_depth_min
            )

        gt_joints2d = batch["smpl_joints2d"].to(device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        if gt_joints2d.dim() == 2:
            gt_joints2d = gt_joints2d.unsqueeze(0).unsqueeze(0)
        elif gt_joints2d.dim() == 3:
            gt_joints2d = gt_joints2d.unsqueeze(1)
        elif gt_joints2d.dim() != 4:
            raise ValueError(f"Unsupported smpl_joints2d shape: {tuple(gt_joints2d.shape)}")
        
        # logging.info(f"avg_scale: {batch['avg_scale'].detach().cpu().numpy()}")
        # logging.info(f"loss gt_extrinsics: {batch['extrinsics'].detach().cpu().numpy()}")
        # logging.info(f"loss gt_intrinsics: {batch['intrinsics'].detach().cpu().numpy()}")
        # logging.info(f"loss smpl_joints3d_world: {batch['smpl_joints3d_world'].detach().cpu().numpy()}")
        # logging.info(f"loss smpl_joints2d: {batch['smpl_joints2d']}")
        # logging.info(f"loss points_world: {points_world.detach().cpu().numpy()}")
        # logging.info(f"loss pred_joints2d: {pred_joints2d.detach().cpu().numpy()}")

        joints2d_conf = batch["smpl_joints2d_confidence"]
        joints2d_conf = joints2d_conf.to(device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        if joints2d_conf.dim() == 1:
            joints2d_conf = joints2d_conf.unsqueeze(0).unsqueeze(0)
        elif joints2d_conf.dim() == 2:
            joints2d_conf = joints2d_conf.unsqueeze(1)
        elif joints2d_conf.dim() != 3:
            raise ValueError(f"Unsupported smpl_joints2d_confidence shape: {tuple(joints2d_conf.shape)}")
        if has_smpl is not None:
            has_smpl_2d = has_smpl.to(device=joints2d_conf.device, dtype=joints2d_conf.dtype)
            if has_smpl_2d.dim() == 1:
                has_smpl_2d = has_smpl_2d[:, None, None]
            elif has_smpl_2d.dim() == 2:
                has_smpl_2d = has_smpl_2d[:, :, None]
            else:
                raise ValueError(f"Unsupported has_smpl shape for 2D joints: {tuple(has_smpl.shape)}")
            if has_smpl_2d.shape[1] != joints2d_conf.shape[1]:
                if has_smpl_2d.shape[1] == 1:
                    has_smpl_2d = has_smpl_2d.expand(-1, joints2d_conf.shape[1], -1)
                elif joints2d_conf.shape[1] == 1:
                    has_smpl_2d = has_smpl_2d[:, :1]
                else:
                    raise ValueError(
                        f"Cannot align has_smpl shape {tuple(has_smpl.shape)} to joints2d_conf shape {tuple(joints2d_conf.shape)}"
                    )
            joints2d_conf = joints2d_conf * has_smpl_2d
        # print(f"pred_joints2d.shape: {pred_joints2d.shape}")
        # print(f"joints2d_conf.shape: {joints2d_conf.shape}")
        # print(f"joints2d_conf: {joints2d_conf}")

        B, S, J_pred = pred_joints2d.shape[:3]
        if gt_joints2d.shape[1] != S:
            gt_joints2d = gt_joints2d[:, :1].expand(-1, S, -1, -1)
        if joints2d_conf.shape[1] != S:
            joints2d_conf = joints2d_conf[:, :1].expand(-1, S, -1)

        # Normalize both pred/gt 2D joints to [-1, 1] in image coordinates (TRAM-style).
        # x_norm = (x - W/2) / (W/2), y_norm = (y - H/2) / (H/2)
        H, W = image_hw
        half_w = torch.tensor(W / 2.0, device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        half_h = torch.tensor(H / 2.0, device=pred_joints2d.device, dtype=pred_joints2d.dtype)

        pred_joints2d_norm = pred_joints2d.clone()
        pred_joints2d_norm[..., 0] = (pred_joints2d_norm[..., 0] - half_w) / half_w.clamp(min=1.0)
        pred_joints2d_norm[..., 1] = (pred_joints2d_norm[..., 1] - half_h) / half_h.clamp(min=1.0)

        # 投影爆掉偵測(診斷用,no-grad):in-frame 應 ~[-1,1],飆高代表深度 z→0 投影爆掉
        smpl_joints2d_max_abs = pred_joints2d_norm.detach().abs().max()

        gt_joints2d_norm = gt_joints2d.clone()
        gt_joints2d_norm[..., 0] = (gt_joints2d_norm[..., 0] - half_w) / half_w.clamp(min=1.0)
        gt_joints2d_norm[..., 1] = (gt_joints2d_norm[..., 1] - half_h) / half_h.clamp(min=1.0)

        joints2d_loss_type = loss_type_joints2d or loss_type
        body_joint_count = min(24, J_pred, gt_joints2d.shape[2], joints2d_conf.shape[2])
        if weight_joints2d > 0.0 and body_joint_count > 0:
            pred_body_joints2d = pred_joints2d_norm[:, :, :body_joint_count]
            gt_body_joints2d = gt_joints2d_norm[:, :, :body_joint_count]
            body_joints2d_conf = joints2d_conf[:, :, :body_joint_count]
            if joints2d_loss_type == "l1":
                diff = (pred_body_joints2d - gt_body_joints2d).abs().sum(dim=-1)
            elif joints2d_loss_type == "l2":
                diff = ((pred_body_joints2d - gt_body_joints2d) ** 2).sum(dim=-1)
            else:
                raise ValueError(joints2d_loss_type)
            valid_f = (body_joints2d_conf > 0.5).to(dtype=diff.dtype)
            # Robust gate: drop joints whose normalized reprojection blew up (depth z→0 →
            # coords >> 1). In-frame is ~[-1,1]; anything beyond joints2d_max_norm is a bad
            # mesh_translate this step. Masking removes its huge gradient into the shared
            # backbone; the mesh_translate L1 loss still pulls these back toward valid depth.
            joints2d_max_norm = float(kwargs.get("joints2d_max_norm", 0.0))
            if joints2d_max_norm > 0.0:
                sane = (
                    pred_body_joints2d.detach().abs().amax(dim=-1) <= joints2d_max_norm
                ).to(dtype=diff.dtype)
                valid_f = valid_f * sane
            loss_joints2d = (diff * valid_f).sum()
            loss_joints2d = check_and_fix_inf_nan(loss_joints2d, "loss_joints2d")
            loss_joints2d = loss_joints2d / valid_f.sum().clamp(min=1.0)

    #* Vertices loss (SMPL space; mirrors TRAM behavior)
    if weight_vertices > 0.0:
        B, T = gt_pose.shape[:2]
        gt_pose_flat = gt_pose.reshape(B * T, -1)
        gt_beta_flat = gt_beta.reshape(B * T, -1)
        if smpl_decode_uses_mesh_translate:
            gt_trans_flat = torch.zeros(B * T, 3, device=gt_pose.device, dtype=gt_pose.dtype)
        else:
            gt_trans_flat = gt_trans.reshape(B * T, -1)

        genders = _resolve_batch_genders(batch, B)
        if len(genders) == B:
            genders = [g for g in genders for _ in range(T)]

        # Match the prediction decode path: keep SMPL LBS and gauge transforms in
        # fp32 so use_gt=True does not leave a small bf16/autocast vertex residual.
        with torch.cuda.amp.autocast(enabled=False):
            gt_joints_zero_flat, gt_vertices_smpl_flat = _decode_smpl_batch(
                pose_aa=gt_pose_flat.float(),
                betas=gt_beta_flat.float(),
                trans=gt_trans_flat.float(),
                genders=genders,
                use_mamma=use_mamma,
            )
        gt_vertices_smpl = gt_vertices_smpl_flat.reshape(B, T, gt_vertices_smpl_flat.shape[1], 3)
        if smpl_decode_uses_mesh_translate:
            gt_joints_zero = gt_joints_zero_flat.reshape(B, T, gt_joints_zero_flat.shape[1], 3)
            # mesh_rot mode: gt_pose[:3] is the cam0-frame mesh_rot (overwritten above),
            # so the decoded GT body is already in cam0 -> scale-only gauge.
            with torch.cuda.amp.autocast(enabled=False):
                avg_scale_f = batch["avg_scale"].float()
                if use_mesh_rot:
                    gt_vertices_smpl = scale_joints_to_batch_gauge(gt_vertices_smpl.float(), avg_scale_f)
                    gt_joints_zero = scale_joints_to_batch_gauge(gt_joints_zero.float(), avg_scale_f)
                else:
                    gt_vertices_smpl = normalize_joints_world_to_batch_gauge(
                        gt_vertices_smpl.float(),
                        batch["raw_extrinsics"].float(),
                        avg_scale_f,
                    )
                    gt_joints_zero = normalize_joints_world_to_batch_gauge(
                        gt_joints_zero.float(),
                        batch["raw_extrinsics"].float(),
                        avg_scale_f,
                    )
            gt_mesh_translate_aligned = gt_mesh_translate.to(device=gt_pose.device).float()
            gt_offset = gt_mesh_translate_aligned - gt_joints_zero[..., 0, :]
            gt_vertices_smpl = gt_vertices_smpl + gt_offset.unsqueeze(-2)

        # L1 in SMPL space
        vertex_diff = (pred_vertices_smpl - gt_vertices_smpl).abs().mean(dim=(-1, -2))
        if has_smpl is not None:
            vertex_mask = has_smpl.to(device=vertex_diff.device, dtype=vertex_diff.dtype)
            if vertex_mask.dim() == 1:
                vertex_mask = vertex_mask[:, None].expand_as(vertex_diff)
            loss_vertices = (vertex_diff * vertex_mask).sum() / vertex_mask.sum().clamp(min=1.0)
        else:
            loss_vertices = vertex_diff.mean()

    # ---- dense-landmark (direct 2D GNLL) + per-person mask losses ----
    # Both use the already-matched (Hungarian) + people-flattened predictions and
    # the shared has_smpl mask, so slots stay bound to the same identity.
    loss_landmark = (pred_pose * 0.0).mean()
    loss_landmark_l2 = (pred_pose * 0.0).mean().detach()
    weight_landmark = float(kwargs.get("weight_landmark", 0.0))
    if (
        weight_landmark > 0.0
        and predictions.get("smpl_landmarks2d") is not None
        and batch.get("smpl_landmarks2d") is not None
    ):
        lmk_dict = compute_landmark_loss(
            pred_xy=predictions["smpl_landmarks2d"],
            pred_logvar=predictions.get("smpl_landmarks_logvar"),
            gt_xy=batch["smpl_landmarks2d"],
            visibility=batch.get("smpl_landmarks2d_visibility"),
            has_smpl=has_smpl,
            loss_type=str(kwargs.get("landmark_loss_type", "gnll")),
        )
        loss_landmark = lmk_dict["loss_landmark"]
        loss_landmark_l2 = lmk_dict["loss_landmark_l2"]

    loss_mask = (pred_pose * 0.0).mean()
    mask_soft_iou = (pred_pose * 0.0).mean().detach()
    weight_mask = float(kwargs.get("weight_mask", 0.0))
    if (
        weight_mask > 0.0
        and predictions.get("person_mask_logits") is not None
        and batch.get("person_mask") is not None
    ):
        mask_dict = compute_mask_loss(
            pred_logits=predictions["person_mask_logits"],
            gt_mask=batch["person_mask"],
            has_smpl=has_smpl,
        )
        loss_mask = mask_dict["loss_mask"]
        mask_soft_iou = mask_dict["mask_soft_iou"]

    total = (
        loss_smpl_losses +
        weight_trans * loss_trans +
        weight_mesh_translate * loss_mesh_translate +
        weight_presence * loss_presence +
        weight_joints2d * loss_joints2d +
        weight_joints3d * loss_joints3d +
        weight_vertices * loss_vertices +
        weight_landmark * loss_landmark +
        weight_mask * loss_mask
    )

    return {
        "loss_smpl": total,
        "loss_smpl_losses": loss_smpl_losses,
        "loss_smpl_trans": loss_trans,
        "loss_mesh_translate": loss_mesh_translate,
        "loss_smpl_presence": loss_presence,
        "loss_smpl_joints2d": loss_joints2d,
        "loss_smpl_joints3d": loss_joints3d,
        "loss_smpl_vertices": loss_vertices,
        "loss_landmark": loss_landmark,
        "loss_landmark_l2": loss_landmark_l2,
        "loss_mask": loss_mask,
        "mask_soft_iou": mask_soft_iou,
        "smpl_presence_positive_prob_mean": smpl_presence_positive_prob_mean,
        "smpl_presence_empty_prob_mean": smpl_presence_empty_prob_mean,
        "smpl_joints2d_max_abs": smpl_joints2d_max_abs,
    }


def compute_smpl_3d_joint_loss(pred_joints_world, gt_joints_norm, batch, loss_type="l1",
                              pelvis_center=True, pelvis_ids=(1,2), normalize_cam=True):
    """
    pred_joints_world: predictions["smpl_joints3d_world"]  (raw world/unity)
    gt_joints_norm:    batch["smpl_joints3d_world"]       (你 _process_batch 後已 normalize)
    batch 需要提供 raw_extrinsics, avg_scale
    """
    raw_extr = batch["raw_extrinsics"]
    avg_scale = batch["avg_scale"]

    if normalize_cam:
        pred_norm = normalize_joints_world_to_batch_gauge(pred_joints_world, raw_extr, avg_scale)
    else:
        pred_norm = pred_joints_world

    if pred_norm.dim() == 4 and gt_joints_norm.dim() == 4:
        if pred_norm.shape[1] != gt_joints_norm.shape[1]:
            if gt_joints_norm.shape[1] % pred_norm.shape[1] == 0:
                views_per_frame = gt_joints_norm.shape[1] // pred_norm.shape[1]
                gt_joints_norm = gt_joints_norm[:, ::views_per_frame]
            elif pred_norm.shape[1] % gt_joints_norm.shape[1] == 0:
                pred_norm = pred_norm[:, :: pred_norm.shape[1] // gt_joints_norm.shape[1]]
            T = min(pred_norm.shape[1], gt_joints_norm.shape[1])
            pred_norm = pred_norm[:, :T]
            gt_joints_norm = gt_joints_norm[:, :T]
        B, T, J_pred, _ = pred_norm.shape
        _, _, J_gt, _ = gt_joints_norm.shape
        J = min(J_pred, J_gt)
        pred_norm = pred_norm[:, :, :J, :].reshape(B * T, J, 3)
        gt_joints_norm = gt_joints_norm[:, :, :J, :].reshape(B * T, J, 3)
    else:
        if pred_norm.dim() == 4:
            pred_norm = pred_norm[:, 0]
        if gt_joints_norm.dim() == 4:
            gt_joints_norm = gt_joints_norm[:, 0]

        if pred_norm.dim() != 3 or gt_joints_norm.dim() != 3:
            raise ValueError(
                f"Expected pred/gt joints to be 3D after alignment, got {tuple(pred_norm.shape)} and {tuple(gt_joints_norm.shape)}"
            )

    # If joint counts differ, fall back to matching the common prefix.
    if pred_norm.shape[1] != gt_joints_norm.shape[1]:
        J = min(pred_norm.shape[1], gt_joints_norm.shape[1])
        pred_norm = pred_norm[:, :J, :]
        gt_joints_norm = gt_joints_norm[:, :J, :]

    if pelvis_center:
        pred_norm, gt_joints_norm = subtract_pelvis(pred_norm, gt_joints_norm, pelvis_ids=pelvis_ids)

    if loss_type == "l1":
        loss_tensor = (pred_norm - gt_joints_norm).abs()
    elif loss_type == "l2":
        loss_tensor = (pred_norm - gt_joints_norm) ** 2
    else:
        raise ValueError(loss_type)

    has_smpl = batch.get("has_smpl", None)
    if has_smpl is not None:
        sample_loss = loss_tensor.mean(dim=(-1, -2))
        mask = has_smpl.to(device=sample_loss.device, dtype=sample_loss.dtype).reshape(-1)
        if sample_loss.numel() != mask.numel():
            if sample_loss.numel() % mask.numel() == 0:
                mask = mask.repeat_interleave(sample_loss.numel() // mask.numel())
            else:
                mask = mask[: sample_loss.numel()]
        return (sample_loss.reshape(-1) * mask).sum() / mask.sum().clamp(min=1.0)

    return loss_tensor.mean()

def normalize_joints_world_to_batch_gauge(joints_world: torch.Tensor, raw_extrinsics: torch.Tensor, avg_scale: torch.Tensor) -> torch.Tensor:
    """
    Apply the SAME gauge transform as normalize_camera_extrinsics_*:
      1) move world into cam0 coordinates: X' = X @ R0^T + t0
      2) divide by avg_scale
    """
    R0 = raw_extrinsics[:, 0, :3, :3]   # (B,3,3)
    t0 = raw_extrinsics[:, 0, :3, 3]    # (B,3)

    if joints_world.dim() == 2:  # (B,3)
        joints0 = torch.bmm(joints_world.unsqueeze(1), R0.transpose(-1, -2)).squeeze(1) + t0
    elif joints_world.dim() == 3:  # (B,J,3)
        # Use bmm to avoid broadcasting introducing an extra singleton dimension.
        joints0 = torch.bmm(joints_world, R0.transpose(-1, -2)) + t0.unsqueeze(1)
    elif joints_world.dim() == 4:  # (B,S,J,3)
        # Broadcast R0 across S (views/time) without adding extra dims.
        joints0 = joints_world @ R0.transpose(-1, -2).unsqueeze(1) + t0.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported joints_world shape: {joints_world.shape}")

    # scale normalize
    view = (-1,) + (1,) * (joints0.dim() - 1)   # (B,1,1,1), (B,1,1), or (B,1)
    joints_norm = joints0 / avg_scale.view(*view).clamp(min=1e-6)
    return joints_norm

def subtract_pelvis(pred_j, gt_j, pelvis_ids=(1,2)):
    if pred_j.dim() == 3:  # (B,J,3)
        pred_pelvis = 0.5 * (pred_j[:, pelvis_ids[0], :] + pred_j[:, pelvis_ids[1], :])
        gt_pelvis   = 0.5 * (gt_j[:, pelvis_ids[0], :] + gt_j[:, pelvis_ids[1], :])
        pred_j = pred_j - pred_pelvis[:, None, :]
        gt_j   = gt_j   - gt_pelvis[:, None, :]
    else:  # (B,S,J,3)
        pred_pelvis = 0.5 * (pred_j[:, :, pelvis_ids[0], :] + pred_j[:, :, pelvis_ids[1], :])
        gt_pelvis   = 0.5 * (gt_j[:, :, pelvis_ids[0], :] + gt_j[:, :, pelvis_ids[1], :])
        pred_j = pred_j - pred_pelvis[:, :, None, :]
        gt_j   = gt_j   - gt_pelvis[:, :, None, :]
    return pred_j, gt_j

def _project_points_opencv(
    points_world: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project 3D world points to pixel coords using OpenCV convention.

    points_world: (B, S, J, 3)
    extrinsics:   (B, S, 3, 4) world->cam
    intrinsics:   (B, S, 3, 3)
    returns:      (B, S, J, 2)
    """
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]

    Xc = torch.einsum("bsij,bsnj->bsni", R, points_world) + t.unsqueeze(-2)
    X = Xc[..., 0]
    Y = Xc[..., 1]
    Z = Xc[..., 2]
    # print(f"X: {X}, Y: {Y}, Z: {Z}")

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    # print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    Z_safe = Z.clamp(min=eps)
    u = fx.unsqueeze(-1) * (X / Z_safe) + cx.unsqueeze(-1)
    v = fy.unsqueeze(-1) * (Y / Z_safe) + cy.unsqueeze(-1)
    # print(f"u: {u}, v: {v}")
    # print(f"extrinsics: {extrinsics.detach().cpu().numpy()}, intrinsics: {intrinsics.detach().cpu().numpy()}")
    # print(f"points_world: {points_world.detach().cpu().numpy()}")

    uv = torch.stack([u, v], dim=-1)
    return uv


def compute_landmark_loss(
    pred_xy: torch.Tensor,       # (B*P, S, L, 2) normalised [-1,1]
    pred_logvar: torch.Tensor,   # (B*P, S, L)
    gt_xy: torch.Tensor,         # (B*P, S, L, 2)
    visibility: torch.Tensor | None,  # (B*P, S, L) in {0,1}
    has_smpl: torch.Tensor | None,    # (B*P,)
    loss_type: str = "gnll",
) -> dict:
    """Dense 512-landmark loss (direct 2D).

    Gaussian NLL with a predicted per-landmark log-variance (mamma-style), so
    hard / occluded points down-weight themselves.  Only visible landmarks of
    matched (has_smpl) people contribute.
    """
    pred_xy = pred_xy.float()
    gt_xy = gt_xy.to(device=pred_xy.device, dtype=pred_xy.dtype)
    pred_logvar = pred_logvar.float()

    # per-landmark squared error summed over (x, y).
    sq = ((pred_xy - gt_xy) ** 2).sum(dim=-1)                     # (B*P, S, L)

    if loss_type == "gnll":
        # 0.5 * ( exp(-logvar) * ||d||^2 + logvar ), clamp logvar for stability.
        logvar = pred_logvar.clamp(-6.0, 6.0)
        per_lmk = 0.5 * (torch.exp(-logvar) * sq + logvar)
    elif loss_type == "l2":
        per_lmk = sq
    else:
        raise ValueError(f"Unknown landmark loss_type: {loss_type}")

    weight = torch.ones_like(per_lmk)
    if visibility is not None:
        weight = weight * visibility.to(device=weight.device, dtype=weight.dtype)
    if has_smpl is not None:
        w_person = has_smpl.to(device=weight.device, dtype=weight.dtype).reshape(-1, 1, 1)
        weight = weight * w_person

    denom = weight.sum().clamp(min=1.0)
    loss_landmark = (per_lmk * weight).sum() / denom
    # raw pixel-ish L2 (unweighted by sigma) for logging.
    loss_landmark_l2 = (sq.sqrt() * weight).sum() / denom

    return {
        "loss_landmark": loss_landmark,
        "loss_landmark_l2": loss_landmark_l2.detach(),
    }
