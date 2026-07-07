"""SMPL / SMPL-X body model, decode, axis-angle & gauge utils (split out of loss_smpl.py)."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import torch
import torch.nn.functional as F
import logging

from dataclasses import dataclass
from pathlib import Path
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from training.train_utils.general import check_and_fix_inf_nan
import numpy as np
from scipy.optimize import linear_sum_assignment

# Compatibility for chumpy (SMPL/SMPL-X .pkl loading) on Python >= 3.11, where
# inspect.getargspec was removed. Loading the SMPL-X .pkl in the loss/training
# path (not just the debug harness) needs this shim.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

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


def set_smplx_model_root(root) -> None:
    """Override the base directory for the per-gender SMPL-X ``model.pkl`` files.

    Expects ``<root>/{female,male,neutral}/model.pkl`` (same layout as the bundled
    ``smplx_models/``). Config-driven via ``loss.smplx_model_dir``. Clears the model
    cache so a later decode reloads from the new location. ``None`` is a no-op (keeps
    the default paths).
    """
    if root is None:
        return
    root = Path(root).expanduser()
    for gender_key in ("female", "male", "neutral"):
        _SMPLX_MODEL_PATHS[gender_key] = str(root / gender_key / "model.pkl")
    _SMPLX_MODEL_CACHE.clear()


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


