import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Sampler

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
EVAL_RESULTS_ROOT = SCRIPT_ROOT / "eval_results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from training.loss import (  # noqa: E402
    _decode_smpl_batch,
    _project_points_opencv,
    compute_gt_mesh_translate,
    extri_intri_to_pose_encoding,
    normalize_joints_world_to_batch_gauge,
    pose_encoding_to_extri_intri,
)
from training.train_utils.general import copy_data_to_device  # noqa: E402
from training.train_utils.normalization import (  # noqa: E402
    normalize_camera_extrinsics_points_and_3djoints_batch,
)


DEFAULT_CONFIG_NAME = "0519_multi_100k"
DEFAULT_CHECKPOINT = (
    "/mnt/train-data-4-hdd/clchen/vggt/training/logs/0519_multi_100k/"
    "ckpts/checkpoint_step_40000.pt"
)
DEFAULT_DATASET_ROOT = "/mnt/train-data-3-hdd/clchen/SMPL_multi_dataset/0522_cloth3d_100k_test/test"


class EvalNoPadDynamicDistributedSampler(Sampler):
    """Distributed eval sampler that never pads or repeats samples."""

    def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        for idx in indices[self.rank :: self.num_replicas]:
            yield (idx, self.image_num, self.aspect_ratio)

    def __len__(self):
        dataset_len = len(self.dataset)
        if self.rank >= dataset_len:
            return 0
        return (dataset_len - 1 - self.rank) // self.num_replicas + 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_parameters(self, aspect_ratio, image_num):
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num


def _init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        use_nccl = args.device == "cuda" and torch.cuda.is_available()
        dist.init_process_group(backend="nccl" if use_nccl else "gloo")

    return distributed, rank, world_size, local_rank


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def _reduce_metric_totals(totals, distributed):
    values = torch.stack([value.to(dtype=torch.float64) for value in totals])
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values.cpu().tolist()


def _gather_batch_result_lines(batch_result_lines, distributed):
    if not distributed:
        return batch_result_lines
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, batch_result_lines)
    return [line for rank_lines in gathered for line in rank_lines]


def _install_eval_no_pad_sampler(val_dataset, rank: int, world_size: int):
    sampler = EvalNoPadDynamicDistributedSampler(
        val_dataset.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(getattr(val_dataset, "shuffle", False)),
        seed=int(getattr(val_dataset, "seed", 0)),
    )
    val_dataset.sampler = sampler
    val_dataset.batch_sampler.sampler = sampler
    return sampler


def _normalize_batch(batch, normalize_cam):
    if normalize_cam:
        (
            normalized_extrinsics,
            normalized_cam_points,
            normalized_world_points,
            normalized_joints3d_world,
            normalized_depths,
            avg_scale,
        ) = normalize_camera_extrinsics_points_and_3djoints_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch.get("cam_points"),
            world_points=batch.get("world_points"),
            joints3d_world=batch.get("smpl_joints3d_world"),
            depths=batch.get("depths"),
            scale_by_extrinsics=True,
            point_masks=batch.get("point_masks"),
        )

        batch["avg_scale"] = avg_scale
        batch["raw_extrinsics"] = batch["extrinsics"].clone()
        batch["extrinsics"] = normalized_extrinsics
        if normalized_cam_points is not None:
            batch["cam_points"] = normalized_cam_points
        if normalized_world_points is not None:
            batch["world_points"] = normalized_world_points
        if normalized_joints3d_world is not None:
            batch["smpl_joints3d_world"] = normalized_joints3d_world
        if normalized_depths is not None:
            batch["depths"] = normalized_depths
    else:
        B = batch["extrinsics"].shape[0]
        device = batch["extrinsics"].device
        batch["avg_scale"] = torch.ones(B, device=device)
        batch["raw_extrinsics"] = batch["extrinsics"].clone()
    return batch


def _load_checkpoint(model, ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {len(unexpected)}")


def _metadata_inputs(batch):
    inputs = {}
    for key in ("views_per_frame", "temporal_num_frames"):
        if key in batch:
            inputs[key] = batch[key]
    return inputs


def _match_joint_dims(pred, gt):
    if pred.dim() == 4 and gt.dim() == 3:
        gt = gt[:, None, :, :]
    if pred.dim() == 3 and gt.dim() == 4:
        pred = pred[:, None, :, :]
    if pred.shape[-2] != gt.shape[-2]:
        j = min(pred.shape[-2], gt.shape[-2])
        pred = pred[..., :j, :]
        gt = gt[..., :j, :]
    return pred, gt


def subtract_root_joint(pred_j, gt_j, root_id=0):
    if pred_j.dim() == 3:
        pred_root = pred_j[:, root_id, :]
        gt_root = gt_j[:, root_id, :]
        pred_j = pred_j - pred_root[:, None, :]
        gt_j = gt_j - gt_root[:, None, :]
    elif pred_j.dim() == 4:
        pred_root = pred_j[:, :, root_id, :]
        gt_root = gt_j[:, :, root_id, :]
        pred_j = pred_j - pred_root[:, :, None, :]
        gt_j = gt_j - gt_root[:, :, None, :]
    else:
        raise ValueError(f"Unsupported joint tensor dim for root alignment: {pred_j.dim()}")
    return pred_j, gt_j


def _sum_count_per_sample(err, valid=None):
    batch_size = err.shape[0]
    err_flat = err.reshape(batch_size, -1)
    if valid is None:
        count = torch.full(
            (batch_size,),
            err_flat.shape[1],
            device=err.device,
            dtype=torch.long,
        )
        return err_flat.sum(dim=1), count

    valid = valid.reshape(batch_size, -1)
    valid_float = valid.to(dtype=err.dtype)
    return (err_flat * valid_float).sum(dim=1), valid.sum(dim=1).to(dtype=torch.long)


def _metric_from_per_sample(sum_tensor, count_tensor, sample_idx, scale=1.0):
    count = int(count_tensor[sample_idx].detach().cpu().item())
    if count == 0:
        return float("nan"), 0
    value = float(sum_tensor[sample_idx].detach().cpu().item()) / count * scale
    return value, count


def _add_metric_to_samples(sample_sum, sample_count, sample_indices, metric_sum, metric_count):
    sample_sum.index_add_(0, sample_indices, metric_sum.to(dtype=sample_sum.dtype))
    sample_count.index_add_(0, sample_indices, metric_count.to(dtype=sample_count.dtype))


def _list_from_batch_value(value, batch_size):
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value] * batch_size
    if len(values) != batch_size:
        values = (values + [None] * batch_size)[:batch_size]
    return values


def _ids_to_str(batch, sample_idx):
    if "ids" not in batch:
        return "NA"
    try:
        ids_i = batch["ids"][sample_idx]
        if isinstance(ids_i, torch.Tensor):
            ids_i = ids_i.detach().cpu().tolist()
        if isinstance(ids_i, (list, tuple)):
            return ",".join([str(int(x)) for x in ids_i])
        return str(ids_i)
    except Exception:
        return "NA"


def _empty_sample_metrics(batch_size, device):
    return {
        "gt_people": torch.zeros(batch_size, device=device, dtype=torch.long),
        "selected_people": torch.zeros(batch_size, device=device, dtype=torch.long),
        "matched_people": torch.zeros(batch_size, device=device, dtype=torch.long),
        "miss": torch.zeros(batch_size, device=device, dtype=torch.long),
        "fp": torch.zeros(batch_size, device=device, dtype=torch.long),
        "mpjpe_sum": torch.zeros(batch_size, device=device, dtype=torch.float64),
        "mpjpe_count": torch.zeros(batch_size, device=device, dtype=torch.long),
        "pa_sum": torch.zeros(batch_size, device=device, dtype=torch.float64),
        "pa_count": torch.zeros(batch_size, device=device, dtype=torch.long),
        "reproj_sum": torch.zeros(batch_size, device=device, dtype=torch.float64),
        "reproj_count": torch.zeros(batch_size, device=device, dtype=torch.long),
        "body_pve_sum": torch.zeros(batch_size, device=device, dtype=torch.float64),
        "body_pve_count": torch.zeros(batch_size, device=device, dtype=torch.long),
    }


def _mpjpe_per_sample(pred, gt, root_align=True, root_id=0):
    pred, gt = _match_joint_dims(pred, gt)
    if root_align:
        pred, gt = subtract_root_joint(pred, gt, root_id=root_id)
    err = torch.linalg.norm(pred - gt, dim=-1)
    return _sum_count_per_sample(err)


def _procrustes_align(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    assert pred.shape == gt.shape and pred.dim() == 3

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)
    y0 = pred - mu_pred
    x0 = gt - mu_gt

    h = y0.transpose(1, 2) @ x0
    u, s, vh = torch.linalg.svd(h)
    r = u @ vh

    mask = torch.linalg.det(r) < 0
    if mask.any():
        u_fix = u.clone()
        u_fix[mask, :, -1] *= -1
        r = u_fix @ vh

    var_y = (y0**2).sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    scale = s.sum(dim=1, keepdim=True).view(-1, 1, 1) / var_y
    trans = mu_gt - scale * (mu_pred @ r)
    return scale * (pred @ r) + trans


def _pa_mpjpe_per_sample(pred, gt, root_align=True, root_id=0):
    pred, gt = _match_joint_dims(pred, gt)
    if root_align:
        pred, gt = subtract_root_joint(pred, gt, root_id=root_id)
    pred_flat = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    gt_flat = gt.reshape(-1, gt.shape[-2], gt.shape[-1])
    pred_aligned = _procrustes_align(pred_flat, gt_flat)
    err = torch.linalg.norm(pred_aligned - gt_flat, dim=-1)
    err = err.reshape(*pred.shape[:-1])
    return _sum_count_per_sample(err)


def _reprojection_error_2d_per_sample(pred, gt, conf=None):
    if pred.shape[-2] != gt.shape[-2]:
        j = min(pred.shape[-2], gt.shape[-2])
        pred = pred[..., :j, :]
        gt = gt[..., :j, :]
        if conf is not None:
            conf = conf[..., :j]
    err = torch.linalg.norm(pred - gt, dim=-1)
    valid = (conf > 0.5) if conf is not None else None
    return _sum_count_per_sample(err, valid)


def _pve_per_sample(pred_vertices, gt_vertices):
    if pred_vertices.shape[-2] != gt_vertices.shape[-2]:
        v = min(pred_vertices.shape[-2], gt_vertices.shape[-2])
        pred_vertices = pred_vertices[..., :v, :]
        gt_vertices = gt_vertices[..., :v, :]
    err = torch.linalg.norm((pred_vertices - gt_vertices).float(), dim=-1)
    return _sum_count_per_sample(err)


def _first_n_joints(joints, n=24):
    return joints[..., : min(n, joints.shape[-2]), :]


def _gender_to_name(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return "neutral"
        value = value.reshape(-1)[0].detach().cpu().item()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        token = value.strip().lower()
        if token.startswith("m"):
            return "male"
        if token.startswith("f"):
            return "female"
        return "neutral"
    try:
        value = int(value)
    except Exception:
        return "neutral"
    if value == 0:
        return "male"
    if value == 1:
        return "female"
    return "neutral"


def _resolve_valid_genders(batch, valid_mask):
    gender = batch.get("smpl_gender", None)
    if gender is None:
        gender = batch.get("smpl_genders", None)
    if gender is None:
        return ["neutral"] * int(valid_mask.sum().item())
    if torch.is_tensor(gender):
        if gender.dim() >= 2:
            gender = gender[:, : valid_mask.shape[1]]
            return [_gender_to_name(v) for v in gender[valid_mask]]
        if gender.dim() == 1 and valid_mask.shape[1] == 1:
            return [_gender_to_name(v) for v in gender]
    return ["neutral"] * int(valid_mask.sum().item())


def _resolve_indexed_genders(batch, batch_indices, person_indices):
    gender = batch.get("smpl_gender", None)
    if gender is None:
        gender = batch.get("smpl_genders", None)
    if gender is None:
        return ["neutral"] * int(batch_indices.numel())
    if torch.is_tensor(gender):
        if gender.dim() >= 2:
            values = gender[batch_indices, person_indices]
            return [_gender_to_name(v) for v in values]
        if gender.dim() == 1:
            values = gender[batch_indices]
            return [_gender_to_name(v) for v in values]
    return ["neutral"] * int(batch_indices.numel())


def _get_gt_people_joints(batch, p_eval):
    gt_world = batch["smpl_joints3d_world"]
    if gt_world.dim() == 5:
        return gt_world[:, 0, :p_eval]
    if gt_world.dim() == 4:
        if gt_world.shape[1] >= p_eval:
            return gt_world[:, :p_eval]
        return gt_world[:, 0, None].expand(-1, p_eval, -1, -1)
    if gt_world.dim() == 3:
        return gt_world[:, None].expand(-1, p_eval, -1, -1)
    raise ValueError(f"Unsupported smpl_joints3d_world shape: {tuple(gt_world.shape)}")


def _get_gt_people_joints2d(batch, p_eval):
    gt_joints2d = batch.get("smpl_joints2d")
    if gt_joints2d is None:
        return None
    if gt_joints2d.dim() == 5:
        return gt_joints2d[:, :, :p_eval].permute(0, 2, 1, 3, 4).contiguous()
    if gt_joints2d.dim() == 4:
        return gt_joints2d[:, None].expand(-1, p_eval, -1, -1, -1)
    raise ValueError(f"Unsupported smpl_joints2d shape: {tuple(gt_joints2d.shape)}")


def _get_gt_people_joints2d_conf(batch, p_eval):
    conf = batch.get("smpl_joints2d_confidence")
    if conf is None:
        return None
    if conf.dim() == 4:
        return conf[:, :, :p_eval].permute(0, 2, 1, 3).contiguous()
    if conf.dim() == 3:
        return conf[:, None].expand(-1, p_eval, -1, -1)
    raise ValueError(f"Unsupported smpl_joints2d_confidence shape: {tuple(conf.shape)}")


def _build_eval_mask(batch, p_eval, pred_pose):
    has_smpl = batch.get("has_smpl", None)
    if has_smpl is None:
        return torch.ones(pred_pose.shape[0], p_eval, device=pred_pose.device, dtype=torch.bool)
    has_smpl = has_smpl.to(device=pred_pose.device)
    if has_smpl.dim() == 1:
        has_smpl = has_smpl[:, None]
    return has_smpl[:, :p_eval] > 0.5


def _presence_probs_from_preds(preds, batch, p_pred, device):
    presence_logits = preds.get("smpl_presence_logits", None)
    if presence_logits is not None:
        if presence_logits.dim() == 1:
            presence_logits = presence_logits[:, None]
        return torch.sigmoid(presence_logits[:, :p_pred].to(device=device))

    # GT mode has no predicted logits; keep it usable as a self-check.
    has_smpl = batch.get("has_smpl", None)
    if has_smpl is None:
        raise KeyError("smpl_presence_logits is required for detection evaluation")
    if has_smpl.dim() == 1:
        has_smpl = has_smpl[:, None]
    probs = torch.zeros(has_smpl.shape[0], p_pred, device=device, dtype=torch.float32)
    copy_count = min(p_pred, has_smpl.shape[1])
    probs[:, :copy_count] = has_smpl[:, :copy_count].to(device=device, dtype=torch.float32)
    return probs


def _pairwise_mean_abs(pred, gt):
    pred = pred.float()
    gt = gt.float()
    diff = (pred[:, None] - gt[None]).abs()
    reduce_dims = tuple(range(2, diff.dim()))
    return diff.mean(dim=reduce_dims)


def _hungarian_match_smpl_cost(
    pred_pose,
    pred_beta,
    pred_trans,
    pred_presence_logits,
    gt_pose,
    gt_beta,
    gt_trans,
    cost_pose_weight,
    cost_beta_weight,
    cost_trans_weight,
    cost_presence_weight,
    pred_mesh_translate=None,
    gt_mesh_translate=None,
    cost_mesh_trans_weight=0.0,
):
    if pred_pose.shape[0] == 0 or gt_pose.shape[0] == 0:
        return []

    cost = (
        cost_pose_weight * _pairwise_mean_abs(pred_pose[..., :72], gt_pose[..., :72])
        + cost_beta_weight * _pairwise_mean_abs(pred_beta, gt_beta)
    )
    if cost_trans_weight > 0.0 and pred_trans is not None and gt_trans is not None:
        cost = cost + cost_trans_weight * _pairwise_mean_abs(pred_trans, gt_trans)
    if (
        cost_mesh_trans_weight > 0.0
        and pred_mesh_translate is not None
        and gt_mesh_translate is not None
    ):
        cost = cost + cost_mesh_trans_weight * _pairwise_mean_abs(
            pred_mesh_translate, gt_mesh_translate
        )
    if (
        cost_presence_weight > 0.0
        and pred_presence_logits is not None
        and pred_presence_logits.numel() == pred_pose.shape[0]
    ):
        pred_presence_logits = pred_presence_logits.float()
        presence_cost = F.binary_cross_entropy_with_logits(
            pred_presence_logits,
            torch.ones_like(pred_presence_logits),
            reduction="none",
        )
        cost = cost + cost_presence_weight * presence_cost[:, None]

    cost = cost.float().detach().cpu()
    finite_mask = torch.isfinite(cost)
    if not bool(finite_mask.any()):
        return []

    finite_costs = cost[finite_mask]
    invalid_cost = finite_costs.max() + 1.0
    row_ind, col_ind = linear_sum_assignment(
        torch.where(finite_mask, cost, invalid_cost).numpy()
    )

    matches = []
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        if bool(finite_mask[pred_idx, gt_idx]):
            matches.append((int(pred_idx), int(gt_idx)))
    return matches


def _resolve_dataset_subdir(dataset_root: Path, sub: str, category: str) -> str:
    # Datasets may either nest each split under a category folder
    # (``out_image/CLOTH3D/...``) or place the ``runs_*`` entries directly under
    # the split folder (``out_image/...``). Prefer the category folder when it
    # exists, otherwise fall back to the bare split folder.
    if category:
        category_dir = dataset_root / sub / category
        if category_dir.exists():
            return str(category_dir)
    return str(dataset_root / sub)


def _apply_dataset_root_override(cfg, dataset_root, category="CLOTH3D"):
    dataset_root = Path(dataset_root)
    # SysSMPLMultiDataset reads images from out_image/<category> and the merged
    # per-frame annotations from out_data/<category>. When <category> is absent,
    # fall back to out_image/out_data directly.
    sys_smpl_dir = _resolve_dataset_subdir(dataset_root, "out_image", category)
    sys_smpl_annotation_dir = _resolve_dataset_subdir(dataset_root, "out_data", category)
    sys_smpl_param_dir = _resolve_dataset_subdir(dataset_root, "out_param", category)

    dataset_cfgs = cfg.data["val"].dataset.get("dataset_configs")
    if not dataset_cfgs:
        raise ValueError("--dataset-root was provided, but no dataset_configs were found in cfg.data.val.dataset")

    updated = 0
    for ds_cfg in dataset_cfgs:
        target = str(ds_cfg.get("_target_", ""))
        if target.endswith("SysSMPLMultiDataset"):
            with open_dict(ds_cfg):
                ds_cfg["SysSMPL_DIR"] = sys_smpl_dir
                ds_cfg["SysSMPL_ANNOTATION_DIR"] = sys_smpl_annotation_dir
                # Only override SysSMPL_PARAM_DIR when the dataset config already
                # uses it; SysSMPLMultiDataset does not accept this argument.
                if "SysSMPL_PARAM_DIR" in ds_cfg:
                    ds_cfg["SysSMPL_PARAM_DIR"] = sys_smpl_param_dir
            updated += 1

    if updated == 0:
        raise ValueError("--dataset-root was provided, but no SysSMPLMultiDataset config was found")
    return sys_smpl_dir, sys_smpl_annotation_dir, sys_smpl_param_dir


def _dataset_info(cfg):
    lines = []
    dataset_cfgs = cfg.data["val"].dataset.get("dataset_configs")
    if not dataset_cfgs:
        return lines
    for idx, ds_cfg in enumerate(dataset_cfgs):
        lines.append(f"[INFO] Dataset[{idx}] target: {ds_cfg.get('_target_', 'NA')}")
        for key in ("SysSMPL_DIR", "SysSMPL_ANNOTATION_DIR", "SysSMPL_PARAM_DIR"):
            if key in ds_cfg:
                lines.append(f"[INFO] Dataset[{idx}] {key}: {ds_cfg.get(key)}")
    return lines


def _append_sample_result_lines(
    batch_result_lines,
    batch,
    rank,
    batch_idx,
    batch_size,
    gt_people_count,
    selected_people_count,
    matched_people_count,
    miss_count,
    fp_count,
    mpjpe_sum,
    mpjpe_count,
    pa_sum,
    pa_count,
    reproj_sum,
    reproj_count,
    body_pve_sum,
    body_pve_count,
):
    seq_names = _list_from_batch_value(batch.get("seq_name"), batch_size)
    for i in range(batch_size):
        seq_mpjpe, seq_mpjpe_count = _metric_from_per_sample(mpjpe_sum, mpjpe_count, i, scale=1000.0)
        seq_pa, seq_pa_count = _metric_from_per_sample(pa_sum, pa_count, i, scale=1000.0)
        seq_reproj, seq_reproj_count = _metric_from_per_sample(reproj_sum, reproj_count, i)
        seq_body_pve, seq_body_pve_count = _metric_from_per_sample(
            body_pve_sum,
            body_pve_count,
            i,
            scale=1000.0,
        )
        sample_gt = int(gt_people_count[i].detach().cpu().item())
        sample_selected = int(selected_people_count[i].detach().cpu().item())
        sample_matched = int(matched_people_count[i].detach().cpu().item())
        sample_miss = int(miss_count[i].detach().cpu().item())
        sample_fp = int(fp_count[i].detach().cpu().item())
        batch_result_lines.append(
            f"rank={rank} batch_idx={batch_idx} sample_idx={i} seq_name={seq_names[i]} "
            f"mpjpe={seq_mpjpe:.6f} pa_mpjpe={seq_pa:.6f} reproj_err={seq_reproj:.6f} "
            f"body_pve={seq_body_pve:.6f} "
            f"mpjpe_count={int(seq_mpjpe_count)} pa_count={int(seq_pa_count)} "
            f"reproj_count={int(seq_reproj_count)} body_pve_count={int(seq_body_pve_count)} "
            f"gt_people={sample_gt} selected_people={sample_selected} "
            f"matched_people={sample_matched} miss_count={sample_miss} fp_count={sample_fp} "
            f"ids={_ids_to_str(batch, i)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute MPJPE, PA-MPJPE, 2D reprojection error, and Body PVE."
    )
    parser.add_argument("--folder-name", default="0519_multi_100k")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name under eval/config/.",
    )
    parser.add_argument("--output-name", default="multi_eval_results.txt")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=(
            "SysSMPL multi dataset split root. The script maps it to "
            "<root>/out_image/<category>, <root>/out_data/<category>, and <root>/out_param/<category>. "
            "When <category> is missing, it falls back to <root>/out_image, etc."
        ),
    )
    parser.add_argument(
        "--dataset-category",
        default="CLOTH3D",
        help=(
            "Category subfolder under out_image/out_data/out_param (e.g. CLOTH3D). "
            "Pass an empty string, or point at a dataset without this level, to use "
            "out_image/out_data directly."
        ),
    )
    parser.add_argument(
        "--output-batch-name",
        default="multi_eval_batch_results.txt",
        help="Output file for per-sample results.",
    )
    parser.add_argument(
        "--skip-batch-results-file",
        action="store_true",
        help="Skip collecting and writing multi_eval_batch_results.txt.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
        help="Threshold on sigmoid(smpl_presence_logits) for selecting predicted people.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit; default runs the full validation set.",
    )
    parser.add_argument(
        "--max-img-per-gpu",
        type=int,
        default=None,
        help="Override cfg.max_img_per_gpu (images per batch per GPU). Default uses the config value.",
    )
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = _init_distributed(args)
    if args.device == "cuda" and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "eval" / "config"),
    ):
        cfg = compose(config_name=args.config_name)
    if args.max_img_per_gpu is not None:
        # Top-level value; data.{train,val}.max_img_per_gpu reference it via
        # ${max_img_per_gpu} interpolation, so this propagates at instantiate time.
        with open_dict(cfg):
            cfg.max_img_per_gpu = int(args.max_img_per_gpu)
    _apply_dataset_root_override(cfg, args.dataset_root, args.dataset_category)

    smpl_cfg = cfg.loss.smpl
    normalize_cam = bool(smpl_cfg.get("normalize_cam", False))
    use_gt_smpl = bool(smpl_cfg.get("use_gt", False))
    use_hungarian = bool(smpl_cfg.get("use_hungarian", True))
    hungarian_cost_pose_weight = float(smpl_cfg.get("hungarian_cost_pose_weight", 1.0))
    hungarian_cost_beta_weight = float(smpl_cfg.get("hungarian_cost_beta_weight", 0.1))
    hungarian_cost_trans_weight = float(smpl_cfg.get("hungarian_cost_trans_weight", 0.0))
    hungarian_cost_mesh_trans_weight = float(smpl_cfg.get("hungarian_cost_mesh_trans_weight", 0.0))
    hungarian_cost_presence_weight = float(smpl_cfg.get("hungarian_cost_presence_weight", 0.0))
    enable_mesh_translate = bool(cfg.model.get("enable_smpl_multi_query_trans", False))

    val_dataset = instantiate(cfg.data.get("val"), _recursive_=False)
    val_dataset.seed = cfg.seed_value
    if distributed:
        _install_eval_no_pad_sampler(val_dataset, rank=rank, world_size=world_size)

    model = None
    if not use_gt_smpl:
        model = instantiate(cfg.model, _recursive_=False)
        _load_checkpoint(model, args.checkpoint)
        model.to(device)
        model.eval()

    total_mpjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_mpjpe_count = torch.zeros((), device=device, dtype=torch.float64)
    total_pa_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_pa_count = torch.zeros((), device=device, dtype=torch.float64)
    total_reproj_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_reproj_count = torch.zeros((), device=device, dtype=torch.float64)
    total_body_pve_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_body_pve_count = torch.zeros((), device=device, dtype=torch.float64)
    total_gt_people = torch.zeros((), device=device, dtype=torch.float64)
    total_selected_people = torch.zeros((), device=device, dtype=torch.float64)
    total_matched_people = torch.zeros((), device=device, dtype=torch.float64)
    total_miss_count = torch.zeros((), device=device, dtype=torch.float64)
    total_fp_count = torch.zeros((), device=device, dtype=torch.float64)
    batch_result_lines = []

    total_samples = len(val_dataset.dataset)
    _rank0_print(rank, f"total_samples: {total_samples}")
    if distributed:
        print(
            f"[INFO][rank {rank}/{world_size}] local_rank={local_rank} "
            f"device={device} local_samples={len(val_dataset.sampler)}"
        )

    data_loader = val_dataset.get_loader(epoch=0)
    processed = 0
    next_progress_log = 500

    for data_iter, batch in enumerate(data_loader):
        if args.max_batches is not None and data_iter >= args.max_batches:
            break
        batch = _normalize_batch(batch, normalize_cam)
        batch = copy_data_to_device(batch, device, non_blocking=True)

        if use_gt_smpl:
            image_hw = batch["images"].shape[-2:]
            gt_pose_encoding = extri_intri_to_pose_encoding(
                batch["extrinsics"],
                batch["intrinsics"],
                image_hw,
                pose_encoding_type="absT_quaR_FoV",
            )
            preds = {
                "smpl_pose": batch["smpl_pose"],
                "smpl_beta": batch["smpl_beta"],
                "smpl_trans": batch["smpl_trans"],
                "pose_enc_list": [gt_pose_encoding],
            }
        else:
            with torch.inference_mode():
                autocast_enabled = device.type == "cuda"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    preds = model(images=batch["images"], smpl_inputs=_metadata_inputs(batch))

        batch_size = int(batch["images"].shape[0])
        sample_metrics = _empty_sample_metrics(batch_size, device)
        if "smpl_joints3d_world" not in batch:
            if not args.skip_batch_results_file:
                _append_sample_result_lines(
                    batch_result_lines,
                    batch,
                    rank,
                    data_iter,
                    batch_size,
                    sample_metrics["gt_people"],
                    sample_metrics["selected_people"],
                    sample_metrics["matched_people"],
                    sample_metrics["miss"],
                    sample_metrics["fp"],
                    sample_metrics["mpjpe_sum"],
                    sample_metrics["mpjpe_count"],
                    sample_metrics["pa_sum"],
                    sample_metrics["pa_count"],
                    sample_metrics["reproj_sum"],
                    sample_metrics["reproj_count"],
                    sample_metrics["body_pve_sum"],
                    sample_metrics["body_pve_count"],
                )
            processed += batch_size
            continue

        pred_pose = preds["smpl_pose"]
        pred_beta = preds["smpl_beta"]
        pred_trans = preds.get("smpl_trans")
        pred_mesh_translate = preds.get("mesh_translate")
        gt_pose = batch["smpl_pose"]
        gt_beta = batch["smpl_beta"]
        gt_trans = batch.get("smpl_trans")

        # When the model predicts mesh_translate (enable_smpl_multi_query_trans),
        # derive the GT root anchor with the same logic as training.loss.
        use_mesh_translate = pred_mesh_translate is not None
        gt_mesh_translate = None
        if use_mesh_translate:
            if not normalize_cam:
                raise ValueError("mesh_translate predictions require normalize_cam=True")
            with torch.cuda.amp.autocast(enabled=False):
                gt_mesh_translate = compute_gt_mesh_translate(batch, normalize_cam=normalize_cam)

        if pred_pose.dim() == 2:
            pred_pose = pred_pose[:, None]
            pred_beta = pred_beta[:, None]
            pred_trans = pred_trans[:, None] if pred_trans is not None else None
            pred_mesh_translate = pred_mesh_translate[:, None] if pred_mesh_translate is not None else None
            gt_pose = gt_pose[:, None]
            gt_beta = gt_beta[:, None]
            gt_trans = gt_trans[:, None] if gt_trans is not None else None
            gt_mesh_translate = gt_mesh_translate[:, None] if gt_mesh_translate is not None else None

        p_pred = pred_pose.shape[1]
        p_gt = gt_pose.shape[1]
        if p_pred == 0 or p_gt == 0:
            if not args.skip_batch_results_file:
                _append_sample_result_lines(
                    batch_result_lines,
                    batch,
                    rank,
                    data_iter,
                    batch_size,
                    sample_metrics["gt_people"],
                    sample_metrics["selected_people"],
                    sample_metrics["matched_people"],
                    sample_metrics["miss"],
                    sample_metrics["fp"],
                    sample_metrics["mpjpe_sum"],
                    sample_metrics["mpjpe_count"],
                    sample_metrics["pa_sum"],
                    sample_metrics["pa_count"],
                    sample_metrics["reproj_sum"],
                    sample_metrics["reproj_count"],
                    sample_metrics["body_pve_sum"],
                    sample_metrics["body_pve_count"],
                )
            processed += batch_size
            continue

        gt_valid_mask = _build_eval_mask(batch, p_gt, pred_pose)
        sample_metrics["gt_people"] = gt_valid_mask.sum(dim=1).to(dtype=torch.long)
        gt_people = int(gt_valid_mask.sum().detach().cpu().item())
        if gt_people == 0:
            if not args.skip_batch_results_file:
                _append_sample_result_lines(
                    batch_result_lines,
                    batch,
                    rank,
                    data_iter,
                    batch_size,
                    sample_metrics["gt_people"],
                    sample_metrics["selected_people"],
                    sample_metrics["matched_people"],
                    sample_metrics["miss"],
                    sample_metrics["fp"],
                    sample_metrics["mpjpe_sum"],
                    sample_metrics["mpjpe_count"],
                    sample_metrics["pa_sum"],
                    sample_metrics["pa_count"],
                    sample_metrics["reproj_sum"],
                    sample_metrics["reproj_count"],
                    sample_metrics["body_pve_sum"],
                    sample_metrics["body_pve_count"],
                )
            processed += batch_size
            continue

        presence_prob = _presence_probs_from_preds(preds, batch, p_pred, device=pred_pose.device)
        pred_selected_mask = presence_prob > float(args.presence_threshold)
        sample_metrics["selected_people"] = pred_selected_mask.sum(dim=1).to(dtype=torch.long)
        selected_people = int(pred_selected_mask.sum().detach().cpu().item())

        pred_pose_flat = pred_pose.reshape(batch_size * p_pred, *pred_pose.shape[2:])
        pred_beta_flat = pred_beta.reshape(batch_size * p_pred, *pred_beta.shape[2:])
        pred_trans_flat = (
            pred_trans.reshape(batch_size * p_pred, *pred_trans.shape[2:])
            if pred_trans is not None
            else torch.zeros(batch_size * p_pred, 3, device=device, dtype=pred_pose.dtype)
        )
        pred_genders = ["neutral"] * (batch_size * p_pred)
        pred_joints_world_flat, _ = _decode_smpl_batch(
            pose_aa=pred_pose_flat,
            betas=pred_beta_flat,
            trans=pred_trans_flat,
            genders=pred_genders,
        )
        pred_joints_people_full = pred_joints_world_flat.reshape(
            batch_size,
            p_pred,
            pred_joints_world_flat.shape[-2],
            3,
        )
        if normalize_cam:
            pred_joints_people_full = normalize_joints_world_to_batch_gauge(
                pred_joints_people_full,
                batch["raw_extrinsics"],
                batch["avg_scale"],
            )

        gt_joints2d = _get_gt_people_joints2d(batch, p_gt)
        if gt_joints2d is None or "pose_enc_list" not in preds:
            raise KeyError("Detection evaluation requires smpl_joints2d and pose_enc_list for reprojection metrics")

        pred_pose_enc = preds["pose_enc_list"][-1]
        image_hw = batch["images"].shape[-2:]
        if use_gt_smpl:
            pred_pose_enc = extri_intri_to_pose_encoding(
                batch["extrinsics"],
                batch["intrinsics"],
                image_hw,
                pose_encoding_type="absT_quaR_FoV",
            )

        pred_extr, pred_intr = pose_encoding_to_extri_intri(
            pred_pose_enc,
            image_size_hw=image_hw,
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True,
        )
        view_count = pred_extr.shape[1]
        points_world = pred_joints_people_full.reshape(batch_size * p_pred, -1, 3)
        points_world = points_world[:, None, :, :].expand(
            batch_size * p_pred,
            view_count,
            points_world.shape[1],
            3,
        )
        pred_extr_rep = pred_extr[:, None].expand(-1, p_pred, -1, -1, -1).reshape(
            batch_size * p_pred,
            view_count,
            3,
            4,
        )
        pred_intr_rep = pred_intr[:, None].expand(-1, p_pred, -1, -1, -1).reshape(
            batch_size * p_pred,
            view_count,
            3,
            3,
        )
        pred_joints2d_full = _project_points_opencv(points_world, pred_extr_rep, pred_intr_rep)
        pred_joints2d_full = pred_joints2d_full.reshape(
            batch_size,
            p_pred,
            view_count,
            pred_joints2d_full.shape[-2],
            2,
        )
        joints2d_conf = _get_gt_people_joints2d_conf(batch, p_gt)
        gt_joints_world_full = _get_gt_people_joints(batch, p_gt)

        matched_pred_indices = []
        matched_gt_indices = []
        matched_batch_indices = []
        batch_miss = 0
        batch_fp = 0
        for sample_idx in range(batch_size):
            pred_indices = torch.where(pred_selected_mask[sample_idx])[0]
            gt_indices = torch.where(gt_valid_mask[sample_idx])[0]
            if gt_indices.numel() == 0:
                sample_fp = int(pred_indices.numel())
                sample_metrics["fp"][sample_idx] = sample_fp
                batch_fp += sample_fp
                continue
            if pred_indices.numel() == 0:
                sample_miss = int(gt_indices.numel())
                sample_metrics["miss"][sample_idx] = sample_miss
                batch_miss += sample_miss
                continue

            presence_logits_i = None
            if "smpl_presence_logits" in preds and preds["smpl_presence_logits"] is not None:
                presence_logits = preds["smpl_presence_logits"]
                if presence_logits.dim() == 1:
                    presence_logits = presence_logits[:, None]
                presence_logits_i = presence_logits[sample_idx, pred_indices]

            if not use_hungarian:
                raise ValueError("eval_multi.py currently requires loss.smpl.use_hungarian=True for multi-person matching")
            matches = _hungarian_match_smpl_cost(
                pred_pose[sample_idx, pred_indices],
                pred_beta[sample_idx, pred_indices],
                pred_trans[sample_idx, pred_indices] if pred_trans is not None else None,
                presence_logits_i,
                gt_pose[sample_idx, gt_indices],
                gt_beta[sample_idx, gt_indices],
                gt_trans[sample_idx, gt_indices] if gt_trans is not None else None,
                hungarian_cost_pose_weight,
                hungarian_cost_beta_weight,
                hungarian_cost_trans_weight,
                hungarian_cost_presence_weight,
                pred_mesh_translate=(
                    pred_mesh_translate[sample_idx, pred_indices]
                    if pred_mesh_translate is not None
                    else None
                ),
                gt_mesh_translate=(
                    gt_mesh_translate[sample_idx, gt_indices]
                    if gt_mesh_translate is not None
                    else None
                ),
                cost_mesh_trans_weight=hungarian_cost_mesh_trans_weight,
            )
            matched_pred_local = {pred_i for pred_i, _ in matches}
            matched_gt_local = {gt_i for _, gt_i in matches}
            sample_fp = int(pred_indices.numel()) - len(matched_pred_local)
            sample_miss = int(gt_indices.numel()) - len(matched_gt_local)
            sample_metrics["matched_people"][sample_idx] = len(matches)
            sample_metrics["fp"][sample_idx] = sample_fp
            sample_metrics["miss"][sample_idx] = sample_miss
            batch_fp += sample_fp
            batch_miss += sample_miss
            for pred_local_idx, gt_local_idx in matches:
                matched_batch_indices.append(sample_idx)
                matched_pred_indices.append(int(pred_indices[pred_local_idx].item()))
                matched_gt_indices.append(int(gt_indices[gt_local_idx].item()))

        matched_count = len(matched_batch_indices)
        if matched_count > 0:
            b_idx = torch.as_tensor(matched_batch_indices, device=device, dtype=torch.long)
            pred_idx = torch.as_tensor(matched_pred_indices, device=device, dtype=torch.long)
            gt_idx = torch.as_tensor(matched_gt_indices, device=device, dtype=torch.long)
            genders = _resolve_indexed_genders(batch, b_idx, gt_idx)

            pred_pose_eval = pred_pose[b_idx, pred_idx]
            pred_beta_eval = pred_beta[b_idx, pred_idx]
            pred_trans_eval = pred_trans[b_idx, pred_idx] if pred_trans is not None else None
            gt_pose_eval = gt_pose[b_idx, gt_idx]
            gt_beta_eval = gt_beta[b_idx, gt_idx]
            gt_trans_eval = gt_trans[b_idx, gt_idx] if gt_trans is not None else None

            if use_mesh_translate:
                # mesh_translate mode (enable_smpl_multi_query_trans): mirror the
                # training.loss geometry. Decode SMPL at zero translation, gauge-
                # normalize joints/vertices, then re-anchor each root to the
                # predicted (pred) / GT mesh_translate. Everything stays in the
                # batch gauge, consistent with the gauge-normalized GT joints.
                zero_trans_eval = torch.zeros(
                    matched_count, 3, device=device, dtype=pred_pose.dtype
                )
                pred_joints_world, pred_body_vertices = _decode_smpl_batch(
                    pose_aa=pred_pose_eval,
                    betas=pred_beta_eval,
                    trans=zero_trans_eval,
                    genders=genders,
                )
                gt_joints_zero, gt_body_vertices = _decode_smpl_batch(
                    pose_aa=gt_pose_eval,
                    betas=gt_beta_eval,
                    trans=zero_trans_eval,
                    genders=genders,
                )
                raw_extr_matched = batch["raw_extrinsics"][b_idx].float()
                avg_scale_matched = batch["avg_scale"][b_idx].float()
                pred_joints_world = normalize_joints_world_to_batch_gauge(
                    pred_joints_world.float(), raw_extr_matched, avg_scale_matched
                )
                pred_body_vertices = normalize_joints_world_to_batch_gauge(
                    pred_body_vertices.float(), raw_extr_matched, avg_scale_matched
                )
                gt_joints_zero = normalize_joints_world_to_batch_gauge(
                    gt_joints_zero.float(), raw_extr_matched, avg_scale_matched
                )
                gt_body_vertices = normalize_joints_world_to_batch_gauge(
                    gt_body_vertices.float(), raw_extr_matched, avg_scale_matched
                )
                pred_mesh_translate_eval = pred_mesh_translate[b_idx, pred_idx].float()
                gt_mesh_translate_eval = gt_mesh_translate[b_idx, gt_idx].float()
                pred_offset = pred_mesh_translate_eval - pred_joints_world[..., 0, :]
                gt_offset = gt_mesh_translate_eval - gt_joints_zero[..., 0, :]
                pred_joints_metric = pred_joints_world + pred_offset[..., None, :]
                pred_body_vertices = pred_body_vertices + pred_offset[..., None, :]
                gt_body_vertices = gt_body_vertices + gt_offset[..., None, :]
                gt_joints_world = gt_joints_world_full[b_idx, gt_idx].to(
                    device=pred_joints_metric.device,
                    dtype=pred_joints_metric.dtype,
                )
            else:
                pred_joints_world, pred_body_vertices = _decode_smpl_batch(
                    pose_aa=pred_pose_eval,
                    betas=pred_beta_eval,
                    trans=pred_trans_eval,
                    genders=genders,
                )
                _, gt_body_vertices = _decode_smpl_batch(
                    pose_aa=gt_pose_eval,
                    betas=gt_beta_eval,
                    trans=gt_trans_eval,
                    genders=genders,
                )
                pred_joints_metric_full = torch.zeros(
                    batch_size,
                    p_gt,
                    pred_joints_world.shape[-2],
                    3,
                    device=device,
                    dtype=pred_joints_world.dtype,
                )
                pred_joints_metric_full[b_idx, gt_idx] = pred_joints_world
                if normalize_cam:
                    pred_joints_metric_full = normalize_joints_world_to_batch_gauge(
                        pred_joints_metric_full,
                        batch["raw_extrinsics"],
                        batch["avg_scale"],
                    )
                pred_joints_metric = pred_joints_metric_full[b_idx, gt_idx]
                gt_joints_world = gt_joints_world_full[b_idx, gt_idx].to(
                    device=pred_joints_metric.device,
                    dtype=pred_joints_metric.dtype,
                )

            # MPJPE / PA-MPJPE are reported in real metric (mm), but the joints
            # above live in the batch gauge (divided by avg_scale during
            # normalization). Multiply by avg_scale to undo the uniform scale and
            # recover true metric distances. The residual gauge rotation/translation
            # is absorbed by MPJPE's root alignment (rotation-invariant norm) and by
            # PA-MPJPE's Procrustes alignment, so this is exact for both metrics.
            metric_scale = batch["avg_scale"][b_idx].to(
                device=pred_joints_metric.device, dtype=pred_joints_metric.dtype
            )
            pred_joints_metric_scaled = pred_joints_metric * metric_scale[:, None, None]
            gt_joints_world_scaled = gt_joints_world * metric_scale[:, None, None]

            mpjpe_sum_ps, mpjpe_count_ps = _mpjpe_per_sample(
                _first_n_joints(pred_joints_metric_scaled, 24),
                _first_n_joints(gt_joints_world_scaled, 24),
            )
            total_mpjpe_sum += mpjpe_sum_ps.sum().to(dtype=torch.float64)
            total_mpjpe_count += mpjpe_count_ps.sum().to(dtype=torch.float64)
            _add_metric_to_samples(
                sample_metrics["mpjpe_sum"],
                sample_metrics["mpjpe_count"],
                b_idx,
                mpjpe_sum_ps,
                mpjpe_count_ps,
            )

            pa_sum_ps, pa_count_ps = _pa_mpjpe_per_sample(
                _first_n_joints(pred_joints_metric_scaled, 24),
                _first_n_joints(gt_joints_world_scaled, 24),
                root_align=True,
            )
            total_pa_sum += pa_sum_ps.sum().to(dtype=torch.float64)
            total_pa_count += pa_count_ps.sum().to(dtype=torch.float64)
            _add_metric_to_samples(
                sample_metrics["pa_sum"],
                sample_metrics["pa_count"],
                b_idx,
                pa_sum_ps,
                pa_count_ps,
            )

            body_pve_sum_ps, body_pve_count_ps = _pve_per_sample(
                pred_body_vertices,
                gt_body_vertices,
            )
            total_body_pve_sum += body_pve_sum_ps.sum().to(dtype=torch.float64)
            total_body_pve_count += body_pve_count_ps.sum().to(dtype=torch.float64)
            _add_metric_to_samples(
                sample_metrics["body_pve_sum"],
                sample_metrics["body_pve_count"],
                b_idx,
                body_pve_sum_ps,
                body_pve_count_ps,
            )

            pred_joints_for_proj = pred_joints_metric[:, None, :, :].expand(
                matched_count,
                view_count,
                pred_joints_metric.shape[1],
                3,
            )
            pred_extr_valid = pred_extr[b_idx]
            pred_intr_valid = pred_intr[b_idx]
            pred_joints2d = _project_points_opencv(pred_joints_for_proj, pred_extr_valid, pred_intr_valid)
            gt_joints2d_valid = gt_joints2d[b_idx, gt_idx].to(
                device=pred_joints2d.device,
                dtype=pred_joints2d.dtype,
            )
            joints2d_conf_valid = None
            if joints2d_conf is not None:
                joints2d_conf_valid = joints2d_conf[b_idx, gt_idx].to(
                    device=pred_joints2d.device,
                    dtype=pred_joints2d.dtype,
                )
            reproj_sum_ps, reproj_count_ps = _reprojection_error_2d_per_sample(
                _first_n_joints(pred_joints2d, 24),
                _first_n_joints(gt_joints2d_valid, 24),
                joints2d_conf_valid[..., :24] if joints2d_conf_valid is not None else None,
            )
            total_reproj_sum += reproj_sum_ps.sum().to(dtype=torch.float64)
            total_reproj_count += reproj_count_ps.sum().to(dtype=torch.float64)
            _add_metric_to_samples(
                sample_metrics["reproj_sum"],
                sample_metrics["reproj_count"],
                b_idx,
                reproj_sum_ps,
                reproj_count_ps,
            )

        total_gt_people += torch.tensor(gt_people, device=device, dtype=torch.float64)
        total_selected_people += torch.tensor(selected_people, device=device, dtype=torch.float64)
        total_matched_people += torch.tensor(matched_count, device=device, dtype=torch.float64)
        total_miss_count += torch.tensor(batch_miss, device=device, dtype=torch.float64)
        total_fp_count += torch.tensor(batch_fp, device=device, dtype=torch.float64)
        if not args.skip_batch_results_file:
            _append_sample_result_lines(
                batch_result_lines,
                batch,
                rank,
                data_iter,
                batch_size,
                sample_metrics["gt_people"],
                sample_metrics["selected_people"],
                sample_metrics["matched_people"],
                sample_metrics["miss"],
                sample_metrics["fp"],
                sample_metrics["mpjpe_sum"],
                sample_metrics["mpjpe_count"],
                sample_metrics["pa_sum"],
                sample_metrics["pa_count"],
                sample_metrics["reproj_sum"],
                sample_metrics["reproj_count"],
                sample_metrics["body_pve_sum"],
                sample_metrics["body_pve_count"],
            )
        processed += batch_size
        if not distributed and processed >= total_samples:
            break
        if processed >= next_progress_log:
            print(f"[INFO][rank {rank}] Processed {processed} local samples")
            next_progress_log += 500

    (
        total_mpjpe_sum,
        total_mpjpe_count,
        total_pa_sum,
        total_pa_count,
        total_reproj_sum,
        total_reproj_count,
        total_body_pve_sum,
        total_body_pve_count,
        total_gt_people,
        total_selected_people,
        total_matched_people,
        total_miss_count,
        total_fp_count,
        evaluated_samples,
    ) = _reduce_metric_totals(
        [
            total_mpjpe_sum,
            total_mpjpe_count,
            total_pa_sum,
            total_pa_count,
            total_reproj_sum,
            total_reproj_count,
            total_body_pve_sum,
            total_body_pve_count,
            total_gt_people,
            total_selected_people,
            total_matched_people,
            total_miss_count,
            total_fp_count,
            torch.tensor(processed, device=device, dtype=torch.float64),
        ],
        distributed=distributed,
    )
    if not args.skip_batch_results_file:
        batch_result_lines = _gather_batch_result_lines(batch_result_lines, distributed)
    precision = total_matched_people / (total_matched_people + total_fp_count) if (total_matched_people + total_fp_count) > 0 else 0.0
    recall = total_matched_people / (total_matched_people + total_miss_count) if (total_matched_people + total_miss_count) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result_lines = [
        f"[INFO] Config: {args.config_name}",
        f"[INFO] Checkpoint: {args.checkpoint}",
        f"[INFO] Device: {gpu_name}",
        f"[INFO] use_gt_smpl: {use_gt_smpl}",
        f"[INFO] normalize_cam: {normalize_cam}",
        f"[INFO] max_img_per_gpu: {cfg.get('max_img_per_gpu')}",
        f"[INFO] enable_smpl_multi_query_trans (mesh_translate): {enable_mesh_translate}",
        f"[INFO] Evaluation mode: detection-aware presence threshold + Hungarian SMPL cost matching",
        f"[INFO] presence_threshold: {float(args.presence_threshold):.6f}",
        f"[INFO] Hungarian cost weights: "
        f"pose={hungarian_cost_pose_weight:.6f} beta={hungarian_cost_beta_weight:.6f} "
        f"trans={hungarian_cost_trans_weight:.6f} mesh_trans={hungarian_cost_mesh_trans_weight:.6f} "
        f"presence={hungarian_cost_presence_weight:.6f}",
        f"[INFO] Distributed eval: {distributed}",
        f"[INFO] World size: {world_size}",
        f"[INFO] Body metrics use first 24 SMPL joints and matched prediction/GT pairs only.",
        f"[INFO] Total samples: {total_samples}",
        f"[INFO] Evaluated samples: {int(evaluated_samples)}",
        f"[INFO] GT people: {int(total_gt_people)}",
        f"[INFO] Selected predicted people: {int(total_selected_people)}",
    ]
    result_lines.extend(_dataset_info(cfg))
    metric_lines = [
        f"[RESULT] precision: {precision:.6f}",
        f"[RESULT] recall: {recall:.6f}",
        f"[RESULT] F1: {f1:.6f}",
        f"[RESULT] matched_people: {int(total_matched_people)}",
        f"[RESULT] miss_count: {int(total_miss_count)}",
        f"[RESULT] fp_count: {int(total_fp_count)}",
    ]

    if total_mpjpe_count > 0:
        mpjpe = (total_mpjpe_sum / total_mpjpe_count) * 1000.0
        result_line = f"[RESULT] MPJPE (first 24 SMPL joints, root-aligned, mm): {mpjpe:.6f}"
        _rank0_print(rank, result_line)
        metric_lines.append(result_line)
    else:
        metric_lines.append("[WARN] No MPJPE computed.")

    if total_pa_count > 0:
        pa_mpjpe = (total_pa_sum / total_pa_count) * 1000.0
        result_line = f"[RESULT] PA-MPJPE (first 24 SMPL joints, root-aligned + Procrustes aligned, mm): {pa_mpjpe:.6f}"
        _rank0_print(rank, result_line)
        metric_lines.append(result_line)
    else:
        metric_lines.append("[WARN] No PA-MPJPE computed.")

    if total_reproj_count > 0:
        reproj_error = total_reproj_sum / total_reproj_count
        result_line = f"[RESULT] 2D reprojection error (first 24 SMPL joints, pixels): {reproj_error:.6f}"
        _rank0_print(rank, result_line)
        metric_lines.append(result_line)
    else:
        metric_lines.append("[WARN] No 2D reprojection error computed.")

    if total_body_pve_count > 0:
        body_pve = (total_body_pve_sum / total_body_pve_count) * 1000.0
        result_line = f"[RESULT] Body PVE (SMPL body vertices, mm): {body_pve:.6f}"
        _rank0_print(rank, result_line)
        metric_lines.append(result_line)
    else:
        metric_lines.append("[WARN] No Body PVE computed.")

    result_lines.extend(metric_lines)

    if rank == 0:
        output_dir = EVAL_RESULTS_ROOT / args.folder_name
        output_path = output_dir / Path(args.output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote results to: {output_path}")

        if not args.skip_batch_results_file:
            batch_output_path = output_dir / Path(args.output_batch_name)
            batch_output_path.parent.mkdir(parents=True, exist_ok=True)
            batch_output_path.write_text("\n".join(batch_result_lines) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote per-sample results to: {batch_output_path}")

    _cleanup_distributed()


if __name__ == "__main__":
    main()
