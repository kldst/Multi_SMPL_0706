#!/usr/bin/env python
"""Verify the MAMMA-style contact loss addition.

A. EQUIVALENCE: compute_contact_loss (no sentinels, has_smpl=1) matches an
   independent port of MAMMA's focal_loss (landmarks/lib/models/models_2d/loss.py).
B. SENTINEL MASK: target == -1 entries (sdf_vertices absent) are ignored.
C. END-TO-END: a VGGT built with landmark_predict_contact=True produces
   smpl_contact_logits / smpl_floor_contact_logits, and compute_smpl_loss returns
   finite loss_contact / loss_floor_contact that backprop. (needs CUDA; skipped otherwise)

Run:
    conda activate mamma
    python debug/test_contact_loss.py
"""
import inspect
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
os.chdir(REPO)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from loss_smpl import compute_contact_loss  # noqa: E402


def mamma_focal_reference(pred_logits, target, alpha=0.9, gamma=2.0):
    """Independent port of MAMMA focal_loss (mean over all entries)."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p = torch.sigmoid(pred_logits)
    pt = target * p + (1 - target) * (1 - p)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def test_equivalence():
    torch.manual_seed(0)
    pred = torch.randn(3, 4, 512)
    target = (torch.rand(3, 4, 512) > 0.8).float()      # sparse positives
    has = torch.ones(3)                                  # all valid people, no sentinels
    ref = mamma_focal_reference(pred, target).item()
    got = compute_contact_loss(pred, target, has).item()
    assert abs(ref - got) < 1e-6, f"contact loss != MAMMA focal ref: {ref} vs {got}"
    print(f"[equivalence] compute_contact_loss vs MAMMA focal: |diff|={abs(ref-got):.2e}  OK")


def test_sentinel_mask():
    torch.manual_seed(1)
    pred = torch.randn(2, 4, 512)
    target = (torch.rand(2, 4, 512) > 0.8).float()
    has = torch.ones(2)
    full = compute_contact_loss(pred, target, has).item()
    # invalidate person 1 entirely with the -1 sentinel -> loss must equal person-0-only.
    target_masked = target.clone()
    target_masked[1] = -1.0
    masked = compute_contact_loss(pred, target_masked, has).item()
    person0_only = mamma_focal_reference(pred[:1], target[:1]).item()
    assert abs(masked - person0_only) < 1e-6, f"sentinel not masked: {masked} vs {person0_only}"
    assert abs(masked - full) > 1e-6, "masking had no effect (test is degenerate)"
    print(f"[sentinel]  masked loss = person-0-only loss ({masked:.6f})  OK  (full={full:.6f})")


def test_end_to_end():
    if not torch.cuda.is_available():
        print("[end-to-end] CUDA unavailable -> skipped")
        return
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.data.datasets.sys_smpl_multi import SysSMPLMultiDataset
    from training.train_utils.normalization import normalize_camera_extrinsics_points_and_3djoints_batch
    from loss_smpl import compute_smpl_loss
    import numpy as np

    with initialize_config_dir(config_dir=os.path.join(REPO, "training", "config"), version_base=None):
        cfg = compose(config_name="mamma_overfit")
    assert bool(cfg.model.landmark_predict_contact), "config should enable landmark_predict_contact"

    model = instantiate(cfg.model, _recursive_=False).cuda().eval()  # random init (plumbing test)
    model.aggregator.requires_grad_(False)  # frozen (matches config) -> cheap backward
    print("[end-to-end] built VGGT with landmark_predict_contact=True (random init, frozen aggregator)")

    cc = OmegaConf.create(dict(img_size=518, patch_size=14, rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=False,
        inside_random=False, allow_duplicate_img=False, fixed_view_sampling=True,
        augs=dict(scales=None)))
    dcfg = cfg.data.train.dataset.dataset_configs[0]
    ds = SysSMPLMultiDataset(common_conf=cc, split="test",
                             SysSMPL_DIR=dcfg.SysSMPL_DIR, SysSMPL_ANNOTATION_DIR=dcfg.SysSMPL_ANNOTATION_DIR,
                             max_num_people=int(dcfg.max_num_people), min_num_images=4,
                             emit_landmarks=True, emit_person_mask=True, emit_contact=True,
                             max_sequences=1, max_frames_per_sequence=1)
    data = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)

    def t(x, dt=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dt).unsqueeze(0)
    imgs = torch.from_numpy(np.stack(data["images"], 0).astype("float32") / 255.0).permute(0, 3, 1, 2).unsqueeze(0).cuda()
    batch = {
        "images": imgs.cpu(),
        "extrinsics": t(np.stack(data["extrinsics"], 0)),
        "intrinsics": t(np.stack(data["intrinsics"], 0)),
        "smpl_pose": t(data["smpl_pose"]), "smpl_beta": t(data["smpl_beta"]),
        "smpl_trans": t(data["smpl_trans"]), "smpl_gender": t(data["smpl_gender"]),
        "has_smpl": t(data["has_smpl"]), "num_people": t(data["num_people"], torch.long),
        "smpl_joints2d": t(data["smpl_joints2d"]), "smpl_joints3d_world": t(data["smpl_joints3d_world"]),
        "smpl_joints2d_confidence": t(data["smpl_joints2d_confidence"]),
        "smpl_landmarks2d": t(data["smpl_landmarks2d"]),
        "smpl_landmarks2d_visibility": t(data["smpl_landmarks2d_visibility"]),
        "smpl_contact": t(data["smpl_contact"]), "smpl_floor_contact": t(data["smpl_floor_contact"]),
        "person_mask": t(data["person_mask"]),
    }
    # cam-normalize to cam0 gauge (as the trainer does)
    ne, _, _, nj, _, avg = normalize_camera_extrinsics_points_and_3djoints_batch(
        extrinsics=batch["extrinsics"], cam_points=None, world_points=None,
        joints3d_world=batch["smpl_joints3d_world"], depths=None, scale_by_extrinsics=True, point_masks=None)
    batch["raw_extrinsics"] = batch["extrinsics"].clone()
    batch["extrinsics"] = ne
    batch["smpl_joints3d_world"] = nj
    batch["avg_scale"] = avg
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(imgs, smpl_inputs={})
    assert "smpl_contact_logits" in pred, "head did not output smpl_contact_logits"
    assert "smpl_floor_contact_logits" in pred, "head did not output smpl_floor_contact_logits"
    B, S, P, L = pred["smpl_contact_logits"].shape
    print(f"[end-to-end] head outputs smpl_contact_logits shape (B,S,P,L)=({B},{S},{P},{L})  OK")

    smpl_cfg = OmegaConf.to_container(cfg.loss.smpl, resolve=True)
    smpl_cfg.pop("weight", None)
    out = compute_smpl_loss(pred, batch, **smpl_cfg)
    lc, lf = out["loss_contact"], out["loss_floor_contact"]
    assert torch.isfinite(lc) and torch.isfinite(lf), f"non-finite contact loss: {lc}, {lf}"
    out["loss_smpl"].backward()
    gc = model.smpl_dense_landmark_head.dec_contact.weight.grad
    assert gc is not None and torch.isfinite(gc).all(), "no/invalid grad into dec_contact"
    print(f"[end-to-end] loss_contact={lc.item():.5f} loss_floor_contact={lf.item():.5f} "
          f"contact_positive_frac={out['contact_positive_frac'].item():.4f}")
    print("[end-to-end] backward OK, gradient flows into dec_contact  OK")


if __name__ == "__main__":
    test_equivalence()
    test_sentinel_mask()
    test_end_to_end()
    print("\nALL CONTACT TESTS PASSED")
