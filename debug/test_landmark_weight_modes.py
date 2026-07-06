#!/usr/bin/env python
"""Verify the landmark loss `weight_mode` flag ("visibility" vs "mamma").

How correctness is checked
--------------------------
1. EQUIVALENCE: an INDEPENDENT numpy re-implementation of MAMMA's
   ``BEDLAM_WD.target_weight`` (ported line-for-line, operating on normalised
   [0,1] coords) is compared point-by-point against
   ``loss_smpl.compute_mamma_landmark_weight`` on random inputs. If they match,
   our port is faithful to MAMMA.
2. GATING: on a synthetic case, assert "visibility" mode zeroes an
   occluded-but-in-frame landmark's coord loss while "mamma" mode keeps it.
3. SELF-CONTAINED: assert the body-parts json and verts_512 matrix both load
   from the repo (training/data/assets), not an external path.

Run:
    conda activate mamma
    python debug/test_landmark_weight_modes.py
"""
import inspect
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from loss_smpl import (  # noqa: E402
    compute_landmark_loss,
    compute_mamma_landmark_weight,
    _BODY_PARTS_JSON,
    _load_landmark_body_parts,
)
from training.data.landmark_mask_gt import DEFAULT_VERTS512_PATHS, load_verts512_matrix  # noqa: E402

L = 512
BETA = 2.0


def mamma_target_weight_reference(gt_norm: np.ndarray, beta: float, hand_weight: float) -> np.ndarray:
    """Independent port of MAMMA BEDLAM_WD target_weight (normalised [0,1] coords).

    Mirrors lib/datasets/BEDLAM_WD.py lines ~316-339 exactly, per landmark-set row.
    """
    parts = json.load(open(_BODY_PARTS_JSON))
    N = gt_norm.shape[0]
    out = np.zeros((N, L), dtype=np.float64)
    for n in range(N):
        j = gt_norm[n]  # (L, 2), normalised
        valid_x = (0.0 <= j[:, 0]) & (j[:, 0] < 1.0)
        valid_y = (0.0 <= j[:, 1]) & (j[:, 1] < 1.0)
        valid = (valid_x & valid_y).astype(np.float64)
        tw = valid.copy()
        outside = valid == 0
        if outside.any():
            o = 2.0 * (j[outside] - 0.5)
            dist = np.linalg.norm(o, axis=1)
            tw[outside] = np.exp(-beta * np.abs(dist - 1.0))
        for key, idx in parts.items():
            idx = np.array(idx, dtype=int)
            if key in ("left_hand", "right_hand", "left_feet", "right_feet") and valid[idx].sum() > 0:
                tw[idx] = tw[idx] + tw[idx]
                if key in ("left_hand", "right_hand"):
                    tw[idx] = tw[idx] * hand_weight
        out[n] = tw
    return out


def test_equivalence_to_mamma():
    rng = np.random.default_rng(0)
    # coords in [-0.5, 1.5] -> mix of in-frame and out-of-frame; avoid exact 0/1.
    gt = rng.uniform(-0.5, 1.5, size=(4, L, 2)).astype(np.float64)
    for hw in (1.0, 2.0):
        ref = mamma_target_weight_reference(gt, BETA, hw)
        got = compute_mamma_landmark_weight(
            torch.from_numpy(gt), beta=BETA, hand_weight=hw
        ).numpy()
        max_diff = np.abs(ref - got).max()
        assert max_diff < 1e-6, f"hand_weight={hw}: mismatch vs MAMMA ref, max_diff={max_diff}"
        print(f"[equivalence] hand_weight={hw}: max|ours-MAMMAref| = {max_diff:.2e}  OK")


def test_gating_behavior():
    gt = torch.full((1, 1, L, 2), 0.5)          # all in-frame
    logvar = torch.zeros(1, 1, L)
    vis = torch.ones(1, 1, L)
    vis[0, 0, 0] = 0.0                           # landmark 0: OCCLUDED (but in-frame)
    has = torch.ones(1)
    pred = gt.clone()
    pred[0, 0, 0] = torch.tensor([0.9, 0.9])     # big error only at the occluded landmark

    lv = compute_landmark_loss(pred, logvar, gt, vis, has, loss_type="l2", weight_mode="visibility")
    lm = compute_landmark_loss(pred, logvar, gt, vis, has, loss_type="l2", weight_mode="mamma")
    assert lv["loss_landmark"].item() == 0.0, "visibility mode must gate the occluded error to 0"
    assert lm["loss_landmark"].item() > 0.0, "mamma mode must still count the occluded error"
    print(f"[gating] visibility={lv['loss_landmark'].item():.6f} (gated)  "
          f"mamma={lm['loss_landmark'].item():.6f} (counted)  OK")


def test_self_contained():
    assert os.path.commonpath([REPO, _BODY_PARTS_JSON]) == REPO, "body-parts json not under repo"
    assert os.path.isfile(_BODY_PARTS_JSON), _BODY_PARTS_JSON
    _load_landmark_body_parts()  # must not raise
    repo_verts = DEFAULT_VERTS512_PATHS[0]
    assert os.path.commonpath([REPO, repo_verts]) == REPO, "verts_512 first path not under repo"
    assert os.path.isfile(repo_verts), f"repo verts_512 missing: {repo_verts}"
    mat = load_verts512_matrix()                 # resolves to the repo copy
    assert mat.shape == (512, 10475), mat.shape
    print(f"[self-contained] body-parts json + verts_512 both load from repo  OK\n"
          f"    json:  {os.path.relpath(_BODY_PARTS_JSON, REPO)}\n"
          f"    verts: {os.path.relpath(repo_verts, REPO)}  shape={mat.shape}")


if __name__ == "__main__":
    test_self_contained()
    test_equivalence_to_mamma()
    test_gating_behavior()
    print("\nALL TESTS PASSED")
