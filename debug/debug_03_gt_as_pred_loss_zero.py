#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch

from mamma_debug_common import (
    RawMammaMultiViewDataset,
    add_common_args,
    process_batch_like_trainer,
    write_summary,
)
from training.loss import compute_gt_mesh_translate, compute_smpl_loss


def scalarize_loss_dict(loss_dict):
    out = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().reshape(-1)[0].item())
        else:
            out[key] = value
    return out


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    out_dir = Path(args.output_root) / "03_gt_as_pred_loss_zero"
    dataset = RawMammaMultiViewDataset(
        mamma_root=args.mamma_root,
        scene_root=args.scene_root,
        num_views=args.num_views,
        image_size=args.image_size,
        seed=args.seed,
        seq_name=args.seq_name,
        frame=args.frame,
        require_visible_joints=args.require_visible_joints,
        min_visible_joints=args.min_visible_joints,
    )
    raw_batch = dataset.sample()
    batch = process_batch_like_trainer(raw_batch)

    with torch.no_grad():
        mesh_translate = compute_gt_mesh_translate(
            batch,
            normalize_cam=True,
            use_mamma=True,
        )
    batch = dict(batch)
    batch["mesh_translate"] = mesh_translate

    B, S = batch["extrinsics"].shape[:2]
    predictions = {
        "smpl_pose": batch["smpl_pose"].clone(),
        "smpl_beta": batch["smpl_beta"].clone(),
        "mesh_translate": batch["mesh_translate"].clone(),
        "pose_enc_list": [torch.zeros((B, S, 9), dtype=torch.float32)],
    }

    loss_dict = compute_smpl_loss(
        predictions,
        batch,
        loss_type="l1",
        loss_type_joints2d="l1",
        loss_type_joints3d="l1",
        weight_pose=1.0,
        weight_beta=0.1,
        weight_trans=0.0,
        weight_mesh_translate=1.0,
        weight_presence=0.0,
        weight_joints2d=1.0,
        weight_joints3d=1.0,
        weight_vertices=1.0,
        use_gt=True,
        normalize_cam=True,
        use_hungarian=False,
        use_mamma=True,
        joints2d_depth_min=1e-6,
    )
    losses = scalarize_loss_dict(loss_dict)
    max_abs_loss = max(abs(v) for k, v in losses.items() if k.startswith("loss_"))
    passed = max_abs_loss <= float(args.tolerance)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "losses.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "tolerance": args.tolerance,
                "max_abs_loss": max_abs_loss,
                "losses": losses,
                "note": "Nonzero joints2d/joints3d here means decoded SMPL-X, camera preprocessing, or stored joints are not mutually consistent.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.json",
        batch,
        extra={
            "mode": "gt_as_pred_loss_zero",
            "passed": passed,
            "tolerance": args.tolerance,
            "max_abs_loss": max_abs_loss,
            "losses": losses,
        },
    )
    print(f"[03] wrote {out_dir}")
    print(json.dumps(losses, indent=2))
    print(f"[03] max_abs_loss={max_abs_loss:.8f} passed={passed}")


if __name__ == "__main__":
    main()
