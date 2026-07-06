#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np

from mamma_debug_common import (
    RawMammaMultiViewDataset,
    add_common_args,
    draw_points,
    draw_points_on_diagnostic_canvas,
    draw_projected_vertices,
    load_smplx_faces,
    make_scene,
    process_batch_like_trainer,
    project_points,
    save_rgb,
    write_summary,
)


def raw_vertices_to_gauge(vertices: np.ndarray, raw_extrinsics: np.ndarray, avg_scale: float) -> np.ndarray:
    R0 = raw_extrinsics[0, :3, :3]
    t0 = raw_extrinsics[0, :3, 3]
    return ((vertices @ R0.T) + t0.reshape(1, 3)) / max(float(avg_scale), 1e-6)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    out_dir = Path(args.output_root) / "02_processed_batch_projection"
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

    images = raw_batch["images_np"]
    extrinsics = batch["extrinsics"][0].numpy()
    intrinsics = batch["intrinsics"][0].numpy()
    joints3d_gauge = batch["smpl_joints3d_world"][0].numpy()
    joints2d = raw_batch["smpl_joints2d"][0].numpy()
    joint_conf = raw_batch["smpl_joints2d_confidence"][0].numpy()
    raw_extrinsics = batch["raw_extrinsics"][0].numpy()
    avg_scale = float(batch["avg_scale"][0].item())

    faces = load_smplx_faces("neutral")
    first_view_vertices = [
        raw_vertices_to_gauge(v, raw_extrinsics, avg_scale)
        for v in raw_batch["gt_vertices_world"][0]
    ]
    make_scene(np.concatenate(first_view_vertices, axis=0), faces, extrinsics, out_dir / "gauge_scene.glb")

    max_reproj = []
    for view_idx, view_name in enumerate(raw_batch["views"]):
        overlay = images[view_idx].copy()
        diagnostic_sets = []
        for person_idx in range(joints3d_gauge.shape[1]):
            proj, depth = project_points(joints3d_gauge[view_idx, person_idx], extrinsics[view_idx], intrinsics[view_idx])
            valid = joint_conf[view_idx, person_idx] > 0
            if np.any(valid):
                max_reproj.append(float(np.nanmax(np.linalg.norm(proj[valid] - joints2d[view_idx, person_idx][valid], axis=-1))))
            overlay = draw_points(overlay, joints2d[view_idx, person_idx][valid], color=(40, 230, 110), radius=4)
            overlay = draw_points(overlay, proj[valid], color=(255, 60, 60), radius=2)
            diagnostic_sets.extend(
                [
                    (f"person_{person_idx}_dataset_joints2d", joints2d[view_idx, person_idx], (40, 230, 110)),
                    (f"person_{person_idx}_projected_gauge_joints3d", proj, (255, 60, 60)),
                ]
            )
            overlay = draw_projected_vertices(
                overlay,
                raw_vertices_to_gauge(raw_batch["gt_vertices_world"][view_idx][person_idx], raw_extrinsics, avg_scale),
                extrinsics[view_idx],
                intrinsics[view_idx],
                color=(60, 170, 255),
            )
        save_rgb(out_dir / f"view_{view_idx:02d}_{view_name}_gauge_projection.png", overlay)
        save_rgb(
            out_dir / f"view_{view_idx:02d}_{view_name}_gauge_diagnostic_canvas.png",
            draw_points_on_diagnostic_canvas(diagnostic_sets),
        )

    write_summary(
        out_dir / "summary.json",
        batch,
        extra={
            "mode": "processed_batch_gauge",
            "avg_scale": avg_scale,
            "legend": "green=dataset joints2d, red=gauge projected joints3d, blue=gauge projected sampled vertices",
            "max_joint_reprojection_px": max(max_reproj) if max_reproj else None,
            "mean_view_max_joint_reprojection_px": float(np.mean(max_reproj)) if max_reproj else None,
        },
    )
    print(f"[02] wrote {out_dir}")
    print(f"[02] avg_scale: {avg_scale}")
    print(f"[02] max joint reprojection px: {max(max_reproj) if max_reproj else 'n/a'}")


if __name__ == "__main__":
    main()
