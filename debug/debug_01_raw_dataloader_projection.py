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
    project_points,
    save_rgb,
    write_summary,
)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    out_dir = Path(args.output_root) / "01_raw_dataloader_projection"
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
    batch = dataset.sample()

    images = batch["images_np"]
    extrinsics = batch["extrinsics"][0].numpy()
    intrinsics = batch["intrinsics"][0].numpy()
    joints3d = batch["smpl_joints3d_world"][0].numpy()
    joints2d = batch["smpl_joints2d"][0].numpy()
    joint_conf = batch["smpl_joints2d_confidence"][0].numpy()
    vertices = batch["gt_vertices_world"]

    faces = load_smplx_faces("neutral")
    first_view_vertices = np.concatenate(vertices[0], axis=0)
    make_scene(first_view_vertices, faces, extrinsics, out_dir / "raw_scene.glb")

    max_reproj = []
    for view_idx, view_name in enumerate(batch["views"]):
        image = images[view_idx]
        overlay = image.copy()
        diagnostic_sets = []
        for person_idx in range(joints3d.shape[1]):
            proj, depth = project_points(joints3d[view_idx, person_idx], extrinsics[view_idx], intrinsics[view_idx])
            valid = joint_conf[view_idx, person_idx] > 0
            if np.any(valid):
                max_reproj.append(float(np.nanmax(np.linalg.norm(proj[valid] - joints2d[view_idx, person_idx][valid], axis=-1))))
            overlay = draw_points(overlay, joints2d[view_idx, person_idx][valid], color=(40, 230, 110), radius=4)
            overlay = draw_points(overlay, proj[valid], color=(255, 60, 60), radius=2)
            diagnostic_sets.extend(
                [
                    (f"person_{person_idx}_dataset_joints2d", joints2d[view_idx, person_idx], (40, 230, 110)),
                    (f"person_{person_idx}_projected_joints3d", proj, (255, 60, 60)),
                ]
            )
            overlay = draw_projected_vertices(
                overlay,
                vertices[view_idx][person_idx],
                extrinsics[view_idx],
                intrinsics[view_idx],
                color=(60, 170, 255),
            )
        save_rgb(out_dir / f"view_{view_idx:02d}_{view_name}_raw_projection.png", overlay)
        save_rgb(
            out_dir / f"view_{view_idx:02d}_{view_name}_raw_diagnostic_canvas.png",
            draw_points_on_diagnostic_canvas(diagnostic_sets),
        )

    write_summary(
        out_dir / "summary.json",
        batch,
        extra={
            "mode": "raw",
            "legend": "green=dataset joints2d, red=projected joints3d, blue=sampled projected vertices",
            "max_joint_reprojection_px": max(max_reproj) if max_reproj else None,
            "mean_view_max_joint_reprojection_px": float(np.mean(max_reproj)) if max_reproj else None,
        },
    )
    print(f"[01] wrote {out_dir}")
    print(f"[01] max joint reprojection px: {max(max_reproj) if max_reproj else 'n/a'}")


if __name__ == "__main__":
    main()
