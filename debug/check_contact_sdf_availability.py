#!/usr/bin/env python
"""Empirically check whether missing-sdf frames are genuinely NON-contact.

sdf_vertices (the person-person contact source) is only exported for ~half the
SMPL_multi_dataset frames. This script tests whether the frames WITHOUT sdf are
actually non-contact by measuring the minimum vertex-vertex distance between the
two closest people (from vertices3d, which is always present).

Finding (2026-07-06, first 8 frames/dataset):
  * EMPTY-sdf frames: median inter-person min-dist ~0.53 m, 0% within 0.20 m
    -> genuinely far apart / not touching.
  * HAS-sdf frames:  median ~0.004 m; sdf-contact>0 <=> touching (~3 mm),
    sdf-contact==0 <=> apart (~12 cm)  -> distance proxy validates sdf.
So "missing sdf" == genuine no-contact, which justifies treating those frames as
negatives (contact_missing_as_negative=True, MAMMA behavior).

Run:
    conda activate mamma
    python debug/check_contact_sdf_availability.py [--per-dataset 8]
"""
import argparse
import glob
import os
import pickle
from itertools import combinations

import numpy as np

ROOT = "/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma"
try:
    from scipy.spatial import cKDTree
    HAVE_KD = True
except Exception:
    HAVE_KD = False


def min_interperson_dist(verts_list, rng, n_sub=1500):
    subs = []
    for v in verts_list:
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (10475, 3):
            return None
        subs.append(v[rng.choice(10475, n_sub, replace=False)])
    best = np.inf
    for a, b in combinations(range(len(subs)), 2):
        if HAVE_KD:
            d, _ = cKDTree(subs[b]).query(subs[a], k=1)
            best = min(best, float(d.min()))
        else:
            A, B = subs[a], subs[b]
            for i in range(0, len(A), 500):
                best = min(best, float(np.linalg.norm(A[i:i+500, None] - B[None], axis=-1).min()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    empty_d, has_d, has_pos = [], [], []
    n = 0
    for ds in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, ds)
        if not os.path.isdir(d):
            continue
        for py in glob.glob(d + "/**/png/*/*/0000.data.pyd", recursive=True)[: args.per_dataset]:
            try:
                data = pickle.load(open(py, "rb"))
            except Exception:
                continue
            if len(data) < 2:
                continue
            persons = list(data.values())
            md = min_interperson_dist([p.get("vertices3d") for p in persons], rng)
            if md is None:
                continue
            n += 1
            if np.asarray(persons[0].get("sdf_vertices", [])).size == 10475:
                has_d.append(md)
                has_pos.append(sum(int((np.asarray(p["sdf_vertices"]) < 0.01).sum()) for p in persons))
            else:
                empty_d.append(md)

    empty_d, has_d, has_pos = map(np.array, (empty_d, has_d, has_pos))
    print(f"analysed {n} multi-person frames (kdtree={HAVE_KD})")
    print(f"\n--- EMPTY-sdf frames: {len(empty_d)} ---")
    for th in (0.02, 0.05, 0.10, 0.20):
        if len(empty_d):
            print(f"  min inter-person dist < {th:.2f} m : {(empty_d < th).mean()*100:5.1f}%  ({int((empty_d<th).sum())}/{len(empty_d)})")
    if len(empty_d):
        print(f"  median = {np.median(empty_d):.3f} m | MIN (closest case) = {empty_d.min():.3f} m")
        print(f"  5 closest empty-sdf frames (m): {np.round(np.sort(empty_d)[:5],3).tolist()}")
    print(f"\n--- HAS-sdf frames: {len(has_d)} (control) ---")
    if len(has_d):
        for th in (0.02, 0.05, 0.10, 0.20):
            print(f"  min inter-person dist < {th:.2f} m : {(has_d < th).mean()*100:5.1f}%")
        m = has_pos > 0
        print(f"  median = {np.median(has_d):.3f} m")
        print(f"  contact>0 frames: {m.mean()*100:.0f}%  median-dist={np.median(has_d[m]) if m.any() else float('nan'):.3f} m")
        print(f"  contact==0 frames: median-dist={np.median(has_d[~m]) if (~m).any() else float('nan'):.3f} m")
    # Contact is sdf < 1cm, i.e. people essentially touching (<~2cm apart). Judge at
    # THAT scale, not at 20cm: a 10-20cm "near-but-not-touching" tail is still no-contact.
    verdict = "missing-sdf == genuine no-contact (none within contact scale)" \
        if len(empty_d) and (empty_d < 0.02).mean() == 0.0 \
        else "some missing-sdf frames are within ~2cm -> check exact min dist before trusting True"
    print(f"\nVERDICT (contact scale <2cm): {verdict}")
    print("  NOTE: for the closest frames, confirm with an EXACT full-vertex min-distance")
    print("        check (subsampled proxy can under/over-estimate by local vertex spacing).")


if __name__ == "__main__":
    main()
