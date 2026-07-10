#!/usr/bin/env python3
"""Standalone corrupt-file scanner for the raw Mamma dataset.

Walks a dataset root and validates every ``.data.pyd`` (pickle), image ``.jpg``,
and instance-mask ``.mask.jpg`` EXACTLY the way the training dataloader does
(``pickle.load`` / ``cv2.imread`` / ``cv2.imread(..., IMREAD_GRAYSCALE)``), so a
file counted here is precisely what would make ``SysSMPLMultiDataset`` skip/swap
it. Runs in parallel with a progress bar and writes a categorised bad-file list.

Usage:
    python scan_corrupt.py /mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma
    python scan_corrupt.py <root> --workers 16 --out corrupt_report.txt
    python scan_corrupt.py <root> --sample 0.05      # quick estimate on 5%

The bad-file list can later be fed to the loader to pre-exclude known-bad views.
"""
import argparse
import os
import pickle
import random
import sys
import time
from functools import partial
from multiprocessing import Pool


def _init_worker():
    """Silence libpng/cv2 C-level stderr spam and pin cv2 to 1 thread per worker."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def _check(task):
    """Return (kind, path) if the file is BAD, else None."""
    path, kind = task
    try:
        if os.path.getsize(path) == 0:
            return (kind, path)
        if kind == "pyd":
            with open(path, "rb") as f:
                pickle.load(f)
        else:
            import cv2
            flag = cv2.IMREAD_GRAYSCALE if kind == "mask" else cv2.IMREAD_COLOR
            if cv2.imread(path, flag) is None:
                return (kind, path)
    except Exception:
        return (kind, path)
    return None


def enumerate_files(root):
    """One os.walk pass -> lists of (path, kind) for pyd / img / mask."""
    pyds, imgs, masks = [], [], []
    n_dir = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n_dir += 1
        if n_dir % 2000 == 0:
            sys.stderr.write(f"\r  scanning dirs: {n_dir} (pyd={len(pyds)} img={len(imgs)} mask={len(masks)})")
            sys.stderr.flush()
        for fn in filenames:
            if fn.endswith(".data.pyd"):
                pyds.append((os.path.join(dirpath, fn), "pyd"))
            elif fn.endswith(".mask.jpg"):
                masks.append((os.path.join(dirpath, fn), "mask"))
            elif fn.endswith((".jpg", ".jpeg", ".png")):
                imgs.append((os.path.join(dirpath, fn), "img"))
    sys.stderr.write("\r" + " " * 80 + "\r")
    return pyds, imgs, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root, e.g. /mnt/.../SMPL_multi_dataset/mamma")
    ap.add_argument("--workers", type=int, default=min((os.cpu_count() or 8), 16))
    ap.add_argument("--out", default="corrupt_report.txt", help="bad-file list output path")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of files to check (1.0 = all)")
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"root not found: {args.root}")

    print(f"[1/3] enumerating files under {args.root} ...")
    pyds, imgs, masks = enumerate_files(args.root)
    totals = {"pyd": len(pyds), "img": len(imgs), "mask": len(masks)}
    print(f"      found  pyd={totals['pyd']:,}  img={totals['img']:,}  mask={totals['mask']:,}  "
          f"(total {sum(totals.values()):,})")

    tasks = pyds + imgs + masks
    if args.sample < 1.0:
        rng = random.Random(0)
        tasks = [t for t in tasks if rng.random() < args.sample]
        print(f"      sampling {args.sample:.0%} -> checking {len(tasks):,} files")

    # progress bar (tqdm if available, else a lightweight fallback)
    try:
        from tqdm import tqdm
        progress = lambda it, total: tqdm(it, total=total, unit="file", smoothing=0.05)
    except Exception:
        def progress(it, total):
            t0 = time.time()
            for i, x in enumerate(it, 1):
                if i % 2000 == 0 or i == total:
                    rate = i / max(time.time() - t0, 1e-6)
                    sys.stderr.write(f"\r  [2/3] checked {i:,}/{total:,} ({100*i/total:.1f}%) {rate:.0f} file/s")
                    sys.stderr.flush()
                yield x
            sys.stderr.write("\n")

    print(f"[2/3] validating with {args.workers} workers ...")
    bad = {"pyd": [], "img": [], "mask": []}
    checked = {"pyd": 0, "img": 0, "mask": 0}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res, task in progress(
            zip(pool.imap_unordered(_check, tasks, chunksize=args.chunksize), tasks),
            total=len(tasks),
        ):
            checked[task[1]] += 1
            if res is not None:
                bad[res[0]].append(res[1])

    print("\n[3/3] summary")
    print("=" * 60)
    grand_bad = 0
    for kind in ("pyd", "img", "mask"):
        b, c = len(bad[kind]), checked[kind]
        grand_bad += b
        pct = (100.0 * b / c) if c else 0.0
        print(f"  {kind:4s}: {b:>6,} bad / {c:>9,} checked  ({pct:.3f}%)")
    print("=" * 60)
    print(f"  TOTAL bad: {grand_bad:,}")

    with open(args.out, "w") as f:
        for kind in ("pyd", "img", "mask"):
            for p in sorted(bad[kind]):
                f.write(f"{kind}\t{p}\n")
    print(f"  bad-file list written -> {args.out}")


if __name__ == "__main__":
    main()
