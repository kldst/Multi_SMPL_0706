# Harmony4D SMPL → SMPL-X

`convert_harmony4d_smpl_to_smplx.py` converts the two neutral-SMPL people in
Harmony4D into the body-only SMPL-X convention used by this repository's MAMMA
training path.

The conversion has two stages:

1. Map the stored 6890-vertex SMPL GT mesh to the 10475-vertex SMPL-X topology
   with `mamma/model_transfer/smpl2smplx_deftrafo_setup.pkl`.
2. Optimize SMPL-X pose, shape and translation against the 8881 valid body
   vertices in `smplx_mask_ids.npy`.

The source world coordinate frame is not changed. Fingers, jaw, and eye poses
are zero because classic SMPL has no corresponding rotations.

## Quick validation run

Run from the repository root:

```bash
source /train-data-3-hdd/yian/conda/etc/profile.d/conda.sh
conda activate mamma
python smpl_transfer/convert_harmony4d_smpl_to_smplx.py \
  --sequence 01_hugging/001_hugging \
  --frames 00001 \
  --steps 300 \
  --visualize 1 \
  --device cuda:0
```

This writes:

```text
smpl_transfer/output/
├── 01_hugging/001_hugging/00001.npy
├── _validation/01_hugging/001_hugging/00001_aria01.png
├── _validation/01_hugging/001_hugging/00001_aria02.png
└── conversion_report.json
```

The PNG compares the deformation-transferred SMPL surface with the fitted
SMPL-X surface in three orthographic views and plots per-vertex error in mm.
`conversion_report.json` contains mean, median, p95, and maximum errors.

Use `--dry-run` to inspect scope without loading a body model:

```bash
python smpl_transfer/convert_harmony4d_smpl_to_smplx.py \
  --sequence 01_hugging/001_hugging --dry-run
```

## Batch conversion

Convert all sequences (existing outputs are resumed/skipped):

```bash
python smpl_transfer/convert_harmony4d_smpl_to_smplx.py \
  --all --steps 300 --device cuda:0
```

For two GPUs, split the sorted frame list into disjoint shards (run in two
terminals). Each process writes a separate report, so report updates do not
race:

```bash
CUDA_VISIBLE_DEVICES=0 python smpl_transfer/convert_harmony4d_smpl_to_smplx.py \
  --all --num-shards 2 --shard-index 0 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python smpl_transfer/convert_harmony4d_smpl_to_smplx.py \
  --all --num-shards 2 --shard-index 1 --device cuda:0
```

Useful options:

- `--sequence ACTIVITY/SEQUENCE` can be repeated.
- `--max-frames-per-sequence N` is useful for a dataset-wide smoke test.
- `--num-shards N --shard-index I` distributes conversion across GPUs.
- `--visualize N` renders the first `N` converted frames of every sequence.
- `--no-store-vertices` stores parameters and joints only, reducing disk use.
- `--overwrite` recomputes files that already exist.
- `--fail-fast` stops at the first corrupt or incompatible source annotation.

## Output schema

Every frame `.npy` is a pickled dictionary containing `aria01`, `aria02`, and
`_meta`. Each person contains:

- Harmony4D loader fields: `global_orient` (3), `body_pose` (69), `betas` (10),
  `transl` (3), `joints` (24, 3), and optionally `vertices` (10475, 3).
- MAMMA-style fields: `pose_world` (165), `shape` (16), `trans_world` (3), and
  `gender="neutral"`.
- `fit_metrics`, reported in millimetres on the valid body-transfer mask.

For `training/data/datasets/harmony4d.py`, set:

```yaml
body_model_type: smplx
smplx_annotation_root: /train-data-3-hdd/yian/Multi_SMPL_0706/smpl_transfer/output
```

The converter intentionally does not create camera-space `vertices2d` or
visibility/contact labels: Harmony4D's multi-view loader projects the fitted
world-space joints using its own calibrated cameras, while MAMMA-only dense
labels are not present in Harmony4D.

## Person-mask GT

`generate_harmony4d_masks.py` first rectifies each Harmony4D fisheye RGB into a
pinhole image and computes its new intrinsic matrix. It then projects the
converted SMPL-X mesh into that rectified image as an identity-safe prompt, and
SAM2 expands it to clothes/hair. Mesh depth and nearest-prompt distance make
person overlap mutually exclusive. The resulting instance PNG uses the raw
MAMMA label convention: background `0`, `aria01=1`, `aria02=2`.

Use the PoseGAM environment, which already contains SAM2 and its tiny model:

```bash
source /train-data-3-hdd/yian/conda/etc/profile.d/conda.sh
conda activate posegam

python smpl_transfer/generate_harmony4d_masks.py \
  --sequence 01_hugging/001_hugging \
  --frames 00001 \
  --camera cam01 \
  --refiner sam2 \
  --visualize 1 \
  --device cuda:0
```

Outputs are written below `smpl_transfer/masks/`:

```text
01_hugging/001_hugging/cam01/00001.mask.png
01_hugging/001_hugging/cam01/masks/mask_00001_01.png
01_hugging/001_hugging/cam01/masks/mask_00001_02.png
01_hugging/001_hugging/cam01/rectified/00001.jpg
01_hugging/001_hugging/cam01/rectified/00001.camera.npz
_validation/01_hugging/001_hugging/cam01/00001_overlay.jpg
mask_report.json
```

The two files in `masks/` follow MAMMA-off's per-person binary-mask layout.
`<frame>.mask.png` is the single mutually-exclusive instance map consumed by
the repository's MAMMA-style mask supervision. For a deterministic body-only
silhouette without SAM2, pass `--refiner none`; this needs only NumPy/OpenCV.

These are calibrated pseudo-GT masks, not manually annotated silhouettes:
SAM2 captures clothing/hair while the fitted mesh fixes identity and overlap.
Always inspect the validation panels on several interaction-heavy frames before
starting training.

Batch generation resumes complete RGB/mask/camera bundles. Multiple GPUs can
use disjoint shards:

```bash
CUDA_VISIBLE_DEVICES=0 python smpl_transfer/generate_harmony4d_masks.py \
  --all --num-shards 2 --shard-index 0 --refiner sam2 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python smpl_transfer/generate_harmony4d_masks.py \
  --all --num-shards 2 --shard-index 1 --refiner sam2 --device cuda:0
```

## Export minimal MAMMA training data

`export_harmony4d_mamma.py` converts the fitted parameters and the already
rectified RGB/mask/camera bundle into the raw directory format discovered by
`SysSMPLMultiDataset`. It does not rectify again. It writes only fields needed
by `training/config/mamma_mask_dpt.yaml`:

- RGB image, rectified to pinhole;
- instance mask (`0=background`, `1/2=person`), generated directly in the
  rectified image coordinates;
- `pose_world`, `shape`, `trans_world`, `gender`, and `person_idx`;
- `cam_int` and `cam_ext`, needed to recreate joints2d/3d and geometry losses.

The exporter intentionally requires both `rectified/<frame>.jpg` and
`rectified/<frame>.camera.npz`. Old masks generated before this ordering change
are not accepted, preventing a distorted mask from being paired with a pinhole
camera.

It deliberately omits vertices2d/3d, landmarks, visibility, contact/SDF, and
normals. A small one-view validation export is:

```bash
conda activate mamma
python smpl_transfer/export_harmony4d_mamma.py \
  --sequence 01_hugging/001_hugging \
  --frames 00053 \
  --camera cam01 \
  --min-views 1 \
  --visualize 1 \
  --overwrite
```

Production export requires eight complete masks per frame by default:

```bash
python smpl_transfer/export_harmony4d_mamma.py --all --min-views 8
```

Output layout:

```text
smpl_transfer/mamma_harmony4d/
├── _validation/h4d_01_hugging_001_hugging/cam01/00053_overlay.jpg
├── png/h4d_01_hugging_001_hugging/cam01/
│   ├── 00053.jpg
│   ├── 00053.mask.png
│   └── 00053.data.pyd
├── train_data.txt
└── export_report.json
```

Point both `SysSMPL_DIR` and `SysSMPL_ANNOTATION_DIR` at
`smpl_transfer/mamma_harmony4d`. This extracted directory is what the current
loader reads directly; tar packaging is unnecessary for training.
