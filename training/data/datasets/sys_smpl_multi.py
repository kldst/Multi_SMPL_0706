import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_depth, read_image_cv2
from training.data.landmark_mask_gt import (
    NUM_SMPLX_VERTS,
    downsample_vertices,
    downsample_visibility,
    rasterize_person_patch_mask,
)


class SysSMPLMultiDataset(BaseDataset):
    """
    Dataset for multi-person raw Mamma_mv_split scenes.

    One multi-person, multi-view sample = the same time step (frame) seen from
    ``img_per_seq`` views. Ground truth is read from the per-view ``.data.pyd``
    (SMPL-X world params, per-camera intrinsics/extrinsics, projected
    ``vertices2d`` / ``vertex_visibility``) plus the per-view instance mask
    ``*.mask.jpg`` (pixel value == person_idx + 1). Layout:

        <root>/.../png/<seq>/<IOI_view>/<frame>.jpg  (+ .data.pyd, .mask.jpg)

    Dense-landmark (``smpl_landmarks2d`` / ``_visibility``) and per-person mask
    (``person_mask``) GT are emitted when ``emit_landmarks`` / ``emit_person_mask``
    are set (derived on the fly via the ``verts_512`` matrix).
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SysSMPL_DIR: str = None,
        SysSMPL_ANNOTATION_DIR: str = None,
        min_num_images: int = 20,
        max_num_people: Optional[int] = None,
        len_train: Optional[int] = None,
        emit_landmarks: bool = False,
        emit_person_mask: bool = False,
        # Resolution of the emitted person_mask GT: mask grid = processed image
        # size // person_mask_stride.  None keeps the legacy patch-grid behaviour
        # (stride = patch_size -> 37x37 for 518/14).  Set 2 to supervise a
        # pixel-level mask head (259x259), matching model.person_mask_down_ratio.
        person_mask_stride: Optional[int] = None,
        emit_contact: bool = False,
        contact_threshold: float = 0.01,
        # When sdf_vertices is absent for a frame, how to label person-person contact:
        #   True  (MAMMA): treat as 0 (no-contact) -> supervised as negatives.
        #   False        : mark -1 -> ignored by the loss (no supervision).
        # Empirically (debug/check_contact_sdf_availability.py) missing-sdf frames are
        # genuinely non-contact (people >0.5m apart), so True is the justified default.
        contact_missing_as_negative: bool = True,
        landmark_matrix_path: Optional[str] = None,
        landmark_visibility_threshold: float = 0.5,
        max_sequences: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
    ):
        super().__init__(common_conf=common_conf)
        # Optional caps to bound the (raw Mamma) build, which cold-reads one pyd
        # per (frame,view). Default None = load everything. Set small (e.g. 1-2
        # sequences) for fast debug / quick-iteration startup on a cold disk.
        self.max_sequences = max_sequences
        self.max_frames_per_sequence = max_frames_per_sequence

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.fixed_view_sampling = getattr(common_conf, "fixed_view_sampling", False)

        if SysSMPL_DIR is None or SysSMPL_ANNOTATION_DIR is None:
            raise ValueError("SysSMPL_DIR and SysSMPL_ANNOTATION_DIR must be specified.")

        self.image_root = Path(SysSMPL_DIR).expanduser()
        # Both roots point at a raw Mamma_mv_split split root; the loader scans
        # for png/<seq>/<IOI_view>/<frame>.data.pyd under them.
        self.data_root = Path(SysSMPL_ANNOTATION_DIR).expanduser()

        if not self.image_root.is_dir():
            raise ValueError(f"Image root not found: {self.image_root}")
        if not self.data_root.is_dir():
            raise ValueError(f"Merged out_data root not found: {self.data_root}")

        self.min_num_images = min_num_images
        self.max_num_people = max_num_people

        # Dense-landmark / per-person-mask GT (raw Mamma_mv_split only). Both are
        # derived on the fly from data the pyd already ships (vertices2d /
        # vertex_visibility / *.mask.jpg) via the ``verts_512`` matrix.
        self.emit_landmarks = bool(emit_landmarks)
        self.emit_person_mask = bool(emit_person_mask)
        self.person_mask_stride = (
            int(person_mask_stride) if person_mask_stride is not None else None
        )
        # MAMMA-style per-landmark contact GT (person-person via sdf_vertices + floor
        # via floor_contact_mask). Needs the verts_512 matrix, so it also loads it.
        self.emit_contact = bool(emit_contact)
        self.contact_threshold = float(contact_threshold)
        self.contact_missing_as_negative = bool(contact_missing_as_negative)
        self.landmark_visibility_threshold = float(landmark_visibility_threshold)
        self.patch_grid = int(self.img_size // self.patch_size)
        self._verts512 = None
        if self.emit_landmarks or self.emit_contact:
            from training.data.landmark_mask_gt import load_verts512_matrix
            self._verts512 = load_verts512_matrix(landmark_matrix_path)

        self.data_store = self._build_sequences()
        inferred_max_people = max(
            (view_annos[0]["num_people"] for view_annos in self.data_store.values()),
            default=0,
        )
        if self.max_num_people is None:
            self.max_num_people = inferred_max_people
        elif inferred_max_people > self.max_num_people:
            raise ValueError(
                f"max_num_people={self.max_num_people} is smaller than observed people count "
                f"{inferred_max_people}. Increase SysSMPLMultiDataset.max_num_people."
            )
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = sum(len(seq) for seq in self.data_store.values())

        if split == "train":
            self.len_train = self.sequence_list_len if len_train is None else min(len_train, self.sequence_list_len)
        elif split == "test":
            self.len_train = self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info("%s: SysSMPLMulti sequences: %d", status, self.sequence_list_len)
        logging.info("%s: SysSMPLMulti total views: %d", status, self.total_frame_num)
        logging.info("SysSMPLMulti max_num_people: %d", self.max_num_people)
        logging.info("SysSMPLMulti data_root: %s", self.data_root)

    @staticmethod
    def _load_pickle(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _parse_camera(intrinsics, extrinsics) -> Optional[Dict[str, np.ndarray]]:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)

        if intrinsics.ndim == 1 and intrinsics.size == 9:
            intrinsics = intrinsics.reshape(3, 3)
        elif intrinsics.shape != (3, 3):
            return None

        if extrinsics.ndim == 1 and extrinsics.size == 12:
            extrinsics = extrinsics.reshape(3, 4)
        elif extrinsics.shape == (4, 4):
            extrinsics = extrinsics[:3, :]
        elif extrinsics.shape != (3, 4):
            return None

        return {"intrinsics": intrinsics, "extrinsics": extrinsics}

    @staticmethod
    def _project_points_opencv_np(points_world: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        E = np.asarray(extrinsics, dtype=np.float64).reshape(3, 4)
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        cam = points @ E[:3, :3].T + E[:3, 3]
        z = cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        pix = (cam @ K.T)[:, :2] / z_safe[:, None]
        return pix.astype(np.float32)

    @staticmethod
    def _looks_like_raw_mamma_sequence(path: Path) -> bool:
        if not path.is_dir():
            return False
        for view_dir in path.iterdir():
            if view_dir.is_dir() and any(view_dir.glob("*.data.pyd")):
                return True
        return False

    @classmethod
    def _find_raw_mamma_sequences(cls, root: Path) -> List[Path]:
        root = Path(root)
        if cls._looks_like_raw_mamma_sequence(root):
            return [root]

        seq_dirs = []
        png_roots = []
        if (root / "png").is_dir():
            png_roots.append(root / "png")
        png_roots.extend(p for p in root.rglob("png") if p.is_dir())
        for png_root in sorted(set(png_roots)):
            for child in sorted(p for p in png_root.iterdir() if p.is_dir()):
                if cls._looks_like_raw_mamma_sequence(child):
                    seq_dirs.append(child)
            if cls._looks_like_raw_mamma_sequence(png_root):
                seq_dirs.append(png_root)
        if seq_dirs:
            return sorted(set(seq_dirs))

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if cls._looks_like_raw_mamma_sequence(current):
                seq_dirs.append(current)
                dirnames[:] = []
                continue
            if current.name != "png" and current.parent.name != "png":
                # Keep traversal broad enough for Mamma_mv_split/.../png/<seq>,
                # but avoid descending into image leaves once possible.
                pass
        return sorted(set(seq_dirs))

    @staticmethod
    def _group_raw_mamma_frames(seq_dir: Path) -> Dict[str, List[str]]:
        frame_to_views: Dict[str, List[str]] = {}
        for view_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            for data_path in view_dir.glob("*.data.pyd"):
                frame = data_path.name[: -len(".data.pyd")]
                if (view_dir / f"{frame}.jpg").is_file() or (view_dir / f"{frame}.png").is_file():
                    frame_to_views.setdefault(frame, []).append(view_dir.name)
        return {frame: sorted(views) for frame, views in frame_to_views.items()}

    @staticmethod
    def _raw_image_path(seq_dir: Path, view_name: str, frame: str) -> Optional[Path]:
        view_dir = seq_dir / view_name
        for suffix in (".jpg", ".png", ".jpeg"):
            image_path = view_dir / f"{frame}{suffix}"
            if image_path.is_file():
                return image_path
        return None

    def _decode_raw_mamma_joints_world(self, pose: np.ndarray, beta: np.ndarray, trans: np.ndarray, gender) -> np.ndarray:
        # Import lazily to keep the classic npz loader lightweight and avoid a
        # module-level dependency from data loading to the loss stack.
        import inspect
        import torch

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        from training.loss import _decode_smpl_batch, _normalize_gender_string

        pose_t = torch.as_tensor(np.asarray(pose, dtype=np.float32).reshape(1, -1)[:, :72])
        beta_t = torch.as_tensor(np.asarray(beta, dtype=np.float32).reshape(1, -1)[:, :10])
        trans_t = torch.as_tensor(np.asarray(trans, dtype=np.float32).reshape(1, 3))
        gender_key = _normalize_gender_string(gender)
        with torch.no_grad():
            joints, _ = _decode_smpl_batch(pose_t, beta_t, trans_t, [gender_key], use_mamma=True)
        return joints[0, :24].detach().cpu().numpy().astype(np.float32)

    def _build_sequences(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        raw_store = self._build_raw_mamma_sequences()
        if not raw_store:
            raise ValueError(
                f"No raw Mamma .data.pyd sequences found under {self.data_root} / {self.image_root}."
            )
        logging.info("SysSMPLMulti raw Mamma sequences/frames: %d", len(raw_store))
        return raw_store

    def _build_raw_mamma_sequences(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        roots = [self.data_root]
        if self.image_root != self.data_root:
            roots.append(self.image_root)

        seq_dirs = []
        for root in roots:
            if root.is_dir():
                seq_dirs.extend(self._find_raw_mamma_sequences(root))
        seq_dirs = sorted(set(seq_dirs))
        if self.max_sequences is not None:
            seq_dirs = seq_dirs[: int(self.max_sequences)]
        if not seq_dirs:
            return {}

        data_store: Dict[str, List[Dict[str, np.ndarray]]] = {}
        bad_pyds: List[str] = []
        total_seqs = len(seq_dirs)
        logging.info(
            "SysSMPLMulti: building index over %d sequences (serial, reads 1 pyd per frame/view)...",
            total_seqs,
        )
        for seq_i, seq_dir in enumerate(seq_dirs, 1):
            grouped = self._group_raw_mamma_frames(seq_dir)
            frame_items = sorted(grouped.items())
            if self.max_frames_per_sequence is not None:
                frame_items = frame_items[: int(self.max_frames_per_sequence)]
            for frame, views in frame_items:
                if len(views) < self.min_num_images:
                    continue

                try:
                    first_people = self._load_pickle(seq_dir / views[0] / f"{frame}.data.pyd")
                except Exception:  # corrupt/unreadable pyd -> skip this frame
                    bad_pyds.append(str(seq_dir / views[0] / f"{frame}.data.pyd"))
                    continue
                if not isinstance(first_people, dict) or not first_people:
                    continue
                person_ids = sorted(first_people.keys(), key=lambda item: int(item))
                people_params = {}
                for person_id in person_ids:
                    person = first_people[person_id]
                    people_params[person_id] = {
                        "person_key": f"person_{int(person_id):02d}",
                        "smpl_pose": np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)[:72],
                        "smpl_beta": np.asarray(person["shape"], dtype=np.float32).reshape(-1)[:10],
                        "smpl_trans": np.asarray(person["trans_world"], dtype=np.float32).reshape(-1)[:3],
                        "gender": person.get("gender", "neutral"),
                        # instance-mask value for this person == person_idx + 1.
                        "person_idx": int(person.get("person_idx", int(person_id))),
                    }

                view_annos = []
                for view_name in views:
                    data_path = seq_dir / view_name / f"{frame}.data.pyd"
                    image_path = self._raw_image_path(seq_dir, view_name, frame)
                    if image_path is None or not data_path.is_file():
                        continue
                    try:
                        frame_people = self._load_pickle(data_path)
                    except Exception:  # corrupt/unreadable pyd -> skip this view
                        bad_pyds.append(str(data_path))
                        continue
                    if not isinstance(frame_people, dict) or not frame_people:
                        continue
                    cam_person = next(iter(frame_people.values()))
                    camera = self._parse_camera(cam_person["cam_int"], cam_person["cam_ext"])
                    if camera is None:
                        continue

                    people_annos = [
                        dict(people_params[person_id])
                        for person_id in person_ids
                        if person_id in people_params and person_id in frame_people
                    ]
                    if not people_annos:
                        continue

                    mask_path = data_path.with_name(f"{frame}.mask.jpg")
                    view_annos.append(
                        {
                            "image_path": str(image_path),
                            "intrinsics": camera["intrinsics"],
                            "extrinsics": camera["extrinsics"],
                            "people": people_annos,
                            "num_people": len(people_annos),
                            "raw_mamma": True,
                            # for on-the-fly landmark / mask GT (loaded lazily in get_data)
                            "data_path": str(data_path),
                            "mask_path": str(mask_path) if mask_path.is_file() else None,
                        }
                    )

                if len(view_annos) >= self.min_num_images and len({a["num_people"] for a in view_annos}) == 1:
                    rel = seq_dir.name
                    try:
                        rel = str(seq_dir.relative_to(self.data_root))
                    except ValueError:
                        pass
                    data_store[f"raw_mamma_{rel}_frame_{frame}"] = view_annos

            if seq_i % 20 == 0 or seq_i == total_seqs:
                logging.info(
                    "SysSMPLMulti: built %d/%d sequences (%d frames so far)",
                    seq_i, total_seqs, len(data_store),
                )

        if bad_pyds:
            logging.warning(
                "SysSMPLMulti: skipped %d unreadable/corrupt pyd file(s) during build (e.g. %s)",
                len(bad_pyds), ", ".join(bad_pyds[:5]),
            )

        return data_store

    def _parse_gender_label(self, g_raw) -> int:
        if isinstance(g_raw, np.ndarray):
            g_raw = g_raw.reshape(-1)[0] if g_raw.size > 0 else "neutral"
        if isinstance(g_raw, np.generic):
            g_raw = g_raw.item()
        if isinstance(g_raw, (bytes, bytearray)):
            g_raw = g_raw.decode("utf-8", errors="ignore")
        if isinstance(g_raw, (list, tuple)) and len(g_raw) > 0:
            g_raw = g_raw[0]

        if isinstance(g_raw, str):
            text = g_raw.strip().lower()
            if text.startswith("m"):
                return 0
            if text.startswith("f"):
                return 1
            return 2

        try:
            value = int(g_raw)
            if value in (0, 1, 2):
                return value
        except Exception:
            pass

        return 2

    def _load_view(self, anno, emit_mask, emit_pyd):
        """Load a view's RGB image (+ instance mask / GT pyd when needed).

        Returns ``(image, mask, pyd)`` or ``(None, None, None)`` when the view is
        UNUSABLE: the image is corrupt/unreadable, OR (when ``emit_mask``) a present
        mask file is corrupt, OR (when ``emit_pyd``) the landmark/contact ``.data.pyd``
        is corrupt/unreadable. A legitimately absent mask (``mask_path`` is None) is
        fine and yields ``mask=None``. Validating the pyd HERE (and caching it) means
        the per-view landmark/contact read in get_data cannot crash: a bad pyd makes
        the caller swap in another view instead. Lets the loader ride out scattered
        bad-block corruption in the dataset (images, masks, AND pyds).
        """
        image = read_image_cv2(anno["image_path"])
        if image is None:
            return None, None, None
        mask = None
        if emit_mask and anno.get("mask_path"):
            mask = cv2.imread(anno["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:  # mask present but unreadable -> reject this view
                return None, None, None
        pyd = None
        if emit_pyd:
            try:
                pyd = self._load_pickle(anno["data_path"])
            except Exception:  # corrupt/unreadable GT pyd -> reject this view
                return None, None, None
            if not isinstance(pyd, dict) or not pyd:
                return None, None, None
        return image, mask, pyd

    def _select_good_views(self, metadata, order, n_target, emit_mask, emit_pyd):
        """Pick ``n_target`` views (in ``order``) whose image+mask+pyd load OK.

        Caches the decoded image/mask/pyd so the caller does not re-read them. Returns
        ``(ids, annos, images, masks, pyds)`` or ``None`` if fewer than ``n_target``
        usable views exist (signals the caller to resample a different sample).
        ``order`` must cover every view index so corrupt ones can be replaced.
        """
        chosen, annos, imgs, masks, pyds = [], [], [], [], []
        for vi in order:
            if len(annos) >= n_target:
                break
            img, msk, pyd = self._load_view(metadata[int(vi)], emit_mask, emit_pyd)
            if img is None:
                self._bad_view_reads = getattr(self, "_bad_view_reads", 0) + 1
                n = self._bad_view_reads
                if n in (1, 10, 100) or n % 500 == 0:
                    logging.warning(
                        "SysSMPLMulti: skipped %d corrupt image/mask/pyd view(s) so far "
                        "(e.g. %s); swapping in another view.",
                        n, metadata[int(vi)]["image_path"],
                    )
                continue
            chosen.append(int(vi)); annos.append(metadata[int(vi)])
            imgs.append(img); masks.append(msk); pyds.append(pyd)

        if len(annos) < n_target:
            if self.allow_duplicate_img and annos:
                good = list(zip(chosen, annos, imgs, masks, pyds))
                k = 0
                while len(annos) < n_target:
                    c, a, i, m, p = good[k % len(good)]; k += 1
                    chosen.append(c); annos.append(a); imgs.append(i); masks.append(m); pyds.append(p)
            else:
                return None
        return chosen, annos, imgs, masks, pyds

    def get_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[List[int]] = None,
        aspect_ratio: float = 1.0,
        _resample_depth: int = 0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]
        n_views = len(metadata)

        # Build the full ordered list of view indices to TRY (corrupt ones get
        # skipped and replaced by later entries), plus the target view count.
        if ids is None:
            if self.fixed_view_sampling:
                requested = img_per_seq or n_views
                if requested > n_views:
                    logging.warning(
                        "Requested %d views but only %d available in %s; clamping to available views.",
                        requested, n_views, seq_name,
                    )
                    requested = n_views
                n_target = requested
                view_order = list(range(n_views))                     # deterministic
            else:
                n_target = int(img_per_seq)
                view_order = list(np.random.permutation(n_views))     # random, full cover
        else:
            n_target = len(ids)
            seen = {int(i) for i in ids}
            view_order = [int(i) for i in ids] + [i for i in range(n_views) if i not in seen]

        _is_raw_sel = bool(metadata[0].get("raw_mamma", False))
        emit_mask_sel = bool(self.emit_person_mask and _is_raw_sel)
        # pyd is (re-)read for landmark/contact GT in the loop below; validate+cache
        # it here so a corrupt pyd swaps the view instead of crashing mid-loop.
        emit_pyd_sel = bool((self.emit_landmarks or self.emit_contact)
                            and self._verts512 is not None and _is_raw_sel)
        sel = self._select_good_views(metadata, view_order, n_target, emit_mask_sel, emit_pyd_sel)
        if sel is None:
            # Not enough readable views in this frame -> resample a different sample.
            if _resample_depth < 20 and self.sequence_list_len > 1:
                logging.warning(
                    "SysSMPLMulti: %s has < %d readable views; resampling another sample.",
                    seq_name, n_target,
                )
                new_idx = random.randint(0, self.sequence_list_len - 1)
                return self.get_data(
                    seq_index=new_idx, img_per_seq=img_per_seq,
                    aspect_ratio=aspect_ratio, _resample_depth=_resample_depth + 1,
                )
            raise RuntimeError(
                f"SysSMPLMulti: could not gather {n_target} readable views for {seq_name} "
                f"after {_resample_depth} resamples (dataset too corrupt?)."
            )
        ids, annos, view_images, view_masks, view_pyds = sel
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        person_count = metadata[0]["num_people"]
        padded_people = int(self.max_num_people)
        person_count = min(person_count, padded_people)
        person_anchor = metadata[0]["people"][:person_count]

        smpl_joints2d_list = []
        smpl_joints3d_world_list = []
        confidences = []

        # Dense-landmark / mask GT are only available for raw Mamma_mv_split
        # (the pyd ships vertices2d / vertex_visibility / *.mask.jpg).
        is_raw = bool(metadata[0].get("raw_mamma", False))
        emit_lmk = self.emit_landmarks and self._verts512 is not None and is_raw
        emit_mask = self.emit_person_mask and is_raw
        emit_ct = self.emit_contact and self._verts512 is not None and is_raw
        landmarks2d_list = []
        landmarks_vis_list = []
        person_mask_list = []
        contact_list = []
        floor_contact_list = []

        smpl_poses = np.zeros((padded_people, 72), dtype=np.float32)
        smpl_betas = np.zeros((padded_people, 10), dtype=np.float32)
        smpl_translations = np.zeros((padded_people, 3), dtype=np.float32)
        smpl_genders = np.full((padded_people,), 2, dtype=np.int64)
        has_smpl = np.zeros((padded_people,), dtype=np.float32)
        person_keys = []

        for person_idx, person in enumerate(person_anchor):
            smpl_poses[person_idx] = np.asarray(person["smpl_pose"], dtype=np.float32).reshape(-1)[:72]
            smpl_betas[person_idx] = np.asarray(person["smpl_beta"], dtype=np.float32).reshape(-1)[:10]
            if "smpl_trans" in person:
                smpl_translations[person_idx] = np.asarray(person["smpl_trans"], dtype=np.float32).reshape(-1)[:3]
            smpl_genders[person_idx] = self._parse_gender_label(person.get("gender", "neutral"))
            has_smpl[person_idx] = 1.0
            person_keys.append(person.get("person_key", f"person_{person_idx}"))

        for view_i, anno in enumerate(annos):
            image_path = anno["image_path"]
            image = view_images[view_i]   # pre-loaded & validated in _select_good_views
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            extri_opencv = np.copy(anno["extrinsics"])
            intri_opencv = np.copy(anno["intrinsics"])
            original_size = np.array(image.shape[:2])

            people = anno["people"][:person_count]
            joints3d_world = np.zeros((padded_people, 24, 3), dtype=np.float32)
            joints2d_orig = np.zeros((padded_people, 24, 2), dtype=np.float32)
            for person_idx, person in enumerate(people):
                # Raw Mamma_mv_split .data.pyd stores vertices3d/joints3d in camera
                # coords with a high-rank joints2d tensor, so recreate the training
                # joint targets from the SMPL-X world params (same convention as the
                # loss) and project them with this view's camera.
                joints_world = self._decode_raw_mamma_joints_world(
                    person["smpl_pose"],
                    person["smpl_beta"],
                    person["smpl_trans"],
                    person.get("gender", "neutral"),
                )
                joints3d_world[person_idx] = joints_world[:24]
                joints2d_orig[person_idx] = self._project_points_opencv_np(
                    joints_world[:24],
                    extri_opencv,
                    intri_opencv,
                )

            # --- optional dense-landmark GT (raw only): M @ vertices2d in ORIG
            # pixels, appended to the track so it gets the same crop/resize as
            # the joints; visibility from M @ vertex_visibility (per view). ---
            landmarks2d_orig = None
            landmarks_vis = None
            contact_gt = None
            floor_contact_gt = None
            if emit_lmk or emit_ct:
                view_pyd = view_pyds[view_i]   # pre-loaded & validated in _select_good_views
                pyd_ids = sorted(view_pyd.keys(), key=lambda x: int(x))
            if emit_lmk:
                landmarks2d_orig = np.zeros((padded_people, 512, 2), dtype=np.float32)
                landmarks_vis = np.zeros((padded_people, 512), dtype=np.float32)
                for person_idx in range(person_count):
                    p = view_pyd[pyd_ids[person_idx]]
                    v2d = np.asarray(p["vertices2d"], dtype=np.float32)      # (10475,2)
                    landmarks2d_orig[person_idx] = downsample_vertices(self._verts512, v2d)
                    vv = np.asarray(p["vertex_visibility"], dtype=np.float32).reshape(-1)
                    landmarks_vis[person_idx] = downsample_visibility(
                        self._verts512, vv[None, :],
                        threshold=self.landmark_visibility_threshold,
                    )[0]
            # --- optional MAMMA-style contact GT (raw only), per (view, person) ---
            #   floor_contact = down( floor_contact_mask )
            #   contact       = down( sdf_vertices < thresh ) * (1 - visible)
            # (MAMMA: a visible landmark is treated as NOT in contact.)
            if emit_ct:
                contact_gt = np.zeros((padded_people, 512), dtype=np.float32)
                floor_contact_gt = np.zeros((padded_people, 512), dtype=np.float32)
                for person_idx in range(person_count):
                    p = view_pyd[pyd_ids[person_idx]]
                    vis512 = downsample_visibility(
                        self._verts512,
                        np.asarray(p["vertex_visibility"], dtype=np.float32).reshape(1, -1),
                        threshold=self.landmark_visibility_threshold,
                    )[0]
                    # person-person contact via SDF. In this dataset sdf_vertices is
                    # only exported for SOME samples; when absent, mark the whole row
                    # invalid (-1) so the loss ignores it (floor contact is always present).
                    sdf = np.asarray(p.get("sdf_vertices", []), dtype=np.float32).reshape(-1)
                    if sdf.size == NUM_SMPLX_VERTS:
                        contact_v = (sdf < self.contact_threshold).astype(np.float32)
                        c512 = downsample_visibility(self._verts512, contact_v[None, :], threshold=0.5)[0]
                        contact_gt[person_idx] = c512 * (1.0 - vis512)
                    else:
                        # no SDF: MAMMA treats as no-contact (0, negatives); otherwise -1 = ignore.
                        contact_gt[person_idx] = 0.0 if self.contact_missing_as_negative else -1.0
                    floor_v = np.asarray(p.get("floor_contact_mask", []), dtype=np.float32).reshape(-1)
                    if floor_v.size == NUM_SMPLX_VERTS:
                        floor_contact_gt[person_idx] = downsample_visibility(
                            self._verts512, floor_v[None, :], threshold=0.5
                        )[0]
                    else:
                        floor_contact_gt[person_idx] = -1.0

            # --- optional per-person instance mask (raw only): transformed with
            # the image via extra_maps (nearest interp preserves labels). ---
            extra_maps = None
            if emit_mask and anno.get("mask_path"):
                instance_mask = view_masks[view_i]   # pre-loaded & validated (not None here)
                if instance_mask is not None:
                    extra_maps = {"person_mask": instance_mask.astype(np.float32)}

            n_joint_pts = padded_people * 24
            if landmarks2d_orig is not None:
                track_in = np.concatenate(
                    [joints2d_orig.reshape(-1, 2), landmarks2d_orig.reshape(-1, 2)], axis=0
                )
            else:
                track_in = joints2d_orig.reshape(-1, 2)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                track_new,
                confidence,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                track=track_in,
                filepath=image_path,
                extra_maps=extra_maps,
            )
            H_final, W_final = image.shape[:2]
            joints2d_new = track_new[:n_joint_pts].reshape(padded_people, 24, 2)
            if confidence is not None:
                joints_conf = confidence[:n_joint_pts].reshape(padded_people, 24)
            else:
                joints_conf = None

            if landmarks2d_orig is not None:
                lmk_px = track_new[n_joint_pts:].reshape(padded_people, 512, 2)
                # normalise to [0, 1] to match the head's sigmoid output convention.
                lmk_norm = np.empty_like(lmk_px)
                lmk_norm[..., 0] = lmk_px[..., 0] / W_final
                lmk_norm[..., 1] = lmk_px[..., 1] / H_final
                # gate visibility by in-frame after the geometric transform.
                if confidence is not None:
                    inframe = confidence[n_joint_pts:].reshape(padded_people, 512)
                    landmarks_vis = landmarks_vis * inframe
                landmarks2d_list.append(lmk_norm.astype(np.float32))
                landmarks_vis_list.append(landmarks_vis.astype(np.float32))

            if emit_ct:
                # contact GT is per-landmark (not a pixel coord) -> no crop/resize.
                contact_list.append(contact_gt.astype(np.float32))
                floor_contact_list.append(floor_contact_gt.astype(np.float32))

            if emit_mask:
                # Match the model's ACTUAL mask-head grid, which follows the
                # processed image size (the dynamic sampler may use a non-square
                # aspect ratio, so it is NOT always img_size//stride).
                #   stride = patch_size (legacy, default) -> the dot-product head's
                #     patch grid (37x37 for 518/14);
                #   stride = person_mask_stride (e.g. 2)  -> the DPT head's
                #     pixel-level grid (259x259 for 518).
                mask_stride = self.person_mask_stride or self.patch_size
                mask_h_final = H_final // mask_stride
                mask_w_final = W_final // mask_stride
                person_mask = np.zeros(
                    (padded_people, mask_h_final, mask_w_final), dtype=np.float32
                )
                if extra_maps is not None and extra_maps.get("person_mask") is not None:
                    mask_final = extra_maps["person_mask"]
                    for person_idx in range(person_count):
                        pv = int(people[person_idx].get("person_idx", person_idx)) + 1
                        person_mask[person_idx] = rasterize_person_patch_mask(
                            mask_final, pv, mask_h_final, mask_w_final
                        )
                person_mask_list.append(person_mask)

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

            smpl_joints3d_world_list.append(joints3d_world)
            smpl_joints2d_list.append(joints2d_new)
            confidences.append(joints_conf)

        smpl_joints2d = np.stack(smpl_joints2d_list, axis=0).astype(np.float32)
        smpl_joints3d_world = np.stack(smpl_joints3d_world_list, axis=0).astype(np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        batch = {
            "seq_name": "syssmpl_multi_" + seq_name,
            "ids": np.asarray(ids, dtype=np.int64),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "smpl_pose": smpl_poses,
            "smpl_beta": smpl_betas,
            "smpl_trans": smpl_translations,
            "smpl_joints2d": smpl_joints2d,
            "smpl_joints3d_world": smpl_joints3d_world,
            "smpl_gender": smpl_genders,
            "has_smpl": has_smpl,
            "num_people": np.asarray(person_count, dtype=np.int64),
            "person_keys": person_keys,
            "smpl_joints2d_confidence": confidences,
            "image_paths": image_paths,
        }

        # Dense-landmark GT: (S, P, 512, 2) normalised 2D + (S, P, 512) visibility.
        if landmarks2d_list:
            batch["smpl_landmarks2d"] = np.stack(landmarks2d_list, axis=0).astype(np.float32)
            batch["smpl_landmarks2d_visibility"] = np.stack(landmarks_vis_list, axis=0).astype(np.float32)
        # Per-person mask GT: (S, P, patch_grid, patch_grid) occupancy in [0,1].
        if person_mask_list:
            batch["person_mask"] = np.stack(person_mask_list, axis=0).astype(np.float32)
        # MAMMA-style contact GT: (S, P, 512) binary person-person + floor contact.
        if contact_list:
            batch["smpl_contact"] = np.stack(contact_list, axis=0).astype(np.float32)
            batch["smpl_floor_contact"] = np.stack(floor_contact_list, axis=0).astype(np.float32)

        return batch
