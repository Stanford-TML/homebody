"""Measured target labels from pinned SAMURAI, with bounded per-target video memory.

Streaming preprocessing and click selection use camera observations only.
Source and license attribution are recorded in THIRD_PARTY_NOTICES.md and the
vendored SAMURAI closure.
"""
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from homebody.primitives.observations import ObservationError

KALMAN_FIELDS = ("kf_mean", "kf_covariance", "stable_frames", "frame_cnt", "history")
MEMORY_KEEP, IMAGES_KEEP = 128, 4
PREDICTOR = "++model._target_=homebody.primitives.cameras._sam2_predictor.StreamingPredictor"
CLOSURE_SHA256 = "4949af25e9436a91b31f35c7f460c137d7ae0fbb2d890881bef7090e088336ec"


def verify_runtime(root):
    root = Path(root).resolve()
    manifest = (root / "CLOSURE.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != CLOSURE_SHA256:
        raise ObservationError("SAMURAI closure manifest is not the pinned closure")
    doc = json.loads(manifest)
    if doc.get("upstream", {}).get("commit") != "76ba195984892b0d1e3db5d9c9f90bb62175680a":
        raise ObservationError("SAMURAI runtime is not the pinned deployment revision")
    for member in doc["members"]:
        relative = Path(member["path"]).relative_to("vendor/samurai")
        if ".." in relative.parts:
            raise ObservationError("SAMURAI closure contains an unsafe path")
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ObservationError(f"Pinned SAMURAI member changed: {relative}")
    expected = {Path(member["path"]).relative_to("vendor/samurai") for member in doc["members"]}
    actual = {p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}}
    if actual - expected:
        raise ObservationError("Unlisted Python/config files in pinned SAMURAI runtime")
    return root


class _Images:
    def __init__(self, first):
        self.frames, self.count = {0: first}, 1

    def append(self, tensor):
        self.frames[self.count] = tensor
        self.count += 1
        for key in tuple(self.frames):
            if key < self.count - IMAGES_KEEP:
                del self.frames[key]

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return self.count


class _Evicting(dict):
    def __setitem__(self, key, value):
        import torch
        names = ("best_iou_score", "object_score_logits", "kf_score")
        held = [(name, value[name]) for name in names
                if isinstance(value.get(name), torch.Tensor) and value[name].is_cuda]
        if held:
            flat = torch.cat([t.reshape(-1).float() for _, t in held]).cpu()
            value = dict(value)
            for (name, tensor), scalar in zip(held, flat.split([t.numel() for _, t in held])):
                value[name] = scalar.reshape(tensor.shape).to(tensor.dtype)
        super().__setitem__(key, value)

    def __missing__(self, key):
        import torch
        return {"best_iou_score": torch.zeros(()), "object_score_logits": torch.full((), -1e4),
                "kf_score": torch.zeros(())}


def choose_mask(masks, scores, pixel):
    """Select a coherent whole-object mask, with slab and area exclusion."""
    masks = np.asarray(masks, dtype=bool)
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    u, v = pixel
    covering = [i for i in range(len(masks)) if masks[i, v, u]]
    if not covering:
        raise ObservationError("Segmentation did not support the selected pixel")
    bounded = []
    for i in covering:
        parents = [areas[j] for j in covering if j != i and areas[j] >= .25
                   and np.count_nonzero(masks[i] & masks[j]) >= .9 * masks[i].sum()]
        slab = bool(parents and areas[i] / min(parents) >= 1 / 3)
        if areas[i] < .25 and not slab:
            bounded.append(i)
    if bounded:
        best = max(scores[i] for i in bounded)
        if best >= .9:
            bounded = [i for i in bounded if scores[i] >= best - .05]
        else:
            bounded = [i for i in bounded if scores[i] >= .3] or [i for i in bounded if scores[i] >= best - .05]
    selected = max(bounded, key=lambda i: areas[i]) if bounded else min(covering, key=lambda i: areas[i])
    return masks[selected]


class _Track:
    def __init__(self, owner, rgb, mask):
        import torch
        from PIL import Image
        self.temp = TemporaryDirectory(prefix="homebody-target-")
        self.frames, self.shape = 1, rgb.shape[:2]
        Image.fromarray(rgb).save(Path(self.temp.name) / "000000.jpg", quality=100, subsampling=0)
        with owner.precision(), torch.inference_mode():
            self.state = owner.model.init_state(video_path=self.temp.name, offload_video_to_cpu=True)
        self.state["images"] = _Images(self.state["images"][0])
        self.state["output_dict"]["non_cond_frame_outputs"] = _Evicting(self.state["output_dict"]["non_cond_frame_outputs"])
        self.kalman = {"kf_mean": None, "kf_covariance": None, "stable_frames": 0, "frame_cnt": 0, "history": {}}
        self.install(owner.model)
        with owner.precision(), torch.inference_mode():
            owner.model.add_new_mask(inference_state=self.state, frame_idx=0, obj_id=1,
                                     mask=torch.from_numpy(np.ascontiguousarray(mask)))
        # Initialize the seed's conditioning memory before the first streaming frame.
        with owner.precision(), torch.inference_mode():
            for _ in owner.model.propagate_in_video(self.state, start_frame_idx=0, max_frame_num_to_track=0):
                pass
        self.capture(owner.model)
        self.mask = mask
        self.stamp = None

    def install(self, model):
        for name in KALMAN_FIELDS:
            setattr(model, name, self.kalman[name])

    def capture(self, model):
        for name in KALMAN_FIELDS:
            self.kalman[name] = getattr(model, name)

    def update(self, owner, rgb, stamp):
        import torch
        if self.stamp == stamp:
            return self.mask
        if self.stamp is not None and stamp < self.stamp:
            raise ObservationError("Target track cannot propagate backwards in capture time")
        if rgb.shape[:2] != self.shape:
            raise ObservationError("Camera dimensions changed during target tracking")
        self.state["images"].append(owner.tensor(rgb))
        self.frames += 1
        self.state["num_frames"] = self.frames
        self.install(owner.model)
        index, mask = self.frames - 1, None
        with owner.precision(), torch.inference_mode():
            for idx, ids, logits in owner.model.propagate_in_video(
                    self.state, start_frame_idx=index, max_frame_num_to_track=1):
                if int(idx) == index:
                    mask = (logits[list(ids).index(1)] > 0).detach().cpu().numpy().squeeze()
        self.capture(owner.model)
        if mask is None:
            raise ObservationError("SAMURAI produced no current target observation")
        tables = [self.state["output_dict"]["non_cond_frame_outputs"]]
        tables += [d["non_cond_frame_outputs"] for key in ("output_dict_per_obj", "temp_output_dict_per_obj")
                   for d in self.state[key].values()]
        tables.append(self.state["frames_already_tracked"])
        tables.append(self.kalman["history"])
        for table in tables:
            for key in tuple(table):
                if key < index - MEMORY_KEEP:
                    del table[key]
        self.mask, self.stamp = np.asarray(mask, dtype=bool), stamp
        return self.mask

    def close(self):
        self.temp.cleanup()


class SAM2Targets:
    """Current visual identities only; tracking does not establish object semantics."""
    def __init__(self, model, image_predictor, *, device="cuda", max_targets=4):
        self.model, self.predictor, self.device = model, image_predictor, device
        self.max_targets = max_targets
        self.tracks = OrderedDict()
        self._next = 1
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self.ready = True
        self.warmup_result = None

    @classmethod
    def from_config(cls, config):
        if config.segmentation_root is None or config.segmentation_checkpoint is None:
            raise ObservationError("REAL requires an explicit pinned SAMURAI runtime and checkpoint")
        root = verify_runtime(config.segmentation_root)
        checkpoint = Path(config.segmentation_checkpoint)
        from homebody.primitives.localization.map_binding import sha256
        if not checkpoint.is_file():
            raise ObservationError("SAM2 checkpoint is missing")
        if not config.segmentation_checkpoint_sha256 or sha256(checkpoint) != config.segmentation_checkpoint_sha256:
            raise ObservationError("SAM2 checkpoint hash differs or is not pinned")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import sam2
        if not Path(sam2.__file__).resolve().is_relative_to(root):
            raise ObservationError("Another SAM2 runtime shadows the pinned SAMURAI package")
        import torch
        if config.segmentation_device == "cuda" and not torch.cuda.is_available():
            raise ObservationError("Configured CUDA segmentation is unavailable")
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2_video_predictor(
            config.segmentation_config, str(checkpoint), device=config.segmentation_device,
            hydra_overrides_extra=[PREDICTOR])
        result = cls(model, SAM2ImagePredictor(model), device=config.segmentation_device)
        result.warmup()
        return result

    def warmup(self):
        """Initialize inference kernels before any sensor/command connection is opened."""
        if self.tracks:
            raise ObservationError("Perception warmup is only permitted before real target selection")
        started = time.perf_counter()
        image = np.full((128, 128, 3), 180, np.uint8)
        image[35:80, 40:75] = [220, 30, 30]
        try:
            self.select(image, 1., (55, 55))
            for index in range(1, 4):
                self.labels(np.roll(image, index, axis=1), 1. + index * .1)
        finally:
            self.retain(set())
            self._next = 1
            self.predictor.reset_predictor()
            for name, value in (("kf_mean", None), ("kf_covariance", None),
                                ("stable_frames", 0), ("frame_cnt", 0), ("history", {})):
                setattr(self.model, name, value)
        self.warmup_result = {"version": 1, "synthetic_only": True,
                              "seconds": time.perf_counter() - started,
                              "remaining_tracks": len(self.tracks), "remaining_cache": len(self._cache)}

    def precision(self):
        import torch
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()

    def tensor(self, rgb):
        import torch
        import torch.nn.functional as F
        pixels = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        image = F.interpolate(pixels.permute(2, 0, 1).float()[None],
                              size=(self.model.image_size, self.model.image_size),
                              mode="bilinear", align_corners=False, antialias=True)[0] / 255.
        mean = torch.tensor((.485, .456, .406), device=self.device)[:, None, None]
        std = torch.tensor((.229, .224, .225), device=self.device)[:, None, None]
        return (image - mean) / std

    @staticmethod
    def _key(rgb, stamp):
        return (float(stamp), hashlib.blake2b(np.ascontiguousarray(rgb).tobytes(), digest_size=16).digest())

    def labels(self, rgb, stamp):
        key = self._key(rgb, stamp)
        with self._lock:
            if key in self._cache:
                return self._cache[key].copy()
            labels = np.zeros(rgb.shape[:2], dtype=np.int32)
            for label, track in self.tracks.items():
                mask = track.update(self, rgb, stamp)
                labels[mask] = label
            self._cache[key] = labels.copy()
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
            return labels

    def select(self, rgb, stamp, pixel, existing_labels=None):
        import torch
        key = self._key(rgb, stamp)
        u, v = pixel
        with self._lock:
            labels = (np.zeros(rgb.shape[:2], dtype=np.int32) if existing_labels is None else
                      np.array(existing_labels, dtype=np.int32, copy=True))
            if labels[v, u] > 0:
                return labels
            if len(self.tracks) >= self.max_targets:
                raise ObservationError("Target-tracking capacity reached; release unused target tracks before another pick")
            with self.precision(), torch.inference_mode():
                self.predictor.set_image(np.ascontiguousarray(rgb))
                masks, scores, _ = self.predictor.predict(point_coords=np.array([[u, v]], dtype=np.float32),
                                                         point_labels=np.ones(1, dtype=np.int32), multimask_output=True)
            mask = choose_mask(masks, scores, pixel)
            label, self._next = self._next, self._next + 1
            track = _Track(self, rgb, mask)
            track.stamp = stamp
            self.tracks[label] = track
            labels[mask] = label
            self._cache.clear()
            self._cache[key] = labels.copy()
            return labels

    def retain(self, labels):
        """Session releases obsolete tracks while keeping every carried target's identity."""
        with self._lock:
            for label in tuple(self.tracks):
                if label not in labels:
                    self.tracks.pop(label).close()
            self._cache.clear()

    def close(self):
        self.retain(set())
        self.ready = False
