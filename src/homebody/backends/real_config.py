"""Explicit REAL connections and calibration; loading configuration has no hardware effects."""
import json
from dataclasses import dataclass, fields
from pathlib import Path

from homebody.primitives.observations import ObservationError


@dataclass(frozen=True)
class RealConfig:
    iface: str
    map_path: Path
    map_sha256: str
    map_frame: str
    alignment_path: Path
    camera_calibration: Path
    map_floor_height_m: float
    head_camera_serial: str
    stereo_calibration_sha256: str
    body_address: str = "127.0.0.1:29400"
    color_endpoint: str = "tcp://127.0.0.1:55566"
    depth_endpoint: str = "tcp://127.0.0.1:55563"
    depth_telemetry_endpoint: str = "tcp://127.0.0.1:55564"
    depth_shape: tuple[int, int] = (480, 848)
    hand_calibration: Path | None = None
    segmentation_root: Path | None = None
    segmentation_checkpoint: Path | None = None
    segmentation_checkpoint_sha256: str | None = None
    segmentation_config: str = "configs/samurai/sam2.1_hiera_t.yaml"
    segmentation_device: str = "cuda"
    camera_max_age_s: float = .5
    camera_pair_skew_s: float = .10
    state_pair_skew_s: float = .10
    sensor_wait_s: float = 2.5
    writer_lock_path: Path | None = None

    @classmethod
    def from_file(cls, path, root=None):
        path = Path(path).resolve()
        doc = json.loads(path.read_text())
        unknown = set(doc) - {f.name for f in fields(cls)}
        if unknown:
            raise ObservationError(f"Unknown REAL configuration keys: {sorted(unknown)}")
        base = Path(root).resolve() if root else path.parent
        for key in ("map_path", "alignment_path", "camera_calibration", "hand_calibration",
                    "segmentation_root", "segmentation_checkpoint", "writer_lock_path"):
            if doc.get(key) is not None:
                doc[key] = (base / doc[key]).resolve()
        if "depth_shape" in doc:
            doc["depth_shape"] = tuple(doc["depth_shape"])
        try:
            return cls(**doc)
        except (TypeError, ValueError) as exc:
            raise ObservationError(f"Invalid REAL configuration: {exc}") from exc

    def __post_init__(self):
        if not self.iface or not self.map_frame:
            raise ValueError("REAL requires an explicit interface and map frame")
        if not isinstance(self.head_camera_serial, str) or not self.head_camera_serial.isdigit():
            raise ValueError("head_camera_serial must be the selected measured camera's serial")
        if (not isinstance(self.stereo_calibration_sha256, str) or len(self.stereo_calibration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.stereo_calibration_sha256)):
            raise ValueError("stereo_calibration_sha256 must pin the producer's calibrated stereo identity")
        if len(self.map_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.map_sha256):
            raise ValueError("map_sha256 must be a lowercase SHA-256")
        import math
        if not isinstance(self.map_floor_height_m, (int, float)) or not math.isfinite(self.map_floor_height_m):
            raise ValueError("map_floor_height_m must be the finite measured floor height in map metres")
        if len(self.depth_shape) != 2 or any(type(v) is not int or v <= 0 for v in self.depth_shape):
            raise ValueError("depth_shape must be two positive integers")
        for name in ("camera_max_age_s", "camera_pair_skew_s", "state_pair_skew_s", "sensor_wait_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 < value <= 5:
                raise ValueError(f"{name} must be finite, positive and at most 5 s")

    def offline_checks(self, *, root=None):
        """Validate inputs without importing GPU packages, opening sockets or taking authority."""
        import importlib.util

        from homebody.backends.g1.task_motion import load_hand_calibration
        from homebody.primitives.cameras.tracking import verify_runtime
        from homebody.primitives.localization.map_binding import (
            Alignment,
            MapBinding,
            sha256,
        )

        checks = {}
        def run(name, check):
            try:
                value = check()
                checks[name] = {"ok": True, "detail": "validated locally"}
                return value
            except (OSError, ValueError, TypeError, KeyError) as exc:
                checks[name] = {"ok": False, "detail": str(exc)}
                return None

        binding = run("map", lambda: MapBinding.load(self.map_path, expected_hash=self.map_sha256,
                                                    frame_id=self.map_frame))
        if binding is not None:
            run("alignment", lambda: Alignment.load(self.alignment_path, binding))
        else:
            checks["alignment"] = {"ok": False, "detail": "map binding invalid"}
        def camera():
            from homebody.primitives.cameras.rgbd import validate_head_calibration
            return validate_head_calibration(self.camera_calibration, root=root)
        run("camera_calibration", camera)
        if self.hand_calibration is not None:
            run("hand_calibration", lambda: load_hand_calibration(self.hand_calibration))
        else:
            checks["hand_calibration"] = {"ok": True,
                "detail": "Dex3 position contact; live feedback and visual retention checked at runtime"}
        def runtime():
            if self.segmentation_root is None:
                raise ObservationError("Pinned SAMURAI runtime path is required")
            return verify_runtime(self.segmentation_root)
        run("segmentation_runtime", runtime)
        def checkpoint():
            if self.segmentation_checkpoint is None or not Path(self.segmentation_checkpoint).is_file():
                raise ObservationError("SAM2 checkpoint is absent")
            if not self.segmentation_checkpoint_sha256 or sha256(self.segmentation_checkpoint) != self.segmentation_checkpoint_sha256:
                raise ObservationError("SAM2 checkpoint hash differs or is not pinned")
            if Path(self.segmentation_checkpoint).stat().st_size < 1024:
                raise ObservationError("SAM2 checkpoint is empty or a pointer, not model weights")
        run("segmentation_checkpoint", checkpoint)
        for package in ("numpy", "scipy", "cv2", "zmq", "cyclonedds", "unitree_sdk2py",
                        "torch", "torchvision", "hydra", "iopath"):
            present = importlib.util.find_spec(package) is not None
            checks[f"dependency:{package}"] = {"ok": present, "detail": "installed" if present else "not installed"}
        checks["live_intrinsics"] = {"ok": None, "detail": "requires received HSCOLOR1 geometry; not checked offline"}
        checks["live_readiness"] = {"ok": None, "detail": "robot, sensor clock agreement, mounted calibration and authority not checked offline"}
        return checks
