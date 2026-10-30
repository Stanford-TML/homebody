"""REAL Robot composition: calibrated observations and the persistent G1 host's commands."""
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from homebody.primitives.actions import ActionLimits, ActionRejected
from homebody.primitives.localization.map_binding import MapBinding, sha256
from homebody.primitives.navigation.task import NavigationEvidence, depth_associated_labels
from homebody.primitives.observations import Frame, Measurement, ObservationError

from .real_config import RealConfig


@dataclass
class RealPorts:
    camera: object
    localization: object
    motion: object
    tracker: object | None


def torso_pose(pose, joints):
    """Registered upright LiDAR pose to torso and pelvis using the deployed URDF lever."""
    from homebody.backends.g1.real import LIDAR_IN_TORSO_XYZ, TORSO_IN_PELVIS_XYZ

    T_torso_lidar = np.eye(4)
    T_torso_lidar[:3, 3] = LIDAR_IN_TORSO_XYZ
    world_torso = pose.T_map_lidar @ np.linalg.inv(T_torso_lidar)
    yaw, roll, pitch = joints[12:15]
    T_pelvis_torso = np.eye(4)
    yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
    T_pelvis_torso[:3, :3] = (yaw_rotation @ Rotation.from_euler("x", roll).as_matrix()
                             @ Rotation.from_euler("y", pitch).as_matrix())
    T_pelvis_torso[:3, 3] = yaw_rotation @ TORSO_IN_PELVIS_XYZ
    world_pelvis = world_torso @ np.linalg.inv(T_pelvis_torso)
    base = np.array([*world_pelvis[:2, 3], np.arctan2(world_pelvis[1, 0], world_pelvis[0, 0])])
    return world_torso, base


def create_ports(config, binding, calibration, *, root=None):
    """Connect explicitly selected sensors and the existing host; start no services."""
    from homebody.primitives.cameras.color_stream import LatestColorClient
    from homebody.primitives.cameras.rgbd import CalibratedRGBD
    from homebody.primitives.cameras.tracking import SAM2Targets
    from homebody.primitives.cameras.verified_depth import connect_verified_depth
    from homebody.primitives.localization.mapped import SuperOdomMappedLocalization

    from .g1.task_motion import G1HostMotion

    opened = []
    try:
        tracker = SAM2Targets.from_config(config)
        opened.append(tracker)
        color = LatestColorClient(config.color_endpoint)
        opened.append(color)
        depth = connect_verified_depth(config)
        opened.append(depth)
        camera = CalibratedRGBD(color, depth, config.camera_calibration, root=root,
                               camera_serial=config.head_camera_serial,
                               stereo_calibration_sha256=config.stereo_calibration_sha256,
                               max_age_s=config.camera_max_age_s,
                               pair_skew_s=config.camera_pair_skew_s, wait_s=config.sensor_wait_s)
        localization = SuperOdomMappedLocalization(config.iface, binding, config.alignment_path,
                                                   wait_s=config.sensor_wait_s)
        opened.append(localization)
        motion = G1HostMotion(config, calibration)
        return RealPorts(camera, localization, motion, tracker)
    except BaseException:
        for port in reversed(opened):
            port.close()
        raise


class RealG1Backend:
    """Robot protocol over injected ports or an explicit live composition."""
    def __init__(self, root, settings, *, config: RealConfig, ports: RealPorts | None = None,
                 clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        self.root, self.settings, self.config = Path(root), settings, config
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.map_binding = MapBinding.load(config.map_path, expected_hash=config.map_sha256,
                                          frame_id=config.map_frame)
        self.calibration = json.loads((self.root / "assets/robot/calibration.json").read_text())
        self.limits = ActionLimits(settings, self.calibration)
        if ports is None:
            checks = config.offline_checks(root=self.root)
            failed = [f"{key}: {result['detail']}" for key, result in checks.items() if result["ok"] is False]
            if failed:
                raise ObservationError("REAL prerequisites unavailable: " + "; ".join(failed))
        self._alignment_hash = sha256(config.alignment_path) if Path(config.alignment_path).is_file() else None
        self.ports = ports or create_ports(config, self.map_binding, self.calibration, root=self.root)
        self._epoch = 1
        self._port_generation = self.ports.motion.generation
        self._closed = False
        self._sequence = 0
        self._frame_key = None
        self._frame = None
        self._T_torso_camera = None
        self._velocity_expiry = 0.

    @property
    def epoch(self):
        generation = self.ports.motion.generation
        if generation != self._port_generation:
            self._port_generation = generation
            self._epoch += 1
            self._frame = None
        return self._epoch

    @property
    def time(self):
        return float(self.clock())

    @property
    def provenance(self):
        return {"backend": "real", "map_sha256": self.map_binding.sha256,
                "map_frame": self.map_binding.frame_id, "map_path": str(self.map_binding.path),
                "camera_calibration_sha256": getattr(self.ports.camera, "calibration_hash", "unavailable"),
                "head_camera_serial": self.config.head_camera_serial,
                "stereo_calibration_sha256": self.config.stereo_calibration_sha256,
                "synchronization": "capture-associated; separate measured timestamps and bounded skew",
                "map_floor_height_m": self.config.map_floor_height_m,
                "alignment_sha256": self._alignment_hash,
                "segmentation_checkpoint_sha256": self.config.segmentation_checkpoint_sha256,
                "segmentation_warmup": getattr(self.ports.tracker, "warmup_result", None),
                "ground_truth": False}

    def readiness(self):
        hand_status = self.retention_status()
        return {"calibrated_rgbd": bool(getattr(self.ports.camera, "calibration_hash", None)),
                "registered_map": bool(self.map_binding.sha256),
                "target_tracking": bool(self.ports.tracker is not None and self.ports.tracker.ready),
                "hand_feedback": (set(hand_status["hands"]) == {"left", "right"} and
                                  not hand_status.get("command_owner", {}).get("fault")),
                "physical_qualification": False}

    def require_manipulation(self):
        missing = [key for key, value in self.readiness().items()
                   if key != "physical_qualification" and not value]
        if missing:
            raise ObservationError("REAL manipulation unavailable: " + ", ".join(missing))

    def retention_status(self):
        """Read hands only; an empty task ledger does not establish an empty real hand."""
        observed = {}
        command_owner = self._command_status()
        try:
            hands, timestamps = self.ports.motion.hand_states()
            for side in ("left", "right"):
                hand, stamp = hands[side], float(timestamps[side])
                closure = float(hand.closure)
                if not np.isfinite([stamp, closure]).all() or not 0 <= closure <= 1:
                    raise ObservationError(f"{side} hand measurement is invalid")
                age = self.time - stamp
                if not 0 <= age <= .1:
                    raise ObservationError(f"{side} hand measurement is stale or its clock is invalid")
                retained = bool(hand.retained(self.settings.grasp))
                contact = getattr(hand, "contact_state", None)
                observed[side] = {"closure": closure, "captured_at": stamp,
                                  "calibrated_retention": retained and contact is None,
                                  "contact_state": contact,
                                  "retention_source": ("dex3_position_contact" if contact is not None
                                                       else "opposing_effort_and_position"),
                                  "retained_or_unknown": closure > self.settings.placement.open_closure_limit or retained}
        except Exception as exc:  # noqa: BLE001 -- uncertain hand state must never permit writer shutdown
            return {"retained_or_unknown": True,
                    "reason": f"Hand state unavailable: {type(exc).__name__}: {exc}", "hands": observed,
                    "command_owner": command_owner}
        if command_owner.get("fault"):
            return {"retained_or_unknown": True, "reason": "Hand command owner faulted; retention is unconfirmed",
                    "hands": observed, "command_owner": command_owner}
        blocked = [side for side, state in observed.items() if state["retained_or_unknown"]]
        return {"retained_or_unknown": bool(blocked),
                "reason": ("Possible unrecorded load in " + ", ".join(blocked) + " hand(s)"
                           if blocked else "Both hands are measured open without retention evidence"),
                "hands": observed, "command_owner": command_owner}

    def _command_status(self):
        status = getattr(self.ports.motion, "command_status", None)
        return status() if status is not None else {}

    def _measurement(self, timestamp=None):
        pose = self.ports.localization.pose_at(timestamp, max_skew_s=self.config.state_pair_skew_s)
        body = self.ports.motion.state_at(timestamp, max_skew_s=self.config.state_pair_skew_s)
        hands, hand_times = self.ports.motion.hand_states(timestamp, max_skew_s=self.config.state_pair_skew_s)
        joints = np.asarray(body["q"], dtype=float)
        if joints.shape != (29,) or not np.isfinite(joints).all():
            raise ObservationError("Body telemetry must contain all 29 measured joints")
        if timestamp is not None and any(abs(t - timestamp) > self.config.state_pair_skew_s
                                         for t in hand_times.values()):
            raise ObservationError("RGB-D and hand evidence have incompatible capture times")
        world_torso, base = torso_pose(pose, joints)
        return pose, body, hands, hand_times, joints, world_torso, base

    def snapshot(self):
        deadline = self.monotonic() + self.config.sensor_wait_s
        for attempt in range(3):
            try:
                return self._snapshot()
            except ObservationError:
                if self._closed or attempt == 2 or self.monotonic() >= deadline:
                    raise
                # Reacquire asynchronous evidence; never relabel an expired packet.
                self.sleep(.01)

    def _snapshot(self):
        if self._closed:
            raise ObservationError("REAL backend is closed")
        capture = self.ports.camera.snapshot()
        pose, body, hands, hand_times, joints, world_torso, base = self._measurement(capture.captured_at)
        key = (self.epoch, capture.sequence)
        if self._frame_key == key and self._frame is not None:
            self._fresh_capture(min(self._frame.time, self._frame.sensor_metadata["depth_time"]))
            return self._frame
        points, obstacle_at = self.ports.localization.obstacles()
        navigation = self.map_binding.with_obstacles(points, floor_height_m=self.config.map_floor_height_m)
        self._T_torso_camera = capture.T_torso_camera
        labels = (np.zeros(capture.depth.shape, dtype=np.int32) if self.ports.tracker is None else
                  self.ports.tracker.labels(capture.rgb, capture.captured_at))
        self._fresh_capture(min(capture.color_time, capture.depth_time))
        if key[0] != self.epoch:
            raise ObservationError("Execution generation changed while preparing RGB-D evidence")
        self._sequence += 1
        metadata = {"schema_version": 1, "clock": "unix_seconds", "accepted_at": self.time,
                          "color_time": capture.color_time, "depth_time": capture.depth_time,
                          "depth_rays_from_world_camera": False,  # IR endpoints reprojected into RGB.
                          "camera_assembled_at": capture.assembled_at,
                          "camera_identity": capture.sensor_identity,
                          "pose_time": pose.captured_at, "registered_pose": True,
                          "navigation_generation": (self.epoch, pose.generation, self.map_binding.sha256),
                          "localization_health": "healthy when sampled",
                          "source_odom_stamp": pose.source_stamp, "body_time": body["t"],
                          "hand_times": hand_times, "hand_time_basis": "DDS receipt; bounded capture association",
                          "pose_time_basis": "registered stamp associated through fastest IMU receipt",
                          "obstacle_time": obstacle_at,
                          "map_frame": self.map_binding.frame_id,
                          "map_sha256": self.map_binding.sha256,
                          "calibration_hash": capture.calibration_hash,
                          "command_owner": self._command_status(),
                          "age_limits_s": {"color_time": self.config.camera_max_age_s,
                                           "depth_time": self.config.camera_max_age_s,
                                           "pose_time": .3, "body_time": .5,
                                           "obstacle_time": .5, "hand_times": .1}}
        frame = Frame(self.epoch, self._sequence, capture.captured_at, capture.rgb, capture.depth,
                      capture.intrinsics, world_torso @ capture.T_torso_camera, world_torso,
                      base, joints, hands, labels, navigation, sensor_metadata=metadata)
        self._frame_key, self._frame = key, frame
        return frame

    def _fresh_capture(self, captured_at):
        age = self.time - captured_at
        if not 0 <= age <= self.config.camera_max_age_s:
            raise ObservationError(f"RGB-D evidence expired during perception ({age:.3f} s; "
                                   f"limit {self.config.camera_max_age_s:.3f} s); holding for a new capture")

    def select_target(self, selected, current, point):
        self.require_manipulation()
        if selected.epoch != self.epoch or current.epoch != self.epoch:
            raise ObservationError("Target selection belongs to an expired execution generation")
        u, v = selected.normalized_pixel(point)
        first = self.ports.tracker.select(selected.rgb, selected.time, (u, v), selected.labels)
        selected = replace(selected, labels=first)
        # Click segmentation can take longer than a control freshness window.
        self._frame = None
        current = self.snapshot()
        return selected, current

    def retain_targets(self, labels):
        if self.ports.tracker is not None:
            self.ports.tracker.retain(set(labels))
        self._frame = None

    def measure(self):
        if self._T_torso_camera is None:
            self.snapshot()
        pose, body, hands, _, joints, world_torso, base = self._measurement()
        return Measurement(self.epoch, int(body["tick"]), min(pose.captured_at, body["t"]),
                           world_torso @ self._T_torso_camera, world_torso, base, joints, hands)

    def navigation_evidence(self):
        pose, body, _, _, _, _, base = self._measurement()
        points, obstacle_at = self.ports.localization.obstacles()
        navigation = self.map_binding.with_obstacles(points, floor_height_m=self.config.map_floor_height_m)
        acquisition, labels = None, None
        try:
            captured, old_body, hands, hand_times, joints, torso, old_base = self._measurement(obstacle_at)
        except ObservationError:
            captured = None
        if captured is not None:
            if captured.generation != pose.generation:
                raise ObservationError("Obstacle scan and current localization generations differ")
            stamps = [captured.captured_at, old_body["t"], *hand_times.values()]
            if all(abs(stamp - obstacle_at) <= 1e-6 for stamp in stamps) and self._T_torso_camera is not None:
                acquisition = Measurement(self.epoch, int(old_body["tick"]), min(stamps),
                    torso @ self._T_torso_camera, torso, old_base, joints, hands)
        if acquisition is not None and self._frame is not None and self.ports.tracker is not None:
            try:
                frame = self.snapshot()
            except ObservationError:
                frame = None
            if (frame is not None and frame.epoch == acquisition.epoch and
                    abs(frame.time - obstacle_at) <= 1e-6):
                labels = depth_associated_labels(points, frame, self.settings.collision.cloud_margin)
                acquisition = frame
        now = self.time
        return NavigationEvidence(self.epoch, now, pose.captured_at, obstacle_at, obstacle_at,
                                  base, navigation, points, pose.quality,
                                  self.map_binding.frame_id, self.map_binding.sha256,
                                  (self.epoch, pose.generation, self.map_binding.sha256),
                                  "SuperOdom registered pose/scan + measured map registration",
                                  True, bool(body.get("active") and not body.get("ramping")),
                                  static_navigation=self.map_binding.navigation,
                                  floor_height_m=self.config.map_floor_height_m,
                                  obstacle_state=acquisition, obstacle_labels=labels)

    def _check(self, epoch):
        self.limits.epoch(epoch, self.epoch)
        if self._closed:
            raise ActionRejected("REAL backend is closed")

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self._check(epoch)
        values = self.limits.velocity(forward, lateral, yaw_rate)
        self.ports.motion.velocity(values)
        self.ports.localization.note_twist(*values)
        self._velocity_expiry = self.monotonic() + min(.2, self.settings.physics.command_timeout)

    def arm_target(self, epoch, side, joints):
        self._check(epoch)
        self.ports.motion.arm(side, self.limits.arm(side, joints))

    def grip(self, epoch, side, closure):
        self._check(epoch)
        self.ports.motion.grip(side, self.limits.closure(side, closure))

    def advance(self, epoch, seconds):
        self._check(epoch)
        duration = self.limits.duration(seconds)
        deadline = self.monotonic() + duration
        while self.monotonic() < deadline:
            self._check(epoch)
            if self.monotonic() >= self._velocity_expiry:
                self.ports.motion.hold()
                self.ports.localization.note_twist(0, 0, 0)
                self._velocity_expiry = float("inf")
            self.sleep(min(.01, max(0, deadline - self.monotonic())))

    def stop(self, epoch):
        self.limits.epoch(epoch, self.epoch)
        self.ports.motion.hold()
        self.ports.localization.note_twist(0, 0, 0)
        self._velocity_expiry = 0.

    def reset(self):
        """Invalidate task commands without moving arms, opening hands or resetting SLAM."""
        self.stop(self.epoch)
        self._epoch += 1
        self._frame = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        failures = []
        for port in (self.ports.motion, self.ports.localization, self.ports.camera, self.ports.tracker):
            if port is not None:
                try:
                    port.close()
                except Exception as exc:  # noqa: BLE001 -- close every owned port; retain failures
                    failures.append(exc)
        if failures:
            raise RuntimeError("REAL ports did not close cleanly") from failures[0]
