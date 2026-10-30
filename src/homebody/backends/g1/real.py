"""The G1 driver over DDS: observations, the 250 Hz body control loop and the body-host client.

Construction only takes the sole-writer lease, and observation providers open only their read
transport; the control loop starts with the first validated body action. Poses are map-frame
metres and radians.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from homebody.backends.g1 import body_host, stall_watch
from homebody.backends.g1 import profile as g1
from homebody.backends.g1.assets import load_policy
from homebody.backends.g1.body_host import (
    LOGFILE,
    STATE_RECOVERY_S,
    STATE_STALE_S,
    BodyClient,
    HostPort,
    RemoteKernel,
)
from homebody.backends.g1.dex3 import Dex3Port
from homebody.backends.g1.kernel import _real_controller_kernel_type, body_stop_reason
from homebody.backends.g1.unitree import UnitreeG1Port
from homebody.backends.g1.wireless import UnitreeWirelessInput
from homebody.backends.g1.writer import SoleWriter, lease_path
from homebody.helpers.geometry import wrap
from homebody.primitives.navigation.contracts import HardwareUnavailable, Refused
from homebody.real2sim.exploration.settings import current

CONTROL_HZ = 250.0  # this driver's in-process body control loop rate
AMO_PERIOD_S = 0.02  # the walking policy's wall-clock period (50 Hz)
LOOP_RESYNC_S = 0.05     # a loop this far behind its schedule restarts it instead of racing
LOOP_RATE_WARN_HZ = 200.0
# IMU heading anchor: map yaw = IMU yaw + an offset learned while still.
IMU_YAW_ALIGN_STILL_S = 1.0      # commanded still this long before the offset is learned
IMU_YAW_ALIGN_TAU_S = 3.0        # the offset's time constant while aligning
# Pose plausibility: the largest LiDAR xy step accepted per read.
POSE_SWAY_M = 0.08
POSE_SPEED_SLACK = 1.3           # x the commanded planar travel since the last accepted read
POSE_DRIFT_M_S = 0.05            # what an uncommanded body may still move
POSE_ORBIT_M_PER_RAD = 0.25      # base travel per radian turned in place
POSE_COAST_M = 0.25              # the stopping tail after a planar burst
POSE_COAST_S = 1.5               # how long after the last planar twist the tail counts
POSE_HOLD_S = 0.6                # a rejected step persisting this long is accepted
TILT_TAU_S = 0.4  # low-pass time constant of the pelvis tilt, about a stride
IMU_YAW_SNAP_RAD = math.radians(25.0)  # a still disagreement this large snaps the offset
IMU_YAW_SNAP_READS = 4  # consecutive still reads before the offset snaps
# Sign check: turning, the IMU yaw and the map yaw move the same way.
IMU_YAW_SIGN_RAD = math.radians(10.0)
IMU_YAW_SIGN_READS = 4           # map-yaw steps against the turn, with none along it
IMU_YAW_SIGN_STEP_RAD = math.radians(0.5)
IMU_YAW_SIGN_WINDOW_S = 3.0


def _yaw_wxyz(q) -> float:
    """Yaw of a (w, x, y, z) quaternion about +z."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_z(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


# The MID-360 lever from the pelvis, from g1_29dof_with_hand.urdf (metres).
TORSO_IN_PELVIS_XYZ = np.array([-0.0039635, 0.0, 0.044])
LIDAR_IN_TORSO_XYZ = np.array([0.0002835, 0.00003, 0.41618])


def _lidar_in_pelvis(q_waist) -> np.ndarray:
    """The MID-360 origin in the pelvis frame for the waist angles (yaw, roll, pitch)."""
    yaw, roll, pitch = (float(v) for v in q_waist)
    r_torso = _rot_z(yaw) @ _rot_x(roll) @ _rot_y(pitch)
    return _rot_z(yaw) @ TORSO_IN_PELVIS_XYZ + r_torso @ LIDAR_IN_TORSO_XYZ


def _pelvis_xy(lidar_xy, pelvis_yaw: float, tilt, q_waist) -> np.ndarray:
    """The pelvis map xy: the LiDAR xy minus the lever rotated by the pelvis attitude."""
    roll, pitch = (float(v) for v in tilt)
    lever = _rot_z(pelvis_yaw) @ _rot_y(pitch) @ _rot_x(roll) @ _lidar_in_pelvis(q_waist)
    return np.asarray(lidar_xy, dtype=float) - lever[:2]


LOWER_DEADMAN_S = 0.2

# The localizer's categorical verdict, used by the staging gate.
ACCEPTED_LOCALIZATION_HEALTH = frozenset({"OK", "DEGENERATE"})
LOC_ODOM_AGE_MAX_S = 1.0  # older odometry is a frozen pose
LOC_STALL_WAIT_S = 2.5  # a frozen sample is re-polled this long before the pose is refused
LOC_STALL_POLL_S = 0.1
# The refusal while the walking policy is not deployed.
BODY_NOT_DEPLOYED = ("the body policy is not deployed — on the remote press Y (initial "
                     "pose), then A (deploy); the console's BODY line shows the state")
HEALTH_ONLY_QUALITY = 0.75  # the quality of a healthy sample without a finite residual


_CLOSE_STATE_INIT_LOCK = threading.Lock()


def _body_host_addr() -> str | None:
    """The body host address from the body settings, or None for the in-process kernel
    (`off`); empty follows a running host."""
    value = current().body.address.strip()
    if value.lower() in ("0", "off", "local"):
        return None
    return value or (body_host.DEFAULT_BIND if body_host.host_alive() else None)


def localization_frozen(status) -> str:
    """Why this bridge sample must not be trusted as a live pose, or "" when it may be."""
    age = getattr(status, "odom_age_s", None)
    if age is not None and age > LOC_ODOM_AGE_MAX_S:
        return (f"localization frozen: the bridge last heard odometry {age:.1f}s ago "
                f"(limit {LOC_ODOM_AGE_MAX_S:.1f}s) — the LiDAR chain has stalled; "
                "check the LiDAR link and restart the odometry container (hardware/g1/superodom/run_superodom_ws.sh)")
    return ""


class RealHardwareRequired(HardwareUnavailable):
    """A real-driver path needs the robot. The message names the missing piece."""


class RealBackend:
    """Physical G1 handle with lazy, fail-closed observation and action transports."""

    def __init__(
        self,
        *,
        iface: str,
        map_artifact: str | Path | None = None,
        side: str = "right",
        lock_path: str | Path | None = None,
        diagnostics_dir: str | Path | None = None,
    ):
        self._iface = str(iface).strip()
        self._map_artifact = map_artifact
        self._diagnostics_dir = diagnostics_dir
        self.side = side
        lease = lease_path(lock_path)
        if _body_host_addr() is not None:   # a body-host client takes the client lease
            lease = lease.with_name(lease.name + ".client")
        self.writer = SoleWriter(lease)  # refuses to start while the lease is taken
        self._epoch = 1
        self._term_stamp: dict[str, float] = {}
        self._control_lock = threading.RLock()
        self._control_tick = threading.Condition(self._control_lock)
        self._control_ticks = 0
        self._control_stop = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._port = self._kernel = None
        self._host = None   # the body host's client: the legs live outside this process
        self._last_command = None  # the kernel's last 50 Hz command, re-armed at 250 Hz
        self._gates = None
        self._loop_rate_hz = 0.0            # the control loop's measured ticks/s (250 nominal)
        self._loop_rate_said = 0.0
        self._imu_yaw: tuple[float, float] | None = None   # (pelvis yaw, monotonic) each tick
        self._tilt: tuple[float, float] | None = None       # (roll, pitch) low-passed
        self._tilt_at = 0.0
        self._yaw_disagree = 0                              # still reads far off the offset
        self._yaw_pairs: list[tuple[float, float, float]] = []   # (t, imu, lidar) while turning
        self._imu_yaw_mirrored = False
        self._q_waist: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._yaw_offset: float | None = None              # map yaw - IMU yaw, learned still
        self._yaw_offset_at = 0.0
        self._twist_zero_since: float | None = None        # when the commanded twist went to 0
        self._pose_held: tuple[np.ndarray, float] | None = None   # (last accepted xy, when)
        self._pose_reject_since: float | None = None       # when the current step was rejected
        self._twist_cmd = (0.0, 0.0, 0.0)                  # the last twist sent to the kernel
        self._twist_at = 0.0
        self._travel_m = 0.0                               # commanded planar travel since accept
        self._turn_rad = 0.0                               # commanded turning since accept
        self._twist_nonzero_at = 0.0                       # when a non-zero planar twist was sent
        self._wireless = None
        self._body_events: list[tuple[float, str]] = []
        self._body_was_active = False
        self._dex3 = None
        self._nav = None
        self._arm_target14: np.ndarray | None = None
        self._last_state = None
        self._control_error: BaseException | None = None
        self._body_stop_published = False
        self._body_stop_diagnostic: str | None = None
        self._close_condition = threading.Condition()
        self._close_phase = "open"
        self._close_errors: tuple[str, ...] = ()
        self._close_owner_thread_id: int | None = None
        self.trace_t: list[float] = []
        self.trace_q: list[np.ndarray] = []
        self._lower_last_command = 0.0
        self._lower_deadman_latched = True

    # -- meta ------------------------------------------------------------------
    def now(self) -> float:
        """The driver clock deadlines are measured against, not a robot state timestamp."""
        return time.monotonic()

    def epoch(self) -> int:
        return self._epoch

    def age_s(self, term: str) -> float:
        stamp = self._term_stamp.get(term)
        if stamp is None:
            raise RealHardwareRequired(f"age of '{term}': that feed has not produced a sample")
        return max(0.0, time.time() - stamp)

    def observation_source(self, term: str) -> str:
        return {
            "rgb": "camera",
            "camera": "calibration",
            "qpos": "joint_encoder",
            "gripper_state": "joint_proprioceptive_gap",
            "base_pose": "lidar_registration",
            "loc_quality": "registration_health",
            "arm_trace": "joint_encoder",
            "map": "live_lidar_map",
            "base_mobility": "controller_state",
            "pelvis_tilt": "imu_attitude",
        }[term]

    # -- observations (each names its missing hardware) -------------------------
    def lidar_obstacles(self):
        raise RealHardwareRequired("registered LiDAR obstacles require the exploration backend")


    def qpos(self):
        self._ensure_port()
        if self._host is not None and not self._host.fresh():
            self._host.interrupt("joint observation waiting for body state")
            # Wait for fresh state; the interrupted motion generation stays invalid.
            if not self._host.wait_fresh(STATE_RECOVERY_S):
                raise RealHardwareRequired("body state did not recover for joint observation")
        with self._control_lock:
            self._raise_control_error()
            state = self._port.read_state()
            self._last_state = state
            self._term_stamp["qpos"] = time.time()
            return np.asarray(state.joint_positions, dtype=np.float32)

    def joint_names(self):
        return g1.BODY_JOINTS  # the profile's order — no hardware needed

    def gripper_state(self):
        dex3 = self._ensure_dex3()
        try:
            state = dex3.state()
        except Exception as exc:
            raise RealHardwareRequired(f"gripper_state: {type(exc).__name__}: {exc}") from exc
        self._term_stamp["gripper_state"] = float(state.pop("stamp"))
        return state

    def base_pose(self):
        sample = self._navigation_status()
        if sample.pose is None:
            raise RealHardwareRequired("localization status has no x/y/yaw pose")
        frozen = localization_frozen(sample)
        give_up = time.monotonic() + LOC_STALL_WAIT_S
        while frozen and time.monotonic() < give_up:
            time.sleep(LOC_STALL_POLL_S)
            sample = self._navigation_status()
            frozen = localization_frozen(sample) if sample.pose is not None else frozen
        if frozen:
            raise RealHardwareRequired(frozen)
        # Stamp the pose at the odometry time.
        self._term_stamp["base_pose"] = (time.time() - sample.age()
                                         - float(getattr(sample, "odom_age_s", None) or 0.0))
        pose = np.asarray(sample.pose, dtype=np.float64)
        pose[:2] = self._plausible_xy(pose[:2])
        q_waist = self._q_waist
        pose[2] = self._anchored_yaw(float(pose[2]) - q_waist[0])   # the head yaw has the waist's
        if self._tilt is not None:
            pose[:2] = _pelvis_xy(pose[:2], float(pose[2]), self._tilt, q_waist)
        return pose

    def _lowpass_tilt(self, quat_wxyz) -> tuple[float, float]:
        """The pelvis roll and pitch, low-passed over TILT_TAU_S."""
        w, x, y, z = (float(v) for v in quat_wxyz)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        now = time.monotonic()
        prev = self._tilt
        if prev is None:
            self._tilt_at = now
            return (roll, pitch)
        a = 1.0 - math.exp(-max(0.0, now - self._tilt_at) / TILT_TAU_S)
        self._tilt_at = now
        return (prev[0] + a * (roll - prev[0]), prev[1] + a * (pitch - prev[1]))

    def _base_still_for(self) -> float:
        """Seconds the base has been commanded still: a zero twist, or no command past the
        dead-man."""
        now = time.monotonic()
        since = self._twist_zero_since
        if since is None and now - self._lower_last_command > LOWER_DEADMAN_S:
            since = self._lower_last_command + LOWER_DEADMAN_S
        return 0.0 if since is None else max(0.0, now - since)

    def _anchored_yaw(self, lidar_yaw: float) -> float:
        """The map heading carried by the pelvis IMU: `imu_yaw + offset`, the offset learned
        from the LiDAR yaw only while the base is still."""
        imu = self._imu_yaw
        if (self._imu_yaw_mirrored or imu is None
                or time.monotonic() - imu[1] > 0.5):
            return lidar_yaw
        imu_yaw, _ = imu
        measured = wrap(lidar_yaw - imu_yaw)
        now = time.monotonic()
        if self._sign_disagrees(now, imu_yaw, lidar_yaw):
            self._imu_yaw_mirrored = True
            print("[pose] the pelvis IMU yaw turns against the map yaw: heading anchor off for "
                  "this run, the LiDAR yaw is used as is", flush=True)
            return lidar_yaw
        if self._yaw_offset is None:
            self._yaw_offset, self._yaw_offset_at = measured, now
        elif self._base_still_for() >= IMU_YAW_ALIGN_STILL_S:
            off = wrap(measured - self._yaw_offset)
            self._yaw_disagree = self._yaw_disagree + 1 if abs(off) > IMU_YAW_SNAP_RAD else 0
            if self._yaw_disagree >= IMU_YAW_SNAP_READS:   # a re-pin, not drift: take it
                self._yaw_offset, self._yaw_disagree = measured, 0
            else:
                dt = max(0.0, now - self._yaw_offset_at)
                a = 1.0 - math.exp(-dt / IMU_YAW_ALIGN_TAU_S)
                self._yaw_offset = wrap(self._yaw_offset + a * off)
            self._yaw_offset_at = now
        else:
            self._yaw_offset_at = now
        return wrap(imu_yaw + self._yaw_offset)

    def _sign_disagrees(self, now: float, imu_yaw: float, lidar_yaw: float) -> bool:
        """Whether, over the last IMU_YAW_SIGN_WINDOW_S of commanded turning, the IMU yaw and
        the map yaw both moved more than IMU_YAW_SIGN_RAD — in opposite directions."""
        if abs(self._twist_cmd[2]) < 0.2:
            self._yaw_pairs = []
            return False
        pairs = [pr for pr in self._yaw_pairs if now - pr[0] <= IMU_YAW_SIGN_WINDOW_S]
        pairs.append((now, imu_yaw, lidar_yaw))
        self._yaw_pairs = pairs
        d_imu = wrap(imu_yaw - pairs[0][1])
        d_lidar = wrap(lidar_yaw - pairs[0][2])
        if not (abs(d_imu) > IMU_YAW_SIGN_RAD and abs(d_lidar) > IMU_YAW_SIGN_RAD
                and d_imu * d_lidar < 0.0):
            return False
        # A mirrored yaw fails every step; one re-registration fails one.
        wz = self._twist_cmd[2]
        steps = [wrap(b[2] - a[2]) for a, b in itertools.pairwise(pairs)]
        against = sum(1 for d in steps if d * wz < -IMU_YAW_SIGN_STEP_RAD)
        with_it = sum(1 for d in steps if d * wz > IMU_YAW_SIGN_STEP_RAD)
        return against >= IMU_YAW_SIGN_READS and with_it == 0

    def _pose_xy_accepted(self, xy: np.ndarray, now: float) -> bool:
        """Read-only acceptance predicate shared by control and the footprint display."""
        held = self._pose_held
        if held is None:
            return True
        last, at = held
        dt_cmd = max(0.0, min(0.1, now - self._twist_at))
        travel = self._travel_m + math.hypot(self._twist_cmd[0], self._twist_cmd[1]) * dt_cmd
        turned = self._turn_rad + abs(self._twist_cmd[2]) * dt_cmd
        coast = POSE_COAST_M if now - self._twist_nonzero_at < POSE_COAST_S else 0.0
        # Sway + commanded travel + turn orbit + stopping tail + drift.
        allowed = (POSE_SWAY_M + POSE_SPEED_SLACK * travel + POSE_ORBIT_M_PER_RAD * turned
                   + coast + POSE_DRIFT_M_S * max(0.0, now - at))
        since = self._pose_reject_since
        return float(np.hypot(*(xy - last))) <= allowed or (
                since is not None and now - since >= POSE_HOLD_S)

    def _plausible_xy(self, xy: np.ndarray) -> np.ndarray:
        """Keep the last accepted xy over implausible LiDAR steps until movement supports them
        or they persist."""
        now = time.monotonic()
        held = self._pose_held
        if held is None:
            self._pose_held, self._travel_m, self._turn_rad = (xy.copy(), now), 0.0, 0.0
            return xy
        if self._pose_xy_accepted(xy, now):
            self._pose_held, self._travel_m, self._turn_rad = (xy.copy(), now), 0.0, 0.0
            self._pose_reject_since = None
            return xy
        if self._pose_reject_since is None:
            self._pose_reject_since = now
        return held[0].copy()

    def loc_quality(self):
        """Localization quality in [0, 1] from the health and the registration residual; a
        healthy sample without a residual scores HEALTH_ONLY_QUALITY."""
        sample = self._navigation_status()
        if sample.health not in ACCEPTED_LOCALIZATION_HEALTH or localization_frozen(sample):
            quality = 0.0
        elif sample.localization_rmse_m is None:
            quality = HEALTH_ONLY_QUALITY
        else:
            quality = 1.0 / (1.0 + max(0.0, sample.localization_rmse_m))
        self._term_stamp["loc_quality"] = time.time() - sample.age()
        return quality

    def map_artifact(self):
        configured = str(self._map_artifact or "").strip()
        if not configured:
            raise RealHardwareRequired("map: exploration requires its live map owner")
        path = Path(configured)
        if not path.is_file():
            raise RealHardwareRequired(f"map: configured artifact does not exist: {path}")
        try:
            with np.load(path, allow_pickle=False) as item:
                raster = np.asarray(item["raster"], dtype=np.uint8)
                transform = np.asarray(item["T_map_px"], dtype=np.float64)
                occupancy = np.asarray(item["occupancy"], dtype=bool)
                center_allowed = (np.asarray(item["center_allowed"], dtype=bool)
                                  if "center_allowed" in item else None)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RealHardwareRequired(f"map: invalid artifact {path}: {exc}") from exc
        if (
            raster.ndim not in (2, 3)
            or transform.shape != (3, 3)
            or occupancy.shape != raster.shape[:2]
            or (center_allowed is not None and center_allowed.shape != occupancy.shape)
            or not np.isfinite(transform).all()
        ):
            raise RealHardwareRequired(
                f"map: incoherent raster/transform/occupancy shapes in {path}"
            )
        if raster.ndim == 2:
            raster = np.repeat(raster[:, :, None], 3, axis=2)
        self._term_stamp["map"] = time.time()
        return {
            "raster": raster,
            "T_map_px": transform,
            "occupancy": occupancy,
            "center_allowed": center_allowed,
            "scan_ref": str(path),
            "scan_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    # -- the body's lifecycle: the remote's Y and A ----------------------------
    def arm_body(self) -> None:
        """Open the kernel at the measured pose and start reading the remote; nothing moves
        until Y (initial pose) and A (deploy)."""
        self._ensure_control()

    def base_mobility(self) -> bool:
        """Whether a lower_body command would move the base: the walking policy deployed
        and past its ramp."""
        self._term_stamp["base_mobility"] = time.time()
        return self.body_deployed()

    def pelvis_tilt(self) -> tuple[float, float]:
        """(roll, pitch) of the pelvis from the IMU, low-passed over the stride; level until
        the first state arrives."""
        self._term_stamp["pelvis_tilt"] = time.time()
        tilt = self._tilt
        return (0.0, 0.0) if tilt is None else (float(tilt[0]), float(tilt[1]))

    def body_deployed(self) -> bool:
        kernel = self._kernel
        if kernel is None:
            return False
        status = kernel.status
        return bool(status.active and not status.initializing)

    def require_body_authority(self) -> None:
        """Raise HardwareUnavailable unless this process still drives a live body through the
        body host: fresh state, its driver claim, the robot link up."""
        host = self._host
        if host is None:
            return
        host.require_fresh()
        state = host.state or {}
        if state.get("driver") != host.claim or state.get("link_live") is False:
            raise HardwareUnavailable("body authority or robot link lost")

    def forget_pose_anchor(self) -> None:
        """Drop what the pose gate and the IMU-yaw anchor learned: the start frame changed."""
        self._pose_held = self._pose_reject_since = self._yaw_offset = None

    def body_link_diagnostics(self) -> dict | None:
        """The body-host link's counters for the console; None without a host."""
        host = self._host
        if host is None:
            return None
        age = None if not host.state_at else max(0, time.monotonic() - host.state_at)
        return {"state_age_s": age,
                "ticks": host.ticks, "interruptions": host.interruptions,
                "last_interruption": host.last_interruption,
                "callback_dropped": host.callback_dropped, "callback_error": host.callback_error,
                "control_error": str(self._control_error or "")}


    def _body_event(self, what: str) -> None:
        self._body_events.append((time.time(), what))
        # A closed stdout must not abort a body stop.
        with contextlib.suppress(BrokenPipeError, OSError, ValueError):
            print(f"[body] {time.strftime('%H:%M:%S')} {what}", flush=True)

    def _note_body_flip(self, status) -> None:
        """A policy that stopped on its own is recorded with the kernel's reason."""
        active = bool(status.active)
        if self._body_was_active and not active:
            self._body_event(body_stop_reason(status) or "policy stopped")
        self._body_was_active = active

    def _not_deployed(self) -> str:
        why = BODY_NOT_DEPLOYED
        if self._body_events:
            stamp, what = self._body_events[-1]
            why += f" (last: {what}, {time.time() - stamp:.0f} s ago)"
        return why

    def body_state(self) -> str:
        """One line for the console: what the body is doing and what the remote should do."""
        kernel = self._kernel
        if kernel is None:
            return "not armed — ARM opens the body in HOLD"
        status = kernel.status
        ticks = int(getattr(self._kernel.profile.loop, "initialization_ticks", 0) or 0)
        if status.initializing and int(status.initialization_tick) >= ticks:
            line = ("INITIAL POSE reached — legs and waist held there, ARMS LIVE for the "
                    "skills; A deploys the walking policy")
        elif status.initializing:
            done = int(status.initialization_tick)
            line = f"INITIAL POSE {done}/{ticks} (Y) — press A to deploy when it settles"
        elif status.active:
            walk = "walking" if status.walk_enabled else "standing"
            line = f"DEPLOYED — policy running, {walk} (B stops)"
        else:
            line = ("HOLD — legs and waist held, ARMS LIVE for the skills; Y (initial pose) "
                    "then A deploys the walking policy")
        reason = body_stop_reason(status)
        if reason and not status.active:
            line += f" · STOPPED: {reason}"
        if self._body_events:
            stamp, what = self._body_events[-1]
            line += f" · last: {what} {time.time() - stamp:.0f} s ago"
        if self._wireless is None:
            line += " · REMOTE NOT READ"
        if self._host is not None:
            msg = self._host.state or {}
            line += f" · body host {msg.get('loop_hz', 0):.0f} Hz ({msg.get('scheduling', '?')})"
        error = self._control_error
        if error is not None:
            line += f" · control error: {type(error).__name__}: {str(error)[:80]}"
        return line

    def _poll_remote(self) -> None:
        """One control tick's worth of the remote: rising-edge commands into the kernel."""
        if self._wireless is None:
            return
        from holosoma_inference.g1_control.runtime import ControllerCommand

        try:
            sample = self._wireless.poll(self._port)
        except Exception as exc:  # noqa: BLE001 -- the remote is a convenience; its failure is a note
            self._body_event(f"remote read failed: {exc}")
            self._wireless = None
            return
        for name in sample.commands:
            if name == "kill":
                self._body_event(f"KILL (L1+R1) — remote keys {getattr(sample, 'key', '?')!r}")
                raise RealHardwareRequired("operator pressed L1+R1 (kill) on the remote")
            self._kernel.handle_command(ControllerCommand(name))
            self._body_event(f"{name} ({sample.key})")

    # -- actions ----------------------------------------------------------------
    def apply_lower_body(self, cmd) -> None:
        if cmd is None:  # code-readiness probes must never command motors
            return
        self._ensure_control()
        from holosoma_inference.g1_control.contracts import VelocityCommand
        from holosoma_inference.g1_control.runtime import ControllerCommand

        with self._control_lock:
            tick = self._control_ticks
            self._raise_control_error()
            if not self.body_deployed():
                raise Refused(self._not_deployed())   # typed: the skill's code, not a crash
            if cmd.state is not None and self._kernel.status.walk_enabled != (cmd.state == "walk"):
                self._kernel.handle_command(ControllerCommand.toggle_walk)
            if cmd.vx is not None or cmd.vy is not None or cmd.wz is not None:
                self._kernel.set_velocity(
                    VelocityCommand(
                        linear_xy=(float(cmd.vx or 0.0), float(cmd.vy or 0.0)),
                        angular_z=float(cmd.wz or 0.0),
                    )
                )
                now_cmd = time.monotonic()
                planar = math.hypot(self._twist_cmd[0], self._twist_cmd[1])
                dt_cmd = max(0.0, min(0.1, now_cmd - self._twist_at))
                self._travel_m += planar * dt_cmd
                self._turn_rad += abs(self._twist_cmd[2]) * dt_cmd
                self._twist_cmd = (float(cmd.vx or 0.0), float(cmd.vy or 0.0), float(cmd.wz or 0.0))
                self._twist_at = now_cmd
                still = not (cmd.vx or cmd.vy or cmd.wz)
                if cmd.vx or cmd.vy:
                    self._twist_nonzero_at = now_cmd
                if not still:
                    self._twist_zero_since = None
                elif self._twist_zero_since is None:
                    self._twist_zero_since = time.monotonic()
            self._lower_last_command = time.monotonic()
            self._lower_deadman_latched = False
        self._await_control_tick(tick)

    def close(self) -> None:
        condition = self._close_lifecycle_condition()
        with condition:
            if self._close_phase == "closing":
                if self._close_owner_thread_id == threading.get_ident():
                    raise RealHardwareRequired("shutdown re-entered by its teardown owner")
                condition.wait_for(lambda: self._close_phase == "closed")
                errors = self._close_errors
                owner = False
            elif self._close_phase == "closed":
                errors = self._close_errors
                owner = False
            else:
                self._close_phase = "closing"
                self._close_owner_thread_id = threading.get_ident()
                errors = ()
                owner = True

        if owner:
            try:
                errors = self._close_once()
            except BaseException as exc:  # noqa: BLE001 -- no waiter left stranded
                errors = (f"shutdown coordinator: {type(exc).__name__}: {exc}",)
            with condition:
                self._close_errors = errors
                self._close_phase = "closed"
                self._close_owner_thread_id = None
                condition.notify_all()

        if errors:
            raise RealHardwareRequired("shutdown incomplete: " + "; ".join(errors))

    def _close_lifecycle_condition(self) -> threading.Condition:
        """Return the close coordinator, creating it for an object built without __init__."""

        # The module lock stops two first closers from each installing a condition.
        with _CLOSE_STATE_INIT_LOCK:
            condition = getattr(self, "_close_condition", None)
            if condition is None:
                condition = threading.Condition()
                self._close_condition = condition
            if not hasattr(self, "_close_phase"):
                self._close_phase = "open"
            if not hasattr(self, "_close_errors"):
                self._close_errors = ()
            if not hasattr(self, "_close_owner_thread_id"):
                self._close_owner_thread_id = None
            return condition

    def _close_once(self) -> tuple[str, ...]:
        """Perform the single teardown lifecycle whose result every caller observes."""

        errors = []
        live_stop = getattr(self, "_live_stop", None)
        if live_stop is not None:
            live_stop.set()

        def attempt(label, operation) -> None:
            try:
                operation()
            except BaseException as exc:  # noqa: BLE001 -- attempt every teardown step
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        nav = getattr(self, "_nav", None)
        if nav is not None:
            attempt("navigation STOP", lambda: nav.stop(timeout_s=2.0))

        control_stop = getattr(self, "_control_stop", None)
        if control_stop is not None:
            attempt("control stop latch", control_stop.set)
        control_thread = getattr(self, "_control_thread", None)
        if control_thread is not None and control_thread is not threading.current_thread():
            attempt("control thread join", lambda: control_thread.join(timeout=2.0))

        if getattr(self, "_kernel", None) is not None:
            # Stop from the last validated state, without another read.
            attempt("body STOP", self._publish_body_stop_once)
            diagnostic = getattr(self, "_body_stop_diagnostic", None)
            if diagnostic is not None:
                errors.append(f"body STOP callback: {diagnostic}")

        port = getattr(self, "_port", None)
        if port is not None:
            attempt("body port close", port.close)
        color = getattr(self, "_color", None)
        if color is not None:
            attempt("colour stream client close", color.close)
        dex3 = getattr(self, "_dex3", None)
        if dex3 is not None:
            attempt("Dex3 close", dex3.close)
        if nav is not None:
            attempt("navigation close", nav.close)
        writer = getattr(self, "writer", None)
        if writer is not None:
            attempt("writer release", writer.release)
        return tuple(errors)

    def _ensure_control(self) -> None:
        if self._kernel is not None:
            self._raise_control_error()
            return
        self._ensure_port()
        if self._host is not None:
            self._start_host_client()
            return
        iface = self._iface
        try:
            port = self._port
            state = self._last_state or port.read_state()
            policy = load_policy(g1.PROFILE)
            kernel = _real_controller_kernel_type()(g1.PROFILE, port, policy)
            arm = np.asarray(state.joint_positions[15:29], dtype=float)
            gates = {side: g1.ArmGate(side=side) for side in ("left", "right")}
            for side, gate in gates.items():
                gate.prime(np.asarray(state.joint_positions[g1.ARM_SLICE[side]], dtype=float))
                gate.arm(1.0, ramp_s=2.0)
            kernel.set_arm_targets(tuple(float(v) for v in arm))
            # The kernel keeps the measured pose until Y then A deploys the policy.
            self._wireless = UnitreeWirelessInput()
            self._body_event("armed: holding the measured pose")
        except Exception as exc:
            raise RealHardwareRequired(
                f"DDS/Holosoma control could not start on {iface}: {type(exc).__name__}: {exc}"
            ) from exc
        self._last_command = None
        self._port, self._kernel, self._gates = port, kernel, gates
        self._last_state, self._arm_target14 = state, arm
        self._settle_to_idle = False
        self._body_stop_published = False
        self._body_stop_diagnostic = None
        self._term_stamp["qpos"] = time.time()
        self._control_stop.clear()
        sys.setswitchinterval(0.0005)   # a 0.5 ms GIL slice for the 250 Hz loop
        self._control_thread = threading.Thread(
            target=self._control_loop, name="homebody-g1-control", daemon=False
        )
        self._control_thread.start()

    def _start_host_client(self) -> None:
        """Arm through the body host, which writes the motors and reads the remote; this
        process mirrors its state and sends intent; Dex3 keeps its own 250 Hz cadence."""
        client = self._host
        client.on_event = lambda text: self._body_event(f"host: {text}")
        client.send(cmd="arm")
        state = self._port.read_state()
        self._last_command = None
        self._kernel, self._gates, self._wireless = RemoteKernel(client), None, "body host"
        self._last_state = state
        self._arm_target14 = np.asarray(state.joint_positions[15:29], dtype=float)
        self._body_stop_published = False
        self._body_stop_diagnostic = None
        self._term_stamp["qpos"] = time.time()
        self._body_event("armed through the body host: the legs stay up whatever this run does")
        self._control_stop.clear()
        # The same 0.5 ms GIL slice as the in-process loop.
        sys.setswitchinterval(0.0005)
        self._control_thread = threading.Thread(
            target=self._host_loop, name="homebody-g1-host-client", daemon=True
        )
        self._control_thread.start()

    def _open_host_port(self, addr: str) -> None:
        client = BodyClient(addr)
        if not client.wait_fresh(3.0):
            client.closed.set()
            raise RealHardwareRequired(
                f"the body host is not answering on {addr}: start it with `python -m "
                f"homebody.backends.g1.body_host start` (hardware/g1/explore.sh does), log {LOGFILE}")
        port = HostPort(client)
        self._host, self._port, self._last_state = client, port, port.read_state()
        self._term_stamp["qpos"] = time.time()

    def _ensure_port(self) -> None:
        """Open state transport without loading policy or publishing a command."""
        if self._port is not None:
            return
        addr = _body_host_addr()
        if addr is not None:
            self._open_host_port(addr)
            return
        if body_host.host_alive():
            raise RealHardwareRequired(
                f"the body host (pid {body_host.running_pid()}) owns the G1's legs; "
                "body.address = off would open a second writer beside it")
        iface = self._iface
        if not iface:
            raise RealHardwareRequired(
                "DDS state: robot.iface is unset; select the single robot-LAN interface"
            )
        try:
            port = UnitreeG1Port(g1.PROFILE, iface)
            state = port.read_state()
        except Exception as exc:
            raise RealHardwareRequired(
                f"DDS state could not open on {iface}: {type(exc).__name__}: {exc}"
            ) from exc
        self._port, self._last_state = port, state
        self._term_stamp["qpos"] = time.time()

    def _lower_deadman(self) -> None:
        """No lower-body command for LOWER_DEADMAN_S: zero the twist (under the control lock)."""
        from holosoma_inference.g1_control.contracts import VelocityCommand

        if (self._lower_deadman_latched
                or time.monotonic() - self._lower_last_command <= LOWER_DEADMAN_S):
            return
        self._kernel.set_velocity(VelocityCommand())
        self._lower_deadman_latched = True
        # The zero counts as a command: the plausibility travel stops accruing.
        t_dead = self._lower_last_command + LOWER_DEADMAN_S
        dt_cmd = max(0.0, min(0.1, t_dead - self._twist_at))
        self._travel_m += math.hypot(*self._twist_cmd[:2]) * dt_cmd
        self._turn_rad += abs(self._twist_cmd[2]) * dt_cmd
        self._twist_cmd, self._twist_at = (0.0, 0.0, 0.0), t_dead
        if self._twist_zero_since is None:
            self._twist_zero_since = t_dead

    def motion_generation(self) -> int:
        """The body-link interruption count a dispatch checks its reference against; 0 without
        a host."""
        if self._host is None:
            return 0
        self._host.check_link()
        return self._host.interruptions

    def motion_interruption_reason(self) -> str:
        return self._host.last_interruption if self._host is not None else ""

    def _host_loop(self) -> None:
        """Mirror each new body state once and clock Dex3 independently in this thread."""
        client = self._host
        period = 1.0 / CONTROL_HZ
        seen = client.ticks
        last_tick = last_arm = time.monotonic()
        next_hand = last_tick + period
        watch = stall_watch.StallWatch(self._diagnostics_dir)
        try:
            watch.arm()
            while not self._control_stop.is_set():
                with client.tick:
                    timeout = (max(0.0, next_hand - time.monotonic())
                               if self._dex3 is not None else STATE_STALE_S)
                    client.tick.wait_for(lambda seen=seen: client.ticks != seen,
                                         timeout=timeout)
                    new_state = client.ticks != seen
                    seen = client.ticks
                if self._control_stop.is_set():
                    break
                tick_at = time.monotonic()
                hand_due = self._dex3 is not None and tick_at >= next_hand
                if hand_due:
                    # A delayed loop resumes from now; never replay missed hand writes.
                    next_hand = tick_at + period
                if tick_at - last_tick >= stall_watch.STALL_S:
                    watch.note_stall(tick_at - last_tick, "homebody-body-client-control")
                last_tick = tick_at
                if tick_at - last_arm >= stall_watch.ARM_WINDOW_S:
                    watch.arm()
                    last_arm = tick_at
                if not client.fresh():
                    client.interrupt("body-state stream interrupted")
                    if time.monotonic() - client.state_at > STATE_RECOVERY_S:
                        raise RealHardwareRequired(
                            f"the body host sent no fresh state for {STATE_RECOVERY_S:.1f} s "
                            f"({LOGFILE}); recovery timed out")
                    # Keep receiving; the next dispatch plans from fresh joints.
                    continue
                with self._control_lock:
                    self._lower_deadman()
                    if not new_state:
                        if hand_due and client.fresh():
                            next_hand = time.monotonic() + period
                            self._dex3.tick(period)
                        continue
                    try:
                        state = self._port.read_state()
                    except HardwareUnavailable:
                        # The stream expired while waiting for the lock: recover above.
                        if not client.fresh():
                            continue
                        raise
                    msg = client.state or {}
                    self._imu_yaw = (_yaw_wxyz(state.quaternion_wxyz), time.monotonic())
                    self._tilt = self._lowpass_tilt(state.quaternion_wxyz)
                    self._q_waist = tuple(float(v) for v in state.joint_positions[12:15])
                    self._note_body_flip(self._kernel.status)
                    self._loop_rate_hz = float(msg.get("loop_hz") or 0.0)
                    if hand_due:
                        next_hand = time.monotonic() + period
                        self._dex3.tick(period)
                    self._last_state = state
                    self._term_stamp["qpos"] = self._term_stamp["arm_trace"] = time.time()
                    self.trace_t.append(self.now())
                    self.trace_q.append(np.asarray(state.joint_positions, dtype=np.float32))
                    self._control_ticks += 1
                    self._control_tick.notify_all()
        except BaseException as exc:  # noqa: BLE001 -- latch the control loop error
            with self._control_tick:
                self._control_error = exc
                self._control_stop.set()
                self._epoch += 1
                self._control_tick.notify_all()
            client.send(cmd="stand")
        finally:
            watch.disarm()

    def _control_loop(self) -> None:
        period = 1.0 / CONTROL_HZ
        next_tick = time.monotonic()
        next_amo = next_tick
        self._loop_rate_hz = 0.0
        rate_at, rate_ticks = next_tick, 0
        last_tick = next_tick
        watch = stall_watch.StallWatch(self._diagnostics_dir)
        try:
            while not self._control_stop.is_set():
                # A tick this late dumps every thread's stack to the stall log.
                tick_at = time.monotonic()
                if tick_at - last_tick >= stall_watch.STALL_S:
                    watch.note_stall(tick_at - last_tick, "homebody-g1-control")
                last_tick = tick_at
                if self._control_ticks % stall_watch.ARM_EVERY_TICKS == 0:
                    watch.arm()
                with self._control_lock:
                    self._lower_deadman()
                    state = self._port.read_state()
                    self._imu_yaw = (_yaw_wxyz(state.quaternion_wxyz), time.monotonic())
                    self._tilt = self._lowpass_tilt(state.quaternion_wxyz)
                    self._q_waist = tuple(float(v) for v in state.joint_positions[12:15])
                    if self._kernel.ramping:
                        # The ramp owns the arms: the target follows the measured pose.
                        measured14 = np.asarray(state.joint_positions[15:29], dtype=float)
                        self._arm_target14 = measured14.copy()
                        self._settle_to_idle = True   # settle to the idle pose once it ends
                        for side, gate in self._gates.items():
                            gate.prime(measured14[g1.ARM_SLICE[side].start - 15:
                                                  g1.ARM_SLICE[side].stop - 15])
                        target = measured14
                        self._kernel.set_arm_gains(None, None)
                    else:
                        if getattr(self, "_settle_to_idle", False):
                            # A ramp cut short by A settles to the idle pose.
                            idle = np.asarray(g1.PROFILE.robot.idle_arm_positions, dtype=float)
                            step = g1.IDLE_SETTLE_RAD_S * period
                            self._arm_target14 = self._arm_target14 + np.clip(
                                idle - self._arm_target14, -step, step)
                            self._settle_to_idle = bool(
                                np.abs(idle - self._arm_target14).max() > 1e-6)
                        target = self._arm_target14.copy()
                        kp14, kd14 = np.zeros(14), np.zeros(14)
                        for side, gate in self._gates.items():
                            selected = g1.ARM_SLICE[side]
                            measured = np.asarray(state.joint_positions[selected], dtype=float)
                            shaped = gate(
                                target[selected.start - 15 : selected.stop - 15], measured,
                                period
                            )
                            target[selected.start - 15 : selected.stop - 15] = shaped["q_out"]
                            kp14[selected.start - 15 : selected.stop - 15] = shaped["kp"]
                            kd14[selected.start - 15 : selected.stop - 15] = shaped["kd"]
                    self._kernel.set_arm_targets(tuple(float(v) for v in target))
                    if not self._kernel.ramping:
                        self._kernel.set_arm_gains(kp14, kd14)
                    self._poll_remote()
                    now = time.monotonic()
                    if now >= next_amo:
                        self._last_command = self._kernel.tick().command
                        self._note_body_flip(self._kernel.status)
                        next_amo += AMO_PERIOD_S
                        if now - next_amo > LOOP_RESYNC_S:
                            next_amo = now + AMO_PERIOD_S
                    else:
                        self._kernel.republish_arms(self._last_command)
                    rate_ticks += 1
                    if now - rate_at >= 1.0:
                        self._loop_rate_hz = rate_ticks / (now - rate_at)
                        rate_at, rate_ticks = now, 0
                        slow = self._loop_rate_hz < LOOP_RATE_WARN_HZ
                        if slow and now - self._loop_rate_said > 30.0:
                            self._loop_rate_said = now
                            # a closed launcher pipe must never stop the body
                            with contextlib.suppress(OSError, ValueError):
                                print(f"[control] loop at {self._loop_rate_hz:.0f} ticks/s of "
                                      f"{CONTROL_HZ:.0f}; the gait keeps its 50 Hz on the wall "
                                      "clock", flush=True)
                    if self._dex3 is not None:
                        self._dex3.tick(period)
                    self._last_state = state
                    self._term_stamp["qpos"] = time.time()
                    # The trace's sample is this tick's row.
                    self._term_stamp["arm_trace"] = self._term_stamp["qpos"]
                    self.trace_t.append(self.now())
                    self.trace_q.append(np.asarray(state.joint_positions, dtype=np.float32))
                    self._control_ticks += 1
                    self._control_tick.notify_all()
                next_tick += period
                if time.monotonic() - next_tick > LOOP_RESYNC_S:
                    next_tick = time.monotonic()      # behind: do not race to catch up
                self._control_stop.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:  # noqa: BLE001 -- latch the control loop error
            with self._control_tick:
                # Latch the loop error first so no action resumes during cleanup.
                self._control_error = exc
                self._control_stop.set()
                try:
                    self._publish_body_stop_once()
                except BaseException:  # noqa: BLE001, S110 -- the loop error stays the latched one
                    pass
                self._epoch += 1
                self._control_tick.notify_all()
        finally:
            watch.disarm()        # a stopped loop is not a stalled one

    def _publish_body_stop_once(self):
        """Publish at most one successful stopped hold from the last valid G1 state."""

        with self._control_lock:
            if self._kernel is None or getattr(self, "_body_stop_published", False):
                return None
            if getattr(self, "_host", None) is not None:   # the host stands; homebody never stops it
                from holosoma_inference.g1_control.runtime import ControllerCommand

                self._kernel.handle_command(ControllerCommand.stop)
                self._body_stop_published = True
                return None
            from holosoma_inference.g1_control.contracts import G1State

            state = self._last_state
            if not isinstance(state, G1State):
                raise RealHardwareRequired(
                    "body STOP has no last validated finite G1State to hold"
                )
            try:
                self._last_command = None
                result = self._kernel.publish_stopped_hold(state)
            except BaseException:
                diagnostic = self._kernel.status.last_error
                if diagnostic is not None and diagnostic.startswith("policy on_stop failed:"):
                    self._body_stop_diagnostic = diagnostic
                raise
            self._body_stop_published = True
            diagnostic = result.status.last_error
            if diagnostic is not None and diagnostic.startswith("policy on_stop failed:"):
                self._body_stop_diagnostic = diagnostic
            return result

    def _raise_control_error(self) -> None:
        if self._control_error is not None:
            raise RealHardwareRequired(
                f"DDS/Holosoma control loop stopped: {type(self._control_error).__name__}: "
                f"{self._control_error}"
            ) from self._control_error

    def _await_control_tick(self, previous: int) -> None:
        with self._control_tick:
            if not self._control_tick.wait_for(
                lambda: self._control_ticks > previous or self._control_error is not None,
                timeout=0.25,
            ):
                raise RealHardwareRequired("DDS/Holosoma control missed its 250 ms tick deadline")
            self._raise_control_error()

    def _ensure_dex3(self):
        if self._dex3 is not None:
            return self._dex3
        iface = self._iface
        if not iface:
            raise RealHardwareRequired("Dex3 DDS: robot.iface is unset")
        try:
            self._dex3 = Dex3Port(iface)
        except Exception as exc:
            raise RealHardwareRequired(
                f"Dex3 DDS could not start on {iface}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._dex3

    def _navigation_status(self):
        raise RealHardwareRequired("localization status requires the exploration backend")

