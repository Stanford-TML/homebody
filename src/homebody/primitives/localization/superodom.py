"""SuperOdometry DDS ingestion, sample freshness admission and start-frame localization.

Poses are read in ws_map and served in the start frame (x forward at Zero, y left, metres).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from .lidar import (
    LIDAR_HEIGHT_M,
    RING_STOP_M,
    SelfLearner,
    drop_self_points,
    ring_ahead,
    ring_ahead_points,
    ring_min,
)

ORIGIN_LIKE_M, ORIGIN_LIKE_RAD = 0.5, 0.3


class SuperOdomLocalization:
    ODOM_GAP_RESET_S = 2.0
    ODOM_JUMP_RESET_M = 1.0  # a jump between two poses this large is a restart
    RING_FRESH_S = 1.5  # a ring distance older than this is stale
    DEAD_RECKON_MAX_S = 10.0  # the pose is carried on the commanded twist this long
    DEAD_RECKON_WALK = 1.0  # fraction of the commanded walk the dead reckoning integrates
    TWIST_ON_WIRE_S = 0.2  # the body host's twist deadman
    GYRO_FRESH_S = 0.3  # a state heading older than this is not the heading now
    GYRO_MATCH_S = 0.05  # a registered stamp is matched to a 250 Hz state sample this close

    def __init__(self, iface: str, *, start: bool = True):
        self.iface = iface
        self.lock = threading.Lock()
        self._sensor_stop = threading.Event()
        self._sensor_threads = []
        self._odom: deque = deque(maxlen=400)
        self._clock = time.time
        self._cmd = (0.0, 0.0, 0.0, 0.0)
        self._dr = [0.0, 0.0, 0.0]
        self._dr_t = 0.0
        self._state: deque = deque(maxlen=1000)
        self._dr_gyro0 = None
        self._imu_reference = None
        self._sensor_offset = None
        self._health = (False, 0.0)
        self._healthy_since = None
        self._freshness = {kind: {"received": 0, "accepted": 0, "rejected": 0,
                                 "received_at": None, "accepted_at": None, "stamp": None,
                                 "last_excess_age_s": None, "last_rejection": ""}
                           for kind in ("odom", "scan")}
        self._scans: deque = deque(maxlen=20)
        self._last_points = np.zeros((0, 3))
        self._registered_scan = (None, None)
        self.ring_hits = np.zeros((0, 3))
        self.ring_m, self.ring_t = float("inf"), 0.0
        self._zero = None
        self.frame_reset, self.frame_reset_why = False, ""
        self.scan_times: deque = deque(maxlen=50)
        self.raw_scan_times: deque = deque(maxlen=50)
        self._pump_errors = 0
        self._ingestion = {"stages": {}, "pumps": {}, "map_queue_max": 0,
                           "map_queue_dropped": 0, "health_false_received": 0,
                           "health_read_gaps": 0, "health_read_gap_max_s": 0.0}
        self.self_learner = SelfLearner()
        if start:
            self._start_dds()

    def _start_dds(self) -> None:
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        from homebody.primitives.localization.dds import initialize

        from .ros2_types import Bool, Imu, Odometry, PointCloud2

        initialize(self.iface)  # one process-wide domain on the robot NIC, shared with RealBackend
        dp = DomainParticipant(0)
        be = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(10))
        # 5 s of odometry queued
        odom_qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(50))
        odom_topic = Topic(dp, "rt/ws/laser_odometry_aligned", Odometry)
        self._rodom = DataReader(dp, odom_topic, qos=odom_qos)
        # the engine's IMU-propagated state at 250 Hz: only its heading is read
        state_qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(250))
        state_topic = Topic(dp, "rt/ws/state_estimation_aligned", Odometry)
        self._rstate = DataReader(dp, state_topic, qos=state_qos)
        scan_topic = Topic(dp, "rt/ws/registered_scan_aligned", PointCloud2)
        # keep only the newest cloud in DDS
        scan_qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self._rscan = DataReader(dp, scan_topic, qos=scan_qos)
        self._rimu = DataReader(dp, Topic(dp, "rt/utlidar/imu_livox_mid360", Imu), qos=state_qos)
        self._rhealth = DataReader(dp, Topic(dp, "rt/ws/state_estimation_health", Bool), qos=be)
        for target, name in ((self._pump, "explore-state"), (self._pump_scans, "explore-cloud")):
            worker = threading.Thread(target=target, name=name, daemon=True)
            self._sensor_threads.append(worker)
            worker.start()

    def _pump(self) -> None:
        self._run_pump("state", self._pump_sensors)

    def _pump_scans(self) -> None:
        self._run_pump("cloud", self._pump_scan)

    def _run_pump(self, name, step) -> None:
        previous = None
        stop_event = getattr(self, "_sensor_stop", None)
        while stop_event is None or not stop_event.is_set():
            started = time.perf_counter()
            with self.lock:
                stats = self._ingestion["pumps"].setdefault(name, {"cycles": 0, "max_gap_s": 0.})
                gap = 0. if previous is None else started - previous
                stats.update(cycles=stats["cycles"] + 1, last_at=self._clock(),
                             last_gap_s=gap, max_gap_s=max(stats["max_gap_s"], gap))
            previous = started
            try:
                self._measure_ingestion(name, step)
            except Exception as exc:  # noqa: BLE001 -- retain the reader after a bad sample
                self._pump_dropped(name, exc)
            time.sleep(0.01)

    def _measure_ingestion(self, name, operation, *args, **kwargs):
        started, cpu = time.perf_counter(), time.thread_time()
        try:
            return operation(*args, **kwargs)
        finally:
            wall, cpu = time.perf_counter() - started, time.thread_time() - cpu
            with self.lock:
                stats = self._ingestion["stages"].setdefault(
                    name, {"count": 0, "max_wall_s": 0., "max_thread_cpu_s": 0.})
                stats.update(count=stats["count"] + 1, last_wall_s=wall,
                             last_thread_cpu_s=cpu, max_wall_s=max(stats["max_wall_s"], wall),
                             max_thread_cpu_s=max(stats["max_thread_cpu_s"], cpu))

    def _take_batch(self, kind, reader, count):
        samples = self._measure_ingestion(f"{kind}_take", reader.take, N=count)
        with self.lock:
            stats = self._ingestion["stages"][f"{kind}_take"]
            stats["last_batch"] = len(samples)
            stats["max_batch"] = max(stats.get("max_batch", 0), len(samples))
        return samples

    def _pump_sensors(self) -> None:
        for s in self._take_batch("imu", self._rimu, 250):
            self._take_imu(s.header.stamp.sec + s.header.stamp.nanosec * 1e-9, self._clock())
        for s in self._take_batch("health", self._rhealth, 10):
            self._take_health(s.data, self._clock())
        self._pump_state()
        self._pump_odometry(np.diag([1.0, -1.0, -1.0, 1.0]))

    def _pump_scan(self) -> None:
        for s in self._take_batch("scan", self._rscan, 1):
            received_at = self._clock()  # application take completed, before XYZ/filter work
            stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
            self.raw_scan_times.append(time.time())  # application takes, not native publish rate
            points = self._measure_ingestion("cloud_xyz", _cloud_xyz, s)
            self._measure_ingestion("scan_filter", self._on_scan, points, stamp,
                                    received_at=received_at)

    def _pump_dropped(self, what: str, exc: BaseException) -> None:
        with self.lock:
            self._pump_errors += 1
            errors = self._pump_errors
        if errors in (1, 10, 100) or errors % 1000 == 0:
            print(f"[source] {what} dropped ({errors}): {exc!r}", flush=True)

    def _pump_state(self) -> None:
        for s in self._take_batch("state", self._rstate, 250):
            q = s.pose.pose.orientation
            self._take_state(
                s.header.stamp.sec + s.header.stamp.nanosec * 1e-9,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                self._clock(),
            )

    def _take_state(self, stamp: float, yaw: float, now: float) -> None:
        """One IMU-propagated heading (ws_map): the same yaw convention as the registered pose's
        x axis (Rx(180) leaves that axis alone), so the two subtract."""
        with self.lock:
            if not math.isfinite(yaw) or self._sensor_age_s(stamp, now) > 0.5:
                return
            if self._state and stamp <= self._state[-1][0]:
                return
            self._state.append((stamp, now, yaw))

    def _take_health(self, healthy: bool, now: float) -> None:
        with self.lock:
            gap = now - self._health[1] if self._health[1] else 0.
            self._ingestion["health_false_received"] += int(not healthy)
            self._ingestion["health_read_gaps"] += int(gap > 0.5)
            self._ingestion["health_read_gap_max_s"] = max(
                self._ingestion["health_read_gap_max_s"], gap)
            if not healthy or gap > 0.5:
                self._healthy_since = None
            if healthy and self._healthy_since is None:
                self._healthy_since = now
            self._health = (healthy, now)

    def health_stable_s(self) -> float:
        """Seconds the engine's health has been continuously true."""
        now = self._clock()
        with self.lock:
            health, since = self._health, self._healthy_since
        if not health[0] or now - health[1] > 0.5 or since is None:
            return 0.
        return max(0., now - since)

    def _take_imu(self, stamp: float, now: float) -> None:
        if not math.isfinite(stamp) or not math.isfinite(now):
            return
        with self.lock:
            if self._imu_reference is not None and stamp <= self._imu_reference[0]:
                if stamp < self._imu_reference[0] - 0.5:
                    self.frame_reset, self.frame_reset_why = True, "raw IMU clock ran backwards"
                return
            self._imu_reference = (stamp, now)
            offset = now - stamp
            self._sensor_offset = offset if self._sensor_offset is None else min(
                self._sensor_offset, offset)

    def _sensor_age_s(self, stamp: float, now: float) -> float:
        """A stamp's age in excess of the fastest raw IMU delivery."""
        offset, reference = self._sensor_offset, self._imu_reference
        if (offset is None or reference is None
                or now - reference[1] > 0.5 or not math.isfinite(stamp)):
            return math.inf
        age = now - stamp - offset
        return math.inf if age < -0.1 else max(0.0, age)

    def _state_yaw_near(self, stamp: float):
        """The state stream's yaw within GYRO_MATCH_S of `stamp`, or None (pump thread only)."""
        best = min(self._state, key=lambda s: abs(s[0] - stamp), default=None)
        return None if best is None or abs(best[0] - stamp) > self.GYRO_MATCH_S else best[2]

    def _gyro_turned(self, now: float):
        """The heading turned since the last registered pose by the gyro's account, or None when
        the state stream is silent or never matched that pose (the commanded wz then counts)."""
        if self._dr_gyro0 is None or not self._state:
            return None
        _, recv, yaw = self._state[-1]
        if now - recv > self.GYRO_FRESH_S:
            return None
        return (yaw - self._dr_gyro0 + math.pi) % (2 * math.pi) - math.pi

    def _pump_odometry(self, rx180: np.ndarray) -> None:
        for s in self._take_batch("odom", self._rodom, 50):
            p, q = s.pose.pose.position, s.pose.pose.orientation
            T = np.eye(4)
            T[:3, :3] = _quat_to_R(q.w, q.x, q.y, q.z)
            T[:3, 3] = (p.x, p.y, p.z)
            stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
            self._take_odometry(stamp, T @ rx180, self._clock())

    def _take_odometry(self, stamp: float, T: np.ndarray, now: float) -> None:
        """One registered pose (ws_map, upright): the restart verdict, then it supersedes the dead
        reckoning."""
        with self.lock:
            self._freshness_received("odom", stamp, now)
            if not np.isfinite(T).all() or self._sensor_age_s(stamp, now) > 0.5:
                self._freshness_admission("odom", "invalid_pose" if not np.isfinite(T).all()
                                          else "sensor_delayed_or_unavailable")
                return
            if self._odom and stamp <= self._odom[-1][0]:
                self.frame_reset = True
                self.frame_reset_why = "registered pose clock ran backwards"
                self._freshness_admission("odom", "clock_reversed")
                return
            if self._odom:
                stamp_last, T_last, _ = self._odom[-1]
                origin_like = (
                    float(np.hypot(T[0, 3], T[1, 3])) < ORIGIN_LIKE_M
                    and abs(math.atan2(T[1, 0], T[0, 0])) < ORIGIN_LIKE_RAD
                )
                was_away = float(np.hypot(T_last[0, 3], T_last[1, 3])) >= ORIGIN_LIKE_M
                # judge the jump against the dead-reckoned position
                self._advance_dr(now)
                expected = T_last[:2, 3] + np.array(self._dr[:2])
                why = reset_verdict(
                    stamp - stamp_last,
                    float(np.hypot(*(T[:2, 3] - expected))),
                    origin_like and was_away,
                )
                if why and not self.frame_reset:
                    self.frame_reset, self.frame_reset_why = True, why
            self._odom.append((stamp, T, now))
            self._freshness_admission("odom", accepted_at=now)
            self._dr, self._dr_t = [0.0, 0.0, 0.0], now
            self._dr_gyro0 = self._state_yaw_near(stamp)

    def note_twist(self, vx: float, vy: float, wz: float) -> None:
        """What the base was just told (every 250 Hz command, zeros included): the dead reckoning
        between registered poses walks exactly this."""
        now = self._clock()
        with self.lock:
            self._advance_dr(now)
            self._cmd = (vx, vy, wz, now)

    def _advance_dr(self, now: float) -> None:
        """Integrate the twist on the wire up to `now` (under the lock) from the last registered
        pose as a unicycle in ws_map; the heading is the gyro's when live, else the commanded wz."""
        vx, vy, wz, at = self._cmd
        dt = max(0.0, min(now, at + self.TWIST_ON_WIRE_S) - self._dr_t)
        turned = self._gyro_turned(now)
        if turned is None:
            turned = self._dr[2] + wz * dt
        if dt > 0.0 and self._odom and (vx or vy):
            T = self._odom[-1][1]
            yaw = math.atan2(T[1, 0], T[0, 0]) + 0.5 * (self._dr[2] + turned)
            k = self.DEAD_RECKON_WALK * dt
            self._dr[0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * k
            self._dr[1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * k
        self._dr[2] = turned
        self._dr_t = now

    def _on_scan(self, pts_ws: np.ndarray, stamp: float, *,
                 received_at: float | None = None) -> None:
        with self.lock:
            self._freshness_received(
                "scan", stamp, self._clock() if received_at is None else received_at)
        if self.odom_age_s() > 0.5 or self._sensor_age_s(stamp, self._clock()) > 0.5:
            self._freshness_admission("scan", "pose_or_sensor_unavailable")
            return
        with self.lock:
            best = min(self._odom, key=lambda o: abs(o[0] - stamp), default=None)
            T = best[1] if best is not None and abs(best[0] - stamp) < 0.2 else None
            zero, learner = self._zero, self.self_learner
            reset = self.frame_reset
        if T is None or zero is None or reset:
            self._freshness_admission("scan", "no_matching_pose_or_start_frame")
            return
        floor_z = float(T[2, 3]) - LIDAR_HEIGHT_M  # re-grounded per scan: odometry z drifts
        # one self filter for the ring and the live map
        registered = drop_self_points(pts_ws, T, floor_z)
        start = self._ws_to_start(registered, floor_z, zero)
        origin = self._ws_to_start(T[:3, 3][None, :], floor_z, zero)[0]
        # the learned self mask: points fixed in the body frame while moving
        yaw = math.atan2(T[1, 0], T[0, 0]) - zero[2]
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = start[:, 0] - origin[0], start[:, 1] - origin[1]
        bx, by = c * dx - s * dy, s * dx + c * dy
        learner.observe(bx, by, start[:, 2], (float(origin[0]), float(origin[1])))
        keep = ~learner.mask(bx, by, start[:, 2])
        start, registered = start[keep], registered[keep]
        distance = ring_min(start, float(origin[0]), float(origin[1]))
        start.flags.writeable = origin.flags.writeable = registered.flags.writeable = False
        with self.lock:
            # drop the cloud when the start frame changed while it was filtered
            if self.frame_reset or self._zero is not zero:
                self._freshness_admission("scan", "frame_changed_during_processing")
                return
            observed_at = self._clock()
            age = self._sensor_age_s(stamp, observed_at)
            if (age > 0.5 or not self._health[0] or observed_at - self._health[1] > 0.5
                    or not self._odom or observed_at - self._odom[-1][2] > 0.5
                    or self._sensor_age_s(self._odom[-1][0], observed_at) > 0.5):
                self._freshness_admission("scan", "stale_after_processing")
                return
            observed_at -= age  # processing a queued scan cannot renew its freshness
            self.scan_times.append(time.time())
            self.ring_m, self.ring_t = distance, observed_at
            self._last_points = start
            self._registered_scan = (registered, observed_at)
            # The map takes the same self-filtered cloud the ring trusts.
            if len(self._scans) == self._scans.maxlen:
                self._ingestion["map_queue_dropped"] += 1
            self._scans.append((start, origin, observed_at))
            self._ingestion["map_queue_max"] = max(
                self._ingestion["map_queue_max"], len(self._scans))
            self._freshness_admission("scan", accepted_at=observed_at)

    def registered_scan(self):
        """Last admitted self-filtered cloud in ws_map, with its capture time."""
        with self.lock:
            return self._registered_scan

    def _freshness_received(self, kind, stamp, now):
        sample = self._freshness[kind]
        sample["received"] += 1
        sample["received_at"], sample["stamp"] = now, stamp
        age = self._sensor_age_s(stamp, now)
        sample["last_excess_age_s"] = age if math.isfinite(age) else None

    def _freshness_admission(self, kind, reason="", *, accepted_at=None):
        sample = self._freshness[kind]
        sample["rejected" if reason else "accepted"] += 1
        sample["last_rejection"] = reason
        if not reason:
            sample["accepted_at"] = accepted_at

    def freshness_status(self, *, stable_s: float = 0.5) -> dict:
        """Small read-only snapshot; receive times mean application take, not DDS wire arrival."""
        now = self._clock()
        with self.lock:
            samples = {kind: dict(sample) for kind, sample in self._freshness.items()}
            ingestion = {key: value for key, value in self._ingestion.items()
                         if key not in {"stages", "pumps"}}
            ingestion["stages"] = {name: dict(row)
                                   for name, row in self._ingestion["stages"].items()}
            ingestion["pumps"] = {
                name: {**row, "age_s": max(0., now - row["last_at"])}
                for name, row in self._ingestion["pumps"].items()}
            ingestion.update(map_queue=len(self._scans), scan_dds_history_depth=1,
                             scan_dds_overwritten=None, errors=self._pump_errors)
            imu, health, since = self._imu_reference, self._health, self._healthy_since

        def age(at):
            return max(0., now - at) if at is not None and math.isfinite(at) else None

        reasons = ["frame_reset"] if self.frame_reset else []
        status = {"raw_imu_receive_age_s": age(imu[1]) if imu else None,
                  "health_receive_age_s": age(health[1]) if health[1] else None,
                  "health_ok": bool(health[0]), "health_stable_s": age(since),
                  "ingestion": ingestion}
        if imu is None:
            reasons.append("imu_missing")
        elif now - imu[1] > .5:
            reasons.append("imu_stale")
        if not health[1]:
            reasons.append("health_missing")
        elif now - health[1] > .5:
            reasons.append("health_stale")
        elif not health[0]:
            reasons.append("health_unhealthy")
        elif self.health_stable_s() < stable_s:
            reasons.append("health_settling")
        for kind, sample in samples.items():
            recv_age, accepted_age = age(sample["received_at"]), age(sample["accepted_at"])
            stamp = sample["stamp"]
            sensor_age = self._sensor_age_s(stamp, now) if stamp is not None else math.inf
            status.update({f"{kind}_receive_age_s": recv_age,
                           f"{kind}_accepted_age_s": accepted_age,
                           f"{kind}_sensor_age_s": (
                               sensor_age if math.isfinite(sensor_age) else None),
                           f"{kind}_last_excess_age_s": sample["last_excess_age_s"],
                           kind: {key: sample[key] for key in
                                  ("received", "accepted", "rejected", "last_rejection")}})
            if recv_age is None:
                reasons.append(f"{kind}_missing")
            elif recv_age > .5:
                reasons.append(f"{kind}_stale")
            elif math.isfinite(sensor_age) and sensor_age > .5:
                reasons.append(f"{kind}_delayed")
            elif sample["last_rejection"]:
                reasons.append(f"{kind}_rejected")
            elif accepted_age is None or accepted_age > .5:
                reasons.append(f"{kind}_unaccepted")
        status.update(reason=reasons[0] if reasons else "ready", reasons=reasons)
        return status

    def _ws_to_start(self, pts: np.ndarray, floor_z: float, zero=None) -> np.ndarray:
        zero = self._zero if zero is None else zero
        if zero is None:
            raise RuntimeError("no start frame: the SLAM restart cleared it")
        x0, y0, yaw0 = zero
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        dx, dy = pts[:, 0] - x0, pts[:, 1] - y0
        out = np.empty((len(pts), 3))
        out[:, 0], out[:, 1] = c * dx - s * dy, s * dx + c * dy
        out[:, 2] = pts[:, 2] - floor_z
        return out

    def _latest(self):
        with self.lock:
            return self._odom[-1] if self._odom else None

    def zeroed(self) -> bool:
        return self._zero is not None

    def zero(self) -> None:
        latest = self._latest()
        if latest is None:
            raise RuntimeError("no /ws/laser_odometry_aligned yet: is SuperOdom running?")
        T = latest[1]
        with self.lock:
            self._zero = (float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0]))
            self.start_frame_metadata = {
                "source_frame": "ws_map", "target_frame": "exploration_start",
                "zero_xyyaw": list(self._zero), "zero_recorded_wall_s": time.time(),
                "odom_stamp_s": latest[0], "odom_receive_s": latest[2],
                "xy_rule": "p_start_xy = R(-yaw0) @ (p_ws_xy - [x0, y0])",
                "z_rule": "p_start_z = p_ws_z - (odom_z_at_scan - lidar_height_m)",
                "lidar_height_m": LIDAR_HEIGHT_M,
                "notes": "Planar origin only; Z is re-grounded per scan, not a fixed SE(3). "
                         "Timestamps here inherit source float precision; native stamps in bag. "
                         "Start-frame epoch IDs are separate from /homebody/slam_epoch IDs.",
            }
            self._scans.clear()  # scans converted to the replaced start frame
            self.self_learner = SelfLearner()
            self._freshness["scan"]["accepted_at"] = None
            self._last_points = np.zeros((0, 3))
            self._registered_scan = (None, None)
            self.ring_m, self.ring_t = float("inf"), 0.0
            self.frame_reset, self.frame_reset_why = False, ""

    def pose(self) -> tuple[float, float, float]:
        """The last registered pose carried to now on the commanded twist, in the start frame."""
        now = self._clock()
        with self.lock:
            if not self._odom or self._zero is None:
                raise RuntimeError("no pose: Zero first")
            _, T, recv = self._odom[-1]
            if now - recv <= self.DEAD_RECKON_MAX_S:
                self._advance_dr(now)
            dx, dy, dyaw = self._dr
            zero = self._zero
        p = self._ws_to_start(np.array([[T[0, 3] + dx, T[1, 3] + dy, 0.0]]), 0.0, zero)[0]
        yaw = math.atan2(T[1, 0], T[0, 0]) + dyaw - zero[2]
        return float(p[0]), float(p[1]), float((yaw + math.pi) % (2 * math.pi) - math.pi)

    def odom_age_s(self) -> float:
        """Seconds since the engine's last registered pose: the console's number."""
        latest = self._latest()
        if latest is None:
            return math.inf
        now = self._clock()
        if not self._health[0] or now - self._health[1] > 0.5:
            return math.inf
        return max(now - latest[2], self._sensor_age_s(latest[0], now))

    def pose_age_s(self) -> float:
        """The age of `pose()`: 0 while the dead reckoning carries it, else the engine's silence."""
        age = self.odom_age_s()
        return 0.0 if age <= self.DEAD_RECKON_MAX_S else age

    def ring_fresh_m(self) -> float:
        """The closest live lidar point, or 0 (refuse motion) when the scan feeding it is stale."""
        return self.ring_m if time.time() - self.ring_t <= 0.5 else 0.0

    def ring_ahead_m(self, vx: float, vy: float) -> float:
        """The closest live lidar point in the direction of a body-frame twist (0 when stale).
        Under RING_STOP_M the offending points are kept in `ring_hits`: the map learns them."""
        if time.time() - self.ring_t > self.RING_FRESH_S:
            return 0.0
        x, y, yaw = self.pose()
        with self.lock:
            points = self._last_points
        heading = yaw + math.atan2(vy, vx)
        ahead = ring_ahead(points, x, y, heading)
        if ahead < RING_STOP_M:
            self.ring_hits = ring_ahead_points(points, x, y, heading, RING_STOP_M)
        return ahead

    def scans(self):
        with self.lock:
            out = list(self._scans)
            self._scans.clear()
        return out

    def live_points(self) -> np.ndarray:
        """The same self-filtered cloud the map ingests."""
        return self._last_points

    def scan_hz(self) -> float:
        t = [v for v in self.scan_times if time.time() - v < 5.0]
        return len(t) / 5.0

    def raw_scan_hz(self, window_s: float = 3.0) -> float:
        t = [v for v in self.raw_scan_times if time.time() - v < window_s]
        return len(t) / window_s

    def reset_samples(self):
        """Drop the start frame and every sample, for an engine restart."""
        with self.lock:
            self._zero = None
            self._odom.clear()
            self._state.clear()
            self._imu_reference = self._sensor_offset = self._healthy_since = None
            self._health = (False, 0.0)
            self._dr, self._cmd, self._dr_gyro0 = [0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 0.0), None
            self._scans.clear()
            self._last_points = np.zeros((0, 3))
            self._registered_scan = (None, None)
            self.ring_m, self.ring_t = float("inf"), 0.0
            self.frame_reset, self.frame_reset_why = False, ""

    def stop(self):
        self._sensor_stop.set()

    def close(self):
        self.stop()
        errors = []
        for worker in self._sensor_threads:
            try:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    errors.append("sensor worker still running")
            except Exception as exc:  # noqa: BLE001 -- finish each sensor owner before reporting
                errors.append(f"sensor worker: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


def reset_verdict(stamp_step_s: float, jump_m: float, origin_like: bool = True) -> str:
    """Why two consecutive odometry samples, judged on header stamps, cannot share one origin,
    or "" when they can."""
    if stamp_step_s > SuperOdomLocalization.ODOM_GAP_RESET_S and origin_like:
        return f"odometry header stamps {stamp_step_s:.1f} s apart and the pose jumped home"
    if stamp_step_s < -0.5:
        return f"odometry header stamps ran {-stamp_step_s:.1f} s backwards"
    if jump_m > SuperOdomLocalization.ODOM_JUMP_RESET_M:
        return f"odometry jumped {jump_m:.2f} m between two samples"
    return ""


def _quat_to_R(w, x, y, z) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _cloud_xyz(msg) -> np.ndarray:
    offsets = {f.name: int(f.offset) for f in msg.fields}
    step = int(msg.point_step)
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, step)
    return np.stack(
        [
            np.frombuffer(raw[:, offsets[c] : offsets[c] + 4].tobytes(), np.float32)
            for c in ("x", "y", "z")
        ],
        1,
    ).astype(np.float64)
