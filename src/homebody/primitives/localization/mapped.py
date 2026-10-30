"""Registered SuperOdom samples bound to a measured deployment-map registration."""
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from homebody.primitives.observations import ObservationError, frozen_array

from .map_binding import Alignment
from .superodom import SuperOdomLocalization


@dataclass(frozen=True)
class RegisteredPose:
    captured_at: float
    source_stamp: float
    T_map_lidar: np.ndarray
    generation: int
    quality: float

    def __post_init__(self):
        object.__setattr__(self, "T_map_lidar", frozen_array(self.T_map_lidar, (4, 4)))


class SuperOdomMappedLocalization:
    """One DDS reader; no localization service/process is started or restarted."""
    def __init__(self, iface, binding, alignment_path, *, source=None, clock=time.time,
                 monotonic=time.monotonic, sleep=time.sleep, wait_s=2.5):
        self.binding = binding
        self.alignment = Alignment.load(alignment_path, binding)
        self.source = source or SuperOdomLocalization(iface, start=False)
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.wait_s = wait_s
        self.generation = 1
        self._closed = False
        if source is None:
            self.source._start_dds()
        try:
            deadline = monotonic() + wait_s
            while self.source._latest() is None and monotonic() < deadline:
                sleep(.01)
            latest = self.source._latest()
            if latest is None:
                raise ObservationError("No current registered SuperOdom pose; existing service must be running")
            stamp, T, _ = latest
            # An old registration cannot silently attach to a fresh ws_map after restart.
            if abs(stamp - self.alignment.anchor_stamp) > 30:
                raise ObservationError("Map registration anchor is not from this current odometry session")
            delta = np.linalg.inv(self.alignment.anchor_pose) @ T
            angle = math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            if np.linalg.norm(delta[:3, 3]) > .15 or angle > math.radians(10):
                raise ObservationError("Robot moved away from the measured registration anchor before binding")
            self.source.zero()
        except BaseException:
            self.close()
            raise

    def _healthy(self):
        if self._closed:
            raise ObservationError("Localization primitive is closed")
        with self.source.lock:
            if self.source.frame_reset:
                raise ObservationError("Localization frame changed; a new map registration is required")
            healthy, at = self.source._health
        if not healthy or not 0 <= self.clock() - at <= .5:
            raise ObservationError("SuperOdom registration health is unavailable or stale")
        status = self.source.freshness_status()
        if status["reasons"]:
            raise ObservationError("SuperOdom evidence unavailable: " + ", ".join(status["reasons"]))

    def pose_at(self, timestamp=None, *, max_skew_s=.1):
        deadline = self.monotonic() + min(self.wait_s, .5)
        while True:
            self._healthy()
            now = self.clock()
            with self.source.lock:
                offset = self.source._sensor_offset
                if offset is None:
                    raise ObservationError("SuperOdom source-to-receiver clock association is unavailable")
                rows = list(self.source._odom)
            if not rows:
                raise ObservationError("No registered localization observations")
            chosen = rows[-1] if timestamp is None else min(rows, key=lambda x: abs(x[0] + offset - timestamp))
            if timestamp is None or abs(chosen[0] + offset - timestamp) <= max_skew_s:
                break
            # Registered poses can arrive after RGB-D from the same instant.
            # Wait only for that pending pose, never extrapolate or widen pairing.
            if (rows[-1][0] + offset < timestamp and 0 <= now - timestamp <= .5
                    and self.monotonic() < deadline):
                self.sleep(.01)
                continue
            raise ObservationError("RGB-D capture has no contemporaneous registered pose")
        stamp, T, received = chosen
        captured = stamp + offset
        if not 0 <= now - captured <= .5 or not 0 <= now - received <= .5:
            raise ObservationError("Registered pose is stale")
        # Health is categorical; 1 denotes the measured healthy verdict, not an ICP fitness.
        return RegisteredPose(captured, stamp, self.alignment.T_map_source @ T, self.generation, 1.)

    def obstacles(self):
        self._healthy()
        points, captured = self.source.registered_scan()
        if points is None or captured is None or not 0 <= self.clock() - captured <= .5:
            raise ObservationError("No fresh registered obstacle cloud in the bound map frame")
        transform = self.alignment.T_map_source
        return points @ transform[:3, :3].T + transform[:3, 3], float(captured)

    def note_twist(self, forward, lateral, yaw):
        self.source.note_twist(forward, lateral, yaw)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.source._sensor_stop.set()
        for worker in self.source._sensor_threads:
            if worker is not threading.current_thread():
                worker.join(timeout=1.)
