"""Task commands over the persistent body host and the guarded Dex3 transport."""
import json
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebody.primitives.actions import ActionRejected
from homebody.primitives.observations import HandState, ObservationError


@dataclass(frozen=True)
class RealHandState(HandState):
    effort_available: bool = True
    contact_state: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.contact_state not in (None, "contact", "empty", "unknown"):
            raise ObservationError("Invalid physical hand contact evidence")

    @property
    def effort_fraction(self):
        if not self.effort_available:
            return None
        return super().effort_fraction

    def retained(self, settings):
        if self.contact_state is not None:
            return self.contact_state == "contact"
        if not self.effort_available:
            raise ObservationError("Physical hand contact evidence is unavailable")
        return super().retained(settings)

    def contact_supported(self, settings):
        if self.contact_state is not None:
            return self.contact_state == "contact"
        if not self.effort_available:
            raise ObservationError("Physical hand contact evidence is unavailable")
        return super().contact_supported(settings)

    def closed_empty(self, settings):
        if self.contact_state is not None:
            from .dex3 import DEX3_EMPTY_CLOSURE
            return self.contact_state == "empty" and self.closure > DEX3_EMPTY_CLOSURE
        return super().closed_empty(settings)


def measured_hand(row, side, calibration, effort_calibration):
    if np.any(row.get("motor_faults", ())):
        raise ObservationError(f"Dex3 {side} hand reports a motor fault")
    joints = np.asarray(row["joints"], dtype=float)
    ray = np.asarray(calibration["hand_closed"][side], dtype=float)
    target = np.asarray(row["target"], dtype=float)
    names = calibration["hand_joints"][side]
    if (joints.shape != (7,) or target.shape != (7,) or ray.shape != (7,)
            or not np.isfinite([joints, target, ray]).all()):
        raise ObservationError("Invalid measured Dex3 joints or command reference")
    gaps, efforts = [], []
    available = effort_calibration is not None and row["tau_raw"] is not None
    tau = None
    if available:
        scaling = effort_calibration[side]
        tau = np.asarray(row["tau_raw"]) * np.asarray(scaling["raw_to_nm"])
        ceilings = np.asarray(scaling["rated_nm"])
        if (tau.shape != (7,) or ceilings.shape != (7,) or not np.isfinite(tau).all()
                or np.any(np.abs(tau) > 10 * ceilings)):
            raise ObservationError("Dex3 torque exceeds the calibrated sensor range")
    for finger in ("thumb", "index", "middle"):
        indices = [i for i, name in enumerate(names) if f"_hand_{finger}_" in name]
        direction = ray[indices]
        norm = float(direction @ direction)
        if norm <= 0:
            raise ObservationError("Hand calibration has no closing ray for a finger")
        gaps.append(max(0., float((target[indices] - joints[indices]) @ direction / norm)))
        efforts.append(0. if not available else float(np.max(np.abs(tau[indices]) / ceilings[indices])))
    closure = float(np.clip(joints @ ray / (ray @ ray), 0, 1))
    return RealHandState(joints, closure, np.array(efforts), np.array(gaps), available,
                         row.get("contact_state"))


def load_hand_calibration(path):
    if path is None:
        return None
    doc = json.loads(Path(path).read_text())
    if not doc.get("provenance") or not doc.get("robot_id"):
        raise ObservationError("Hand effort calibration requires robot identity and measured provenance")
    for side in ("left", "right"):
        for name in ("raw_to_nm", "rated_nm"):
            value = np.asarray(doc[side][name], dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ObservationError(f"Invalid {side} hand {name}")
    return doc


class G1HostMotion:
    """One task client, one hand writer; never constructs a low-command body writer."""
    def __init__(self, config, calibration, *, body=None, hands=None, lease=None,
                 clock=time.time, monotonic=time.monotonic, sleep=time.sleep, start=True):
        from .body_host import BodyClient
        from .dex3 import Dex3Port
        from .writer import SoleWriter, lease_path

        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.calibration = calibration
        self.effort_calibration = load_hand_calibration(config.hand_calibration)
        self._lock = threading.RLock()
        self._history = deque(maxlen=1000)
        self._hand_history = {side: deque(maxlen=1000) for side in ("left", "right")}
        self._stop = threading.Event()
        self._fault = None
        self._fault_status = {}
        self._fault_transitions = deque(maxlen=256)
        self._fault_transition_count = 0
        self._closed = False
        self._generation = 1
        self._body_interruptions = 0
        self._arms = None
        self._velocity = np.zeros(3)
        self._velocity_until = 0.
        self.body = self.hands = self.lease = None
        self._worker = None
        try:
            path = lease_path(config.writer_lock_path)
            self.lease = lease or SoleWriter(path.with_name(path.name + ".client"))
            self.body = body or BodyClient(config.body_address)
            self.body.on_state = self._receive
            if not self.body.wait_fresh(config.sensor_wait_s):
                raise ObservationError("Existing body host did not provide fresh telemetry")
            self._receive(self.body.state)
            self.hands = hands or Dex3Port(config.iface)
            self._arms = np.asarray(self.body.state["q"], dtype=float)[15:29].copy()
            if start:
                # Bind the existing host while standing before any task command.
                self.hold()
                self._worker = threading.Thread(target=self._hand_loop, name="homebody-real-hands", daemon=True)
                self._worker.start()
        except BaseException:
            self.close()
            raise

    def _receive(self, message):
        with self._lock:
            self._history.append(dict(message))

    @property
    def generation(self):
        count = self.body.interruptions
        if count != self._body_interruptions:
            self._body_interruptions = count
            self._generation += 1
        return self._generation

    def _check(self):
        if self._closed:
            raise ActionRejected("REAL motion primitive is closed")
        if self._fault is not None:
            raise ActionRejected(f"REAL hand command loop failed: {self._fault}")
        self.body.require_fresh()
        state = self.body.state
        if state.get("driver") not in (None, self.body.claim):
            raise ActionRejected("Another body client owns execution authority")
        if not state.get("active") or state.get("ramping") or not state.get("link_live", True):
            raise ActionRejected("G1 must be deployed and standing with a live body link")

    def state_at(self, timestamp=None, max_skew_s=.1):
        self.body.require_fresh()
        with self._lock:
            rows = tuple(self._history)
        state = (rows[-1] if timestamp is None else min(rows, key=lambda row: abs(row["t"] - timestamp)))
        if timestamp is not None and abs(state["t"] - timestamp) > max_skew_s:
            raise ObservationError("No contemporaneous body joint sample for RGB-D capture")
        if not 0 <= self.clock() - state["t"] <= .5:
            raise ObservationError("Body telemetry is stale")
        return dict(state)

    def _record_hands(self, *, require_both=True):
        rows = self.hands.measured_state(require_both=require_both)
        with self._lock:
            for side, row in rows.items():
                history = self._hand_history[side]
                if not history or row["captured_at"] > history[-1]["captured_at"]:
                    history.append(row)
        return rows

    def hand_states(self, timestamp=None, max_skew_s=.1):
        current = self._record_hands()
        now = self.clock()
        if any(not 0 <= now - row["captured_at"] <= .1 for row in current.values()):
            raise ObservationError("Dex3 observation is stale")
        with self._lock:
            rows = {side: (history[-1] if timestamp is None else
                           min(history, key=lambda row: abs(row["captured_at"] - timestamp)))
                    for side, history in self._hand_history.items()}
        if timestamp is not None and any(abs(row["captured_at"] - timestamp) > max_skew_s
                                         for row in rows.values()):
            raise ObservationError("No contemporaneous hand evidence for RGB-D capture")
        return {side: measured_hand(row, side, self.calibration, self.effort_calibration)
                for side, row in rows.items()}, {side: row["captured_at"] for side, row in rows.items()}

    def velocity(self, values):
        values = np.asarray(values, dtype=float)
        with self._lock:
            self._check()
            self._velocity = values.copy()
            self._velocity_until = self.monotonic() + .2
            self.body.send(walk=bool(np.any(values)), vel=values.tolist())

    def arm(self, side, joints):
        with self._lock:
            self._check()
            sl = slice(0, 7) if side == "left" else slice(7, 14)
            self._arms[sl] = joints
            self.body.set_arms(self._arms)

    def grip(self, side, closure):
        with self._lock:
            self._check()
            self.hands.set_targets(**{side: np.asarray(self.calibration["hand_closed"][side]) * closure})

    def _send_hold(self):
        # A task STOP preserves cached arm references and the Dex3 command worker.
        with self._lock:
            self._velocity[:] = 0
            self._velocity_until = 0.
        self.body.send(cmd="stand")

    def hold(self):
        before = int((self.body.state or {}).get("tick", -1))
        previous_driver = (self.body.state or {}).get("driver")
        self._send_hold()
        deadline = self.monotonic() + .3
        while self.monotonic() < deadline:
            self.body.require_fresh()
            state = self.body.state or {}
            driver = state.get("driver")
            if driver not in (None, self.body.claim, previous_driver):
                raise ActionRejected("Cannot confirm hold: body authority changed")
            if (driver == self.body.claim and int(state.get("tick", -1)) > before
                    and state.get("walk") is False):
                return
            self.sleep(.004)
        raise ActionRejected("Body host did not confirm a later held tick; STOP is unverified")

    def _hand_loop(self):
        previous = self.monotonic()
        while not self._stop.wait(.004):
            now = self.monotonic()
            if self._fault is not None:
                self._maintain_fault_hold()
                previous = now
                continue
            try:
                self.hands.tick(min(.02, max(0., now - previous)))
                self._record_hands(require_both=False)
                if now >= self._velocity_until and np.any(self._velocity):
                    self._send_hold()
            except Exception as exc:  # noqa: BLE001 -- any command-loop fault holds the robot
                with self._lock:
                    self._fault = exc
                self._maintain_fault_hold()
            previous = now

    def _maintain_fault_hold(self):
        try:
            status = self.hands.hold_tick()
        except Exception as exc:  # noqa: BLE001 -- retain owner after an unavailable hold publisher
            status = {"published": {}, "observation_error": None,
                      "publication_errors": {"unknown": f"{type(exc).__name__}: {exc}"}}
        base_error = None
        try:
            self._send_hold()
        except Exception as exc:  # noqa: BLE001 -- base transport failure cannot terminate grip publication
            base_error = f"{type(exc).__name__}: {exc}"
        status = {"fault": f"{type(self._fault).__name__}: {self._fault}",
                  "hand_reference_published": status["published"],
                  "hand_observation_error": status["observation_error"],
                  "hand_publication_errors": status["publication_errors"],
                  "base_hold_error": base_error, "physical_retention_confirmed": False}
        with self._lock:
            if status != self._fault_status:
                self._fault_transition_count += 1
                self._fault_transitions.append({"sequence": self._fault_transition_count,
                                                "time": self.clock(), **deepcopy(status)})
                self._fault_status = status

    def command_status(self):
        """Passive status; a retained worker does not certify a retained physical object."""
        with self._lock:
            return {**deepcopy(self._fault_status),
                    "transitions": deepcopy(list(self._fault_transitions)),
                    "omitted_transition_count": self._fault_transition_count - len(self._fault_transitions)}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.)
        try:
            if self.body is not None:
                self.body.close()
        finally:
            try:
                if self.hands is not None:
                    self.hands.close()
            finally:
                if self.lease is not None:
                    self.lease.release()
