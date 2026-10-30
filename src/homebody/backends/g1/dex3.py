"""Dex3 DDS transport owned by :class:`RealBackend`.

Both hands take the seven-joint gripper payload; every motor is rate-limited and closure is
measured. Contact is a fingertip joint short of its commanded reference. One reader thread per
hand reads at 250 Hz and the control tick takes the newest sample.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from homebody.backends.g1 import profile as g1
from homebody.backends.g1.hand_math import hand_open_bound
from homebody.primitives.localization.dds import initialize
from homebody.primitives.navigation.observations.gripper_state import HELD, UNKNOWN
from homebody.real2sim.exploration.settings import current

DEX3_EMPTY_CLOSURE = 0.9  # closure past which a close without contact is empty
DEX3_DROPPED_CLOSURE = 0.96  # closure past which a verified grasp is dropped

_CMD = {"left": "rt/dex3/left/cmd", "right": "rt/dex3/right/cmd"}
_STATE = {"left": "rt/dex3/left/state", "right": "rt/dex3/right/state"}
_SIDES = ("left", "right")
_CONTACT_GAP_RAD = 0.08
_CONTACT_SAMPLES = 3
_CONTACT_STALL_S = 0.40  # a closing hand with a tip gap this long has contact
_CONTACT_LOSS_SAMPLES = 3
_CONTACT_LOSS_S = _CONTACT_STALL_S  # a latch clears once the gap is gone this long
_STATE_FRESH_S = 0.10  # a hand sample this recent is live
_STATE_WAIT_S = 0.05
_INIT_WAIT_S = 1.0  # start-up wait for discovery
_READ_PERIOD_S = 1.0 / 250.0  # reader poll period
_TAU_RATING = np.array([2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])  # rated torque per finger, Nm
_LEAD_SHRINK_RAD = 0.01    # lead shrink per tick over the torque limit
_LEAD_RESTORE_RAD = 0.002  # lead restore per tick under it
_TAU_IMPLAUSIBLE = 10.0  # tau_est past this times the rating is not Nm and is ignored
_TEMP_PLAUSIBLE_C = (-20.0, 150.0)   # outside this the field is not degrees and is ignored


def _tip_indices(side: str) -> np.ndarray:
    """The distal joint of the thumb, index and middle finger."""
    names = g1.HAND_JOINTS[side]
    return np.asarray([max(i for i, name in enumerate(names) if f"_{digit}_" in name)
                       for digit in ("thumb", "index", "middle")], dtype=int)


def _tip_gap(commanded, measured, side: str, threshold: float = _CONTACT_GAP_RAD) -> bool:
    """Whether any fingertip joint is short of its closing reference by `threshold`."""
    command = np.asarray(commanded, dtype=float)
    actual = np.asarray(measured, dtype=float)
    tips = _tip_indices(side)
    close = np.asarray(g1.HAND_CLOSED[side], dtype=float)[tips]
    direction = np.sign(close - np.asarray(g1.HAND_OPEN, dtype=float)[tips])
    shortfall = direction * (command[tips] - actual[tips])
    return bool(np.max(shortfall) >= threshold)


class Dex3Port:
    def __init__(self, interface_name: str) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        except ImportError as exc:
            raise ImportError("Dex3 requires unitree_sdk2py with HG HandCmd/HandState") from exc
        initialize(interface_name)
        self._new_command = unitree_hg_msg_dds__HandCmd_
        self._init_state()
        self._pub, self._sub = {}, {}
        for side in _SIDES:
            self._pub[side] = ChannelPublisher(_CMD[side], HandCmd_)
            self._sub[side] = ChannelSubscriber(_STATE[side], HandState_)
            self._pub[side].Init()
            self._sub[side].Init()
        self._start_readers()
        self.refresh(require_both=True, wait_s=_INIT_WAIT_S)

    def _init_state(self) -> None:
        hands = current().hands
        self._kp, self._kd = hands.kp, hands.kd
        self._lead_cap = np.full(7, hands.lead_rad)
        self._lead_min = max(hands.lead_min_rad, _CONTACT_GAP_RAD + 0.02)  # above the contact gap
        self._tau_fraction = hands.torque_fraction
        self._temp_ease_c, self._temp_rest_c = hands.temp_ease_c, hands.temp_rest_c
        self._lock = threading.RLock()
        self._arrival = threading.Condition()
        self._inbox = {"left": None, "right": None}    # (count, q, wall time) off the reader
        self._taken = {"left": 0, "right": 0}
        self._measured = {"left": None, "right": None}
        self._target = {"left": None, "right": None}
        self._command = {"left": None, "right": None}
        self._published = {"left": None, "right": None}
        self._hold_reference = None
        self._stamp = {"left": 0.0, "right": 0.0}
        self._latched = {"left": None, "right": None}
        self._latched_once = {"left": False, "right": False}   # a grasp verified on this close
        self._gap_samples = {"left": 0, "right": 0}
        self._gap_since = {"left": None, "right": None}
        self._loss_samples = {"left": 0, "right": 0}
        self._loss_since = {"left": 0, "right": 0}
        self._tau = {"left": None, "right": None}          # newest |tau_est| per finger, Nm
        self._hot = {"left": None, "right": None}          # newest per-finger temperature, C
        self._hot_peak = {side: np.zeros(7) for side in _SIDES}
        self._fault = {"left": None, "right": None}        # the packet's own motorstate word
        self._said_fault = {"left": False, "right": False}
        self._said_hot = {"left": False, "right": False}
        self._lead = {side: np.full(7, self._lead_cap) for side in _SIDES}
        self._tau_peak = {side: np.zeros(7) for side in _SIDES}
        self._said_over = {"left": False, "right": False}
        self._said_untrusted = {"left": False, "right": False}

    def _start_readers(self) -> None:
        self._stop = threading.Event()
        self._readers = [threading.Thread(target=self._read_loop, args=(side,), daemon=True,
                                          name=f"dex3-{side}") for side in _SIDES]
        for thread in self._readers:
            thread.start()

    def _read_loop(self, side: str) -> None:
        """Read one hand's state at the reader period until stopped."""
        while not self._stop.is_set():
            t0 = time.monotonic()
            sample = self._sub[side].Read(timeout=_STATE_WAIT_S)
            if sample is not None:
                self._on_state(side, sample)
            self._stop.wait(max(0.0, _READ_PERIOD_S - (time.monotonic() - t0)))

    def _on_state(self, side: str, sample) -> None:
        """Keep the newest joint positions, torques, temperatures and faults; `refresh`
        validates them."""
        motors = sample.motor_state[:7]
        q = np.asarray([motor.q for motor in motors], dtype=float)
        tau = np.asarray([motor.tau_est for motor in motors], dtype=float)
        hot = np.asarray([max(motor.temperature) for motor in motors], dtype=float)
        fault = np.asarray([int(getattr(motor, "motorstate", 0)) for motor in motors], dtype=int)
        with self._arrival:
            count = self._inbox[side][0] + 1 if self._inbox[side] is not None else 1
            self._inbox[side] = (count, q, time.time(), tau, hot, fault)
            self._arrival.notify_all()

    def _await_fresh(self, wait_s: float) -> None:
        deadline = time.monotonic() + wait_s
        with self._arrival:
            while True:
                now = time.time()
                missing = [side for side in _SIDES if self._inbox[side] is None
                           or now - self._inbox[side][2] > _STATE_FRESH_S]
                remaining = deadline - time.monotonic()
                if not missing or remaining <= 0.0:
                    break
                self._arrival.wait(remaining)
        if missing:
            raise TimeoutError(f"no Dex3 state from {', '.join(missing)} hand(s)")

    def refresh(self, *, require_both: bool = False, wait_s: float = _STATE_WAIT_S) -> None:
        """Take each hand's newest sample once. Only `require_both` (never the tick) waits."""
        if require_both:
            self._await_fresh(wait_s)
        with self._lock:
            for side in _SIDES:
                with self._arrival:
                    item = self._inbox[side]
                if item is None or item[0] == self._taken[side]:
                    continue
                self._taken[side], q, stamp, tau, hot, fault = item
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise RuntimeError(f"Dex3 {side} state contains invalid joint positions")
                if np.isfinite(tau).all():
                    self._tau[side] = np.abs(tau)
                    self._tau_peak[side] = np.maximum(self._tau_peak[side], np.abs(tau))
                if np.isfinite(hot).all():
                    self._hot[side] = hot
                    self._hot_peak[side] = np.maximum(self._hot_peak[side], hot)
                self._fault[side] = fault
                if fault.any() and not self._said_fault[side]:
                    self._said_fault[side] = True
                    print(f"[dex3] {side} hand motor fault: motorstate {fault.tolist()} at "
                          f"{hot.tolist()} C — a tripped motor stops answering until it clears",
                          flush=True)
                self._measured[side] = q
                self._stamp[side] = stamp
                command = self._command[side]
                has_contact = command is not None and _tip_gap(command, q, side)
                now = time.time()
                if has_contact:
                    if self._gap_samples[side] == 0:
                        self._gap_since[side] = now
                    self._gap_samples[side] += 1
                    self._loss_samples[side] = 0
                else:
                    self._gap_samples[side] = 0
                    self._gap_since[side] = None
                    if self._latched[side] is not None:
                        if self._loss_samples[side] == 0:
                            self._loss_since[side] = now
                        self._loss_samples[side] += 1
                stalled = (self._gap_samples[side] >= _CONTACT_SAMPLES
                           and self._gap_since[side] is not None
                           and now - self._gap_since[side] >= _CONTACT_STALL_S)
                if stalled and self._latched[side] is None:
                    self._latched[side] = g1.grip_from_hand_q(command, side)
                    self._latched_once[side] = True
                # A lost gap clears the latch only past DEX3_DROPPED_CLOSURE.
                if (self._loss_samples[side] >= _CONTACT_LOSS_SAMPLES
                        and now - self._loss_since[side] >= _CONTACT_LOSS_S
                        and g1.grip_from_hand_q(q, side) > DEX3_DROPPED_CLOSURE):
                    self._latched[side] = None
                    self._loss_samples[side] = 0
                target_grip = g1.grip_from_hand_q(
                    self._target[side] if self._target[side] is not None else q, side)
                if self._latched[side] is not None and target_grip < self._latched[side]:
                    self._latched[side] = None
                    self._gap_samples[side] = 0
                    self._loss_samples[side] = 0
                if target_grip < 0.05:
                    self._latched_once[side] = False    # commanded open
                if self._command[side] is None:
                    self._command[side] = q.copy()
                    self._target[side] = q.copy()

    def set_targets(self, left=None, right=None) -> None:
        with self._lock:
            for side, value in (("left", left), ("right", right)):
                if value is None:
                    continue
                q = np.asarray(value, dtype=float)
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise ValueError(f"Dex3 {side} target must be finite (7,)")
                lo = np.minimum(hand_open_bound(side), g1.HAND_CLOSED[side])
                hi = np.maximum(hand_open_bound(side), g1.HAND_CLOSED[side])
                self._target[side] = np.clip(q, lo, hi)

    def tick(self, dt: float) -> None:
        self.refresh()
        with self._lock:
            for side in _SIDES:
                current, target = self._command[side], self._target[side]
                if current is None or target is None:
                    continue
                step = g1.HAND_MAX_RATE * min(max(float(dt), 0.0), 4.0 / 250.0)
                current = current + np.clip(target - current, -step, step)
                current = self._bounded_squeeze(side, current)
                self._command[side] = current
                self._publish(side, current)

    def _publish(self, side, reference):
        msg = self._new_command()
        for i in range(7):
            msg.motor_cmd[i].mode = 1
            msg.motor_cmd[i].q = float(reference[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = self._kp
            msg.motor_cmd[i].kd = self._kd
        if self._pub[side].Write(msg) is False:
            raise RuntimeError(f"Dex3 {side} publication failed")
        self._published[side] = reference.copy()

    def hold_tick(self):
        """Republish only applied references, with protective easing and no grip advance."""
        with self._lock:
            if self._hold_reference is None:
                self._hold_reference = {side: None if q is None else q.copy()
                                        for side, q in self._published.items()}
            for side, reference in self._hold_reference.items():
                if reference is not None:
                    self._target[side] = reference.copy()
        observation_error = None
        try:
            self.refresh()
        except Exception as exc:  # noqa: BLE001 -- retain the last valid bounded reference
            observation_error = f"{type(exc).__name__}: {exc}"
        published, errors = {}, {}
        with self._lock:
            for side, reference in self._hold_reference.items():
                try:
                    if reference is None:
                        raise RuntimeError("No successfully published hand reference")
                    safe = self._bounded_squeeze(side, reference.copy())
                    direction = np.sign(np.asarray(g1.HAND_CLOSED[side]) - np.asarray(g1.HAND_OPEN))
                    if np.any(direction * (safe - reference) > 1e-9):
                        raise RuntimeError("Held reference conflicts with current lead bounds; refusing tighter closure")
                    self._publish(side, safe)
                    self._hold_reference[side] = safe.copy()
                    self._command[side] = safe.copy()
                    self._target[side] = safe.copy()
                    published[side] = True
                except Exception as exc:  # noqa: BLE001 -- one failed hand cannot stop the other publisher
                    published[side] = False
                    errors[side] = f"{type(exc).__name__}: {exc}"
        return {"published": published, "observation_error": observation_error,
                "publication_errors": errors}

    def _tau_trusted(self, side: str, tau) -> bool:
        """Whether this hand's `tau_est` reads as newton-metres; reported once when not."""
        if bool((tau > _TAU_IMPLAUSIBLE * _TAU_RATING).any()):
            if not self._said_untrusted[side]:
                self._said_untrusted[side] = True
                print(f"[dex3] {side} hand: tau_est {np.round(tau, 1).tolist()} is not Nm "
                      f"against a {_TAU_RATING.tolist()} rating — the torque limit is off on "
                      f"this hand, the {self._lead_cap} rad lead cap still holds", flush=True)
            return False
        return True

    def _bounded_squeeze(self, side: str, command):
        """The command clipped within each finger's lead of the measured joints; a finger past
        its torque limit shrinks its lead."""
        measured = self._measured[side]
        if measured is None:
            return command
        tau, lead = self._tau[side], self._lead[side]
        if tau is not None and not self._tau_trusted(side, tau):
            tau = None
        if tau is not None:
            over = tau > self._tau_fraction * _TAU_RATING
            self._lead[side] = lead = np.where(
                over, np.maximum(lead - _LEAD_SHRINK_RAD, self._lead_min),
                np.minimum(lead + _LEAD_RESTORE_RAD, self._lead_cap))
            if over.any() and not self._said_over[side]:
                self._said_over[side] = True
                print(f"[dex3] {side} hand at its torque limit: "
                      f"tau {np.round(tau, 2).tolist()} Nm of {_TAU_RATING.tolist()}, "
                      f"lead now {np.round(lead, 3).tolist()} rad", flush=True)
            elif not over.any():
                self._said_over[side] = False
        lead = lead * self._thermal_ease(side)
        return np.clip(command, measured - lead, measured + lead)

    def _thermal_ease(self, side: str):
        """Per finger in [0, 1], the share of its lead left at this temperature: full below the
        ease temperature, zero at the rest temperature."""
        hot = self._hot[side]
        if hot is None or not ((_TEMP_PLAUSIBLE_C[0] <= hot) & (hot <= _TEMP_PLAUSIBLE_C[1])).all():
            return np.ones(7)
        rest, start = self._temp_rest_c, self._temp_ease_c
        ease = np.clip((rest - hot) / max(rest - start, 1e-6), 0.0, 1.0)
        if (ease < 1.0).any() and not self._said_hot[side]:
            self._said_hot[side] = True
            print(f"[dex3] {side} hand easing off: {hot.tolist()} C, squeeze now "
                  f"{np.round(100.0 * ease).astype(int).tolist()} % of its lead", flush=True)
        elif (ease >= 1.0).all():
            self._said_hot[side] = False
        return ease

    def state(self) -> dict:
        self.refresh(require_both=True)
        with self._lock:
            return {
                "closure_l": g1.grip_from_hand_q(self._measured["left"], "left"),
                "closure_r": g1.grip_from_hand_q(self._measured["right"], "right"),
                "held_l": self._held("left"),
                "held_r": self._held("right"),
                "stamp": min(self._stamp.values()),
            }

    def measured_state(self, *, require_both: bool = True) -> dict:
        """Raw timestamped hand evidence; tau_est units require deployment calibration."""
        self.refresh(require_both=require_both)
        with self._lock:
            return {side: {"joints": self._measured[side].copy(),
                           "target": self._target[side].copy(),
                           "tau_raw": None if self._tau[side] is None else self._tau[side].copy(),
                           "contact_state": ("contact" if self._held(side) is HELD else
                                             "empty" if self._held(side) is None else "unknown"),
                           "motor_faults": self._fault[side].copy(),
                           "captured_at": self._stamp[side]}
                    for side in _SIDES if self._measured[side] is not None and self._target[side] is not None}

    def _held(self, side: str):
        if self._latched[side] is not None:
            return HELD
        target = self._target[side]
        if target is None:
            return UNKNOWN
        if g1.grip_from_hand_q(target, side) < 0.05:
            return None
        measured = g1.grip_from_hand_q(self._measured[side], side)
        if self._latched_once[side]:
            # Verified on this close: empty only past the dropped closure.
            return None if measured > DEX3_DROPPED_CLOSURE else UNKNOWN
        # Closed near fully with no latch: empty.
        return None if measured > DEX3_EMPTY_CLOSURE else UNKNOWN

    def close(self) -> None:
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
            for thread in self._readers:
                thread.join(timeout=2.0 * _STATE_WAIT_S)
        for channel in (*self._pub.values(), *self._sub.values()):
            function = getattr(channel, "Close", None)
            if callable(function):
                function()
