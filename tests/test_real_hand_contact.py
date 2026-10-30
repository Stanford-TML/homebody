"""Physical position contact passes through the REAL composition without fabricated effort."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_real_backend import ROOT, Clock

from homebody.backends.g1.task_motion import G1HostMotion, measured_hand
from homebody.motion.carry import Evidence, evidence
from homebody.primitives.observations import ObservationError
from homebody.session.settings import Settings

CALIBRATION = json.loads((ROOT / "assets/robot/calibration.json").read_text())


def native_hand(state, closure=.6):
    ray = np.asarray(CALIBRATION["hand_closed"]["right"])
    return measured_hand({"joints": ray * closure, "target": ray,
                          "tau_raw": np.full(7, 1674.6), "contact_state": state,
                          "motor_faults": np.zeros(7, dtype=int)}, "right", CALIBRATION, None)


def test_real_carry_requires_visible_lift_not_only_persistent_contact():
    points = np.array([[-.01, 0, 1.], [.01, 0, 1.], [0, .01, 1.]])
    palm = np.eye(4)
    record = {"label": 1, "local_points": points.copy(), "palm_to_payload": np.eye(4),
              "payload_half": np.array([.05, .05, 1.1]), "baseline_upper": 1.,
              "baseline_palm_height": 0., "lift_verified": False}
    current = SimpleNamespace(hands={"right": native_hand("contact")}, visible=lambda label: True,
                              target_points=lambda label: points, palm=lambda arm, side: palm)
    settings = Settings().grasp
    assert evidence(current, None, "right", record, settings) == Evidence.ENCLOSED
    current.visible = lambda label: False
    assert evidence(current, None, "right", record, settings) == Evidence.UNKNOWN
    current.visible = lambda label: True
    points[:, 2] += .2
    palm[2, 3] += .2
    assert evidence(current, None, "right", record, settings) == Evidence.HELD
    current.hands["right"] = native_hand("unknown")
    assert evidence(current, None, "right", record, settings) == Evidence.UNKNOWN
    current.hands["right"] = native_hand("empty", .99)
    assert evidence(current, None, "right", record, settings) == Evidence.LOST


def test_hand_adapter_refuses_stale_or_faulted_native_samples(tmp_path):
    clock = Clock()
    row = {"joints": np.zeros(7), "target": np.zeros(7), "tau_raw": None,
           "contact_state": "empty", "motor_faults": np.zeros(7), "captured_at": clock()}
    motion = G1HostMotion.__new__(G1HostMotion)
    import threading
    from collections import deque
    motion._lock, motion.clock = threading.RLock(), clock
    motion.calibration, motion.effort_calibration = CALIBRATION, None
    motion._hand_history = {side: deque() for side in ("left", "right")}
    motion.hands = SimpleNamespace(measured_state=lambda **kw: {side: dict(row) for side in motion._hand_history})
    states, _ = motion.hand_states()
    assert not states["right"].retained(Settings().grasp)
    clock.sleep(.2)
    with pytest.raises(ObservationError, match="stale"):
        motion.hand_states()
    row["captured_at"] = clock()
    row["motor_faults"] = np.ones(7)
    with pytest.raises(ObservationError, match="motor fault"):
        motion.hand_states()
