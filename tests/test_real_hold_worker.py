"""An observation fault cannot silently terminate the healthy grip publisher."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_real_backend import ROOT, Clock, map_config

from homebody.backends.g1 import profile as g1
from homebody.backends.g1.dex3 import Dex3Port
from homebody.backends.g1.task_motion import G1HostMotion
from homebody.primitives.actions import ActionRejected


def hands():
    port = Dex3Port.__new__(Dex3Port)
    port._init_state()
    port.refresh = lambda **kwargs: None
    port._new_command = lambda: SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
    sent = {side: [] for side in ("left", "right")}
    port._pub = {side: SimpleNamespace(Write=lambda msg, side=side:
                                     sent[side].append(np.array([motor.q for motor in msg.motor_cmd])))
                 for side in sent}
    for side in sent:
        reference = np.asarray(g1.HAND_CLOSED[side]) * .35
        port._measured[side] = reference.copy()
        port._command[side] = reference.copy()
        port._target[side] = np.asarray(g1.HAND_CLOSED[side]) * .8
        port._publish(side, reference)
    return port, sent


def test_hold_tick_applies_thermal_protection_without_reclosing_after_cooling():
    port, sent = hands()
    before = sent["right"][-1].copy()
    direction = np.sign(np.asarray(g1.HAND_CLOSED["right"]) - np.asarray(g1.HAND_OPEN))
    port._measured["right"] = before - direction * .08
    port._hot["right"] = np.full(7, port._temp_rest_c)
    status = port.hold_tick()
    assert all(status["published"].values())
    eased = sent["right"][-1].copy()
    assert np.all(direction * (eased - before) <= 0)
    np.testing.assert_allclose(eased, port._measured["right"])
    port._hot["right"][:] = 20
    port.hold_tick()
    np.testing.assert_array_equal(sent["right"][-1], eased)


class BoundedStop:
    def __init__(self, count):
        self.count, self.calls = count, 0
    def wait(self, duration):
        self.calls += 1
        return self.calls > self.count
    def set(self):
        self.count = 0


@pytest.mark.parametrize("failure", ["evidence", "publication", "base_hold"])
def test_worker_latches_fault_and_keeps_hold_only_publication(tmp_path, failure):
    clock = Clock()
    port, sent = hands()
    base_calls = []
    def send(**command):
        base_calls.append(command)
        if failure == "base_hold":
            raise OSError("base transport unavailable")
    body = SimpleNamespace(state={"q": np.zeros(29), "t": clock(), "tick": 1,
                                  "active": True, "ramping": False, "link_live": True},
                           claim="ours", interruptions=0, wait_fresh=lambda _: True,
                           require_fresh=lambda: None, send=send,
                           close=lambda: None)
    motion = G1HostMotion(map_config(tmp_path), json.loads((ROOT/'assets/robot/calibration.json').read_text()),
                          body=body, hands=port, lease=SimpleNamespace(release=lambda: None),
                          clock=clock, monotonic=clock, sleep=clock.sleep, start=False)
    motion._stop = BoundedStop(4)
    normal_ticks = []
    tick = port.tick
    def normal_tick(duration):
        normal_ticks.append(duration)
        tick(duration)
    port.tick = normal_tick
    evidence_calls = []
    def record(**kwargs):
        evidence_calls.append(True)
        raise RuntimeError("transient evidence bookkeeping failure")
    motion._record_hands = record
    if failure == "publication":
        port._pub["left"].Write = lambda msg: False
    motion._hand_loop()
    assert len(normal_ticks) == 1
    assert len(base_calls) == 4
    assert len(sent["right"]) >= 5
    assert motion._fault is not None
    with pytest.raises(ActionRejected, match="command loop failed"):
        motion.grip("right", 1.)
    status = motion.command_status()
    assert status["fault"] and status["transitions"]
    assert status["hand_reference_published"]["right"]
    if failure == "publication":
        assert status["hand_publication_errors"]["left"]
    if failure == "base_hold":
        assert "base transport unavailable" in status["base_hold_error"]
    frozen = status["transitions"][0]["hand_reference_published"]
    frozen["right"] = False
    assert motion.command_status()["transitions"][0]["hand_reference_published"]["right"]
    right_after_first = sent["right"][1]
    for reference in sent["right"][2:]:
        np.testing.assert_array_equal(reference, right_after_first)
