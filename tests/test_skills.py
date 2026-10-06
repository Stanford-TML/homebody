"""The skill framework: frames, refusals, cancellation, resets and backend errors, with fakes."""
from dataclasses import replace

import numpy as np
import pytest

from homebody.primitives.observations import Frame, HandState, NavigationMap, ObservationError
from homebody.skills.contract import Context, Runner


def frame(epoch=1, sequence=1, time=0.0):
    return Frame(epoch, sequence, time, np.zeros((6, 8, 3), dtype=np.uint8),
                 np.ones((6, 8)), np.array([[10, 0, 4], [0, 10, 3], [0, 0, 1]]),
                 np.eye(4), np.eye(4), np.array([1.0, 1.0, 0.0]), np.zeros(29),
                 {s: HandState(np.zeros(7), 0, np.zeros(3), np.zeros(3)) for s in ("left", "right")},
                 np.ones((6, 8), dtype=int),
                 NavigationMap(np.zeros((40, 40), dtype=bool), np.diag([0.1, 0.1, 1]), 0.1))


class FakeBackend:
    def __init__(self, initial=None):
        self.frame = initial or frame()
        self.commands = []
        self.stopped = 0
        self.velocity = np.zeros(3)
        self.reset_on_advance = False
        self.freeze_pose = False

    @property
    def epoch(self):
        return self.frame.epoch

    @property
    def time(self):
        return self.frame.time

    def measure(self):
        return self.snapshot()

    def snapshot(self):
        return self.frame

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        assert epoch == self.epoch
        self.commands.append((forward, lateral, yaw_rate))
        self.velocity = np.array([forward, lateral, yaw_rate])

    def advance(self, epoch, seconds):
        assert epoch == self.epoch
        x, y, yaw = self.frame.base_pose
        forward, lateral, turn = self.velocity
        pose = [x + seconds * (forward * np.cos(yaw) - lateral * np.sin(yaw)),
                y + seconds * (forward * np.sin(yaw) + lateral * np.cos(yaw)),
                yaw + turn * seconds]
        self.frame = replace(self.frame, time=self.time + seconds,
                             epoch=self.epoch + int(self.reset_on_advance),
                             sequence=self.frame.sequence + 1,
                             base_pose=self.frame.base_pose if self.freeze_pose else pose)

    def arm_target(self, epoch, side, joints):
        assert epoch == self.epoch

    def stop(self, epoch):
        assert epoch == self.epoch
        self.velocity[:] = 0
        self.stopped += 1


def run(backend, name, args, **context):
    events = []
    result = Runner(Context(backend, backend, {}, emit=events.append, **context)).run(
        name, args, backend.frame)
    terminals = [event for event in events if event["type"] == "skill_finished"]
    assert len(terminals) == 1
    assert terminals[0]["result"] == result
    return result


def test_frame_is_owned_and_readonly():
    original = frame()
    with pytest.raises(ValueError):
        original.depth[0, 0] = 7
    with pytest.raises(TypeError):
        original.hands["right"] = None
    assert original.select(2, 2) == 1
    background = replace(original, labels=np.zeros((6, 8), dtype=int))
    with pytest.raises(ObservationError, match="background"):
        background.select(2, 2)


def test_preparation_refusal_has_no_actuation():
    backend = FakeBackend()
    result = run(backend, "navigate", {"x": -100, "y": 0})
    assert result.code == "NO_ROUTE"
    assert backend.commands == []
    assert backend.stopped == 0


def test_cancel_before_preparation_no_actuation():
    backend = FakeBackend()
    result = run(backend, "navigate", {"x": 2, "y": 1}, cancelled=lambda: True)
    assert result.code == "CANCELLED"
    assert not backend.commands


def test_reset_during_execution_stops_old_generation():
    """No old command, not even a stop, is sent into the new generation."""
    backend = FakeBackend()
    backend.reset_on_advance = True
    result = run(backend, "navigate", {"x": 2, "y": 1})
    assert result.code == "STALE_EPOCH"
    assert len(backend.commands) == 1
    assert backend.stopped == 0


@pytest.mark.parametrize("value", [10**1000, float("inf"), float("nan")])
def test_numeric_overflow_is_a_declared_refusal(value):
    backend = FakeBackend()
    assert run(backend, "navigate", {"x": value, "y": 1}).code == "INVALID_ARGUMENT"
    with pytest.raises(ObservationError):
        backend.frame.normalized_pixel([value, 0])
    assert not backend.commands


def test_unexpected_backend_error_propagates_with_one_error_terminal():
    class BrokenBackend(FakeBackend):
        def advance(self, epoch, seconds):
            raise RuntimeError("programming failure")

    backend, events = BrokenBackend(), []
    runner = Runner(Context(backend, backend, {}, emit=events.append))
    with pytest.raises(RuntimeError, match="programming failure"):
        runner.run("navigate", {"x": 2, "y": 1}, backend.frame)
    results = [e["result"] for e in events if e["type"] == "skill_finished"]
    assert len(results) == 1 and results[0].code == "ERROR"
    assert backend.stopped == 1


def test_stale_current_observation_refuses_motion():
    class StaleBackend(FakeBackend):
        @property
        def time(self):
            return self.frame.time + 2.0

    backend = StaleBackend()
    assert run(backend, "navigate", {"x": 2, "y": 1}).code == "STALE_OBSERVATION"
    assert not backend.commands


def test_normalized_click_converts_once_on_captured_dimensions():
    captured = frame()
    assert captured.normalized_pixel([0, 0]) == (0, 0)
    assert captured.normalized_pixel([1000, 1000]) == (7, 5)
    assert captured.normalized_pixel([500, 500]) == (3, 2)
    for bad in ([True, 500], [1001, 500], [0, float("nan")], "500,500"):
        with pytest.raises(ObservationError):
            captured.normalized_pixel(bad)
