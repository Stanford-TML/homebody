"""Checked navigation consumes accepted observations through the shared skill context."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_skills import FakeBackend, frame

from homebody.skills.contract import Context, SkillFailure
from homebody.skills.navigate_real import task as navigation_task


class CountedBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.reads = 0
        self.current_time = 0.
        self.packet = self.frame
        self.after_read = lambda: None

    @property
    def time(self):
        return self.current_time

    def snapshot(self):
        self.reads += 1
        self.after_read()
        return self.packet


@pytest.mark.parametrize("failure", ["stale", "future", "packet_epoch", "cancel_before", "cancel_during"])
def test_unaccepted_observation_never_enters_navigation_history(failure):
    robot, calls, stopped = CountedBackend(), [], []
    if failure == "stale":
        robot.current_time = 5.
    elif failure == "future":
        robot.packet = frame(time=1.)
    elif failure == "packet_epoch":
        robot.packet = frame(epoch=2)
    elif failure == "cancel_before":
        stopped.append(True)
    else:
        robot.after_read = lambda: stopped.append(True)
    ctx = Context(robot, robot, {}, cancelled=lambda: bool(stopped),
                  navigation=SimpleNamespace(observe_frame=lambda *_: calls.append("history")),
                  emit=lambda _: calls.append("event"))
    with pytest.raises(SkillFailure) as raised:
        ctx.observe(1)
    assert raised.value.code == ("CANCELLED" if failure.startswith("cancel") else "STALE_OBSERVATION")
    assert not calls and ctx.navigation_history is None
    assert robot.reads == (0 if failure == "cancel_before" else 1)
    assert not robot.commands


def registered_frame(**changes):
    packet = frame()
    camera = np.eye(4)
    camera[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    camera[:3, 3] = [1., 1., .5]
    meta = {"registered_pose": True, "navigation_generation": (1, 1),
            "map_frame": "test-room", "map_sha256": "test-map-v1", "calibration_hash": "cal-v1"}
    return replace(packet, world_camera=camera, sensor_metadata=meta, **changes)


@pytest.mark.parametrize("missing", ["registered_pose", "navigation_generation", "map_frame",
                                     "map_sha256", "calibration_hash"])
def test_actual_hook_discards_history_without_acquisition_provenance(monkeypatch, missing):
    robot = CountedBackend()
    packet = registered_frame()
    monkeypatch.setattr(navigation_task, "robot_boxes", lambda *_: [])
    ctx = Context(robot, robot, {}, navigation=navigation_task)
    navigation_task.observe_frame(ctx, packet)
    assert ctx.navigation_history is not None
    metadata = dict(packet.sensor_metadata)
    del metadata[missing]
    navigation_task.observe_frame(ctx, replace(packet, sensor_metadata=metadata))
    assert ctx.navigation_history is None and robot.reads == 0
