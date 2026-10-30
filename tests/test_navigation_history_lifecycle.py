"""Moved-object history cannot revive when verified carry ownership ends."""
from dataclasses import replace

import numpy as np
import pytest
from test_navigation_observation_history import CountedBackend, registered_frame

from homebody.skills.contract import Context
from homebody.skills.navigate_real import task


def packet(time, *, acquired=None, shift=0.):
    frame = registered_frame(time=time)
    labels = frame.labels.copy()
    labels[0, 0] = 62
    camera = frame.world_camera.copy()
    camera[1, 3] += shift
    metadata = {**frame.sensor_metadata, "depth_time": time if acquired is None else acquired}
    return replace(frame, labels=labels, world_camera=camera, sensor_metadata=metadata)


@pytest.mark.parametrize("record", [
    {"state": "uncertain", "label": 62, "epoch": 1},
    {"state": "pending", "label": 62, "epoch": 1},
    {"state": "verified", "label": 62, "epoch": 2},
])
def test_unverified_or_foreign_carry_cannot_retire_observed_hits(monkeypatch, record):
    monkeypatch.setattr(task, "robot_boxes", lambda *_: [])
    robot = CountedBackend()
    ctx = Context(robot, robot, {}, navigation=task)
    task.observe_frame(ctx, packet(0.))
    history = ctx.navigation_history
    ghost = history.points(0., history.binding)[0]
    ctx.carried["right"] = record
    task.observe_frame(ctx, packet(.5))
    assert np.any(np.all(history.points(.5, history.binding) == ghost, axis=1))
