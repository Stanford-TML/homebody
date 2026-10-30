"""Columns reuse must skip allocation without retaining expired or changed evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.helpers.volume import ColumnHistory
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task


def setup_bridge(monkeypatch):
    binding = (1, "room", "map", 1, "calibration")
    history = ColumnHistory(seconds=1., period=0.)
    history.add(.05, binding, np.array([[1., 1., .4], [2., 2., .8]]), np.array([0, 7]))
    frame = SimpleNamespace(epoch=1, time=1., sensor_metadata={})
    outer = SimpleNamespace(navigation_history=history, observations=SimpleNamespace(time=1.),
                            observe=lambda epoch: frame)
    planning = SimpleNamespace(arms={}, carried={}, settings=SimpleNamespace(
        motion=SimpleNamespace(observation_age=.25),
        collision=SimpleNamespace(column_margin=.01, column_spacing=.02)))
    bridge = task._Bridge(planning, frame, outer)
    bridge._last_look = bridge._sample_read = 1.
    sample = SimpleNamespace(epoch=1, frame_id="room", map_hash="map", generation=1,
                             obstacles=np.empty((0, 3)))
    state = SimpleNamespace(frame_id=(1, 1), boxes=[])
    monkeypatch.setattr(task, "robot_boxes", lambda measured, *_: measured.boxes)
    calls, original = [], history.points

    def materialize(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(history, "points", materialize)
    return bridge, sample, state, history, calls


def test_binding_mismatch_refuses_before_materialization(monkeypatch):
    bridge, sample, state, history, calls = setup_bridge(monkeypatch)
    bridge.columns(sample, state)
    history.add(1., (1, "other-room", "map", 1, "calibration"),
                np.array([[1., 1., .4]]), np.array([0]))
    with pytest.raises(SkillFailure, match="Support history and navigation frames changed"):
        bridge.columns(sample, state)
    assert bridge.outer.navigation_history is None and len(calls) == 1


def box(point):
    return (np.array(point, float), np.eye(3), np.full(3, .05))
