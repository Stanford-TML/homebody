"""Intermediate route points may be passed only through a checked onward connector."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.motion.exploration.gait import occupancy_grid
from homebody.motion.exploration.localizer import PoseGate
from homebody.primitives.navigation.context import Ctx, Deadline
from homebody.real2sim.exploration import settings
from homebody.skills.navigate_real import controller


@pytest.fixture
def rig(monkeypatch):
    configured = settings.load(Path(__file__).resolve().parents[1])
    monkeypatch.setitem(settings._installed, "settings", configured)
    occupied = np.zeros((201, 301), bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    art = SimpleNamespace(occupancy=occupied,
                          T_map_px=np.array([[.02, 0., -2.], [0., .02, -2.], [0., 0., 1.]]))
    state = SimpleNamespace(time=0., pose=np.array([0., 0., 0.]), events=[], commands=[],
                            on_step=lambda _: None)

    def observe(term):
        return {"base_pose": state.pose.copy(), "map": art, "loc_quality": 1.,
                "gripper_state": SimpleNamespace(held_l=None, held_r=None)}[term]

    observe._owns_pose_consistency = True

    def step(channel, command):
        assert channel == "lower_body"
        state.commands.append(command)
        vx, vy, wz = command.vx or 0., command.vy or 0., command.wz or 0.
        yaw = state.pose[2]
        state.pose += .004 * np.array([vx * math.cos(yaw) - vy * math.sin(yaw),
                                       vx * math.sin(yaw) + vy * math.cos(yaw), wz])
        state.time = round(state.time + .004, 9)
        state.on_step(command)

    ctx = Ctx(observe, step, lambda: state.time, Deadline(lambda: state.time, 20.),
              lambda _: None, lambda kind, data: state.events.append((kind, data)), PoseGate())
    return state, art, ctx


@pytest.mark.parametrize("straight", [False, True])
def test_blocked_shortcut_is_retained_until_onward_footprint_fits(rig, straight):
    state, art, ctx = rig
    if straight:
        state.pose[:] = (1.01, -.23, 0.)
        waypoint, following, start = (1., 0.), (2., 0.), (0., 0.)
        art.occupancy[82, 171] = True  # (1.42, -.36): near the robot's off-line shortcut.
    else:
        state.pose[:] = (-.30, 0., 0.)
        waypoint, following, start = (0., 0.), (0., 1.), (-1., 0.)
        art.occupancy[122, 79] = True  # (-.42, .44): near the diagonal corner shortcut.
    checker, _ = occupancy_grid(art, .30, exact_footprint=True)
    assert controller._line_free(checker, state.pose[:2], waypoint)
    assert controller._line_free(checker, waypoint, following)
    assert not controller._line_free(checker, state.pose[:2], following)
    previous = [tuple(state.pose[:2])]

    def checked_step(_):
        xy = tuple(state.pose[:2])
        assert controller._line_free(checker, previous[0], xy)
        previous[0] = xy

    state.on_step = checked_step
    controller._walk_waypoints(ctx, (waypoint,), {"L": None, "R": None}, 0,
                               checker=checker, preserve_goal=True, goal_xy=following,
                               route_start=start, radius_m=.30, clearance_margin_m=.15)
    assert any(c.vx for c in state.commands)
    assert controller._line_free(checker, state.pose[:2], following)
    assert not ctx.deadline.expired()
