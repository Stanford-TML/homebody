"""Shared manipulation checks under strict and default task policies."""
from dataclasses import replace

import numpy as np
import pytest
from fakes import PlanningArm
from test_skills import FakeBackend, frame

from homebody.motion.carry import CarryState
from homebody.primitives.observations import HandState
from homebody.skills import place
from homebody.skills.contract import Context, Runner
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.registry import task_registry


def carried_frame():
    current = frame()
    hands = dict(current.hands)
    hands["right"] = HandState(np.zeros(7), .8, np.ones(3), np.ones(3))
    return replace(current, hands=hands)


def carry():
    return {"right": {"label": 1, "state": CarryState.VERIFIED,
                      "palm_to_payload": np.eye(4), "payload_half": np.full(3, .05)}}


class NoReachArm(PlanningArm):
    def solve(self, *args, **kwargs):
        return None


def test_real_unreachable_place_refuses_without_opening_but_sim_keeps_its_fallback():
    robot = FakeBackend(carried_frame())
    robot.grip = lambda *args: pytest.fail("Unreachable placement must never open a hand")
    ctx = Context(robot, robot, {"right": NoReachArm(), "left": NoReachArm()}, carried=carry(),
                  manipulation=REAL_MANIPULATION)
    args = {"x_ahead_m": 3., "y_left_m": 0.}
    real = Runner(ctx, task_registry()).run("place", args, robot.frame)
    assert real.code == "UNREACHABLE"
    assert robot.commands == []
    default = Context(robot, robot, {"right": NoReachArm(), "left": NoReachArm()}, carried=carry())
    plan = place.prepare(args, robot.frame, robot.frame, default.planning())
    assert plan.pre_joints is None
    assert np.allclose(plan.point[:2], [3., 0.])


def test_real_pick_cannot_start_second_object_or_overwrite_track():
    robot = FakeBackend(carried_frame())
    robot.grip = lambda *args: pytest.fail("Second pick cannot open a hand")
    ctx = Context(robot, robot, {}, carried=carry(), manipulation=REAL_MANIPULATION,
                  target_selector=lambda *args: pytest.fail("Do not replace the carried target's track"))
    result = Runner(ctx, task_registry()).run("pick", {"hand": "left", "point_normalized_1000": [500, 500]}, robot.frame)
    assert result.code == "HAND_OCCUPIED"
    assert robot.commands == []


@pytest.mark.parametrize("side", ["left", "right"])
def test_new_pick_cannot_open_or_ignore_an_unrecorded_closed_hand(side):
    robot = FakeBackend(carried_frame())
    robot.grip = lambda *args: pytest.fail("An unresolved hand cannot be opened by a new pick")
    ctx = Context(robot, robot, {}, manipulation=REAL_MANIPULATION,
                  target_selector=lambda *args: pytest.fail("Refuse before replacing a target track"))
    result = Runner(ctx, task_registry()).run("pick", {"hand": side, "point_normalized_1000": [500, 500]}, robot.frame)
    assert result.code == "HOLD_UNCERTAIN"
    assert robot.commands == []
