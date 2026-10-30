"""Pre-grasp REAL walk-in failures retain their checked posture and failure result."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_skills import FakeBackend

from homebody.skills import pick
from homebody.skills.contract import Context, Result, Runner
from homebody.skills.manipulation_policy import REAL_MANIPULATION


class PickBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.arm_commands = []
        self.grip_commands = []

    def arm_target(self, epoch, side, joints):
        super().arm_target(epoch, side, joints)
        self.arm_commands.append((side, np.asarray(joints).copy()))

    def grip(self, epoch, side, closure):
        assert epoch == self.epoch
        self.grip_commands.append((side, closure))


def setup_attempt(monkeypatch, *, real=True, robot=None, navigation_result=None):
    robot, events, resets = robot or PickBackend(), [], []
    navigation_result = navigation_result or Result(
        "navigate", "ARRIVAL_NOT_VERIFIED", "settled outside the accepted stance",
        {"distance_m": .123, "arrival_distance_m": .1})
    navigation = SimpleNamespace(execute=lambda plan, ctx: navigation_result)
    policy = {"manipulation": REAL_MANIPULATION} if real else {}
    ctx = Context(robot, robot, {}, emit=events.append, navigation=navigation, **policy)
    plan = pick.ApproachPlan(robot.epoch, "right", 1, object())
    monkeypatch.setattr(pick, "prepare", lambda *args: plan)
    monkeypatch.setattr(pick, "reset_arm", lambda ctx, epoch, side: resets.append(side) or True)
    runner = Runner(ctx, {"pick": pick})
    return robot, ctx, runner, events, resets


def terminals(events):
    return [event["result"] for event in events if event["type"] == "skill_finished"]


def test_real_failed_walk_in_keeps_failure_and_never_resets_or_grasps(monkeypatch):
    robot, _, runner, events, resets = setup_attempt(monkeypatch)
    monkeypatch.setattr(pick, "grasp", lambda *args: pytest.fail("Grasp must not start"))
    result = runner.run("pick", {}, robot.frame)
    assert result.code == "ARRIVAL_NOT_VERIFIED"
    assert result.message == "settled outside the accepted stance"
    assert result.details == {"distance_m": .123, "arrival_distance_m": .1,
                              "pick_phase": "walk_in", "grasp_started": False}
    assert resets == [] and robot.arm_commands == [] and robot.grip_commands == []
    assert robot.stopped == 1 and not robot.velocity.any()
    assert terminals(events) == [result]
