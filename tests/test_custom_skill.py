"""A complete example skill: wave one empty hand.

Everything above the fake robot is what `skills/wave.py` would hold. The tests register this
module the way registry.py registers navigate, pick and place, and run it through the
production Runner on a fake robot that implements primitives.Robot and nothing more.
"""
import re
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.primitives.actions import ActionRejected
from homebody.primitives.observations import ARM, Frame, HandState, NavigationMap
from homebody.primitives.robot import Robot
from homebody.session.settings import Settings
from homebody.skills.contract import Context, Result, Runner, SkillFailure, declared, side_argument
from homebody.skills.registry import PROMPTS, REGISTRY, SKILLS

NEEDS = ("arm_target", "advance")
FAILURES = ("HAND_OCCUPIED", "ARM_NOT_REACHED")
PROMPT = ('wave: hand "right" or "left". Raises that empty hand, swings it {wave.swings} times '
          'and lowers it to where it started; refused while the hand holds an object.')
SHOULDER_PITCH, WRIST_ROLL = 0, 4


@dataclass(frozen=True)
class Wave:
    """The skill's tunables, as a group in session/config.py with a `wave` field on Settings."""
    lift: float = 1.0
    swing: float = 0.4
    swings: int = 2
    tolerance: float = 0.05
    timeout: float = 4.0


@dataclass(frozen=True)
class WaveSettings(Settings):
    wave: Wave = field(default_factory=Wave)


@dataclass(frozen=True)
class Plan:
    epoch: int
    side: str
    poses: tuple


def prepare(arguments, selected, current, planning):
    side = side_argument(arguments)
    if side in planning.carried:
        raise SkillFailure("HAND_OCCUPIED", f"The {side} hand holds an object; place it first")
    settings, start = planning.settings.wave, current.arm_joints(side).copy()
    raised = start.copy()
    raised[SHOULDER_PITCH] -= settings.lift
    out, back = raised.copy(), raised.copy()
    out[WRIST_ROLL] += settings.swing
    back[WRIST_ROLL] -= settings.swing
    swings = [("Waving", pose) for _ in range(settings.swings) for pose in (out, back)]
    return Plan(current.epoch, side, (("Raising arm", raised), *swings, ("Lowering arm", start)))


def execute(plan, ctx):
    for stage, joints in plan.poses:
        ctx.stage(stage)
        reach(plan, ctx, joints)
    return Result("wave", "OK", f"Waved the {plan.side} hand and lowered it", {"side": plan.side})


def abort(plan, ctx):
    """Lower the arm again after a failure; the failure's own code stands."""
    ctx.stage("Lowering arm")
    reach(plan, ctx, plan.poses[-1][1])


def reach(plan, ctx, joints):
    """Hold JOINTS as the arm target until the measured arm is within tolerance of it."""
    settings = ctx.settings.wave
    frame = ctx.observe(plan.epoch)
    deadline = frame.time + settings.timeout
    while np.abs(frame.arm_joints(plan.side) - joints).max() > settings.tolerance:
        if frame.time >= deadline:
            raise SkillFailure("ARM_NOT_REACHED", f"The {plan.side} arm stopped short of its pose")
        ctx.actions.arm_target(plan.epoch, plan.side, joints)
        ctx.advance(plan.epoch)
        frame = ctx.observe(plan.epoch)


class FakeRobot:
    """primitives.Robot and nothing more: the arms move toward their targets at RATE rad/s,
    and no joint goes below FLOOR, as if an arm met an obstacle there."""

    def __init__(self, rate=1.0, floor=-np.inf):
        self.epoch, self.time, self.sequence = 1, 0.0, 0
        self.joints, self.targets = np.zeros(29), np.zeros(29)
        self.rate, self.floor = rate, floor
        self.commands, self.trace = [], []

    def measure(self):
        return self.snapshot()

    def snapshot(self):
        self.sequence += 1
        hand = HandState(np.zeros(7), 0.0, np.zeros(3), np.zeros(3))
        return Frame(self.epoch, self.sequence, self.time, np.zeros((4, 4, 3), np.uint8),
                     np.ones((4, 4)), np.eye(3), np.eye(4), np.eye(4), np.zeros(3), self.joints,
                     {"left": hand, "right": hand}, np.zeros((4, 4), int),
                     NavigationMap(np.zeros((4, 4), bool), np.eye(3), 0.1))

    def command(self, name, epoch):
        if epoch != self.epoch:
            raise ActionRejected("Command belongs to an expired execution generation")
        self.commands.append(name)

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.command("base_velocity", epoch)

    def arm_target(self, epoch, side, joints):
        self.command("arm_target", epoch)
        self.targets[ARM[side]] = joints

    def grip(self, epoch, side, closure):
        self.command("grip", epoch)

    def advance(self, epoch, seconds):
        self.command("advance", epoch)
        step = np.clip(self.targets - self.joints, -self.rate * seconds, self.rate * seconds)
        self.joints = np.maximum(self.joints + step, self.floor)
        self.trace.append(self.joints)
        self.time += seconds

    def stop(self, epoch):
        self.command("stop", epoch)
        self.targets = self.joints.copy()


@pytest.fixture
def wave(monkeypatch):
    """This module, registered as `wave` through the check registry.py applies to every skill."""
    monkeypatch.setitem(REGISTRY, "wave", declared("wave", sys.modules[__name__]))


def run(robot, arguments, **context):
    events = []
    runner = Runner(Context(robot, robot, {}, WaveSettings(), emit=events.append, **context))
    result = runner.run("wave", arguments, robot.snapshot())
    return result, [event["stage"] for event in events if event["type"] == "skill_stage"]


def test_a_registered_skill_is_offered_to_the_model(wave):
    assert PROMPTS["wave"] == PROMPT
    assert tuple(PROMPTS) == (*SKILLS, "wave")
    assert PROMPT in "\n".join(PROMPTS.values())


def test_the_runner_raises_swings_and_lowers_the_hand(wave):
    robot, settings = FakeRobot(), Wave()
    assert isinstance(robot, Robot)
    result, stages = run(robot, {"hand": "left"})
    assert result == Result("wave", "OK", "Waved the left hand and lowered it", {"side": "left"})
    assert stages == ["Preparing", "Raising arm", "Waving", "Lowering arm"]
    left, tolerance = np.array(robot.trace)[:, ARM["left"]], settings.tolerance
    assert left[:, SHOULDER_PITCH].min() == pytest.approx(-settings.lift, abs=tolerance)
    assert np.ptp(left[:, WRIST_ROLL]) == pytest.approx(2 * settings.swing, abs=2 * tolerance)
    assert np.abs(robot.joints).max() <= settings.tolerance
    assert set(robot.commands[:-1]) == set(NEEDS) and robot.commands[-1] == "base_velocity"


def test_a_hand_that_holds_an_object_is_refused_before_anything_moves(wave):
    robot = FakeRobot()
    result, stages = run(robot, {"hand": "right"}, carried={"right": {"state": "verified"}})
    assert result.code == "HAND_OCCUPIED" and stages == ["Preparing"]
    assert robot.commands == []


def test_a_blocked_arm_ends_with_its_declared_code_and_is_lowered_again(wave):
    robot = FakeRobot(floor=-0.3)
    result, stages = run(robot, {"hand": "right"})
    assert result.code == "ARM_NOT_REACHED" and result.message.endswith("stopped short of its pose")
    assert stages == ["Preparing", "Raising arm", "Lowering arm"]
    assert np.abs(robot.joints).max() <= Wave().tolerance
    assert robot.commands[-1] == "stop"


def test_a_robot_without_a_needed_command_refuses_the_skill_before_it_moves(wave):
    class Armless(FakeRobot):
        arm_target = None

    robot = Armless()
    result, _ = run(robot, {"hand": "right"})
    assert result.code == "ACTION_REJECTED" and result.message.endswith("arm_target")
    assert robot.commands == []


@pytest.mark.parametrize("change, problem", [
    ({"NEEDS": ("arm_target", "wave_hand")}, "NEEDS must be a tuple of Robot commands"),
    ({"NEEDS": {"actions": ("arm_target",)}}, "NEEDS must be a tuple of Robot commands"),
    ({"FAILURES": "ARM_NOT_REACHED"}, "FAILURES must be a tuple"),
    ({"PROMPT": "greet: {}"}, "PROMPT must start with 'wave: '"),
    ({"execute": None}, "does not declare execute"),
])
def test_registration_names_what_a_skill_declares_wrongly(change, problem):
    declarations = {"NEEDS": NEEDS, "FAILURES": FAILURES, "PROMPT": PROMPT,
                    "prepare": prepare, "execute": execute, **change}
    present = {key: value for key, value in declarations.items() if value is not None}
    with pytest.raises(TypeError, match=re.escape(problem)):
        declared("wave", SimpleNamespace(**present))


def test_the_skill_settings_group_loads_and_validates_like_the_others(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[wave]\nlift = 0.8\nswings = 3\n")
    assert WaveSettings.load(path).wave == Wave(lift=0.8, swings=3)
    with pytest.raises(ValueError, match="wave.swing must contain positive"):
        WaveSettings(wave=Wave(swing=-0.4))
