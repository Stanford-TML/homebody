"""Task navigation through declared Robot observations, including interrupted admission."""
from dataclasses import replace

import numpy as np
import pytest
from test_skills import FakeBackend

from homebody.primitives.navigation.sim_task import SimNavigationEvidence
from homebody.skills.contract import Context, Runner, SkillFailure
from homebody.skills.navigate_real import task as navigate_real
from homebody.skills.registry import task_registry


class RobotSurface:
    """A robot boundary with explicit commands and no simulator/evaluator access."""

    def __init__(self):
        self.robot = FakeBackend()
        self.grips = []

    def __getattr__(self, name):
        raise AssertionError(f"Navigation accessed undeclared robot state: {name}")

    @property
    def epoch(self):
        return self.robot.epoch

    @property
    def time(self):
        return self.robot.time

    def snapshot(self):
        return self.robot.snapshot()

    def measure(self):
        return self.robot.measure()

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.robot.base_velocity(epoch, forward, lateral, yaw_rate)

    def arm_target(self, epoch, side, joints):
        self.robot.arm_target(epoch, side, joints)

    def grip(self, epoch, side, closure):
        self.grips.append((epoch, side, closure))

    def advance(self, epoch, seconds):
        self.robot.advance(epoch, seconds)

    def stop(self, epoch):
        self.robot.stop(epoch)


def task_navigation():
    surface = RobotSurface()
    selected = surface.snapshot()
    sensors = SimNavigationEvidence(surface, selected.navigation)
    events = []
    context = Context(surface, surface, {}, navigation=navigate_real,
                      navigation_evidence=sensors, emit=events.append)
    return surface, context, Runner(context, task_registry()), events


def test_default_task_navigation_uses_observations_without_simulator_or_evaluator_state():
    surface, context, runner, events = task_navigation()
    goal = [1.6, 1.]
    result = runner.run("navigate", {"goal_xy_m": goal, "yaw": .4, "snap": False},
                        surface.snapshot())
    assert result.ok, result
    assert np.linalg.norm(surface.measure().base_pose[:2] - goal) <= context.settings.motion.arrival_distance
    assert abs(surface.measure().base_pose[2] - .4) <= context.settings.motion.arrival_angle
    assert any(event.get("kind") == "drive.request" for event in events)
    assert sum(event["type"] == "skill_finished" for event in events) == 1
    assert np.allclose(surface.robot.velocity, 0)
    assert not surface.grips


@pytest.mark.parametrize("fault,code", [
    ("stop", "CANCELLED"),
    ("epoch", "STALE_EPOCH"),
    ("map_generation", "STALE_EPOCH"),
    ("obstacle_clock", "STALE_OBSERVATION"),
    ("unregistered", "STALE_OBSERVATION"),
])
def test_default_task_navigation_rechecks_authority_and_sensor_binding_before_motion(fault, code):
    surface, context, runner, events = task_navigation()
    attempt = runner.prepare("navigate", {"goal_xy_m": [1.6, 1.], "snap": False}, surface.snapshot())
    assert not attempt.finished
    sensors = context.navigation_evidence
    if fault == "stop":
        context.cancelled = lambda: True
    elif fault == "epoch":
        surface.robot.frame = replace(surface.robot.frame, epoch=surface.epoch + 1)
    else:
        changed = {"map_generation": {"generation": "reset-map"},
                   "obstacle_clock": {"obstacle_time": -10.},
                   "unregistered": {"registered": False}}[fault]
        def changed_evidence():
            sample = sensors()
            if fault == "obstacle_clock":
                changed["obstacle_state"] = replace(sample.obstacle_state, time=-10.)
            return replace(sample, **changed)
        context.navigation_evidence = changed_evidence
    result = runner.execute(attempt)
    assert result.code == code, result
    assert not any(any(command) for command in surface.robot.commands)
    assert not surface.grips
    assert sum(event["type"] == "skill_finished" for event in events) == 1


@pytest.mark.parametrize("fault,code", [("stop", "CANCELLED"), ("map_generation", "STALE_EPOCH")])
def test_docking_rechecks_authority_and_sensor_binding_before_motion(fault, code):
    surface, context, _, _ = task_navigation()
    sensors, samples = context.navigation_evidence, []
    if fault == "stop":
        context.cancelled = lambda: True
    else:
        def reset_after_binding():
            samples.append(sensors())
            return samples[-1] if len(samples) == 1 else replace(samples[-1], generation="reset-map")
        context.navigation_evidence = reset_after_binding
    with pytest.raises(SkillFailure) as failure:
        navigate_real.dock(context, surface.epoch, .2)
    assert failure.value.code == code
    assert not any(any(command) for command in surface.robot.commands)


def test_default_task_navigation_missing_evidence_has_no_fallback_or_motion():
    surface, context, runner, events = task_navigation()
    context.navigation_evidence = None
    result = runner.run("navigate", {"goal_xy_m": [1.6, 1.], "snap": False}, surface.snapshot())
    assert result.code == "ACTION_REJECTED", result
    assert not surface.robot.commands and not surface.grips
    assert sum(event["type"] == "skill_finished" for event in events) == 1


def test_sim_launcher_uses_the_default_sim_composition(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from homebody import __main__ as cli
    from homebody.backends import mujoco
    from homebody.helpers import kinematics
    from homebody.session import recorder, session
    from homebody.session.settings import Settings
    from homebody.vlm import observations

    surface = RobotSurface()
    sensors = SimNavigationEvidence(surface, surface.snapshot().navigation)
    surface.navigation_evidence = sensors
    surface.observer_frame = surface.snapshot
    surface.shoulder_frame = surface.snapshot
    surface.close = lambda: None
    captured = []

    class SessionSelection:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

        def run_task(self, task):
            return {"status": "model_done"}

    monkeypatch.setattr(mujoco, "MujocoBackend", lambda *args, **kwargs: surface)
    monkeypatch.setattr(kinematics, "Arm", lambda *args: object())
    monkeypatch.setattr(recorder, "Recorder", lambda *args: SimpleNamespace(
        path=tmp_path, event=lambda *args, **kwargs: None, close=lambda: None,
        setup_failure=lambda *args: None))
    monkeypatch.setattr(session, "Session", SessionSelection)
    monkeypatch.setattr(cli, "make", lambda *args: object())
    monkeypatch.setattr(observations, "semantic_prior", lambda *args: {})
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(root=root, scene=root / "assets/real2sim/src_kitchen",
                           spawns={}, headless=True, task="Navigate to the requested point")
    assert cli.run(args, Settings(), tmp_path) == 0
    assert len(captured) == 1
    for key in ("registry", "navigation", "manipulation", "navigation_evidence"):
        assert key not in captured[0], key


def test_sim_and_real_share_task_policy_and_measured_hand_prompt(tmp_path):
    from types import SimpleNamespace

    from test_real_backend import backend as make_robot
    from test_real_manipulation import carried_frame

    from homebody.session.real import compose
    from homebody.session.session import Session
    from homebody.session.settings import Settings
    from homebody.skills.manipulation_policy import REAL_MANIPULATION
    from homebody.ui.console import Controls

    settings = Settings()
    real_robot, motion, _clock = make_robot(tmp_path)
    real = compose(real_robot, object(), settings, SimpleNamespace(), Controls(),
                   root=tmp_path, memory_root=tmp_path / "memories", memory_name="room",
                   arms={}, jaw_samples={})
    sim = Session(RobotSurface(), object(), settings, SimpleNamespace(), Controls(),
                  arms={}, jaw_samples={}, registry=task_registry(),
                  navigation=navigate_real, manipulation=REAL_MANIPULATION)
    assert sim.context.manipulation == real.context.manipulation
    for name in ("navigate", "pick", "place"):
        first, second = sim.registry[name], real.registry[name]
        assert first.PROMPT == second.PROMPT and first.FAILURES == second.FAILURES
        assert first.prepare is second.prepare and first.execute is second.execute
    current = carried_frame()
    prompts = [instance.prompt("Place the object at the requested target", current)
               for instance in (sim, real)]
    for prefix in ("Hands now:", "Task contract:", "Current task:", "Current observation:"):
        blocks = [next(block for block in prompt.split("\n\n") if block.startswith(prefix))
                  for prompt in prompts]
        assert blocks[0] == blocks[1]
    assert "may hold an untracked object" in prompts[0]
    assert "instead of placing it elsewhere" in prompts[0]
    assert not motion.events


def payload_filter_scene(palm_to_payload):
    from fakes import PlanningArm
    from test_real_manipulation import carried_frame, carry

    acquired = carried_frame()
    torso = np.eye(4)
    torso[:2, 3] = [1., 1.]
    acquired = replace(acquired, world_torso=torso)
    robot = FakeBackend(acquired)
    context = Context(robot, robot, {"right": PlanningArm()}, carried=carry())
    context.carried["right"]["palm_to_payload"] = palm_to_payload
    static = acquired.navigation.occupied.copy()
    static[10, 10] = True
    navigation = replace(acquired.navigation, occupied=static)
    current_torso = torso.copy()
    current_torso[0, 3] += 1.
    robot.frame = replace(acquired, time=.1, world_torso=current_torso)
    return acquired, robot, context, navigation


def test_payload_filter_keeps_other_identities_distant_returns_and_static_structure():
    from test_real_navigation import evidence, marked

    acquired, robot, context, navigation = payload_filter_scene(np.eye(4))
    margin = context.settings.collision.cloud_margin
    match = context.settings.grasp.recovery_match
    half = context.carried["right"]["payload_half"][0]
    points = np.array([[1. + half + margin / 2, 1., 1.],
                       [1. + half + margin / 2, 1., 1.],
                       [1. + half + margin / 2, 1., 1.],
                       [1. + half + .8 * margin, 1. + half + .8 * margin, 1.],
                       [1. + half + match / 2, 1., 1.],
                       [1. + half + 2 * match, 1., 1.]])
    sample = evidence(robot, obstacles=points, static_navigation=navigation,
                      navigation=marked(navigation, points),
                      obstacle_state=acquired, obstacle_time=acquired.time,
                      obstacle_labels=np.array([1, 0, 2, 1, 1, 1]))
    bridge = navigate_real._Bridge(context.planning(), robot.frame)
    filtered = bridge.without_own_payload(sample)
    assert np.array_equal(filtered.obstacles, points[[1, 2, 5]])
    assert np.array_equal(filtered.obstacle_labels, [0, 2, 1])
    assert filtered.navigation.occupied[10, 10]
    assert filtered.navigation.occupied[10, 11]
    unassociated = bridge.without_own_payload(replace(sample, obstacle_labels=None))
    assert np.array_equal(unassociated.obstacles, points)
    context.carried["right"]["state"] = "uncertain"
    uncertain = bridge.without_own_payload(sample)
    assert np.array_equal(uncertain.obstacles, points)


def test_real_scan_generation_change_refuses_identity_evidence(tmp_path):
    from test_real_backend import backend as make_robot

    from homebody.primitives.observations import ObservationError

    robot, motion, _clock = make_robot(tmp_path)
    original = robot.ports.localization.pose_at
    robot.ports.localization.pose_at = lambda stamp=None, **kwargs: replace(
        original(stamp, **kwargs), generation=1 if stamp is None else 2)
    with pytest.raises(ObservationError, match="generations differ"):
        robot.navigation_evidence()
    assert not motion.events
