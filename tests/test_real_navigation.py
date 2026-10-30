"""The actual exploration controller through task Runner and explicit sensor evidence."""
from dataclasses import replace

import numpy as np
import pytest
from test_skills import FakeBackend, frame

from homebody.helpers.raster import world_to_px
from homebody.primitives.navigation.task import NavigationEvidence
from homebody.primitives.observations import ARM
from homebody.skills.contract import Context, Runner
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import task as navigate_real


def evidence(robot, **changes):
    state = robot.measure()
    return replace(NavigationEvidence(state.epoch, state.time, state.time, state.time, state.time,
        state.base_pose, robot.frame.navigation, np.zeros((0, 3)), 1., "test-room", "test-map-v1",
        (state.epoch, 1), "test_sensor_schedule", True, True), **changes)


def marked(navigation, points):
    """NAVIGATION with the cells of POINTS occupied, as evidence producers mark live obstacles."""
    occupied = navigation.occupied.copy()
    pixels = np.rint(world_to_px(navigation.T_map_px, points[:, :2])).astype(int)
    occupied[pixels[:, 1], pixels[:, 0]] = True
    return replace(navigation, occupied=occupied)


def context(robot, **kwargs):
    events = []
    ctx = Context(robot, robot, {}, navigation=navigate_real,
                  manipulation=REAL_MANIPULATION,
                  navigation_evidence=lambda: evidence(robot), emit=events.append, **kwargs)
    return ctx, Runner(ctx, {"navigate": navigate_real}), events


@pytest.mark.parametrize("change", [{"registered": False}, {"pose_time": -1.},
    {"obstacle_time": -1.}, {"map_time": -4.}, {"epoch": 2}, {"localization_quality": .1}])
def test_invalid_navigation_evidence_never_moves(change):
    robot = FakeBackend()
    ctx, runner, _ = context(robot)
    ctx.navigation_evidence = lambda: evidence(robot, **change)
    result = runner.run("navigate", {"x": 1.6, "y": 1.}, robot.frame)
    assert result.code in ("STALE_OBSERVATION", "ACTION_REJECTED")
    assert robot.commands == []


def test_stop_during_controller_execution_holds_without_facing_or_arrival():
    robot = FakeBackend()
    _ctx, runner, events = context(robot, cancelled=lambda: robot.time > .4)
    result = runner.run("navigate", {"x": 2., "y": 1., "yaw": 1.}, robot.frame)
    assert result.code == "CANCELLED"
    assert np.allclose(robot.velocity, 0)
    assert len([event for event in events if event["type"] == "skill_finished"]) == 1


def test_disconnected_goal_refuses_in_real_controller_preparation():
    current = frame()
    occupancy = current.navigation.occupied.copy()
    occupancy[:, 15:18] = True
    current = replace(current, navigation=replace(current.navigation, occupied=occupancy))
    robot = FakeBackend(current)
    _ctx, runner, _ = context(robot)
    result = runner.run("navigate", {"x": 2.8, "y": 1.}, robot.frame)
    assert result.code == "NO_ROUTE"
    assert robot.commands == []


def test_localization_loss_during_execution_never_claims_arrival():
    robot = FakeBackend()
    ctx, runner, events = context(robot)
    ctx.navigation_evidence = lambda: evidence(robot, registered=robot.time < .4)
    result = runner.run("navigate", {"x": 2., "y": 1.}, robot.frame)
    assert result.code == "STALE_OBSERVATION"
    assert np.allclose(robot.velocity, 0.)
    assert len([event for event in events if event["type"] == "skill_finished"]) == 1


def test_failed_hold_is_not_reported_as_a_verified_arrival():
    from homebody.primitives.actions import ActionRejected
    class FailedHold(FakeBackend):
        def base_velocity(self, epoch, forward, lateral, yaw_rate):
            if not any((forward, lateral, yaw_rate)):
                raise ActionRejected("Hold was not acknowledged")
            super().base_velocity(epoch, forward, lateral, yaw_rate)
    robot = FailedHold()
    _ctx, runner, events = context(robot)
    result = runner.run("navigate", {"x": 1.6, "y": 1.}, robot.frame)
    assert result.code == "ACTION_REJECTED"
    assert robot.stopped == 1
    assert len([event for event in events if event["type"] == "skill_finished"]) == 1


def loaded_navigation(*, doorway=False, compactable=False):
    from types import SimpleNamespace

    from fakes import PlanningArm
    from test_real_manipulation import carried_frame, carry

    class MovingFrame(FakeBackend):
        def arm_target(self, epoch, side, joints):
            super().arm_target(epoch, side, joints)
            measured = self.frame.joints.copy()
            measured[ARM[side]] = joints
            self.frame = replace(self.frame, joints=measured)

        def advance(self, epoch, seconds):
            super().advance(epoch, seconds)
            torso = np.eye(4)
            torso[:2, 3] = self.frame.base_pose[:2]
            self.frame = replace(self.frame, world_torso=torso)

    current = carried_frame()
    torso = np.eye(4)
    torso[:2, 3] = [1.5, 3.]
    occupied = np.zeros((60, 60), bool)
    if doorway:
        occupied[:, 30] = True
        occupied[24:36, 30] = False  # A 1.2 m opening for component-aware passage checks.
    current = replace(current, base_pose=np.array([1.5, 3., 0.]), world_torso=torso,
                      navigation=replace(current.navigation, occupied=occupied))
    robot = MovingFrame(current)
    ctx, runner, events = context(robot)
    ctx.arms = {"right": PlanningArm()}
    arm = ctx.arms["right"]
    arm.chain = ()
    arm.collision = SimpleNamespace(boxes=arm.collision.boxes, hand_boxes=arm.collision.hand_boxes,
                                    torso=arm.collision.torso, finger_joints=(), bounds={})
    ctx.carried.update(carry())
    ctx.carried["right"]["payload_half"] = np.array([.2 if compactable else .595, .05, .05])
    if compactable:
        ctx.carried["right"]["palm_to_payload"][0, 3] = .4
    return robot, ctx, runner, events


def test_laterally_large_payload_cannot_pass_a_door_that_admits_the_base():
    robot, ctx, runner, _ = loaded_navigation(doorway=True)
    ctx.carried["right"]["payload_half"] = np.array([.05, .70, .05])
    loaded = runner.run("navigate", {"goal_xy_m": [4.5, 3.], "snap": False}, robot.frame)
    assert not loaded.ok
    assert robot.frame.base_pose[0] < 3.
    assert any(any(command) for command in robot.commands)
    assert "right" in ctx.carried
    stopped_commands = len(robot.commands)
    ctx.carried.clear()
    robot.frame = replace(robot.frame, hands=frame().hands)
    unloaded = runner.prepare("navigate", {"goal_xy_m": [4.5, 3.], "snap": False}, robot.frame)
    assert not unloaded.finished
    assert len(robot.commands) == stopped_commands


def test_default_exploration_controller_rejects_radius_over_point_six():
    from homebody.primitives.navigation.contracts import DriveFailure
    robot = FakeBackend()
    ctx, _, _ = context(robot)
    live = navigate_real._Bridge(ctx.planning(), robot.frame).context()
    args = {"goal_xy_m": [2., 1.], "radius_m": .63, "planning_radius_m": .63,
            "execution_radius_m": .60, "preserve_goal": True,
            "max_footprint_radius_m": 5.}  # Tool arguments cannot grant a composition capability.
    with pytest.raises(DriveFailure, match="0.20..0.60"):
        navigate_real.controller.prepare(live, args)
    assert robot.commands == []


def test_measured_payload_growth_invalidates_prepared_component_envelope():
    robot, ctx, runner, _ = loaded_navigation()
    attempt = runner.prepare("navigate", {"goal_xy_m": [2.1, 3.], "snap": False}, robot.frame)
    assert not attempt.finished
    assert attempt.plan.radius_m < .4
    ctx.carried["right"]["payload_half"] = np.array([.6, .05, .05])
    assert runner.execute(attempt).code == "STALE_EPOCH"
    assert not any(any(command) for command in robot.commands)


def observed_low_wall(robot, ctx):
    """A base-blocking support whose measured height clears the held payload."""
    occupied = robot.frame.navigation.occupied.copy()
    occupied[:, 18] = True  # 25 cm from the base: inside even the base-only footprint.
    robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
    x, y = np.meshgrid(np.arange(1.69, 1.92, .02), np.arange(2.7, 3.32, .02))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .4)))
    ctx.navigation_evidence = lambda: evidence(robot, obstacles=points)


def test_loaded_controller_escapes_inflated_start_without_crossing_obstacle():
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    observed_low_wall(robot, ctx)
    # Keep the held object inside the observed map throughout the full 1 m retreat.
    result = runner.run("navigate", {"goal_xy_m": [.5, 3.], "snap": False}, robot.frame)
    assert result.ok, result
    recoveries = [row for row in events if row["type"] == "navigation"
                  and row["kind"] == "recovery.complete"]
    assert recoveries and recoveries[0]["details"]["padding_escape"]
    assert all(command[0] <= 0 for command in robot.commands)
    assert np.linalg.norm(robot.frame.base_pose[:2] - [.5, 3.]) <= ctx.settings.motion.arrival_distance
    assert len([row for row in events if row["type"] == "skill_finished"]) == 1


@pytest.mark.parametrize("blocked_start", [False, True])
def test_pending_compaction_occurs_held_then_reprepares_same_goal(monkeypatch, blocked_start):
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    ctx.carried["right"]["compact_pending"] = True
    if blocked_start:
        observed_low_wall(robot, ctx)
    calls = []
    def compact(bounded, epoch, side):
        assert np.allclose(robot.velocity, 0)
        assert "arm_target" in navigate_real.NEEDS
        if blocked_start:
            departures = [row for row in events if row.get("kind") == "drive.departure_restored"]
            assert departures and not departures[-1]["details"]["arrived"]
        calls.append((robot.time, robot.frame.base_pose.copy()))
        bounded.actions.arm_target(epoch, side, np.array([-.4, 0., 0., 0., 0., 0., 0.]))
        bounded.carried[side].pop("compact_pending")
        bounded.advance(epoch, .2)
        return robot.frame
    monkeypatch.setattr(navigate_real, "compact_carry", compact)
    goal = [.8, 3.] if blocked_start else [2.1, 3.]
    result = runner.run("navigate", {"goal_xy_m": goal, "snap": False}, robot.frame)
    assert result.ok, result
    assert len(calls) == 1
    refreshed = [row for row in events if row.get("kind") == "carry.reprepared"]
    assert len(refreshed) == 1 and refreshed[0]["details"]["goal_xy"] == goal
    assert refreshed[0]["details"]["planning_radius_m"] < .63
    assert refreshed[0]["details"]["payload_signature"] == navigate_real.payload_signature(ctx.carried)
    assert result.details["goal_xy"] == goal
    assert len([row for row in events if row["type"] == "skill_started"]) == 1
    assert len([row for row in events if row["type"] == "skill_finished"]) == 1


@pytest.mark.parametrize("fault", ["stop", "hold", "timeout", "epoch"])
def test_compaction_fault_does_not_continue_or_release_payload(monkeypatch, fault):
    from homebody.skills.contract import SkillFailure
    robot, ctx, runner, events = loaded_navigation()
    ctx.carried["right"]["compact_pending"] = True
    cancelled = [False]
    ctx.cancelled = lambda: cancelled[0]
    def compact(bounded, epoch, side):
        assert np.allclose(robot.velocity, 0)
        if fault == "hold":
            bounded.carried[side]["state"] = "uncertain"
            raise SkillFailure("HOLD_UNCERTAIN", "Measured retention failed during compaction")
        if fault == "stop":
            cancelled[0] = True
        elif fault == "timeout":
            bounded.advance(epoch, ctx.settings.motion.drive_timeout)
        elif fault == "epoch":
            robot.reset_on_advance = True
            bounded.advance(epoch, .1)
        return robot.frame
    monkeypatch.setattr(navigate_real, "compact_carry", compact)
    result = runner.run("navigate", {"goal_xy_m": [2.1, 3.], "snap": False}, robot.frame)
    assert result.code == {"stop": "CANCELLED", "hold": "HOLD_UNCERTAIN",
                           "timeout": "TIMEOUT", "epoch": "STALE_EPOCH"}[fault]
    assert "right" in ctx.carried and robot.frame.hands["right"].closure > .15
    assert not any(any(command) for command in robot.commands)
    assert not any(row.get("kind") == "carry.reprepared" for row in events)
    assert len([row for row in events if row["type"] == "skill_finished"]) == 1
