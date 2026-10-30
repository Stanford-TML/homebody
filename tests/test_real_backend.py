"""REAL composition through injected sensor and command ports, without robot access."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.backends.real import RealG1Backend, RealPorts
from homebody.backends.real_config import RealConfig
from homebody.primitives.actions import ActionRejected
from homebody.primitives.cameras.rgbd import RGBDCapture
from homebody.primitives.localization.map_binding import Alignment, MapBinding, sha256
from homebody.primitives.localization.mapped import RegisteredPose
from homebody.primitives.observations import HandState, ObservationError
from homebody.session.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def map_config(tmp_path):
    path = tmp_path / "map.npz"
    occupancy = np.zeros((8, 10), dtype=bool)
    occupancy[0, 0] = True
    T = np.array([[0, -.1, 2.], [.1, 0, -1.], [0, 0, 1.]])
    np.savez(path, occupancy=occupancy, T_map_px=T)
    return RealConfig("test0", path, sha256(path), "room-v1", tmp_path / "alignment.json", tmp_path / "camera.json",
                      map_floor_height_m=0., head_camera_serial="123456789012", stereo_calibration_sha256="a" * 64)


def test_alignment_requires_explicit_matching_map_and_rigid_transform(tmp_path):
    config = map_config(tmp_path)
    binding = MapBinding.load(config.map_path, expected_hash=config.map_sha256, frame_id=config.map_frame)
    doc = {"map_sha256": config.map_sha256, "map_frame": config.map_frame, "source_frame": "ws_map",
               "units": "m", "provenance": "fixture measured anchor", "anchor_odom_stamp_s": 12.,
               "anchor_T_source_lidar": np.eye(4).tolist(), "T_map_source": np.eye(4).tolist()}
    config.alignment_path.write_text(json.dumps(doc))
    assert Alignment.load(config.alignment_path, binding).anchor_stamp == 12.
    doc["map_sha256"] = "1" * 64
    config.alignment_path.write_text(json.dumps(doc))
    with pytest.raises(ObservationError, match="another map"):
        Alignment.load(config.alignment_path, binding)


class Clock:
    def __init__(self):
        self.value = 100.
    def __call__(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class Motion:
    generation = 1
    def __init__(self, clock):
        self.effort_calibration = {"measured": True}
        self.clock, self.events = clock, []
        self.joints = np.zeros(29)
    def state_at(self, timestamp=None, max_skew_s=.1):
        return {"q": self.joints.copy(), "tick": int(self.clock() * 250), "t": timestamp or self.clock(), "active": True}
    def hand_states(self, timestamp=None, max_skew_s=.1):
        states = {side: HandState(np.zeros(7), 0., np.zeros(3), np.zeros(3)) for side in ("left", "right")}
        return states, {side: self.clock() for side in states}
    def velocity(self, values): self.events.append(("velocity", tuple(values)))
    def arm(self, side, joints): self.events.append(("arm", side, tuple(joints)))
    def grip(self, side, closure): self.events.append(("grip", side, closure))
    def hold(self): self.events.append(("hold",))
    def close(self): self.events.append(("close",))


def backend(tmp_path):
    clock, config = Clock(), map_config(tmp_path)
    motion = Motion(clock)
    capture = RGBDCapture(1, clock(), clock(), clock(), clock(), np.zeros((4, 5, 3), dtype=np.uint8),
                         np.ones((4, 5)), np.eye(3), np.eye(4), "measured-head")
    camera = SimpleNamespace(snapshot=lambda: capture, calibration_hash="measured-head", close=lambda: None)
    T = np.eye(4)
    T[:3, 3] = [1., 2., 1.1]
    localizer = SimpleNamespace(pose_at=lambda stamp=None, **kw: RegisteredPose(stamp or clock(), 5., T, 1, 1.),
                               obstacles=lambda: (np.array([[1., 2., .6]]), clock()),
                               note_twist=lambda *v: None, close=lambda: None)
    b = RealG1Backend(ROOT, Settings(), config=config, ports=RealPorts(camera, localizer, motion, None),
                      clock=clock, monotonic=clock, sleep=clock.sleep)
    return b, motion, clock


def test_fresh_rgb_cannot_hide_expired_paired_depth(tmp_path):
    robot, motion, clock = backend(tmp_path)
    capture = robot.ports.camera.snapshot()
    robot.ports.camera.snapshot = lambda: replace(capture, depth_time=clock() - .52)
    with pytest.raises(ObservationError, match="expired during perception"):
        robot.snapshot()
    assert not motion.events


@pytest.mark.parametrize("case,blocked", [("open", False), ("closed", True), ("retained", True),
                                          ("missing", True), ("stale", True), ("fault", True)])
def test_passive_retention_guard_catches_unrecorded_loads_and_missing_evidence(tmp_path, case, blocked):
    robot, motion, clock = backend(tmp_path)
    hands, timestamps = motion.hand_states()
    if case == "closed":
        hands["right"] = replace(hands["right"], closure=.7)
    elif case == "retained":
        hands["left"] = replace(hands["left"], finger_effort_fraction=np.array([.2, .2, 0.]),
                                 finger_closure_gap=np.array([.2, .2, 0.]))
    elif case == "missing":
        del hands["right"]
    elif case == "stale":
        timestamps["right"] = clock() - .2
    def observe():
        if case == "fault":
            raise ObservationError("No hand packets")
        return hands, timestamps
    motion.hand_states = observe
    robot.ports.camera.snapshot = lambda: pytest.fail("Retention must not require camera")
    robot.ports.localization.pose_at = lambda *args, **kw: pytest.fail("Retention must not require pose")
    status = robot.retention_status()
    assert status["retained_or_unknown"] is blocked
    assert status["reason"]
    assert not motion.events


def test_deadman_wait_zeroes_base_without_releasing_hand(tmp_path):
    robot, motion, clock = backend(tmp_path)
    robot.grip(robot.epoch, "right", .9)
    robot.base_velocity(robot.epoch, .2, 0, 0)
    robot.advance(robot.epoch, .3)
    assert clock() >= 100.3
    assert ("hold",) in motion.events
    assert [e for e in motion.events if e[0] == "grip"] == [("grip", "right", .9)]


def test_missing_assets_offline_check_is_inert(tmp_path, monkeypatch):
    import socket
    config = map_config(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("offline check opened transport"))
    result = config.offline_checks()
    assert result["map"]["ok"]
    assert not result["camera_calibration"]["ok"]
    assert result["live_readiness"]["ok"] is None


def test_body_authority_loss_refuses_before_hand_or_arm_command(tmp_path):
    from homebody.backends.g1.task_motion import G1HostMotion
    config = map_config(tmp_path)
    sent = []
    body = SimpleNamespace(state={"q": np.zeros(29), "t": 100., "driver": "another", "active": True},
                           claim="ours", interruptions=0, wait_fresh=lambda timeout: True,
                           require_fresh=lambda: None, close=lambda: None,
                           send=lambda **cmd: sent.append(cmd), set_arms=lambda q: sent.append(q))
    hands = SimpleNamespace(set_targets=lambda **cmd: sent.append(cmd), close=lambda: None)
    calibration = json.loads((ROOT / "assets/robot/calibration.json").read_text())
    primitive = G1HostMotion(config, calibration, body=body, hands=hands,
                             lease=SimpleNamespace(release=lambda: None), start=False)
    with pytest.raises(ActionRejected, match="Another body client"):
        primitive.grip("right", 0.)
    assert sent == []
    primitive.close()


def test_body_hold_is_not_success_without_later_confirmation(tmp_path):
    from homebody.backends.g1.task_motion import G1HostMotion
    clock = Clock()
    state = {"q": np.zeros(29), "t": clock(), "tick": 3, "walk": True,
             "driver": "ours", "active": True}
    sent = []
    body = SimpleNamespace(state=state, claim="ours", interruptions=0,
                           wait_fresh=lambda timeout: True, require_fresh=lambda: None,
                           close=lambda: None, send=lambda **kw: sent.append(kw))
    hands = SimpleNamespace(close=lambda: None)
    calibration = json.loads((ROOT / "assets/robot/calibration.json").read_text())
    primitive = G1HostMotion(map_config(tmp_path), calibration, body=body, hands=hands,
                             lease=SimpleNamespace(release=lambda: None), clock=clock,
                             monotonic=clock, sleep=clock.sleep, start=False)
    with pytest.raises(ActionRejected, match="STOP is unverified"):
        primitive.hold()
    assert sent == [{"cmd": "stand"}]
    assert clock() == pytest.approx(100.3, abs=.005)
    primitive.close()


def test_production_prerequisites_refuse_before_opening_any_sensor(tmp_path, monkeypatch):
    import homebody.backends.real as module
    monkeypatch.setattr(module, "create_ports", lambda *args: pytest.fail("opened hardware"))
    with pytest.raises(ObservationError, match="prerequisites unavailable"):
        module.RealG1Backend(ROOT, Settings(), config=map_config(tmp_path))


def test_slow_perception_cannot_send_pick_commands_from_stale_frame(tmp_path):
    from homebody.skills import pick
    from homebody.skills.manipulation_policy import REAL_MANIPULATION
    robot, motion, clock = backend(tmp_path)
    selected = robot.snapshot()
    capture = robot.ports.camera.snapshot()
    robot.ports.camera.snapshot = lambda: replace(capture, sequence=2, captured_at=clock())
    def slow_labels(rgb, stamp):
        clock.sleep(.51)
        return np.ones(rgb.shape[:2], dtype=np.int32)
    robot.ports.tracker = SimpleNamespace(ready=True, labels=slow_labels,
        select=lambda *args: np.ones(selected.labels.shape, dtype=np.int32))
    planning = SimpleNamespace(manipulation=REAL_MANIPULATION, carried={}, target_selector=robot.select_target)
    with pytest.raises(ObservationError, match="expired during perception"):
        pick.prepare({"point_normalized_1000": [500, 500]}, selected, selected, planning)
    assert not motion.events


def test_pinned_segmentation_closure_and_head_calibration_unchanged():
    from homebody.primitives.cameras.tracking import verify_runtime
    assert verify_runtime(ROOT / "vendor/samurai") == ROOT / "vendor/samurai"
    doc = json.loads((ROOT / "assets/calib/g1_head.json").read_text())
    assert sha256(ROOT / doc["urdf"]) == doc["urdf_sha256"]
