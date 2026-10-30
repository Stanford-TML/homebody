"""Actual grasp/STOP/strict-place behavior under the opt-in REAL task policy in MuJoCo."""
import json
from pathlib import Path

import numpy as np
from test_session import ImageOnlyRecorder, Provider

from homebody.backends.mujoco import MujocoBackend
from homebody.helpers.geometry import inverse, points_in
from homebody.helpers.grasp import visible_shape
from homebody.motion.carry import continuing
from homebody.session.real_sim import compose
from homebody.session.settings import Settings
from homebody.ui.console import Controls
from homebody.vlm.provider import Decision


def test_real_policy_retains_actual_carry_across_stop_and_refuses_unreachable_place(tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root / "configs/simulation.toml", scene=root / "assets/real2sim/src_kitchen")
    robot = MujocoBackend(root, settings)
    recorder = ImageOnlyRecorder(root, settings, tmp_path / "runs", backend="mujoco-real-policy")
    controls = Controls()

    class PickThenStop:
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Decision("Pick the observed bag", "pick", {"point_normalized_1000": [778, 328], "hand": "right"})
            controls.stop()
            return Decision("Discard this late decision", "done", {})

    try:
        initial = robot.snapshot()
        label = initial.select(*initial.normalized_pixel([778, 328]))
        original = visible_shape(initial.target_points(label), settings.grasp.shape_quantiles).center
        session = compose(robot, PickThenStop(), settings, recorder, controls, root=root,
                          memory_root=tmp_path / "memory", memory_name="carry-test")
        outcome = session.run_task("Pick the selected bag, then wait for the operator")
        assert outcome["status"] == "cancelled"
        assert outcome["session_retained"]
        assert continuing(robot.snapshot(), session.context.arms["right"], "right",
                          session.context.carried["right"], settings.grasp)
        assert not session.end_session()
        assert not controls.closing.is_set()
        held_targets = {side: values.copy() for side, values in robot.controller.hand_target.items()}
        controls.cancel.clear()
        session.provider = Provider([
            Decision("Unreachable target", "place", {"hand": "right", "x_ahead_m": 20., "y_left_m": 0.}),
            Decision("Refusal recorded", "done", {})], ticks=0)
        session.run_task("Test an unreachable requested placement")
        results = lambda: [json.loads(line)["result"] for line in (recorder.path / "events.jsonl").read_text().splitlines()
                           if json.loads(line)["kind"] == "result"]
        assert results()[-1]["code"] == "UNREACHABLE"
        assert all(np.array_equal(robot.controller.hand_target[side], value)
                   for side, value in held_targets.items())
        current = robot.snapshot()
        assert continuing(current, session.context.arms["right"], "right",
                          session.context.carried["right"], settings.grasp)
        assert not session.end_session()
        target = points_in(inverse(current.world_torso), original)
        palm = current.palm(session.context.arms["right"], "right")
        center = points_in(inverse(current.world_torso), points_in(
            palm, session.context.carried["right"]["palm_to_payload"][:3, 3]))
        session.provider = Provider([
            Decision("Return to observed location", "place", {"hand": "right", "x_ahead_m": float(target[0]),
                "y_left_m": float(target[1]), "delta_z_m": float(target[2] + .03 - center[2])}),
            Decision("Placement result recorded", "done", {})], ticks=0)
        session.run_task("Return the object to its previously observed position")
        assert results()[-1]["code"] == "OK"
        assert results()[-1]["details"]["short_of_point_m"] == 0.
        assert not session.context.carried
        assert session.end_session()
        assert robot.evaluation_state()["physics_valid"]
    finally:
        recorder.close()
        robot.close()
