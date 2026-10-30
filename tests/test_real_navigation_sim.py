"""The checked task binding runs on actual MuJoCo observations, physics and AMO commands."""
from dataclasses import replace
from pathlib import Path

from homebody.backends.mujoco import MujocoBackend
from homebody.session.settings import Settings
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import task as navigate_real


def test_real_task_policy_composition_keeps_scoring_outside_the_session(tmp_path):
    from test_session import ImageOnlyRecorder, Provider

    from homebody.session.real_sim import compose
    from homebody.ui.console import Controls
    from homebody.vlm.observations import semantic_prior
    from homebody.vlm.provider import Decision

    root = Path(__file__).resolve().parents[1]
    settings = Settings(camera=replace(Settings().camera, width=160, height=120),
                        agent=replace(Settings().agent, done_settle_seconds=0.))
    robot = MujocoBackend(root, settings)
    recorder = ImageOnlyRecorder(root, settings, tmp_path / "runs", backend="mujoco-real-policy")
    provider = Provider([Decision("Remember current image", "remember", {"content": "Current room view"}),
                         Decision("Recall it", "recall", {"k": 1}),
                         Decision("Stored and recalled", "done", {})], ticks=0)
    try:
        session = compose(robot, provider, settings, recorder, Controls(), root=root,
                          memory_root=tmp_path / "memory", memory_name="room")
        assert session.evaluator is None
        assert session.semantic_context == semantic_prior(robot.scene)
        assert session.context.navigation is navigate_real
        assert session.context.manipulation == REAL_MANIPULATION
        assert session.context.navigation_evidence().obstacles.size > 0
        result = session.run_task("Remember this view, then recall the saved image")
        assert result["status"] == "model_done"
        assert len(provider.images[-1]) == 5
        assert all(names == ("navigate", "pick", "place", "remember", "recall", "done") for names in provider.names)
        assert next(recorder.path.glob("tasks/*/physics.jsonl")).read_text() == ""
        assert robot.evaluation_state()["physics_valid"]
    finally:
        recorder.close()
        robot.close()
