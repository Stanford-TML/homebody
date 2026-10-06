"""The offline cockpit replays recorded events through the console's own overlays."""
import json

import numpy as np
import pytest
from fakes import PlanningArm
from PIL import Image

from homebody.helpers.geometry import transform
from homebody.session.recorder import json_value
from homebody.skills import navigate, pick
from homebody.ui.theme import ACCENT, MASK, PLAN, RELEASE
from tools import compose_cockpit


def record_frame(session, height=480, width=640):
    """One recorded frame as the Recorder saves it: a visible segment 1 m ahead of the camera."""
    folder = session / "frames" / "001-00000001"
    folder.mkdir(parents=True)
    labels = np.zeros((height, width), np.int32)
    labels[160:320, 240:400] = 5
    Image.fromarray(np.full((height, width, 3), 120, np.uint8)).save(folder / "rgb.png")
    np.savez_compressed(folder / "sensors.npz", depth=np.ones((height, width), np.float32),
                        labels=labels, intrinsics=np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]]),
                        world_camera=np.eye(4), world_torso=transform([-.4, 0., 0.]),
                        base_pose=np.zeros(3), joints=np.zeros(29), right_hand_joints=np.zeros(7))
    (folder / "state.json").write_text(json.dumps({
        "epoch": 1, "sequence": 1, "simulation_seconds": 2.0,
        "hands": {"right": {"closure": 0., "finger_effort_fraction": [0, 0, 0],
                            "finger_closure_gap": [0, 0, 0]}}}))


def test_synthetic_session_composes_click_and_release_rings_and_palm_path(tmp_path, monkeypatch):
    """A still-recording session's partial last line is left out, and only decisions, plans and
    recorded frames draw. Each picture is the 640x480 ego view over half-size depth and
    segmentation, beside a 720 px map and a 480 px top-down."""
    session, scene = tmp_path / "session-synthetic", tmp_path / "scene"
    record_frame(session)
    scene.mkdir()
    np.savez(scene / "map.npz", occupancy=np.zeros((20, 20), bool),
             T_map_px=np.array([[.1, 0., -1.], [0., -.1, 1.], [0., 0., 1.]]))
    (session / "provenance.json").write_text(json.dumps({"settings": {"topdown": {
        "ahead": [0., .8], "left": [-.4, .4], "pixels_per_metre": 600.}}}))
    plan = {"epoch": 1, "side": "right", "label": 5, "point": [.3, .1, 1.], "pre_joints": [0.] * 7,
            "pre": transform([.1, .05, 1.2]).tolist(), "palm": transform([.1, .1, 1.1]).tolist(),
            "withdraw": np.eye(4).tolist(), "support": True}
    rows = [{"kind": "scene_package", "wall_time": 0., "path": "/nowhere/scene"},
            {"kind": "task_started", "wall_time": 1., "task": "Release the held object"},
            {"kind": "decision", "wall_time": 2., "step": 0, "epoch": 1, "sequence": 1,
             "decision": {"text": "The visible support is clear.", "skill": "place",
                          "arguments": {"point_normalized_1000": [500, 500], "hand": "right"}}},
            {"kind": "skill", "wall_time": 3., "event": {"type": "skill_prepared",
                                                       "skill": "place", "plan": plan}},
            {"kind": "observer_frame", "wall_time": 3.5, "epoch": 1, "simulation_time": 2.,
             "video_frame": 0, "path": "observer/00000000.png", "model_visible": False},
            {"kind": "frame", "wall_time": 4., "epoch": 1, "sequence": 1, "simulation_time": 2.,
             "path": "frames/001-00000001"},
            {"kind": "result", "wall_time": 5., "step": 0, "result": {"skill": "place", "code": "OK",
                                                                      "message": "Released", "details": {}}}]
    (session / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"ki')
    monkeypatch.setattr(compose_cockpit, "Arm", lambda urdf, side: PlanningArm())
    video, count = compose_cockpit.compose(session, scene=scene, frames=True)
    assert (video, count) == (session / "cockpit.mp4", 2)
    assert video.stat().st_size > 0
    pictures = sorted((session / "cockpit").glob("*.png"))
    assert [path.name for path in pictures] == ["000000.png", "000001.png"]
    decision_view, frame_view = (np.asarray(Image.open(path)) for path in pictures)
    assert decision_view.shape == frame_view.shape == (720, 1840, 3)
    ego, topdown = frame_view[:480, :640], frame_view[:480, 1360:]

    def present(region, colour):
        return bool(np.all(region == colour, axis=-1).any())

    assert present(ego, MASK) and present(ego, PLAN)
    assert present(topdown, ACCENT) and present(topdown, RELEASE)
    assert present(decision_view[:480, :640], MASK) and not present(decision_view[:480, :640], PLAN)
    assert not present(decision_view[:480, 1360:], RELEASE)


def test_a_decision_whose_frame_was_never_recorded_is_one_clear_error(tmp_path, monkeypatch):
    session, scene = tmp_path / "session-partial", tmp_path / "scene"
    (session / "frames").mkdir(parents=True)
    scene.mkdir()
    np.savez(scene / "map.npz", occupancy=np.zeros((20, 20), bool),
             T_map_px=np.array([[.1, 0., -1.], [0., -.1, 1.], [0., 0., 1.]]))
    (session / "provenance.json").write_text(json.dumps({"settings": {}}))
    row = {"kind": "decision", "wall_time": 2., "step": 0, "epoch": 1, "sequence": 9,
           "decision": {"text": "", "skill": "look", "arguments": {}}}
    (session / "events.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(compose_cockpit, "Arm", lambda urdf, side: PlanningArm())
    with pytest.raises(FileNotFoundError, match="001-00000009 holds no recorded frame"):
        compose_cockpit.compose(session, scene=scene)
    with pytest.raises(SystemExit) as stopped:
        compose_cockpit.main([str(session), "--scene", str(scene)])
    assert str(stopped.value).startswith("error: ") and "001-00000009" in str(stopped.value)
    assert not (session / "cockpit.mp4").exists()


def test_a_replay_that_fails_midway_closes_the_video_it_started(tmp_path, monkeypatch):
    """The frames composed before a missing one are a finished video, not an open stream."""
    session, scene = tmp_path / "session-partial", tmp_path / "scene"
    record_frame(session)
    scene.mkdir()
    np.savez(scene / "map.npz", occupancy=np.zeros((20, 20), bool),
             T_map_px=np.array([[.1, 0., -1.], [0., -.1, 1.], [0., 0., 1.]]))
    (session / "provenance.json").write_text(json.dumps({"settings": {}}))
    rows = [{"kind": "frame", "wall_time": float(sequence), "epoch": 1, "sequence": sequence,
             "simulation_time": 2., "path": f"frames/001-{sequence:08d}"} for sequence in (1, 2)]
    (session / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(compose_cockpit, "Arm", lambda urdf, side: PlanningArm())
    with pytest.raises(FileNotFoundError, match="001-00000002 holds no recorded frame") as failure:
        compose_cockpit.compose(session, scene=scene)
    assert failure.value and b"moov" in (session / "cockpit.mp4").read_bytes()


def test_recorded_walk_in_plan_is_rebuilt_with_its_drive_plan_and_unknown_records_are_skipped():
    walk = navigate.Plan(1, np.array([1., .5]), .3, [np.array([0., 0.]), np.array([1., .5])],
                      requested_goal=np.array([0., 0.]))
    recorded = json.loads(json.dumps(pick.ApproachPlan(1, "right", 7, walk), default=json_value))
    rebuilt = compose_cockpit.plan(recorded)
    assert isinstance(rebuilt, pick.ApproachPlan) and isinstance(rebuilt.drive_plan, navigate.Plan)
    np.testing.assert_allclose(rebuilt.drive_plan.waypoints, [[0., 0.], [1., .5]])
    assert rebuilt.drive_plan.departure_goal is None and rebuilt.label == 7
    assert compose_cockpit.plan({"epoch": 1, "unknown": []}) is None


def test_a_session_without_frames_replays_from_its_videos_and_decision_images(tmp_path):
    """run_wave.py deletes frames/; the camera and room videos and each decision's images remain."""
    import imageio_ffmpeg
    session = tmp_path / "session-light"
    decision = session / "tasks/task-000/decisions/step-000"
    decision.mkdir(parents=True)
    for name, size in (("observations.mp4", (64, 48)), ("room.mp4", (96, 54))):
        video = imageio_ffmpeg.write_frames(str(session / name), size, fps=2, macro_block_size=1,
                                            ffmpeg_log_level="error")
        video.send(None)
        for shade in (40, 200):
            video.send(np.full((size[1], size[0], 3), shade, np.uint8))
        video.close()
    for index in range(4):
        Image.fromarray(np.full((100, 100, 3), 255, np.uint8)).save(decision / f"observation-{index}.png")
    (decision / "prompt.txt").write_text('{"map_click_from_world": [[100.0, 0.0, 500.0], [0.0, -100.0, 500.0]]}')
    rows = [{"kind": "task_started", "task": "Tidy"},
            {"kind": "observer_frame", "stream": "room", "video_frame": 0, "simulation_time": 1.},
            {"kind": "frame", "video_frame": 0, "simulation_time": 1., "path": "frames/001-00000001"},
            {"kind": "decision", "step": 0, "epoch": 1, "sequence": 1, "decision": {
                "text": "Walk", "skill": "navigate", "arguments": {"goal_xy_m": [1., 1.]}}},
            {"kind": "frame", "video_frame": 1, "simulation_time": 2., "path": "frames/001-00000002"}]
    (session / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output, count = compose_cockpit.compose(session)
    assert count == 3 and output.stat().st_size > 0
    ringed = compose_cockpit.ringed(decision / "observation-2.png",
                                    compose_cockpit.clicks(compose_cockpit.Decision(
                                        "Walk", "navigate", {"goal_xy_m": [1., 1.]}), decision), 2)
    assert tuple(ringed[40, 50 + 6]) != (255, 255, 255)  # the goal ring at map click (600, 400)
