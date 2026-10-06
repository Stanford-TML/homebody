"""Session lifecycle tests use explicit fakes; they do not claim Astra/physics validation."""
import json
import re
import runpy
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.primitives.observations import Frame, HandState, NavigationMap
from homebody.session.config import Agent
from homebody.session.recorder import Recorder
from homebody.session.session import Session
from homebody.session.settings import Settings
from homebody.skills.registry import SKILLS
from homebody.ui.console import Controls
from homebody.vlm.observations import describe, map_preview
from homebody.vlm.provider import Decision, ProviderBusy, ProviderError


def test_terminal_file_failure_does_not_mask_error_during_close(tmp_path):
    recorder = Recorder(tmp_path, Settings(), tmp_path / "runs")
    path = recorder.start_task("test", epoch=1)
    (path / "outcome.json").mkdir()
    with pytest.raises(IsADirectoryError):
        recorder.end_task({"status": "model_done"})
    recorder.close()
    assert (path / "outcome.json").is_dir()


class Backend:
    def __init__(self):
        self.time = 0.0
        self.epoch = 1
        self.sequence = 0
        self.sink = None
        self.stops = 0
        self.controls = None
        self.state_calls = 0
        self.fail_terminal = False

    def measure(self):
        return self.snapshot()

    def snapshot(self):
        self.sequence += 1
        return Frame(self.epoch, self.sequence, self.time,
                     np.zeros((16, 16, 3), dtype=np.uint8), np.ones((16, 16)),
                     np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]]), np.eye(4), np.eye(4),
                     np.zeros(3), np.zeros(29),
                     {"right": HandState(np.zeros(7), 0.0, np.zeros(3), np.zeros(3))},
                     np.ones((16, 16), dtype=np.int32),
                     NavigationMap(np.zeros((20, 20)),
                                   np.array([[0.1, 0, -1], [0, -0.1, 1], [0, 0, 1]]), 0.1))

    def base_velocity(self, epoch, forward, lateral, yaw):
        assert epoch == self.epoch
        assert (forward, lateral, yaw) == (0.0, 0.0, 0.0)

    def stop(self, epoch):
        assert epoch == self.epoch
        self.stops += 1

    def advance(self, epoch, seconds):
        assert epoch == self.epoch
        self.time += seconds
        if self.sink:
            self.sink(self.evaluation_state())

    def set_evaluation_sink(self, sink):
        self.sink = sink
        if sink:
            sink(self.evaluation_state())

    def evaluation_state(self):
        self.state_calls += 1
        if self.fail_terminal and self.state_calls == 5:
            raise RuntimeError("terminal sensor failure")
        return {"schema_version": 1, "time": self.time, "epoch": self.epoch,
                "physics_valid": True, "robot_upright": True, "warnings": [], "objects": {},
                "hidden_target": "SECRET_EVALUATOR"}

    def reset(self):
        self.epoch += 1
        self.time = 0
        self.sequence = 0
        self.controls.closing.set()


class Provider:
    def __init__(self, decisions, *, controls=None, error=None, ticks=3):
        self.decisions = iter(decisions)
        self.ticks = ticks
        self.controls = controls
        self.error = error
        self.prompts = []
        self.images = []
        self.names = []

    def generate(self, prompt, *, tick, **kwargs):
        self.prompts.append(prompt)
        self.images.append(kwargs["images"])
        self.names.append(kwargs["names"])
        for _ in range(self.ticks):
            tick()
        if self.controls:
            self.controls.stop()
        if self.error:
            raise self.error
        return next(self.decisions)


class ImageOnlyRecorder(Recorder):
    def _append_video(self, rgb, stream="observations", fps=4):
        pass


@pytest.fixture
def build(tmp_path):
    sessions = []

    def create(provider, *, settings=None, observer_frame=None, backend=None, controls=None, ui=None,
               scene_update=None):
        settings = settings or Settings(agent=Agent(done_settle_seconds=0.))
        backend, controls = backend or Backend(), controls or Controls()
        recorder = ImageOnlyRecorder(tmp_path / "source", settings, tmp_path / "runs")
        session = Session(backend, provider, settings, recorder, controls,
                          arms={}, jaw_samples={}, observer_frame=observer_frame, ui=ui,
                          scene_update=scene_update)
        sessions.append(session)
        return session

    yield create
    for session in sessions:
        session.recorder.close()


def outcome(session):
    return json.loads(next(session.recorder.path.glob("tasks/*/outcome.json")).read_text())


def test_thinking_advances_physics_and_does_not_expose_scorer(build):
    provider = Provider([Decision("Observed completion", "done", {"summary": "claim"})])
    session = build(provider)
    result = session.run_task("Inspect the current view")
    assert result["status"] == "model_done"
    assert session.backend.time >= 0.06
    assert session.backend.stops == 0
    assert all("SECRET_EVALUATOR" not in text for text in provider.prompts)
    assert session.backend.sink is None
    assert outcome(session)["independently_scored"] is False
    times = [row["time"] for row in physics_rows(session)]
    assert len(times) == 4 and times[0] == 0.0 and times[-1] == session.backend.time
    assert all(a < b for a, b in pairwise(times))


def test_cancel_between_decision_and_execution(build):
    provider = Provider([Decision("Move", "navigate", {"x": 10, "y": 10})])
    session = build(provider)
    provider.controls = session.controls
    result = session.run_task("Move")
    assert result["status"] == "cancelled"
    events = (session.recorder.path / "events.jsonl").read_text()
    assert "skill_started" not in events
    assert session.backend.stops > 0


def test_stop_before_task_is_not_lost(build):
    provider = Provider([])
    session = build(provider)
    session.controls.stop()
    assert session.run_task("Queued task")["status"] == "cancelled"
    assert not provider.prompts


@pytest.mark.parametrize("error, status", [
    (ProviderError("No account access"), "provider_error"),
    (RuntimeError("programming failure"), "error"), (KeyboardInterrupt(), "interrupted")])
def test_provider_failure_is_recorded_and_only_a_programming_error_propagates(build, error, status):
    session = build(Provider([], error=error))
    if isinstance(error, ProviderError):
        assert session.run_task("Observe")["status"] == status
        assert outcome(session)["reason"] == "No account access"
    else:
        with pytest.raises(type(error)):
            session.run_task("Observe")
    assert outcome(session)["status"] == status
    assert session.backend.sink is None


def test_provider_receives_only_the_current_views_and_the_scene_memory(build):
    provider = Provider([Decision("Refused", "navigate", {}), Decision("Finished", "done", {})])
    session = build(provider)
    session.semantic_context = {"entities": [{"label": "Island", "reference_anchor_ws_map_m": [1, 2, 3]}]}
    assert session.run_task("Observe")["status"] == "model_done"
    for images, prompt, names in zip(provider.images, provider.prompts, provider.names):
        assert [path.name for path in images] == ["rgb.png", "depth_preview.png", "map.png", "topdown.png"]
        assert len({path.parent for path in images}) == 1
        assert "\n- Island at [1, 2]\n" in prompt
        assert set(names) == {*SKILLS, "done"}
    assert provider.images[0][0].parent != provider.images[1][0].parent


@pytest.mark.parametrize("skill, arguments", [
    ("pick", {"point_normalized_1000": [625, 375], "hand": "left"}),
    ("navigate", {"goal_map_1000": [250, 600], "facing_map_1000": [350, 650]}),
    ("place", {"point_normalized_1000": [350, 700], "hand": "right"}),
    ("place", {"release_point_1000": [600, 400], "hand": "right"}),
])
def test_structured_visual_click_uses_normal_runner_and_captured_frame(
        build, monkeypatch, skill, arguments):
    from homebody.skills import navigate, place
    from homebody.skills.contract import Result, Runner, click_argument
    from homebody.skills.registry import REGISTRY

    decision = Decision.parse(json.dumps({"text": "Current visible target", "skill": skill,
                                         "arguments_json": json.dumps(arguments)}), tuple(REGISTRY))
    provider = Provider([decision, Decision("Finished", "done", {})])
    selections, prepared, executed = [], [], []
    ui = SimpleNamespace(update=lambda *args: None, show=lambda frame: None,
                         skill_event=lambda event: None,
                         select=lambda proposed, frame: selections.append((proposed, frame)))
    session = build(provider, ui=ui)

    def prepare(proposed, selected, current, planning):
        assert proposed == arguments
        assert selected.frame_id == selections[0][1].frame_id
        assert current.sequence > selected.sequence and current.time > selected.time
        assert provider.images[0][0].parent.name == f"{selected.epoch:03d}-{selected.sequence:08d}"
        if skill == "navigate":
            resolved = navigate.goal_arguments(proposed, selected.navigation)
        elif skill == "place":
            resolved = place.destination(proposed, selected, planning.settings)
        else:
            resolved = click_argument(proposed, selected)
        prepared.append(resolved)
        return selected.frame_id

    def execute(plan, context):
        executed.append(plan)
        return Result(skill, "OK", "Explicit test skill transport; no physical motion")

    monkeypatch.setitem(REGISTRY, skill, SimpleNamespace(
        NEEDS=(), FAILURES=(), PROMPT=REGISTRY[skill].PROMPT, prepare=prepare, execute=execute))
    assert isinstance(session.runner, Runner)
    assert session.run_task("Use the visible target")["status"] == "model_done"
    assert len(prepared) == len(executed) == 1
    assert executed[0] == selections[0][1].frame_id
    assert session.history[-1]["result"]["code"] == "OK"
    assert all("SECRET_EVALUATOR" not in prompt for prompt in provider.prompts)
    observation = json.loads(provider.prompts[0].split("Current observation:\n")[1].split(
        "\n\nRecent steps")[0])
    assert observation == {"base_pose": [0.0, 0.0, 0.0],
                           "map_click_from_world": [[500.0, 0.0, 500.0], [0.0, -500.0, 500.0]]}


class PrivilegedBackend(Backend):
    """Evaluation state as the simulator reports it: object poses, contacts and a file path."""
    def evaluation_state(self):
        state = super().evaluation_state()
        state["objects"] = {"carton_zz": {
            "position": [7.7731, -2.2185, 0.9113], "quaternion_wxyz": [1, 0, 0, 0],
            "corners": [[7.8431, -2.1485, 0.9813]] * 8, "linear_velocity": [0, 0, 0],
            "angular_velocity": [0, 0, 0], "hand_contacts": ["right"],
            "support_contacts": ["island_zz"], "object_contacts": []}}
        state["scene_file"] = "/home/someone/assets/physics.json"
        return state


def test_prompt_carries_no_evaluation_state_or_file_path(build, monkeypatch):
    """Only the model's decisions, the skills' public results, the robot's own sensors and
    the scan's annotations reach the prompt: never object poses, contacts, paths or tracebacks.
    The public result detail is kept, and the private state is still recorded for scoring."""
    from homebody.skills.contract import Result
    from homebody.skills.registry import REGISTRY

    provider = Provider([Decision("Pick it", "pick", {"point_normalized_1000": [500, 500], "hand": "right"}),
                         Decision("Finished", "done", {})])
    monkeypatch.setitem(REGISTRY, "pick", SimpleNamespace(
        NEEDS=(), FAILURES=(), PROMPT=REGISTRY["pick"].PROMPT,
        prepare=lambda proposed, selected, current, planning: selected.frame_id,
        execute=lambda plan, context: Result("pick", "OK", "Lift verified", {
            "palm_rise": 0.1234, "traceback": 'File "/home/someone/src/homebody/skills/pick.py", line 9'})))
    session = build(provider, backend=PrivilegedBackend())
    assert session.run_task("Pick the carton")["status"] == "model_done"
    assert len(provider.prompts) == 2
    assert '"palm_rise": 0.123}' in provider.prompts[1]
    assert '"frame_sequence"' not in provider.prompts[1] and '"epoch"' not in provider.prompts[1]
    private = ("7.7731", "-2.2185", "0.9113", "7.8431", "island_zz", "carton_zz", "SECRET_EVALUATOR",
               "scene_file", "/home/", "/tmp", "Traceback", "File \"", "pick.py", "runs/",
               str(session.recorder.path), str(provider.images[0][0].parent))
    for prompt in provider.prompts:
        assert not [word for word in private if word in prompt]
    rows = physics_rows(session)
    assert rows[-1]["objects"]["carton_zz"]["position"] == [7.7731, -2.2185, 0.9113]


@pytest.mark.parametrize("name", ["wait", "speak", "teleport"])
def test_unknown_skills_are_rejected_even_from_a_nonconforming_provider(build, name):
    session = build(Provider([Decision("Unsupported memory", name, {})]))
    result = session.run_task("Observe")
    assert result["status"] == "provider_error"
    assert result["reason"] == f"Unsupported skill: {name}"
    assert result["steps"] == 0


def test_reset_clears_task_history_and_carried_state(build):
    session = build(Provider([]))
    session.history.append({"operator_task": "Old task"})
    session.context.carried["right"] = {"observation": "old epoch"}
    session.controls.reset()
    session.backend.controls = session.controls
    session.run()
    assert session.backend.epoch == 2
    assert not session.history and not session.context.carried


@pytest.mark.parametrize("carried, line", [
    ({}, "Hands now: the left hand is free; the right hand is free."),
    ({"left": {"state": "verified"}, "right": {"state": "pending"}},
     ("Hands now: the left hand HOLDS the object from its last pick, not placed yet; only a "
      "place by that hand opens it; the right hand's grasp is uncertain; repeat pick on its "
      "target to recheck it.")),
])
def test_prompt_states_what_each_hand_holds(build, carried, line):
    session = build(Provider([]))
    session.context.carried.update(carried)
    assert line in session.prompt("Inspect", session.backend.snapshot()).split("\n\n")


def test_prompt_numbers_are_settings_limits_or_coordinate_conventions(build):
    """Every number in the instructions and skills is rendered from the settings, or is an
    image index, a click bound or the yaw origin; a changed limit changes the prompt."""
    defaults = Settings()
    settings = replace(defaults, motion=replace(defaults.motion, stance_reach=0.52),
                       placement=replace(defaults.placement, release_height=0.02, max_release_height=0.27,
                                         forward_reach=0.43),
                       topdown=replace(defaults.topdown, ahead=(0.1, 1.2), left=(-0.6, 0.65)))
    session = build(Provider([]), settings=settings)
    instructions = session.prompt("Inspect", session.backend.snapshot()).split("\n\nInitial-scene memory:")[0]
    assert "{" not in instructions and "}" not in instructions
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", instructions)) == {
        "0", "1", "2", "3", "4", "1000", "0.52", "0.02", "0.27", "0.43", "0.1", "1.2", "0.6", "0.65"}


def test_each_click_argument_names_the_attachment_the_recorder_sends(build):
    session = build(Provider([]))
    text = session.prompt("Inspect", session.backend.snapshot())
    listed = re.findall(r"^([1-4])\. [^\n]*?(RGB|depth preview|map|from above)", text, re.MULTILINE)
    assert listed == [("1", "RGB"), ("2", "depth preview"), ("3", "map"), ("4", "from above")]
    for argument, index in (("point_normalized_1000", "1"), ("goal_map_1000", "3"),
                            ("facing_map_1000", "3"), ("release_point_1000", "4")):
        assert re.search(rf"{argument}[^.]*?image (\d)", text).group(1) == index


def test_map_click_from_world_inverts_the_drive_map_click():
    from homebody.skills.navigate import map_point
    from homebody.vlm.observations import model_observation
    navigation = NavigationMap(np.zeros((1024, 768)),
                               np.array([[0.0172, 0, -2.94], [0, -0.0172, 8.39], [0, 0, 1]]), 0.0172)
    observed = model_observation(SimpleNamespace(navigation=navigation, base_pose=np.array([3.2, 2.1, -.74])))
    assert observed["base_pose"] == [3.2, 2.1, -0.74]
    for world in ([3.2, 2.1], [-2.9, 8.3], [10.1, -9.0]):
        click = np.array(observed["map_click_from_world"]) @ [*world, 1.]
        np.testing.assert_allclose(map_point({"goal_map_1000": click.tolist()}, "goal", navigation), world,
                                   atol=2e-3)


def test_step_limit_is_not_success(build):
    settings = replace(Settings(), agent=replace(Settings().agent, max_steps=1))
    session = build(Provider([Decision("Refused", "navigate", {})]), settings=settings)
    assert session.run_task("Observe")["status"] == "step_limit"


def test_recorder_keeps_metric_depth_and_reuses_exact_frame(build):
    session = build(Provider([]))
    frame = session.backend.snapshot()
    paths = session.recorder.frame(frame)
    assert session.recorder.frame(frame) == paths
    session.recorder.flush()
    with np.load(paths[0].parent / "sensors.npz") as saved:
        np.testing.assert_array_equal(saved["depth"], frame.depth)
        for side in frame.hands:
            np.testing.assert_array_equal(saved[f"{side}_hand_joints"], frame.hands[side].joints)
    assert describe(frame)["map_world_xy_from_pixel"] == frame.navigation.T_map_px.tolist()
    picture = map_preview(frame)
    assert tuple(picture[10, 10]) == (10, 125, 180)
    assert len(list((session.recorder.path / "frames").iterdir())) == 1


def test_a_failed_frame_write_fails_loudly_after_closing_every_file(build, monkeypatch):
    from homebody.session import recorder as recording

    def full_disk(*_args):
        raise OSError("No space left on device")

    monkeypatch.setattr(recording, "write_frame", full_disk)
    session = build(Provider([]))
    frame = session.backend.snapshot()
    session.recorder.frame(frame)
    with pytest.raises(OSError, match="No space"):
        session.recorder.close()
    assert session.recorder._events.closed
    assert "frame_write_failed" in (session.recorder.path / "events.jsonl").read_text()


def test_after_done_the_robot_stands_still_before_the_final_sample(build):
    """The scripted fixture stands still after each release; a session stands still after
    the model's done, so an object its last action released is judged at rest."""
    session = build(Provider([Decision("Observed", "done", {"summary": "done"})]),
                    settings=Settings(agent=Agent(done_settle_seconds=1.)))
    session.run_task("Inspect")
    times = [row["time"] for row in physics_rows(session)]
    assert times[-1] - times[0] >= 1.


def test_terminal_state_read_failure_still_records_outcome(build):
    provider = Provider([Decision("Observed", "done", {})])
    session = build(provider)
    session.backend.fail_terminal = True
    result = session.run_task("Inspect")
    assert result["status"] == "error"
    assert "terminal sensor failure" in outcome(session)["cleanup_error"]
    assert outcome(session)["status_before_cleanup"] == "model_done"
    assert session.backend.sink is None
    assert len(physics_rows(session)) == 4


def test_setup_failure_is_an_attempt(build):
    session = build(Provider([]))
    session.recorder.setup_failure("Move cartons", "No graphics context")
    assert outcome(session)["status"] == "setup_error"
    assert outcome(session)["steps"] == 0


def test_the_prior_carries_no_operator_alias_and_never_loads_live_physics(tmp_path):
    """The task's own name for an object (task_scene.json) is the operator's, not the scan's,
    so the model must ground "coffee bag" in the scan's description itself."""
    from homebody.vlm.observations import semantic_prior
    path = tmp_path / "semantics.json"
    path.write_text(json.dumps({"entities": [{"id": "source-item", "label": "Wrapped parcel",
                                             "reference_anchor_ws_map_m": [1, 2, 3],
                                             "portable_candidate": True,
                                             "internal_evidence": "OMIT"}]}))
    (tmp_path / "task_scene.json").write_text(json.dumps({
        "semantic_aliases": {"source-item": "Coffee bag"},
        "identity_provenance": "Operator-defined task identity",
        "robot_start": [9, 9, 9], "movable_entities": ["source-item", "private-list"]}))
    prior = semantic_prior(tmp_path)
    assert prior["entities"][0]["label"] == "Wrapped parcel"
    assert "Coffee bag" not in json.dumps(prior)
    assert prior["entities"][0]["reference_anchor_ws_map_m"] == [1, 2, 3]
    assert "private-list" not in json.dumps(prior)
    assert "OMIT" not in json.dumps(prior)
    assert "robot_start" not in prior
    assert "identity_provenance" not in json.dumps(prior) and "Operator-defined" not in json.dumps(prior)


def test_prior_places_every_entity_where_the_scan_saw_it(tmp_path):
    from homebody.vlm.observations import annotations, semantic_prior
    (tmp_path / "semantics.json").write_text(json.dumps({"entities": [
        {"id": "juice_carton", "label": "Sideways juice carton", "visual_description": "Orange print.",
         "location_description": "In front of the toaster on the counter.", "reference_anchor_ws_map_m": [1, 2, 3],
         "portable_candidate": True, "object_specific_limits": "Contents unknown."},
        {"id": "toaster", "label": "Toaster oven", "visual_description": "Chrome box; behind the juice carton.",
         "location_description": "On the counter. Behind juice carton.", "portable_candidate": False,
         "reference_anchor_ws_map_m": [1.3042, 2.0071, 3], "object_specific_limits": "No heat state."}]}))
    assert annotations(semantic_prior(tmp_path)) == (
        "- Sideways juice carton (portable) at [1, 2]: Orange print. In front of the toaster on the counter.\n"
        "- Toaster oven at [1.3, 2.01]: Chrome box; behind the juice carton. On the counter. Behind juice carton.")


def test_the_prompt_tells_the_model_what_the_scan_said_never_what_the_simulator_moves():
    """The simulator's movable_entities never reach the prompt: every entity keeps the scan's
    anchor, and what is marked portable is the scan's own judgement, so the four objects the
    tasks move look no different from the other portable things."""
    from homebody.vlm.observations import semantic_prior

    scene = Path(__file__).parents[1] / "assets/real2sim/src_kitchen"
    source = json.loads((scene / "semantics.json").read_text())
    movable = set(json.loads((scene / "task_scene.json").read_text())["movable_entities"])
    prior = semantic_prior(scene)["entities"]
    portable = {raw["id"] for raw, entry in zip(source["entities"], prior, strict=True) if entry.get("portable")}
    assert portable == {raw["id"] for raw in source["entities"] if raw.get("portable_candidate")}
    assert movable < portable
    for raw, entry in zip(source["entities"], prior, strict=True):
        assert entry.get("reference_anchor_ws_map_m") == raw.get("reference_anchor_ws_map_m")


def test_room_camera_is_recorded_during_tasks_but_never_passed_to_provider(build):
    provider = Provider([Decision("Observed", "done", {})])
    room = np.full((16, 16, 3), 127, dtype=np.uint8)
    session = build(provider, observer_frame=lambda: room)
    for _ in range(30):
        session.heartbeat()
    assert not any((session.recorder.path / "frames").iterdir())
    session.run_task("Observe")
    saved = list((session.recorder.path / "observer").glob("*.png"))
    assert len(saved) == 1
    assert len(provider.images[0]) == 4
    assert all("observer" not in path.parts for path in provider.images[0])
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["kind"] == "observer_frame")["model_visible"] is False


def test_every_operator_video_frame_has_its_simulation_time(tmp_path):
    """The shoulder stream samples whenever a snapshot is a quarter second newer, so its
    frames are timed only by their events; the room stream also keeps PNGs."""
    recorder = ImageOnlyRecorder(tmp_path, Settings(), tmp_path / "runs")
    view = np.zeros((16, 16, 3), dtype=np.uint8)
    for time, stream in ((0., "room"), (.3, "shoulder"), (.7, "shoulder"), (2., "room")):
        recorder.observer(view, epoch=1, simulation_time=time, stream=stream, fps=4)
    recorder.close()
    events = [json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()]
    frames = [(row["stream"], row["video_frame"], row["simulation_time"], row["path"])
              for row in events if row["kind"] == "observer_frame"]
    assert frames == [("room", 0, 0., "observer/00000000.png"), ("shoulder", 0, .3, None),
                      ("shoulder", 1, .7, None), ("room", 1, 2., "observer/00000001.png")]
    assert "events.jsonl" in json.loads((recorder.path / "provenance.json").read_text())["video"]


def physics_rows(session):
    path = next(session.recorder.path.glob("tasks/*/physics.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("error, expected", [(None, "model_done"),
                                             (ProviderError("immediate failure"), "provider_error")])
def test_immediate_terminal_task_does_not_duplicate_initial_sample(build, error, expected):
    session = build(Provider([Decision("Finished", "done", {})], error=error, ticks=0))
    assert session.run_task("Inspect")["status"] == expected
    samples = physics_rows(session)
    assert len(samples) == 1
    assert samples[0]["time"] == 0.0


def test_equal_time_dedup_cannot_clear_failure_evidence(build):
    session = build(Provider([]))
    recorder = session.recorder
    recorder.start_task("Record flags", 1)
    state = {"epoch": 1, "time": 0.0, "physics_valid": False, "robot_upright": False,
             "warnings": ["first"], "objects": {"latest": False}}
    recorder.physics(state)
    recorder.physics({**state, "physics_valid": True, "robot_upright": True,
                      "warnings": ["second"], "objects": {"latest": True}})
    recorder.end_task({"status": "test"})
    samples = physics_rows(session)
    assert len(samples) == 1
    assert samples[0]["physics_valid"] is False and samples[0]["robot_upright"] is False
    assert samples[0]["warnings"] == ["first", "second"]
    assert samples[0]["objects"] == {"latest": True}


@pytest.mark.parametrize("last", [{"epoch": 1, "time": 0.5}, {"epoch": 2, "time": 1.0}])
def test_clock_regression_or_reset_is_never_deduplicated(build, last):
    session = build(Provider([]))
    session.recorder.start_task("Record discontinuity", 1)
    session.recorder.physics({"epoch": 1, "time": 1.0})
    session.recorder.physics(last)
    session.recorder.end_task({"status": "test"})
    assert physics_rows(session) == [{"epoch": 1, "time": 1.0}, last]


def test_done_summary_type_failure_is_recorded_without_poisoning_ui(build):
    session = build(Provider([Decision("Finished", "done", {"summary": ["not text"]})]))
    assert session.run_task("Inspect")["status"] == "provider_error"
    assert outcome(session)["reason"] == "done.summary must be text"


@pytest.mark.parametrize("terminal_warning", [False, True])
def test_recorded_terminal_dedup_is_compatible_with_independent_scorer(build, terminal_warning):
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "evaluation/evaluate.py"))["score_attempt"]
    session = build(Provider([]))
    state = session.backend.evaluation_state()
    state["objects"] = {"carton": {
        "position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
        "corners": [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)],
        "linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
        "hand_contacts": [], "robot_contacts": [], "support_contacts": ["table"]}}
    target = {"schema_version": 1, "frame": "ws_map", "objects": {
        "carton": {"bounds": [[-1, -1, -1], [1, 1, 1]], "support_contacts": ["table"]}}}
    session.recorder.start_task("Record settled object", 1)
    for index in range(21):
        state["time"] = index / 10
        session.recorder.physics(state)
    if terminal_warning:
        state.update(physics_valid=False, robot_upright=False, warnings=["terminal_warning"])
    session.recorder.physics(state)
    session.recorder.end_task({"status": "model_done"})
    result = score(physics_rows(session), target)
    assert result["sample_count"] == 21
    assert result["success"] is not terminal_warning
    assert "nonmonotonic_time" not in result["invalid_reasons"]
    if terminal_warning:
        assert result["invalid_reasons"] == ["invalid_physics", "physics_warning", "robot_fall"]


class InvalidPhysicsBackend(Backend):
    def __init__(self, reason="Robot fell"):
        super().__init__()
        self.reason = reason
        self.invalid = False
        self.fail_next_advance = True
        self.actuation = []
        self.advance_count = 0
        self.fail_reset = False

    def base_velocity(self, epoch, forward, lateral, yaw):
        assert not self.invalid, "Invalid physics received another base command"
        self.actuation.append(("base", epoch))
        super().base_velocity(epoch, forward, lateral, yaw)

    def advance(self, epoch, seconds):
        from homebody.primitives.actions import PhysicsInvalid
        assert not self.invalid, "Invalid physics was advanced again"
        self.advance_count += 1
        self.actuation.append(("advance", epoch))
        if self.fail_next_advance:
            self.time += seconds
            self.invalid = True
            self.fail_next_advance = False
            raise PhysicsInvalid(self.reason)
        super().advance(epoch, seconds)

    def stop(self, epoch):
        self.actuation.append(("stop", epoch))
        super().stop(epoch)

    def evaluation_state(self):
        state = super().evaluation_state()
        if self.invalid:
            state.update(physics_valid=False, robot_upright=False, warnings=[self.reason])
        return state

    def reset(self):
        from homebody.primitives.actions import PhysicsInvalid
        self.epoch += 1
        self.time, self.sequence = 0., 0
        self.actuation.append(("reset", self.epoch))
        if self.fail_reset:
            raise PhysicsInvalid("Reset warm-up failed")
        self.invalid = False


class ScriptedControls(Controls):
    def __init__(self, commands):
        super().__init__()
        self.commands = iter(commands)

    def next(self):
        from homebody.ui.console import Command
        item = next(self.commands, "close")
        if item == "close":
            self.closing.set()
            return Command("close")
        return item() if callable(item) else item


class Display:
    def __init__(self):
        from types import SimpleNamespace
        self.status = SimpleNamespace(content="Ready")
        self.explanation = SimpleNamespace(content="")
        self.thought = SimpleNamespace(content="")
        self.stage = SimpleNamespace(content="")
        self._skill_stages = []
        self._stage_times = []
        self._stage_result = None
        self._stage_finished_at = None
        self._active_skill = ""
        self._stage_sim_start = self._stage_sim_now = None
        self.ego = SimpleNamespace(image=np.ones((16, 32, 3), dtype=np.uint8), label="Old frame")
        self.map = SimpleNamespace(image=np.ones((16, 16, 3), dtype=np.uint8))
        self.topdown = SimpleNamespace(image=np.ones((16, 16, 3), dtype=np.uint8), label="Old capture")
        self._bubble_message = ""
        self.model_name = "Codex (gpt-6-astra)"
        self.submit_button = SimpleNamespace(disabled=False)
        self.reset_button = SimpleNamespace(disabled=False)
        self.selections = []
        self.events = []

    def clear_selection(self):
        self.selections.clear()

    def clear_view(self):
        from homebody.ui.console import Console
        Console.clear_view(self)

    def select(self, decision, frame):
        self.selections.append((decision, frame.frame_id))

    def skill_event(self, event):
        self.events.append(event)

    def show(self, frame):
        pass

    def update(self, status, text):
        from homebody.ui.console import Console
        Console.update(self, status, text)


def test_idle_physics_failure_keeps_ui_resettable_and_blocks_queued_tasks(build):
    from homebody.ui.console import Command
    backend, ui, provider = InvalidPhysicsBackend(), Display(), Provider([])
    stopped_commands = []
    def inspect_failure():
        """Returning no command exercises an idle heartbeat in the invalid generation."""
        assert session.requires_reset
        assert ui.status.content == "Reset required"
        assert ui.submit_button.disabled and not ui.reset_button.disabled
        stopped_commands[:] = backend.actuation
    def inspect_blocked():
        assert backend.actuation == stopped_commands
        assert not provider.prompts
        assert outcome(session)["status"] == "physics_invalid"
        return Command("stop")
    def inspect_stopped():
        assert backend.actuation == stopped_commands
        assert session.requires_reset
        return Command("reset")
    def inspect_reset():
        """Returning no command resumes the normal heartbeat in the new generation."""
        assert not session.requires_reset and backend.epoch == 2
        assert ui.status.content == "Ready" and not ui.submit_button.disabled
        assert not ui.reset_button.disabled
    controls = ScriptedControls([None, inspect_failure, Command("task", "Blocked task"),
                                 inspect_blocked, inspect_stopped, inspect_reset])
    session = build(provider, backend=backend, controls=controls, ui=ui)
    session.run()
    assert backend.advance_count == 2
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "physics_invalid"]) == 1
    assert len([event for event in events if event["kind"] == "scene_reset"]) == 1


def test_provider_heartbeat_failure_is_recorded_and_reset_resumes_tasks(build):
    from homebody.ui.console import Command
    backend = InvalidPhysicsBackend("Robot fell")
    provider = Provider([Decision("Finished", "done", {})])
    controls = ScriptedControls([Command("reset")])
    session = build(provider, backend=backend, controls=controls)
    result = session.run_task("Inspect")
    assert result["status"] == "physics_invalid" and result["requires_reset"]
    assert outcome(session)["reason"] == "Robot fell"
    assert physics_rows(session)[-1]["physics_valid"] is False
    commands = backend.actuation.copy()
    assert session.run_task("Refused until reset")["status"] == "physics_invalid"
    assert backend.actuation == commands and len(provider.prompts) == 1
    session.run()
    assert not session.requires_reset and backend.epoch == 2
    assert session.run_task("Inspect after reset")["status"] == "model_done"
    assert len(provider.prompts) == 2


def test_skill_physics_failure_preserves_typed_terminal_and_reset_latch(build):
    backend = InvalidPhysicsBackend()
    provider = Provider([Decision("Hold current stance", "navigate", {"x": 0., "y": 0.})], ticks=0)
    session = build(provider, backend=backend)
    assert session.run_task("Verify stance")["status"] == "physics_invalid"
    assert session.requires_reset
    events = [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]
    terminal = [row["event"]["result"] for row in events
                if row["kind"] == "skill" and row["event"]["type"] == "skill_finished"]
    assert len(terminal) == 1 and terminal[0]["code"] == "PHYSICS_INVALID"
    commands = backend.actuation.copy()
    session.heartbeat()
    assert backend.actuation == commands


def test_failed_reset_clears_the_view_keeps_the_ui_alive_and_requires_another_reset(build):
    """The human scene follows the new generation even when its reset fails."""
    from homebody.ui.console import Command
    backend, display, generations = InvalidPhysicsBackend(), Display(), []
    backend.fail_reset = True
    display.selections.append("old target")
    display.stage.content = "old stage"
    def retry_reset():
        assert session.requires_reset and backend.epoch == 2
        assert session.status == "Reset required"
        assert generations == [(1, 0.), (1, .02), (2, 0.)]
        assert not display.ego.image.any() and not display.map.image.any()
        assert "waiting for fresh observations" in display.ego.label
        assert not display.selections and not display.stage.content
        backend.fail_reset = False
        return Command("reset")
    controls = ScriptedControls([None, Command("reset"), None, retry_reset])
    session = build(Provider([]), backend=backend, controls=controls, ui=display,
                    scene_update=lambda: generations.append((backend.epoch, backend.time)))
    session.run()
    assert backend.epoch == 3 and not session.requires_reset
    assert backend.advance_count == 1


def test_unrelated_idle_programming_error_still_propagates(build):
    class BrokenBackend(Backend):
        def advance(self, epoch, seconds):
            raise RuntimeError("Idle programming defect")
    session = build(Provider([]), backend=BrokenBackend(), controls=ScriptedControls([None]))
    with pytest.raises(RuntimeError, match="Idle programming defect"):
        session.run()
    assert not session.requires_reset


def test_human_scene_updates_during_thinking_without_entering_provider_inputs(build):
    backend, times = Backend(), []
    def scene_update():
        times.append(backend.time)
        return {"hidden_geometry": "SECRET_SCENE_GRAPH"}
    provider = Provider([Decision("Done", "done", {})], ticks=4)
    session = build(provider, backend=backend, scene_update=scene_update)
    session.run_task("Observe")
    np.testing.assert_allclose(times, [0, .02, .04, .06, .08])
    assert all("SECRET_SCENE_GRAPH" not in text for text in provider.prompts)
    assert len(provider.images[0]) == 4
    assert [path.name for path in provider.images[0]] == ["rgb.png", "depth_preview.png", "map.png", "topdown.png"]


def test_session_forwards_selected_capture_and_skill_events(build):
    display = Display()
    provider = Provider([Decision("This route is invalid", "navigate", {"x": "bad", "y": 1}),
                         Decision("Finished", "done", {})], ticks=1)
    session = build(provider, ui=display)
    assert session.run_task("Check route")["status"] == "model_done"
    assert display.selections[0][0].text == "This route is invalid"
    assert display.selections[0][1] == display.events[0]["frame_id"]
    assert [event["type"] for event in display.events] == [
        "skill_started", "skill_stage", "skill_finished"]
    assert display.events[-1]["result"].code == "INVALID_ARGUMENT"


def test_a_watched_session_is_paced_to_the_wall_clock_and_restarts_when_behind(monkeypatch):
    from homebody.session import session as module
    clock = {"wall": 0.0}
    sleeps = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["wall"])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(round(seconds, 6)))
    backend = SimpleNamespace(time=0.0, epoch=1)

    def advance(epoch, seconds):
        backend.time += seconds
    backend.advance = advance
    observed = module.ObservedBackend(backend, lambda frame: None, 1e9, real_time=True)
    observed._last = 0.0
    observed.advance(1, 0.1)               # anchors the pace
    observed.advance(1, 0.1)               # 0.1 s of physics in no wall time: wait it out
    assert sleeps == [0.1]
    clock["wall"] = 5.0                    # far behind (a long render): restart, no catch-up
    observed.advance(1, 0.1)
    assert sleeps == [0.1]
    unpaced = module.ObservedBackend(backend, lambda frame: None, 1e9)
    unpaced._last = backend.time
    unpaced.advance(1, 0.1)
    assert sleeps == [0.1]


class BusyThenAnswers(Provider):
    """At capacity for the first BUSY requests, then answers."""
    def __init__(self, decisions, busy, error=None):
        super().__init__(decisions, ticks=1)
        self.busy, self.calls, self.attempts, self.failure = busy, 0, [], error

    def generate(self, prompt, *, attempt, **kwargs):
        self.calls += 1
        self.attempts.append(attempt.name)
        if self.calls <= self.busy:
            raise (self.failure or ProviderBusy("Codex exited 1: Selected model is at capacity."))
        return super().generate(prompt, **kwargs)


def test_a_busy_service_is_asked_again_and_the_task_goes_on(build):
    """A service at capacity is asked again; the task does not end on the refusal."""
    provider = BusyThenAnswers([Decision("Finished", "done", {"summary": "ok"})], busy=2)
    session = build(provider, settings=Settings(agent=Agent(done_settle_seconds=0., busy_wait=0.01)))
    assert session.run_task("Inspect")["status"] == "model_done"
    assert provider.attempts == ["step-000", "step-000-retry1", "step-000-retry2"]


def test_busy_retries_run_out_and_other_errors_are_not_retried(build):
    busy = BusyThenAnswers([], busy=9)
    session = build(busy, settings=Settings(agent=Agent(done_settle_seconds=0., busy_wait=0.01, busy_retries=2)))
    assert session.run_task("Inspect")["status"] == "provider_error" and busy.calls == 3
    broken = BusyThenAnswers([], busy=9, error=ProviderError("Codex exited 1: invalid schema"))
    session = build(broken)
    assert session.run_task("Inspect")["status"] == "provider_error" and broken.calls == 1
