"""score_session.py scores a session's task against the sequence's own targets, settled by
default, and counts only a completed task."""
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORER = runpy.run_path(str(ROOT / "tools/score_session.py"))
TARGETS = {"parcel_a": {"bounds": [[-1, -1, 0], [1, 1, 1]], "support_contacts": ["table"]},
           "carton_b": {"bounds": [[4, 4, 0], [6, 6, 1]], "support_contacts": ["island"]}}
RESTING = {"parcel_a": ([0, 0, .1], "table"), "carton_b": ([5, 5, .1], "island")}


def rows(count=21, objects=RESTING):
    """COUNT samples a tenth of a second apart with each object released and settled."""
    def state(position, support):
        x, y, z = position
        return {"position": [x, y, z], "quaternion_wxyz": [1, 0, 0, 0],
                "corners": [[x + dx, y + dy, z + dz] for dx in (-.1, .1) for dy in (-.1, .1) for dz in (-.1, .1)],
                "linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0], "hand_contacts": [],
                "robot_contacts": [], "support_contacts": [support], "object_contacts": []}
    return [{"schema_version": 1, "time": index / 10, "epoch": 1, "physics_valid": True, "robot_upright": True,
             "warnings": [], "objects": {name: state(*where) for name, where in objects.items()}}
            for index in range(count)]


@pytest.fixture
def session(tmp_path):
    """A recorded session with one task; returns a scorer over it."""
    sequence = tmp_path / "sequence.json"
    sequence.write_text(json.dumps({"schema_version": 1, "steps": [
        {"fixture_id": name, "evaluator_target": target} for name, target in TARGETS.items()]}))
    task = tmp_path / "session-test/tasks/task-000"
    task.mkdir(parents=True)

    def score(physics, status, *options, capsys, typed="Put the bag on the table."):
        task.joinpath("task.json").write_text(json.dumps({"task": typed}))
        task.joinpath("physics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in physics))
        if status is not None:
            task.joinpath("outcome.json").write_text(json.dumps({"status": status, "finished_unix": 1}))
        code = SCORER["main"]([str(task.parents[1]), "--sequence", str(sequence), *options])
        captured = capsys.readouterr()
        score.err = captured.err
        return code, json.loads(captured.out)
    return score


def test_a_finished_task_with_settled_objects_in_their_targets_passes(session, capsys):
    code, report = session(rows(), "model_done", capsys=capsys)
    assert code == 0 and report["success"] and report["passed"] and report["task_status"] == "model_done"
    assert report["scoring_policy"] == "settled_placement" and report["task"] == "task-000"
    assert report["objects"] == {"parcel_a": {"terminal_reason": None}, "carton_b": {"terminal_reason": None}}


@pytest.mark.parametrize("status", ["cancelled", "step_limit", "error", "provider_error", "interrupted", None])
def test_settled_physics_without_a_completed_task_is_not_a_result(session, capsys, status):
    code, report = session(rows(), status, capsys=capsys)
    assert code == 1 and report["success"] and not report["passed"] and report["task_status"] == status


def test_a_short_placement_needs_terminal_only_to_pass_without_settling(session, capsys):
    short = rows(count=5)
    code, report = session(short, "model_done", capsys=capsys)
    assert code == 1 and not report["success"] and report["scoring_policy"] == "settled_placement"
    code, report = session(short, "model_done", "--terminal-only", capsys=capsys)
    assert code == 0 and report["success"] and report["scoring_policy"] == "terminal_placement"


def test_swap_exchanges_the_destinations_and_only_narrows_the_objects(session, capsys):
    code, report = session(rows(), "model_done", "--swap", capsys=capsys)
    assert code == 1 and report["objects"]["parcel_a"]["terminal_reason"] == "outside_target"
    code, report = session(rows(), "model_done", "--only", "carton_b", capsys=capsys)
    assert code == 0 and list(report["objects"]) == ["carton_b"]


def test_every_refusal_applied_to_a_passing_record_fails_it_with_its_reason(session, capsys):
    code, report = session(rows(), "model_done", "--perturb", capsys=capsys)
    assert code == 0 and report["success"]
    assert report["refusals"] == dict.fromkeys(
        ["hand_contact", "robot_contact", "outside_target", "unsupported", "moving",
         "invalid_physics", "robot_fall", "exchanged_destinations"], True)
    code, report = session(rows(), "model_done", "--perturb", "--terminal-only", capsys=capsys)
    assert code == 0 and "moving" not in report["refusals"]


def test_each_refusal_is_applied_to_every_object_not_only_one(session, capsys, monkeypatch):
    seen = []
    original = SCORER["OBJECT_REFUSALS"]["hand_contact"]
    monkeypatch.setitem(SCORER["OBJECT_REFUSALS"], "hand_contact",
                        lambda state, step, span: (seen.append(id(state)), original(state, step, span)))
    session(rows(), "model_done", "--perturb", capsys=capsys)
    assert len(seen) == 2 * 20  # both objects, every sample of the window after the first


def test_an_untargeted_object_knocked_off_fails_and_its_shift_is_refused(session, capsys):
    """--only scores carton_b; parcel_a must stay where it started."""
    code, report = session(rows(), "model_done", "--only", "carton_b", "--perturb", capsys=capsys)
    assert code == 0 and report["untargeted"] == {"parcel_a": {"terminal_reason": None}}
    assert report["refusals"]["disturbed"]
    knocked = rows()
    knocked[-1]["objects"]["parcel_a"]["support_contacts"] = ["floor"]
    code, report = session(knocked, "model_done", "--only", "carton_b", capsys=capsys)
    assert code == 1 and report["untargeted"]["parcel_a"]["terminal_reason"] == "disturbed"


def test_a_session_with_several_tasks_is_refused_unless_one_is_named(session, capsys, tmp_path):
    session(rows(), "model_done", capsys=capsys)
    other = tmp_path / "session-test/tasks/task-001"
    other.mkdir()
    other.joinpath("physics.jsonl").write_text(json.dumps(rows(1)[0]) + "\n")
    sequence = str(tmp_path / "sequence.json")
    with pytest.raises(SystemExit, match=r"holds 2 tasks \(task-000, task-001\); name one with --task"):
        SCORER["main"]([str(tmp_path / "session-test"), "--sequence", sequence])
    assert SCORER["main"]([str(tmp_path / "session-test"), "--sequence", sequence, "--task", "task-000"]) == 0
    assert json.loads(capsys.readouterr().out)["task"] == "task-000"


def test_a_task_other_than_the_scenes_default_is_warned_about(session, capsys):
    default = json.loads((ROOT / "assets/real2sim/src_kitchen/task_scene.json").read_text())["default_task"]
    session(rows(), "model_done", capsys=capsys, typed=default)
    assert "warning" not in session.err
    session(rows(), "model_done", capsys=capsys, typed="Put the bag on the shelf.")
    assert "warning" in session.err and "--target OBJECT=ENTITY" in session.err
    session(rows(), "model_done", "--swap", capsys=capsys, typed="Put the bag on the shelf.")
    assert "warning" not in session.err


def test_an_only_name_without_a_target_is_refused_by_name(session, capsys):
    with pytest.raises(SystemExit) as refused:
        session(rows(), "model_done", "--only", "carton_c", capsys=capsys)
    assert refused.value.code == 2 and "['carton_c']" in capsys.readouterr().err


def test_a_record_cut_off_by_a_crash_is_one_clear_error(session, tmp_path):
    """A session killed mid-write leaves a partial last line; the tool names it."""
    task = tmp_path / "session-test/tasks/task-000"
    task.joinpath("physics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows(2)) + '{"ti')
    with pytest.raises(SystemExit, match=r"physics.jsonl line 3 is not a JSON sample"):
        SCORER["main"]([str(task.parents[1]), "--sequence", str(tmp_path / "sequence.json")])


def test_a_session_without_a_recorded_task_is_one_clear_error(tmp_path):
    (tmp_path / "session-empty").mkdir()
    with pytest.raises(SystemExit, match="holds no recorded task"):
        SCORER["main"]([str(tmp_path / "session-empty")])


def test_target_scores_against_a_scene_entity_or_the_sequence_destination(session, capsys, tmp_path):
    """--target OBJECT=ENTITY: a sequence destination keeps its target; another entity is a
    surface target from its scanned bounds, resting on that entity."""
    scene = tmp_path / "scene"
    scene.mkdir()
    scene.joinpath("semantics.json").write_text(json.dumps({"entities": [
        {"id": "shelf", "bounds_ws_map_m": [[-1, -1, -.5], [1, 1, .05]]}]}))
    on_shelf = rows(objects={"parcel_a": ([0, 0, .1], "shelf"), "carton_b": ([5, 5, .1], "island")})
    code, report = session(on_shelf, "model_done", "--scene", str(scene), "--target", "parcel_a=shelf",
                           "--target", "carton_b=island", "--perturb", capsys=capsys)
    assert code == 0 and report["success"] and "exchanged_destinations" not in report["refusals"]
    code, report = session(rows(), "model_done", "--scene", str(scene), "--target", "parcel_a=shelf",
                           capsys=capsys)
    assert code == 1 and report["objects"]["parcel_a"]["terminal_reason"] == "wrong_support"
    with pytest.raises(SystemExit):
        session(rows(), "model_done", "--scene", str(scene), "--target", "parcel_a=attic", capsys=capsys)


def test_motion_is_refused_as_motion_for_an_object_resting_against_its_targets_edge(session, capsys):
    """The jitter leaves the last sample in place, so a carton touching the bin's wall still
    fails as moving rather than as outside the target."""
    edge = {**RESTING, "parcel_a": ([.895, 0, .1], "table")}
    code, report = session(rows(objects=edge), "model_done", "--perturb", capsys=capsys)
    assert code == 0 and report["passed"] and report["refusals"]["moving"]


@pytest.mark.parametrize("count", [21, 22, 23])
def test_motion_is_refused_as_motion_whatever_the_record_length(session, capsys, count):
    """The jitter is never zero, so the sample before the last always reads as moving."""
    _, report = session(rows(count), "model_done", "--perturb", capsys=capsys)
    assert report["passed"] and report["refusals"]["moving"]
