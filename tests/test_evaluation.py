"""Independent scoring rejects claims, bad traces and incomplete final placements."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "evaluation/evaluate.py"
SPEC = importlib.util.spec_from_file_location("physical_evaluation", SOURCE)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)
TARGET = {"schema_version": 1, "frame": "ws_map", "objects": {
    "carton": {"bounds": [[-1, -1, 0], [1, 1, 1]], "support_contacts": ["table"]}}}


def sample(time):
    return {"schema_version": 1, "time": time, "epoch": 0, "physics_valid": True,
            "robot_upright": True, "warnings": [], "objects": {"carton": {
                "position": [0, 0, 0.1], "quaternion_wxyz": [1, 0, 0, 0],
                "corners": [[x, y, z] for x in [-0.1, 0.1]
                            for y in [-0.1, 0.1] for z in [0, 0.2]],
                "linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
                "hand_contacts": [], "robot_contacts": [], "support_contacts": ["table"]}}}


def trace():
    return [sample(i / 10) for i in range(21)]


def test_complete_contained_released_trace_reports_settling():
    result = evaluate.score_attempt(trace(), TARGET)
    assert result["success"] and result["settled_seconds"] == 2
    assert result["scoring_policy"] == "settled_placement"


def interior_target():
    """TARGET with rotated walls and a sloped wall that fit inside its broad outer bounds."""
    goal = copy.deepcopy(TARGET)
    goal["objects"]["carton"]["interior_halfspaces"] = [
        [1, 1, 0, -.3], [-1, -1, 0, -.3], [1, -1, -.2, -.3],
        [-1, 1, 0, -.3], [0, 0, 1, -.25]]
    return goal


def test_interior_planes_add_to_support_requirements():
    goal = interior_target()
    assert evaluate.score_attempt(trace(), goal)["success"]
    rows = trace()
    rows[-1]["objects"]["carton"]["support_contacts"] = ["floor"]
    assert evaluate.score_attempt(rows, goal)["objects"]["carton"]["terminal_reason"] == "wrong_support"
    assert evaluate.score_attempt(trace()[:-1], goal, require_settled=False)["success"]
    assert not evaluate.score_attempt(trace()[:-1], goal)["success"]


@pytest.mark.parametrize("corner", [[.2, .2, .1], [.2, -.2, .1], [0, 0, .3]])
def test_one_protruding_corner_fails_despite_center_and_outer_bounds_inside(corner):
    rows = trace()
    rows[-1]["objects"]["carton"]["corners"][7] = corner
    assert evaluate.score_attempt(rows, TARGET)["success"]
    result = evaluate.score_attempt(rows, interior_target())
    assert not result["success"]
    assert result["objects"]["carton"]["terminal_reason"] == "outside_target"


@pytest.mark.parametrize("planes", [[], [[0, 0, 0, 1]], [[1, 0, 0]],
                                    [[float("nan"), 0, 0, 0]], [[1, 0, 0, 0]] * 33])
def test_invalid_or_unbounded_interior_planes_rejected(planes):
    goal = copy.deepcopy(TARGET)
    goal["objects"]["carton"]["interior_halfspaces"] = planes
    with pytest.raises(ValueError):
        evaluate.score_attempt(trace(), goal)


def test_overflowing_plane_arithmetic_cannot_certify_containment():
    goal = copy.deepcopy(TARGET)
    goal["objects"]["carton"].update(bounds=[[-1e308] * 3, [1e308] * 3],
                                    interior_halfspaces=[[1e308, 0, 0, 0]])
    rows = trace()
    rows[-1]["objects"]["carton"]["corners"][0] = [-1e308, 0, 0]
    assert not evaluate.score_attempt(rows, goal)["success"]


def test_a_sample_without_robot_contacts_is_invalid_not_contact_free():
    """Every recorded session samples robot_contacts; one missing is not read as none."""
    rows = trace()
    del rows[-1]["objects"]["carton"]["robot_contacts"]
    result = evaluate.score_attempt(rows, TARGET)
    assert not result["success"] and "invalid_sample" in result["invalid_reasons"]


def with_bystander(rows, **change):
    """ROWS with an untargeted box resting on a shelf, CHANGE applied at the last sample."""
    for row in rows:
        box = copy.deepcopy(row["objects"]["carton"])
        box.update(position=[3, 0, .1], corners=[[x + 3, y, z] for x, y, z in box["corners"]],
                   support_contacts=["shelf"])
        row["objects"]["box"] = box
    rows[-1]["objects"]["box"].update(change)
    return rows


@pytest.mark.parametrize("change", [
    {"position": [3 + evaluate.DISTURBED_HORIZONTAL + .001, 0, .1]}, {"position": [3, .06, .1]},
    {"support_contacts": ["floor"]}, {"support_contacts": []}, {"hand_contacts": ["left"]},
    {"robot_contacts": ["torso_link"]}, {"quaternion_wxyz": [0, 1, 0, 0]}])
def test_an_untargeted_object_moved_tipped_dropped_or_held_fails_the_attempt(change):
    assert evaluate.score_attempt(with_bystander(trace()), TARGET)["untargeted"] == {"box": {"terminal_reason": None}}
    result = evaluate.score_attempt(with_bystander(trace(), **change), TARGET)
    assert not result["success"] and result["untargeted"]["box"]["terminal_reason"] == "disturbed"
    assert result["objects"]["carton"]["terminal_reason"] is None


def test_an_untargeted_object_nudged_within_tolerance_passes():
    nudged = with_bystander(trace(), position=[3.03, .03, .05])
    assert evaluate.score_attempt(nudged, TARGET, require_settled=True)["success"]


def test_an_object_that_starts_on_the_floor_may_stay_there():
    rows = with_bystander(trace())
    for row in rows:
        row["objects"]["box"]["support_contacts"] = ["floor"]
    assert evaluate.score_attempt(rows, TARGET, require_settled=True)["success"]


def test_rotated_overhanging_corner_fails_even_if_center_inside():
    rows = trace()
    rows[-1]["objects"]["carton"]["corners"][0] = [1.01, 0, 0.1]
    assert not evaluate.score_attempt(rows, TARGET)["success"]


def test_surface_scoring_accepts_supported_tilt_without_losing_height_guards():
    goal = {"schema_version": 1, "frame": "ws_map", "objects": {
        "bag": {"bounds": [[4.0, 1.0, -.339], [5.0, 2.0, .22]],
                "support_contacts": ["island"]}}}
    rows = trace()
    for row in rows:
        bag = row["objects"].pop("carton")
        bag.update(position=[4.624, 1.298, -.281], support_contacts=["island"],
                   corners=[[x, y, z] for x in [4.54, 4.71]
                            for y in [1.21, 1.38] for z in [-.350, -.21]])
        row["objects"]["bag"] = bag
    assert evaluate.score_attempt(rows, goal)["success"]

    below = copy.deepcopy(rows)
    below[-1]["objects"]["bag"]["position"][2] = -.34
    assert evaluate.score_attempt(below, goal)["objects"]["bag"]["terminal_reason"] == "outside_target"

    above = copy.deepcopy(rows)
    above[-1]["objects"]["bag"]["corners"][7][2] = .221
    assert evaluate.score_attempt(above, goal)["objects"]["bag"]["terminal_reason"] == "outside_target"

    inside_bin = copy.deepcopy(goal)
    inside_bin["objects"]["bag"]["interior_halfspaces"] = [[0, 0, 1, -.3]]
    through_floor = copy.deepcopy(rows)
    through_floor[-1]["objects"]["bag"]["corners"][0][2] = -.339 - evaluate.CONTACT_PENETRATION - .005
    assert evaluate.score_attempt(through_floor, inside_bin)["objects"]["bag"]["terminal_reason"] == "outside_target"


@pytest.mark.parametrize("change", ["gap", "reset", "warning", "fall", "invalid", "nan"])
def test_bad_evidence_cannot_become_success(change):
    rows = trace()
    if change == "gap":
        rows = rows[:2] + rows[5:]
    elif change == "reset":
        rows[2]["epoch"] = 1
    elif change == "warning":
        rows[0]["warnings"] = ["physics_unstable"]
    elif change == "fall":
        rows[0]["robot_upright"] = False
    elif change == "invalid":
        del rows[0]["physics_valid"]
    else:
        rows[0]["time"] = float("nan")
    assert not evaluate.score_attempt(rows, TARGET)["success"]


def test_all_objects_must_be_placed_together_at_the_terminal_sample():
    goal = copy.deepcopy(TARGET)
    goal["objects"]["bag"] = copy.deepcopy(goal["objects"]["carton"])
    rows = trace()
    for row in rows:
        row["objects"]["bag"] = copy.deepcopy(row["objects"]["carton"])
    rows[10]["objects"]["bag"]["hand_contacts"] = ["right_hand"]
    assert evaluate.score_attempt(rows, goal, require_settled=False)["success"]
    assert not evaluate.score_attempt(rows, goal)["success"]
    rows[-1]["objects"]["bag"]["hand_contacts"] = ["right_hand"]
    assert not evaluate.score_attempt(rows, goal, require_settled=False)["success"]


def test_model_claim_does_not_supply_physics(tmp_path):
    """A completion claim without a physics record is unscored, and one beside a failing
    record is a scored failure: only the recorded physics can make an attempt succeed."""
    (tmp_path / "outcome.json").write_text('{"status": "model_done", "finished_unix": 1, "model_claim": "done", "success": true}')
    result = evaluate.score_directory(tmp_path)
    assert not result["success"] and result["status"] == "unscored"
    rows = trace()
    rows[-1]["objects"]["carton"]["hand_contacts"] = ["right_hand"]
    (tmp_path / "target.json").write_text(json.dumps(TARGET))
    (tmp_path / "physics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = evaluate.score_directory(tmp_path)
    assert result["status"] == "scored" and result["execution_status"] == "model_done"
    assert not result["physical_success"] and not result["success"]


def test_empty_and_invalid_target_not_success():
    assert not evaluate.score_attempt([], TARGET)["success"]
    with pytest.raises(ValueError):
        evaluate.score_attempt(trace(), {"schema_version": 1, "frame": "ws_map", "objects": {}})


@pytest.mark.parametrize("value", [10**1000, float("inf"), float("nan")])
def test_overflowing_evidence_is_invalid_rather_than_a_scoring_crash(value):
    rows = trace()
    rows[0]["objects"]["carton"]["position"][0] = value
    result = evaluate.score_attempt(rows, TARGET)
    assert not result["success"]
    assert "invalid_sample" in result["invalid_reasons"]


@pytest.mark.parametrize("terminal", [None, "{", '{}',
    '{"status":"interrupted","finished_unix":1}',
    '{"status":"error","finished_unix":1}'])
def test_settled_prefix_without_completed_task_cannot_count_as_success(tmp_path, terminal):
    (tmp_path / "target.json").write_text(json.dumps(TARGET))
    (tmp_path / "physics.jsonl").write_text("\n".join(json.dumps(row) for row in trace()))
    if terminal is not None:
        (tmp_path / "outcome.json").write_text(terminal)
    result = evaluate.score_directory(tmp_path)
    assert result["physical_success"]
    assert not result["success"]


@pytest.mark.parametrize("status", ["physics_invalid", "skill_failed", "target_not_visible"])
def test_every_recorded_terminal_status_is_scored_and_only_a_completion_counts(tmp_path, status):
    """The session and the scripted fixture end tasks with these too; a valid record of a
    failed task is a scored failure, not an unreadable one."""
    (tmp_path / "target.json").write_text(json.dumps(TARGET))
    (tmp_path / "physics.jsonl").write_text("\n".join(json.dumps(row) for row in trace()))
    (tmp_path / "outcome.json").write_text(json.dumps({"status": status, "finished_unix": 1}))
    result = evaluate.score_directory(tmp_path)
    assert (result["status"], result["execution_status"]) == ("scored", status)
    assert result["physical_success"] and not result["success"]


def test_completion_is_the_model_or_the_scripted_sequence_finishing():
    assert evaluate.COMPLETED == {evaluate.MODEL_COMPLETED, evaluate.SCRIPTED_COMPLETED} == {
        "model_done", "skills_completed"}


@pytest.mark.parametrize("status", ["model_done", "skills_completed"])
def test_completed_task_still_needs_independent_physics_success(tmp_path, status):
    (tmp_path / "target.json").write_text(json.dumps(TARGET))
    (tmp_path / "outcome.json").write_text(json.dumps({"status": status, "finished_unix": 1}))
    path = tmp_path / "physics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in trace()))
    assert evaluate.score_directory(tmp_path)["success"]
    misplaced = sample(0)
    misplaced["objects"]["carton"]["position"][0] = 2
    for corner in misplaced["objects"]["carton"]["corners"]:
        corner[0] += 2
    path.write_text(json.dumps(misplaced))
    assert not evaluate.score_directory(tmp_path)["success"]


def test_directory_scores_settled_unless_asked_for_the_terminal_sample(tmp_path):
    (tmp_path / "target.json").write_text(json.dumps(TARGET))
    (tmp_path / "physics.jsonl").write_text("\n".join(json.dumps(row) for row in trace()[:5]))
    (tmp_path / "outcome.json").write_text('{"status":"model_done","finished_unix":1}')
    assert evaluate.score_directory(tmp_path, require_settled=False)["success"]
    assert not evaluate.score_directory(tmp_path)["success"]


def test_object_resting_on_another_object_is_supported_by_what_that_object_rests_on():
    goal = copy.deepcopy(TARGET)
    goal["objects"]["box"] = goal["objects"]["carton"]
    rows = trace()
    for row in rows:
        carton = row["objects"]["carton"]
        row["objects"]["box"] = {**copy.deepcopy(carton), "position": [0, 0, 0.3],
                                 "corners": [[c[0], c[1], c[2] + 0.2] for c in carton["corners"]],
                                 "support_contacts": [], "object_contacts": ["carton"]}
        carton["object_contacts"] = ["box"]
    assert evaluate.score_attempt(rows, goal)["success"]
    rows[-1]["objects"]["carton"]["support_contacts"] = ["floor"]
    assert evaluate.score_attempt(rows, goal)["objects"]["box"]["terminal_reason"] == "wrong_support"
    rows[-1]["objects"]["box"]["object_contacts"] = []
    assert evaluate.score_attempt(rows, goal)["objects"]["box"]["terminal_reason"] == "unsupported"


def test_a_container_floor_allows_the_soft_contact_tolerance_its_walls_do():
    """An object resting on a container's soft floor sinks a little into its contact, as
    one touching a wall may; the tolerance covers both, and deeper than that still fails."""
    goal = {"schema_version": 1, "frame": "ws_map", "objects": {
        "carton": {"bounds": [[0., 0., -1.2], [1., 1., -.5]], "support_contacts": ["bin"],
                   "interior_halfspaces": [[1., 0., 0., -1.]]}}}
    rows = trace()
    for row in rows:
        carton = row["objects"]["carton"]
        carton.update(position=[.5, .5, -1.18], support_contacts=["bin"],
                      corners=[[x, y, z] for x in [.4, .6] for y in [.4, .6] for z in [-1.2011, -1.16]])
    assert evaluate.score_attempt(rows, goal)["success"]
    sunk = copy.deepcopy(rows)
    for corner in sunk[-1]["objects"]["carton"]["corners"][::2]:
        corner[2] = -1.2 - evaluate.CONTACT_PENETRATION - .001
    assert evaluate.score_attempt(sunk, goal)["objects"]["carton"]["terminal_reason"] == "outside_target"


@pytest.mark.parametrize(("change", "reason"), [
    (lambda carton: carton["position"].__setitem__(0, carton["position"][0] + .004), "moving"),
    (lambda carton: carton.update(quaternion_wxyz=[0.9999500004166653, 0, 0, 0.009999833334166664]), "rotating"),
])
def test_terminal_placement_allows_motion_but_the_settled_score_rejects_it(change, reason):
    """Motion is the pose change since a sample at least MOTION_INTERVAL older; only the
    settled policy (the default) fails it."""
    rows = trace()
    change(rows[-1]["objects"]["carton"])
    terminal = evaluate.score_attempt(rows, TARGET, require_settled=False)
    settled = evaluate.score_attempt(rows, TARGET)
    assert terminal["success"] and terminal["objects"]["carton"]["terminal_reason"] is None
    assert not settled["success"] and settled["objects"]["carton"]["terminal_reason"] == reason


def test_instantaneous_contact_jitter_of_an_object_at_rest_does_not_fail_it():
    """Contact jitter at rest can exceed the angular limit for single steps; motion is
    judged over MOTION_INTERVAL, so the pose that does not move passes."""
    rows = trace()
    for row in rows[1:]:
        row["objects"]["carton"].update(angular_velocity=[0, 0, .3], linear_velocity=[.05, 0, 0])
    assert evaluate.score_attempt(rows, TARGET)["success"]
