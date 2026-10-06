"""Score a recorded session against the fixture's evaluator targets.

    score_session.py runs/session-* [--swap] [--only juice_carton ...] [--perturb] [--task task-000]
    score_session.py runs/session-* --target juice_carton=empty_table [--target ...]

`--target OBJECT=ENTITY` scores an object against a scene entity (an `id` in the scene
package's semantics.json). An entity the sequence does not already score becomes a surface
target from its scanned bounds. `--perturb` also checks that each tampered record fails for
its own reason. Every object no target names must stay where it started.
"""
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path

from homebody.session.config import DEFAULT_SCENE

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 3.0
SURFACE_BAND = (0.06, 0.5)  # m below and above a surface's top


def shifted(state, dx):
    state["position"][0] += dx
    for corner in state["corners"]:
        corner[0] += dx


OBJECT_REFUSALS = {
    "hand_contact": lambda state, step, span: state.update(hand_contacts=["right"]),
    "robot_contact": lambda state, step, span: state.update(robot_contacts=["right_elbow_link"]),
    "outside_target": lambda state, step, span: shifted(state, span + 0.5),
    "unsupported": lambda state, step, span: state.update(support_contacts=[], object_contacts=[]),
    "moving": lambda state, step, span: shifted(state, 0.006 * (1 + step % 2)),
}
ENDS_IN_PLACE = {"moving"}  # last sample untouched, so only stillness can catch it
SAMPLE_REFUSALS = {"invalid_physics": "physics_valid", "robot_fall": "robot_upright"}


def exchanged(targets):
    """TARGETS with its two destinations exchanged between the two groups of objects."""
    groups = {}
    for name, target in targets.items():
        groups.setdefault(json.dumps(target, sort_keys=True), []).append(name)
    if len(groups) != 2:
        raise SystemExit(f"error: --swap needs exactly two destinations, the sequence has {len(groups)}")
    first, second = groups.values()
    return {**{name: targets[second[0]] for name in first}, **{name: targets[first[0]] for name in second}}


def refusals(rows, target, swapped, score, settled, disturbance):
    """Whether each tampering of the record's last WINDOW seconds fails the verdict with its
    own reason."""
    end = rows[-1]["time"]  # the first sample stays, it is where every object started
    first = max(1, next(index for index, row in enumerate(rows) if row["time"] >= end - WINDOW))

    def scored(change, last=True):
        copies = rows[:first] + copy.deepcopy(rows[first:])
        for step, row in enumerate(copies[first:] if last else copies[first:-1], first):
            change(row, step)
        return score(copies, target)

    def refused(name, change, span, reason, kind="objects"):
        result = scored(lambda row, step: change(row["objects"][name], step, span), reason not in ENDS_IN_PLACE)
        return not result["success"] and result[kind][name]["terminal_reason"] == reason

    results = {}
    for reason, change in OBJECT_REFUSALS.items():
        if reason == "moving" and not settled:
            continue
        results[reason] = all(refused(name, change, goal["bounds"][1][0] - goal["bounds"][0][0], reason)
                              for name, goal in target["objects"].items())
    others = sorted(rows[-1]["objects"].keys() - target["objects"].keys())
    if others:
        results["disturbed"] = all(refused(name, lambda state, step, span: shifted(state, span), disturbance,
                                           "disturbed", "untargeted") for name in others)
    for reason, flag in SAMPLE_REFUSALS.items():
        result = scored(lambda row, step, flag=flag: row.update({flag: False}))
        results[reason] = not result["success"] and reason in result["invalid_reasons"]
    if swapped is not None:
        results["exchanged_destinations"] = not score(rows, {**target, "objects": swapped})["success"]
    return results


def entity_targets(args, parser, sequence_targets):
    """Each --target's evaluator target, from the sequence or the entity's scanned bounds."""
    entities = json.loads((args.scene / "semantics.json").read_text())["entities"]
    bounds = {entity["id"]: entity["bounds_ws_map_m"] for entity in entities}
    known = {target["support_contacts"][0]: target for target in sequence_targets.values()}
    targets = {}
    for item in args.target:
        name, _, entity = item.partition("=")
        if entity in known:
            targets[name] = known[entity]
        elif entity in bounds:
            low, high = bounds[entity]
            targets[name] = {"bounds": [[low[0], low[1], high[2] - SURFACE_BAND[0]],
                                        [high[0], high[1], high[2] + SURFACE_BAND[1]]],
                             "support_contacts": [entity]}
        else:
            parser.error(f"--target {item!r}: {entity!r} is not an entity id in {args.scene / 'semantics.json'}")
    return targets


def warn_if_other_task(path, sequence, scene):
    """Warn when the typed task is not the one the sequence's targets score."""
    typed = json.loads(path.read_text()).get("task") if path.exists() else None
    expected = sequence.get("task") or json.loads((scene / "task_scene.json").read_text())["default_task"]
    if typed != expected:
        print(f"warning: the session's task {typed!r} is not {expected!r}, the task the sequence's targets "
              "score; for another task name each object's destination with --target OBJECT=ENTITY",
              file=sys.stderr)


def samples(path):
    """The physics record's samples. Exits naming a malformed line."""
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"error: {path} line {number} is not a JSON sample ({error.msg})") from error
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--sequence", type=Path, default=ROOT / "configs/continuous_four.json",
                        help="The scripted sequence whose evaluator targets score the objects")
    parser.add_argument("--swap", action="store_true", help="Exchange the sequence's two destinations")
    parser.add_argument("--only", nargs="+", default=None, help="Score only these objects")
    parser.add_argument("--perturb", action="store_true",
                        help="Also check that each tampered record fails for its own reason")
    parser.add_argument("--target", action="append", default=[], metavar="OBJECT=ENTITY",
                        help="Score OBJECT against the scene entity ENTITY (an id in semantics.json)")
    parser.add_argument("--scene", type=Path, default=ROOT / DEFAULT_SCENE, help="The scene package")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Score the last sample alone, without the two-second stillness window (lenient)")
    parser.add_argument("--task", help="The task directory to score (tasks/TASK) when the session holds several")
    args = parser.parse_args(argv)
    sequence = json.loads(args.sequence.read_text())
    targets = {step["fixture_id"]: step["evaluator_target"] for step in sequence["steps"]}
    swapped = exchanged(targets)
    if args.swap:
        targets, swapped = swapped, targets
    if args.only:
        unknown = sorted(set(args.only) - targets.keys())
        if unknown:
            parser.error(f"--only names objects {args.sequence} has no target for: {unknown}")
        targets = {name: targets[name] for name in args.only}
        swapped = {name: swapped[name] for name in args.only}
    if args.target:
        if args.swap or args.only:
            parser.error("--target names its own destinations; it takes no --swap or --only")
        targets, swapped = entity_targets(args, parser, targets), None
    recorded = sorted(args.session.glob(f"tasks/{args.task or 'task-*'}/physics.jsonl"))
    if not recorded:
        raise SystemExit(f"error: {args.session} holds no recorded task" + (f" {args.task}" if args.task else ""))
    if len(recorded) > 1:
        raise SystemExit(f"error: {args.session} holds {len(recorded)} tasks "
                         f"({', '.join(path.parent.name for path in recorded)}); name one with --task")
    physics = recorded[0]
    if not (args.target or args.swap or args.only):
        warn_if_other_task(physics.parent / "task.json", sequence, args.scene)
    evaluator = runpy.run_path(ROOT / "evaluation/evaluate.py")
    settled = not args.terminal_only

    def score_policy(rows, target):
        return evaluator["score_attempt"](rows, target, require_settled=settled)
    rows, target = samples(physics), {"schema_version": 1, "frame": "ws_map", "objects": targets}
    score = score_policy(rows, target)
    outcome = physics.parent / "outcome.json"
    status = json.loads(outcome.read_text()).get("status") if outcome.exists() else None
    report = {**score, "task": physics.parent.name, "task_status": status}
    if args.perturb:
        report["refusals"] = refusals(rows, target, swapped, score_policy, settled,
                                      2 * evaluator["DISTURBED_HORIZONTAL"])
    report["passed"] = bool(score["success"] and status in evaluator["COMPLETED"]
                            and all(report.get("refusals", {}).values()))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
