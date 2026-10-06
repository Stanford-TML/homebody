"""Print a session's decisions, skill results and object contact changes by simulated time.

    timeline.py runs/session-<stamp>-<id>/ [--stages] [--task N]
"""
import argparse
import json
from pathlib import Path


def events(session):
    """(simulated seconds, kind, event) for each event, in recorded order."""
    now = 0.0
    for line in (session / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if "simulation_time" in event:
            now = event["simulation_time"]
            if event["kind"] in ("frame", "observer_frame"):
                continue
        yield now, event["kind"], event


def object_changes(physics):
    """(simulated seconds, text) at each change of an object's hand or support contacts."""
    last = {}
    for line in physics.read_text().splitlines():
        sample = json.loads(line)
        for name, state in sample["objects"].items():
            now = (tuple(state["hand_contacts"]), tuple(state["support_contacts"]))
            before = last.get(name)
            last[name] = now
            if before is None or before == now:
                continue
            if now[0] and not before[0]:
                yield sample["time"], f"{name}: in the {'/'.join(now[0])} hand"
            elif before[0] and not now[0]:
                yield sample["time"], f"{name}: left the {'/'.join(before[0])} hand"
            if now[1] != before[1]:
                where = ", ".join(now[1]) if now[1] else "nothing (falling or held)"
                yield sample["time"], f"{name}: resting on {where}"


def rows(session, task_index, stages):
    task = sorted(session.glob("tasks/task-*"))[task_index]
    decisions = {}
    for now, kind, event in events(session):
        if kind == "decision":
            decision = event["decision"]
            decisions[event["step"]] = now
            arguments = decision.get("arguments") or decision.get("arguments_json")
            yield now, f"#{event['step']:<3} {decision['skill']:<9} {json.dumps(arguments)}  \"{decision.get('text', '')}\""
        elif kind == "result":
            result = event["result"]
            took = now - decisions.get(event["step"], now)
            yield now, f"     -> {result['code']} ({took:.0f} s): {result.get('message', '')}"
        elif kind == "skill" and stages and event["event"].get("type") == "skill_stage":
            yield now, f"        . {event['event']['stage']}"
        elif kind == "skill" and "reason" in event["event"]:
            detail = event["event"]
            yield now, f"        ! {detail['type']}{' (' + detail['side'] + ')' if 'side' in detail else ''}: {detail['reason']}"
        elif kind == "task_finished":
            claim = event.get("model_claim") or event.get("reason") or ""
            yield now, f"TASK {event['status']} after {event.get('steps')} steps: {claim}"
    physics = task / "physics.jsonl"
    if physics.exists():
        for now, text in object_changes(physics):
            yield now, f"   ~ {text}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--stages", action="store_true", help="Also each skill stage")
    parser.add_argument("--task", type=int, default=-1, help="Task index in the session (default: last)")
    args = parser.parse_args(argv)
    if not (args.session / "events.jsonl").exists():
        raise SystemExit(f"error: {args.session} has no events.jsonl")
    for now, text in sorted(rows(args.session, args.task, args.stages), key=lambda row: row[0]):
        print(f"{now:8.1f} s  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
