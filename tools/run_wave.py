"""Run a wave of headless agent sessions and score each with the evaluator.

    run_wave.py configs/waves/default.json --model opus --output runs/waves/w1 [--jobs 2]
                [--repeat 3] [--keep-frames] [--terminal-only]

A wave file is a JSON list of tasks:

    {"name": "bags_to_table", "task": "Put both coffee bags on the foreground round table.",
     "targets": {"parcel_left": "empty_table", "parcel_corner": "empty_table"},
     "place_objects": ["juice_carton=3.19,3.42,-1.86"]}

`targets` maps each scored object to a scene entity (score_session.py --target), `"swap":
true` scores the sequence's swapped destinations, and `place_objects` starts objects
elsewhere (homebody --place-object). Each session runs in its own process under
OUTPUT/<name>[_<repeat>] and is scored with --perturb. summary.json is rewritten as each
one finishes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from homebody.session.config import MODELS

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    tasks = json.loads(Path(path).read_text())
    names = [task.get("name") for task in tasks]
    if not tasks or len(set(names)) != len(names) or not all(names) or not all(task.get("task") for task in tasks):
        raise SystemExit(f"error: {path} must list tasks, each with a unique name and a task")
    return tasks


def score_arguments(task, terminal_only=False):
    policy = ["--perturb"] + (["--terminal-only"] if terminal_only else [])
    if task.get("targets"):
        return [arg for name, entity in task["targets"].items() for arg in ("--target", f"{name}={entity}")] + policy
    return (["--swap"] if task.get("swap") else []) + policy


def run_one(task, name, args):
    """Run and score one session and return its summary row."""
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "homebody", "--headless", "--root", str(ROOT), "--model", args.model,
               "--task", task["task"], "--output", str(output)]
    for spawn in task.get("place_objects", []):
        command += ["--place-object", spawn]
    if args.settings:
        command += ["--settings", str(args.settings)]
    env = dict(os.environ)  # carries the MUJOCO_GL that importing homebody set
    earlier = set(output.glob("session-*"))  # a rerun scores only its own session
    start = time.monotonic()
    with open(output / "log.txt", "w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {"name": name, "task": task["task"], "wall_s": round(time.monotonic() - start)}
    sessions = sorted(set(output.glob("session-*")) - earlier)
    if not sessions:
        return {**row, "verdict": "NO_SESSION", "log": str(output / "log.txt")}
    scored = subprocess.run([sys.executable, str(ROOT / "tools/score_session.py"), str(sessions[-1]),
                             *score_arguments(task, args.terminal_only)], env=env, capture_output=True, text=True, check=False)
    try:
        report = json.loads(scored.stdout)
    except json.JSONDecodeError:
        return {**row, "verdict": "SCORE_ERROR", "session": str(sessions[-1]), "error": scored.stderr[-500:]}
    if args.cockpit:
        subprocess.run([sys.executable, str(ROOT / "tools/compose_cockpit.py"), str(sessions[-1])],
                       env=env, capture_output=True, check=False)
    if not args.keep_frames:
        shutil.rmtree(sessions[-1] / "frames", ignore_errors=True)
    failed = {item: result["terminal_reason"] for item, result in {**report["objects"], **report["untargeted"]}.items()
              if result["terminal_reason"]}
    unrefused = sorted(reason for reason, refused in report["refusals"].items() if not refused)
    verdict = "FAIL" if not report["success"] else "UNSOUND" if unrefused else "SUCCESS" if report["passed"] else "FAIL"
    return {**row, "verdict": verdict, "scoring_policy": report["scoring_policy"],
            "task_status": report["task_status"], "failed_objects": failed, "unrefused": unrefused,
            "invalid": report.get("invalid_reasons", []), "session": str(sessions[-1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wave", type=Path)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Sessions at once (each renders on the GPU)")
    parser.add_argument("--repeat", type=int, default=1, help="Runs of each task")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Score the last sample alone, without the stillness window (lenient)")
    parser.add_argument("--cockpit", action="store_true",
                        help="Render each session's cockpit.mp4 from its frames before they are deleted")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.repeat < 1:
        parser.error("--jobs and --repeat must be positive")
    args.output = args.output.resolve()
    runs = [(task, task["name"] if args.repeat == 1 else f"{task['name']}_{index}")
            for task in load(args.wave) for index in range(args.repeat)]
    rows = []

    def finished(row):
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(
            {"model": args.model, "wave": str(args.wave), "rows": rows}, indent=2))
        print(f"{row['verdict']:<11} {row['name']:<28} {row['wall_s']:>5} s  {row.get('failed_objects') or ''}",
              flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for future in [pool.submit(run_one, task, name, args) for task, name in runs]:
            finished(future.result())
    passed = sum(row["verdict"] == "SUCCESS" for row in rows)
    print(f"{passed} of {len(rows)} succeeded ({args.model}); {args.output / 'summary.json'}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
