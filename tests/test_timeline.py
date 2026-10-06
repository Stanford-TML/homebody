"""timeline.py puts decisions, results and object contact changes on one simulated-time axis."""
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMELINE = runpy.run_path(str(ROOT / "tools/timeline.py"))


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_a_drop_shows_between_the_decision_and_its_result(tmp_path, capsys):
    task = tmp_path / "tasks/task-000"
    task.mkdir(parents=True)
    write(tmp_path / "events.jsonl", [
        {"kind": "frame", "simulation_time": 10.0},
        {"kind": "decision", "step": 0, "simulation_time": 11.0,
         "decision": {"skill": "navigate", "arguments": {"goal_xy_m": [1, 2]}, "text": "Walk"}},
        {"kind": "skill", "simulation_time": 12.0, "event": {"type": "skill_stage", "stage": "Following route"}},
        {"kind": "result", "step": 0, "simulation_time": 20.0,
         "result": {"code": "CARRY_CLEARED", "message": "dropped"}},
        {"kind": "frame", "simulation_time": 21.0},
        {"kind": "task_finished", "status": "model_done", "steps": 2, "model_claim": "Gave up"}])

    def state(hand, support):
        return {"hand_contacts": hand, "support_contacts": support}
    write(task / "physics.jsonl", [
        {"time": 10.0, "objects": {"box": state(["right"], [])}},
        {"time": 15.0, "objects": {"box": state([], [])}},
        {"time": 15.5, "objects": {"box": state([], ["floor"])}}])
    assert TIMELINE["main"]([str(tmp_path), "--stages"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(" s  ")[0].strip() for line in lines] == ["11.0", "12.0", "15.0", "15.5", "20.0", "21.0"]
    assert "box: left the right hand" in lines[2] and "resting on floor" in lines[3]
    assert "CARRY_CLEARED (9 s): dropped" in lines[4] and "TASK model_done after 2 steps: Gave up" in lines[5]
