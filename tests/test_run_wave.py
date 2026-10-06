"""run_wave.py: a wave file's tasks, how each is scored, and the summary it keeps."""
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WAVE = runpy.run_path(str(ROOT / "tools/run_wave.py"))


@pytest.mark.parametrize("wave", ["default.json", "other_tasks.json", "fun.json"])
def test_the_shipped_waves_load(wave):
    assert WAVE["load"](ROOT / "configs/waves" / wave)


def test_a_task_is_scored_settled_and_perturbed_by_its_targets_its_swap_or_the_sequence():
    assert WAVE["score_arguments"]({"targets": {"a": "shelf", "b": "island"}}) == [
        "--target", "a=shelf", "--target", "b=island", "--perturb"]
    assert WAVE["score_arguments"]({"swap": True}) == ["--swap", "--perturb"]
    assert WAVE["score_arguments"]({}) == ["--perturb"]
    assert WAVE["score_arguments"]({}, terminal_only=True) == ["--perturb", "--terminal-only"]


@pytest.mark.parametrize(("success", "passed", "refusals", "verdict"), [
    (True, True, {"hand_contact": True}, "SUCCESS"), (True, False, {"hand_contact": True}, "FAIL"),
    (True, False, {"hand_contact": False}, "UNSOUND"), (False, False, {"hand_contact": False}, "FAIL")])
def test_a_session_verdict_is_the_scorers_pass_and_every_refusal(tmp_path, monkeypatch, success, passed, refusals,
                                                                  verdict):
    report = {"passed": passed, "success": success, "scoring_policy": "settled_placement", "task_status": "model_done",
              "objects": {"a": {"terminal_reason": None}}, "untargeted": {"b": {"terminal_reason": "disturbed"}},
              "refusals": refusals, "invalid_reasons": []}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "-m" in command:
            (tmp_path / "out/t/session-1").mkdir(parents=True)
        return type("Done", (), {"stdout": json.dumps(report), "stderr": ""})()
    monkeypatch.setattr(WAVE["subprocess"], "run", run)
    args = type("Args", (), {"output": tmp_path / "out", "model": "opus", "settings": None, "cockpit": False,
                             "keep_frames": True, "terminal_only": False})()
    row = WAVE["run_one"]({"task": "t"}, "t", args)
    assert row["verdict"] == verdict and row["failed_objects"] == {"b": "disturbed"}
    assert calls[-1][-1] == "--perturb"


@pytest.mark.parametrize("tasks", [[], [{"name": "a"}], [{"name": "a", "task": "t"}, {"name": "a", "task": "u"}]])
def test_a_wave_needs_named_unique_tasks(tmp_path, tasks):
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(tasks))
    with pytest.raises(SystemExit):
        WAVE["load"](path)


def test_the_summary_counts_each_repeat(tmp_path, monkeypatch, capsys):
    path = tmp_path / "wave.json"
    path.write_text(json.dumps([{"name": "a", "task": "t"}, {"name": "b", "task": "u"}]))
    verdicts = iter(["SUCCESS", "FAIL", "SUCCESS", "SUCCESS"])
    main = WAVE["main"]
    monkeypatch.setitem(main.__globals__, "run_one",
                        lambda task, name, args: {"name": name, "task": task["task"], "wall_s": 1,
                                                  "verdict": next(verdicts)})
    assert main([str(path), "--model", "opus", "--output", str(tmp_path / "out"), "--repeat", "2"]) == 1
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert [row["name"] for row in summary["rows"]] == ["a_0", "a_1", "b_0", "b_1"]
    assert "3 of 4 succeeded (opus)" in capsys.readouterr().out
