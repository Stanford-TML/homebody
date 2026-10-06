"""Transport checks use an explicit test executable, never a substituted live provider."""
import json
import sys
import time

import pytest

from homebody.session.settings import Agent
from homebody.vlm.claude import ClaudeCode
from homebody.vlm.codex import DISABLED_FEATURES, Codex
from homebody.vlm.provider import Cancelled, Decision, ProviderError, schema


@pytest.mark.parametrize("value", [
    '{}', '[]', '{"text":"a","skill":"invented","arguments_json":"{}"}',
    '{"text":"a","skill":"done","arguments_json":"[]"}',
    '{"text":"a","skill":"done","arguments_json":"{\\"x\\":NaN}"}',
])
def test_decision_rejects_invalid_shapes(value):
    with pytest.raises(ProviderError):
        Decision.parse(value, ("done",))


def test_tool_surfaces_disabled_and_no_asset_paths(tmp_path):
    command = Codex(Agent(), executable=sys.executable).command(
        tmp_path, (tmp_path / "observation.png",))
    for feature in ("shell_tool", "unified_exec", "view_image", "plugins", "apps"):
        assert feature in DISABLED_FEATURES
        index = command.index(feature)
        assert command[index - 1] == "--disable"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="high"' in command  # astra's own effort
    assert 'model_reasoning_effort="xhigh"' in Codex(Agent(reasoning_effort="max"),
                                                     executable=sys.executable).command(tmp_path, ())
    assert 'model_reasoning_effort="medium"' in Codex(Agent(reasoning_effort="medium"),
                                                      executable=sys.executable).command(tmp_path, ())
    assert command[-1] == "-"
    assert not any("assets" in part or "evaluation" in part for part in command)


@pytest.fixture
def executable(tmp_path):
    def create(mode):
        path = tmp_path / f"test-transport-{mode}"
        source = '''import json
from pathlib import Path
import sys
import time
args = sys.argv
prompt = sys.stdin.read()
time.sleep(0.025)
'''
        if mode == "success":
            source += '''output = Path(args[args.index("--output-last-message") + 1])
output.write_text(json.dumps({"text":"Observed", "skill":"done", "arguments_json":"{}"}))
for event in ({"type":"thread.started"}, {"type":"turn.started"},
              {"type":"item.completed","item":{"type":"agent_message","text":"Observed"}},
              {"type":"turn.completed"}):
    print(json.dumps(event), flush=True)
'''
        elif mode == "slow":
            source += "time.sleep(10)\n"
        elif mode == "error":
            source += "print('Explicit test transport failure', file=sys.stderr)\nsys.exit(1)\n"
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o700)
        return str(path)
    return create


def test_provider_ticks_retains_input_and_structured_output(tmp_path, executable):
    ticks = []
    image = tmp_path / "current.png"
    image.write_bytes(b"fixture")
    attempt = tmp_path / "attempt"

    def tick():
        ticks.append(1)
        time.sleep(0.005)

    decision = Codex(Agent(), executable("success")).generate(
        "my prompt", images=(image,), names=("done",), attempt=attempt, tick=tick,
        cancelled=lambda: False)
    assert ticks
    assert decision.skill == "done"
    assert (attempt / "prompt.txt").read_text() == "my prompt"
    assert (attempt / "observation-0.png").read_bytes() == b"fixture"
    assert json.loads((attempt / "status.json").read_text())["status"] == "ok"
    assert (attempt / "decision.json").is_file()


@pytest.mark.parametrize("mode, agent, cancelled, error, match, status, returncode", [
    ("slow", Agent(), True, Cancelled, None, "cancelled", "killed"),
    ("slow", Agent(timeout=0.05), False, ProviderError, "timed out", "error", "killed"),
    ("error", Agent(), False, ProviderError, "Explicit test transport failure", "error", 1),
])
def test_provider_cancel_timeout_and_transport_failure_terminate_without_fallback(
        tmp_path, executable, mode, agent, cancelled, error, match, status, returncode):
    attempt = tmp_path / "attempt"
    with pytest.raises(error, match=match):
        Codex(agent, executable(mode)).generate(
            "prompt", images=(), names=("done",), attempt=attempt,
            tick=lambda: time.sleep(0.005), cancelled=lambda: cancelled)
    recorded = json.loads((attempt / "status.json").read_text())
    assert recorded["status"] == status
    assert (recorded["returncode"] < 0) if returncode == "killed" else (recorded["returncode"] == returncode)
    assert not (attempt / "decision.json").exists()


def test_claude_code_runs_restricted_on_its_attachments_only(tmp_path, monkeypatch):
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("TEST_CLAUDE_SEEN", str(seen))
    fake = tmp_path / "test-claude"
    fake.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
json.dump({"argv": sys.argv[1:], "stdin": sys.stdin.read(), "files": sorted(os.listdir())},
          open(os.environ["TEST_CLAUDE_SEEN"], "w"))
reply = {"text": "Observed", "skill": "done", "arguments_json": "{}"}
print(json.dumps({"type": "result", "structured_output": reply}))
''')
    fake.chmod(0o700)
    image = tmp_path / "current.png"
    image.write_bytes(b"fixture")
    attempt = tmp_path / "attempt"
    decision = ClaudeCode(Agent(model="opus"), str(fake)).generate(
        "my prompt", images=(image,), names=("done",), attempt=attempt,
        tick=lambda: time.sleep(0.005), cancelled=lambda: False)
    assert decision.skill == "done"
    assert json.loads(seen.read_text()) == {
        "argv": ["--print", "--output-format", "json", "--restricted", "--safe-mode", "--strict-mcp-config",
                 "--json-schema", json.dumps(schema(("done",))), "--tools", "Read",
                 "--allowedTools", "Read(./observation-*)", "--permission-mode", "dontAsk", "--no-session-persistence", "--effort", "high", "--model", "claude-opus-5-5"],
        "stdin": "my prompt\nAttachment 1: read ./observation-0.png",
        "files": ["observation-0.png"]}
    for name in ("prompt.txt", "stdout.json", "stderr.txt"):
        assert (attempt / name).is_file()
    assert json.loads((attempt / "status.json").read_text())["status"] == "ok"


@pytest.mark.parametrize("arguments", [
    '{"x":1e999}', '{"x":-1e999}', '{"x":' + str(10 ** 1000) + '}',
    '{"x":-' + str(10 ** 1000) + '}', '{"nested":[{"x":1e999}]}',
    '{"nested":[{"x":' + str(10 ** 1000) + '}]}',
])
def test_decision_rejects_nonfinite_or_out_of_range_numeric_leaves(arguments):
    raw = json.dumps({"text": "Move", "skill": "navigate", "arguments_json": arguments})
    with pytest.raises(ProviderError, match="finite floating-point range"):
        Decision.parse(raw, ("navigate",))


def test_finite_numeric_arguments_remain_valid():
    arguments = {"x": 0, "nested": [sys.float_info.max, -sys.float_info.max, 10 ** 308],
                 "option": True, "label": "text"}
    raw = json.dumps({"text": "Observe", "skill": "done",
                      "arguments_json": json.dumps(arguments)})
    assert Decision.parse(raw, ("done",)).arguments == arguments


def test_a_model_is_chosen_by_one_name_that_fixes_its_provider_and_id():
    assert (Agent(model="astra").provider, Agent(model="astra").model_id) == ("codex", "gpt-6-astra")
    assert (Agent(model="opus").provider, Agent(model="opus").model_id) == ("claude-code", "claude-opus-5-5")
    with pytest.raises(ValueError, match="agent.model must be one of astra"):
        Agent(model="gpt-6-astra")


def test_every_named_model_has_a_provider_and_a_way_to_check_it():
    from homebody.session.config import MODELS
    from homebody.vlm.catalog import PROVIDERS, requirement
    for name, (provider, _, effort) in MODELS.items():
        assert provider in PROVIDERS and PROVIDERS[provider].LABEL, name
        executable, key = requirement(provider)
        assert (executable is None) != (key is None), name
        assert (effort is not None) == (executable is not None), name  # only the CLIs take an effort


def test_claude_code_runs_at_its_model_effort_high_by_default():
    command = ClaudeCode(Agent(model="opus"), executable=sys.executable).command(("done",))
    assert command[command.index("--effort") + 1] == "high"



@pytest.mark.parametrize("model, expected", [
    ("astra", ("codex", "gpt-6-astra", "high")), ("astra-api", ("openai-api", "gpt-6-astra", None)),
    ("opus", ("claude-code", "claude-opus-5-5", "high")), ("opus-api", ("claude-api", "claude-opus-5-5", None))])
def test_every_model_is_a_named_choice(model, expected):
    agent = Agent(model=model)
    assert (agent.provider, agent.model_id, agent.effort) == expected


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="agent.model must be one of"):
        Agent(model="claude-code:claude-sonnet-5-5")


def test_a_busy_service_is_told_apart_from_a_failure_and_the_cli_reason_is_reported(tmp_path):
    from homebody.vlm.provider import ProviderBusy, failure, reported_errors
    assert isinstance(failure("Codex exited 1: Selected model is at capacity."), ProviderBusy)
    assert isinstance(failure("HTTP 429: Too Many Requests"), ProviderBusy)
    assert not isinstance(failure("Codex exited 1: invalid schema"), ProviderBusy)
    out = tmp_path / "stdout.jsonl"
    out.write_text('{"type":"turn.started"}\n{"type":"error","message":"Selected model is at capacity."}\n')
    assert reported_errors(out) == "Selected model is at capacity."


def test_a_claude_code_overload_is_read_from_its_result_envelope(tmp_path):
    from homebody.vlm.provider import ProviderBusy, failure, reported_errors
    stdout = tmp_path / "stdout.json"
    stdout.write_text(json.dumps({"is_error": True, "api_error_status": 529, "terminal_reason": "api_error",
                                  "result": "API Error: 529 Overloaded. This is a server-side issue"}))
    assert isinstance(failure(f"Claude Code exited 1: {reported_errors(stdout)}"), ProviderBusy)
