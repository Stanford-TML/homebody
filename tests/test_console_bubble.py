"""Model and provider text is shown as text, never as executable browser markup."""
from types import SimpleNamespace

from homebody.ui.console import stage_progress, thought_bubble


def test_model_explanation_is_escaped_in_bubble():
    output = thought_bubble('<script>alert("run")</script> & inspect the carton')
    assert "<script>" not in output
    assert '&lt;script&gt;alert(&quot;run&quot;)&lt;/script&gt; &amp; inspect' in output


def test_thinking_state_is_labelled_without_invented_model_reasoning():
    output = thought_bubble(thinking=True)
    assert "Agent Thinking about the next step…" in output
    assert '<div class="hb-message"></div>' in output


def test_selected_model_is_named_and_escaped():
    output = thought_bubble("Inspecting", model='gpt-6-astra <script>', thinking=True)
    assert "Agent: gpt-6-astra &lt;script&gt; Thinking about the next step…" in output
    assert "<script>" not in output


def test_claude_model_is_named_without_decorative_dots():
    from homebody.session.config import Agent
    from homebody.vlm.catalog import model_label

    output = thought_bubble("Inspecting the scene", model=model_label(Agent(model="opus")),
                            thinking=True)
    assert "Agent: Claude (claude-opus-5-5, high effort) Thinking about the next step…" in output
    assert "hb-dots" not in output


def test_skill_progress_highlights_current_stage_and_keeps_completed_steps():
    steps = ["Preparing", "Opening hand", "Approaching"]
    running = stage_progress("pick", steps)
    assert "Pick: Approaching" in running and "Stage 3" in running
    assert running.count('class="is-done"') == 2
    assert '<li class="is-current">Approaching</li>' in running
    finished = stage_progress("pick", steps, SimpleNamespace(code="OK", message="Lift verified"))
    assert "Result: OK" in finished and "Lift verified" in finished
    assert 'class="is-current"' not in finished
    failed = stage_progress("pick", steps, SimpleNamespace(code="TARGET_LOST", message="No view"))
    assert '<li class="is-failed">Approaching</li>' in failed


def test_stage_clock_updates_and_freezes_at_the_result(monkeypatch):
    from fakes import mocked_console

    clock = [10.]
    monkeypatch.setattr("homebody.ui.console.time.monotonic", lambda: clock[0])
    console = mocked_console(monkeypatch)
    console.skill_event({"type": "skill_stage", "skill": "pick", "stage": "Preparing",
                         "simulation_time": 5.})
    clock[0] = 12.5
    console.skill_event({"type": "skill_stage", "skill": "pick", "stage": "Approaching",
                         "simulation_time": 6.})
    clock[0] = 18.
    console._refresh_stage(simulation_time=9.)
    assert "8.0s elapsed (4.0s sim)" in console.stage.content
    assert 'Preparing<span class="hb-step-time">2.5s</span>' in console.stage.content
    assert 'Approaching<span class="hb-step-time">5.5s</span>' in console.stage.content
    console.skill_event({"type": "skill_finished", "result": SimpleNamespace(
        skill="pick", code="OK", message="Held"), "simulation_time": 9.})
    clock[0] = 40.
    console._refresh_stage(simulation_time=20.)
    assert "8.0s elapsed (4.0s sim)" in console.stage.content


def test_model_and_provider_text_reach_the_page_only_as_escaped_html(monkeypatch):
    """viser renders markdown as MDX, where `{...}` is JavaScript the browser runs."""
    from html import escape

    from fakes import mocked_console
    console = mocked_console(monkeypatch)
    claim = 'Placed {["a", "b"].join("+")} <b>here</b> & stopped'
    console.update("Executing", claim)
    assert console.explanation.content == "" and escape(claim) in console.thought.content
    console.update("model_done", claim)
    assert console.explanation.kind == "add_html" and escape(claim) in console.explanation.content
    assert console.status.kind == "add_markdown" and console.status.content == "Model reports done"
    console.update("provider_error", "Claude Code exited 1: {'error': '<overloaded>'}")
    assert console.status.content == "Provider error" and "&lt;overloaded&gt;" in console.explanation.content
