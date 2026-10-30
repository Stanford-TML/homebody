"""The REAL view displays snapshots and enqueues commands without accessing a backend."""
from types import SimpleNamespace

from homebody.ui.console import Console, Controls


class Widget(SimpleNamespace):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_click(self, callback):
        self.click_callback = callback

    def on_update(self, callback):
        self.update_callback = callback

    def click(self):
        self.click_callback(None)

    def remove(self):
        self.removed = True


class Gui:
    def __init__(self):
        self.widgets = []

    def __getattr__(self, method):
        assert method.startswith("add_")
        def add(*args, **kwargs):
            widget = Widget(kind=method, args=args, removed=False, disabled=False,
                            visible=True, value=kwargs.get("initial_value"), **{
                                key: value for key, value in kwargs.items()
                                if key not in ("disabled", "visible", "initial_value")})
            widget.disabled = kwargs.get("disabled", False)
            widget.visible = kwargs.get("visible", True)
            if method in ("add_html", "add_markdown"):
                widget.content = args[0]
            if method == "add_image":
                widget.image = args[0]
            if method in ("add_button", "add_checkbox"):
                widget.label = args[0]
            self.widgets.append(widget)
            return widget
        return add


def console(monkeypatch, workflow="real"):
    import homebody.ui.console as module
    server = SimpleNamespace(gui=Gui(), stop=lambda: None)
    opened = []
    monkeypatch.setattr(module, "claim", lambda *args: None)
    def open_server(*args):
        opened.append(args)
        return server
    monkeypatch.setattr(module, "open_server", open_server)
    controls = Controls()
    view = Console(controls, scene_name="Measured room", workflow=workflow)
    return view, controls, opened


def test_real_stop_and_end_enqueue_distinct_session_commands(monkeypatch):
    view, controls, _ = console(monkeypatch)
    view.stop_button.click()
    assert controls.cancel.is_set() and controls.next().name == "stop"
    assert not controls.closing.is_set()
    view.end_button.click()
    assert [controls.next().name, controls.next().name] == ["stop", "end"]
    assert not controls.closing.is_set()


def test_pause_resume_is_control_only_and_preserves_model_text(monkeypatch):
    view, controls, _ = console(monkeypatch)
    view.update("Executing", "Inspecting a current observation")
    text = view.thought.content
    view.pause_button.click()
    assert controls.paused.is_set() and controls.pause_requested.is_set()
    assert view.pause_button.label == "Resume" and view.thought.content == text
    view.pause_button.click()
    assert not controls.paused.is_set() and view.pause_button.label == "Pause"
    assert controls.next() is None


def test_handover_requires_stopped_task_carry_and_explicit_operator_confirmation(monkeypatch):
    view, controls, _ = console(monkeypatch)
    state = {"carried": {"right": {"state": "verified"}}, "task_active": True,
             "retention": {"handover_token": "stopped-1"}}
    view.workflow_state(state)
    view.handover_confirm.value = True
    view.handover_button.click()
    assert controls.next() is None
    view.workflow_state({**state, "task_active": False})
    view.handover_confirm.value = False
    view.handover_button.click()
    assert controls.next() is None
    view.handover_confirm.value = True
    view.handover_confirm.update_callback(None)
    assert not view.handover_button.disabled
    view.handover_button.click()
    assert controls.next().name == "handover" and controls.next() is None
    assert not view.handover_confirm.value and view.handover_button.disabled
