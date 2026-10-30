"""Fault retention exercises the REAL launcher with no hardware or model effects."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from homebody.session.retention import RetentionGuard
from homebody.session.settings import Settings


@pytest.mark.parametrize("hand_status", [
    {"retained_or_unknown": True, "reason": "untracked_closed_hand", "hands": {"right": {"closure": .6}}},
    {"retained_or_unknown": True, "reason": "hand_state_unavailable", "hands": {}},
])
@pytest.mark.parametrize("before_session", [False, True])
def test_unknown_load_retains_backend_after_startup_or_session_fault(tmp_path, monkeypatch, hand_status, before_session):
    import homebody.__main__ as entry
    import homebody.backends.real as backend_module
    import homebody.backends.real_config as config_module
    import homebody.session.real as composition
    import homebody.session.recorder as recorder_module
    import homebody.ui.console as console_module
    import homebody.ui.real_scene as scene_module

    calls, control_box, guards = [], [], []

    def queue_exit():
        token = guards[-1].status()["handover_token"]
        for name in ("end", "handover", "end"):
            control_box[0].queue.put(console_module.Command(name, confirmation=token))

    import homebody.session.retention as retention_module
    def make_guard(*args, **kwargs):
        guard = RetentionGuard(*args, **kwargs)
        guards.append(guard)
        return guard
    monkeypatch.setattr(retention_module, "RetentionGuard", make_guard)

    class Backend:
        epoch = 1
        map_binding = SimpleNamespace(navigation=None)
        def __init__(self, *args, **kwargs):
            self.provenance = {}

        def require_manipulation(self):
            if before_session:
                queue_exit()
                raise RuntimeError("Calibration setup failed after motion owner connected")

        def retention_status(self):
            return hand_status

        def readiness(self):
            return {}

        def stop(self, epoch):
            calls.append("hold")

        def close(self):
            assert "operator_handover" in calls and "session_end" in calls
            calls.append("backend_closed")

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.path = tmp_path

        def event(self, kind, **kwargs):
            calls.append(kind)

        def close(self):
            pass

    class Console:
        server = SimpleNamespace(get_host=lambda: "fixture", get_port=lambda: 0)

        def __init__(self, controls, **kwargs):
            control_box.append(controls)

        def update(self, *args):
            pass

        def workflow_state(self, state):
            assert "retention" in state

        def close(self):
            pass

    def compose(backend, provider, settings, recorder, controls, **kwargs):
        session = SimpleNamespace(context=SimpleNamespace(carried={}, arms={}),
                                  heartbeat=lambda: None, status="Ready",
                                  observed=SimpleNamespace(snapshot=lambda: None),
                                  memory=SimpleNamespace(path=tmp_path), update=lambda *a: None)
        session.retention = make_guard(backend, recorder, controls, carried=dict, update=session.update)
        def run():
            queue_exit()
            raise RuntimeError("Sensor failed before any pick")
        session.run = run
        return session

    config = SimpleNamespace(offline_checks=lambda **kwargs: {}, map_frame="fixture", map_floor_height_m=0.)
    monkeypatch.setattr(config_module.RealConfig, "from_file", lambda *args: config)
    monkeypatch.setattr(backend_module, "RealG1Backend", Backend)
    monkeypatch.setattr(recorder_module, "Recorder", Recorder)
    monkeypatch.setattr(console_module, "Console", Console)
    monkeypatch.setattr(scene_module, "RealSceneView", lambda *args, **kwargs: None)
    monkeypatch.setattr(composition, "compose", compose)
    monkeypatch.setattr(entry, "make", lambda settings: object())
    args = SimpleNamespace(root=tmp_path, real_config=Path("fixture.json"), task=None,
                           host="fixture", port=0, memory_root=tmp_path, memory="fixture")
    assert entry.run_real(args, Settings(), tmp_path) == 1
    assert calls.index("end_refused") < calls.index("operator_handover") < calls.index("backend_closed")
