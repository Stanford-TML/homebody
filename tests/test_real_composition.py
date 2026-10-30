"""The public REAL command with effects replaced below the primitive boundary."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_real_backend import backend as make_robot
from test_session import ImageOnlyRecorder, Provider

from homebody import __main__ as cli
from homebody.backends.real_config import RealConfig
from homebody.vlm.provider import Decision

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("native_contact", [False, True])
@pytest.mark.parametrize("startup_gap", [0, 1])
def test_public_real_command_runs_registered_memory_and_closes_without_actuation(tmp_path, monkeypatch, native_contact, startup_gap):
    import homebody.session.recorder as recording
    import homebody.ui.console as ui
    import homebody.ui.real_scene as scene
    from homebody.backends import real

    robot, motion, clock = make_robot(tmp_path)
    if native_contact:
        from homebody.backends.g1.task_motion import RealHandState
        hands, _ = motion.hand_states()
        physical = {side: RealHandState(hand.joints, hand.closure, hand.finger_effort_fraction,
                                        hand.finger_closure_gap, False, "empty")
                    for side, hand in hands.items()}
        motion.effort_calibration = None
        motion.hand_states = lambda timestamp=None, **kw: (physical, {side: clock() for side in physical})
    capture = robot.ports.camera.snapshot()
    robot.ports.camera.snapshot = lambda: replace(capture, sequence=int(clock()*1000),
        captured_at=clock(), color_time=clock(), depth_time=clock(), assembled_at=clock())
    robot.ports.tracker = SimpleNamespace(ready=True,
        labels=lambda rgb, stamp: __import__('numpy').zeros(rgb.shape[:2], dtype='int32'),
        retain=lambda labels: None, close=lambda: None)
    provider = Provider([Decision("Remember", "remember", {"content": "Current room"}),
                         Decision("Recall", "recall", {}), Decision("Done", "done", {})], ticks=0)
    snapshot = robot.snapshot
    snapshots = 0
    def observe():
        nonlocal snapshots
        snapshots += 1
        if snapshots == startup_gap:
            from homebody.primitives.observations import ObservationError
            assert not provider.names
            raise ObservationError("SuperOdom evidence unavailable: scan_delayed")
        return snapshot()
    robot.snapshot = observe
    statuses = []
    monkeypatch.setattr(real, "RealG1Backend", lambda *a, **k: robot)
    monkeypatch.setattr(RealConfig, "from_file", lambda *a, **k: robot.config)
    monkeypatch.setattr(RealConfig, "offline_checks", lambda self, **kwargs: {"fixture": {"ok": True}})
    monkeypatch.setattr(scene, "RealSceneView", lambda *args, **kwargs: None)
    monkeypatch.setattr(recording, "Recorder", ImageOnlyRecorder)
    monkeypatch.setattr(cli, "make", lambda agent: provider)

    class View:
        def __init__(self, controls, **kwargs):
            self.controls = controls
            self.server = SimpleNamespace(get_host=lambda: "fixture", get_port=lambda: 1)
        def show(self, frame): pass
        def workflow_state(self, state): pass
        def recalled(self, hits): pass
        def select(self, decision, frame): pass
        def skill_event(self, event): pass
        def update(self, status, text=""):
            statuses.append(status)
            if status == "model_done": self.controls.end()
        def close(self): pass

    monkeypatch.setattr(ui, "Console", View)
    code = cli.main(["run", "--backend", "real", "--root", str(ROOT),
                     "--real-config", str(tmp_path/'explicit.json'), "--memory", "room",
                     "--memory-root", str(tmp_path/'memories'), "--task", "Remember and recall",
                     "--output", str(tmp_path/'runs')])
    assert code == 0
    assert ("Waiting for sensors" in statuses) == bool(startup_gap)
    assert "Stopped" not in statuses[:statuses.index("Thinking")]
    assert all(names == ("navigate", "pick", "place", "remember", "recall", "done") for names in provider.names)
    assert len(provider.images[-1]) == 5
    assert (tmp_path/'memories/room/memory.json').is_file()
    assert all(row[0] in ("velocity", "hold", "close") for row in motion.events)
    assert all(row[1] == (0., 0., 0.) for row in motion.events if row[0] == "velocity")
    assert motion.events[-1] == ("close",)


@pytest.mark.parametrize("extra", [[], ["--real-config", "config.json"],
                                   ["--real-config", "config.json", "--memory", "room", "--headless", "--task", "pick"]])
def test_incomplete_real_launch_refuses_before_factory_or_provider(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, "run_real", lambda *a, **k: pytest.fail("Incomplete configuration reached run"))
    with pytest.raises(SystemExit) as result:
        cli.main(["run", "--backend", "real", "--root", str(ROOT), *extra])
    assert result.value.code == 2


def test_real_check_does_not_construct_runtime_or_provider(tmp_path, monkeypatch):
    from homebody.backends import real
    monkeypatch.setattr(real, "RealG1Backend", lambda *a, **k: pytest.fail("check constructed robot"))
    monkeypatch.setattr(cli, "make", lambda *a: pytest.fail("check initialized provider"))
    assert cli.main(["check", "--root", str(ROOT), "--backend", "real",
                     "--real-config", str(tmp_path/'missing.json')]) == 1
