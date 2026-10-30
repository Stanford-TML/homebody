"""The `homebody` command line: run, check and provider-check; `homebody real2sim ...` runs REAL2SIM."""
import argparse
import errno
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

from homebody.session.config import DEFAULT_ENV, MODELS
from homebody.session.settings import Settings
from homebody.vlm.catalog import PROVIDERS, make, model_label, requirement
from homebody.vlm.provider import ProviderError


def parser():
    value = argparse.ArgumentParser(prog="homebody", description="HomeBody: shared G1 skills in SIM or an explicitly configured REAL session")
    value.add_argument("command", nargs="?", choices=("run", "check", "provider-check"),
                       default="run")
    value.add_argument("--root", type=Path,
                       help="Release root containing assets and configs")
    value.add_argument("--backend", choices=("sim", "real"), default="sim",
                       help="Task engine (default sim); REAL connects to already-running services")
    value.add_argument("--real-config", type=Path, help="Explicit REAL connection, map and calibration JSON")
    value.add_argument("--memory", help="Persistent named photo-memory set for REAL tasks")
    value.add_argument("--memory-root", type=Path, help="Named-memory storage (default XDG state/homebody/memory)")
    value.add_argument("--env", help="The environment by name, a folder of assets/real2sim/ "
                                     f"(default: {DEFAULT_ENV})")
    value.add_argument("--scene", type=Path, help="An environment folder by path, instead of --env")
    value.add_argument("--settings", type=Path,
                       help="TOML overrides for the selected workflow; SIM also loads simulation and scene settings")
    value.add_argument("--model", choices=tuple(MODELS), help="The model (default astra)")
    value.add_argument("--task", help="Explicit initial task; ordinary launch waits for input")
    value.add_argument("--place-object", action="append", default=[], metavar="NAME=X,Y,YAW",
                       help="Start a movable object upright at map X, Y (m) and YAW (rad) on the "
                            "surface there instead of its scanned position; repeatable")
    value.add_argument("--headless", action="store_true", help="Run one explicit task without UI")
    value.add_argument("--host", default="127.0.0.1", help="The console's address (default 127.0.0.1)")
    value.add_argument("--port", type=int, default=8080, help="The console's port (default 8080)")
    value.add_argument("--output", type=Path, help="Evidence root (default: ROOT/runs)")
    return value


def check(root: Path, scene: Path, settings) -> int:
    missing = []
    for package in ("numpy", "scipy", "mujoco", "torch", "PIL", "trimesh", "viser", "imageio_ffmpeg",
                    "holosoma_inference"):
        available = find_spec(package) is not None
        print(f"{'OK' if available else 'MISSING'} package {package}")
        if not available:
            missing.append(package)
    print(f"Scene package: {scene}")
    for directory, names in (
            (scene, ("scene.xml", "task_scene.json", "map.npz", "physics.json",
                     "semantics.json")),
            (root, ("assets/robot/g1_29dof_with_hand.urdf", "assets/robot/calibration.json",
                    "assets/robot/g1_grasp/g1_closd_grasp_deploy.xml",
                    "assets/models/amo_jit.pt", "assets/models/adapter_jit.pt",
                    "assets/models/adapter_norm_stats.pt"))):
        for name in names:
            available = (directory / name).is_file()
            print(f"{'OK' if available else 'MISSING'} {name}")
            if not available:
                missing.append(name)
    if settings.agent.provider not in PROVIDERS:
        print(f"MISSING provider {settings.agent.provider!r} for model {settings.agent.model!r} "
              f"in vlm/catalog.py PROVIDERS")
        missing.append(settings.agent.provider)
    executable, variable = requirement(settings.agent.provider) if settings.agent.provider in PROVIDERS else (None, None)
    if executable:
        found = shutil.which(executable)
        print(f"{'OK' if found else 'MISSING'} {executable} executable")
        if not found:
            missing.append(executable)
    if variable:
        found = bool(os.environ.get(variable))
        print(f"{'OK' if found else 'MISSING'} {variable} in the environment")
        if not found:
            missing.append(variable)
    print(f"Model: {settings.agent.model}: {model_label(settings.agent)} via {settings.agent.provider}; "
          f"login not tested (homebody provider-check --model {settings.agent.model} sends one request)")
    print("No simulator, web server, provider request, or robot service was started.")
    return int(bool(missing))


def provider_check(settings, output):
    path = output / ("provider-check-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                     + "-" + uuid.uuid4().hex[:6])
    provider = make(settings.agent)
    try:
        decision = provider.generate(
            'Return {"text":"connection checked","skill":"done","arguments_json":"{}"}. '
            'Do not use tools.', images=(), names=("done",), attempt=path,
            tick=lambda: time.sleep(0.02), cancelled=lambda: False)
        print(f"{settings.agent.provider}: {decision.text}")
        return 0
    except ProviderError as exc:
        print(f"{settings.agent.provider} check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"Provider evidence: {path}")


def run(args, settings, output):
    from homebody.backends.mujoco import MujocoBackend
    from homebody.helpers.kinematics import Arm
    from homebody.session.recorder import Recorder
    from homebody.session.session import Session
    from homebody.ui.console import Console, Controls
    from homebody.vlm.observations import semantic_prior

    recorder = Recorder(args.root, settings, output)
    recorder.event("scene_package", path=str(args.scene), object_spawns=args.spawns)
    backend = ui = None
    assembled = False
    controls = Controls()
    try:
        if not args.headless:
            task_scene = json.loads((args.scene / "task_scene.json").read_text())
            ui = Console(controls, host=args.host, port=args.port,
                         initial_task=args.task or task_scene.get("default_task", ""),
                         scene_name=task_scene.get("name", "Simulation"),
                         model_name=model_label(settings.agent),
                         topdown_settings=settings.topdown, render_in_background=True)
        backend = MujocoBackend(args.root, settings, scene=args.scene, spawns=args.spawns)
        scene_update = None
        if ui is not None:
            from homebody.ui.scene import SceneView
            ui.scene_view = SceneView(ui.server, backend)
            scene_update = ui.scene_view.update
        urdf = args.root / "assets/robot/g1_29dof_with_hand.urdf"
        arms = {side: Arm(urdf, side) for side in ("left", "right")}
        if ui is not None:
            ui.arms = arms
        calibration = json.loads((args.root / "assets/robot/calibration.json").read_text())
        provider = make(settings.agent)
        session = Session(backend, provider, settings, recorder, controls,
                          arms=arms, jaw_samples=calibration["jaw_samples"], ui=ui,
                          observer_frame=backend.observer_frame if args.headless else None,
                          shoulder_frame=backend.shoulder_frame if args.headless else None,
                          scene_update=scene_update, real_time=ui is not None,
                          semantic_context=semantic_prior(args.scene))
        assembled = True
        print(f"Evidence: {recorder.path}")
        if args.headless:
            outcome = session.run_task(args.task)
            print(json.dumps(outcome))
            return 0 if outcome["status"] == "model_done" else 1
        ui.update("Ready")
        if args.task:
            controls.submit(args.task)
        print(f"Console: http://{ui.server.get_host()}:{ui.server.get_port()}")
        session.run()
        return 0
    except KeyboardInterrupt:
        controls.cancel.set()
        recorder.event("interrupted", reason="KeyboardInterrupt")
        if args.task and not assembled:
            recorder.setup_failure(args.task, "Interrupted while initializing simulation")
        return 130
    except Exception as exc:  # noqa: BLE001 -- record every failure in the run's evidence
        recorder.event("session_error", error=f"{type(exc).__name__}: {exc}")
        if args.task and not assembled:
            recorder.setup_failure(args.task, f"{type(exc).__name__}: {exc}")
        hint = " (the port is in use; pass --port)" if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE else ""
        print(f"Session failed: {exc}{hint}\nEvidence: {recorder.path}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()
        if ui is not None:
            ui.close()
        recorder.close()


def check_real(args):
    """Validate local deployment files without opening sensor or command connections."""
    from homebody.backends.real_config import RealConfig
    try:
        config = RealConfig.from_file(args.real_config, args.root)
        checks = config.offline_checks(root=args.root)
        for name, result in checks.items():
            state = {True: "OK", False: "MISSING", None: "UNTESTED"}[result["ok"]]
            print(f"{state} {name}: {result['detail']}")
        print("Local files only: no robot connection, service startup, camera stream or provider request.")
        print("Physical calibration, hand retention and loaded motion remain supervised qualification gates.")
        return int(any(result["ok"] is False for result in checks.values()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"REAL configuration invalid: {error}", file=sys.stderr)
        return 1


def _fault_hold(guard, controls, error):
    """Keep the healthy host client alive while an operator resolves a retained load."""
    try:
        guard.recorder.event("retention_fault", retention=guard.status())
    except (Exception, KeyboardInterrupt) as record_error:  # noqa: BLE001 -- evidence failure cannot release ownership
        print(f"Fault evidence unavailable: {record_error}. Live client retained.", file=sys.stderr)
    try:
        guard.backend.stop(guard.backend.epoch)
    except (Exception, KeyboardInterrupt) as hold_error:  # noqa: BLE001 -- retain unconfirmed grip ownership
        print(f"Hold is unconfirmed: {hold_error}. Live client retained; operator intervention required.", file=sys.stderr)
    try:
        guard.update("Stopped", f"{error} Object state retained. Complete a supervised handover before End session.")
    except (Exception, KeyboardInterrupt) as display_error:  # noqa: BLE001 -- display failure cannot release
        print(f"Fault display unavailable: {display_error}", file=sys.stderr)
    while not controls.closing.is_set():
        try:
            command = controls.next()
            if command is None:
                time.sleep(.02)
            elif command.name == "handover":
                guard.handover(command.confirmation)
            elif command.name == "end":
                guard.end()
            elif command.name == "task":
                guard.update("Stopped", "This fault requires a new validated session after handover.")
        except KeyboardInterrupt:
            controls.end()
        except Exception as command_error:  # noqa: BLE001 -- retain ownership after a failed hold or evidence write
            print(f"Fault recovery command failed: {command_error}. Session remains open.", file=sys.stderr)


def run_real(args, settings, output):
    """Connect explicit primitives; Session remains the only task/provider owner."""
    from homebody.backends.real import RealG1Backend
    from homebody.backends.real_config import RealConfig
    from homebody.session.real import compose
    from homebody.session.recorder import Recorder
    from homebody.session.retention import RetentionGuard
    from homebody.ui.console import Console, Controls

    backend = ui = session = guard = None
    recorder = Recorder(args.root, settings, output, backend="real")
    controls = Controls()
    failed = False
    previous_interrupt = None
    if threading.current_thread() is threading.main_thread():
        previous_interrupt = signal.signal(signal.SIGINT, lambda signum, frame: controls.end())
    try:
        config = RealConfig.from_file(args.real_config, args.root)
        missing = [name for name, check in config.offline_checks(root=args.root).items() if check["ok"] is False]
        if missing:
            raise ValueError("Missing REAL prerequisites: " + ", ".join(missing))
        ui = Console(controls, host=args.host, port=args.port, initial_task=args.task or "",
                     scene_name=config.map_frame, model_name=model_label(settings.agent),
                     topdown_settings=settings.topdown, render_in_background=True, workflow="real")
        backend = RealG1Backend(args.root, settings, config=config)
        def startup_update(status, message=""):
            ui.update(status, message)
            ui.workflow_state({"carried": {}, "task_active": False, "paused": False,
                               "recording": str(recorder.path), "memory": None,
                               "readiness": backend.readiness(), "retention": guard.status()})
        guard = RetentionGuard(backend, recorder, controls, carried=dict, update=startup_update)
        backend.require_manipulation()
        from homebody.ui.real_scene import RealSceneView
        ui.scene_view = RealSceneView(ui.server, args.root,
                                      navigation=backend.map_binding.navigation,
                                      floor_height_m=config.map_floor_height_m)
        recorder.event("deployment", **backend.provenance)
        memory_root = args.memory_root or Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "homebody/memory"
        session = compose(backend, make(settings.agent), settings, recorder, controls,
                          root=args.root, memory_root=memory_root, memory_name=args.memory, ui=ui)
        guard = session.retention
        ui.arms = session.context.arms
        session.heartbeat()
        if session.status != "Waiting for sensors":
            session.update("Ready", "Verified sensor capabilities. Start task authorizes its registered actions.")
        print(f"Evidence: {recorder.path}\nMemory: {session.memory.path}")
        print(f"Console: http://{ui.server.get_host()}:{ui.server.get_port()}")
        if args.task:
            controls.submit(args.task)
        while not controls.closing.is_set():
            try:
                session.run()
            except KeyboardInterrupt:
                controls.stop()
                recorder.event("interrupted", reason="KeyboardInterrupt", carried=session.context.carried.copy())
                if not session.end_session():
                    print("Object retained: session remains open for placement or confirmed handover.", file=sys.stderr)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- retain terminal evidence and uncertain loads
        failed = True
        try:
            recorder.event("session_error", error=f"{type(error).__name__}: {error}")
        except Exception as record_error:  # noqa: BLE001 -- disk failure must not bypass retained grip
            print(f"Could not record session failure: {record_error}", file=sys.stderr)
        print(f"REAL session failed: {error}\nEvidence: {recorder.path}", file=sys.stderr)
        if guard is not None and guard.status()["retained_or_unknown"]:
            _fault_hold(guard, controls, error)
        elif args.task and session is None:
            recorder.setup_failure(args.task, f"{type(error).__name__}: {error}")
    finally:
        # Includes failures before Session composition and interruptions during fault entry.
        # A closing UI flag alone cannot authorize dropping an unknown load's command owner.
        while guard is not None and guard.status()["retained_or_unknown"]:
            controls.closing.clear()
            try:
                _fault_hold(guard, controls, "Shutdown is waiting for physical load resolution.")
            except KeyboardInterrupt:
                controls.end()
        failures = []
        for name, resource in (("backend", backend), ("console", ui), ("recorder", recorder)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:  # noqa: BLE001 -- every owned resource must be closed
                    failures.append(f"{name}: {error}")
        if failures:
            failed = True
            print("REAL resources failed to close: " + "; ".join(failures), file=sys.stderr)
        if previous_interrupt is not None:
            signal.signal(signal.SIGINT, previous_interrupt)
    return 1 if failed else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["real2sim"]:
        from homebody.real2sim.cli import main as real2sim
        return real2sim(argv[1:])
    value = parser()
    args = value.parse_args(argv)
    if args.root is None:
        candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
        args.root = next((path for path in candidates
                          if (path / "assets").is_dir() and
                          (path / "configs/simulation.toml").is_file()), None)
        if args.root is None:
            value.error("Run from the release folder or pass --root PATH containing assets/configs")
    args.root = args.root.resolve()
    if args.backend == "real":
        if args.real_config is None:
            value.error("--backend real requires --real-config PATH")
        if args.place_object:
            value.error("--place-object is only available in SIM")
        if args.command == "run" and args.headless:
            value.error("REAL tasks require the persistent operator console; --headless is for SIM")
        if args.command == "run" and not args.memory:
            value.error("REAL tasks require an explicit persistent --memory NAME")
    elif args.real_config or args.memory or args.memory_root:
        value.error("REAL configuration and persistent memory flags require --backend real")
    environments = sorted(path.name for path in (args.root / "assets/real2sim").iterdir() if path.is_dir())
    if args.env and args.env not in environments:
        value.error(f"--env must be one of {', '.join(environments)}")
    if args.env and args.scene:
        value.error("Pass --env or --scene, not both")
    args.scene = (args.scene or args.root / "assets/real2sim" / (args.env or DEFAULT_ENV)).resolve()
    if args.command == "run" and args.headless and not (args.task and args.task.strip()):
        value.error("--headless requires an explicit --task")
    args.spawns = {}
    for item in args.place_object:
        name, _, numbers = item.partition("=")
        try:
            args.spawns[name] = [float(v) for v in numbers.split(",")]
        except ValueError:
            args.spawns[name] = []
        if not name or len(args.spawns[name]) != 3 or not all(map(math.isfinite, args.spawns[name])):
            value.error(f"--place-object takes NAME=X,Y,YAW in map metres and radians, not {item!r}")
    if not 1 <= args.port <= 65535:
        value.error("--port must be between 1 and 65535")
    if args.spawns and (args.scene / "task_scene.json").is_file():
        movable = json.loads((args.scene / "task_scene.json").read_text()).get("movable_entities", ())
        unknown = sorted(set(args.spawns) - set(movable))
        if unknown:
            value.error(f"--place-object: no movable object {', '.join(unknown)}; "
                        f"this environment's are {', '.join(movable)}")
    try:
        settings = Settings.load(args.root / "configs/simulation.toml" if args.backend == "sim" else None,
                                 scene=args.scene if args.backend == "sim" else None, override=args.settings)
    except (ValueError, TypeError, KeyError, OSError) as error:
        value.error(f"settings: {error}")
    if args.model:
        settings = replace(settings, agent=replace(settings.agent, model=args.model))
    output = (args.output or args.root / "runs").resolve()
    if args.command == "check":
        if args.backend == "real":
            return check_real(args)
        return check(args.root, args.scene, settings)
    if args.command == "provider-check":
        return provider_check(settings, output)
    return run_real(args, settings, output) if args.backend == "real" else run(args, settings, output)


if __name__ == "__main__":
    raise SystemExit(main())
