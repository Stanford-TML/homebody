# Agent context: HomeBody SIM, REAL2SIM and REAL

A MuJoCo Unitree G1 that a vision-language model drives through skills in a real2sim
kitchen (SIM), plus G1 exploration, capture and scene reconstruction (REAL2SIM). The
README has the install and run commands. REAL runs the task skills on a physical G1, only supervised;
this preliminary release ships the vanilla open-source SuperOdometry, so REAL results can vary significantly.

## Layout

| Path | What it is |
| --- | --- |
| `src/homebody/skills/` | The skills, each one module: prompt line, arguments, `prepare` / `execute` / `abort`, declared failure codes. `contract.py` is the protocol and Runner; `registry.py` registers the task skills for SIM and REAL. `navigate_sim.py` is the SIM navigator, `navigate_real/` the checked navigation of REAL2SIM and REAL. |
| `src/homebody/motion/` | Shared robot motions: `arm.py` (servos, joint streams, the one arm reset `reset_arm`), `carry.py` (holding, tucking, recovery); `exploration/` is the G1 gait, drive math and localization checks. |
| `src/homebody/helpers/` | Pure math shared by SIM, REAL and REAL2SIM: geometry, kinematics, collision, grasp, navigation, trajectories, `raster.py` (pixels, clicks, map transforms), `floor.py` (walkable-floor networks), `region.py` (task-region polygons), `volume.py` (column evidence for REAL docking). |
| `src/homebody/primitives/` | The robot interface (`robot.py` `Robot`), observation records, action limits, and the live localization, camera and navigation primitives. |
| `src/homebody/backends/` | `mujoco.py` implements `Robot` with physics; `real.py` composes calibrated observations and G1 commands for REAL tasks; `g1/` is the G1 driver for exploration and REAL. |
| `src/homebody/vlm/` | Providers (Codex, Claude Code, two APIs), `catalog.py`, `agent_prompt.md`, what the model sees each turn (`observations.py`). |
| `src/homebody/session/` | The decision loop (`session.py`), settings (`config.py` defaults; `configs/simulation.toml` and a scene's `settings.toml` override), recording; `exploration.py` is `ExplorationSession`, and `live_context.py` the context its skills run in, reading the explorer through `LiveWorld` (`skills/live.py`). |
| `src/homebody/ui/` | The live console (viser). |
| `src/homebody/real2sim/` | `exploration/` is the live map, decisions and app behind `homebody real2sim explore` (`skills.py` registers its skills, `explore_prompt.md` is its prompt, `config.py` its settings); `runner.py` reconstructs a scene offline, reading the capture through `src/homebody/capture*.py` and exporting it with `src/homebody/scene_package.py`. |
| `hardware/g1/` | The G1 launcher, SuperOdometry and recorder image recipes, phone relay templates. |
| `configs/` | `simulation.toml` and `exploration.toml` settings, `waves/` task sets, the `capture/` schema. |
| `assets/` | Robot model and calibration, the walking-policy weights, the SRC Kitchen scene package (`assets/real2sim/src_kitchen/`), `manifest.json` (hashes; `tools/verify_assets.py` checks it). |
| `evaluation/evaluate.py` | The independent physics scorer; reads only the recorded physics after a run. |
| `tools/` | Scripted checks (`check_manipulation.py`), waves of model runs (`run_wave.py`), scoring, timelines, cockpit replays, scene building. |
| `tests/` | Fast synthetic tests; `test_custom_skill.py` is the worked example of a new skill. |
| `vendor/holosoma_inference/` | The walking-policy kernel, derived from Holosoma; its hashes are pinned in `assets/manifest.json`. |
| `.agents/skills/homebody-real2sim/` | Capture audit and reconstruction into a HomeBody/MuJoCo scene; see `guides/real2sim.md`. |

## Rules that hold the design together

* **Skills reach the robot only through a declared surface:** `Robot` for SIM skills, `SURFACE` (the
  live actions) for exploration skills. No skill imports a backend or the app: the click, raster, floor-network and region geometry they share lives in `helpers/`.
* **The model gets the prompt and the images, nothing else.** No live object poses,
  simulator state or evaluator targets reach a prompt or a skill. Scan labels, anchors and the
  scan's own portable judgement go in as initial-scene memory, for every entity alike.
* **Prompts are general.** A skill's prompt line describes the robot, not this kitchen or
  these tasks: no scene nouns, no hints for one benchmark.
* **Every number has one home, and no environment variable sets it.** SIM: `session/config.py`,
  overridden by `configs/simulation.toml`, a scene's `settings.toml` and `--settings`. REAL2SIM: the
  groups in `real2sim/exploration/config.py`, overridden by the `[group]` tables of
  `configs/exploration.toml` and `--settings`. Unknown keys are refused.
* **Where extensions go:** a SIM skill is one module plus a name in `SKILLS` (a skill
  with tunables also adds a settings group in `session/config.py` and `session/settings.py`);
  `SKILLS` also feeds REAL through `task_registry` ([REAL composition](guides/architecture.md#real-task-composition));
  an exploration motion is one module plus a name in `LIVE_SKILLS` in `real2sim/exploration/skills.py`,
  and an exploration skill the model can call takes four places: that module, its name in
  `LIVE_SKILLS`, its branch in `vlm_pick.parse_decision` and its branch in `decision_interpretation.py`;
  a provider is one class (`LABEL`, `EXECUTABLE` or `KEY`, `generate`) plus `PROVIDERS` in
  `vlm/catalog.py` and a `MODELS` row in `session/config.py`;
  a scene is one folder under `assets/real2sim/`; a backend implements `Robot`.
* **One arm reset.** Every return to rest goes through `motion/arm.reset_arm`, the same for
  both arms (y mirrored).
* **Exploration has one motion owner and one writer lock.** STOP closes the outputs while the body
  host keeps balancing, Pause resumes, Ctrl-backslash is the emergency halt that releases the legs,
  and help/check/plan start no services. Start robot services only when the operator asks.
* **Comments stay short and describe the code;** prose belongs in docstrings or the commit message.

## What a hardware backend must add around `Robot`

Capture timestamps (a frame's images, depth, joints and pose captured together, stamped at
capture; `advance` waits in real time), a deadman that stops the base when commands stop,
one commander at a time (an operator stop starts a new epoch and refuses older commands),
and calibration (arm limits, jaw samples, camera intrinsics and extrinsics). `__main__.py`
selects `MujocoBackend` or `RealG1Backend`; simulator-only state is recorded for SIM scoring.
Exploration runs its skills over an asynchronous live context ([architecture](guides/architecture.md)).

## Docs

`README.md` is written for people: install and run commands, detail in `<details>` drop-downs.
`docs/` is the project website, with the operator and architecture guides in `guides/`.
`runs/` holds recordings and is never committed.

## Checking a change

`pytest tests`, `ruff check src tests tools`, `homebody check`,
`python tools/verify_assets.py`, and the scripted four-object run
`tools/check_manipulation.py --sequence configs/continuous_four.json --no-visuals`, scored with
`tools/score_session.py`. Model behaviour is measured as rates over repeated `tools/run_wave.py`
sessions, never from one run.
