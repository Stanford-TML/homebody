# Agent context: HomeBody SIM

A MuJoCo Unitree G1 that a vision-language model drives through skills in a real2sim
kitchen. The README's "SIM" section has install and run commands. The real-robot backend and
the scanning pipeline are the only parts not implemented yet; they come with the REAL2SIM
and REAL releases.

## Layout

| Path | What it is |
| --- | --- |
| `src/homebody/skills/` | The skills (`navigate.py`, `pick.py`, `place.py`), each one module: prompt line, arguments, `prepare` / `execute` / `abort`, declared failure codes. `registry.py` lists them. |
| `src/homebody/motion/` | Shared robot motions: `arm.py` (servos, joint streams, the one arm reset `reset_arm`), `carry.py` (holding, tucking, recovery). |
| `src/homebody/helpers/` | Pure math: geometry, kinematics, collision, grasp, navigation, trajectories. |
| `src/homebody/primitives/` | The robot interface (`robot.py` `Robot`), observation records, action limits, `NotReleased`. |
| `src/homebody/backends/` | `mujoco.py` implements `Robot` with physics; `real.py` is the real-G1 backend, not yet implemented, coming with the REAL release. |
| `src/homebody/vlm/` | Providers (Codex, Claude Code, two APIs), `catalog.py`, `agent_prompt.md`, what the model sees each turn (`observations.py`). |
| `src/homebody/session/` | The decision loop, settings (`config.py` defaults; `configs/simulation.toml` and a scene's `settings.toml` override), recording. |
| `src/homebody/ui/` | The live console (viser). |
| `src/homebody/real2sim/` | The scanning pipeline's stages (`explore`, `capture`, `reconstruct`, `package`, `validate`), not yet implemented, coming with the REAL2SIM release. |
| `assets/` | Robot model and calibration, the walking-policy weights, the SRC Kitchen scene package (`assets/real2sim/src_kitchen/`), `manifest.json` (hashes; `tools/verify_assets.py` checks it). |
| `evaluation/evaluate.py` | The independent physics scorer; reads only the recorded physics after a run. |
| `tools/` | Scripted checks (`check_manipulation.py`), waves of model runs (`run_wave.py`), scoring, timelines, cockpit replays, scene building. |
| `vendor/holosoma_inference/` | The walking-policy kernel, derived from Holosoma; its hashes are pinned in `assets/manifest.json`. |

## Rules that hold the design together

* **Skills reach the robot only through `Robot`.** No skill imports the simulator; a real
  backend must work without changing a skill.
* **The model gets the prompt and the images, nothing else.** No live object poses,
  simulator state or evaluator targets reach a prompt or a skill. Scan labels, anchors and the
  scan's own portable judgement go in as initial-scene memory, for every entity alike.
* **Prompts are general.** A skill's prompt line describes the robot, not this kitchen or
  these tasks: no scene nouns, no hints for one benchmark.
* **Extension is one place each:** a skill is one module plus a name in `SKILLS` (a skill
  with tunables also adds a settings group in `session/config.py` and `session/settings.py`);
  a provider is one class (`LABEL`, `EXECUTABLE` or `KEY`, `generate`) plus `PROVIDERS` in
  `vlm/catalog.py` and a `MODELS` row in `session/config.py`;
  a scene is one folder under `assets/real2sim/`; a backend implements `Robot`.
* **One arm reset.** Every return to rest goes through `motion/arm.reset_arm`, the same for
  both arms (y mirrored).
* **Comments stay short;** prose belongs in docstrings or the commit message.

## What a hardware backend must add around `Robot`

Capture timestamps (a frame's images, depth, joints and pose captured together, stamped at
capture; `advance` waits in real time), a deadman that stops the base when commands stop,
one commander at a time (an operator stop starts a new epoch and refuses older commands),
and calibration (arm limits, jaw samples, camera intrinsics and extrinsics). `__main__.py`
constructs `MujocoBackend` directly, and the session records simulator-only state for the
scorer, so a real backend needs those made optional first.

## Docs

`README.md` is the project page's README plus the SIM section, written for people: keep it
short, with detail in `<details>` drop-downs. `docs/` is the project website. `runs/` holds recordings and is never committed.

## Checking a change

`pytest tests`, `ruff check src tests tools`, `.venv/bin/homebody check`,
`.venv/bin/python tools/verify_assets.py`, and the scripted
four-object run `tools/check_manipulation.py --sequence configs/continuous_four.json
--no-visuals`, scored with `tools/score_session.py`. Model behaviour is measured as rates
over repeated `tools/run_wave.py` sessions, never from one run.
