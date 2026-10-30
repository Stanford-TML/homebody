# Architecture

SIM and REAL2SIM share the skill contract, Runner, model providers and common geometry.
Their task loops have separate retry, step-limit and recording policies. REAL task
composition uses the shared Session with the capabilities described below.

| | SIM | REAL2SIM |
|---|---|---|
| Entry | `homebody` | `homebody real2sim explore` |
| Loop | `Session` in `session/session.py` | `ExplorationSession` in `session/exploration.py` |
| Robot | `MujocoBackend`, a `Robot` (`backends/mujoco.py`) | the G1: body host and driver in `backends/g1/`, offered to skills as live actions |
| Skills | `SKILLS` in `skills/registry.py` | `LIVE_SKILLS` in `real2sim/exploration/skills.py` |
| Settings | `session/config.py`, `configs/simulation.toml`, a scene's `settings.toml`, `--settings` | `real2sim/exploration/config.py`, `configs/exploration.toml`, `--settings` |
| Prompt | `vlm/agent_prompt.md` plus each skill's `PROMPT` | `real2sim/exploration/explore_prompt.md` |
| Recording | `runs/session-<time>-<id>/` | `runs/explore/` plus the sensor bag |
| Add a skill | one module in `skills/` and its name in `SKILLS`; with tunables, a settings group in `session/config.py` and `session/settings.py` (`tests/test_custom_skill.py` is a complete example) | one module in `skills/` and its name in `LIVE_SKILLS`; if the model may call it, also its branch in `vlm_pick.parse_decision` and in `decision_interpretation.py` (four places) |
| Add a setting | a field in `session/config.py`, never an environment variable | a field in `real2sim/exploration/config.py`, never an environment variable |
| Add a model, scene or backend | a `MODELS` row in `session/config.py` (a new service: a provider class in `vlm/` and a `PROVIDERS` entry); a folder under `assets/real2sim/` ([format](real2sim.md#homebody-scene-package)); a `Robot` implementation (`primitives/robot.py`) plus what CLAUDE.md lists for hardware | the same models; scenes come from [reconstruction](real2sim.md) |

A skill module in `src/homebody/skills/` declares `PROMPT`, `NEEDS`, `FAILURES`, `prepare`, `execute` and optionally
`OPTIONAL`, `SURFACE` and `abort`; `skills/contract.py` checks them (`declared`). `Runner.prepare` plans from one frame,
`Runner.execute` runs the plan against a `Context` (the skill's only robot handle), and each attempt ends in one
`Result` with a declared code. A skill never imports a backend or the app (shared click, raster, floor-network and
region geometry lives in `helpers/`): SIM skills reach `Robot`, exploration skills only their declared live actions
(`SURFACE = LiveActions`) and the explorer's `LiveWorld` in `skills/live.py`.

## REAL task composition

`homebody run --backend real --real-config ... --memory ...` selects `session/real.py`: Session and Runner on
`RealG1Backend` with checked navigation, strict one-object manipulation and the `remember` and `recall` skills, where
the SIM launcher uses `skills/navigate_sim.py` and permissive manipulation. `session/real_sim.py`, used by the
REAL-policy tests, runs the REAL task policy on MuJoCo; both compose through `session/task.py` (calibration loading,
map-bound memory, persistent Session assembly), with backend adapters for map binding, navigation evidence, target
selection and readiness. `tools/check_manipulation.py --real-policy` runs the scripted sequence with REAL navigation and
manipulation, without Session or memory.

Adding a task skill to `SKILLS` (`skills/registry.py`) makes it available in both; `task_registry` binds REAL navigation
(`skills/navigate_real/task.py`) and adds memory skills when the composition requests them, and the session's
manipulation policy selects each skill's prompt and refusal list. REAL2SIM keeps its independent `LIVE_SKILLS`.

Skills never import the concrete REAL pieces: `primitives/localization/mapped.py` (registered SuperOdom evidence on an
explicit deployment map), `primitives/cameras/rgbd.py` (calibrated RGB-D with capture-time state), `tracking.py`
(measured visual target identities) and `backends/g1/task_motion.py` (bounded commands through the body host and guarded
Dex3 transport). The Session owns provider requests, historical-image attachments, task recording and the STOP/Pause/End
lifecycle; memory storage owns map binding, persistence and removal. The REAL scene view consumes frozen frames and
cannot query hardware.

## Checked navigation and sensors

Checked navigation (`skills/navigate_real/`) walks REAL task routes and exploration routes:
`controller.py` prepares and follows them; `navigation.py` rechecks them, recovers a blocked departure, settles,
verifies arrival and checks facing, consuming primitives, never a backend (tunables: the `drive` and `gait` groups).
`skills/navigate_real/task.py`, the REAL navigate skill, binds task Robot observations and
commands to this controller; SIM tasks use `skills/navigate_sim.py`, and
neither navigator imports the other. REAL supplies registered localization and LiDAR obstacles; MuJoCo supplies
measured pose, rendered depth and rendered segmentation labels for the REAL policy.
`pick_point` resolves the requested map coordinates and checks the current permission; a rejected destination stays
rejected even when recovery restores a usable departure. Primitives own acquisition and bounded motion:

| Owner | Responsibility |
|---|---|
| `primitives/localization/superodom.py` | SuperOdometry DDS readers, health continuity, registered pose and scan admission, dead reckoning, the exploration start frame |
| `primitives/localization/lidar.py` | Self filtering, floor exclusion and near-field distances |
| `primitives/cameras/context.py`, `color_stream.py` | Phone RGB client and its newest-frame colour stream subscriber, freshness, receipt-time image buffer and the live camera's lifetime |
| `primitives/navigation/` | The live actions (`live.py` is the `LiveActions` protocol) and observations skills see, the command envelope and live errors |
| `backends/g1/` | The DDS transport, the single motor writer and the body host that keeps the robot balancing |

Received and accepted observations carry separate counters and timestamps; a queued cloud keeps its observation time,
and a scan updates the live cloud or map only when it matches a buffered odometry sample of the same start-frame
generation. The phone image is uncalibrated RGB stamped at decode time, without metric depth or an exposure pose; SIM
`Frame` and `Measurement` come from `MujocoBackend` physics. No observation read starts or restarts a service.
