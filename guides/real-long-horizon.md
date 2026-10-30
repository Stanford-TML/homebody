# REAL tasks

REAL runs the HomeBody task skills (navigate, pick, place, `remember` and `recall`) on a physical Unitree G1 with Dex3
hands, calibrated head RGB-D and SuperOdometry localization. This preliminary release ships the vanilla open-source
SuperOdometry, so results can vary significantly. Run it only supervised, with the robot supported and the remote in hand.

## How to run

After the [setup](#setup), with the [head RGB-D producers](head-rgbd.md#how-to-run) and registered SuperOdom localization
running, start the body host:

```bash
homebody real2sim explore setup --iface enp3s0   # starts the body host; use your robot Ethernet interface
```

On the G1 remote, **Y** initializes the pose; after the ramp, **A** switches to standing mode, where the walking policy
balances the robot. A is refused while the pelvis is tilted more than 25°. Then check the local files and start the console:

```bash
homebody check --backend real --real-config configs/real.local.json   # local files only; starts nothing
homebody run --backend real --real-config configs/real.local.json --memory kitchen --model astra --port 8080
```

Open http://127.0.0.1:8080, check readiness and the measured scene, enter a task and press **Start task**
(`--task "..."` submits one at launch). `homebody run` connects to the running camera, localization and body-host
services and starts none of them: it does not initialize AMO, restart SuperOdom or open the hands. Keep one control
owner. Named memory defaults to `$XDG_STATE_HOME/homebody/memory` (or `~/.local/state/homebody/memory`; `--memory-root`
selects another location), each task is recorded under `runs/session-*`, and the console's 3D view uses frozen
observations and never queries hardware.

### Stop safely

- **Pause** holds motion; Resume replans from fresh observations and cannot resume a stale action or use a model answer
  returned after pausing.
- **STOP task** cancels execution, confirms the base hold where the link is healthy and keeps the session, memory, arm
  references and grip worker, so another task can start in the same console.
- **End session** closes only when the carry ledger and measured hands show no retained load; an untracked closed hand
  or unavailable hand evidence also blocks it. Place the object, or complete a supervised handover and confirm it in
  the console: the confirmation records the removal, does not open a gripper and expires before the next task.
- Ctrl-C requests task cancellation and a safe session end; with an object held, the session stays open for placement
  or handover. Software cannot keep the grip after process death, SIGKILL, power loss or loss of the body host.
- All of these leave the body host balancing the robot. To take it down, support the robot, then press **B** (or
  **L1+R1**) on the remote: the policy stops and every joint is damped. Then
  `PYTHONPATH=src:vendor python -m homebody.backends.g1.body_host stop` ends the body host; it refuses while the policy
  is active or initializing. `PYTHONPATH=src:vendor python -m homebody.backends.g1.body_host halt` is the immediate
  halt: it stops the policy, damps every joint and ends the body host, so support the robot first.

## Setup

1. After the [exploration install](exploration.md#install): `pip install -e '.[exploration,real]'` and
   `cp configs/real.example.json configs/real.local.json`.
2. Download the SAM2 checkpoint (`source_uri` in `assets/real_perception.json`) to `.cache/models/sam2.1_hiera_tiny.pt`
   and check its `size_bytes` and `sha256`; it is not in Git, and SIM tasks do not need it.
3. Set up the producers in the [head RGB-D guide](head-rgbd.md).
4. Fill in `configs/real.local.json` from the fields and evidence below and run the check above. `homebody check` and
   `homebody --help` read local files only: no robot connection, service startup, camera stream or provider request.

### Configuration fields

The example's placeholders are invalid and cannot arm a robot. Relative paths resolve from the release root, unknown
keys are refused, and omitted fields take the defaults in `backends/real_config.py`. Settings come from
`session/config.py`; REAL does not read `configs/simulation.toml`, and `--settings` is an explicit override, not a
substitute for measured calibration.

| Field | Value |
|---|---|
| `iface` | Workstation interface on the robot network |
| `map_path`, `map_sha256`, `map_frame` | Deployment occupancy map, its exact SHA-256 and its named frame |
| `map_floor_height_m` | Measured map floor height (need not be Z=0), used by obstacle height filtering and the operator map plane |
| `alignment_path` | Current SuperOdom registration JSON |
| `camera_calibration` | Head optical-to-torso calibration, `assets/calib/g1_head.json` |
| `head_camera_serial`, `stereo_calibration_sha256`, `color_endpoint`, `depth_endpoint`, `depth_telemetry_endpoint` | Head camera identity and streams ([head RGB-D](head-rgbd.md#configuration-fields)) |
| `hand_calibration` | `null`, unless measured Dex3 effort calibration exists |
| `body_address` | Body host, default `127.0.0.1:29400` |
| `segmentation_root`, `segmentation_checkpoint`, `segmentation_checkpoint_sha256` | `vendor/samurai` (the pinned SAMURAI runtime), the SAM2 weights and their SHA-256 |

### Deployment evidence

- **Map:** a boolean occupancy NPZ (`True` means blocked) and square metric `T_map_px`, mapping pixel centers to world
  XY, with an exact SHA-256 and a named right-handed, metre, Z-up frame. A reconstruction package alone does not
  establish live registration.
- **Registration:** a current SuperOdom registration JSON with `map_sha256`, `map_frame`, `source_frame: "ws_map"`,
  `units: "m"`, measured `provenance`, `anchor_odom_stamp_s`, `anchor_T_source_lidar` and `T_map_source`, both
  right-handed rigid 4×4 matrices. The initial anchor must match current registered localization; identity is never
  inferred.
- **Head mount:** `g1_head.json` gives the measured optical-to-torso transform for the shipped URDF and
  includes the optical-axis correction; do not apply another rotation. `rot_vec_deg` and `translation_m` are the total
  correction to the raw URDF `d435_joint` pose, `T = [Rodrigues(rot_vec_deg) @ R_urdf | t_urdf + translation_m]` in torso
  axes, rotating about the camera centre. The URDF hash binds the robot model, not the physical mount; verify the mount on
  your robot, and a changed mount requires measured calibration.
- **Hands:** fresh, fault-free Dex3 feedback for both hands; no tactile sensing or torque conversion is needed. A supplied
  `hand_calibration` must contain `robot_id`, measured `provenance` and seven positive `raw_to_nm` and `rated_nm` values
  for each hand. Missing effort is recorded as unavailable, never as a measured zero or guessed force.

## Qualification before deployment

Supervised physical qualification must cover registration, depth scale, grasp retention, body-host motor timing and
stability, loaded clearance in clutter and at doorways and barriers, STOP while carrying followed by continuation, and
placement at the intended target. A rendered scene or a passing software check does not replace this qualification.
