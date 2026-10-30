# Head RGB-D producers

REAL manipulation needs capture-associated, calibrated RGB and metric depth from one calibrated device, the G1's head
D435i, with actual intrinsics and the camera-to-torso transform; raw depth or an iPhone RGB stream cannot satisfy this.
Four producer stages publish them in the foreground, separately from `homebody run`, and do not start AMO, move the
robot, open its hands or take over a camera another service holds.

## How to run

Paths are relative to each host's checkout; `--python` selects another interpreter. On the **camera host**, in two
terminals, then on the **workstation**, in two terminals, with the camera host's address:

```bash
python3 src/homebody/primitives/cameras/deployment.py run capture --calibration configs/local/head-stereo.json
python3 src/homebody/primitives/cameras/deployment.py run color --calibration configs/local/head-stereo.json
python src/homebody/primitives/cameras/deployment.py run server --calibration configs/local/head-stereo.json \
  --engine /path/to/verified.engine --engine-sha256 VERIFIED_ENGINE_SHA256 --stereo-endpoint tcp://CAMERA_HOST:55562
python src/homebody/primitives/cameras/deployment.py run relay --calibration configs/local/head-stereo.json \
  --engine-sha256 VERIFIED_ENGINE_SHA256
```

| Stage | Host | Publishes |
|---|---|---|
| `capture` | camera host | IR pairs on 55562 |
| `color` | camera host | HSCOLOR1 RGB on 55566 |
| `server` | workstation | HSDPTH01 depth on localhost:55565 and FFSTEL01 telemetry on localhost:55564 |
| `relay` | workstation | HMDPTH01 metric depth on localhost:55563 |

`plan` in place of `run` prints the exact command, and `check` checks source and asset identity only; neither contacts
a camera, starts a process, imports CUDA or checks modules, device availability or engine compatibility. Stop each
producer with Ctrl-C; task STOP leaves perception running. A file lease keeps a second copy of the same producer for
that camera from starting. Then continue with the [REAL guide](real-long-horizon.md#how-to-run).

## Setup

1. **Hosts.** Clone the same release checkout onto both hosts. The camera host needs only the standalone scripts and the
   vendor closure, not the Python 3.11 HomeBody application or a locomotion runtime, plus a D435i, supported
   librealsense/`pyrealsense2`, Python 3.8 or newer, NumPy, OpenCV and pyzmq compatible with its JetPack/OS and
   RealSense SDK; configure USB permissions and device ownership separately. The workstation (inference host) needs
   CUDA-enabled PyTorch, TensorRT and NumPy/OpenCV/pyzmq in one compatible environment; the runner uses TensorRT's named
   tensor API, and the engine must match that host's GPU, TensorRT and CUDA runtime.
2. **Stereo calibration.** Supply `configs/local/head-stereo.json`, measured from the actual device and identical on both
   hosts: serial, firmware, both rectified IR intrinsics, baseline/extrinsics and the 848×480@60 Y8 profile, in the
   `CalibrationRecord` format of the pinned `fast_stereo/calibration.py`. Capture checks the live device against every field.
3. **Engine.** Supply a compatible TensorRT engine and its SHA-256, or build one on the workstation from an authorized
   Fast-FoundationStereo ONNX export (with an adjacent `model.onnx.data`, also pass
   `--onnx-data-sha256 VERIFIED_TENSOR_DATA_SHA256`):

   ```bash
   CAMERA_TOOL=src/homebody/primitives/cameras/deployment.py
   python "$CAMERA_TOOL" check build --onnx /path/to/model.onnx --onnx-sha256 VERIFIED_ONNX_SHA256 --output /path/to/new.engine
   python "$CAMERA_TOOL" run build --onnx /path/to/model.onnx --onnx-sha256 VERIFIED_ONNX_SHA256 --output /path/to/new.engine
   sha256sum /path/to/new.engine
   ```

   The builder does not overwrite an output file. Its default FP16 flag requires a compatible TensorRT version; on
   versions without it, `--fp32` skips the flag and does not rewrite the ONNX precision. Setup neither includes nor
   downloads weights or engines. The reference export uses four iterations at 288×384, takes normalized left/right float
   images and returns disparity; export, checkpoint preparation and their licenses
   ([NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)) are external requirements.
4. **Head mount.** Device stereo calibration does not establish the mount transform; REAL also requires the separately
   measured calibration selected by `camera_calibration` ([REAL setup](real-long-horizon.md#deployment-evidence)).
5. **Clocks.** Synchronize the camera host and workstation wall clocks: RGB-D pairing and capture freshness use capture
   wall time, not socket receipt time. Run the relay on the workstation that runs `homebody run`, so its completion
   clock shares that clock domain.

## Configuration fields

| `configs/real.local.json` field | Value |
|---|---|
| `head_camera_serial` | The camera's actual 12-digit serial |
| `stereo_calibration_sha256` | The canonical digest `plan relay` prints as `--expected-calibration-sha256`, not the raw JSON file's byte hash |
| `color_endpoint`, `depth_endpoint`, `depth_telemetry_endpoint` | `tcp://CAMERA_HOST:55566`, `tcp://127.0.0.1:55563`, `tcp://127.0.0.1:55564` |

Localhost endpoints need local publishers or explicit forwarding; for a robot-hosted RGB publisher, use its robot
address. REAL joins depth with the FFSTEL01 telemetry by exact frame ID and capture time and verifies the configured
camera serial and stereo calibration identity: RGB must name the same camera, and telemetry must run alongside depth,
since a depth image alone is insufficient.

## Limits and safety

- Do not copy another camera's calibration, edit its serial to pass the check, or substitute an unverified export on the
  strength of its name. Keep these unauthenticated camera streams on a trusted network.
- Do not use 55564 for RGB: it is telemetry. Explicit endpoint overrides support other topologies.
- A busy RGB camera produces a refusal; release it through its owner's documented operator procedure.
- The RGB producer uses RealSense global timestamps and refuses device-clock-only, stale or future frames; the stereo
  capture also requires global timestamps.
- The pinned Fast-FoundationStereo runtime, `vendor/mjlab_depth`, is listed with its license in
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); HomeBody's deployment and RGB publication are in `primitives/cameras`.

## Qualification before deployment

Hash matching checks identity, not engine correctness: validate the resulting depth against measured surfaces before manipulation. Verify the clock domains on the actual device, since
offline checks cannot establish their accuracy. On a bandwidth-constrained link, `capture` accepts `--jpeg-quality` and
`--output-rate-hz`; qualify the resulting latency and image quality.
