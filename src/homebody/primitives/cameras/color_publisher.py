"""The depth node's colour stream: the D435i colour imager on its own pipeline, one JPEG per
frame on a PUB socket, the camera's geometry in every header. Runs on the G1 (Python 3.8).

    HSCOLOR1 message, ONE zmq part so a CONFLATE subscriber always holds the newest frame:
        b"HSCOLOR1" | uint32 LE header length | header JSON | JPEG bytes

The header carries K_ir, the stereo baseline, the colour intrinsics and the IR1->colour
extrinsic, all read from the device (one calibration graph), so the workstation needs no
other calibration source. The stereo module belongs to the capture process; this one opens
only the colour node. An occupied camera is refused without stopping its owner.

The camera is chosen by serial (--serial or --exclude-serial), since an
unselected pipeline takes whichever D435i enumerates first. With two cameras and no selection
it refuses.

--depth-endpoint also streams the camera's own depth when no capture process owns its stereo
module: uint16 millimetres in the left-IR frame, K_ir in the colour header, on a
second PUB as HMDPTH01 packets, the head's Fast-FoundationStereo wire format.
"""
# ruff: noqa: UP031 -- this producer runs on Python 3.8.
import argparse
import json
import math
import struct
import sys
import time

import cv2
import numpy as np
import zmq

MAGIC = b"HSCOLOR1"
DEPTH_MAGIC = b"HMDPTH01"
DEPTH_HEADER = struct.Struct("<8sBBHQQQQIII")  # HMDPTH01 metric-depth wire contract
DEFAULT_ENDPOINT = "tcp://0.0.0.0:55566"
FPS_LADDER = (60, 30, 15, 6)   # tried in order; 60 caps exposure at 16.7 ms

class CameraOwnershipError(RuntimeError):
    """The selected RGB device is held by another process; no owner is terminated."""


def pack(header, jpeg):
    """Serialize one HSCOLOR1 message; the inverse is color_stream.unpack."""
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(body)) + body + bytes(jpeg)


def _pick_serial(rs, serial, exclude):
    """The one D435i this publisher streams: --serial, else the only one not --exclude-serial."""
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if serial:
        if serial not in serials:
            raise RuntimeError("D435i %s is not connected (found %s)" % (serial, serials or "none"))
        return serial
    others = [s for s in serials if s != exclude]
    if len(others) != 1:
        raise RuntimeError("expected exactly one D435i besides %s, found %s: pass --serial" % (
            exclude or "(none excluded)", others or "none"))
    return others[0]


def _start_color_pipeline(rs, width, height, ladder, serial, depth=False):
    last = None
    for fps in ladder:
        pipe = rs.pipeline()
        started = False
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
        if depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(fps))
        try:
            prof = pipe.start(cfg)
            started = True
            # RGB takes the global-time clock the depth producer requires; capture_time
            # refuses an unconverted device clock.
            for sensor in prof.get_device().query_sensors():
                if (sensor.get_info(rs.camera_info.name) == "RGB Camera"
                        and sensor.supports(rs.option.global_time_enabled)):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
            return pipe, prof, int(fps)
        except Exception as exc:
            if started:
                pipe.stop()
                raise RuntimeError("RGB global-time setup failed; camera released") from exc
            last = exc
            if "resource busy" in str(exc).lower() or "errno=16" in str(exc).lower():
                raise CameraOwnershipError(
                    "RGB camera is busy; stop its current owner explicitly before retrying") from exc
    raise RuntimeError("no colour profile accepted (%s)" % str(last)[:80])


def capture_time(frame, rs, received_ns):
    """Use a genuine SDK exposure timestamp, never retrieval time as capture."""
    domain = frame.get_frame_timestamp_domain()
    if domain == rs.timestamp_domain.global_time:
        basis = "realsense_global_time"
    elif domain == rs.timestamp_domain.system_time:
        basis = "realsense_system_time"
    else:
        raise RuntimeError("RGB exposure has no wall-clock timestamp; global time is required")
    milliseconds = float(frame.get_timestamp())
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise RuntimeError("RGB exposure timestamp is invalid")
    captured_ns = round(milliseconds * 1_000_000)
    if not 0 <= received_ns - captured_ns <= 500_000_000:
        raise RuntimeError("RGB exposure is stale or future dated; refusing a fresh receipt label")
    return captured_ns, basis


def _geometry(rs, prof, ir_width):
    """K_ir, baseline, colour intrinsics and the IR1->colour extrinsic from the device's own
    stream profiles -- never across pipelines."""
    profiles = [sp for sensor in prof.get_device().query_sensors()
                for sp in sensor.get_stream_profiles()]

    def infrared(index):
        return next(sp for sp in profiles if sp.stream_type() == rs.stream.infrared
                    and sp.stream_index() == index
                    and sp.as_video_stream_profile().width() == ir_width)

    s1, s2 = infrared(1), infrared(2)
    ir = s1.as_video_stream_profile().get_intrinsics()
    baseline = float(np.linalg.norm(np.asarray(s1.get_extrinsics_to(s2).translation, dtype=float)))
    cs = prof.get_stream(rs.stream.color).as_video_stream_profile()
    ci = cs.get_intrinsics()
    col = next(sp for sp in profiles if sp.stream_type() == rs.stream.color)
    ex = s1.get_extrinsics_to(col)
    R = np.asarray(ex.rotation, dtype=float).reshape(3, 3).T   # rs is column-major
    return {
        "K_ir": [[ir.fx, 0.0, ir.ppx], [0.0, ir.fy, ir.ppy], [0.0, 0.0, 1.0]],
        "baseline_m": baseline,
        "color_intr": [ci.fx, ci.fy, ci.ppx, ci.ppy, float(ci.width), float(ci.height)],
        "ir2c_R": R.tolist(),
        "ir2c_t": [float(v) for v in ex.translation],
    }


def serve(endpoint, width, height, ir_width, jpeg_quality, duration_s, serial="",
          exclude_serial="", max_fps=60, depth_endpoint="", depth_hz=15.0, report_s=5.0):
    import pyrealsense2 as rs

    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    dsock = None
    if depth_endpoint:
        dsock = context.socket(zmq.PUB)
        dsock.setsockopt(zmq.SNDHWM, 1)
        dsock.setsockopt(zmq.LINGER, 0)
        dsock.bind(depth_endpoint)
    depth_sequence = 0
    deadline = time.monotonic() + duration_s
    sequence = 0
    ladder = tuple(f for f in FPS_LADDER if f <= max_fps) or FPS_LADDER[-1:]
    while time.monotonic() < deadline:
        pipe = None
        try:
            chosen = _pick_serial(rs, serial, exclude_serial)
        except Exception as exc:  # noqa: BLE001 - isolate a failed camera frame or device
            print("colour camera: %s -- retry in 2s" % str(exc)[:120], flush=True)
            time.sleep(2.0)
            continue
        try:
            pipe, prof, fps = _start_color_pipeline(rs, width, height, ladder, chosen,
                                                    depth=dsock is not None)
        except CameraOwnershipError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate a failed camera frame or device
            print("colour start failed: %s -- retry in 2s" % str(exc)[:90], flush=True)
        if pipe is None:
            time.sleep(2.0)
            continue
        try:
            geometry = _geometry(rs, prof, ir_width)
            scale = prof.get_device().first_depth_sensor().get_depth_scale() if dsock else 0.001
        except Exception as exc:  # noqa: BLE001 - isolate a failed camera frame or device
            print("colour geometry failed: %s -- restarting" % str(exc)[:90], flush=True)
            pipe.stop()
            time.sleep(2.0)
            continue
        print("COLOR up | D435i %s | %dx%d @ %d fps | q%d | %s | depth %s | baseline %.4f m" % (
            chosen, width, height, fps, jpeg_quality, endpoint,
            "%s @ %.0f Hz" % (depth_endpoint, depth_hz) if dsock else "off",
            geometry["baseline_m"]), flush=True)
        sent = dropped = depth_sent = 0
        last_depth_ns = 0
        bytes_sent = 0
        report_at = time.monotonic() + report_s
        try:
            while time.monotonic() < deadline:
                frames = pipe.wait_for_frames(5000)
                color = frames.get_color_frame()
                if not color:
                    continue
                received_ns = time.time_ns()
                captured_ns, timestamp_basis = capture_time(color, rs, received_ns)
                ok, jpeg = cv2.imencode(".jpg", np.asanyarray(color.get_data()),
                                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    continue
                header = {
                    "sequence": sequence,
                    "capture_realtime_ns": captured_ns,
                    "capture_timestamp_basis": timestamp_basis,
                    "sdk_receive_realtime_ns": received_ns,
                    "publish_realtime_ns": time.time_ns(),
                    "width": width, "height": height,
                    "jpeg_quality": jpeg_quality,
                    "camera_serial": chosen,
                    "geometry": geometry,
                }
                sequence += 1
                message = pack(header, jpeg.tobytes())
                try:
                    sock.send(message, flags=zmq.NOBLOCK)
                    sent += 1
                    bytes_sent += len(message)
                except zmq.Again:
                    dropped += 1
                # 5 ms slack so capture jitter cannot push the gate to a later colour frame
                if dsock is not None and captured_ns - last_depth_ns >= 1e9 / depth_hz - 5e6:
                    depth = frames.get_depth_frame()
                    if depth:
                        z = np.asanyarray(depth.get_data())
                        mm = z if abs(scale - 0.001) < 1e-9 else np.clip(
                            np.rint(z.astype(np.float32) * (scale * 1000.0)), 0, 65535)
                        body = np.ascontiguousarray(mm, dtype="<u2").tobytes()
                        packet = DEPTH_HEADER.pack(DEPTH_MAGIC, 1, 1, 0, depth_sequence, captured_ns,
                                                   time.monotonic_ns(), 0, z.shape[0], z.shape[1],
                                                   len(body)) + body
                        try:
                            dsock.send(packet, flags=zmq.NOBLOCK)
                            depth_sequence += 1
                            depth_sent += 1
                        except zmq.Again:
                            pass
                        last_depth_ns = captured_ns
                now = time.monotonic()
                if now >= report_at:
                    print("COLOR | %.1f Hz | sent %d dropped %d | %.0f kB/frame | depth %.1f Hz" % (
                        sent / report_s, sent, dropped,
                        bytes_sent / max(sent, 1) / 1e3, depth_sent / report_s), flush=True)
                    sent = dropped = depth_sent = 0
                    bytes_sent = 0
                    report_at = now + report_s
        except Exception as exc:  # noqa: BLE001 - isolate a failed camera frame or device
            print("colour loop error: %s -- restarting" % str(exc)[:90], flush=True)
        try:
            pipe.stop()
        except Exception as exc:  # noqa: BLE001 - isolate a failed camera frame or device
            print("colour stop failed: %s" % str(exc)[:90], flush=True)
    sock.close(0)
    if dsock is not None:
        dsock.close(0)
    context.term()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ir-width", type=int, default=848,
                        help="the stereo profile whose intrinsics the header carries")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--duration-s", type=float, default=86400.0)
    parser.add_argument("--serial", default="", help="the D435i to stream, by serial")
    parser.add_argument("--exclude-serial", default="",
                        help="stream the only connected D435i whose serial differs from this one")
    parser.add_argument("--max-fps", type=int, default=60)
    parser.add_argument("--depth-endpoint", default="",
                        help="also stream this camera's own depth (HMDPTH01) on this PUB endpoint")
    parser.add_argument("--depth-hz", type=float, default=15.0)
    args = parser.parse_args(argv)
    serve(args.endpoint, args.width, args.height, args.ir_width, args.jpeg_quality,
          args.duration_s, args.serial, args.exclude_serial, args.max_fps, args.depth_endpoint,
          args.depth_hz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
