#!/usr/bin/env python3
# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Foreground relay for native-rate Fast-FoundationStereo output."""

from __future__ import annotations

import argparse
from pathlib import Path

from holosoma_inference.runtime.depth_vision_relay import run_relay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-depth", default="tcp://127.0.0.1:55565")
    parser.add_argument("--source-telemetry", default="tcp://127.0.0.1:55564")
    parser.add_argument("--output", default="tcp://127.0.0.1:55563")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--expected-engine-sha256")
    parser.add_argument("--expected-camera-sha256")
    parser.add_argument("--expected-calibration-sha256")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--startup-timeout-s", type=float, default=15.0)
    args = parser.parse_args()
    if args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive")
    expected_values = (
        args.expected_engine_sha256,
        args.expected_camera_sha256,
        args.expected_calibration_sha256,
    )
    if any(expected_values) and not all(expected_values):
        parser.error("expected FFS identities must be supplied together")
    expected_identities = (
        {
            "engine_sha256": args.expected_engine_sha256,
            "camera_sha256": args.expected_camera_sha256,
            "calibration_sha256": args.expected_calibration_sha256,
        }
        if all(expected_values)
        else None
    )
    print(
        "VISION relay starting | "
        f"depth={args.source_depth} telemetry={args.source_telemetry} output={args.output} | "
        f"relay_identity_gate={'explicit' if expected_identities is not None else 'transport-owned'}",
        flush=True,
    )
    try:
        run_relay(
            source_depth_endpoint=args.source_depth,
            source_telemetry_endpoint=args.source_telemetry,
            output_endpoint=args.output,
            shape=(args.height, args.width),
            expected_identities=expected_identities,
            ready_file=args.ready_file,
            startup_timeout_s=args.startup_timeout_s,
        )
    except KeyboardInterrupt:
        print("VISION stopped", flush=True)
    except RuntimeError as exc:
        raise SystemExit(f"VISION FAILED | {exc}") from exc


if __name__ == "__main__":
    main()
