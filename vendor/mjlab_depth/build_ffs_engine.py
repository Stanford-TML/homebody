#!/usr/bin/env python3
# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Build the Fast-FoundationStereo TensorRT engine for this machine's GPU.

With TRT 10.x, parse the ORIGINAL (float32) single-file ONNX and let the
TensorRT builder do precision assignment with the FP16 flag. Do NOT feed a
graph-level fp16-converted ONNX to TRT 10.11 -- the blanket conversion leaves
float activations feeding half convolution weights, which its parser refuses.

TRT >= 11 removed the FP16 builder flag entirely (strongly-typed networks);
there, build with --fp32 or from an fp16 ONNX, or use an existing engine that
still deserializes.

    python vendor/mjlab_depth/build_ffs_engine.py \
        --onnx  /path/to/fast_foundationstereo.onnx \
        --output /path/to/new/fast_foundationstereo.engine

Engines are GPU- and TRT-version-specific; rebuild after driver/TRT upgrades.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fp32", action="store_true", help="Skip the FP16 builder flag")
    parser.add_argument("--workspace-gb", type=float, default=6.0)
    args = parser.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT >= 10 networks are always explicit-batch; flag 0 is the modern form.
    network = builder.create_network(0)
    parser_trt = trt.OnnxParser(network, logger)
    print(f"parsing {args.onnx} (sha256 {sha256_of(args.onnx)[:12]}…)")
    # parse_from_file resolves external weight data (.onnx.data) beside the model.
    if not parser_trt.parse_from_file(str(args.onnx)):
        for i in range(parser_trt.num_errors):
            print(f"  parser error: {parser_trt.get_error(i)}")
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30))
    )
    if not args.fp32:
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            raise SystemExit(
                f"TensorRT {trt.__version__} has no FP16 builder flag (TRT >= 11 "
                "is strongly typed: precision comes from the ONNX graph). Either "
                "build in an environment with TRT 10.x, pass --fp32, or feed an "
                "fp16 ONNX. Note: an engine built earlier may still deserialize -- "
                "try it with TrtFfsRunner before rebuilding."
            )
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input  {t.name} {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name} {tuple(t.shape)} {t.dtype}")

    print(f"building ({'fp32' if args.fp32 else 'fp16'}, TRT {trt.__version__})…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(serialized))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"engine sha256: {sha256_of(args.output)}")


if __name__ == "__main__":
    main()
