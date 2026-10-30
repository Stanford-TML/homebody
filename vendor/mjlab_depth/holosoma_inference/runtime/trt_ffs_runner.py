# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""TensorRT runner for a single-file Fast-FoundationStereo engine.

Derived from TML-Humanoid-Deploy's ``utils/ffs_runner.py``, extended with the
per-stage H2D / infer / D2H timings that the FFSTEL01 telemetry contract reports.

The single-ONNX export expects pre-normalized float32 inputs
``(pixel_0_255 - imagenet_mean) / imagenet_std`` and returns disparity in
pixels at the engine resolution. Pre/post processing stays on the GPU; the
only host<->device traffic is the uint8 pair in and the float32 disparity out.
"""

from __future__ import annotations

import numpy as np
import torch

_IMAGENET_MEAN = (123.675, 116.28, 103.53)
_IMAGENET_STD = (58.395, 57.12, 57.375)


class TrtFfsRunner:
    def __init__(self, engine_path: str, device: str = "cuda"):
        import tensorrt as trt

        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path} (TRT {trt.__version__}); "
                "rebuild the engine on this machine with "
                "vendor/mjlab_depth/build_ffs_engine.py."
            )
        self.context = self.engine.create_execution_context()

        ins, outs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ins.append(name)
            else:
                outs.append(name)
        if len(ins) != 2:
            raise RuntimeError(f"expected 2 inputs (left,right), got {ins}")
        self.in_left = next((n for n in ins if "left" in n.lower()), ins[0])
        self.in_right = next(
            (n for n in ins if "right" in n.lower()),
            ins[1] if ins[1] != self.in_left else ins[0],
        )
        self.out_name = outs[0]

        shape = tuple(self.engine.get_tensor_shape(self.in_left))  # (1,3,H,W)
        self.height, self.width = int(shape[2]), int(shape[3])

        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        self._out = torch.empty(out_shape, device=self.device, dtype=torch.float32)
        self._in_bufs: dict = {}
        # (host_to_device_ms, infer_ms, device_to_host_ms) of the last call.
        self.last_timings_ms: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """uint8 (H,W) mono or (H,W,3) -> normalized (1,3,H,W) float32 cuda."""
        t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device, non_blocking=True)
        if t.ndim == 2:
            t = t.unsqueeze(-1).expand(-1, -1, 3)
        t = t.permute(2, 0, 1).unsqueeze(0).float()
        if t.shape[2] != self.height or t.shape[3] != self.width:
            t = torch.nn.functional.interpolate(
                t, (self.height, self.width), mode="bilinear", align_corners=False
            )
        return ((t - self._mean) / self._std).contiguous()

    @torch.no_grad()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        import time

        t0 = time.perf_counter()
        l = self._preprocess(left)
        r = self._preprocess(right)
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        self.context.set_input_shape(self.in_left, tuple(l.shape))
        self.context.set_input_shape(self.in_right, tuple(r.shape))
        # Keep references alive until execution finishes.
        self._in_bufs = {"l": l, "r": r}
        self.context.set_tensor_address(self.in_left, int(l.data_ptr()))
        self.context.set_tensor_address(self.in_right, int(r.data_ptr()))
        self.context.set_tensor_address(self.out_name, int(self._out.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.execute_async_v3(stream)
        torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        disp = self._out.squeeze().float().cpu().numpy()
        t3 = time.perf_counter()
        self.last_timings_ms = (
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
        )
        return disp
