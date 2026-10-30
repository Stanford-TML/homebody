"""The pinned SAM 2 factory builds HomeBody's streaming predictor with its batched mask heads."""

import pytest

from homebody.primitives.cameras._sam2_heads import forward_sam_heads


def test_the_pinned_factory_builds_the_streaming_predictor(monkeypatch):
    """The upstream factory must construct HomeBody's predictor, or the upstream heads run silently."""
    for module in ("torch", "hydra", "iopath", "torchvision"):
        pytest.importorskip(module)
    from pathlib import Path

    from homebody.backends.real_config import RealConfig
    from homebody.primitives.cameras import tracking

    root = tracking.verify_runtime(Path(__file__).resolve().parents[1] / "vendor/samurai")
    monkeypatch.syspath_prepend(str(root))
    from sam2.build_sam import build_sam2_video_predictor

    from homebody.primitives.cameras._sam2_predictor import StreamingPredictor

    model = build_sam2_video_predictor(RealConfig.segmentation_config, None, device="cpu",
                                       hydra_overrides_extra=[tracking.PREDICTOR])
    assert type(model) is StreamingPredictor
    assert model._empty_prompt_cache == {} and model._forward_sam_heads.__func__ is forward_sam_heads
