"""SAMURAI predictor with batched mask-head transfers."""
from sam2.sam2_video_predictor import SAM2VideoPredictor

from ._sam2_heads import forward_sam_heads


class StreamingPredictor(SAM2VideoPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._empty_prompt_cache = {}

    _forward_sam_heads = forward_sam_heads
