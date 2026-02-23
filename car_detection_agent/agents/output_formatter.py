"""
Output formatter / merge agent — combines high-confidence and VLM-verified
detections, applies NMS once, and stores the final result.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from config import config
from utils.nms_utils import apply_nms

if TYPE_CHECKING:
    from graph.state import PipelineState


def merge_agent(state: "PipelineState") -> "PipelineState":
    """Merge detection lists and apply Non-Maximum Suppression.

    Parameters
    ----------
    state : PipelineState
        Must have ``high_conf_detections`` and optionally
        ``vlm_verified_detections``.

    Returns
    -------
    PipelineState
        Updated with ``final_detections`` and ``latency_ms["merge"]``.
    """
    t0 = time.perf_counter()

    combined = list(state.high_conf_detections) + list(
        state.vlm_verified_detections
    )
    state.final_detections = apply_nms(combined, config.iou_threshold)

    state.latency_ms["merge"] = (time.perf_counter() - t0) * 1000
    return state
