"""
Entry point for the multi-agent car detection pipeline.

Usage::

    python main.py <image_path>
    python main.py              # defaults to test.jpg
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

import comet_ml

from agents.attribute_extractor import attribute_agent_async
from graph.pipeline import pipeline
from graph.state import PipelineState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def run(image_path: str) -> dict:
    """Execute the full car-detection pipeline on a single image.

    Parameters
    ----------
    image_path : str
        Filesystem path to the input image.

    Returns
    -------
    dict
        Structured JSON-serialisable result containing detections,
        car count, attributes, per-agent latencies, and image quality.
    """
    state = PipelineState(image_path=image_path)

    t_start = time.perf_counter()

    # Run the LangGraph pipeline synchronously (agents are CPU/GPU-bound)
    result_state: PipelineState = pipeline.invoke(state)

    # Async attribute extraction (non-blocking, fires after merge)
    await attribute_agent_async(result_state)

    total_ms = (time.perf_counter() - t_start) * 1000
    result_state.latency_ms["total"] = total_ms

    # ── Comet ML logging ────────────────────────────────────────────────
    try:
        experiment = comet_ml.Experiment(
            project_name="car-detection-agent",
            auto_metric_logging=False,
            auto_param_logging=False,
            auto_output_logging="native",
        )
        for key, value in result_state.latency_ms.items():
            experiment.log_metric(f"latency_ms/{key}", value)
        car_count = (
            result_state.attributes.car_count
            if result_state.attributes
            else 0
        )
        experiment.log_metric("car_count", car_count)
        experiment.log_metric(
            "vlm_triggered", 1 if result_state.needs_vlm else 0
        )
        experiment.end()
    except Exception:
        logger.warning("Comet ML logging failed — continuing.", exc_info=True)

    # ── Build result dict ───────────────────────────────────────────────
    return {
        "detections": [d.model_dump() for d in result_state.final_detections],
        "car_count": (
            result_state.attributes.car_count
            if result_state.attributes
            else 0
        ),
        "attributes": (
            result_state.attributes.model_dump()
            if result_state.attributes
            else {}
        ),
        "latency_ms": result_state.latency_ms,
        "image_quality": result_state.image_quality,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(json.dumps(asyncio.run(run(path)), indent=2))
