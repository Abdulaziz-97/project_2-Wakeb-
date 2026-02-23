"""
Attribute extractor agent — async, non-blocking post-processing.

Counts detected cars and populates :attr:`CarAttributes` on the pipeline
state.  Designed to be ``await``-ed after the main graph completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph.state import CarAttributes

if TYPE_CHECKING:
    from graph.state import PipelineState


async def attribute_agent_async(state: "PipelineState") -> "PipelineState":
    """Count final car detections and store attributes.

    Parameters
    ----------
    state : PipelineState
        Must have ``final_detections`` populated by the merge agent.

    Returns
    -------
    PipelineState
        Updated with ``attributes.car_count``.
    """
    car_count = sum(
        1 for d in state.final_detections if d.class_name == "car"
    )
    state.attributes = CarAttributes(car_count=car_count)
    return state
