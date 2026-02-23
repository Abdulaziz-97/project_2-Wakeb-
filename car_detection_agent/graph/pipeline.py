"""
LangGraph pipeline definition for the multi-agent car detection system.

Builds a :class:`StateGraph` with conditional routing:

  preprocess → detector → (route) → vlm_verifier → merge → END
                                   ↘ merge → END    (fast path)
                                   ↘ END             (rejected)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.preprocessor import preprocess_agent
from agents.detector import detector_agent
from agents.vlm_verifier import vlm_verifier_agent
from agents.output_formatter import merge_agent
from graph.state import PipelineState


def route_after_detection(state: PipelineState) -> str:
    """Decide the next node after the detector finishes.

    Returns
    -------
    str
        ``"end"`` if the image was rejected, ``"vlm_verifier"`` when
        low-confidence detections exist, or ``"merge"`` for the fast path.
    """
    if state.image_quality == "rejected":
        return "end"
    if state.needs_vlm:
        return "vlm_verifier"
    return "merge"


def build_pipeline() -> StateGraph:
    """Construct and compile the LangGraph car-detection pipeline.

    Returns
    -------
    CompiledGraph
        Ready to be invoked via ``pipeline.invoke(state)``.
    """
    graph = StateGraph(PipelineState)

    # ── Nodes ───────────────────────────────────────────────────────────
    graph.add_node("preprocess", preprocess_agent)
    graph.add_node("detector", detector_agent)
    graph.add_node("vlm_verifier", vlm_verifier_agent)
    graph.add_node("merge", merge_agent)

    # ── Entry ───────────────────────────────────────────────────────────
    graph.set_entry_point("preprocess")

    # ── Edges ───────────────────────────────────────────────────────────
    graph.add_edge("preprocess", "detector")

    graph.add_conditional_edges(
        "detector",
        route_after_detection,
        {
            "vlm_verifier": "vlm_verifier",
            "merge": "merge",
            "end": END,
        },
    )

    graph.add_edge("vlm_verifier", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


# Expose a ready-to-use compiled pipeline
pipeline = build_pipeline()
