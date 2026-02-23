"""
Pipeline state schema for the multi-agent car detection pipeline.

All agents receive and return :class:`PipelineState`.  Heavy objects such as
numpy arrays are supported via ``arbitrary_types_allowed``.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class Detection(BaseModel):
    """A single object detection."""

    box: list[float]  # [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_name: str
    verified_by: str = "yolo"  # "yolo" or "vlm"


class CarAttributes(BaseModel):
    """Aggregated attributes extracted from final detections."""

    car_count: int = 0
    # extend later


class PipelineState(BaseModel):
    """
    Shared mutable state passed through every node in the LangGraph pipeline.
    """

    # ── Input ────────────────────────────────────────────────────────
    image_path: str = ""
    raw_image: Optional[Any] = None  # numpy ndarray BGR

    # ── Preprocessing ────────────────────────────────────────────────
    preprocessed_image: Optional[Any] = None  # numpy ndarray 640×640
    image_quality: str = "ok"
    # values: "ok" | "dark" | "blurry" | "rejected"

    # ── Detection outputs ────────────────────────────────────────────
    all_detections: list[Detection] = []
    high_conf_detections: list[Detection] = []
    low_conf_detections: list[Detection] = []

    # ── Router flag ──────────────────────────────────────────────────
    needs_vlm: bool = False

    # ── VLM output ───────────────────────────────────────────────────
    vlm_verified_detections: list[Detection] = []

    # ── Final ────────────────────────────────────────────────────────
    final_detections: list[Detection] = []
    attributes: Optional[CarAttributes] = None

    # ── Diagnostics ──────────────────────────────────────────────────
    result: Optional[dict] = None
    latency_ms: dict[str, float] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)
