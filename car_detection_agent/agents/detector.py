"""
Fast detector agent — runs YOLOv26s FP16 ONNX inference via ONNXRuntime.

The ONNX session is loaded **once** at module level to avoid repeated
initialisation overhead.  The agent parses the standard YOLO output tensor
(1, 84, 8400), filters by pre-threshold 0.25, and splits results into
high- and low-confidence buckets for downstream routing.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from config import config
from graph.state import Detection

if TYPE_CHECKING:
    from graph.state import PipelineState

logger = logging.getLogger(__name__)

# ── COCO 80-class label list (index 2 = "car") ─────────────────────────────
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Module-level ONNX session (loaded once) ─────────────────────────────────
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Lazily initialise the ONNX session on first call."""
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            config.yolo_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return _session


def detector_agent(state: "PipelineState") -> "PipelineState":
    """Run YOLOv26s FP16 inference and split detections by confidence.

    Parameters
    ----------
    state : PipelineState
        Must contain a valid ``preprocessed_image`` (640×640 uint8 BGR).

    Returns
    -------
    PipelineState
        Populated with ``all_detections``, ``high_conf_detections``,
        ``low_conf_detections``, ``needs_vlm``, and
        ``latency_ms["detector"]``.
    """
    t0 = time.perf_counter()

    # Skip if the image was already rejected upstream
    if state.image_quality == "rejected":
        state.latency_ms["detector"] = (time.perf_counter() - t0) * 1000
        return state

    session = _get_session()
    input_name: str = session.get_inputs()[0].name

    # ── Prepare input blob ──────────────────────────────────────────────
    img = state.preprocessed_image.astype(np.float32) / 255.0  # normalise
    blob = np.transpose(img, (2, 0, 1))  # HWC → CHW
    blob = np.expand_dims(blob, axis=0)  # add batch dim → (1, 3, 640, 640)

    # ── Inference ───────────────────────────────────────────────────────
    outputs = session.run(None, {input_name: blob})
    preds = outputs[0]  # shape: (1, 84, 8400)

    # ── Parse predictions ───────────────────────────────────────────────
    preds = preds[0].T  # (8400, 84)

    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    class_scores = preds[:, 4:]  # (8400, 80)
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Pre-threshold at 0.25
    mask = confidences > 0.25
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # ── Build Detection objects ─────────────────────────────────────────
    all_dets: list[Detection] = []
    high_dets: list[Detection] = []
    low_dets: list[Detection] = []

    for i in range(len(confidences)):
        cid = int(class_ids[i])
        cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else "unknown"
        conf = float(confidences[i])
        det = Detection(
            box=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            confidence=conf,
            class_name=cname,
            verified_by="yolo",
        )
        all_dets.append(det)

        if cname == config.target_class:
            if conf >= config.confidence_threshold:
                high_dets.append(det)
            else:
                low_dets.append(det)

    # Cap VLM crops
    low_dets = low_dets[: config.max_vlm_crops]

    state.all_detections = all_dets
    state.high_conf_detections = high_dets
    state.low_conf_detections = low_dets
    state.needs_vlm = len(low_dets) > 0

    state.latency_ms["detector"] = (time.perf_counter() - t0) * 1000
    return state
