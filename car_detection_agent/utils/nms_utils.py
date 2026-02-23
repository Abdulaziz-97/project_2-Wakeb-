"""
Non-Maximum Suppression utility wrapping :func:`torchvision.ops.nms`.

Operates on :class:`Detection` objects for seamless integration with the
pipeline state.
"""

from __future__ import annotations

import torch
import torchvision

from graph.state import Detection


def apply_nms(
    detections: list[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Apply NMS to a list of :class:`Detection` objects.

    Parameters
    ----------
    detections : list[Detection]
        Input detections (may contain overlapping boxes).
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    list[Detection]
        Filtered detections after NMS.
    """
    if not detections:
        return []

    boxes = torch.tensor(
        [d.box for d in detections], dtype=torch.float32
    )
    scores = torch.tensor(
        [d.confidence for d in detections], dtype=torch.float32
    )

    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

    return [detections[i] for i in keep_indices.tolist()]
