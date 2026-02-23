"""
Preprocessor agent — first node in the LangGraph pipeline.

Responsibilities:
  1. Load the image from disk via OpenCV.
  2. Reject images that are too small or unreadable.
  3. Enhance dark images using CLAHE on the L channel (LAB colour space).
  4. Apply letterbox resize to the configured input size (default 640×640).
  5. Record preprocessing latency.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import config
from utils.image_utils import letterbox

if TYPE_CHECKING:
    from graph.state import PipelineState


def preprocess_agent(state: "PipelineState") -> "PipelineState":
    """Read, validate, optionally enhance, and resize the input image.

    Parameters
    ----------
    state : PipelineState
        Must have ``image_path`` set to a valid filesystem path.

    Returns
    -------
    PipelineState
        Updated in-place with ``raw_image``, ``preprocessed_image``,
        ``image_quality``, and ``latency_ms["preprocess"]``.
    """
    t0 = time.perf_counter()

    # 1. Read image -----------------------------------------------------------
    img = cv2.imread(state.image_path)

    if img is None:
        state.image_quality = "rejected"
        state.latency_ms["preprocess"] = (time.perf_counter() - t0) * 1000
        return state

    h, w = img.shape[:2]

    # 2. Reject images below minimum dimension --------------------------------
    if min(h, w) < config.min_image_dim:
        state.image_quality = "rejected"
        state.latency_ms["preprocess"] = (time.perf_counter() - t0) * 1000
        return state

    # 3. Dark-image enhancement -----------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    if mean_brightness < 50:
        state.image_quality = "dark"
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Store original & letterbox -------------------------------------------
    state.raw_image = img.copy()
    state.preprocessed_image = letterbox(img, config.input_size)

    # 5. Latency --------------------------------------------------------------
    state.latency_ms["preprocess"] = (time.perf_counter() - t0) * 1000
    return state
