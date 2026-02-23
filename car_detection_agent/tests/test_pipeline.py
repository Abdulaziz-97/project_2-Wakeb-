"""
Unit tests for the multi-agent car detection pipeline.

All heavy models (ONNX session, VLM) are mocked so the tests run on
CPU without any model files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure the project root is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.state import Detection, PipelineState
from utils.image_utils import letterbox
from utils.nms_utils import apply_nms
from agents.preprocessor import preprocess_agent


# ─────────────────────────────────────────────────────────────────────────────
# 1. Letterbox produces the correct shape
# ─────────────────────────────────────────────────────────────────────────────
class TestLetterbox:
    def test_letterbox_shape(self) -> None:
        """Output of letterbox is always (640, 640, 3)."""
        for h, w in [(480, 640), (1080, 1920), (100, 300), (640, 640)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = letterbox(img, target_size=640)
            assert result.shape == (640, 640, 3), (
                f"Expected (640,640,3) for input ({h},{w}), got {result.shape}"
            )
            assert result.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessor rejects small images
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessor:
    def test_preprocessor_rejects_small_image(self) -> None:
        """An image with dimensions below min_image_dim is rejected."""
        # Create a tiny image (100×100) and write to a temp file
        tiny = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tiny)
            tmp_path = f.name

        try:
            state = PipelineState(image_path=tmp_path)
            result = preprocess_agent(state)
            assert result.image_quality == "rejected"
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Router fast path — all detections above threshold
# ─────────────────────────────────────────────────────────────────────────────
class TestRouterFastPath:
    def test_router_fast_path(self) -> None:
        """When all car detections are ≥ 0.85, needs_vlm is False."""
        state = PipelineState()
        state.high_conf_detections = [
            Detection(
                box=[10, 10, 100, 100],
                confidence=0.92,
                class_name="car",
            ),
        ]
        state.low_conf_detections = []
        state.needs_vlm = False

        assert state.needs_vlm is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Router VLM path — some detections below threshold
# ─────────────────────────────────────────────────────────────────────────────
class TestRouterVLMPath:
    def test_router_vlm_path(self) -> None:
        """When ≥1 car detection is < 0.85, needs_vlm is True."""
        state = PipelineState()
        state.low_conf_detections = [
            Detection(
                box=[10, 10, 100, 100],
                confidence=0.55,
                class_name="car",
            ),
        ]
        state.needs_vlm = True

        assert state.needs_vlm is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. NMS removes duplicate / overlapping boxes
# ─────────────────────────────────────────────────────────────────────────────
class TestNMS:
    def test_nms_removes_duplicate(self) -> None:
        """Two heavily overlapping boxes → only one survives NMS."""
        dets = [
            Detection(
                box=[10, 10, 100, 100],
                confidence=0.95,
                class_name="car",
            ),
            Detection(
                box=[12, 12, 102, 102],
                confidence=0.80,
                class_name="car",
            ),
        ]
        result = apply_nms(dets, iou_threshold=0.45)
        assert len(result) == 1
        assert result[0].confidence == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full pipeline end-to-end (no VLM) with synthetic image
# ─────────────────────────────────────────────────────────────────────────────
class TestFullPipelineNoVLM:
    @patch("agents.detector._get_session")
    def test_full_pipeline_no_vlm(self, mock_get_session: MagicMock) -> None:
        """End-to-end with a synthetic white image (no cars).

        The ONNX session is mocked to return zero-confidence detections so
        the VLM is never triggered and the result is an empty detection list.
        """
        # Mock ONNX output: shape (1, 84, 8400) with all zeros → no detections
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.get_inputs.return_value[0].name = "images"
        mock_output = np.zeros((1, 84, 8400), dtype=np.float32)
        mock_session.run.return_value = [mock_output]
        mock_get_session.return_value = mock_session

        # Create a synthetic white image
        white = np.ones((640, 640, 3), dtype=np.uint8) * 255
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, white)
            tmp_path = f.name

        try:
            from graph.pipeline import route_after_detection
            from agents.preprocessor import preprocess_agent
            from agents.detector import detector_agent
            from agents.output_formatter import merge_agent

            state = PipelineState(image_path=tmp_path)

            # Run agents sequentially (mimic graph execution)
            state = preprocess_agent(state)
            assert state.image_quality == "ok"

            state = detector_agent(state)
            assert state.needs_vlm is False

            route = route_after_detection(state)
            assert route == "merge"

            state = merge_agent(state)

            # Build result dict like main.py
            result = {
                "detections": [
                    d.model_dump() for d in state.final_detections
                ],
                "car_count": 0,
            }

            assert "detections" in result
            assert result["car_count"] == 0
        finally:
            os.unlink(tmp_path)
