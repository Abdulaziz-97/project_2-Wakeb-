"""
VLM verifier agent — crop-level verification using Qwen2.5-VL-3B-Instruct.

The VLM is lazily loaded on first call to ``_get_vlm()``.  For each
low-confidence detection the agent crops the original image (with padding),
sends the crop to the VLM for a single-word vehicle-type classification,
and promotes verified cars.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image

from config import config
from graph.state import Detection

if TYPE_CHECKING:
    from graph.state import PipelineState

logger = logging.getLogger(__name__)

# ── Module-level VLM model + processor (lazy-loaded once) ───────────────────
_vlm_model: Any = None
_vlm_processor: Any = None


def _get_vlm():
    """Lazily initialise the Qwen2.5-VL model and processor."""
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.vlm_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _vlm_processor = AutoProcessor.from_pretrained(config.vlm_model_id)
    return _vlm_model, _vlm_processor


def vlm_verifier_agent(state: "PipelineState") -> "PipelineState":
    """Verify low-confidence detections by sending crops to a VLM.

    Parameters
    ----------
    state : PipelineState
        Must have ``raw_image`` (BGR numpy) and ``low_conf_detections``.

    Returns
    -------
    PipelineState
        Populated with ``vlm_verified_detections`` and
        ``latency_ms["vlm_verifier"]``.
    """
    t0 = time.perf_counter()

    verified: list[Detection] = []

    if state.raw_image is None or not state.low_conf_detections:
        state.vlm_verified_detections = verified
        state.latency_ms["vlm_verifier"] = (time.perf_counter() - t0) * 1000
        return state

    # Convert BGR numpy → PIL RGB
    rgb = state.raw_image[:, :, ::-1]  # BGR → RGB
    pil_image = Image.fromarray(rgb.copy())
    img_h, img_w = state.raw_image.shape[:2]

    model, processor = _get_vlm()

    # Lazy imports for heavy deps (not needed at module level)
    import torch
    from qwen_vl_utils import process_vision_info

    for det in state.low_conf_detections:
        try:
            # Crop with 10 px padding, clamped to image bounds
            x1 = max(0, int(det.box[0]) - 10)
            y1 = max(0, int(det.box[1]) - 10)
            x2 = min(img_w, int(det.box[2]) + 10)
            y2 = min(img_h, int(det.box[3]) + 10)

            crop = pil_image.crop((x1, y1, x2, y2))

            # Build VLM prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {
                            "type": "text",
                            "text": (
                                "What vehicle type is in this image? "
                                "Answer with one word only: , truck, bus, "
                                "motorcycle, or none."
                            ),
                        },
                    ],
                }
            ]

            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=5)

            # Decode — strip the prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            label = output_text.strip().split()[-1].lower()

            if label in ("car", "vehicle", "automobile"):
                verified.append(
                    Detection(
                        box=det.box,
                        confidence=max(det.confidence, 0.70),
                        class_name="car",
                        verified_by="vlm",
                    )
                )

        except Exception:
            logger.warning(
                "VLM verification failed for crop %s — skipping.",
                det.box,
                exc_info=True,
            )
            continue

    state.vlm_verified_detections = verified
    state.latency_ms["vlm_verifier"] = (time.perf_counter() - t0) * 1000
    return state
