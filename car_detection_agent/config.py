"""
Configuration module for the car detection multi-agent pipeline.

Defines all tunable parameters as a Pydantic v2 BaseModel with sensible
defaults.  Import the global ``config`` instance anywhere in the project.
"""

from pydantic import BaseModel


class Config(BaseModel):
    """Central configuration for every agent in the pipeline."""

    yolo_onnx_path: str = "models/yolo26s_fp16.onnx"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    confidence_threshold: float = 0.85
    iou_threshold: float = 0.45
    target_class: str = "car"
    input_size: int = 640
    min_image_dim: int = 320
    max_vlm_crops: int = 5
    device: str = "cuda"


config = Config()
