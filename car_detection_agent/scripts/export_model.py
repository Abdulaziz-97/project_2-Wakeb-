"""
One-time model export script: YOLOv26s → FP16 ONNX.

Exports the Ultralytics YOLOv26s model to ONNX format, then converts the
floating-point weights from FP32 to FP16 for faster inference while
keeping I/O tensor types unchanged.

Usage::

    python scripts/export_model.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
from ultralytics import YOLO


def main() -> None:
    """Export YOLOv26s to FP16 ONNX and save to models/ directory."""
    # 1. Load YOLOv26s from Ultralytics
    model = YOLO("yolo26s.pt")

    # 2. Export to ONNX (FP32)
    onnx_fp32_path = model.export(
        format="onnx",
        simplify=True,
        imgsz=640,
        opset=17,
    )
    print(f"FP32 ONNX exported to: {onnx_fp32_path}")

    # 3. Load the FP32 ONNX model
    onnx_model = onnx.load(onnx_fp32_path)

    # 4. Convert to FP16 (keep I/O types intact for compatibility)
    onnx_fp16_model = convert_float_to_float16(
        onnx_model, keep_io_types=True
    )

    # 5. Save FP16 ONNX
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    fp16_path = os.path.join(output_dir, "yolo26s_fp16.onnx")
    onnx.save(onnx_fp16_model, fp16_path)

    size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"FP16 ONNX saved to: {fp16_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
