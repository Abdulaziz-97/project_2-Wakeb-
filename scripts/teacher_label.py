#!/usr/bin/env python3
"""
teacher_label.py – Generate class-aware pseudo-labels using SAM3 PCS.

Uses SAM3SemanticPredictor (Promptable Category Segmentation) with text prompts
to produce class-specific instance masks.

Strategy:
  - For suv, sedan, bus: single PCS call with text prompt.
  - For truck: multiple sub-prompts ("truck", "pickup truck", "box truck",
    "semi truck"), merge overlapping masks via IoU into class_id=3.

Inputs:
  - tiles_manifest.jsonl
  - prompts.yaml (class prompts + exemplar boxes + merge settings)

Outputs per tile:
  - {tile_stem}_preds.json   (instances: class_id, bbox, polygon, score)
  - {tile_stem}_overlay.png  (QA visualization)
"""
import argparse, json, os, sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════
class Instance:
    """Single detected instance."""
    def __init__(self, class_id: int, class_name: str, bbox_xyxy: list,
                 polygon: list, score: float, prompt_used: str = ""):
        self.class_id = class_id
        self.class_name = class_name
        self.bbox_xyxy = bbox_xyxy
        self.polygon = polygon
        self.score = score
        self.prompt_used = prompt_used

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": self.bbox_xyxy,
            "polygon": self.polygon,
            "score": round(self.score, 4),
            "prompt_used": self.prompt_used,
        }


# ═══════════════════════════════════════════════════════════════════
# SAM3 PCS Teacher
# ═══════════════════════════════════════════════════════════════════
class SAM3PCSTeacher:
    """
    Teacher using SAM3 Promptable Category Segmentation (PCS).
    Uses SAM3SemanticPredictor with text prompts for class-aware masks.
    """

    def __init__(self):
        self.predictor = None

    def load(self, model_path: str, device: str = "0") -> None:
        """Load SAM3SemanticPredictor."""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=model_path,
                half=True,
                verbose=False,
                device=device,
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"[PCS Teacher] Loaded SAM3SemanticPredictor from {model_path}")
        except Exception as e:
            print(f"[PCS Teacher] Failed to load: {e}", file=sys.stderr)
            self.predictor = None

    def predict_class(self, image_path: str, prompts: List[str],
                      class_id: int, class_name: str) -> List[Instance]:
        """
        Run PCS for one class (may have multiple text prompts, e.g. truck bundle).
        Returns instances all assigned to the given class_id.
        """
        if self.predictor is None:
            return []

        all_instances = []
        try:
            self.predictor.set_image(image_path)

            for prompt_text in prompts:
                results = self.predictor(text=[prompt_text])

                if results and len(results) > 0:
                    result = results[0]
                    if result.masks is not None and len(result.masks) > 0:
                        masks = result.masks.data.cpu().numpy()
                        # Get boxes and confidences
                        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
                        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

                        for idx in range(len(masks)):
                            mask = masks[idx]
                            polygon = self._mask_to_polygon(mask)
                            if polygon is None or len(polygon) < 3:
                                continue

                            bbox = (boxes[idx].tolist() if boxes is not None and idx < len(boxes)
                                    else self._polygon_to_bbox(polygon))
                            score = (float(confs[idx]) if confs is not None and idx < len(confs)
                                     else 0.5)

                            all_instances.append(Instance(
                                class_id=class_id,
                                class_name=class_name,
                                bbox_xyxy=bbox,
                                polygon=polygon,
                                score=score,
                                prompt_used=prompt_text,
                            ))

        except Exception as e:
            print(f"  [PCS Teacher] Error predicting '{class_name}': {e}", file=sys.stderr)

        return all_instances

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> Optional[list]:
        """Convert binary mask to largest polygon."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return None
        epsilon = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx.reshape(-1, 2).tolist()

    @staticmethod
    def _polygon_to_bbox(polygon: list) -> list:
        pts = np.array(polygon)
        return [
            float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()),
        ]


# ═══════════════════════════════════════════════════════════════════
# IoU dedup for truck sub-prompts
# ═══════════════════════════════════════════════════════════════════
def bbox_iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping(instances: List[Instance], iou_thr: float = 0.5) -> List[Instance]:
    """Merge overlapping instances (same class_id) by IoU, keep highest score."""
    if not instances:
        return instances

    instances.sort(key=lambda x: x.score, reverse=True)
    keep = []
    suppressed = set()

    for i, inst_i in enumerate(instances):
        if i in suppressed:
            continue
        keep.append(inst_i)
        for j in range(i + 1, len(instances)):
            if j in suppressed:
                continue
            if instances[j].class_id == inst_i.class_id:
                if bbox_iou(inst_i.bbox_xyxy, instances[j].bbox_xyxy) >= iou_thr:
                    suppressed.add(j)
    return keep


def cross_class_nms(instances: List[Instance], iou_thr: float = 0.7) -> List[Instance]:
    """
    Suppress cross-class overlaps (e.g. same vehicle detected as both sedan & suv).
    Keep the higher-confidence one.
    """
    if not instances:
        return instances

    instances.sort(key=lambda x: x.score, reverse=True)
    keep = []
    suppressed = set()

    for i, inst_i in enumerate(instances):
        if i in suppressed:
            continue
        keep.append(inst_i)
        for j in range(i + 1, len(instances)):
            if j in suppressed:
                continue
            if bbox_iou(inst_i.bbox_xyxy, instances[j].bbox_xyxy) >= iou_thr:
                suppressed.add(j)
    return keep


# ═══════════════════════════════════════════════════════════════════
# Overlay visualization
# ═══════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    0: (0, 200, 0),      # suv – green
    1: (200, 200, 0),    # sedan – yellow
    2: (200, 0, 0),      # bus – blue (BGR)
    3: (0, 0, 200),      # truck – red (BGR)
}


def draw_overlay(image: np.ndarray, instances: List[Instance]) -> np.ndarray:
    """Draw polygons + labels on image for QA."""
    overlay = image.copy()
    for inst in instances:
        color = CLASS_COLORS.get(inst.class_id, (128, 128, 128))
        pts = np.array(inst.polygon, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
        x, y = int(inst.bbox_xyxy[0]), max(int(inst.bbox_xyxy[1]) - 5, 15)
        label = f"{inst.class_name} {inst.score:.2f}"
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Teacher labeling with SAM3 PCS")
    parser.add_argument("--tiles-dir", required=True, help="Dir containing tiles + manifest")
    parser.add_argument("--output-dir", required=True, help="Dir for predictions + overlays")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--prompts", default="configs/prompts.yaml")
    parser.add_argument("--device", default=None, help="Override device (e.g. 0, cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts_cfg = load_config(args.prompts)
    device = args.device or str(cfg["training"]["device"])

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(args.tiles_dir, "tiles_manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = [json.loads(line) for line in f]

    print(f"Loaded {len(manifest)} tiles from manifest.")

    # Load teacher
    teacher = SAM3PCSTeacher()
    sam_path = cfg["paths"].get("sam_model", "sam3.pt")
    teacher.load(sam_path, device)

    if teacher.predictor is None:
        print("[ERROR] Failed to load SAM3SemanticPredictor. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Parse class prompts from config
    class_configs = []
    for class_key, class_info in prompts_cfg.get("class_prompts", {}).items():
        class_configs.append({
            "class_id": class_info["class_id"],
            "class_name": class_key,
            "prompts": class_info["prompts"],
            "merge_into": class_info.get("merge_into", class_info["class_id"]),
        })

    merge_iou = prompts_cfg.get("merge", {}).get("iou_threshold", 0.5)
    score_thr = prompts_cfg.get("merge", {}).get("score_threshold", 0.25)

    print(f"Classes: {[c['class_name'] for c in class_configs]}")
    print(f"Merge IoU: {merge_iou}, Score threshold: {score_thr}")

    # Process tiles
    total_instances = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, entry in enumerate(manifest):
        tile_path = entry["tile_path"]
        tile_name = Path(tile_path).stem

        if not os.path.exists(tile_path):
            print(f"  [WARN] Tile not found: {tile_path}", file=sys.stderr)
            continue

        # Run PCS for each class
        all_instances = []
        for cc in class_configs:
            instances = teacher.predict_class(
                image_path=tile_path,
                prompts=cc["prompts"],
                class_id=cc["merge_into"],
                class_name=cc["class_name"],
            )
            all_instances.extend(instances)

        # Filter by score
        all_instances = [inst for inst in all_instances if inst.score >= score_thr]

        # Merge overlapping same-class instances (truck sub-prompts)
        all_instances = merge_overlapping(all_instances, merge_iou)

        # Cross-class NMS (same vehicle detected as multiple classes)
        all_instances = cross_class_nms(all_instances, iou_thr=0.7)

        total_instances += len(all_instances)
        for inst in all_instances:
            class_counts[inst.class_id] = class_counts.get(inst.class_id, 0) + 1

        # Save predictions JSON
        pred_path = os.path.join(args.output_dir, f"{tile_name}_preds.json")
        with open(pred_path, "w") as f:
            json.dump(
                {
                    "tile_name": entry["tile_name"],
                    "source_image": entry["source_image"],
                    "tile_w": entry["tile_w"],
                    "tile_h": entry["tile_h"],
                    "instances": [inst.to_dict() for inst in all_instances],
                },
                f, indent=2,
            )

        # Save overlay
        img = cv2.imread(tile_path)
        if img is not None:
            overlay = draw_overlay(img, all_instances)
            overlay_path = os.path.join(overlay_dir, f"{tile_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay)

        if (i + 1) % 5 == 0 or i == len(manifest) - 1:
            print(f"  [{i+1}/{len(manifest)}] {tile_name}: {len(all_instances)} instances")

    print(f"\n{'='*60}")
    print(f"Teacher Labeling Summary (SAM3 PCS)")
    print(f"{'='*60}")
    print(f"  Total instances: {total_instances}")
    print(f"  Per class:")
    names = {0: "suv", 1: "sedan", 2: "bus", 3: "truck"}
    for cid in sorted(class_counts):
        print(f"    {cid}: {names.get(cid, '?'):8s} → {class_counts[cid]}")
    print(f"  Predictions: {args.output_dir}")
    print(f"  Overlays:    {overlay_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
