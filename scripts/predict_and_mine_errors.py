#!/usr/bin/env python3
"""
predict_and_mine_errors.py – Run student inference and mine hard tiles.

Outputs:
  - Overlay images (predictions drawn on tiles)
  - to_correct.txt  (ranked list of hard tiles for human review)

Hard-tile heuristics:
  1. Low average confidence
  2. Many overlapping detections (high IoU pairs)
  3. Fragmented masks (small area ratio)
  4. Class confusion (bus ↔ truck co-occurrence at similar locations)
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml


CLASS_NAMES = {0: "suv", 1: "sedan", 2: "bus", 3: "truck"}
CLASS_COLORS = {
    0: (0, 200, 0),      # suv – green
    1: (200, 200, 0),    # sedan – yellow
    2: (200, 0, 0),      # bus – blue (BGR)
    3: (0, 0, 200),      # truck – red (BGR)
}


def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_difficulty_score(detections: list, img_area: float, cfg: dict) -> dict:
    """
    Compute difficulty metrics for a tile.
    Returns dict with individual scores and overall difficulty.
    """
    mcfg = cfg.get("mining", {})
    low_conf_thr = mcfg.get("low_conf_threshold", 0.35)
    overlap_iou_thr = mcfg.get("overlap_iou_threshold", 0.5)
    min_mask_area = mcfg.get("min_mask_area_ratio", 0.001)

    if not detections:
        return {"difficulty": 0.0, "reasons": [], "n_detections": 0}

    scores_list = [d["score"] for d in detections]
    avg_conf = np.mean(scores_list)
    low_conf_count = sum(1 for s in scores_list if s < low_conf_thr)

    # Check overlapping detections
    boxes = [d["bbox"] for d in detections]
    overlap_pairs = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if bbox_iou(boxes[i], boxes[j]) > overlap_iou_thr:
                overlap_pairs += 1

    # Check fragmented masks (small area)
    fragmented = 0
    for d in detections:
        mask_area = d.get("mask_area", 0)
        if mask_area > 0 and (mask_area / img_area) < min_mask_area:
            fragmented += 1

    # Check class confusion (bus ↔ truck at similar locations)
    bus_truck_confusion = 0
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            ci, cj = detections[i]["class_id"], detections[j]["class_id"]
            if {ci, cj} == {2, 3}:  # bus & truck
                if bbox_iou(boxes[i], boxes[j]) > 0.3:
                    bus_truck_confusion += 1

    # Weighted difficulty score
    difficulty = 0.0
    reasons = []

    if avg_conf < 0.5:
        difficulty += (0.5 - avg_conf) * 2
        reasons.append(f"low_avg_conf={avg_conf:.2f}")

    if low_conf_count > 0:
        difficulty += low_conf_count * 0.3
        reasons.append(f"low_conf_detections={low_conf_count}")

    if overlap_pairs > 0:
        difficulty += overlap_pairs * 0.5
        reasons.append(f"overlapping_pairs={overlap_pairs}")

    if fragmented > 0:
        difficulty += fragmented * 0.4
        reasons.append(f"fragmented_masks={fragmented}")

    if bus_truck_confusion > 0:
        difficulty += bus_truck_confusion * 1.0
        reasons.append(f"bus_truck_confusion={bus_truck_confusion}")

    return {
        "difficulty": round(difficulty, 3),
        "reasons": reasons,
        "n_detections": len(detections),
        "avg_confidence": round(avg_conf, 3),
    }


def draw_predictions(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw prediction overlays on image."""
    overlay = image.copy()

    for det in detections:
        class_id = det["class_id"]
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        score = det["score"]
        name = CLASS_NAMES.get(class_id, "?")

        # Draw mask polygon
        if "polygon" in det and det["polygon"]:
            pts = np.array(det["polygon"], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)

        # Draw bbox
        bbox = det["bbox"]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)

        # Label
        label = f"{name} {score:.2f}"
        y = max(int(bbox[1]) - 5, 15)
        cv2.putText(image, label, (int(bbox[0]), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


def main():
    parser = argparse.ArgumentParser(description="Run student inference & mine hard tiles")
    parser.add_argument("--tiles-dir", required=True, help="Dir with tile images")
    parser.add_argument("--model", required=True, help="Path to trained student .pt weights")
    parser.add_argument("--output-dir", required=True, help="Dir for overlays and reports")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or str(cfg["training"]["device"])
    tile_size = cfg["tiling"]["tile_size"]
    top_k = cfg.get("mining", {}).get("top_k_hard", 50)

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Collect tile images
    exts = {".jpg", ".jpeg", ".png"}
    tiles = sorted(
        p for p in Path(args.tiles_dir).iterdir()
        if p.suffix.lower() in exts
    )

    if not tiles:
        print(f"[ERROR] No tile images found in {args.tiles_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running inference on {len(tiles)} tiles …")

    tile_scores = []  # (tile_path, difficulty_info)

    for i, tile_path in enumerate(tiles):
        img = cv2.imread(str(tile_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        img_area = h * w

        # Run inference
        results = model(str(tile_path), device=device, conf=args.conf,
                        imgsz=tile_size, verbose=False)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            masks = result.masks

            if boxes is not None:
                for j in range(len(boxes)):
                    class_id = int(boxes.cls[j])
                    score = float(boxes.conf[j])
                    bbox = boxes.xyxy[j].cpu().numpy().tolist()

                    polygon = []
                    mask_area = 0
                    if masks is not None and j < len(masks):
                        # Extract polygon from mask
                        mask_data = masks.data[j].cpu().numpy()
                        mask_uint8 = (mask_data * 255).astype(np.uint8)
                        # Resize mask to image size if needed
                        if mask_uint8.shape != (h, w):
                            mask_uint8 = cv2.resize(mask_uint8, (w, h))
                        contours, _ = cv2.findContours(
                            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            polygon = largest.reshape(-1, 2).tolist()
                            mask_area = cv2.contourArea(largest)

                    detections.append({
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, "?"),
                        "score": score,
                        "bbox": bbox,
                        "polygon": polygon,
                        "mask_area": mask_area,
                    })

        # Compute difficulty
        diff_info = compute_difficulty_score(detections, img_area, cfg)
        diff_info["tile"] = str(tile_path)
        tile_scores.append((str(tile_path), diff_info))

        # Draw overlay
        overlay = draw_predictions(img, detections)
        overlay_path = os.path.join(overlay_dir, tile_path.stem + "_pred.png")
        cv2.imwrite(overlay_path, overlay)

        # Save predictions JSON
        pred_path = os.path.join(args.output_dir, tile_path.stem + "_student_preds.json")
        with open(pred_path, "w") as f:
            json.dump({
                "tile": tile_path.name,
                "detections": detections,
                "difficulty": diff_info,
            }, f, indent=2)

        if (i + 1) % 10 == 0 or i == len(tiles) - 1:
            print(f"  [{i+1}/{len(tiles)}] {tile_path.name}: "
                  f"{len(detections)} detections, difficulty={diff_info['difficulty']:.2f}")

    # Rank hard tiles
    tile_scores.sort(key=lambda x: x[1]["difficulty"], reverse=True)
    hard_tiles = tile_scores[:top_k]

    # Write to_correct.txt
    to_correct_path = os.path.join(args.output_dir, "to_correct.txt")
    with open(to_correct_path, "w") as f:
        f.write("# Hard tiles ranked by difficulty (highest first)\n")
        f.write("# Format: difficulty_score | tile_path | reasons\n\n")
        for tile_path, info in hard_tiles:
            reasons = ", ".join(info["reasons"]) if info["reasons"] else "none"
            f.write(f"{info['difficulty']:.3f} | {tile_path} | {reasons}\n")

    print(f"\n{'='*60}")
    print(f"Error Mining Summary")
    print(f"{'='*60}")
    print(f"  Total tiles processed:  {len(tiles)}")
    print(f"  Hard tiles flagged:     {len(hard_tiles)}")
    print(f"  Overlays:               {overlay_dir}")
    print(f"  Hard tile list:         {to_correct_path}")
    if hard_tiles:
        print(f"\n  Top 5 hardest tiles:")
        for tile_path, info in hard_tiles[:5]:
            print(f"    {info['difficulty']:.3f}  {Path(tile_path).name}  ({', '.join(info['reasons'])})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
