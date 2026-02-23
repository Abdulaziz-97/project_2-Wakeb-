#!/usr/bin/env python3
"""
teacher_label_yolo.py – Substitute teacher using a pre-trained YOLO11s-seg model.

Used when sam3.pt is not available. Downloads yolo11s-seg.pt automatically via
Ultralytics and runs instance segmentation on each tile, mapping COCO classes
to the project's 4-class taxonomy.

COCO → Project class mapping:
  COCO  1 (bicycle)  → skip
  COCO  2 (car)      → 1 (sedan)   [includes SUV fallback if folder hint available]
  COCO  3 (motorcycle) → skip
  COCO  5 (bus)      → 2 (bus)
  COCO  7 (truck)    → 3 (truck)
  COCO  8 (boat)     → skip

Outputs per tile (same schema as teacher_label.py):
  - {tile_stem}_preds.json
  - overlays/{tile_stem}_overlay.png
"""
import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── COCO class-id → (project class_id, project class_name) ──────────
COCO_TO_PROJECT = {
    2: (1, "sedan"),   # car → sedan (covers most passenger vehicles)
    5: (2, "bus"),
    7: (3, "truck"),
}

# ── Folder-name hint: if a tile's source image came from one of these
#    subdirectories, we can override the class assignment.
FOLDER_HINTS = {
    "suv":   (0, "suv"),
    "sedan": (1, "sedan"),
    "bus":   (2, "bus"),
    "truck": (3, "truck"),
}

CLASS_COLORS = {
    0: (0, 200, 0),      # suv   – green
    1: (200, 200, 0),    # sedan – yellow
    2: (200, 0, 0),      # bus   – blue (BGR)
    3: (0, 0, 200),      # truck – red  (BGR)
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mask_to_polygon(mask: np.ndarray):
    """Convert binary mask → largest polygon (list of [x, y] pairs)."""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def draw_overlay(img: np.ndarray, instances: list) -> np.ndarray:
    """Draw filled polygons + labels on image for QA."""
    overlay = img.copy()
    for inst in instances:
        color = CLASS_COLORS.get(inst["class_id"], (128, 128, 128))
        pts = np.array(inst["polygon"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        x = int(inst["bbox_xyxy"][0])
        y = max(int(inst["bbox_xyxy"][1]) - 5, 15)
        label = f"{inst['class_name']} {inst['score']:.2f}"
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv2.addWeighted(overlay, 0.35, img, 0.65, 0)


def infer_folder_hint(tile_name: str, manifest_lookup: dict) -> str | None:
    """Return the source subdir name (e.g. 'bus') from the manifest, if available."""
    entry = manifest_lookup.get(tile_name)
    if entry:
        src = entry.get("source_image", "")
        # source_image is the basename; but we stored the full path hint separately
        return entry.get("source_subdir")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Substitute teacher: YOLO11s-seg pseudo-labeller"
    )
    parser.add_argument("--tiles-dir", required=True, help="Dir with tiles + manifest")
    parser.add_argument("--output-dir", required=True, help="Dir for predictions + overlays")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc.")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Confidence threshold (default 0.20 for aerial images)")
    parser.add_argument("--model", default="yolo11s-seg.pt",
                        help="YOLO model to use as teacher (auto-downloaded if not found)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or str(cfg["training"]["device"])
    score_threshold = args.conf

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(args.tiles_dir, "tiles_manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = []
    with open(manifest_path) as f:
        for line in f:
            manifest.append(json.loads(line))

    print(f"Loaded {len(manifest)} tiles from manifest.")

    # Load YOLO model (auto-downloads if not present)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    print(f"Loading teacher model: {args.model}  (downloads automatically if missing)")
    model = YOLO(args.model)
    print(f"Teacher model loaded.")

    tile_size = cfg["tiling"]["tile_size"]
    total_instances = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, entry in enumerate(manifest):
        tile_path = entry["tile_path"]
        tile_name = Path(tile_path).stem

        if not os.path.exists(tile_path):
            print(f"  [WARN] Tile not found: {tile_path}", file=sys.stderr)
            continue

        tile_w = entry.get("tile_w", tile_size)
        tile_h = entry.get("tile_h", tile_size)

        # Get folder hint for class override
        subdir_hint = entry.get("source_subdir")
        forced_class = FOLDER_HINTS.get(subdir_hint.lower()) if subdir_hint else None

        # Run YOLO inference
        results = model(
            tile_path, device=device, conf=score_threshold,
            imgsz=tile_size, verbose=False,
            classes=list(COCO_TO_PROJECT.keys()),  # only vehicle classes
        )

        instances = []
        if results and len(results) > 0:
            result = results[0]
            boxes  = result.boxes
            masks  = result.masks

            if boxes is not None:
                for j in range(len(boxes)):
                    coco_cls = int(boxes.cls[j])
                    score = float(boxes.conf[j])
                    bbox = boxes.xyxy[j].cpu().numpy().tolist()

                    # Map COCO → project class
                    if forced_class:
                        class_id, class_name = forced_class
                    elif coco_cls in COCO_TO_PROJECT:
                        class_id, class_name = COCO_TO_PROJECT[coco_cls]
                    else:
                        continue   # skip irrelevant COCO classes

                    # Extract polygon from mask
                    polygon = []
                    if masks is not None and j < len(masks):
                        mask_data = masks.data[j].cpu().numpy()
                        mask_u8 = (mask_data * 255).astype(np.uint8)
                        h, w = cv2.imread(tile_path).shape[:2]
                        if mask_u8.shape != (h, w):
                            mask_u8 = cv2.resize(mask_u8, (w, h))
                        poly = mask_to_polygon(mask_u8 / 255.0)
                        if poly and len(poly) >= 3:
                            polygon = poly

                    if len(polygon) < 3:
                        continue  # skip if no valid polygon extracted

                    instances.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox_xyxy": bbox,
                        "polygon": polygon,
                        "score": round(score, 4),
                        "prompt_used": f"yolo_teacher_coco_{coco_cls}",
                    })
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

        total_instances += len(instances)

        # tile_name already contains subdir prefix (e.g. bus__Pasted image__x0_y0)
        unique_tile_id = tile_name

        # Save predictions JSON
        pred_path = os.path.join(args.output_dir, f"{unique_tile_id}_preds.json")
        with open(pred_path, "w") as f:
            json.dump({
                "tile_name": entry["tile_name"],
                "source_image": entry["source_image"],
                "source_subdir": subdir_hint or "",
                "tile_w": tile_w,
                "tile_h": tile_h,
                "instances": instances,
            }, f, indent=2)

        # Save overlay
        img = cv2.imread(tile_path)
        if img is not None and instances:
            overlay = draw_overlay(img, instances)
            cv2.imwrite(os.path.join(overlay_dir, f"{unique_tile_id}_overlay.png"), overlay)

        if (i + 1) % 5 == 0 or i == len(manifest) - 1:
            print(f"  [{i+1}/{len(manifest)}] {unique_tile_id}: {len(instances)} instances")

    print(f"\n{'='*60}")
    print(f"YOLO Teacher Labeling Summary")
    print(f"{'='*60}")
    print(f"  Total instances: {total_instances}")
    print(f"  Per class:")
    names = {0: "suv", 1: "sedan", 2: "bus", 3: "truck"}
    for cid in sorted(class_counts):
        print(f"    {cid}: {names.get(cid, '?'):8s} → {class_counts[cid]}")
    print(f"  Predictions dir: {args.output_dir}")
    print(f"  Overlays dir:    {overlay_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
