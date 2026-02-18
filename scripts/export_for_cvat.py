#!/usr/bin/env python3
"""
export_for_cvat.py – Convert teacher predictions to COCO JSON for CVAT import.

Reads per-tile *_preds.json files and produces a single annotations.json
that CVAT can import via "Upload Annotations → COCO 1.0".
"""
import argparse, json, os, sys, datetime
from pathlib import Path

import yaml


CLASS_MAP = {0: "suv", 1: "sedan", 2: "bus", 3: "truck"}


def polygon_to_segmentation(polygon: list) -> list:
    """Flatten [[x1,y1],[x2,y2],...] → [x1,y1,x2,y2,...] for COCO."""
    flat = []
    for pt in polygon:
        flat.extend([float(pt[0]), float(pt[1])])
    return flat


def polygon_area(polygon: list) -> float:
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Export teacher preds to COCO JSON for CVAT")
    parser.add_argument("--preds-dir", required=True, help="Dir with *_preds.json files")
    parser.add_argument("--tiles-dir", required=True, help="Dir with tile images")
    parser.add_argument("--output", required=True, help="Output COCO annotations.json path")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tile_size = cfg["tiling"]["tile_size"]

    # Collect all prediction files
    pred_files = sorted(Path(args.preds_dir).glob("*_preds.json"))
    if not pred_files:
        print(f"[ERROR] No *_preds.json files found in {args.preds_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction files.")

    # Build COCO structure
    categories = [{"id": cid, "name": cname} for cid, cname in CLASS_MAP.items()]

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for pred_file in pred_files:
        with open(pred_file) as f:
            pred = json.load(f)

        tile_name = pred["tile_name"]
        # Find the tile image
        tile_img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(args.tiles_dir, Path(tile_name).stem + ext)
            if not os.path.exists(candidate):
                candidate = os.path.join(args.tiles_dir, tile_name)
            if os.path.exists(candidate):
                tile_img_path = candidate
                break

        if tile_img_path is None:
            tile_img_path = os.path.join(args.tiles_dir, tile_name)

        images.append({
            "id": img_id,
            "file_name": os.path.basename(tile_img_path) if tile_img_path else tile_name,
            "width": pred.get("tile_w", tile_size),
            "height": pred.get("tile_h", tile_size),
        })

        for inst in pred.get("instances", []):
            polygon = inst.get("polygon", [])
            if len(polygon) < 3:
                continue

            bbox_xyxy = inst.get("bbox_xyxy", [0, 0, 0, 0])
            bbox_xywh = [
                bbox_xyxy[0],
                bbox_xyxy[1],
                bbox_xyxy[2] - bbox_xyxy[0],
                bbox_xyxy[3] - bbox_xyxy[1],
            ]

            seg = polygon_to_segmentation(polygon)
            area = polygon_area(polygon)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": inst["class_id"],
                "segmentation": [seg],
                "bbox": bbox_xywh,
                "area": area,
                "iscrowd": 0,
                "score": inst.get("score", 1.0),
            })
            ann_id += 1

        img_id += 1

    coco = {
        "info": {
            "description": "Teacher predictions for aerial vehicle segmentation",
            "date_created": datetime.datetime.now().isoformat(),
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Exported {len(annotations)} annotations across {len(images)} images → {args.output}")


if __name__ == "__main__":
    main()
