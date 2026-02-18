#!/usr/bin/env python3
"""
import_from_cvat.py – Parse CVAT export and normalize into canonical annotations.

Supported CVAT export formats:
  1. COCO 1.0 JSON  (default)
  2. CVAT XML 1.1  (images + polygons)

Outputs canonical annotation JSONs (same schema as teacher_label.py output)
into the canonical_annots_dir.
"""
import argparse, json, os, sys, re
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import yaml


# Stable class mapping
NAME_TO_ID = {"suv": 0, "sedan": 1, "bus": 2, "truck": 3}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}

# Aliases that should map to truck
TRUCK_ALIASES = {"pickup", "pickup truck", "box truck", "semi truck",
                 "semi", "tractor-trailer", "tractor trailer", "delivery truck"}


def normalize_class(name: str) -> tuple:
    """Return (class_id, class_name) or (None, None) if unknown."""
    name_lower = name.strip().lower()
    if name_lower in NAME_TO_ID:
        return NAME_TO_ID[name_lower], name_lower
    if name_lower in TRUCK_ALIASES:
        return 3, "truck"
    return None, None


# ═══════════════════════════════════════════════════════════════════
# COCO JSON parser
# ═══════════════════════════════════════════════════════════════════
def parse_coco_json(coco_path: str) -> dict:
    """Parse COCO JSON into {image_filename: [instances]}."""
    with open(coco_path) as f:
        coco = json.load(f)

    # Category map
    cat_map = {}
    for cat in coco.get("categories", []):
        cat_map[cat["id"]] = cat["name"]

    # Image map
    img_map = {}
    for img in coco.get("images", []):
        img_map[img["id"]] = {
            "file_name": img["file_name"],
            "width": img.get("width", 1536),
            "height": img.get("height", 1536),
        }

    # Group annotations by image
    result = defaultdict(lambda: {"instances": [], "width": 1536, "height": 1536})
    for ann in coco.get("annotations", []):
        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue

        cat_name = cat_map.get(ann["category_id"], "unknown")
        class_id, class_name = normalize_class(cat_name)
        if class_id is None:
            print(f"  [WARN] Unknown class '{cat_name}', skipping annotation {ann['id']}")
            continue

        # Parse segmentation → polygon
        polygon = []
        seg = ann.get("segmentation", [])
        if seg and isinstance(seg[0], list):
            flat = seg[0]
            polygon = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]

        # Parse bbox
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        else:
            bbox_xyxy = [0, 0, 0, 0]

        fname = img_info["file_name"]
        result[fname]["width"] = img_info["width"]
        result[fname]["height"] = img_info["height"]
        result[fname]["instances"].append({
            "class_id": class_id,
            "class_name": class_name,
            "bbox_xyxy": bbox_xyxy,
            "polygon": polygon,
            "score": ann.get("score", 1.0),
            "prompt_used": "cvat_import",
        })

    return dict(result)


# ═══════════════════════════════════════════════════════════════════
# CVAT XML parser
# ═══════════════════════════════════════════════════════════════════
def parse_cvat_xml(xml_path: str) -> dict:
    """Parse CVAT XML 1.1 export into {image_filename: [instances]}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = defaultdict(lambda: {"instances": [], "width": 1536, "height": 1536})

    for image_el in root.findall(".//image"):
        fname = image_el.get("name", "")
        width = int(image_el.get("width", 1536))
        height = int(image_el.get("height", 1536))
        result[fname]["width"] = width
        result[fname]["height"] = height

        for poly_el in image_el.findall("polygon"):
            label = poly_el.get("label", "unknown")
            class_id, class_name = normalize_class(label)
            if class_id is None:
                print(f"  [WARN] Unknown class '{label}' in {fname}, skipping.")
                continue

            points_str = poly_el.get("points", "")
            # Format: "x1,y1;x2,y2;..."
            polygon = []
            for pt in points_str.split(";"):
                parts = pt.strip().split(",")
                if len(parts) == 2:
                    polygon.append([float(parts[0]), float(parts[1])])

            if len(polygon) < 3:
                continue

            pts_np = [[p[0], p[1]] for p in polygon]
            xs = [p[0] for p in pts_np]
            ys = [p[1] for p in pts_np]
            bbox_xyxy = [min(xs), min(ys), max(xs), max(ys)]

            result[fname]["instances"].append({
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xyxy": bbox_xyxy,
                "polygon": polygon,
                "score": 1.0,
                "prompt_used": "cvat_import",
            })

    return dict(result)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Import CVAT export into canonical annotations")
    parser.add_argument("--input", required=True,
                        help="Path to COCO JSON or CVAT XML export")
    parser.add_argument("--output-dir", required=True,
                        help="Dir for canonical annotation JSONs")
    parser.add_argument("--format", choices=["coco", "cvat_xml"], default="coco",
                        help="Export format (default: coco)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing {args.format} export: {args.input}")

    if args.format == "coco":
        annotations = parse_coco_json(args.input)
    elif args.format == "cvat_xml":
        annotations = parse_cvat_xml(args.input)
    else:
        print(f"[ERROR] Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)

    # Write canonical JSONs
    total_instances = 0
    for fname, data in annotations.items():
        stem = Path(fname).stem
        out_path = os.path.join(args.output_dir, f"{stem}_preds.json")
        out_data = {
            "tile_name": fname,
            "source_image": fname,
            "tile_w": data["width"],
            "tile_h": data["height"],
            "instances": data["instances"],
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        total_instances += len(data["instances"])

    print(f"Imported {total_instances} annotations across {len(annotations)} images → {args.output_dir}")


if __name__ == "__main__":
    main()
