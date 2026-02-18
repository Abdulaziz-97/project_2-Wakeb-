#!/usr/bin/env python3
"""
build_yolo_dataset.py – Build an Ultralytics YOLO segmentation dataset.

Creates:
  yolo_dataset/
    images/train/   images/val/
    labels/train/   labels/val/
    data.yaml

Label format (per line):  class_id x1 y1 x2 y2 ... xN yN  (normalized 0-1)
"""
import argparse, json, os, sys, shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {0: "suv", 1: "sedan", 2: "bus", 3: "truck"}


def polygon_to_yolo_seg(polygon: list, img_w: int, img_h: int) -> str | None:
    """
    Convert pixel polygon → YOLO seg format (normalized x y pairs).
    Returns None if invalid.
    """
    if len(polygon) < 3:
        return None

    coords = []
    for pt in polygon:
        x_norm = pt[0] / img_w
        y_norm = pt[1] / img_h
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        coords.extend([x_norm, y_norm])

    # Validate: even number of floats, ≥ 3 points (6 values)
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    return " ".join(f"{v:.6f}" for v in coords)


def validate_label_line(line: str) -> bool:
    """Validate a single YOLO seg label line."""
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
        return False
    try:
        class_id = int(parts[0])
        if class_id not in CLASS_NAMES:
            return False
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            return False
        if len(coords) < 6:
            return False
        for c in coords:
            if c < -0.01 or c > 1.01:  # small tolerance
                return False
    except ValueError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build YOLO segmentation dataset")
    parser.add_argument("--annots-dir", required=True,
                        help="Dir with canonical *_preds.json annotation files")
    parser.add_argument("--tiles-dir", required=True,
                        help="Dir with tile images + tiles_manifest.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output yolo_dataset/ dir")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tile_size = cfg["tiling"]["tile_size"]

    # Load manifest for split info
    manifest_path = os.path.join(args.tiles_dir, "tiles_manifest.jsonl")
    split_map = {}  # tile_name → split
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                stem = Path(entry["tile_name"]).stem
                split_map[stem] = entry["split"]
    else:
        print(f"[WARN] No manifest found at {manifest_path}. All images assigned to 'train'.")

    # Create directory structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)

    # Process annotation files
    annot_files = sorted(Path(args.annots_dir).glob("*_preds.json"))
    if not annot_files:
        print(f"[ERROR] No annotation files in {args.annots_dir}", file=sys.stderr)
        sys.exit(1)

    stats = {"train": 0, "val": 0, "total_instances": 0,
             "invalid_labels": 0, "skipped_unknown": 0}

    for annot_file in annot_files:
        with open(annot_file) as f:
            pred = json.load(f)

        tile_name = pred["tile_name"]
        tile_stem = Path(tile_name).stem
        tile_w = pred.get("tile_w", tile_size)
        tile_h = pred.get("tile_h", tile_size)

        # Determine split
        split = split_map.get(tile_stem, "train")

        # Find and copy tile image
        tile_src = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(args.tiles_dir, tile_stem + ext)
            if os.path.exists(candidate):
                tile_src = candidate
                break
        if tile_src is None:
            candidate = os.path.join(args.tiles_dir, tile_name)
            if os.path.exists(candidate):
                tile_src = candidate

        if tile_src is None:
            print(f"  [WARN] Tile image not found for {tile_name}, skipping.")
            continue

        # Copy image
        dst_img = os.path.join(args.output_dir, "images", split,
                               tile_stem + Path(tile_src).suffix)
        if not os.path.exists(dst_img):
            shutil.copy2(tile_src, dst_img)

        # Build label file
        label_lines = []
        for inst in pred.get("instances", []):
            class_id = inst.get("class_id", -1)
            if class_id not in CLASS_NAMES:
                stats["skipped_unknown"] += 1
                continue

            polygon = inst.get("polygon", [])
            seg_str = polygon_to_yolo_seg(polygon, tile_w, tile_h)
            if seg_str is None:
                stats["invalid_labels"] += 1
                continue

            line = f"{class_id} {seg_str}"
            if validate_label_line(line):
                label_lines.append(line)
                stats["total_instances"] += 1
            else:
                stats["invalid_labels"] += 1

        # Write label file (even if empty — YOLO needs the file)
        dst_label = os.path.join(args.output_dir, "labels", split, tile_stem + ".txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

        stats[split] += 1

    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(args.output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    data_yaml_path = os.path.join(args.output_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"YOLO Dataset Summary")
    print(f"{'='*60}")
    print(f"  Output:           {args.output_dir}")
    print(f"  Train images:     {stats['train']}")
    print(f"  Val images:       {stats['val']}")
    print(f"  Total instances:  {stats['total_instances']}")
    print(f"  Invalid labels:   {stats['invalid_labels']}")
    print(f"  Skipped unknown:  {stats['skipped_unknown']}")
    print(f"  data.yaml:        {data_yaml_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
