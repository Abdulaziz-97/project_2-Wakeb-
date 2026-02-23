#!/usr/bin/env python3
"""
tile_images.py – Slice aerial images into 1536×1536 tiles with overlap.

Outputs:
  - Tiled image files in output_dir/
  - tiles_manifest.jsonl  (one JSON object per line)

Each manifest line:
  {
    "tile_path": "...",  "source_image": "...",
    "x_offset": int,  "y_offset": int,
    "tile_w": int,  "tile_h": int,
    "split": "train"|"val"
  }
"""
import argparse, json, os, sys, hashlib, math
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def deterministic_split(source_name: str, train_ratio: float, seed: int) -> str:
    """Assign a source image to train or val deterministically."""
    h = int(hashlib.md5(f"{seed}_{source_name}".encode()).hexdigest(), 16)
    return "train" if (h % 1000) / 1000.0 < train_ratio else "val"


def tile_image(
    img_path: str,
    output_dir: str,
    tile_size: int = 1536,
    overlap: int = 384,
    train_ratio: float = 0.8,
    seed: int = 42,
    source_subdir: str = "",
) -> list:
    """Tile one image and return manifest entries."""
    stride = tile_size - overlap
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read {img_path}, skipping.", file=sys.stderr)
        return []

    h, w = img.shape[:2]
    src_name = Path(img_path).stem
    split = deterministic_split(src_name, train_ratio, seed)

    entries = []
    # Calculate number of tiles needed
    n_x = max(1, math.ceil((w - tile_size) / stride) + 1) if w > tile_size else 1
    n_y = max(1, math.ceil((h - tile_size) / stride) + 1) if h > tile_size else 1

    for yi in range(n_y):
        for xi in range(n_x):
            x0 = min(xi * stride, max(0, w - tile_size))
            y0 = min(yi * stride, max(0, h - tile_size))
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            # Handle edge case: image smaller than tile_size
            if x1 > w or y1 > h:
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                crop = img[y0 : min(y1, h), x0 : min(x1, w)]
                tile[: crop.shape[0], : crop.shape[1]] = crop
            else:
                tile = img[y0:y1, x0:x1]

            prefix = f"{source_subdir}__" if source_subdir else ""
            tile_name = f"{prefix}{src_name}__x{x0}_y{y0}.jpg"
            tile_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(tile_path, tile, [cv2.IMWRITE_JPEG_QUALITY, 95])

            entries.append(
                {
                    "tile_path": tile_path,
                    "tile_name": tile_name,
                    "source_image": os.path.basename(img_path),
                    "source_subdir": source_subdir,
                    "source_w": w,
                    "source_h": h,
                    "x_offset": x0,
                    "y_offset": y0,
                    "tile_w": tile_size,
                    "tile_h": tile_size,
                    "split": split,
                }
            )
    return entries


def main():
    parser = argparse.ArgumentParser(description="Tile aerial images.")
    parser.add_argument("--input-dir", required=True, help="Dir with raw aerial images")
    parser.add_argument("--output-dir", required=True, help="Dir for tiled outputs")
    parser.add_argument("--config", default="configs/project.yaml", help="Project config")
    parser.add_argument("--recursive", action="store_true",
                        help="Search input-dir recursively for images in subfolders")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_size = cfg["tiling"]["tile_size"]
    overlap = cfg["tiling"]["overlap"]
    train_ratio = cfg["split"]["train_ratio"]
    seed = cfg["split"]["seed"]

    os.makedirs(args.output_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if args.recursive:
        images = sorted(
            p
            for p in Path(args.input_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
            and p.parent != Path(args.input_dir)  # skip loose root-level files
        )
    else:
        images = sorted(
            p
            for p in Path(args.input_dir).iterdir()
            if p.suffix.lower() in exts
        )

    if not images:
        print(f"[ERROR] No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images. Tiling {tile_size}×{tile_size}, overlap {overlap} …")

    manifest = []
    for img_path in images:
        # Capture the immediate parent folder name as a class hint (e.g. "bus", "Truck")
        subdir = img_path.parent.name if img_path.parent.name != Path(args.input_dir).name else ""
        entries = tile_image(
            str(img_path), args.output_dir, tile_size, overlap, train_ratio, seed,
            source_subdir=subdir,
        )
        manifest.extend(entries)
        print(f"  {img_path.name}  →  {len(entries)} tiles  (split={entries[0]['split'] if entries else '?'})")

    manifest_path = os.path.join(args.output_dir, "tiles_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    # Summary
    train_count = sum(1 for e in manifest if e["split"] == "train")
    val_count = sum(1 for e in manifest if e["split"] == "val")
    print(f"\nDone. {len(manifest)} tiles total  (train={train_count}, val={val_count})")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
