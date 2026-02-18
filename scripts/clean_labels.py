#!/usr/bin/env python3
"""
clean_labels.py – Filter noisy pseudo-labels from teacher predictions.

Removes:
  1. Tiny masks (bbox area < min_area pixels²)
  2. Low-confidence predictions (score < min_score)
  3. Huge masks that are likely background noise (bbox area > max_frac of tile)
  4. Masks with too few polygon points
  
Rewrites the _preds.json files in-place (with backup).
"""
import argparse, json, os, shutil
from pathlib import Path


def clean_instances(instances, tile_w, tile_h, min_score=0.30,
                    min_area_px=200, max_area_frac=0.25, min_poly_pts=4):
    """Filter a list of instances, returning only good ones."""
    tile_area = tile_w * tile_h
    kept = []
    reasons = {"low_score": 0, "tiny": 0, "huge": 0, "few_pts": 0}

    for inst in instances:
        score = inst.get("score", 0)
        if score < min_score:
            reasons["low_score"] += 1
            continue

        b = inst.get("bbox_xyxy", [0, 0, 0, 0])
        bbox_area = (b[2] - b[0]) * (b[3] - b[1])
        if bbox_area < min_area_px:
            reasons["tiny"] += 1
            continue
        if bbox_area / tile_area > max_area_frac:
            reasons["huge"] += 1
            continue

        poly = inst.get("polygon", [])
        if len(poly) < min_poly_pts:
            reasons["few_pts"] += 1
            continue

        kept.append(inst)

    return kept, reasons


def main():
    parser = argparse.ArgumentParser(description="Clean noisy pseudo-labels")
    parser.add_argument("--preds-dir", required=True, help="Dir with *_preds.json")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--min-area", type=int, default=200,
                        help="Min bbox area in pixels²")
    parser.add_argument("--max-area-frac", type=float, default=0.25,
                        help="Max bbox area as fraction of tile")
    parser.add_argument("--min-poly-pts", type=int, default=4)
    parser.add_argument("--backup", action="store_true",
                        help="Backup originals before overwriting")
    args = parser.parse_args()

    pred_files = sorted(Path(args.preds_dir).glob("*_preds.json"))
    if not pred_files:
        print("No prediction files found!")
        return

    total_before = 0
    total_after = 0
    all_reasons = {"low_score": 0, "tiny": 0, "huge": 0, "few_pts": 0}

    for pf in pred_files:
        with open(pf) as f:
            data = json.load(f)

        instances = data.get("instances", [])
        tw = data.get("tile_w", 1536)
        th = data.get("tile_h", 1536)

        total_before += len(instances)

        cleaned, reasons = clean_instances(
            instances, tw, th,
            min_score=args.min_score,
            min_area_px=args.min_area,
            max_area_frac=args.max_area_frac,
            min_poly_pts=args.min_poly_pts,
        )
        total_after += len(cleaned)
        for k, v in reasons.items():
            all_reasons[k] += v

        if args.backup:
            shutil.copy2(pf, str(pf) + ".bak")

        data["instances"] = cleaned
        with open(pf, "w") as f:
            json.dump(data, f, indent=2)

    removed = total_before - total_after
    print(f"{'='*60}")
    print(f"Label Cleanup Summary")
    print(f"{'='*60}")
    print(f"  Before:      {total_before} instances")
    print(f"  After:       {total_after} instances")
    print(f"  Removed:     {removed} ({removed/total_before*100:.1f}%)")
    print(f"  Reasons:")
    for k, v in all_reasons.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
