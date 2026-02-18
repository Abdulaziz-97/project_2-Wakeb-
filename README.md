# Aerial Vehicle Instance Segmentation Pipeline

**Teacher → Verify/Correct → Retrain** loop for aerial vehicle segmentation using YOLO-seg.

## Classes (Stable IDs)

| ID | Class | Notes |
|----|-------|-------|
| 0  | suv   | Sport utility vehicles |
| 1  | sedan | Passenger cars / sedans |
| 2  | bus   | City bus, shuttle bus |
| 3  | truck | Pickup + box truck + semi/tractor-trailer |

> **Important:** Buses must **never** be labeled as truck. Truck includes pickup, box truck, and semi/tractor-trailer only.

---

## Repository Structure

```
project_2-Wakeb-/
├── configs/
│   ├── project.yaml          # Paths, tiling params, training config
│   └── prompts.yaml          # Teacher prompts per class, merge settings
├── scripts/
│   ├── tile_images.py        # Step 1: Tile raw images
│   ├── teacher_label.py      # Step 2: Generate pseudo-labels (SAM3)
│   ├── export_for_cvat.py    # Step 3: Export to COCO JSON for CVAT
│   ├── import_from_cvat.py   # Step 4: Import corrected CVAT annotations
│   ├── build_yolo_dataset.py # Step 5: Build YOLO seg dataset
│   ├── train.py              # Step 6: Train YOLO student
│   └── predict_and_mine_errors.py  # Step 7: Inference + error mining
├── data/
│   ├── tiles/                # Tiled images + manifest
│   ├── teacher_preds/        # Teacher predictions + overlays
│   ├── cvat_export/          # COCO JSON for CVAT import
│   └── canonical_annots/     # Normalized annotations
├── yolo_dataset/             # Final YOLO training dataset
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── runs/                     # Training runs
├── sam3.pt                   # SAM3 model weights
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies (if not already)
pip install -r requirements.txt
```

---

## Pipeline Commands

### Step 1 — Tile Images

```bash
python scripts/tile_images.py \
  --input-dir /home/wakeb/Downloads/my_20_images \
  --output-dir data/tiles \
  --config configs/project.yaml
```

Produces 1536×1536 tiles with 384px overlap (stride 1152) and `tiles_manifest.jsonl`.

### Step 2 — Teacher Labeling (SAM3)

```bash
python scripts/teacher_label.py \
  --tiles-dir data/tiles \
  --output-dir data/teacher_preds \
  --config configs/project.yaml \
  --prompts configs/prompts.yaml \
  --device 0
```

Generates per-tile prediction JSONs and QA overlay PNGs.

### Step 3 — Export for CVAT

```bash
python scripts/export_for_cvat.py \
  --preds-dir data/teacher_preds \
  --tiles-dir data/tiles \
  --output data/cvat_export/annotations.json \
  --config configs/project.yaml
```

Creates a COCO-format `annotations.json` that CVAT can import.

### Step 4 — Import from CVAT (after human correction)

```bash
# From COCO JSON export:
python scripts/import_from_cvat.py \
  --input /path/to/cvat_coco_export/annotations.json \
  --output-dir data/canonical_annots \
  --format coco

# From CVAT XML export:
python scripts/import_from_cvat.py \
  --input /path/to/cvat_export.xml \
  --output-dir data/canonical_annots \
  --format cvat_xml
```

### Step 5 — Build YOLO Dataset

```bash
python scripts/build_yolo_dataset.py \
  --annots-dir data/canonical_annots \
  --tiles-dir data/tiles \
  --output-dir yolo_dataset \
  --config configs/project.yaml
```

Creates `yolo_dataset/` with `images/`, `labels/`, and `data.yaml`.

### Step 6 — Train YOLO Student

**Python wrapper:**
```bash
python scripts/train.py \
  --config configs/project.yaml \
  --epochs 100 \
  --batch 4 \
  --device 0
```

**Direct Ultralytics CLI:**
```bash
yolo segment train \
  data=yolo_dataset/data.yaml \
  model=yolo11n-seg.pt \
  epochs=100 \
  imgsz=1536 \
  batch=4 \
  device=0 \
  patience=20
```

### Step 7 — Predict & Mine Errors

```bash
python scripts/predict_and_mine_errors.py \
  --tiles-dir data/tiles \
  --model runs/segment/train/weights/best.pt \
  --output-dir data/error_mining \
  --config configs/project.yaml \
  --device 0
```

Outputs:
- `data/error_mining/overlays/` — prediction overlays for QA
- `data/error_mining/to_correct.txt` — ranked hard tiles

---

## Iteration Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    ITERATION LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Teacher label (SAM3)         → pseudo-labels            │
│  2. Export to CVAT               → COCO JSON                │
│  3. Human verify/correct in CVAT → corrected annotations    │
│  4. Import from CVAT             → canonical annotations    │
│  5. Build YOLO dataset           → images/ + labels/        │
│  6. Train YOLO student           → best.pt                  │
│  7. Predict & mine errors        → to_correct.txt           │
│  8. Go to step 2 with hard tiles → re-correct in CVAT       │
│                                                             │
│  Repeat until student performance is satisfactory.          │
│                                                             │
│  PRODUCTION: Use YOLO student only (no teacher needed).     │
└─────────────────────────────────────────────────────────────┘
```

### Active Learning Strategy

1. **Round 1**: Teacher labels all tiles → human corrects in CVAT → train student.
2. **Round 2+**: Student predicts on tiles → mine hard tiles (`to_correct.txt`) → human corrects only the hard tiles → merge with existing annotations → retrain.
3. **Convergence**: Stop when `to_correct.txt` has few/no hard tiles, or mAP plateaus.

---

## Tiling Parameters

| Parameter | Value |
|-----------|-------|
| Tile size | 1536×1536 |
| Overlap   | 384 px |
| Stride    | 1152 px |
| Training imgsz | 1536 |

---

## Notes

- **Train/val split** is by **source image** (not tile) to prevent data leakage.
- **Truck definition**: includes pickup, box truck, semi/tractor-trailer. Buses are class 2 only.
- **Production inference**: Use the trained YOLO student only. Teacher (SAM3) is for labeling/QA only.
- **SAM3 model**: Place `sam3.pt` in the project root. The teacher script will auto-load it.
