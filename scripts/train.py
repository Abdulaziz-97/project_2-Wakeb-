#!/usr/bin/env python3
"""
train.py – Train a YOLO segmentation student model.

Wraps Ultralytics YOLO training with project-specific defaults.
Equivalent CLI command is documented in README.md.
"""
import argparse, os, sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument("--config", default="configs/project.yaml", help="Project config")
    parser.add_argument("--data", default=None, help="Override data.yaml path")
    parser.add_argument("--model", default=None, help="Override pretrained model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--name", default="train", help="Experiment name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    data_yaml = args.data or os.path.join(cfg["paths"]["yolo_dataset_dir"], "data.yaml")
    model = args.model or tcfg["model"]
    epochs = args.epochs or tcfg["epochs"]
    batch = args.batch or tcfg["batch"]
    imgsz = args.imgsz or tcfg["imgsz"]
    device = args.device or str(tcfg["device"])
    workers = tcfg.get("workers", 8)
    patience = tcfg.get("patience", 20)

    # Validate data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}", file=sys.stderr)
        print("  Run build_yolo_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*60}")
    print(f"YOLO Segmentation Training")
    print(f"{'='*60}")
    print(f"  Model:    {model}")
    print(f"  Data:     {data_yaml}")
    print(f"  Epochs:   {epochs}")
    print(f"  Batch:    {batch}")
    print(f"  ImgSz:    {imgsz}")
    print(f"  Device:   {device}")
    print(f"  Workers:  {workers}")
    print(f"  Patience: {patience}")
    print(f"{'='*60}\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    yolo = YOLO(model)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        patience=patience,
        project=os.path.join(os.path.dirname(cfg["paths"]["yolo_dataset_dir"]), "runs", "segment"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # On-the-fly augmentations from config (no extra images created)
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg:
        print("Augmentations (on-the-fly):")
        for k, v in aug_cfg.items():
            train_kwargs[k] = v
            print(f"  {k}: {v}")
        print()

    if args.resume:
        train_kwargs["resume"] = True

    results = yolo.train(**train_kwargs)

    print(f"\nTraining complete.")
    print(f"Best weights: {yolo.trainer.best}")
    print(f"Last weights: {yolo.trainer.last}")


if __name__ == "__main__":
    main()
