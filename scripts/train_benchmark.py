#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_benchmark.py — CCTV-SPCrime detection benchmark (solves Tables IV and V).

Trains and evaluates variants of the YOLO family (YOLOv8n, YOLO11n, YOLO26n) UNDER THE SAME
SPLIT and with FIXED HYPERPARAMETERS (reproducibility), and generates:
  - results_models.csv  -> comparison between models (Table IV of the paper)
  - results_perclass.csv-> results by class of the adopted model (Table V)

HYPERPARAMETERS (they document the reproducibility section of the paper):
  imgsz=640, epochs=100, batch=16, optimizer=SGD, lr0=0.01, lrf=0.01,
  momentum=0.937, weight_decay=0.0005, patience=20, seed=0.
  Reported inference confidence threshold: 0.37 (max F1 on validation).

DEPENDENCIES
    pip install ultralytics
Use
    python train_benchmark.py --data dataset/data.yaml \
        --models yolov8n.pt yolo11n.pt yolo26n.pt \
        --primary yolo26n.pt --epochs 100 --batch 16 --seed 0
--------------------------------------------------------------------------------
"""

import argparse
import csv
import sys
from pathlib import Path

HP = dict(imgsz=640, optimizer="SGD", lr0=0.01, lrf=0.01,
          momentum=0.937, weight_decay=0.0005, patience=20)
CONF_REPORT = 0.37  # reported confidence threshold (max F1 on validation)


def main():
    ap = argparse.ArgumentParser(description="YOLO Benchmark on CCTV-SPCrime (same split).")
    ap.add_argument("--data", default="dataset/data.yaml", help="dataset's data.yaml")
    ap.add_argument("--models", nargs="+",
                    default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"],
                    help="base weights to compare")
    ap.add_argument("--primary", default="yolo26n.pt",
                    help="adopted model (generates the table by class)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs_benchmark")
    ap.add_argument("--out-models", default="results_models.csv")
    ap.add_argument("--out-perclass", default="results_perclass.csv")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ERROR: install Ultralytics -> pip install ultralytics")

    if not Path(args.data).exists():
        sys.exit(f"ERROR: data.yaml not found: {args.data}")

    model_rows, perclass_rows = [], []

    for weights in args.models:
        tag = Path(weights).stem
        print(f"\n=== {tag}: train ({args.epochs} epoch) ===")
        model = YOLO(weights)
        model.train(
            data=args.data, epochs=args.epochs, batch=args.batch, seed=args.seed,
            project=args.project, name=f"{tag}_train", exist_ok=True, verbose=False,
            **HP,
        )
        print(f"=== {tag}: evaluation on the TEST set ===")
        metrics = model.val(
            data=args.data, split="test", conf=CONF_REPORT,
            project=args.project, name=f"{tag}_test", exist_ok=True, verbose=False,
        )
        b = metrics.box
        model_rows.append({
            "model": tag,
            "mAP50": round(float(b.map50), 4),
            "mAP50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4),
            "recall": round(float(b.mr), 4),
        })

        # table by class only for the adopted model
        if Path(weights).stem == Path(args.primary).stem:
            names = metrics.names if hasattr(metrics, "names") else model.names
            maps = list(b.maps)  # mAP50-95 per class
            # P/R per class when available
            try:
                p_c, r_c = list(b.p), list(b.r)
            except Exception:
                p_c, r_c = [None] * len(maps), [None] * len(maps)
            for i, ap50_95 in enumerate(maps):
                cname = names[i] if isinstance(names, (list, dict)) else str(i)
                if isinstance(names, dict):
                    cname = names.get(i, str(i))
                perclass_rows.append({
                    "class_id": i, "class": cname,
                    "precision": round(p_c[i], 4) if i < len(p_c) and p_c[i] is not None else "",
                    "recall": round(r_c[i], 4) if i < len(r_c) and r_c[i] is not None else "",
                    "mAP50_95": round(float(ap50_95), 4),
                })

    # save results
    with open(args.out_models, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mAP50", "mAP50_95", "precision", "recall"])
        w.writeheader(); w.writerows(model_rows)
    with open(args.out_perclass, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class", "precision", "recall", "mAP50_95"])
        w.writeheader(); w.writerows(perclass_rows)

    print("\n=== TABLE IV (comparison of models) ===")
    print(f"{'model':<12}{'mAP50':>8}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for r in model_rows:
        print(f"{r['model']:<12}{r['mAP50']:>8}{r['mAP50_95']:>10}{r['precision']:>8}{r['recall']:>8}")
    print(f"\nFiles: {args.out_models}, {args.out_perclass}")
    print(f"Hyperparameters (for reproducibility): {HP}, epochs={args.epochs}, "
          f"batch={args.batch}, seed={args.seed}, conf_report={CONF_REPORT}")


if __name__ == "__main__":
    main()
