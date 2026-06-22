#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perclass_from_weights.py

DEPENDENCES
    pip install ultralytics
USE
    python perclass_from_weights.py \
        --weights runs_benchmark/yolo11n_train/weights/best.pt \
        --data dataset/data.yaml \
        --out-csv results_perclass_yolo11n.csv
--------------------------------------------------------------------------------
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ERROR: pip install ultralytics")

CONF_REPORT = 0.37  # mesmo limiar de reporte usado no benchmark


def main():
    ap = argparse.ArgumentParser(description="Trained weights")
    ap.add_argument("--weights", required=True, help="best.pt of the trained model (e.g., YOLO11n)")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=CONF_REPORT)
    ap.add_argument("--out-csv", default="results_perclass.csv")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"ERROR: weights not found: {args.weights}")

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, conf=args.conf, verbose=False)
    b = metrics.box
    names = metrics.names if hasattr(metrics, "names") else model.names

    maps = list(b.maps)                       # mAP50-95 per class (indix = class_id)
    try:
        p_c, r_c = list(b.p), list(b.r)       # precision / recall per class
    except Exception:
        p_c, r_c = [None] * len(maps), [None] * len(maps)

    rows = []
    for i, ap50_95 in enumerate(maps):
        if isinstance(names, dict):
            cname = names.get(i, str(i))
        elif isinstance(names, (list, tuple)) and i < len(names):
            cname = names[i]
        else:
            cname = str(i)
        p = p_c[i] if i < len(p_c) and p_c[i] is not None else None
        r = r_c[i] if i < len(r_c) and r_c[i] is not None else None
        f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None
        rows.append({
            "class_id": i,
            "class": cname,
            "precision": round(float(p), 2) if p is not None else "",
            "recall": round(float(r), 2) if r is not None else "",
            "F1": round(float(f1), 2) if f1 is not None else "",
            "mAP50_95": round(float(ap50_95), 2),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class", "precision", "recall", "F1", "mAP50_95"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved in: {args.out_csv}  ({len(rows)} classes)")
    print(f"{'id':>3}  {'class':<22}{'P':>7}{'R':>7}{'F1':>7}{'mAP50-95':>10}")
    for r in rows:
        print(f"{r['class_id']:>3}  {r['class']:<22}{str(r['precision']):>7}{str(r['recall']):>7}"
              f"{str(r['F1']):>7}{str(r['mAP50_95']):>10}")


if __name__ == "__main__":
    main()
