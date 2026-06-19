#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dataset.py — Check the integrity of the YOLO dataset from CCTV-SPCrime.

Checks:
  1. data.yaml (read nc and names)
  2. corrupted / unreadable / truncated images
  3. image <-> label pairing (orphan labels; images without labels = negatives)
  4. malformed labels (line != 5 fields, non-numeric values)
  5. class_id out of [0, nc-1]
  6. boxes outside of [0,1] (and, optionally, boxes that go beyond the image)
  7. duplicates between splits (train/val/test leakage) by name and, optionally, by hash

Exits with code != 0 if there are ERRORS (useful for CI). WARNINGS don't fail.

Use
    python validate_dataset.py --dataset-dir dataset
    python validate_dataset.py --dataset-dir dataset --check-hash --report relatorio.txt
--------------------------------------------------------------------------------
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")

# optional image reader (PIL preferred; if not, OpenCV; if not, skip the check)
_READER = None
try:
    from PIL import Image
    _READER = "pil"
except ImportError:
    try:
        import cv2
        _READER = "cv2"
    except ImportError:
        _READER = None


def parse_data_yaml(path):
    """Minimal parser: extracts nc and names (format ' 0: name' or ' - name')."""
    nc, names = None, []
    if not Path(path).exists():
        return nc, names
    in_names = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("nc:"):
            try:
                nc = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            in_names = False
        elif line.startswith("names:"):
            in_names = True
        elif in_names and line.strip():
            if not line.startswith(" "):
                in_names = False
                continue
            s = line.strip()
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif ":" in s:
                names.append(s.split(":", 1)[1].strip())
    if nc is None and names:
        nc = len(names)
    return nc, names


def read_image_dims(path):
    """Returns (w, h) or None if corrupted/unreadable. None from _READER => skip."""
    if _READER == "pil":
        try:
            with Image.open(path) as im:
                im.verify()              # detects truncation/corruption
            with Image.open(path) as im:
                return im.size           # (w, h)
        except Exception:
            return None
    elif _READER == "cv2":
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    return (0, 0)  # without reader: doesn't validate pixels, assumes readable


def validate_label_file(path, nc):
    """Returns (n_objs, per_class_counts, errors, warnings)."""
    errors, warnings = [], []
    per_class = defaultdict(int)
    n_objs = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return 0, per_class, [f"{path}: error reading ({e})"], warnings

    for ln, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            errors.append(f"{path}:{ln}: expected 5 fields, I found {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path}:{ln}: non-numeric values")
            continue

        if nc is not None and not (0 <= cls < nc):
            errors.append(f"{path}:{ln}: class_id {cls} out of [0,{nc-1}]")
        for nm, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{path}:{ln}: {nm}={v} out of [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{path}:{ln}: non-positive width/height (w={w}, h={h})")
        # box overflowing the image (warning)
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or \
           y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            warnings.append(f"{path}:{ln}: box goes beyond the image limits")
        per_class[cls] += 1
        n_objs += 1
    return n_objs, per_class, errors, warnings


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Validates the integrity of the YOLO dataset.")
    ap.add_argument("--dataset-dir", default="dataset", help="root with data.yaml, images/, labels/")
    ap.add_argument("--check-hash", action="store_true",
                    help="also detects CONTENT duplicates between splits (slower)")
    ap.add_argument("--report", help="save the report as a text file")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        sys.exit(f"ERROR: folder not found: {root}")

    errors, warnings, info = [], [], []
    nc, names = parse_data_yaml(root / "data.yaml")
    if nc is None:
        warnings.append("data.yaml absent or without 'nc'/'names'; limited class validation.")
    else:
        info.append(f"data.yaml: nc={nc}, names={names}")
    if _READER is None:
        warnings.append("PIL/OpenCV not installed; corrupted image check skipped "
                        "(pip install pillow).")

    name_to_splits = defaultdict(list)     # basename -> [splits] (name leak)
    hash_to_items = defaultdict(list)      # hash -> [(split, name)] (content leak)
    totals = {s: {"images": 0, "labels": 0, "objects": 0, "negatives": 0} for s in SPLITS}
    per_class_total = defaultdict(int)

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            warnings.append(f"split '{split}' no image folder; ignored.")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        img_stems = {p.stem for p in images}
        label_files = list(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        lbl_stems = {p.stem for p in label_files}

        totals[split]["images"] = len(images)
        totals[split]["labels"] = len(label_files)

        # corrupted images + leak
        for img in images:
            dims = read_image_dims(img)
            if dims is None:
                errors.append(f"corrupted/unreadable image: {img}")
            name_to_splits[img.name].append(split)
            if args.check_hash and dims is not None:
                hash_to_items[file_hash(img)].append((split, img.name))

        # orphan labels (label without image) = ERROR
        for st in lbl_stems - img_stems:
            errors.append(f"orphan label (no image): {split}/labels/{st}.txt")

        # images without labels = negatives (NOTICE/INFO)
        n_neg = len(img_stems - lbl_stems)
        totals[split]["negatives"] = n_neg

        # validates each existing label that has an image
        for lf in label_files:
            if lf.stem not in img_stems:
                continue
            n_obj, per_class, errs, warns = validate_label_file(lf, nc)
            errors.extend(errs)
            warnings.extend(warns)
            totals[split]["objects"] += n_obj
            for c, n in per_class.items():
                per_class_total[c] += n

    # leak by name between splits
    for name, splits in name_to_splits.items():
        uniq = sorted(set(splits))
        if len(uniq) > 1:
            errors.append(f"LEAK: '{name}' appears in multiple splits: {uniq}")

    # vazamento por conteúdo (hash)
    if args.check_hash:
        for hsh, items in hash_to_items.items():
            splits = sorted({s for s, _ in items})
            if len(splits) > 1:
                names_list = ", ".join(f"{s}/{n}" for s, n in items)
                errors.append(f"LEAK (identical content) between {splits}: {names_list}")

    # ---------------- report ----------------
    out = []
    out.append("=" * 64)
    out.append("VALIDATION REPORT — CCTV-SPCrime")
    out.append("=" * 64)
    for i in info:
        out.append(f"[info] {i}")
    out.append("")
    out.append(f"{'split':<8}{'images':>9}{'labels':>9}{'objects':>9}{'negatives':>11}")
    for s in SPLITS:
        t = totals[s]
        out.append(f"{s:<8}{t['images']:>9}{t['labels']:>9}{t['objects']:>9}{t['negatives']:>11}")
    if per_class_total:
        out.append("")
        out.append("Objects by class_id:")
        for c in sorted(per_class_total):
            label = names[c] if names and 0 <= c < len(names) else "?"
            out.append(f"  {c} ({label}): {per_class_total[c]}")
    out.append("")
    out.append(f"WARNING: {len(warnings)}")
    for w in warnings[:50]:
        out.append(f"  [WARNING] {w}")
    if len(warnings) > 50:
        out.append(f"  ... (+{len(warnings)-50} notic)")
    out.append("")
    out.append(f"ERRORS: {len(errors)}")
    for e in errors[:100]:
        out.append(f"  [ERROR] {e}")
    if len(errors) > 100:
        out.append(f"  ... (+{len(errors)-100} erros)")
    out.append("")
    out.append("RESULT: " + ("FAIL ❌" if errors else "PASSED ✅"))
    report = "\n".join(out)
    print(report)
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved in: {args.report}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
