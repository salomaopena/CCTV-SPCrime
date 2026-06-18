#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dataset.py — Verifica a integridade do dataset YOLO do CCTV-SPCrime.

Checagens:
  1. data.yaml (lê nc e names)
  2. imagens corrompidas / ilegíveis / truncadas
  3. pareamento imagem <-> rótulo (rótulos órfãos; imagens sem rótulo = negativos)
  4. rótulos malformados (linha != 5 campos, valores não numéricos)
  5. class_id fora de [0, nc-1]
  6. caixas fora de [0,1] (e, opcional, caixa que extrapola a imagem)
  7. duplicatas entre splits (vazamento train/val/test) por nome e, opcional, por hash

Sai com código != 0 se houver ERROS (útil para CI). AVISOS não falham.

USO
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

# leitor de imagem opcional (PIL preferido; senão OpenCV; senão pula a checagem)
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
    """Parser mínimo: extrai nc e names (formato '  0: nome' ou '  - nome')."""
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
    """Retorna (w, h) ou None se corrompida/ilegível. None de _READER => pula."""
    if _READER == "pil":
        try:
            with Image.open(path) as im:
                im.verify()              # detecta truncamento/corrupção
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
    return (0, 0)  # sem leitor: não valida pixels, assume legível


def validate_label_file(path, nc):
    """Retorna (n_objs, per_class_counts, errors, warnings)."""
    errors, warnings = [], []
    per_class = defaultdict(int)
    n_objs = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return 0, per_class, [f"{path}: erro ao ler ({e})"], warnings

    for ln, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            errors.append(f"{path}:{ln}: esperados 5 campos, achei {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path}:{ln}: valores não numéricos")
            continue

        if nc is not None and not (0 <= cls < nc):
            errors.append(f"{path}:{ln}: class_id {cls} fora de [0,{nc-1}]")
        for nm, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{path}:{ln}: {nm}={v} fora de [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{path}:{ln}: largura/altura não-positiva (w={w}, h={h})")
        # caixa extrapolando a imagem (aviso)
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or \
           y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            warnings.append(f"{path}:{ln}: caixa extrapola os limites da imagem")
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
    ap = argparse.ArgumentParser(description="Valida a integridade do dataset YOLO.")
    ap.add_argument("--dataset-dir", default="dataset", help="raiz com data.yaml, images/, labels/")
    ap.add_argument("--check-hash", action="store_true",
                    help="também detecta duplicatas de CONTEÚDO entre splits (mais lento)")
    ap.add_argument("--report", help="salva o relatório em arquivo de texto")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        sys.exit(f"ERRO: pasta não encontrada: {root}")

    errors, warnings, info = [], [], []
    nc, names = parse_data_yaml(root / "data.yaml")
    if nc is None:
        warnings.append("data.yaml ausente ou sem 'nc'/'names'; validação de classe limitada.")
    else:
        info.append(f"data.yaml: nc={nc}, names={names}")
    if _READER is None:
        warnings.append("PIL/OpenCV não instalados; checagem de imagens corrompidas pulada "
                        "(pip install pillow).")

    name_to_splits = defaultdict(list)     # basename -> [splits] (vazamento por nome)
    hash_to_items = defaultdict(list)      # hash -> [(split, name)] (vazamento por conteúdo)
    totals = {s: {"images": 0, "labels": 0, "objects": 0, "negatives": 0} for s in SPLITS}
    per_class_total = defaultdict(int)

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            warnings.append(f"split '{split}' sem pasta de imagens; ignorado.")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        img_stems = {p.stem for p in images}
        label_files = list(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        lbl_stems = {p.stem for p in label_files}

        totals[split]["images"] = len(images)
        totals[split]["labels"] = len(label_files)

        # imagens corrompidas + vazamento
        for img in images:
            dims = read_image_dims(img)
            if dims is None:
                errors.append(f"imagem corrompida/ilegível: {img}")
            name_to_splits[img.name].append(split)
            if args.check_hash and dims is not None:
                hash_to_items[file_hash(img)].append((split, img.name))

        # rótulos órfãos (label sem imagem) = ERRO
        for st in lbl_stems - img_stems:
            errors.append(f"rótulo órfão (sem imagem): {split}/labels/{st}.txt")

        # imagens sem rótulo = negativos (AVISO/INFO)
        n_neg = len(img_stems - lbl_stems)
        totals[split]["negatives"] = n_neg

        # valida cada rótulo existente que tem imagem
        for lf in label_files:
            if lf.stem not in img_stems:
                continue
            n_obj, per_class, errs, warns = validate_label_file(lf, nc)
            errors.extend(errs)
            warnings.extend(warns)
            totals[split]["objects"] += n_obj
            for c, n in per_class.items():
                per_class_total[c] += n

    # vazamento por nome entre splits
    for name, splits in name_to_splits.items():
        uniq = sorted(set(splits))
        if len(uniq) > 1:
            errors.append(f"VAZAMENTO: '{name}' aparece em múltiplos splits: {uniq}")

    # vazamento por conteúdo (hash)
    if args.check_hash:
        for hsh, items in hash_to_items.items():
            splits = sorted({s for s, _ in items})
            if len(splits) > 1:
                names_list = ", ".join(f"{s}/{n}" for s, n in items)
                errors.append(f"VAZAMENTO (conteúdo idêntico) entre {splits}: {names_list}")

    # ---------------- relatório ----------------
    out = []
    out.append("=" * 64)
    out.append("RELATÓRIO DE VALIDAÇÃO — CCTV-SPCrime")
    out.append("=" * 64)
    for i in info:
        out.append(f"[info] {i}")
    out.append("")
    out.append(f"{'split':<8}{'imagens':>9}{'rótulos':>9}{'objetos':>9}{'negativos':>11}")
    for s in SPLITS:
        t = totals[s]
        out.append(f"{s:<8}{t['images']:>9}{t['labels']:>9}{t['objects']:>9}{t['negatives']:>11}")
    if per_class_total:
        out.append("")
        out.append("Objetos por class_id:")
        for c in sorted(per_class_total):
            label = names[c] if names and 0 <= c < len(names) else "?"
            out.append(f"  {c} ({label}): {per_class_total[c]}")
    out.append("")
    out.append(f"AVISOS: {len(warnings)}")
    for w in warnings[:50]:
        out.append(f"  [aviso] {w}")
    if len(warnings) > 50:
        out.append(f"  ... (+{len(warnings)-50} avisos)")
    out.append("")
    out.append(f"ERROS: {len(errors)}")
    for e in errors[:100]:
        out.append(f"  [ERRO] {e}")
    if len(errors) > 100:
        out.append(f"  ... (+{len(errors)-100} erros)")
    out.append("")
    out.append("RESULTADO: " + ("FALHOU ❌" if errors else "PASSOU ✅"))
    report = "\n".join(out)
    print(report)
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"\nRelatório salvo em: {args.report}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
