#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py — Divide os frames coletados no formato YOLO para o CCTV-SPCrime.

Lê a pasta produzida por extract_frames.py (raw_frames/<classe>/*.jpg), divide
cada classe em treino/validação/teste (padrão 80/10/10, ESTRATIFICADO por classe
e reprodutível por seed), monta a estrutura YOLO (images/ + labels/), gera o
data.yaml e ATUALIZA o CSV de proveniência com a coluna 'split' e o novo caminho.

--------------------------------------------------------------------------------
USO
    python split_dataset.py \
        --input-dir raw_frames \
        --output-dir dataset \
        --provenance-csv provenance/frames_provenance.csv \
        --train 0.8 --val 0.1 --test 0.1 --seed 0 --mode copy

ESTRUTURA DE SAÍDA
    dataset/
      data.yaml
      images/{train,val,test}/...jpg
      labels/{train,val,test}/...txt   (rótulos copiados se já existirem)

Observações:
  - As classes vêm das subpastas de --input-dir. As classes de incidente entram
    no data.yaml (names); 'background'/'normal' são tratadas como NEGATIVOS
    (imagens sem objeto) e não viram classes.
  - Se ainda não houver rótulos (.txt), só as imagens são divididas; anote depois
    no CVAT e coloque os .txt em labels/<split>/ com o mesmo nome-base.
--------------------------------------------------------------------------------
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
CANONICAL_CLASSES = [
    "accident", "suspicious_behavior", "crime", "fire",
    "intrusion", "suspicious_object", "fall", "vandalism",
]
NEGATIVE_CLASSES = {"background", "normal"}


def log(msg):
    print(msg, flush=True)


def list_images(folder):
    return sorted([p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT])


def split_counts(n, r_train, r_val, r_test):
    """Divide n itens garantindo val/test >= 1 quando n >= 3."""
    n_test = int(round(n * r_test))
    n_val = int(round(n * r_val))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    # garante que treino não fique negativo
    while n_val + n_test >= n and (n_val + n_test) > 0:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def place(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)


def unique_name(name, used):
    """Evita colisão de nomes-base entre classes."""
    if name not in used:
        used.add(name)
        return name
    stem, ext = os.path.splitext(name)
    i = 1
    while f"{stem}_{i}{ext}" in used:
        i += 1
    new = f"{stem}_{i}{ext}"
    used.add(new)
    return new


def write_data_yaml(path, out_dir, class_names):
    lines = [
        f"path: {os.path.abspath(out_dir)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_provenance(prov_csv, basename_to_split, basename_to_newpath):
    """Acrescenta a coluna 'split' e atualiza frame_filename no CSV."""
    p = Path(prov_csv)
    if not p.exists():
        log(f"AVISO: proveniência '{prov_csv}' não encontrada; pulei a atualização.")
        return
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if "split" not in fields:
        fields = fields + ["split"]

    updated = 0
    for r in rows:
        base = os.path.basename(r.get("frame_filename", ""))
        if base in basename_to_split:
            r["split"] = basename_to_split[base]
            r["frame_filename"] = basename_to_newpath[base]
            updated += 1
        else:
            r.setdefault("split", "unused")

    backup = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, backup)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log(f"Proveniência atualizada ({updated} linhas com split). Backup: {backup.name}")


def main():
    ap = argparse.ArgumentParser(description="Divide raw_frames/ no formato YOLO (train/val/test).")
    ap.add_argument("--input-dir", default="raw_frames", help="pasta com subpastas por classe")
    ap.add_argument("--output-dir", default="dataset", help="pasta de saída YOLO")
    ap.add_argument("--provenance-csv", default="provenance/frames_provenance.csv",
                    help="CSV de proveniência por frame a atualizar (opcional)")
    ap.add_argument("--labels-dir", default=None,
                    help="pasta de rótulos .txt existentes (espelha as classes); opcional")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    ap.add_argument("--overwrite", action="store_true", help="sobrescreve a pasta de saída")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        sys.exit("ERRO: train+val+test deve somar 1.0")

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        sys.exit(f"ERRO: pasta de entrada não existe: {in_dir}")

    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.overwrite:
        sys.exit(f"ERRO: '{out_dir}' já existe. Use --overwrite para sobrescrever.")
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        sys.exit(f"ERRO: nenhuma subpasta de classe em {in_dir}")

    rng = random.Random(args.seed)
    used_names = set()
    basename_to_split, basename_to_newpath = {}, {}
    summary = {}            # classe -> {split: count}
    found_incident_classes = []

    for cdir in class_dirs:
        cls = cdir.name
        imgs = list_images(cdir)
        if not imgs:
            log(f"AVISO: classe '{cls}' sem imagens; ignorada.")
            continue
        if cls not in NEGATIVE_CLASSES and cls not in found_incident_classes:
            found_incident_classes.append(cls)

        rng.shuffle(imgs)
        n_tr, n_va, n_te = split_counts(len(imgs), args.train, args.val, args.test)
        assigned = ([("train", p) for p in imgs[:n_tr]]
                    + [("val", p) for p in imgs[n_tr:n_tr + n_va]]
                    + [("test", p) for p in imgs[n_tr + n_va:]])
        summary[cls] = {"train": n_tr, "val": n_va, "test": n_te}

        for split, img in assigned:
            name = unique_name(img.name, used_names)
            dst_img = out_dir / "images" / split / name
            place(img, dst_img, args.mode)

            # rótulo correspondente, se existir
            if args.labels_dir:
                lbl = Path(args.labels_dir) / cls / (img.stem + ".txt")
                if lbl.exists():
                    place(lbl, out_dir / "labels" / split / (Path(name).stem + ".txt"),
                          "copy" if args.mode == "symlink" else args.mode)

            basename_to_split[img.name] = split
            basename_to_newpath[img.name] = str(Path("images") / split / name)

    # data.yaml: classes de incidente na ordem canônica (interseção), depois extras
    ordered = [c for c in CANONICAL_CLASSES if c in found_incident_classes]
    extras = [c for c in found_incident_classes if c not in CANONICAL_CLASSES]
    class_names = ordered + sorted(extras)
    if extras:
        log(f"AVISO: classes fora da lista canônica adicionadas ao fim: {extras}")
    write_data_yaml(out_dir / "data.yaml", out_dir, class_names)

    # proveniência
    if args.provenance_csv:
        update_provenance(args.provenance_csv, basename_to_split, basename_to_newpath)

    # resumo
    log("\nResumo da divisão (imagens por classe):")
    log(f"{'classe':<22}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    tot = {"train": 0, "val": 0, "test": 0}
    for cls in sorted(summary):
        s = summary[cls]
        for k in tot:
            tot[k] += s[k]
        log(f"{cls:<22}{s['train']:>7}{s['val']:>6}{s['test']:>6}{sum(s.values()):>7}")
    log(f"{'TOTAL':<22}{tot['train']:>7}{tot['val']:>6}{tot['test']:>6}"
        f"{sum(tot.values()):>7}")
    log(f"\nData YAML: {out_dir/'data.yaml'}  |  classes: {class_names}")


if __name__ == "__main__":
    main()
