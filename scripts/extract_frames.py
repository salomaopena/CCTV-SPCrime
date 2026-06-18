#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_frames.py — Coleta de frames para o CCTV-SPCrime.

Lê um MANIFESTO (CSV) com a lista de vídeos e seus metadados de licença,
baixa (URL direta ou via yt-dlp) ou usa um arquivo local, extrai frames com
FPS fixo, redimensiona para 640x640 (letterbox), aplica filtragem de qualidade
(desfoque / muito escuro / muito claro / quase-duplicatas), opcionalmente
anonimiza faces (LGPD) e ESCREVE UMA LINHA DE PROVENIÊNCIA POR FRAME no CSV.

--------------------------------------------------------------------------------
DEPENDÊNCIAS
    pip install opencv-python
    (opcional) pip install yt-dlp          # para baixar de Pixabay/YouTube/etc.
    (opcional) pip install tqdm            # barra de progresso

USO RÁPIDO
    # 1) gerar um manifesto-modelo e preencher:
    python extract_frames.py --init-manifest manifest.csv

    # 2) processar:
    python extract_frames.py \
        --manifest manifest.csv \
        --output-dir raw_frames \
        --provenance-csv provenance/frames_provenance.csv \
        --fps 2 --size 640 --anonymize

FORMATO DO MANIFESTO (CSV, cabeçalho obrigatório):
    source_id,source_name,target_class,license,url_or_path,attribution,notes
  - source_id     : id curto da fonte (ex.: d-fire, caucafall, pixabay)
  - target_class  : uma das 8 classes (accident, suspicious_behavior, crime,
                    fire, intrusion, suspicious_object, fall, vandalism)
  - license       : licença da fonte (ex.: CC0, CC BY 4.0, MIT, "uso-pesquisa")
  - url_or_path   : caminho local do vídeo OU URL (direta .mp4/.webm ou yt-dlp)
  - attribution   : crédito a exibir, se a licença exigir (ou vazio)

IMPORTANTE (licença): só baixe/redistribua fontes cuja licença permita. Itens
"benchmark_only" do provenance_sources.csv NÃO devem ser redistribuídos.
--------------------------------------------------------------------------------
"""

import argparse
import csv
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("ERRO: instale as dependências -> pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:  # fallback sem barra de progresso
    def tqdm(x, **kwargs):
        return x

VIDEO_EXT = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")
VALID_CLASSES = {
    "accident", "suspicious_behavior", "crime", "fire",
    "intrusion", "suspicious_object", "fall", "vandalism",
    "background", "normal",  # úteis para negativos
}
PROV_FIELDS = [
    "frame_filename", "target_class", "source_id", "source_name", "license",
    "source_url", "attribution", "source_video", "frame_index", "src_fps",
    "extracted_fps", "blur_var", "brightness", "anonymized", "extracted_at", "notes",
]


# --------------------------------------------------------------------------- #
# Utilidades                                                                   #
# --------------------------------------------------------------------------- #
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def is_direct_video_url(s):
    return is_url(s) and s.lower().split("?")[0].endswith(VIDEO_EXT)


def download_video(url, dest_dir):
    """Baixa um vídeo. URL direta -> download simples; senão tenta yt-dlp."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_direct_video_url(url):
        import urllib.request
        fname = url.split("?")[0].split("/")[-1]
        out = dest_dir / fname
        log(f"  baixando (direto): {fname}")
        urllib.request.urlretrieve(url, out)
        return out

    # yt-dlp (Pixabay/YouTube/Vimeo/etc.)
    if shutil.which("yt-dlp") is None:
        log("  AVISO: yt-dlp não encontrado; pulei o download desta URL.")
        return None
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")
    log("  baixando (yt-dlp)...")
    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4/bestvideo", "-o", out_tmpl, url],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"  ERRO yt-dlp: {e.stderr.strip().splitlines()[-1] if e.stderr else e}")
        return None
    # pega o arquivo mais recente da pasta
    vids = sorted(dest_dir.glob("*"), key=os.path.getmtime, reverse=True)
    return vids[0] if vids else None


def letterbox(img, size):
    """Redimensiona preservando a proporção e preenche para size x size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # cinza padrão YOLO
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def stretch_resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def blur_var(gray):
    """Variância do Laplaciano: baixo = borrado."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray):
    return float(gray.mean())


def small_signature(gray):
    """Assinatura 32x32 para detecção de quase-duplicatas."""
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)


def is_duplicate(sig_a, sig_b, threshold):
    if sig_a is None or sig_b is None:
        return False
    return float(np.abs(sig_a - sig_b).mean()) < threshold


def load_face_cascade():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        log("  AVISO: cascade de faces não carregou; anonimização desativada.")
        return None
    return cascade


def anonymize_faces(img, cascade):
    """Desfoca faces detectadas (anonimização básica, LGPD). Para produção,
    recomenda-se um detector DNN mais robusto (faces de perfil/baixa resolução)."""
    if cascade is None:
        return img, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        k = max(15, (w // 2) | 1)  # kernel ímpar proporcional
        img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return img, len(faces)


def load_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, 1):
            r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if not r.get("url_or_path"):
                continue
            cls = r.get("target_class", "").strip()
            if cls and cls not in VALID_CLASSES:
                log(f"  AVISO (linha {i}): classe '{cls}' não está na lista padrão.")
            rows.append(r)
    return rows


def open_provenance(csv_path):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists() or p.stat().st_size == 0
    f = open(p, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=PROV_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


# --------------------------------------------------------------------------- #
# Processamento de um vídeo                                                    #
# --------------------------------------------------------------------------- #
def process_video(video_path, meta, args, writer, cascade):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"  ERRO: não consegui abrir {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0:
        src_fps = 30.0  # fallback
        log("  AVISO: FPS de origem desconhecido; assumindo 30.")
    step = max(1, int(round(src_fps / args.fps)))
    if args.fps > src_fps:
        log(f"  AVISO: fps alvo ({args.fps}) > fps origem ({src_fps:.1f}); usando todos os frames.")

    out_dir = Path(args.output_dir) / meta["target_class"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    kept, idx, last_sig = 0, 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- filtragem de qualidade ----
        bvar = blur_var(gray)
        bright = brightness(gray)
        if bvar < args.blur_threshold:
            idx += 1
            continue
        if bright < args.dark_threshold or bright > args.bright_threshold:
            idx += 1
            continue
        sig = small_signature(gray)
        if is_duplicate(sig, last_sig, args.dedup_threshold):
            idx += 1
            continue
        last_sig = sig

        # ---- redimensionamento ----
        out_img = (letterbox(frame, args.size) if args.resize_mode == "letterbox"
                   else stretch_resize(frame, args.size))

        # ---- anonimização (opcional) ----
        n_faces = 0
        if args.anonymize:
            out_img, n_faces = anonymize_faces(out_img, cascade)

        # ---- gravação ----
        fname = f"{meta['source_id']}__{stem}__f{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), out_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # ---- proveniência (uma linha por frame) ----
        writer.writerow({
            "frame_filename": str(Path(meta["target_class"]) / fname),
            "target_class": meta["target_class"],
            "source_id": meta.get("source_id", ""),
            "source_name": meta.get("source_name", ""),
            "license": meta.get("license", ""),
            "source_url": meta.get("url_or_path", ""),
            "attribution": meta.get("attribution", ""),
            "source_video": stem,
            "frame_index": idx,
            "src_fps": round(src_fps, 2),
            "extracted_fps": args.fps,
            "blur_var": round(bvar, 1),
            "brightness": round(bright, 1),
            "anonymized": ("yes" if args.anonymize else "no") + (f"({n_faces})" if n_faces else ""),
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": meta.get("notes", ""),
        })
        kept += 1
        idx += 1

    cap.release()
    return kept


# --------------------------------------------------------------------------- #
# Manifesto-modelo                                                             #
# --------------------------------------------------------------------------- #
def write_template(path):
    p = Path(path)
    if p.exists():
        sys.exit(f"ERRO: '{path}' já existe. Apague-o ou escolha outro nome.")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "source_name", "target_class", "license",
                    "url_or_path", "attribution", "notes"])
        w.writerow(["d-fire", "D-Fire", "fire", "CC0",
                    "/caminho/local/incendio_01.mp4", "", "exemplo local"])
        w.writerow(["pixabay", "Pixabay", "background", "Pixabay License",
                    "https://pixabay.com/videos/id-12345/", "",
                    "URL via yt-dlp; verificar pessoas reconheciveis"])
        w.writerow(["wikimedia", "Wikimedia Commons", "normal", "CC BY 4.0",
                    "https://upload.wikimedia.org/.../arquivo.webm",
                    "Autor (CC BY 4.0)", "download direto"])
    log(f"Manifesto-modelo criado em '{path}'. Edite e rode com --manifest.")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description="Extrai frames de vídeos para o CCTV-SPCrime.")
    ap.add_argument("--init-manifest", metavar="CSV",
                    help="cria um manifesto-modelo e sai")
    ap.add_argument("--manifest", help="CSV com a lista de vídeos e metadados")
    ap.add_argument("--output-dir", default="raw_frames",
                    help="pasta de saída dos frames (padrão: raw_frames)")
    ap.add_argument("--provenance-csv", default="provenance/frames_provenance.csv",
                    help="CSV de proveniência por frame")
    ap.add_argument("--download-dir", default="downloads",
                    help="pasta temporária para vídeos baixados")
    ap.add_argument("--fps", type=float, default=2.0, help="frames por segundo a extrair")
    ap.add_argument("--size", type=int, default=640, help="lado do quadro de saída (px)")
    ap.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox preserva proporção (padrão); stretch distorce p/ quadrado")
    ap.add_argument("--blur-threshold", type=float, default=100.0,
                    help="variância do Laplaciano mínima (abaixo = borrado, descarta)")
    ap.add_argument("--dark-threshold", type=float, default=25.0,
                    help="brilho médio mínimo (abaixo = muito escuro, descarta)")
    ap.add_argument("--bright-threshold", type=float, default=235.0,
                    help="brilho médio máximo (acima = estourado, descarta)")
    ap.add_argument("--dedup-threshold", type=float, default=6.0,
                    help="diferença média mínima vs. frame anterior (abaixo = duplicata)")
    ap.add_argument("--anonymize", action="store_true",
                    help="desfoca faces detectadas (LGPD)")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="não apaga os vídeos baixados ao final")
    return ap


def main():
    args = build_parser().parse_args()

    if args.init_manifest:
        write_template(args.init_manifest)
        return
    if not args.manifest:
        sys.exit("ERRO: informe --manifest CSV (ou use --init-manifest para criar um modelo).")

    rows = load_manifest(args.manifest)
    if not rows:
        sys.exit("ERRO: manifesto vazio ou sem coluna 'url_or_path'.")

    cascade = load_face_cascade() if args.anonymize else None
    prov_file, writer = open_provenance(args.provenance_csv)

    total_frames, total_videos, t0 = 0, 0, time.time()
    try:
        for meta in tqdm(rows, desc="vídeos"):
            src = meta["url_or_path"]
            log(f"Fonte '{meta.get('source_id','?')}' / classe '{meta.get('target_class','?')}'")

            # resolve o vídeo: local x download
            if is_url(src):
                video_path = download_video(src, args.download_dir)
                downloaded = True
            else:
                video_path = Path(src)
                downloaded = False
                if not video_path.exists():
                    log(f"  ERRO: arquivo não encontrado: {src}")
                    continue
            if not video_path:
                continue

            kept = process_video(video_path, meta, args, writer, cascade)
            prov_file.flush()
            total_frames += kept
            total_videos += 1
            log(f"  -> {kept} frames mantidos")

            if downloaded and not args.keep_downloads:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
    finally:
        prov_file.close()

    dt = time.time() - t0
    log(f"CONCLUÍDO: {total_frames} frames de {total_videos} vídeos em {dt:.1f}s.")
    log(f"Frames em: {args.output_dir}/  |  Proveniência: {args.provenance_csv}")


if __name__ == "__main__":
    main()
