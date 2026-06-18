"""
extract_frames.py — Frame collection for CCTV-SPCrime.

Reads a MANIFESTO (CSV) with the list of videos and their license metadata, downloads (direct URL or via yt-dlp) or uses a local file, extracts frames at a fixed FPS, resizes to 640x640 (letterbox), applies quality filtering (blur / too dark / too bright / near-duplicates), optionally anonymizes faces (LGPD) and WRITES A SOURCE LINE PER FRAME in the CSV.

--------------------------------------------------------------------------------
DEPENDENCIES
    pip install opencv-python
    (optional) pip install yt-dlp          # to download from Pixabay/YouTube/etc.
    (optional) pip install tqdm            # progress bar

QUICK USE
    # 1) generate a sample manifesto and fill it out:
    python extract_frames.py --init-manifest manifest.csv

    # 2) process:
    python extract_frames.py \
        --manifest manifest.csv \
        --output-dir raw_frames \
        --provenance-csv provenance/frames_provenance.csv \
        --fps 2 --size 640 --anonymize

MANIFESTO FORMAT (CSV, header required):
    source_id,source_name,target_class,license,url_or_path,attribution,notes
  - source_id     : short font ID (e.g., d-fire, caucafall, pixabay)
  - target_class  : one of the 8 classes (accident, suspicious_behavior, crime, fire, intrusion,            suspicious_object, fall, vandalism)
  - license       : source license CC BY 4.0
  - url_or_path   : local path of the resource OR URL (direct .mp4/.webm or yt-dlp)
  - attribution   : CC BY 4.0
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
    sys.exit("ERROR: install the dependencies -> pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:  # fallback without progress bar
    def tqdm(x, **kwargs):
        return x

VIDEO_EXT = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")
VALID_CLASSES = {
    "accident", "suspicious_behavior", "crime", "fire",
    "intrusion", "suspicious_object", "fall", "vandalism",
    "background", "normal",  # useful for negatives
}
PROV_FIELDS = [
    "frame_filename", "target_class", "source_id", "source_name", "license",
    "source_url", "attribution", "source_video", "frame_index", "src_fps",
    "extracted_fps", "blur_var", "brightness", "anonymized", "extracted_at", "notes",
]


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def is_direct_video_url(s):
    return is_url(s) and s.lower().split("?")[0].endswith(VIDEO_EXT)


def download_video(url, dest_dir):
    """Download a video. Direct URL -> simple download; otherwise try yt-dlp."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_direct_video_url(url):
        import urllib.request
        fname = url.split("?")[0].split("/")[-1]
        out = dest_dir / fname
        log(f"  downloading (direct): {fname}")
        urllib.request.urlretrieve(url, out)
        return out

    # yt-dlp (Pixabay/YouTube/Vimeo/etc.)
    if shutil.which("yt-dlp") is None:
        log("  WARNING: yt-dlp not found; I skipped downloading this URL.")
        return None
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")
    log("  downloading (yt-dlp)...")
    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4/bestvideo", "-o", out_tmpl, url],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"  ERROR yt-dlp: {e.stderr.strip().splitlines()[-1] if e.stderr else e}")
        return None
    # grab the latest file from the folder
    vids = sorted(dest_dir.glob("*"), key=os.path.getmtime, reverse=True)
    return vids[0] if vids else None


def letterbox(img, size):
    """Resizes while keeping the proportion and fills to size x size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # standard YOLO gray
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def stretch_resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def blur_var(gray):
    """Laplacian variance: low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray):
    return float(gray.mean())


def small_signature(gray):
    """32x32 signature for near-duplicate detection."""
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)


def is_duplicate(sig_a, sig_b, threshold):
    if sig_a is None or sig_b is None:
        return False
    return float(np.abs(sig_a - sig_b).mean()) < threshold


def load_face_cascade():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        log("  WARNING: face cascade didn't load; anonymization disabled.")
        return None
    return cascade


def anonymize_faces(img, cascade):
    """Blur detected faces (basic anonymization, LGPD). For production, it's recommended to use a more robust DNN detector (side/low-resolution faces)."""
    if cascade is None:
        return img, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        k = max(15, (w // 2) | 1)  # odd proportional kernel
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
                log(f"  WARNING (line {i}): class '{cls}' it's not on the default list.")
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
# Processing a video                                                    #
# --------------------------------------------------------------------------- #
def process_video(video_path, meta, args, writer, cascade):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"  ERROR: I couldn’t open it {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0:
        src_fps = 30.0  # fallback
        log("  NOTICE: Unknown source FPS; assuming 30.")
    step = max(1, int(round(src_fps / args.fps)))
    if args.fps > src_fps:
        log(f"  WARNING: target fps ({args.fps}) > origin fps ({src_fps:.1f}); using all the frames.")

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

        # ---- resizing ----
        out_img = (letterbox(frame, args.size) if args.resize_mode == "letterbox"
                   else stretch_resize(frame, args.size))

        # ---- aanonymization (optional) ----
        n_faces = 0
        if args.anonymize:
            out_img, n_faces = anonymize_faces(out_img, cascade)

        # ---- recording ----
        fname = f"{meta['source_id']}__{stem}__f{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), out_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # ---- provenance (one line per frame) ----
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
# Sample manifesto                                                             #
# --------------------------------------------------------------------------- #
def write_template(path):
    p = Path(path)
    if p.exists():
        sys.exit(f"ERROR: '{path}' It already exists. Delete it or choose another name.")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "source_name", "target_class", "license",
                    "url_or_path", "attribution", "notes"])
        w.writerow(["d-fire", "D-Fire", "fire", "CC0",
                    "/caminho/local/incendio_01.mp4", "", "local example"])
        w.writerow(["pixabay", "Pixabay", "background", "Pixabay License",
                    "https://pixabay.com/videos/id-12345/", "",
                    "URL via yt-dlp: check recognizable people"])
        w.writerow(["wikimedia", "Wikimedia Commons", "normal", "CC BY 4.0",
                    "https://upload.wikimedia.org/.../arquivo.webm",
                    "Author (CC BY 4.0)", "direct download"])
    log(f"Template manifesto created in '{path}'. Edit and run with --manifest.")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description="Extracts frames from videos for CCTV-SPCrime.")
    ap.add_argument("--init-manifest", metavar="CSV",
                    help="create a sample manifesto and leave")
    ap.add_argument("--manifest", help="CSV with the list of videos and metadata")
    ap.add_argument("--output-dir", default="raw_frames",
                    help="output folder for the frames (default: raw_frames)")
    ap.add_argument("--provenance-csv", default="provenance/frames_provenance.csv",
                    help="Provenance CSV by frame")
    ap.add_argument("--download-dir", default="downloads",
                    help="temporary folder for downloaded videos")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second to extract")
    ap.add_argument("--size", type=int, default=640, help="side of the output frame (px)")
    ap.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox preserves proportion (default); stretch distorts to square")
    ap.add_argument("--blur-threshold", type=float, default=100.0,
                    help="minimum Laplacian variance (lower = blurry, discard)")
    ap.add_argument("--dark-threshold", type=float, default=25.0,
                    help="minimum average brightness (below = very dark, discard)")
    ap.add_argument("--bright-threshold", type=float, default=235.0,
                    help="maximum average brightness (above = blown out, discard)")
    ap.add_argument("--dedup-threshold", type=float, default=6.0,
                    help="minimum average difference vs. previous frame (below = duplicate)")
    ap.add_argument("--anonymize", action="store_true",
                    help="blur detected faces (LGPD)")
    ap.add_argument("--keep-downloads", action="store_true",
                    help="doesn't delete the downloaded videos at the end")
    return ap


def main():
    args = build_parser().parse_args()

    if args.init_manifest:
        write_template(args.init_manifest)
        return
    if not args.manifest:
        sys.exit("ERROR: report --manifest CSV (or use --init-manifest to create a template).")

    rows = load_manifest(args.manifest)
    if not rows:
        sys.exit("ERROR: empty or columnless manifesto 'url_or_path'.")

    cascade = load_face_cascade() if args.anonymize else None
    prov_file, writer = open_provenance(args.provenance_csv)

    total_frames, total_videos, t0 = 0, 0, time.time()
    try:
        for meta in tqdm(rows, desc="videos"):
            src = meta["url_or_path"]
            log(f"Font '{meta.get('source_id','?')}' / class '{meta.get('target_class','?')}'")

            # resolve o vídeo: local x download
            if is_url(src):
                video_path = download_video(src, args.download_dir)
                downloaded = True
            else:
                video_path = Path(src)
                downloaded = False
                if not video_path.exists():
                    log(f"  ERROR: file not found: {src}")
                    continue
            if not video_path:
                continue

            kept = process_video(video_path, meta, args, writer, cascade)
            prov_file.flush()
            total_frames += kept
            total_videos += 1
            log(f"  -> {kept} kept frames")

            if downloaded and not args.keep_downloads:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
    finally:
        prov_file.close()

    dt = time.time() - t0
    log(f"COMPLETED: {total_frames} frames of{total_videos} videos in {dt:.1f}s.")
    log(f"Frames in: {args.output_dir}/  |  Provenance: {args.provenance_csv}")


if __name__ == "__main__":
    main()
