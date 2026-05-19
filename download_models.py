"""
download_models.py — One-shot downloader for optional ML model assets.

Currently fetches:
  - GFPGANv1.4.onnx (~333 MB) for HD face restoration after face swap.

Usage:
    python download_models.py
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "backend", "dsp_models")

# Mirrors tried in order — first successful one wins.
GFPGAN_URLS = [
    "https://huggingface.co/Meeperomi/GFPGANv1.4-onnx/resolve/main/GFPGANv1.4.onnx",
    "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx",
]
GFPGAN_DEST = os.path.join(MODELS_DIR, "GFPGANv1.4.onnx")


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, 100 * downloaded / total_size)
        sys.stdout.write(
            f"\r  {pct:5.1f}%  {_human(downloaded)} / {_human(total_size)}"
        )
        sys.stdout.flush()


def download(urls, dest):
    if os.path.isfile(dest):
        print(f"[skip] Already present: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for url in urls:
        print(f"[download] {url}")
        print(f"        -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest, _progress)
            print()
            size = os.path.getsize(dest)
            print(f"[done] {_human(size)} written.")
            return
        except Exception as e:
            print(f"\n[warn] Mirror failed: {e}")
            if os.path.isfile(dest):
                os.remove(dest)
            continue
    print("[error] All mirrors exhausted.")
    sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print(" Deepfake Interaction System — Model Downloader")
    print("=" * 60)
    download(GFPGAN_URLS, GFPGAN_DEST)
    print("\nAll done. Restart the server to pick up the new model.")
