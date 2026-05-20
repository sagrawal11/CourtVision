#!/usr/bin/env python3
"""
Download multiple PlaySight share links to local MP4 files.

Run from the repo root (with tennis_env activated):

    python annotation_collaboration/scripts/bulk_download_playsight.py \\
        annotation_collaboration/scripts/urls.txt
    python annotation_collaboration/scripts/bulk_download_playsight.py \\
        --url "https://my.playsight.com/share?svkey=..." \\
        -o annotation_collaboration/downloads

Requires: ffmpeg on PATH, pip install requests beautifulsoup4 yt-dlp
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from services.playsight import (  # noqa: E402
    PlaySightImportError,
    download_playsight_video,
    is_playsight_url,
)

DEFAULT_OUT = REPO_ROOT / "annotation_collaboration" / "downloads"
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_urls_file(path: str) -> Path:
    """Resolve urls file from cwd, then next to this script."""
    p = Path(path)
    if p.is_file():
        return p.resolve()
    beside_script = SCRIPT_DIR / path
    if beside_script.is_file():
        return beside_script.resolve()
    raise FileNotFoundError(
        f"URLs file not found: {path!r}\n"
        f"  Tried: {Path(path).resolve()}\n"
        f"  Tried: {beside_script}\n"
        f"  Your file is probably: {SCRIPT_DIR / 'urls.txt'}"
    )


def _safe_stem(url: str, index: int) -> str:
    m = re.search(r"svkey=([^&]+)", url)
    if m:
        return f"playsight_{m.group(1).replace('/', '_')}"
    return f"playsight_{index:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-download PlaySight videos to MP4.")
    parser.add_argument(
        "urls_file",
        nargs="?",
        help="Text file with one PlaySight URL per line (# comments allowed)",
    )
    parser.add_argument("--url", action="append", default=[], help="Single URL (repeatable)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    urls: list[str] = list(args.url)
    if args.urls_file:
        text = _resolve_urls_file(args.urls_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)

    if not urls:
        parser.error("Provide urls.txt and/or --url")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["# video_id,local_path,source_url"]
    ok, fail = 0, 0

    for i, url in enumerate(urls, start=1):
        if not is_playsight_url(url):
            print(f"[{i}/{len(urls)}] SKIP (not PlaySight): {url[:80]}")
            fail += 1
            continue

        stem = _safe_stem(url, i)
        dest = out_dir / f"{stem}.mp4"
        print(f"\n[{i}/{len(urls)}] {stem}")
        print(f"  → {dest}")

        try:
            result = download_playsight_video(url, str(dest))
            mb = result.size_bytes / 1_048_576
            print(f"  OK ({mb:.1f} MB)")
            manifest_lines.append(f"{stem},{result.local_path},{url}")
            ok += 1
        except PlaySightImportError as e:
            print(f"  FAILED: {e}")
            fail += 1

    manifest_path = out_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nDone: {ok} ok, {fail} failed")
    print(f"Manifest: {manifest_path}")
    print("Upload the .mp4 files via the annotation web app when ready.")


if __name__ == "__main__":
    main()
