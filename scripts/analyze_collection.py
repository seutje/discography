#!/usr/bin/env python3
"""Batch-run the track analyzer over a directory tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze every audio file under a directory.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="Directory to scan.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output"), help="Report output directory.")
    parser.add_argument("--framework", type=Path, help="Force one analyzer framework for every track.")
    parser.add_argument("--beat-dir", type=Path, help="Directory containing precomputed beat_this .beats files.")
    parser.add_argument("--limit", type=int, help="Analyze only the first N matching files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = Path(__file__).with_name("analyze_track.py")
    audio_files = sorted(
        path
        for path in args.root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO and ".venv" not in path.parts
    )
    if args.limit:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        print(f"No supported audio files found under {args.root}", file=sys.stderr)
        return 1

    failures = 0
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] {audio_path}")
        cmd = [sys.executable, str(analyzer), str(audio_path), "--output-dir", str(args.output_dir)]
        if args.framework:
            cmd.extend(["--framework", str(args.framework)])
        if args.beat_dir:
            candidates = [
                args.beat_dir / f"{audio_path.stem}.beats",
                args.beat_dir / audio_path.relative_to(args.root).with_suffix(".beats"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    cmd.extend(["--beat-file", str(candidate)])
                    break
        completed = subprocess.run(cmd, text=True)
        if completed.returncode:
            failures += 1
            print(f"Failed: {audio_path}", file=sys.stderr)

    if failures:
        print(f"Finished with {failures} failure(s).", file=sys.stderr)
        return 1
    print(f"Finished {len(audio_files)} track(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
