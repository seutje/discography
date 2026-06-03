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
    parser.add_argument("--ollama-model", help="Optional Ollama model for bounded LLM scoring adjustment.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--ollama-timeout", type=float, default=240.0, help="Seconds to wait for each Ollama adjustment.")
    parser.add_argument("--ollama-num-ctx", type=int, default=16384, help="Ollama context window for each scoring adjustment.")
    parser.add_argument("--llm-max-axis-delta", type=float, default=1.5, help="Maximum LLM adjustment per axis around the Python score.")
    parser.add_argument(
        "--transcription-backend",
        choices=("none", "auto", "faster-whisper", "whisper-cli"),
        default="none",
        help="Optional speech-to-text backend for lyric intelligibility/repetition checks.",
    )
    parser.add_argument("--transcription-model", default="base", help="Whisper model name/path for transcription checks.")
    parser.add_argument("--transcription-language", default="en", help="Language code for transcription.")
    parser.add_argument("--transcription-timeout", type=float, default=900.0, help="Seconds to wait for whisper-cli transcription.")
    parser.add_argument("--transcription-device", default="auto", help="Transcription device, e.g. auto, cpu, cuda.")
    parser.add_argument("--transcription-compute-type", default="default", help="faster-whisper compute type.")
    parser.add_argument("--transcription-model-dir", type=Path, default=Path(".cache/whisper"), help="Writable directory for Whisper model downloads/cache.")
    parser.add_argument("--transcription-vad-filter", action="store_true", help="Enable faster-whisper VAD filtering. Off by default because VAD can drop sung vocals.")
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
        if args.ollama_model:
            cmd.extend(
                [
                    "--ollama-model",
                    args.ollama_model,
                    "--ollama-url",
                    args.ollama_url,
                    "--ollama-timeout",
                    str(args.ollama_timeout),
                    "--ollama-num-ctx",
                    str(args.ollama_num_ctx),
                    "--llm-max-axis-delta",
                    str(args.llm_max_axis_delta),
                ]
            )
        if args.transcription_backend != "none":
            cmd.extend(
                [
                    "--transcription-backend",
                    args.transcription_backend,
                    "--transcription-model",
                    args.transcription_model,
                    "--transcription-language",
                    args.transcription_language,
                    "--transcription-timeout",
                    str(args.transcription_timeout),
                    "--transcription-device",
                    args.transcription_device,
                    "--transcription-compute-type",
                    args.transcription_compute_type,
                    "--transcription-model-dir",
                    str(args.transcription_model_dir),
                ]
            )
            if args.transcription_vad_filter:
                cmd.append("--transcription-vad-filter")
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
