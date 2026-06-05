#!/usr/bin/env python3
"""Generate corrected, timed lyric JSON assets from repo lyric sheets.

The lyric sheet is the source of truth. Whisper provides rough word timing,
then the script aligns those words back to the corrected sheet and emits
line-level timings for the static GitHub Pages player.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
DEFAULT_OUTPUT_ROOT = ROOT / "gh-pages" / "data" / "lyrics"
DEFAULT_MODEL_CACHE = ROOT / ".cache" / "whisper"
IGNORED_DIRS = {
    ".agents",
    ".cache",
    ".codex",
    ".git",
    ".github",
    ".venv",
    "_site",
    "analysis-output",
    "analysis_outputs",
    "analyzer",
    "deploy",
    "gh-pages",
    "reports",
    "scripts",
    "suno-runs",
}


@dataclass
class LyricLine:
    index: int
    section: str
    text: str
    tokens: list[str]


def normalize_tokens(text: str) -> list[str]:
    text = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def parse_metadata(raw: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in re.findall(r"\[([A-Za-z_ -]+):\s*(.*?)\]", raw, flags=re.S):
        metadata[key.upper()] = value.strip()
    return metadata


def parse_lyrics(path: Path) -> tuple[dict[str, str], list[LyricLine]]:
    raw = read_text(path).replace("\r\n", "\n")
    metadata = parse_metadata(raw)
    marker = "[LYRICS]"
    lyric_text = raw.split(marker, 1)[1] if marker in raw else raw
    section = ""
    lines: list[LyricLine] = []
    for raw_line in lyric_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bracket = re.fullmatch(r"\[(.+?)\]", line)
        if bracket:
            section = bracket.group(1).strip()
            continue
        lines.append(LyricLine(len(lines), section, line, normalize_tokens(line)))
    return metadata, lines


def audio_path_for_text(text_path: Path) -> Path | None:
    audio_dir = text_path.parent / "audio"
    for suffix in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{text_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_whisper_model(model_name: str, model_cache: Path) -> Any:
    import whisper

    model_cache.mkdir(parents=True, exist_ok=True)
    return whisper.load_model(model_name, download_root=str(model_cache))


def transcribe(model: Any, audio_path: Path, language: str | None, word_timestamps: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": 0,
        "verbose": False,
        "fp16": False,
        "word_timestamps": word_timestamps,
    }
    if language:
        kwargs["language"] = language
    try:
        result = model.transcribe(str(audio_path), **kwargs)
        result["_timing_backend"] = "whisper-word-timestamps" if word_timestamps else "whisper-segment-timestamps"
        return result
    except AttributeError as exc:
        if not word_timestamps:
            raise
        if "triton" not in str(exc).lower() and "src" not in str(exc).lower():
            raise
        kwargs["word_timestamps"] = False
        result = model.transcribe(str(audio_path), **kwargs)
        result["_timing_backend"] = "whisper-segment-timestamps"
        return result


def observed_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                token = normalize_tokens(str(word.get("word") or ""))
                if not token:
                    continue
                words.append(
                    {
                        "token": token[0],
                        "start": float(word.get("start", segment.get("start", 0))),
                        "end": float(word.get("end", segment.get("end", 0))),
                    }
                )
            continue

        tokens = normalize_tokens(str(segment.get("text") or ""))
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        span = max(0.01, end - start)
        for index, token in enumerate(tokens):
            token_start = start + span * index / max(1, len(tokens))
            token_end = start + span * (index + 1) / max(1, len(tokens))
            words.append({"token": token, "start": token_start, "end": token_end})
    return words


def line_token_refs(lines: list[LyricLine]) -> list[dict[str, int | str]]:
    refs: list[dict[str, int | str]] = []
    for line in lines:
        for token in line.tokens:
            refs.append({"token": token, "line": line.index})
    return refs


def match_tokens(refs: list[dict[str, int | str]], words: list[dict[str, Any]]) -> dict[int, list[int]]:
    ref_tokens = [str(ref["token"]) for ref in refs]
    word_tokens = [str(word["token"]) for word in words]
    matcher = difflib.SequenceMatcher(None, ref_tokens, word_tokens, autojunk=False)
    matched: dict[int, list[int]] = {}
    for block in matcher.get_matching_blocks():
        if not block.size:
            continue
        for offset in range(block.size):
            ref_index = block.a + offset
            word_index = block.b + offset
            line_index = int(refs[ref_index]["line"])
            matched.setdefault(line_index, []).append(word_index)
    return matched


def weighted_fill(lines: list[LyricLine], timings: list[dict[str, float | None]], total_duration: float) -> None:
    matched_indices = [index for index, timing in enumerate(timings) if timing["start"] is not None]

    def fill_block(start_line: int, end_line: int, start_time: float, end_time: float) -> None:
        if start_line > end_line:
            return
        end_time = max(start_time + 0.1, end_time)
        block = lines[start_line : end_line + 1]
        weights = [max(1, len(line.tokens)) for line in block]
        total = sum(weights)
        cursor = start_time
        for line, weight in zip(block, weights):
            span = (end_time - start_time) * weight / total
            timings[line.index]["start"] = round(cursor, 3)
            timings[line.index]["end"] = round(min(end_time, cursor + span), 3)
            timings[line.index]["source"] = "interpolated"
            cursor += span

    if not matched_indices:
        fill_block(0, len(lines) - 1, 0, total_duration)
        return

    first = matched_indices[0]
    fill_block(0, first - 1, 0, float(timings[first]["start"] or 0))
    for previous, current in zip(matched_indices, matched_indices[1:]):
        fill_block(
            previous + 1,
            current - 1,
            float(timings[previous]["end"] or timings[previous]["start"] or 0),
            float(timings[current]["start"] or total_duration),
        )
    last = matched_indices[-1]
    fill_block(last + 1, len(lines) - 1, float(timings[last]["end"] or timings[last]["start"] or 0), total_duration)


def align_lines(lines: list[LyricLine], words: list[dict[str, Any]], total_duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = line_token_refs(lines)
    matches = match_tokens(refs, words)
    timings: list[dict[str, Any]] = [{"start": None, "end": None, "source": "unmatched", "matched_words": 0} for _ in lines]
    matched_word_total = 0
    for line in lines:
        word_indices = sorted(set(matches.get(line.index, [])))
        if not word_indices:
            continue
        matched_word_total += len(word_indices)
        start = min(float(words[index]["start"]) for index in word_indices)
        end = max(float(words[index]["end"]) for index in word_indices)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            timings[line.index] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "source": "whisper-aligned",
                "matched_words": len(word_indices),
            }
    weighted_fill(lines, timings, total_duration)

    previous_end = 0.0
    for timing in timings:
        start = max(previous_end, float(timing["start"] or 0))
        end = max(start + 0.1, float(timing["end"] or start + 0.1))
        timing["start"] = round(start, 3)
        timing["end"] = round(end, 3)
        previous_end = end

    output_lines = []
    for line, timing in zip(lines, timings):
        output_lines.append(
            {
                "index": line.index,
                "section": line.section,
                "start": timing["start"],
                "end": timing["end"],
                "text": line.text,
                "timing_source": timing["source"],
                "matched_words": timing["matched_words"],
            }
        )
    total_ref_tokens = sum(len(line.tokens) for line in lines)
    diagnostics = {
        "line_count": len(lines),
        "reference_word_count": total_ref_tokens,
        "observed_word_count": len(words),
        "matched_word_count": matched_word_total,
        "matched_word_ratio": round(matched_word_total / total_ref_tokens, 3) if total_ref_tokens else 0,
    }
    return output_lines, diagnostics


def concise_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(float(segment.get("start", 0)), 3),
            "end": round(float(segment.get("end", 0)), 3),
            "text": str(segment.get("text") or "").strip(),
        }
        for segment in transcript.get("segments") or []
    ]


def build_payload(text_path: Path, audio_path: Path, args: argparse.Namespace, model: Any) -> dict[str, Any]:
    metadata, lines = parse_lyrics(text_path)
    transcript = transcribe(model, audio_path, args.language, args.word_timestamps)
    total_duration = max((float(segment.get("end", 0)) for segment in transcript.get("segments") or []), default=0)
    words = observed_words(transcript)
    timed_lines, diagnostics = align_lines(lines, words, max(0.1, total_duration or 0.1))
    return {
        "schema": "discography-timed-lyrics-v1",
        "album": text_path.parent.name,
        "title": metadata.get("TITLE") or text_path.stem.split(" - ", 1)[-1],
        "text_path": text_path.relative_to(ROOT).as_posix(),
        "audio_path": audio_path.relative_to(ROOT).as_posix(),
        "duration_seconds": round(float(total_duration or 0), 3),
        "model": args.model,
        "language": transcript.get("language") or args.language,
        "timing_granularity": "line",
        "timing_backend": transcript.get("_timing_backend", "unknown"),
        "diagnostics": diagnostics,
        "transcript_segments": concise_segments(transcript),
        "lines": timed_lines,
    }


def text_paths_for_album(album: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        prefix = path.stem.split(" - ", 1)[0]
        return (int(prefix) if prefix.isdigit() else 9999, path.name.lower())

    return sorted(album.glob("*.txt"), key=key)


def discover_albums() -> list[Path]:
    albums = []
    for path in ROOT.iterdir():
        if not path.is_dir() or path.name in IGNORED_DIRS:
            continue
        if any(audio_path_for_text(text_path) for text_path in text_paths_for_album(path)):
            albums.append(path)
    return sorted(albums, key=lambda path: path.name.lower())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate timed lyric JSON for a repo album.")
    parser.add_argument("--album", action="append", help="Album directory to process. May be passed more than once.")
    parser.add_argument("--all", action="store_true", help="Process every album directory with lyric sheets and audio.")
    parser.add_argument("--exclude-album", action="append", default=[], help="Album directory name to skip. May be passed more than once.")
    parser.add_argument("--model", default="base", help="Whisper model name, e.g. tiny, base, small.")
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE, help="Directory for Whisper model files.")
    parser.add_argument("--language", default="en", help="Whisper language code. Use an empty string for auto-detect.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--word-timestamps", action="store_true", help="Use Whisper word timestamps when the local install supports them.")
    parser.add_argument("--force", action="store_true", help="Regenerate files that already exist.")
    return parser.parse_args()


def process_album(album: Path, args: argparse.Namespace, model: Any) -> int:
    if not album.exists():
        raise SystemExit(f"Album not found: {album}")

    written = 0
    for text_path in text_paths_for_album(album):
        audio_path = audio_path_for_text(text_path)
        if not audio_path:
            print(f"skip missing audio: {text_path.relative_to(ROOT)}")
            continue
        output_path = args.output_root / album.name / f"{text_path.stem}.json"
        if output_path.exists() and not args.force:
            print(f"skip existing: {output_path.relative_to(ROOT)}")
            continue
        print(f"timing {text_path.relative_to(ROOT)}")
        payload = build_payload(text_path, audio_path, args, model)
        write_json(output_path, payload)
        ratio = payload["diagnostics"]["matched_word_ratio"]
        print(f"wrote {output_path.relative_to(ROOT)} ({ratio:.0%} word match)")
        written += 1
    return written


def selected_albums(args: argparse.Namespace) -> list[Path]:
    excluded = {name.casefold() for name in args.exclude_album}
    if args.all:
        albums = discover_albums()
    else:
        albums = [ROOT / name for name in (args.album or ["Closed Doors"])]
    return [album for album in albums if album.name.casefold() not in excluded]


def main() -> int:
    args = parse_args()
    if args.language == "":
        args.language = None

    albums = selected_albums(args)
    if not albums:
        print("No albums selected.")
        return 0

    print(f"Loading Whisper model: {args.model}")
    model = load_whisper_model(args.model, args.model_cache)

    written = 0
    for album in albums:
        print(f"album {album.name}")
        written += process_album(album, args, model)
    print(f"Generated {written} timed lyric file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
