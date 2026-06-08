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
LOW_INFORMATION_ADLIB_TOKENS = {
    "ah",
    "ay",
    "eh",
    "ha",
    "hey",
    "hm",
    "hmm",
    "huh",
    "la",
    "mm",
    "mmm",
    "no",
    "oh",
    "ooh",
    "uh",
    "um",
    "woah",
    "yeah",
    "yo",
}
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


def is_low_information_adlib(tokens: list[str]) -> bool:
    """Detect transcript-only vocalization loops that destabilize lyric alignment."""
    if len(tokens) < 3:
        return False
    meaningful = [token for token in tokens if token not in {"a", "an", "the"}]
    if not meaningful:
        return False
    if not set(meaningful) <= LOW_INFORMATION_ADLIB_TOKENS:
        return False
    unique_ratio = len(set(meaningful)) / len(meaningful)
    return unique_ratio <= 0.5


def alignment_token_indices(tokens: list[str]) -> list[int]:
    if is_low_information_adlib(tokens):
        return []

    kept: list[int] = []
    index = 0
    while index < len(tokens):
        best_loop: tuple[int, int] | None = None
        for size in range(1, min(4, len(tokens) - index) + 1):
            phrase = tokens[index : index + size]
            repeats = 1
            while tokens[index + repeats * size : index + (repeats + 1) * size] == phrase:
                repeats += 1
            low_information = set(phrase) <= LOW_INFORMATION_ADLIB_TOKENS
            if repeats >= 4 or (low_information and repeats >= 3):
                best_loop = max(best_loop or (0, 0), (size, repeats), key=lambda item: item[0] * item[1])

        if best_loop:
            size, repeats = best_loop
            phrase = tokens[index : index + size]
            if not set(phrase) <= LOW_INFORMATION_ADLIB_TOKENS:
                kept.extend(range(index, index + size))
            index += size * repeats
            continue

        kept.append(index)
        index += 1

    return kept


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


def split_metadata_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;\n|]+", value) if item.strip()]


def timing_lines(lines: list[LyricLine], metadata: dict[str, str]) -> tuple[list[LyricLine], dict[str, Any]]:
    skipped_sections = set(split_metadata_list(metadata.get("TIMING_SKIP_SECTIONS", "")))
    if not skipped_sections:
        return lines, {"skipped_line_count": 0, "skipped_sections": []}

    kept: list[LyricLine] = []
    skipped_count = 0
    for line in lines:
        if line.section in skipped_sections:
            skipped_count += 1
            continue
        kept.append(LyricLine(len(kept), line.section, line.text, line.tokens))

    return kept, {
        "skipped_line_count": skipped_count,
        "skipped_sections": sorted(skipped_sections),
    }


def timing_section_starts(metadata: dict[str, str]) -> dict[str, float]:
    starts: dict[str, float] = {}
    for item in split_metadata_list(metadata.get("TIMING_SECTION_STARTS", "")):
        section, separator, value = item.partition("=")
        if not separator:
            continue
        try:
            starts[section.strip()] = float(value.strip())
        except ValueError:
            continue
    return starts


def timing_line_starts(metadata: dict[str, str]) -> list[dict[str, str | float | None]]:
    starts: list[dict[str, str | float | None]] = []
    for item in split_metadata_list(metadata.get("TIMING_LINE_STARTS", "")):
        text, separator, value = item.partition("=")
        if not separator:
            continue
        try:
            start = float(value.strip())
        except ValueError:
            continue
        section: str | None = None
        text = text.strip()
        if "::" in text:
            section, text = [part.strip() for part in text.split("::", 1)]
        starts.append(
            {
                "section": section.casefold() if section else None,
                "text": text.casefold(),
                "label": f"{section}::{text}" if section else text,
                "start": start,
            }
        )
    return starts


def apply_section_start_overrides(timed_lines: list[dict[str, Any]], metadata: dict[str, str]) -> dict[str, Any]:
    starts = timing_section_starts(metadata)
    applied: dict[str, float] = {}
    if not starts:
        return {"section_start_overrides": applied}

    seen_sections: set[str] = set()
    previous_end = 0.0
    for line in timed_lines:
        section = str(line.get("section") or "")
        if section in starts and section not in seen_sections:
            start = max(previous_end, starts[section])
            line["start"] = round(start, 3)
            line["end"] = round(max(start + 0.1, float(line.get("end", start + 0.1))), 3)
            line["timing_source"] = f"{line.get('timing_source', 'timed')}-section-override"
            applied[section] = round(start, 3)
        seen_sections.add(section)
        previous_end = max(previous_end, float(line.get("end", line.get("start", 0))))
    return {"section_start_overrides": applied}


def apply_line_start_overrides(timed_lines: list[dict[str, Any]], metadata: dict[str, str]) -> dict[str, Any]:
    starts = timing_line_starts(metadata)
    applied: dict[str, float] = {}
    if not starts:
        return {"line_start_overrides": applied}

    for line in timed_lines:
        text_key = str(line.get("text") or "").strip().casefold()
        section_key = str(line.get("section") or "").strip().casefold()
        for override in starts:
            if text_key != override["text"]:
                continue
            if override["section"] is not None and section_key != override["section"]:
                continue
            start = float(override["start"])
            line["start"] = round(start, 3)
            line["end"] = round(max(start + 0.1, float(line.get("end", start + 0.1))), 3)
            line["timing_source"] = f"{line.get('timing_source', 'timed')}-line-override"
            applied[str(override["label"])] = round(start, 3)
            break
    return {"line_start_overrides": applied}


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


def alignment_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in transcript.get("segments") or []
        if alignment_token_indices(normalize_tokens(str(segment.get("text") or "")))
    ]


def observed_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in alignment_segments(transcript):
        segment_tokens = normalize_tokens(str(segment.get("text") or ""))
        kept_token_indices = alignment_token_indices(segment_tokens)
        segment_words = segment.get("words") or []
        if segment_words:
            kept_word_indices = set(kept_token_indices)
            word_token_index = 0
            for word in segment_words:
                token = normalize_tokens(str(word.get("word") or ""))
                if not token:
                    continue
                if word_token_index not in kept_word_indices:
                    word_token_index += 1
                    continue
                words.append(
                    {
                        "token": token[0],
                        "start": float(word.get("start", segment.get("start", 0))),
                        "end": float(word.get("end", segment.get("end", 0))),
                    }
                )
                word_token_index += 1
            continue

        tokens = [segment_tokens[index] for index in kept_token_indices]
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        span = max(0.01, end - start)
        original_count = max(1, len(segment_tokens))
        for original_index, token in zip(kept_token_indices, tokens):
            token_start = start + span * original_index / original_count
            token_end = start + span * (original_index + 1) / original_count
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


def compact_word_clusters(line: LyricLine, word_indices: list[int], words: list[dict[str, Any]]) -> list[list[int]]:
    indices = sorted(set(word_indices))
    if len(indices) <= 1:
        return [indices] if indices else []

    max_gap = 3.2
    max_span = max(4.0, min(12.0, len(line.tokens) * 1.15))
    clusters: list[list[int]] = []
    current = [indices[0]]
    for index in indices[1:]:
        previous = current[-1]
        gap = float(words[index]["start"]) - float(words[previous]["end"])
        span = float(words[index]["end"]) - float(words[current[0]]["start"])
        if gap <= max_gap and span <= max_span:
            current.append(index)
        else:
            clusters.append(current)
            current = [index]
    clusters.append(current)
    return clusters


def select_word_cluster(
    line: LyricLine,
    word_indices: list[int],
    words: list[dict[str, Any]],
    cursor: int,
) -> list[int]:
    clusters = compact_word_clusters(line, word_indices, words)
    if not clusters:
        return []

    max_count = max(len(cluster) for cluster in clusters)
    good_count = max(1, max_count - 2)
    after_cursor = [cluster for cluster in clusters if cluster[0] >= cursor and len(cluster) >= good_count]
    if after_cursor:
        return min(after_cursor, key=lambda cluster: (cluster[0], -(len(cluster))))

    def score(cluster: list[int]) -> tuple[int, float, int]:
        span = float(words[cluster[-1]]["end"]) - float(words[cluster[0]]["start"])
        return (len(cluster), -span, -cluster[0])

    return max(clusters, key=score)


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


def ordered_overlap_score(reference: list[str], observed: list[str]) -> float:
    if not reference or not observed:
        return 0.0

    def token_matches(left: str, right: str) -> bool:
        if left == right:
            return True
        if left in {"a", "an", "the"} and right in {"a", "an", "the"}:
            return True
        if {left, right} <= {"if", "in"}:
            return True
        if left.endswith("s") and left[:-1] == right:
            return True
        if right.endswith("s") and right[:-1] == left:
            return True
        return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.74

    matched = 0
    observed_index = 0
    for token in reference:
        while observed_index < len(observed):
            if token_matches(token, observed[observed_index]):
                matched += 1
                observed_index += 1
                break
            observed_index += 1
    return matched / len(reference)


def fuzzy_segment_candidate(
    line: LyricLine,
    segments: list[dict[str, Any]],
    previous_end: float,
    cluster_start: float | None,
) -> dict[str, Any] | None:
    if not line.tokens:
        return None
    threshold = 0.5 if len(line.tokens) >= 5 else 0.72
    search_end = cluster_start - 0.25 if cluster_start is not None else previous_end + 22.0
    candidates = []
    for segment in segments:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        if end < previous_end - 0.5:
            continue
        if start > search_end:
            break
        score = ordered_overlap_score(line.tokens, normalize_tokens(str(segment.get("text") or "")))
        if score >= threshold:
            candidates.append((score, start, end))
    if not candidates:
        return None
    score, start, end = max(candidates, key=lambda item: (item[0], -item[1]))
    return {"start": start, "end": end, "score": score}


def align_lines(
    lines: list[LyricLine],
    words: list[dict[str, Any]],
    total_duration: float,
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = line_token_refs(lines)
    matches = match_tokens(refs, words)
    timings: list[dict[str, Any]] = [{"start": None, "end": None, "source": "unmatched", "matched_words": 0} for _ in lines]
    matched_word_total = 0
    cursor = 0
    previous_end = 0.0
    for line in lines:
        word_indices = select_word_cluster(line, matches.get(line.index, []), words, cursor)
        cluster_start = min((float(words[index]["start"]) for index in word_indices), default=None)
        fuzzy = fuzzy_segment_candidate(line, segments, previous_end, cluster_start)
        use_fuzzy = fuzzy is not None and (
            not word_indices or (cluster_start is not None and cluster_start - previous_end > 10)
        )
        if use_fuzzy:
            start = float(fuzzy["start"])
            end = float(fuzzy["end"])
            timings[line.index] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "source": "whisper-segment-fuzzy",
                "matched_words": 0,
            }
            previous_end = max(previous_end, end)
            continue
        if not word_indices:
            continue
        cursor = max(cursor, word_indices[-1] + 1)
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
            previous_end = max(previous_end, end)
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


def repeated_short_line_candidates(lines: list[LyricLine]) -> dict[tuple[str, ...], str]:
    counts: dict[tuple[str, ...], int] = {}
    labels: dict[tuple[str, ...], str] = {}
    for line in lines:
        key = tuple(line.tokens)
        if not 2 <= len(key) <= 3:
            continue
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, line.text)
    return {key: labels[key] for key, count in counts.items() if count >= 2}


def repeated_candidate_span(tokens: list[str], candidates: dict[tuple[str, ...], str]) -> tuple[tuple[str, ...], int] | None:
    fillers = {"ah", "ay", "eh", "hm", "hmm", "uh", "um", "yeah", "yo"}
    trimmed = [token for token in tokens if token not in fillers]
    if not trimmed:
        return None
    for candidate in sorted(candidates, key=len, reverse=True):
        size = len(candidate)
        if len(trimmed) % size:
            continue
        repeats = len(trimmed) // size
        if repeats < 1:
            continue
        if all(tuple(trimmed[index : index + size]) == candidate for index in range(0, len(trimmed), size)):
            return candidate, repeats
    return None


def has_matching_line_near(lines: list[dict[str, Any]], candidate: tuple[str, ...], start: float, end: float) -> bool:
    for line in lines:
        if tuple(normalize_tokens(str(line.get("text") or ""))) != candidate:
            continue
        line_start = float(line.get("start", 0))
        line_end = float(line.get("end", line_start))
        overlaps = max(start, line_start) <= min(end, line_end) + 0.5
        nearby = abs(start - line_start) <= 1.0 or abs(end - line_end) <= 1.0
        if overlaps or nearby:
            return True
    return False


def insert_repeated_adlibs(
    timed_lines: list[dict[str, Any]],
    sheet_lines: list[LyricLine],
    transcript: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    candidates = repeated_short_line_candidates(sheet_lines)
    if not candidates:
        return timed_lines, 0

    additions: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        tokens = normalize_tokens(str(segment.get("text") or ""))
        match = repeated_candidate_span(tokens, candidates)
        if not match:
            continue
        candidate, repeats = match
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        if end <= start or has_matching_line_near(timed_lines, candidate, start, end):
            continue
        span = (end - start) / repeats
        for repeat_index in range(repeats):
            line_start = start + span * repeat_index
            line_end = start + span * (repeat_index + 1)
            if has_matching_line_near(timed_lines + additions, candidate, line_start, line_end):
                continue
            additions.append(
                {
                    "index": -1,
                    "section": "Detected ad-lib",
                    "start": round(line_start, 3),
                    "end": round(line_end, 3),
                    "text": candidates[candidate],
                    "timing_source": "transcript-repeated-adlib",
                    "matched_words": len(candidate),
                    "line_type": "transcript-adlib",
                }
            )

    if not additions:
        return timed_lines, 0

    merged = sorted([*timed_lines, *additions], key=lambda line: (float(line["start"]), float(line["end"])))
    for index, line in enumerate(merged):
        line["index"] = index
        line.setdefault("line_type", "lyric-sheet")
    return merged, len(additions)


def build_payload(text_path: Path, audio_path: Path, args: argparse.Namespace, model: Any) -> dict[str, Any]:
    metadata, parsed_lines = parse_lyrics(text_path)
    lines, timing_line_diagnostics = timing_lines(parsed_lines, metadata)
    transcript = transcribe(model, audio_path, args.language, args.word_timestamps)
    total_duration = max((float(segment.get("end", 0)) for segment in transcript.get("segments") or []), default=0)
    alignable_segments = alignment_segments(transcript)
    transcript_segments = concise_segments(transcript)
    alignment_transcript_segments = concise_segments({"segments": alignable_segments})
    words = observed_words(transcript)
    timed_lines, diagnostics = align_lines(lines, words, max(0.1, total_duration or 0.1), alignment_transcript_segments)
    diagnostics.update(apply_section_start_overrides(timed_lines, metadata))
    diagnostics.update(apply_line_start_overrides(timed_lines, metadata))
    timed_lines, inserted_adlibs = insert_repeated_adlibs(timed_lines, lines, transcript)
    diagnostics.update(timing_line_diagnostics)
    diagnostics["inserted_adlib_count"] = inserted_adlibs
    diagnostics["ignored_adlib_segment_count"] = len(transcript.get("segments") or []) - len(alignable_segments)
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
        "transcript_segments": transcript_segments,
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


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate timed lyric JSON for a repo album.")
    parser.add_argument("--album", action="append", help="Album directory to process. May be passed more than once.")
    parser.add_argument("--track", action="append", default=[], help="Only process matching track title or text-file stem. May be passed more than once.")
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
        if args.track:
            needles = [value.casefold() for value in args.track]
            haystacks = [text_path.stem.casefold(), text_path.stem.split(" - ", 1)[-1].casefold()]
            if not any(needle in haystack for needle in needles for haystack in haystacks):
                continue
        audio_path = audio_path_for_text(text_path)
        if not audio_path:
            print(f"skip missing audio: {text_path.relative_to(ROOT)}")
            continue
        output_path = args.output_root / album.name / f"{text_path.stem}.json"
        if output_path.exists() and not args.force:
            print(f"skip existing: {display_path(output_path)}")
            continue
        print(f"timing {text_path.relative_to(ROOT)}")
        payload = build_payload(text_path, audio_path, args, model)
        write_json(output_path, payload)
        ratio = payload["diagnostics"]["matched_word_ratio"]
        print(f"wrote {display_path(output_path)} ({ratio:.0%} word match)")
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
