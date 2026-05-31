#!/usr/bin/env python3
"""Analyze an MP3 and produce a framework-guided music report."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import mutagen
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")
TRANSFORMATION_TERMS = {
    "build",
    "drop",
    "dropout",
    "filter",
    "filtered",
    "shift",
    "shifts",
    "switch",
    "evolve",
    "evolving",
    "returns",
    "recurs",
    "reprise",
    "variation",
    "transform",
    "transformed",
    "layer",
    "layered",
    "layers",
    "density",
    "sparse",
    "fuller",
    "silence",
    "bridge",
    "break",
    "halftime",
    "half-time",
    "double-time",
    "modulation",
    "pivot",
    "reverb",
    "delay",
    "distortion",
    "glitch",
    "granular",
    "vocoder",
    "automation",
    "tape",
    "wobble",
    "texture",
    "textures",
    "counter",
    "countermelody",
    "harmony",
    "harmonies",
}


@dataclass(frozen=True)
class TrackText:
    path: Path | None
    raw: str
    tags: dict[str, str]
    lyrics: str
    lyric_sections: list[tuple[str, str]]


def db(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def count_syllables_rough(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease_rough(text: str) -> float | None:
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    words = re.findall(r"[A-Za-z']+", text)
    if not words or not sentences:
        return None
    syllables = sum(count_syllables_rough(word) for word in words)
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes}:{sec:05.2f}"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n...[truncated]...\n" + text[-keep:]


def run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    return json.loads(completed.stdout)


def parse_beat_this_file(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None

    beats: list[float] = []
    positions: list[int | None] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            beats.append(float(parts[0]))
        except ValueError:
            continue
        if len(parts) > 1:
            try:
                positions.append(int(float(parts[1])))
            except ValueError:
                positions.append(None)
        else:
            positions.append(None)

    if len(beats) < 2:
        return {
            "path": str(path),
            "beat_count": len(beats),
            "downbeat_count": 0,
            "available": False,
        }

    beat_array = np.array(beats)
    intervals = np.diff(beat_array)
    intervals = intervals[(intervals > 0.15) & (intervals < 3.0)]
    median_interval = safe_float(np.median(intervals)) if intervals.size else 0.0
    interval_cv = safe_float(np.std(intervals) / (np.mean(intervals) + 1e-9)) if intervals.size else 1.0

    downbeats = [beat for beat, position in zip(beats, positions) if position == 1]
    downbeat_intervals = np.diff(np.array(downbeats)) if len(downbeats) > 1 else np.array([])
    downbeat_cv = (
        safe_float(np.std(downbeat_intervals) / (np.mean(downbeat_intervals) + 1e-9))
        if downbeat_intervals.size
        else 1.0
    )

    tempo = 60.0 / median_interval if median_interval else 0.0
    return {
        "path": str(path),
        "available": True,
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "first_beat": beats[0],
        "last_beat": beats[-1],
        "tempo_bpm": tempo,
        "double_tempo_bpm": tempo * 2 if tempo else 0.0,
        "half_tempo_bpm": tempo / 2 if tempo else 0.0,
        "median_beat_interval_seconds": median_interval,
        "beat_interval_cv": interval_cv,
        "beat_grid_stability": clamp(1.0 - interval_cv * 4.0, 0.0, 1.0),
        "median_bar_interval_seconds": safe_float(np.median(downbeat_intervals)) if downbeat_intervals.size else None,
        "bar_interval_cv": downbeat_cv if downbeat_intervals.size else None,
        "bar_grid_stability": clamp(1.0 - downbeat_cv * 3.0, 0.0, 1.0) if downbeat_intervals.size else None,
        "beats": beats,
        "beat_positions": positions,
        "downbeats": downbeats,
        "sample_beats": beats[:12],
        "sample_downbeats": downbeats[:8],
    }


def read_track_text(path: Path | None) -> TrackText:
    if not path:
        return TrackText(None, "", {}, "", [])

    raw = path.read_text(encoding="utf-8")
    tags: dict[str, str] = {}
    for match in re.finditer(r"^\[([A-Z][A-Z0-9 _/-]*):\s*(.*?)\]\s*$", raw, flags=re.MULTILINE):
        tags[match.group(1).strip().upper()] = match.group(2).strip()

    lyric_start = re.search(r"^\[LYRICS\]\s*$", raw, flags=re.MULTILINE)
    lyrics = raw[lyric_start.end() :] if lyric_start else raw

    sections: list[tuple[str, str]] = []
    current_name = "Text"
    current_lines: list[str] = []
    for line in lyrics.splitlines():
        header = re.match(r"^\[([^\]]+)\]\s*$", line.strip())
        if header:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = header.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return TrackText(path, raw, tags, lyrics.strip(), [(name, body) for name, body in sections if body])


def guess_companion_text(audio_path: Path) -> Path | None:
    if audio_path.parent.name.lower() != "audio":
        candidate = audio_path.with_suffix(".txt")
        return candidate if candidate.exists() else None
    candidate = audio_path.parent.parent / f"{audio_path.stem}.txt"
    return candidate if candidate.exists() else None


def choose_framework(audio_path: Path, text: TrackText, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit

    analyzer_dir = Path("analyzer")
    genre = text.tags.get("GENRE", "").lower()
    candidates = {
        "rap": analyzer_dir / "Hip hop & rap.txt",
        "hip": analyzer_dir / "Hip hop & rap.txt",
        "trap": analyzer_dir / "Hip hop & rap.txt",
        "metal": analyzer_dir / "metal.txt",
        "djent": analyzer_dir / "metal.txt",
        "prog": analyzer_dir / "metal.txt",
        "indie": analyzer_dir / "Indie.txt",
        "rock": analyzer_dir / "Indie.txt",
        "pop": analyzer_dir / "Indie.txt",
        "ambient": analyzer_dir / "Sorelian.txt",
        "art song": analyzer_dir / "Sorelian.txt",
        "classical": analyzer_dir / "Sorelian.txt",
    }
    for needle, framework in candidates.items():
        if needle in genre and framework.exists():
            return framework

    album = audio_path.parent.parent.name.lower() if audio_path.parent.name.lower() == "audio" else audio_path.parent.name.lower()
    if "stderr" in album and (analyzer_dir / "Hip hop & rap.txt").exists():
        return analyzer_dir / "Hip hop & rap.txt"
    if (analyzer_dir / "Sorelian.txt").exists():
        return analyzer_dir / "Sorelian.txt"
    return None


def estimate_key(chroma: np.ndarray) -> tuple[str, float]:
    vector = np.mean(chroma, axis=1)
    if np.linalg.norm(vector) == 0:
        return "Unknown", 0.0

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = (a - np.mean(a)) / (np.std(a) + 1e-9)
        b = (b - np.mean(b)) / (np.std(b) + 1e-9)
        return float(np.mean(a * b))

    scores: list[tuple[str, float]] = []
    for i, name in enumerate(PITCH_CLASSES):
        scores.append((f"{name} major", corr(vector, np.roll(MAJOR_PROFILE, i))))
        scores.append((f"{name} minor", corr(vector, np.roll(MINOR_PROFILE, i))))
    scores.sort(key=lambda item: item[1], reverse=True)
    best = scores[0]
    confidence = max(0.0, min(1.0, (best[1] - scores[1][1] + 0.1) / 0.6))
    return best[0], confidence


def pick_boundaries(feature_matrix: np.ndarray, duration: float, frame_times: np.ndarray) -> list[float]:
    if feature_matrix.shape[1] < 8:
        return [0.0, duration]

    means = np.mean(feature_matrix, axis=1, keepdims=True)
    stds = np.std(feature_matrix, axis=1, keepdims=True) + 1e-9
    normed = (feature_matrix - means) / stds
    changes = np.linalg.norm(np.diff(normed, axis=1), axis=0)
    if len(changes) >= 9:
        kernel = np.ones(9) / 9
        changes = np.convolve(changes, kernel, mode="same")

    min_gap = 12.0
    max_internal = max(2, min(9, int(duration // 35)))
    order = np.argsort(changes)[::-1]
    chosen: list[float] = []
    for idx in order:
        if len(chosen) >= max_internal:
            break
        if idx + 1 >= len(frame_times):
            continue
        t = float(frame_times[idx + 1])
        if t < 8 or t > duration - 8:
            continue
        if all(abs(t - other) >= min_gap for other in chosen):
            chosen.append(t)
    return [0.0, *sorted(chosen), duration]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return safe_float(np.dot(a, b) / denom)


def bounded_ratio_delta(a: float, b: float) -> float:
    a = max(float(a), 1e-9)
    b = max(float(b), 1e-9)
    return min(1.0, abs(math.log2(a / b)))


def vector_distance(a: np.ndarray, b: np.ndarray) -> float:
    return clamp(1.0 - cosine_similarity(a, b), 0.0, 1.0)


def section_frame_indices(frame_times: np.ndarray, start: float, end: float) -> np.ndarray:
    indices = np.where((frame_times >= start) & (frame_times < end))[0]
    if indices.size:
        return indices
    center = (start + end) / 2
    nearest = int(np.argmin(np.abs(frame_times - center)))
    return np.array([nearest])


def section_beat_stats(beat_this: dict[str, Any] | None, start: float, end: float) -> dict[str, Any]:
    if not beat_this or not beat_this.get("available"):
        return {
            "beat_count": 0,
            "downbeat_count": 0,
            "median_beat_interval_seconds": None,
            "beat_interval_cv": None,
            "bar_interval_cv": None,
        }

    beats = np.array(beat_this.get("beats") or [], dtype=float)
    positions = beat_this.get("beat_positions") or []
    if beats.size == 0:
        return {
            "beat_count": 0,
            "downbeat_count": 0,
            "median_beat_interval_seconds": None,
            "beat_interval_cv": None,
            "bar_interval_cv": None,
        }

    mask = (beats >= start) & (beats < end)
    section_beats = beats[mask]
    section_positions = [position for position, keep in zip(positions, mask.tolist()) if keep]
    intervals = np.diff(section_beats)
    intervals = intervals[(intervals > 0.15) & (intervals < 3.0)]
    median_interval = safe_float(np.median(intervals)) if intervals.size else None
    interval_cv = safe_float(np.std(intervals) / (np.mean(intervals) + 1e-9)) if intervals.size else None

    section_downbeats = np.array(
        [beat for beat, position in zip(section_beats.tolist(), section_positions) if position == 1],
        dtype=float,
    )
    downbeat_intervals = np.diff(section_downbeats) if section_downbeats.size > 1 else np.array([])
    bar_interval_cv = (
        safe_float(np.std(downbeat_intervals) / (np.mean(downbeat_intervals) + 1e-9))
        if downbeat_intervals.size
        else None
    )

    return {
        "beat_count": int(section_beats.size),
        "downbeat_count": int(section_downbeats.size),
        "median_beat_interval_seconds": median_interval,
        "beat_interval_cv": interval_cv,
        "bar_interval_cv": bar_interval_cv,
    }


def classify_rule_changes(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    global_beat_interval: float | None,
) -> list[str]:
    rules: list[str] = []
    beat_interval = current.get("median_beat_interval_seconds")
    if global_beat_interval and beat_interval:
        ratio = beat_interval / global_beat_interval
        if 1.45 <= ratio <= 2.65:
            rules.append("halftime-feel")
        elif 0.38 <= ratio <= 0.70:
            rules.append("double-time-feel")

    if current.get("beat_interval_cv") is not None and current["beat_interval_cv"] > 0.18:
        rules.append("local-grid-deviation")
    if current.get("bar_interval_cv") is not None and current["bar_interval_cv"] > 0.15:
        rules.append("bar-grid-deviation")

    if previous:
        chroma_delta = vector_distance(current["chroma_vector"], previous["chroma_vector"])
        timbre_delta = vector_distance(current["mfcc_vector"], previous["mfcc_vector"])
        rhythm_delta = bounded_ratio_delta(current["onset_rate"], previous["onset_rate"])
        centroid_delta = bounded_ratio_delta(current["centroid_mean"], previous["centroid_mean"])
        bandwidth_delta = bounded_ratio_delta(current["bandwidth_mean"], previous["bandwidth_mean"])
        flatness_delta = abs(current["flatness_mean"] - previous["flatness_mean"])
        rms_delta = abs(current["rms_db_mean"] - previous["rms_db_mean"]) / 10.0

        if chroma_delta > 0.14:
            rules.append("harmonic-pivot")
        if rhythm_delta > 0.28:
            rules.append("rhythmic-law-shift")
        if timbre_delta > 0.08 or centroid_delta > 0.24:
            rules.append("timbre-law-shift")
        if bandwidth_delta > 0.22 or flatness_delta > 0.05 or rms_delta > 0.40:
            rules.append("texture-law-shift")

    return sorted(set(rules))


def section_evolving_grammar_analysis(
    section_spans: list[dict[str, Any]],
    frame_times: np.ndarray,
    chroma: np.ndarray,
    mfcc: np.ndarray,
    rms: np.ndarray,
    centroid: np.ndarray,
    bandwidth: np.ndarray,
    flatness: np.ndarray,
    zcr: np.ndarray,
    onset_env: np.ndarray,
    onset_frames: np.ndarray,
    sr: int,
    hop_length: int,
    beat_this: dict[str, Any] | None,
) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    global_beat_interval = beat_this.get("median_beat_interval_seconds") if beat_this else None
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    for section in section_spans:
        start = float(section["start"])
        end = float(section["end"])
        duration = max(float(section["duration"]), 1e-9)
        idx = section_frame_indices(frame_times, start, end)
        beat_stats = section_beat_stats(beat_this, start, end)
        onset_count = int(np.sum((onset_times >= start) & (onset_times < end)))

        chroma_vector = np.mean(chroma[:, idx], axis=1)
        mfcc_vector = np.mean(mfcc[:, idx], axis=1)
        profile = {
            "index": section["index"],
            "start": start,
            "end": end,
            "duration": duration,
            "chroma_vector": chroma_vector,
            "mfcc_vector": mfcc_vector,
            "onset_rate": onset_count / duration,
            "onset_strength_mean": safe_float(np.mean(onset_env[idx])),
            "rms_db_mean": db(float(np.mean(rms[idx]))),
            "centroid_mean": safe_float(np.mean(centroid[idx])),
            "bandwidth_mean": safe_float(np.mean(bandwidth[idx])),
            "flatness_mean": safe_float(np.mean(flatness[idx])),
            "zcr_mean": safe_float(np.mean(zcr[idx])),
            **beat_stats,
        }
        previous = profiles[-1] if profiles else None
        profile["rule_changes"] = classify_rule_changes(previous, profile, global_beat_interval)
        profiles.append(profile)

    adjacent_changes: list[dict[str, Any]] = []
    for previous, current in zip(profiles, profiles[1:]):
        harmonic_delta = vector_distance(current["chroma_vector"], previous["chroma_vector"])
        timbre_delta = vector_distance(current["mfcc_vector"], previous["mfcc_vector"])
        rhythm_delta = bounded_ratio_delta(current["onset_rate"], previous["onset_rate"])
        texture_delta = max(
            bounded_ratio_delta(current["centroid_mean"], previous["centroid_mean"]),
            bounded_ratio_delta(current["bandwidth_mean"], previous["bandwidth_mean"]),
            abs(current["flatness_mean"] - previous["flatness_mean"]) * 4.0,
            abs(current["rms_db_mean"] - previous["rms_db_mean"]) / 10.0,
        )
        composite = clamp(harmonic_delta * 8.0 + timbre_delta * 8.0 + rhythm_delta * 3.0 + texture_delta * 3.0)
        adjacent_changes.append(
            {
                "from": previous["index"],
                "to": current["index"],
                "harmonic_delta": round(harmonic_delta, 3),
                "timbre_delta": round(timbre_delta, 3),
                "rhythm_delta": round(rhythm_delta, 3),
                "texture_delta": round(texture_delta, 3),
                "composite_change": round(composite, 3),
                "rule_changes": current["rule_changes"],
            }
        )

    transformed_returns: list[dict[str, Any]] = []
    copied_returns = 0
    for i, first in enumerate(profiles):
        for second in profiles[i + 2 :]:
            harmonic_similarity = cosine_similarity(first["chroma_vector"], second["chroma_vector"])
            timbre_similarity = cosine_similarity(first["mfcc_vector"], second["mfcc_vector"])
            rhythm_delta = bounded_ratio_delta(first["onset_rate"], second["onset_rate"])
            texture_delta = max(
                bounded_ratio_delta(first["centroid_mean"], second["centroid_mean"]),
                bounded_ratio_delta(first["bandwidth_mean"], second["bandwidth_mean"]),
                abs(first["flatness_mean"] - second["flatness_mean"]) * 4.0,
            )
            shared_identity = max(harmonic_similarity, timbre_similarity)
            transformed = shared_identity > 0.78 and (rhythm_delta > 0.18 or texture_delta > 0.18)
            copied = shared_identity > 0.94 and rhythm_delta < 0.08 and texture_delta < 0.08
            if transformed:
                transformed_returns.append(
                    {
                        "sections": [first["index"], second["index"]],
                        "harmonic_similarity": round(harmonic_similarity, 3),
                        "timbre_similarity": round(timbre_similarity, 3),
                        "rhythm_delta": round(rhythm_delta, 3),
                        "texture_delta": round(texture_delta, 3),
                    }
                )
            elif copied:
                copied_returns += 1

    section_count = max(len(profiles), 1)
    section_contrast_score = clamp(
        np.mean([change["composite_change"] for change in adjacent_changes]) if adjacent_changes else 0.0
    )
    cumulative_deltas: list[float] = []
    for idx, current in enumerate(profiles[1:], start=1):
        previous_sections = profiles[:idx]
        nearest = min(
            (
                vector_distance(current["chroma_vector"], previous["chroma_vector"]) * 7.0
                + vector_distance(current["mfcc_vector"], previous["mfcc_vector"]) * 7.0
                + bounded_ratio_delta(current["onset_rate"], previous["onset_rate"]) * 3.0
            )
            for previous in previous_sections
        )
        cumulative_deltas.append(clamp(nearest))
    cumulative_evolution_score = clamp(np.mean(cumulative_deltas) if cumulative_deltas else 0.0)

    rule_change_count = sum(len(profile["rule_changes"]) for profile in profiles)
    rule_change_score = clamp(rule_change_count / max(section_count - 1, 1) * 2.5)
    transformed_return_score = clamp(len(transformed_returns) / max(section_count / 2.0, 1.0) * 5.0 - copied_returns * 0.6)

    beat_grid = beat_this.get("beat_grid_stability") if beat_this and beat_this.get("available") else 0.0
    grid_deviation_values = [
        value
        for profile in profiles
        for value in (profile.get("beat_interval_cv"), profile.get("bar_interval_cv"))
        if value is not None
    ]
    raw_grid_deviation = clamp(np.mean(grid_deviation_values) * 20.0) if grid_deviation_values else 0.0
    controlled_grid_deviation_score = clamp(raw_grid_deviation * (0.4 + (beat_grid or 0.0) * 0.6))

    overall_score = clamp(
        section_contrast_score * 0.22
        + cumulative_evolution_score * 0.18
        + transformed_return_score * 0.22
        + controlled_grid_deviation_score * 0.16
        + rule_change_score * 0.22
    )

    public_profiles = []
    for profile in profiles:
        public_profiles.append(
            {
                "index": profile["index"],
                "start": round(profile["start"], 3),
                "end": round(profile["end"], 3),
                "duration": round(profile["duration"], 3),
                "onset_rate": round(profile["onset_rate"], 3),
                "rms_db_mean": round(profile["rms_db_mean"], 2),
                "centroid_mean": round(profile["centroid_mean"], 1),
                "bandwidth_mean": round(profile["bandwidth_mean"], 1),
                "flatness_mean": round(profile["flatness_mean"], 4),
                "beat_count": profile["beat_count"],
                "downbeat_count": profile["downbeat_count"],
                "median_beat_interval_seconds": profile["median_beat_interval_seconds"],
                "beat_interval_cv": profile["beat_interval_cv"],
                "bar_interval_cv": profile["bar_interval_cv"],
                "rule_changes": profile["rule_changes"],
            }
        )

    return {
        "section_contrast_score": round(section_contrast_score, 3),
        "cumulative_evolution_score": round(cumulative_evolution_score, 3),
        "transformed_return_score": round(transformed_return_score, 3),
        "controlled_grid_deviation_score": round(controlled_grid_deviation_score, 3),
        "rule_change_score": round(rule_change_score, 3),
        "overall_score": round(overall_score, 3),
        "transformed_returns": transformed_returns[:12],
        "copied_return_count": copied_returns,
        "adjacent_changes": adjacent_changes,
        "sections": public_profiles,
    }


def production_transform_metrics(text: TrackText) -> dict[str, Any]:
    fields = [
        text.tags.get("PRODUCTION", ""),
        text.tags.get("VOCALS", ""),
        text.tags.get("MOOD", ""),
        text.tags.get("GENRE", ""),
    ]
    joined = " ".join(fields).lower()
    words = re.findall(r"[a-z-]+", joined)
    matched_terms = sorted({word for word in words if word in TRANSFORMATION_TERMS})

    phrase_patterns = {
        "dropout": r"\bdrop\s*out\b|\bdropouts?\b",
        "density-change": r"\b(build|fuller|sparse|density|layered?|layers?)\b",
        "space-change": r"\b(reverb|delay|room|distant|close|wide|stereo|filtered?)\b",
        "texture-change": r"\b(glitch|granular|distortion|tape|wobble|noise|vocoder|texture)\b",
        "formal-pivot": r"\b(bridge|break|shift|switch|pivot|modulation|reprise|returns?)\b",
    }
    phrase_hits = sorted(name for name, pattern in phrase_patterns.items() if re.search(pattern, joined))
    score = clamp(len(matched_terms) * 0.65 + len(phrase_hits) * 0.9)
    return {
        "score": round(score, 3),
        "matched_terms": matched_terms,
        "phrase_hits": phrase_hits,
    }


def analyze_audio(path: Path, sample_rate: int = 22050, beat_file: Path | None = None) -> dict[str, Any]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    stereo, native_sr = sf.read(path, always_2d=True)
    duration = librosa.get_duration(y=y, sr=sr)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo_values = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo = safe_float(tempo_values[0])

    meter = pyln.Meter(native_sr)
    mono_native = np.mean(stereo, axis=1)
    loudness = safe_float(meter.integrated_loudness(mono_native), default=-99.0)
    peak = safe_float(np.max(np.abs(stereo)))
    clipping_ratio = float(np.mean(np.abs(stereo) >= 0.999))
    crest_factor = db(peak / (float(np.sqrt(np.mean(np.square(mono_native)))) + 1e-12))

    frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)
    features_for_boundaries = np.vstack([mfcc, chroma, rms.reshape(1, -1)])
    boundaries = pick_boundaries(features_for_boundaries, duration, frame_times)
    section_spans = [
        {
            "index": i + 1,
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "duration": boundaries[i + 1] - boundaries[i],
        }
        for i in range(len(boundaries) - 1)
    ]
    beat_this = parse_beat_this_file(beat_file)
    evolving_grammar = section_evolving_grammar_analysis(
        section_spans=section_spans,
        frame_times=frame_times,
        chroma=chroma,
        mfcc=mfcc,
        rms=rms,
        centroid=centroid,
        bandwidth=bandwidth,
        flatness=flatness,
        zcr=zcr,
        onset_env=onset_env,
        onset_frames=onset_frames,
        sr=sr,
        hop_length=hop_length,
        beat_this=beat_this,
    )

    if mfcc.shape[1] > 2:
        normed = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-9)
        sim = np.matmul(normed.T, normed)
        upper = sim[np.triu_indices_from(sim, k=8)]
        recurrence = safe_float(np.mean(upper > 0.92)) if upper.size else 0.0
    else:
        recurrence = 0.0

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(np.square(harmonic)))
    percussive_energy = float(np.mean(np.square(percussive)))
    hp_ratio = harmonic_energy / (percussive_energy + 1e-12)

    key, key_confidence = estimate_key(chroma)
    ffprobe = run_ffprobe(path)
    audio = mutagen.File(path)

    return {
        "path": str(path),
        "duration_seconds": duration,
        "duration": fmt_time(duration),
        "sample_rate": native_sr,
        "channels": int(stereo.shape[1]),
        "bitrate": getattr(audio.info, "bitrate", None) if audio and audio.info else None,
        "ffprobe": ffprobe,
        "tempo_bpm": tempo,
        "estimated_key": key,
        "key_confidence": key_confidence,
        "integrated_lufs": loudness,
        "peak_dbfs": db(peak),
        "crest_factor_db": crest_factor,
        "clipping_ratio": clipping_ratio,
        "rms_db_mean": db(float(np.mean(rms))),
        "rms_db_std": safe_float(np.std(20 * np.log10(rms + 1e-9))),
        "spectral_centroid_mean": safe_float(np.mean(centroid)),
        "spectral_centroid_std": safe_float(np.std(centroid)),
        "spectral_bandwidth_mean": safe_float(np.mean(bandwidth)),
        "rolloff_mean": safe_float(np.mean(rolloff)),
        "flatness_mean": safe_float(np.mean(flatness)),
        "zero_crossing_rate_mean": safe_float(np.mean(zcr)),
        "onset_count": int(len(onset_frames)),
        "onset_rate_per_second": len(onset_frames) / max(duration, 1.0),
        "harmonic_percussive_ratio": hp_ratio,
        "recurrence_ratio": recurrence,
        "beat_this": beat_this,
        "evolving_grammar": evolving_grammar,
        "detected_sections": section_spans,
    }


def lyric_metrics(text: TrackText) -> dict[str, Any]:
    lyrics = text.lyrics
    clean_lines = [line.strip() for line in lyrics.splitlines() if line.strip() and not re.match(r"^\[[^\]]+\]$", line.strip())]
    words = re.findall(r"[A-Za-z0-9']+", lyrics.lower())
    unique_words = set(words)
    repeated_lines = len(clean_lines) - len(set(line.lower() for line in clean_lines))
    end_words = []
    for line in clean_lines:
        found = re.findall(r"[A-Za-z']+", line.lower())
        if found:
            end_words.append(found[-1])
    rhyme_suffixes = [word[-3:] if len(word) >= 3 else word for word in end_words]
    rhyme_reuse = 0.0
    if rhyme_suffixes:
        rhyme_reuse = 1.0 - (len(set(rhyme_suffixes)) / len(rhyme_suffixes))

    reading_ease = flesch_reading_ease_rough(lyrics) if words else None
    production_transform = production_transform_metrics(text)

    return {
        "has_text": bool(text.raw),
        "title": text.tags.get("TITLE"),
        "declared_genre": text.tags.get("GENRE"),
        "declared_tempo": text.tags.get("TEMPO"),
        "declared_key": text.tags.get("KEY"),
        "declared_mood": text.tags.get("MOOD"),
        "production_notes": text.tags.get("PRODUCTION"),
        "vocal_notes": text.tags.get("VOCALS"),
        "section_count": len(text.lyric_sections),
        "section_names": [name for name, _ in text.lyric_sections],
        "line_count": len(clean_lines),
        "word_count": len(words),
        "unique_word_ratio": len(unique_words) / max(len(words), 1),
        "repeated_line_ratio": repeated_lines / max(len(clean_lines), 1),
        "rhyme_suffix_reuse": rhyme_reuse,
        "avg_words_per_line": len(words) / max(len(clean_lines), 1),
        "reading_ease": reading_ease,
        "production_transform": production_transform,
    }


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def score_framework(audio: dict[str, Any], lyrics: dict[str, Any], framework_name: str) -> dict[str, Any]:
    duration = audio["duration_seconds"]
    sections = audio["detected_sections"]
    section_count = len(sections)
    section_durations = [section["duration"] for section in sections]
    if section_durations:
        section_balance = 1.0 - min(1.0, np.std(section_durations) / (np.mean(section_durations) + 1e-9))
    else:
        section_balance = 0.0

    declared_tempo = lyrics.get("declared_tempo") or ""
    beat_this = audio.get("beat_this") or {}
    tempo_candidates = [audio["tempo_bpm"]]
    if beat_this.get("available"):
        tempo_candidates.extend(
            [
                beat_this.get("tempo_bpm", 0.0),
                beat_this.get("double_tempo_bpm", 0.0),
                beat_this.get("half_tempo_bpm", 0.0),
            ]
        )
    tempo_match_bonus = 0.0
    tempo_number = re.search(r"(\d+(?:\.\d+)?)", declared_tempo)
    if tempo_number:
        declared_bpm = float(tempo_number.group(1))
        diffs = [abs(candidate - declared_bpm) for candidate in tempo_candidates if candidate]
        diff = min(diffs) if diffs else abs(audio["tempo_bpm"] - declared_bpm)
        tempo_match_bonus = max(0.0, 1.0 - diff / 12.0)

    clipping_penalty = min(2.0, audio["clipping_ratio"] * 400)
    loudness_ok = 1.0 if -18 <= audio["integrated_lufs"] <= -7 else 0.4
    dynamic_score = clamp((audio["crest_factor_db"] - 5.0) / 1.2)
    recurrence_score = clamp(audio["recurrence_ratio"] * 35)
    onset_score = clamp(audio["onset_rate_per_second"] * 3.0)
    text_structure = clamp((lyrics.get("section_count") or 0) * 0.8)
    lyric_density = clamp((lyrics.get("word_count") or 0) / 55)
    lyric_repetition = clamp((lyrics.get("repeated_line_ratio") or 0) * 20)
    rhyme_score = clamp((lyrics.get("rhyme_suffix_reuse") or 0) * 18)
    spectral_width = clamp(audio["spectral_bandwidth_mean"] / 450)
    brightness_control = clamp(10 - abs(audio["spectral_centroid_mean"] - 2500) / 350)
    novelty = clamp((section_count - 2) * 0.7 + min(4.0, audio["spectral_centroid_std"] / 350) + onset_score * 0.15)
    eg_features = audio.get("evolving_grammar") or {}
    production_transform = lyrics.get("production_transform") or {}
    beat_grid = beat_this.get("beat_grid_stability") if beat_this.get("available") else None
    bar_grid = beat_this.get("bar_grid_stability") if beat_this.get("available") else None
    beat_stability_score = clamp(((beat_grid or 0.0) * 0.65 + (bar_grid or beat_grid or 0.0) * 0.35) * 10)
    downbeat_score = clamp((beat_this.get("downbeat_count") or 0) / max(duration / 20.0, 1.0)) if beat_this.get("available") else 0.0

    sc = clamp(
        2.0
        + section_count * 0.65
        + section_balance * 2.0
        + tempo_match_bonus * 1.2
        + beat_stability_score * 0.14
        + downbeat_score * 0.08
        - clipping_penalty
    )
    mi = clamp(
        2.0
        + recurrence_score * 0.35
        + lyric_repetition * 0.15
        + rhyme_score * 0.10
        + text_structure * 0.10
        + beat_stability_score * 0.08
    )
    bp = clamp(2.0 + loudness_ok * 1.7 + dynamic_score * 0.28 + brightness_control * 0.25 + spectral_width * 0.16 - clipping_penalty)
    eg = clamp(
        1.2
        + novelty * 0.10
        + text_structure * 0.04
        + lyric_density * 0.03
        + eg_features.get("section_contrast_score", 0.0) * 0.20
        + eg_features.get("cumulative_evolution_score", 0.0) * 0.18
        + eg_features.get("transformed_return_score", 0.0) * 0.18
        + eg_features.get("controlled_grid_deviation_score", 0.0) * 0.14
        + eg_features.get("rule_change_score", 0.0) * 0.18
        + production_transform.get("score", 0.0) * 0.16
    )
    cd = clamp((sc + mi + bp) / 3 * 0.6 + lyric_density * 0.18 + min(2.0, section_count * 0.18))

    framework_lower = framework_name.lower()
    if any(term in framework_lower for term in ("rap", "hip-hop")):
        mi = clamp(mi + rhyme_score * 0.12 + onset_score * 0.08 + beat_stability_score * 0.08)
        eg = clamp(
            eg
            + min(0.5, audio["onset_rate_per_second"] * 0.14)
            + beat_stability_score * 0.03
            + eg_features.get("controlled_grid_deviation_score", 0.0) * 0.05
        )
    elif any(term in framework_lower for term in ("metal", "prog", "djent")):
        mi = clamp(mi + onset_score * 0.12 + spectral_width * 0.08 + beat_stability_score * 0.06)
        bp = clamp(bp + min(1.0, audio["harmonic_percussive_ratio"] * 0.2))
        eg = clamp(eg + eg_features.get("rule_change_score", 0.0) * 0.08 + eg_features.get("controlled_grid_deviation_score", 0.0) * 0.08)
    elif "indie" in framework_lower:
        cd = clamp(cd + lyric_density * 0.06 + lyric_repetition * 0.08)
        eg = clamp(eg + production_transform.get("score", 0.0) * 0.05 + eg_features.get("transformed_return_score", 0.0) * 0.05)
    else:
        bp = clamp(bp + min(1.0, audio["harmonic_percussive_ratio"] * 0.12))

    axis_avg = (sc + mi + bp + eg + cd) / 5
    cdpd = clamp((mi * 0.45 + sc * 0.35 + cd * 0.20) / 10, 0, 1)
    nge = clamp((eg * 0.50 + novelty * 0.18 + eg_features.get("overall_score", 0.0) * 0.32) / 10, 0, 1)
    hmii = int(
        round(
            clamp(
                1.5 + spectral_width * 0.18 + onset_score * 0.12 + recurrence_score * 0.12 + beat_stability_score * 0.06,
                1,
                8,
            )
        )
    )

    rung = estimate_rung(axis_avg, cdpd, nge, hmii, clipping_penalty, section_count)
    confidence = clamp(4.5 + bool(lyrics.get("has_text")) * 1.2 + min(1.0, duration / 180) + min(1.0, section_count / 6), 0, 8)

    return {
        "axes": {
            "SC_structural_coherence": round(sc, 2),
            "MI_motivic_integration": round(mi, 2),
            "BP_beauty_spatial_poise": round(bp, 2),
            "EG_evolving_grammar": round(eg, 2),
            "CD_carry_depth": round(cd, 2),
        },
        "core_metrics": {
            "CDPD": round(cdpd, 3),
            "NGE": round(nge, 3),
            "HMII_peak_estimate": hmii,
        },
        "eg_evidence": {
            "section_contrast_score": eg_features.get("section_contrast_score", 0.0),
            "cumulative_evolution_score": eg_features.get("cumulative_evolution_score", 0.0),
            "transformed_return_score": eg_features.get("transformed_return_score", 0.0),
            "controlled_grid_deviation_score": eg_features.get("controlled_grid_deviation_score", 0.0),
            "rule_change_score": eg_features.get("rule_change_score", 0.0),
            "production_transform_score": production_transform.get("score", 0.0),
            "overall_score": eg_features.get("overall_score", 0.0),
        },
        "rung_estimate": rung,
        "confidence_0_10": round(confidence, 2),
        "rationale": build_rationale(audio, lyrics, section_balance, tempo_match_bonus, clipping_penalty),
    }


def estimate_rung(axis_avg: float, cdpd: float, nge: float, hmii: int, clipping_penalty: float, section_count: int) -> dict[str, Any]:
    if clipping_penalty > 1.5 or section_count <= 1:
        number = 4
    elif axis_avg < 4:
        number = 5
    elif axis_avg < 5:
        number = 6
    elif axis_avg < 5.8:
        number = 7
    elif axis_avg < 6.6:
        number = 8
    elif axis_avg < 7.4:
        number = 9
    elif axis_avg < 8.1:
        number = 10
    elif nge >= 0.50 and cdpd >= 0.60:
        number = 11
    else:
        number = 10

    if number >= 11 and hmii >= 5 and nge >= 0.55:
        number = 12
    if number >= 12 and cdpd >= 0.65 and nge >= 0.58:
        number = 13

    labels = {
        4: "Half-Form",
        5: "Draft",
        6: "Serviceable",
        7: "Competent Template",
        8: "Strong Craft, Some Signature",
        9: "Near-Perfection, No Rupture",
        10: "Technical Perfection, No Rupture",
        11: "First Rupture",
        12: "Recursion Threshold",
        13: "Inhabited Form",
    }
    return {
        "number": number,
        "label": labels.get(number, "Unmapped"),
        "note": "Heuristic ceiling is conservative; tiers above 13 need close listening and stronger evidence of new grammar.",
    }


def compact_audio_for_llm(audio: dict[str, Any]) -> dict[str, Any]:
    beat_this = dict(audio.get("beat_this") or {})
    beat_this.pop("beats", None)
    beat_this.pop("beat_positions", None)
    beat_this.pop("downbeats", None)
    ffprobe = audio.get("ffprobe") or {}
    return {
        "path": audio["path"],
        "duration": audio["duration"],
        "tempo_bpm": round(audio["tempo_bpm"], 2),
        "estimated_key": audio["estimated_key"],
        "key_confidence": round(audio["key_confidence"], 3),
        "integrated_lufs": round(audio["integrated_lufs"], 2),
        "peak_dbfs": round(audio["peak_dbfs"], 2),
        "crest_factor_db": round(audio["crest_factor_db"], 2),
        "clipping_ratio": audio["clipping_ratio"],
        "spectral_centroid_mean": round(audio["spectral_centroid_mean"], 2),
        "spectral_centroid_std": round(audio["spectral_centroid_std"], 2),
        "spectral_bandwidth_mean": round(audio["spectral_bandwidth_mean"], 2),
        "flatness_mean": round(audio["flatness_mean"], 5),
        "onset_rate_per_second": round(audio["onset_rate_per_second"], 3),
        "harmonic_percussive_ratio": round(audio["harmonic_percussive_ratio"], 3),
        "recurrence_ratio": round(audio["recurrence_ratio"], 3),
        "detected_sections": audio["detected_sections"],
        "beat_this": beat_this,
        "evolving_grammar": audio.get("evolving_grammar"),
        "ffprobe_streams": ffprobe.get("streams", [])[:3],
    }


def build_ollama_prompt(
    audio: dict[str, Any],
    text: dict[str, Any],
    scoring: dict[str, Any],
    framework_raw: str,
    track_text: TrackText,
    max_framework_chars: int = 9000,
    max_lyrics_chars: int = 6500,
) -> list[dict[str, str]]:
    payload = {
        "task": "Adjust the framework scoring using the measured audio/text evidence. Do not invent unmeasured facts.",
        "base_scoring": scoring,
        "audio_features": compact_audio_for_llm(audio),
        "text_features": text,
        "lyrics_and_notes_excerpt": truncate_text(track_text.raw, max_lyrics_chars),
        "framework_excerpt": truncate_text(framework_raw, max_framework_chars),
        "rules": [
            "Return valid JSON only.",
            "Use the base Python scores as the anchor; adjust only when the framework/text/evidence justifies it.",
            "Axis scores must be floats from 0 to 10.",
            "Prefer deltas within +/-1.5 of base axis scores. Larger moves require explicit evidence.",
            "Core metrics CDPD and NGE must be 0..1; HMII_peak_estimate must be an integer 1..10.",
            "Give concise evidence-backed reasons, not generic praise.",
            "Flag uncertainty where audio-only evidence is weak or section detection looks suspicious.",
        ],
        "required_json_shape": {
            "axes": {
                "SC_structural_coherence": "float 0..10",
                "MI_motivic_integration": "float 0..10",
                "BP_beauty_spatial_poise": "float 0..10",
                "EG_evolving_grammar": "float 0..10",
                "CD_carry_depth": "float 0..10",
            },
            "core_metrics": {
                "CDPD": "float 0..1",
                "NGE": "float 0..1",
                "HMII_peak_estimate": "integer 1..10",
            },
            "rung_estimate": {"number": "integer 1..23", "label": "string", "note": "string"},
            "confidence_0_10": "float 0..10",
            "adjustments": {
                "SC_structural_coherence": {"delta": "float", "reason": "string"},
                "MI_motivic_integration": {"delta": "float", "reason": "string"},
                "BP_beauty_spatial_poise": {"delta": "float", "reason": "string"},
                "EG_evolving_grammar": {"delta": "float", "reason": "string"},
                "CD_carry_depth": {"delta": "float", "reason": "string"},
            },
            "rationale": ["string"],
            "review_flags": ["string"],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a cautious music-analysis rubric judge. You receive measured audio features, "
                "lyrics/production notes, and a framework excerpt. You cannot hear the audio directly. "
                "Calibrate scores from evidence and preserve uncertainty."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def normalize_llm_scoring(raw: dict[str, Any], base: dict[str, Any], max_axis_delta: float) -> dict[str, Any]:
    base_axes = base["axes"]
    raw_axes = raw.get("axes") or {}
    axes: dict[str, float] = {}
    adjustments: dict[str, dict[str, Any]] = {}
    for axis, base_value in base_axes.items():
        proposed = safe_float(raw_axes.get(axis), base_value)
        bounded = clamp(proposed, base_value - max_axis_delta, base_value + max_axis_delta)
        bounded = round(clamp(bounded), 2)
        axes[axis] = bounded

        raw_adjustment = (raw.get("adjustments") or {}).get(axis) or {}
        reason = str(raw_adjustment.get("reason") or "LLM adjusted from measured evidence.").strip()
        adjustments[axis] = {
            "delta": round(bounded - base_value, 2),
            "reason": truncate_text(reason, 500),
        }

    raw_core = raw.get("core_metrics") or {}
    base_core = base["core_metrics"]
    core_metrics = {
        "CDPD": round(clamp(safe_float(raw_core.get("CDPD"), base_core["CDPD"]), 0.0, 1.0), 3),
        "NGE": round(clamp(safe_float(raw_core.get("NGE"), base_core["NGE"]), 0.0, 1.0), 3),
        "HMII_peak_estimate": int(round(clamp(safe_float(raw_core.get("HMII_peak_estimate"), base_core["HMII_peak_estimate"]), 1, 10))),
    }

    raw_rung = raw.get("rung_estimate") or {}
    base_rung = base["rung_estimate"]
    rung_number = int(round(clamp(safe_float(raw_rung.get("number"), base_rung["number"]), 1, 23)))
    rung = {
        "number": rung_number,
        "label": str(raw_rung.get("label") or base_rung.get("label") or "LLM-adjusted rung"),
        "note": truncate_text(str(raw_rung.get("note") or "LLM-adjusted estimate from measured evidence."), 500),
    }

    rationale = raw.get("rationale") if isinstance(raw.get("rationale"), list) else []
    review_flags = raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else []
    return {
        "axes": axes,
        "core_metrics": core_metrics,
        "rung_estimate": rung,
        "confidence_0_10": round(clamp(safe_float(raw.get("confidence_0_10"), base["confidence_0_10"])), 2),
        "adjustments": adjustments,
        "rationale": [truncate_text(str(item), 700) for item in rationale[:8]],
        "review_flags": [truncate_text(str(item), 500) for item in review_flags[:8]],
        "model_note": f"Axis deltas bounded to +/-{max_axis_delta} around Python base scores.",
    }


def call_ollama_adjuster(
    model: str,
    ollama_url: str,
    audio: dict[str, Any],
    text: dict[str, Any],
    scoring: dict[str, Any],
    framework_raw: str,
    track_text: TrackText,
    timeout: float,
    num_ctx: int,
    max_axis_delta: float,
) -> dict[str, Any]:
    messages = build_ollama_prompt(audio, text, scoring, framework_raw, track_text)
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": num_ctx,
        },
    }
    endpoint = ollama_url.rstrip("/") + "/api/chat"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "model": model,
            "url": ollama_url,
            "error": str(exc),
        }

    content = (response_body.get("message") or {}).get("content", "")
    try:
        raw = extract_json_object(content)
        normalized = normalize_llm_scoring(raw, scoring, max_axis_delta=max_axis_delta)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "model": model,
            "url": ollama_url,
            "error": f"Could not parse Ollama JSON response: {exc}",
            "raw_content": truncate_text(content, 4000),
        }

    normalized.update(
        {
            "available": True,
            "model": model,
            "url": ollama_url,
            "created_at": response_body.get("created_at"),
            "eval_count": response_body.get("eval_count"),
            "eval_duration": response_body.get("eval_duration"),
        }
    )
    return normalized


def build_rationale(
    audio: dict[str, Any],
    lyrics: dict[str, Any],
    section_balance: float,
    tempo_match_bonus: float,
    clipping_penalty: float,
) -> list[str]:
    notes = [
        f"Detected {len(audio['detected_sections'])} audio sections over {audio['duration']} with section-balance score {section_balance:.2f}.",
        f"Tempo estimate is {audio['tempo_bpm']:.1f} BPM; declared tempo alignment contributes {tempo_match_bonus:.2f}.",
        f"Integrated loudness is {audio['integrated_lufs']:.1f} LUFS, peak is {audio['peak_dbfs']:.1f} dBFS, clipping ratio is {audio['clipping_ratio']:.5f}.",
        f"Estimated key is {audio['estimated_key']} with confidence {audio['key_confidence']:.2f}.",
        f"Audio recurrence ratio is {audio['recurrence_ratio']:.3f}; this is a rough proxy for repeated musical cells, not true motif recognition.",
    ]
    beat_this = audio.get("beat_this") or {}
    if beat_this.get("available"):
        notes.append(
            f"beat_this detected {beat_this['beat_count']} beats and {beat_this['downbeat_count']} downbeats; "
            f"beat-grid stability is {beat_this['beat_grid_stability']:.2f}."
        )
    eg_features = audio.get("evolving_grammar") or {}
    if eg_features:
        notes.append(
            "EG evidence: "
            f"section contrast {eg_features.get('section_contrast_score', 0.0):.2f}, "
            f"cumulative evolution {eg_features.get('cumulative_evolution_score', 0.0):.2f}, "
            f"transformed returns {eg_features.get('transformed_return_score', 0.0):.2f}, "
            f"grid deviation {eg_features.get('controlled_grid_deviation_score', 0.0):.2f}, "
            f"rule changes {eg_features.get('rule_change_score', 0.0):.2f}."
        )
    if lyrics.get("has_text"):
        notes.append(
            f"Companion text has {lyrics['section_count']} lyric/production sections, {lyrics['word_count']} words, "
            f"line repetition {lyrics['repeated_line_ratio']:.2f}, rhyme-suffix reuse {lyrics['rhyme_suffix_reuse']:.2f}."
        )
        production_transform = lyrics.get("production_transform") or {}
        if production_transform.get("score", 0.0):
            notes.append(
                f"Production/vocal notes contain transformation cues "
                f"({', '.join(production_transform.get('phrase_hits') or production_transform.get('matched_terms') or [])}); "
                f"intent score {production_transform['score']:.2f}."
            )
    if clipping_penalty:
        notes.append("Clipping/headroom risk reduced the craft and spatial-poise estimates.")
    return notes


def make_markdown(report: dict[str, Any]) -> str:
    audio = report["audio"]
    lyrics = report["text"]
    scoring = report["framework_scoring"]
    llm_scoring = report.get("llm_adjusted_scoring") or {}
    displayed_scoring = llm_scoring if llm_scoring.get("available") else scoring
    framework = report["framework"]

    lines = [
        f"# Track Analysis: {lyrics.get('title') or Path(audio['path']).stem}",
        "",
        f"- Audio: `{audio['path']}`",
        f"- Text: `{report['text_path']}`" if report.get("text_path") else "- Text: not provided",
        f"- Framework: `{framework.get('path')}`" if framework.get("path") else "- Framework: not selected",
        f"- Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Duration: {audio['duration']}",
        f"- Tempo: {audio['tempo_bpm']:.1f} BPM" + (f" (declared: {lyrics.get('declared_tempo')})" if lyrics.get("declared_tempo") else ""),
        f"- Key: {audio['estimated_key']} ({audio['key_confidence']:.2f} confidence)" + (f" (declared: {lyrics.get('declared_key')})" if lyrics.get("declared_key") else ""),
        f"- Loudness: {audio['integrated_lufs']:.1f} LUFS; peak {audio['peak_dbfs']:.1f} dBFS; crest {audio['crest_factor_db']:.1f} dB",
        f"- Detected sections: {len(audio['detected_sections'])}",
        f"- Rung estimate: {displayed_scoring['rung_estimate']['number']} - {displayed_scoring['rung_estimate']['label']}",
        f"- Confidence: {displayed_scoring['confidence_0_10']}/10",
    ]
    if llm_scoring.get("available"):
        lines.append(f"- LLM adjustment: `{llm_scoring['model']}` via Ollama")
    lines.extend(["", "## Framework Scores", ""])
    if llm_scoring.get("available"):
        lines.extend(["### LLM-Adjusted", ""])
        for name, value in llm_scoring["axes"].items():
            base_value = scoring["axes"].get(name, value)
            lines.append(f"- {name}: {value}/10 ({value - base_value:+.2f})")
        for name, value in llm_scoring["core_metrics"].items():
            lines.append(f"- {name}: {value}")
        lines.extend(["", "### Python Base", ""])
    for name, value in scoring["axes"].items():
        lines.append(f"- {name}: {value}/10")
    for name, value in scoring["core_metrics"].items():
        lines.append(f"- {name}: {value}")

    if llm_scoring.get("available"):
        lines.extend(["", "## LLM Adjustment Rationale", ""])
        for axis, item in llm_scoring.get("adjustments", {}).items():
            lines.append(f"- {axis}: {item['delta']:+.2f}. {item['reason']}")
        if llm_scoring.get("rationale"):
            lines.append("")
            for note in llm_scoring["rationale"]:
                lines.append(f"- {note}")
        if llm_scoring.get("review_flags"):
            lines.extend(["", "Review flags:"])
            for flag in llm_scoring["review_flags"]:
                lines.append(f"- {flag}")
    elif llm_scoring:
        lines.extend(["", "## LLM Adjustment", "", f"- Unavailable: {llm_scoring.get('error', 'unknown error')}"])

    beat_this = audio.get("beat_this") or {}
    lines.extend(["", "## Beat Grid", ""])
    if beat_this.get("available"):
        lines.extend(
            [
                f"- Source: `{beat_this['path']}`",
                f"- Beats: {beat_this['beat_count']}; downbeats: {beat_this['downbeat_count']}",
                f"- beat_this tempo: {beat_this['tempo_bpm']:.1f} BPM; double-time candidate: {beat_this['double_tempo_bpm']:.1f} BPM",
                f"- Beat-grid stability: {beat_this['beat_grid_stability']:.2f}",
            ]
        )
        if beat_this.get("bar_grid_stability") is not None:
            lines.append(f"- Bar-grid stability: {beat_this['bar_grid_stability']:.2f}")
    else:
        lines.append("- No beat_this beat file was provided.")

    eg_features = audio.get("evolving_grammar") or {}
    lines.extend(["", "## Evolving Grammar Signals", ""])
    if eg_features:
        evidence = scoring.get("eg_evidence") or {}
        lines.extend(
            [
                f"- Section contrast: {evidence.get('section_contrast_score', 0.0):.2f}/10",
                f"- Cumulative evolution: {evidence.get('cumulative_evolution_score', 0.0):.2f}/10",
                f"- Transformed returns: {evidence.get('transformed_return_score', 0.0):.2f}/10",
                f"- Controlled grid deviation: {evidence.get('controlled_grid_deviation_score', 0.0):.2f}/10",
                f"- Rule changes: {evidence.get('rule_change_score', 0.0):.2f}/10",
                f"- Production/arrangement intent: {evidence.get('production_transform_score', 0.0):.2f}/10",
            ]
        )
        transformed = eg_features.get("transformed_returns") or []
        if transformed:
            pairs = ", ".join(f"{item['sections'][0]}->{item['sections'][1]}" for item in transformed[:6])
            lines.append(f"- Transformed-return pairs: {pairs}")
        rule_summaries = [
            f"S{section['index']}: {', '.join(section['rule_changes'])}"
            for section in eg_features.get("sections", [])
            if section.get("rule_changes")
        ]
        if rule_summaries:
            lines.append(f"- Section rule changes: {'; '.join(rule_summaries[:8])}")
    else:
        lines.append("- No evolving-grammar feature block was computed.")

    lines.extend(["", "## Section Map", ""])
    for section in audio["detected_sections"]:
        lines.append(
            f"- Section {section['index']}: {fmt_time(section['start'])} - {fmt_time(section['end'])} "
            f"({section['duration']:.1f}s)"
        )

    lines.extend(["", "## Text Signals", ""])
    if lyrics.get("has_text"):
        lines.extend(
            [
                f"- Genre: {lyrics.get('declared_genre') or 'not declared'}",
                f"- Mood: {lyrics.get('declared_mood') or 'not declared'}",
                f"- Sections: {', '.join(lyrics.get('section_names') or [])}",
                f"- Words: {lyrics['word_count']}; lines: {lyrics['line_count']}; unique-word ratio: {lyrics['unique_word_ratio']:.2f}",
                f"- Repeated-line ratio: {lyrics['repeated_line_ratio']:.2f}; rhyme-suffix reuse: {lyrics['rhyme_suffix_reuse']:.2f}",
            ]
        )
    else:
        lines.append("- No companion text was available.")

    lines.extend(["", "## Rationale", ""])
    for note in scoring["rationale"]:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Section boundaries, key, motif recurrence, HMII, and rung placement are heuristic.",
            "- EG signals are computed from section-level harmony, timbre, rhythm, beat-grid, and production-note proxies; final SC/MI/BP/EG/CD judgment still benefits from close listening.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.analysis.json"
    md_path = output_dir / f"{stem}.analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(make_markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an MP3 and apply a local music-analysis framework heuristically.")
    parser.add_argument("audio", type=Path, help="Path to an MP3/audio file.")
    parser.add_argument("--text", type=Path, help="Companion lyrics/production-notes text file. Guessed by default for album/audio layouts.")
    parser.add_argument("--framework", type=Path, help="Framework file from analyzer/. Guessed from declared genre by default.")
    parser.add_argument("--beat-file", type=Path, help="Precomputed beat_this .beats file to use for beat/downbeat scoring.")
    parser.add_argument("--ollama-model", help="Optional Ollama model for a bounded LLM scoring adjustment, e.g. qwen3:8b.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--ollama-timeout", type=float, default=240.0, help="Seconds to wait for the Ollama adjustment.")
    parser.add_argument("--ollama-num-ctx", type=int, default=16384, help="Ollama context window for scoring adjustment.")
    parser.add_argument("--llm-max-axis-delta", type=float, default=1.5, help="Maximum LLM adjustment per axis around the Python score.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output"), help="Directory for Markdown and JSON reports.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Analysis sample rate for librosa.")
    parser.add_argument("--stdout", action="store_true", help="Print Markdown report to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = args.audio
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    text_path = args.text or guess_companion_text(audio_path)
    track_text = read_track_text(text_path)
    framework_path = choose_framework(audio_path, track_text, args.framework)
    framework_raw = framework_path.read_text(encoding="utf-8") if framework_path and framework_path.exists() else ""

    audio = analyze_audio(audio_path, sample_rate=args.sample_rate, beat_file=args.beat_file)
    text = lyric_metrics(track_text)
    framework_name = framework_path.name if framework_path else "Unselected framework"
    scoring = score_framework(audio, text, framework_name)
    llm_scoring = None
    if args.ollama_model:
        llm_scoring = call_ollama_adjuster(
            model=args.ollama_model,
            ollama_url=args.ollama_url,
            audio=audio,
            text=text,
            scoring=scoring,
            framework_raw=framework_raw,
            track_text=track_text,
            timeout=args.ollama_timeout,
            num_ctx=args.ollama_num_ctx,
            max_axis_delta=args.llm_max_axis_delta,
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audio": audio,
        "text_path": str(text_path) if text_path else None,
        "text": text,
        "framework": {
            "path": str(framework_path) if framework_path else None,
            "name": framework_name,
            "characters_loaded": len(framework_raw),
        },
        "framework_scoring": scoring,
    }
    if llm_scoring is not None:
        report["llm_adjusted_scoring"] = llm_scoring

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", audio_path.with_suffix("").as_posix()).strip("_")
    json_path, md_path = write_outputs(report, args.output_dir, stem)
    if args.stdout:
        print(make_markdown(report))
    else:
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
