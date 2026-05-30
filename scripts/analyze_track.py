#!/usr/bin/env python3
"""Analyze an MP3 and produce a framework-guided music report."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
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


def analyze_audio(path: Path, sample_rate: int = 22050) -> dict[str, Any]:
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
    tempo_match_bonus = 0.0
    tempo_number = re.search(r"(\d+(?:\.\d+)?)", declared_tempo)
    if tempo_number:
        diff = abs(audio["tempo_bpm"] - float(tempo_number.group(1)))
        diff = min(diff, abs(audio["tempo_bpm"] * 2 - float(tempo_number.group(1))))
        diff = min(diff, abs(audio["tempo_bpm"] / 2 - float(tempo_number.group(1))))
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

    sc = clamp(2.0 + section_count * 0.65 + section_balance * 2.0 + tempo_match_bonus * 1.2 - clipping_penalty)
    mi = clamp(2.0 + recurrence_score * 0.35 + lyric_repetition * 0.15 + rhyme_score * 0.10 + text_structure * 0.10)
    bp = clamp(2.0 + loudness_ok * 1.7 + dynamic_score * 0.28 + brightness_control * 0.25 + spectral_width * 0.16 - clipping_penalty)
    eg = clamp(1.5 + novelty * 0.25 + text_structure * 0.08 + lyric_density * 0.04)
    cd = clamp((sc + mi + bp) / 3 * 0.6 + lyric_density * 0.18 + min(2.0, section_count * 0.18))

    framework_lower = framework_name.lower()
    if any(term in framework_lower for term in ("rap", "hip-hop")):
        mi = clamp(mi + rhyme_score * 0.12 + onset_score * 0.08)
        eg = clamp(eg + min(0.6, audio["onset_rate_per_second"] * 0.18))
    elif any(term in framework_lower for term in ("metal", "prog", "djent")):
        mi = clamp(mi + onset_score * 0.12 + spectral_width * 0.08)
        bp = clamp(bp + min(1.0, audio["harmonic_percussive_ratio"] * 0.2))
    elif "indie" in framework_lower:
        cd = clamp(cd + lyric_density * 0.06 + lyric_repetition * 0.08)
    else:
        bp = clamp(bp + min(1.0, audio["harmonic_percussive_ratio"] * 0.12))

    axis_avg = (sc + mi + bp + eg + cd) / 5
    cdpd = clamp((mi * 0.45 + sc * 0.35 + cd * 0.20) / 10, 0, 1)
    nge = clamp((eg * 0.65 + novelty * 0.35) / 10, 0, 1)
    hmii = int(round(clamp(1.5 + spectral_width * 0.18 + onset_score * 0.12 + recurrence_score * 0.12, 1, 8)))

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
    if lyrics.get("has_text"):
        notes.append(
            f"Companion text has {lyrics['section_count']} lyric/production sections, {lyrics['word_count']} words, "
            f"line repetition {lyrics['repeated_line_ratio']:.2f}, rhyme-suffix reuse {lyrics['rhyme_suffix_reuse']:.2f}."
        )
    if clipping_penalty:
        notes.append("Clipping/headroom risk reduced the craft and spatial-poise estimates.")
    return notes


def make_markdown(report: dict[str, Any]) -> str:
    audio = report["audio"]
    lyrics = report["text"]
    scoring = report["framework_scoring"]
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
        f"- Rung estimate: {scoring['rung_estimate']['number']} - {scoring['rung_estimate']['label']}",
        f"- Confidence: {scoring['confidence_0_10']}/10",
        "",
        "## Framework Scores",
        "",
    ]
    for name, value in scoring["axes"].items():
        lines.append(f"- {name}: {value}/10")
    for name, value in scoring["core_metrics"].items():
        lines.append(f"- {name}: {value}")

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
            "- The script can measure audio and text signals, but final SC/MI/BP/EG/CD judgment still benefits from close listening.",
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

    audio = analyze_audio(audio_path, sample_rate=args.sample_rate)
    text = lyric_metrics(track_text)
    framework_name = framework_path.name if framework_path else "Unselected framework"
    scoring = score_framework(audio, text, framework_name)

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
