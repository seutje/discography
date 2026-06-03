#!/usr/bin/env python3
"""Local web UI and callback receiver for the Suno iteration pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import traceback
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import analyze_track
import suno_iterate


ROOT = Path.cwd()
OUTPUT_ROOT = ROOT / "suno-runs" / "web"
FRONTEND_PATH = Path(__file__).with_name("suno_frontend.html")
STATE_LOCK = threading.RLock()
CALLBACK_EVENTS: dict[tuple[str, int], threading.Event] = {}
WAV_CALLBACK_EVENTS: dict[tuple[str, int, int], threading.Event] = {}
WORKERS: dict[str, threading.Thread] = {}
CALLBACK_TOKEN = ""
ALLOW_REMOTE_DASHBOARD = False


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id


def job_state_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def load_job(job_id: str) -> dict[str, Any]:
    return read_json(job_state_path(job_id), {})


def save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = now()
    write_json(job_state_path(job["id"]), job)


def patch_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with STATE_LOCK:
        job = load_job(job_id)
        job.update(updates)
        save_job(job)
        return job


def add_log(job_id: str, message: str) -> None:
    with STATE_LOCK:
        job = load_job(job_id)
        job.setdefault("log", []).append({"time": now(), "message": message})
        save_job(job)


def list_song_files() -> list[dict[str, str]]:
    songs = []
    for path in ROOT.rglob("*.txt"):
        if any(part in {".git", ".venv", ".cache", "analyzer", "analysis-output", "suno-runs"} for part in path.parts):
            continue
        if path.name == "requirements.txt":
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            spec = suno_iterate.parse_track_text(path)
            title = spec.title
        except Exception:
            title = path.stem
        songs.append({"path": rel, "title": title})
    return sorted(songs, key=lambda item: item["path"].lower())


def text_analysis(relative_path: str) -> dict[str, Any]:
    text_path = safe_project_path(relative_path)
    if not text_path.exists() or text_path.suffix.lower() != ".txt":
        raise FileNotFoundError(f"text file not found: {relative_path}")
    track_text = analyze_track.read_track_text(text_path)
    metrics = analyze_track.lyric_metrics(track_text)
    spec = suno_iterate.parse_track_text(text_path)
    style = suno_iterate.render_style_prompt(spec)
    prompt = suno_iterate.truncate(spec.lyrics, suno_iterate.PROMPT_LIMIT)
    pseudo_audio_path = text_path.parent / "audio" / f"{text_path.stem}.mp3"
    framework = analyze_track.choose_framework(pseudo_audio_path, track_text, None)
    missing = [field for field in ("TITLE", "GENRE", "MOOD", "TEMPO", "KEY", "VOCALS", "PRODUCTION") if not track_text.tags.get(field)]
    warnings: list[str] = []
    if len(spec.lyrics) > suno_iterate.PROMPT_LIMIT:
        warnings.append(f"Lyrics are {len(spec.lyrics)} characters and will be truncated to {suno_iterate.PROMPT_LIMIT}.")
    elif len(spec.lyrics) > suno_iterate.PROMPT_LIMIT * 0.95:
        warnings.append(f"Lyrics are close to Suno's limit: {len(spec.lyrics)}/{suno_iterate.PROMPT_LIMIT}.")
    if len(style) > suno_iterate.STYLE_LIMIT:
        warnings.append(f"Style prompt is {len(style)} characters and will be truncated to {suno_iterate.STYLE_LIMIT}.")
    elif len(style) > suno_iterate.STYLE_LIMIT * 0.95:
        warnings.append(f"Style prompt is close to Suno's limit: {len(style)}/{suno_iterate.STYLE_LIMIT}.")
    if missing:
        warnings.append("Missing metadata fields: " + ", ".join(missing) + ".")
    if metrics["section_count"] < 4:
        warnings.append("Few lyric sections detected; Suno may produce a less structured arrangement.")
    if metrics["word_count"] < 120:
        warnings.append("Low word count; generation may feel underwritten for this album's spoken-word style.")

    return {
        "path": relative_path,
        "title": metrics.get("title") or spec.title,
        "framework": str(framework) if framework else None,
        "suno": {
            "prompt_chars": len(spec.lyrics),
            "prompt_limit": suno_iterate.PROMPT_LIMIT,
            "prompt_will_truncate": len(spec.lyrics) > suno_iterate.PROMPT_LIMIT,
            "style_chars": len(style),
            "style_limit": suno_iterate.STYLE_LIMIT,
            "style_will_truncate": len(style) > suno_iterate.STYLE_LIMIT,
            "title_chars": len(spec.title),
            "title_limit": suno_iterate.TITLE_LIMIT,
        },
        "metrics": metrics,
        "missing_metadata": missing,
        "warnings": warnings,
        "album_audio_target": pseudo_audio_path.relative_to(ROOT).as_posix(),
        "prompt_preview": prompt[:800],
        "style_preview": style,
    }


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def bounded_float(value: Any, fallback: float, low: float = 0.0, high: float = 1.0) -> float:
    result = finite_float(value)
    if result is None:
        result = fallback
    if result < low or result > high:
        raise ValueError(f"value must be between {low:g} and {high:g}")
    return round(result, 2)


def normalize_vocal_gender(value: Any) -> str | None:
    gender = str(value or "").strip().lower()
    if not gender:
        return None
    if gender not in {"m", "f"}:
        raise ValueError("vocal_gender must be empty, m, or f")
    return gender


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def track_number_from_name(name: str) -> int | None:
    stem = Path(name).stem
    prefix = stem.split(" - ", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def album_from_audio_path(audio_path: str) -> str:
    parts = Path(audio_path).parts
    if "audio" in parts:
        index = parts.index("audio")
        if index > 0:
            return parts[index - 1]
    if len(parts) > 1:
        return parts[0]
    return "Unsorted"


def relative_project_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def audio_path_for_text(text_path: Path) -> Path | None:
    audio_dir = text_path.parent / "audio"
    for suffix in suno_iterate.AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{text_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def summarize_analysis_report(report_path: Path) -> dict[str, Any] | None:
    report = read_json(report_path, {})
    audio = report.get("audio") or {}
    text = report.get("text") or {}
    framework = report.get("framework") or {}
    base_scoring = report.get("framework_scoring") or {}
    llm_scoring = report.get("llm_adjusted_scoring") or {}
    scoring = llm_scoring if llm_scoring.get("available") else base_scoring
    axes = {key: finite_float(value) for key, value in (scoring.get("axes") or {}).items()}
    axes = {key: value for key, value in axes.items() if value is not None}
    core_metrics = {key: finite_float(value) for key, value in (scoring.get("core_metrics") or {}).items()}
    core_metrics = {key: value for key, value in core_metrics.items() if value is not None}
    if not axes and not core_metrics:
        return None

    audio_path = str(audio.get("path") or "")
    score_values = [value for value in axes.values() if value is not None]
    source_type = "job" if "suno-runs" in report_path.parts else "batch"
    rel_report_path = report_path.relative_to(ROOT).as_posix()
    rung = scoring.get("rung_estimate") or {}
    return {
        "report_path": rel_report_path,
        "source_type": source_type,
        "generated_at": report.get("generated_at"),
        "title": text.get("title") or Path(audio_path).stem or report_path.stem,
        "album": album_from_audio_path(audio_path),
        "track_number": track_number_from_name(audio_path),
        "audio_path": audio_path,
        "framework": framework.get("name") or Path(str(framework.get("path") or "")).name or "Unselected",
        "score": mean(score_values),
        "axes": axes,
        "core_metrics": core_metrics,
        "rung_number": finite_float(rung.get("number")),
        "rung_label": rung.get("label"),
        "confidence": finite_float(scoring.get("confidence_0_10")),
        "duration_seconds": finite_float(audio.get("duration_seconds")),
        "tempo_bpm": finite_float(audio.get("tempo_bpm")),
        "estimated_key": audio.get("estimated_key"),
        "key_confidence": finite_float(audio.get("key_confidence")),
        "integrated_lufs": finite_float(audio.get("integrated_lufs")),
        "peak_dbfs": finite_float(audio.get("peak_dbfs")),
        "crest_factor_db": finite_float(audio.get("crest_factor_db")),
        "clipping_ratio": finite_float(audio.get("clipping_ratio")),
        "onset_rate_per_second": finite_float(audio.get("onset_rate_per_second")),
        "recurrence_ratio": finite_float(audio.get("recurrence_ratio")),
        "section_count": finite_float(text.get("section_count")),
        "word_count": finite_float(text.get("word_count")),
        "line_count": finite_float(text.get("line_count")),
        "repeated_line_ratio": finite_float(text.get("repeated_line_ratio")),
        "rhyme_suffix_reuse": finite_float(text.get("rhyme_suffix_reuse")),
    }


def analysis_records() -> list[dict[str, Any]]:
    records = []
    for report_path in sorted(ROOT.rglob("*.analysis.json")):
        if ".venv" in report_path.parts or ".git" in report_path.parts:
            continue
        if "suno-runs" in report_path.parts:
            continue
        try:
            summary = summarize_analysis_report(report_path)
        except Exception:
            continue
        if summary:
            records.append(summary)
    return records


def album_stats(records: list[dict[str, Any]], total_tracks: int | None = None) -> dict[str, Any]:
    scores = [record["score"] for record in records if record.get("score") is not None]
    rung_numbers = [record["rung_number"] for record in records if record.get("rung_number") is not None]
    tempos = [record["tempo_bpm"] for record in records if record.get("tempo_bpm") is not None]
    durations = [record["duration_seconds"] for record in records if record.get("duration_seconds") is not None]
    axis_keys = sorted({key for record in records for key in (record.get("axes") or {})})
    core_keys = sorted({key for record in records for key in (record.get("core_metrics") or {})})
    return {
        "track_count": total_tracks if total_tracks is not None else len(records),
        "analyzed_count": len(records),
        "average_score": mean(scores),
        "average_rung": mean(rung_numbers),
        "average_tempo_bpm": mean(tempos),
        "total_duration_seconds": round(sum(durations), 3),
        "axes": {
            key: mean([record["axes"][key] for record in records if record.get("axes", {}).get(key) is not None])
            for key in axis_keys
        },
        "core_metrics": {
            key: mean([record["core_metrics"][key] for record in records if record.get("core_metrics", {}).get(key) is not None])
            for key in core_keys
        },
    }


def analysis_statistics() -> dict[str, Any]:
    records = analysis_records()
    return {
        "records": records,
        "summary": {"count": len(records), **album_stats(records)},
    }


def album_catalog() -> list[dict[str, Any]]:
    records = analysis_records()
    records_by_audio = {
        str(record.get("audio_path") or "").lower(): record
        for record in records
        if record.get("audio_path")
    }
    records_by_album_title: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("album") or "").lower(), str(record.get("title") or "").lower())
        records_by_album_title.setdefault(key, record)

    ignored = {
        ".agents",
        ".cache",
        ".codex",
        ".git",
        ".venv",
        "analysis-output",
        "analyzer",
        "deploy",
        "scripts",
        "suno-runs",
    }
    albums: list[dict[str, Any]] = []
    for album_dir in sorted((path for path in ROOT.iterdir() if path.is_dir() and path.name not in ignored), key=lambda path: path.name.lower()):
        text_paths = sorted(album_dir.glob("*.txt"), key=lambda path: (track_number_from_name(path.name) or 9999, path.name.lower()))
        if not text_paths:
            continue
        tracks = []
        album_records = []
        for index, text_path in enumerate(text_paths, start=1):
            try:
                title = suno_iterate.parse_track_text(text_path).title
            except Exception:
                title = text_path.stem.split(" - ", 1)[-1]
            audio_path = audio_path_for_text(text_path)
            audio_rel = relative_project_path(audio_path) if audio_path else ""
            record = records_by_audio.get(audio_rel.lower()) or records_by_album_title.get((album_dir.name.lower(), title.lower()))
            if record:
                album_records.append(record)
            tracks.append(
                {
                    "index": index,
                    "track_number": track_number_from_name(text_path.name) or index,
                    "title": title,
                    "text_path": relative_project_path(text_path),
                    "audio_path": audio_rel,
                    "audio_url": public_url_for_path(audio_path) if audio_path else "",
                    "analysis": record,
                }
            )
        albums.append(
            {
                "name": album_dir.name,
                "track_count": len(tracks),
                "tracks": tracks,
                "stats": album_stats(album_records, len(tracks)),
            }
        )
    return albums


def album_detail(album_name: str) -> dict[str, Any]:
    for album in album_catalog():
        if album["name"] == album_name:
            return album
    raise FileNotFoundError(album_name)


def public_url_for_path(path: Path) -> str:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return "/media?path=" + urllib.parse.quote(rel)


def enrich_job(job: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(job)
    for iteration in enriched.get("iterations", []):
        for candidate in iteration.get("candidates", []):
            audio = candidate.get("audio_path")
            if audio:
                candidate["audio_url"] = public_url_for_path(ROOT / audio)
            image = candidate.get("image_path")
            if image:
                candidate["image_url"] = public_url_for_path(ROOT / image)
            wav = (candidate.get("wav_conversion") or {}).get("local_wav_path")
            if wav and (ROOT / wav).exists():
                candidate["wav_url"] = public_url_for_path(ROOT / wav)
    return enriched


def normalize_audio_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "title": item.get("title") or "Untitled",
        "duration": item.get("duration"),
        "tags": item.get("tags"),
        "audio_url": item.get("audioUrl") or item.get("audio_url") or item.get("sourceAudioUrl") or item.get("source_audio_url"),
        "stream_audio_url": item.get("streamAudioUrl") or item.get("stream_audio_url"),
        "image_url": item.get("imageUrl") or item.get("image_url") or item.get("sourceImageUrl") or item.get("source_image_url"),
    }


def find_iteration_and_candidate(
    job: dict[str, Any], iteration_number: int, candidate_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    for iteration in job.get("iterations", []):
        if int(iteration.get("iteration", -1)) != iteration_number:
            continue
        for candidate in iteration.get("candidates", []):
            if int(candidate.get("index", -1)) == candidate_index:
                return iteration, candidate
    raise FileNotFoundError(f"candidate {candidate_index} in iteration {iteration_number} was not found")


def resolve_candidate_audio_id(job_id: str, iteration_number: int, candidate_index: int, candidate: dict[str, Any]) -> str:
    audio_id = candidate.get("suno_audio_id") or candidate.get("audio_id") or candidate.get("id")
    if audio_id:
        return str(audio_id)

    record = read_json(job_dir(job_id) / f"iteration_{iteration_number:02d}" / "record.json", {})
    raw_items = [normalize_audio_item(item) for item in suno_iterate.extract_audio_items(record)]
    raw_item = raw_items[candidate_index - 1] if 0 <= candidate_index - 1 < len(raw_items) else {}
    audio_id = raw_item.get("id")
    if not audio_id:
        raise ValueError("selected candidate has no Suno audio ID; rerun or use a candidate from a live Suno response")
    candidate["suno_audio_id"] = str(audio_id)
    return str(audio_id)


def wav_callback_url(public_base_url: str, job_id: str, iteration_number: int, candidate_index: int) -> str:
    if not public_base_url:
        raise ValueError("public_base_url is required so Suno can reach the WAV callback")
    return (
        f"{public_base_url.rstrip('/')}/api/suno/wav-callback/{CALLBACK_TOKEN}/"
        f"{urllib.parse.quote(job_id)}/{iteration_number}/{candidate_index}"
    )


def suno_api_key_for_job(job: dict[str, Any]) -> str:
    env_file = str((job.get("settings") or {}).get("env_file") or ".env")
    env = {**suno_iterate.load_env(ROOT / env_file), **os.environ}
    api_key = env.get("SUNO_API_KEY", "")
    if not api_key:
        raise RuntimeError("SUNO_API_KEY is missing.")
    return api_key


def album_wav_destination(job: dict[str, Any]) -> Path:
    source_text = safe_project_path(job["song_text"])
    return source_text.parent / "wav" / f"{source_text.stem}.wav"


def wav_result_url(payload: dict[str, Any]) -> str:
    data = payload.get("data") or {}
    response = data.get("response") or {}
    return str(data.get("audioWavUrl") or response.get("audioWavUrl") or "")


def wav_result_task_id(payload: dict[str, Any]) -> str:
    data = payload.get("data") or {}
    return str(data.get("task_id") or data.get("taskId") or payload.get("taskId") or "")


def save_candidate_wav_updates(
    job_id: str, iteration_number: int, candidate_index: int, updates: dict[str, Any]
) -> dict[str, Any]:
    with STATE_LOCK:
        job = load_job(job_id)
        _, candidate = find_iteration_and_candidate(job, iteration_number, candidate_index)
        conversion = candidate.setdefault("wav_conversion", {})
        conversion.update({key: value for key, value in updates.items() if value is not None})
        save_job(job)
        return enrich_job(job)


def download_wav_to_album(job_id: str, iteration_number: int, candidate_index: int, wav_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not wav_url:
        raise ValueError("Suno WAV result did not include audioWavUrl")
    job = load_job(job_id)
    destination = album_wav_destination(job)
    destination_rel = destination.relative_to(ROOT).as_posix()
    save_candidate_wav_updates(
        job_id,
        iteration_number,
        candidate_index,
        {
            "status": "downloading",
            "audio_wav_url": wav_url,
            "callback": payload,
            "task_id": wav_result_task_id(payload) or (payload.get("data") or {}).get("taskId"),
            "local_wav_path": destination_rel,
            "download_started_at": now(),
        },
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        suno_iterate.download_file(wav_url, tmp, timeout=float((job.get("settings") or {}).get("request_timeout", 60)))
        tmp.replace(destination)
    except Exception as exc:
        save_candidate_wav_updates(
            job_id,
            iteration_number,
            candidate_index,
            {"status": "error", "message": f"WAV download failed: {exc}", "failed_at": now()},
        )
        raise

    add_log(job_id, f"Saved WAV for iteration {iteration_number} candidate {candidate_index} to {destination_rel}.")
    return save_candidate_wav_updates(
        job_id,
        iteration_number,
        candidate_index,
        {
            "status": "complete",
            "audio_wav_url": wav_url,
            "callback": payload,
            "local_wav_path": destination_rel,
            "completed_at": now(),
        },
    )


def download_wav_to_album_async(job_id: str, iteration_number: int, candidate_index: int, wav_url: str, payload: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=download_wav_to_album,
        args=(job_id, iteration_number, candidate_index, wav_url, payload),
        daemon=True,
    )
    thread.start()


def refresh_wav_conversion(job_id: str, iteration_number: int, candidate_index: int) -> dict[str, Any]:
    job = load_job(job_id)
    _, candidate = find_iteration_and_candidate(job, iteration_number, candidate_index)
    conversion = candidate.get("wav_conversion") or {}
    task_id = conversion.get("task_id")
    if not task_id:
        raise ValueError("WAV conversion has no task ID to check")

    local_path = conversion.get("local_wav_path")
    if conversion.get("status") == "complete" and local_path and (ROOT / local_path).exists():
        return {"ok": True, "job": enrich_job(job), "wav_conversion": conversion}
    if conversion.get("status") == "complete" and conversion.get("audio_wav_url"):
        updated_job = download_wav_to_album(job_id, iteration_number, candidate_index, conversion["audio_wav_url"], conversion.get("callback") or {})
        return {"ok": True, "job": updated_job, "wav_conversion": find_iteration_and_candidate(updated_job, iteration_number, candidate_index)[1]["wav_conversion"]}
    if conversion.get("status") == "downloading":
        if local_path and (ROOT / local_path).exists():
            updated_job = save_candidate_wav_updates(
                job_id,
                iteration_number,
                candidate_index,
                {"status": "complete", "completed_at": now()},
            )
            _, updated_candidate = find_iteration_and_candidate(updated_job, iteration_number, candidate_index)
            return {"ok": True, "job": updated_job, "wav_conversion": updated_candidate["wav_conversion"]}
        if conversion.get("audio_wav_url"):
            updated_job = download_wav_to_album(job_id, iteration_number, candidate_index, conversion["audio_wav_url"], conversion.get("callback") or {})
            _, updated_candidate = find_iteration_and_candidate(updated_job, iteration_number, candidate_index)
            return {"ok": True, "job": updated_job, "wav_conversion": updated_candidate["wav_conversion"]}
        return {"ok": True, "job": enrich_job(job), "wav_conversion": conversion}

    api_key = suno_api_key_for_job(job)
    timeout = float((job.get("settings") or {}).get("request_timeout", 60))
    query = urllib.parse.urlencode({"taskId": task_id})
    response = suno_iterate.suno_request("GET", f"/api/v1/wav/record-info?{query}", api_key, timeout=timeout)
    write_json(
        job_dir(job_id) / f"iteration_{iteration_number:02d}" / f"wav_candidate_{candidate_index:02d}_record.json",
        response,
    )
    data = response.get("data") or {}
    success_flag = str(data.get("successFlag") or "").upper()
    wav_url = wav_result_url(response)
    if success_flag == "SUCCESS" and wav_url:
        updated_job = download_wav_to_album(job_id, iteration_number, candidate_index, wav_url, response)
        _, updated_candidate = find_iteration_and_candidate(updated_job, iteration_number, candidate_index)
        return {"ok": True, "job": updated_job, "wav_conversion": updated_candidate["wav_conversion"]}
    if success_flag in {"CREATE_TASK_FAILED", "GENERATE_WAV_FAILED", "CALLBACK_EXCEPTION"}:
        updated_job = save_candidate_wav_updates(
            job_id,
            iteration_number,
            candidate_index,
            {
                "status": "error",
                "record": response,
                "message": data.get("errorMessage") or response.get("msg") or success_flag,
                "failed_at": now(),
            },
        )
        _, updated_candidate = find_iteration_and_candidate(updated_job, iteration_number, candidate_index)
        return {"ok": True, "job": updated_job, "wav_conversion": updated_candidate["wav_conversion"]}

    updated_job = save_candidate_wav_updates(
        job_id,
        iteration_number,
        candidate_index,
        {"status": "pending", "record": response, "checked_at": now()},
    )
    _, updated_candidate = find_iteration_and_candidate(updated_job, iteration_number, candidate_index)
    return {"ok": True, "job": updated_job, "wav_conversion": updated_candidate["wav_conversion"]}


def initiate_wav_conversion(job_id: str, iteration_number: int, candidate_index: int) -> dict[str, Any]:
    should_refresh = False
    with STATE_LOCK:
        job = load_job(job_id)
        if not job:
            raise FileNotFoundError(f"job not found: {job_id}")
        iteration, candidate = find_iteration_and_candidate(job, iteration_number, candidate_index)
        source_task_id = iteration.get("task_id")
        if not source_task_id:
            raise ValueError("selected iteration has no Suno task ID")
        audio_id = resolve_candidate_audio_id(job_id, iteration_number, candidate_index, candidate)
        settings = job.get("settings") or {}
        callback_url = wav_callback_url(str(settings.get("public_base_url") or ""), job_id, iteration_number, candidate_index)
        existing = candidate.get("wav_conversion") or {}
        if existing.get("status") in {"submitted", "pending", "complete", "downloading"}:
            should_refresh = True

    if should_refresh:
        return refresh_wav_conversion(job_id, iteration_number, candidate_index)

    api_key = suno_api_key_for_job(job)

    payload = {"taskId": str(source_task_id), "audioId": audio_id, "callBackUrl": callback_url}
    timeout = float((job.get("settings") or {}).get("request_timeout", 60))
    response = suno_iterate.suno_request("POST", "/api/v1/wav/generate", api_key, payload=payload, timeout=timeout)
    code = response.get("code")
    if code not in (0, 200, "0", "200", None):
        raise RuntimeError(f"Suno WAV conversion rejected request: {response}")
    data = response.get("data") or {}
    wav_task_id = data.get("taskId") or response.get("taskId") or source_task_id

    with STATE_LOCK:
        job = load_job(job_id)
        iteration, candidate = find_iteration_and_candidate(job, iteration_number, candidate_index)
        candidate["suno_audio_id"] = audio_id
        candidate["wav_conversion"] = {
            "status": "submitted",
            "task_id": str(wav_task_id),
            "source_task_id": str(source_task_id),
            "audio_id": audio_id,
            "callback_url": callback_url,
            "requested_at": now(),
            "response": response,
        }
        save_job(job)
        write_json(
            job_dir(job_id) / f"iteration_{iteration_number:02d}" / f"wav_candidate_{candidate_index:02d}_request.json",
            {"payload": payload, "response": response},
        )
    add_log(job_id, f"Requested WAV conversion for iteration {iteration_number} candidate {candidate_index}.")
    return {"ok": True, "job": enrich_job(load_job(job_id)), "wav_conversion": candidate["wav_conversion"]}


def download_image_if_present(item: dict[str, Any], iter_dir: Path, index: int, timeout: float) -> Path | None:
    url = item.get("image_url")
    if not url:
        return None
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    path = iter_dir / f"{index:02d}_cover{ext}"
    suno_iterate.download_file(url, path, timeout=timeout)
    return path


def callback_payload_has_complete_audio(payload: dict[str, Any]) -> bool:
    if int(payload.get("code", 0) or 0) != 200:
        return False
    data = payload.get("data") or {}
    return data.get("callbackType") == "complete" and bool(data.get("data"))


def wait_for_callback_or_poll(
    job_id: str,
    iteration: int,
    task_id: str,
    api_key: str,
    poll_seconds: float,
    task_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    event_key = (job_id, iteration)
    event = CALLBACK_EVENTS.setdefault(event_key, threading.Event())
    deadline = datetime.now().timestamp() + task_timeout
    while datetime.now().timestamp() < deadline:
        callback_path = job_dir(job_id) / f"iteration_{iteration:02d}" / "callback_latest.json"
        callback = read_json(callback_path, {})
        if callback_payload_has_complete_audio(callback):
            add_log(job_id, f"Complete Suno callback received for task {task_id}.")
            return callback

        add_log(job_id, f"Waiting for Suno callback or poll result for task {task_id}.")
        if event.wait(timeout=poll_seconds):
            event.clear()
            continue

        try:
            record = suno_iterate.poll_generation(task_id, api_key, 0.1, 0.2, request_timeout)
            if suno_iterate.extract_audio_items(record):
                add_log(job_id, f"Polling found completed Suno result for task {task_id}.")
                return record
        except TimeoutError:
            pass
    raise TimeoutError(f"Suno task {task_id} did not complete before timeout.")


def candidate_report_summary(report_path: Path, score_mode: str) -> dict[str, Any]:
    score, axes, report = suno_iterate.score_report(report_path, score_mode)
    llm = report.get("llm_adjusted_scoring") or {}
    scoring = llm if llm.get("available") else report["framework_scoring"]
    quality = report.get("transcription_quality") or {}
    return {
        "score": round(score, 3),
        "axes": axes,
        "core_metrics": scoring.get("core_metrics", {}),
        "rung_estimate": scoring.get("rung_estimate"),
        "transcription_quality": {
            key: quality.get(key)
            for key in (
                "available",
                "quality_score_0_10",
                "lyric_alignment_ratio",
                "word_count_ratio",
                "likely_duplicate_passes",
                "flags",
                "error",
            )
            if key in quality
        },
        "report_path": report_path.relative_to(ROOT).as_posix(),
    }


def run_job(job_id: str) -> None:
    job = load_job(job_id)
    settings = job["settings"]
    env = {**suno_iterate.load_env(ROOT / settings.get("env_file", ".env")), **os.environ}
    api_key = env.get("SUNO_API_KEY", "")
    try:
        spec = suno_iterate.parse_track_text(ROOT / job["song_text"])
        revision_notes: list[str] = []
        working_spec = spec
        api_calls = 0
        patch_job(job_id, status="running", started_at=now())
        add_log(job_id, "Job started.")

        for iteration in range(1, int(settings["max_iterations"]) + 1):
            iter_dir = job_dir(job_id) / f"iteration_{iteration:02d}"
            callback_url = f"{settings['public_base_url'].rstrip('/')}/api/suno/callback/{CALLBACK_TOKEN}/{job_id}/{iteration}"
            revision_result = None
            if revision_notes:
                add_log(job_id, f"Revising text for iteration {iteration} from analysis feedback.")
                working_spec, revision_result = suno_iterate.revise_spec_with_feedback(
                    working_spec,
                    revision_notes,
                    settings.get("ollama_model") or suno_iterate.DEFAULT_OLLAMA_MODEL,
                    settings["ollama_url"],
                    float(settings["ollama_timeout"]),
                )
            payload = suno_iterate.build_payload(
                working_spec,
                revision_notes,
                callback_url,
                float(settings.get("style_weight", 0.75)),
                float(settings.get("weirdness_constraint", 0.75)),
                settings.get("vocal_gender") or None,
            )
            verification_errors = suno_iterate.verify_suno_spec(working_spec)
            if verification_errors:
                raise RuntimeError(f"Iteration {iteration} did not pass Suno input verification: {'; '.join(verification_errors)}")
            text_path = suno_iterate.write_iteration_inputs(iter_dir, working_spec, payload, revision_notes)
            if revision_result:
                write_json(iter_dir / "text_revision.json", revision_result)

            with STATE_LOCK:
                job = load_job(job_id)
                job.setdefault("iterations", []).append(
                    {
                        "iteration": iteration,
                        "status": "prepared",
                        "payload_path": (iter_dir / "payload.json").relative_to(ROOT).as_posix(),
                        "text_path": text_path.relative_to(ROOT).as_posix(),
                        "revision_notes": revision_notes,
                        "candidates": [],
                    }
                )
                save_job(job)

            if not settings["live"]:
                patch_job(job_id, status="dry-run-complete", completed_at=now())
                add_log(job_id, "Dry run completed; no Suno API call was made.")
                return
            if not api_key:
                raise RuntimeError("SUNO_API_KEY is missing.")
            if api_calls >= int(settings["max_api_calls"]):
                patch_job(job_id, status="stopped", stop_reason="max_api_calls", completed_at=now())
                add_log(job_id, "Stopped because max_api_calls was reached.")
                return

            api_calls += 1
            add_log(job_id, f"Submitting iteration {iteration} to Suno.")
            task_id = suno_iterate.submit_generation(payload, api_key, timeout=float(settings["request_timeout"]))
            write_json(iter_dir / "task.json", {"taskId": task_id, "callbackUrl": callback_url})
            update_iteration(job_id, iteration, status="submitted", task_id=task_id)

            record = wait_for_callback_or_poll(
                job_id,
                iteration,
                task_id,
                api_key,
                float(settings["poll_seconds"]),
                float(settings["task_timeout"]),
                float(settings["request_timeout"]),
            )
            write_json(iter_dir / "record.json", record)
            update_iteration(job_id, iteration, status="downloading")
            add_log(job_id, f"Downloading generated audio for iteration {iteration}.")

            audio_paths = suno_iterate.download_audio(record, iter_dir, timeout=float(settings["request_timeout"]))
            raw_items = [normalize_audio_item(item) for item in suno_iterate.extract_audio_items(record)]
            if not audio_paths:
                raise RuntimeError(f"Suno task {task_id} did not expose downloadable audio URLs.")

            candidates: list[dict[str, Any]] = []
            for index, audio_path in enumerate(audio_paths, start=1):
                raw_item = raw_items[index - 1] if index - 1 < len(raw_items) else {}
                image_path = None
                try:
                    image_path = download_image_if_present(raw_item, iter_dir, index, float(settings["request_timeout"]))
                except Exception as exc:
                    add_log(job_id, f"Cover download skipped for candidate {index}: {exc}")
                candidates.append(
                    {
                        "index": index,
                        "title": raw_item.get("title") or audio_path.stem,
                        "duration": raw_item.get("duration"),
                        "tags": raw_item.get("tags"),
                        "suno_audio_id": raw_item.get("id"),
                        "audio_path": audio_path.relative_to(ROOT).as_posix(),
                        "image_path": image_path.relative_to(ROOT).as_posix() if image_path else None,
                        "analysis_status": "pending",
                    }
                )
            update_iteration(job_id, iteration, status="analyzing", candidates=candidates)
            add_log(job_id, f"Downloaded {len(candidates)} candidate(s); starting analysis.")

            report_paths = []
            analysis_dir = job_dir(job_id) / "analysis"
            beat_dir = iter_dir / "beat_this"
            framework = ROOT / settings["framework"] if settings.get("framework") else None
            for candidate in candidates:
                audio_path = ROOT / candidate["audio_path"]
                add_log(job_id, f"Analyzing candidate {candidate['index']}: {candidate['title']}.")
                beat_file = suno_iterate.run_beat_this(
                    audio_path,
                    beat_dir,
                    settings.get("beat_this_gpu") or suno_iterate.DEFAULT_BEAT_THIS_GPU,
                )
                candidate["beat_file"] = beat_file.relative_to(ROOT).as_posix()
                report_path = suno_iterate.run_analyzer(
                    audio_path,
                    text_path,
                    analysis_dir,
                    framework,
                    settings.get("ollama_model") or None,
                    settings["ollama_url"],
                    float(settings["ollama_timeout"]),
                    beat_file,
                    settings.get("transcription_backend") or suno_iterate.DEFAULT_TRANSCRIPTION_BACKEND,
                    settings.get("transcription_model") or suno_iterate.DEFAULT_TRANSCRIPTION_MODEL,
                    settings.get("transcription_language") or "en",
                    settings.get("transcription_device") or "auto",
                    settings.get("transcription_compute_type") or "default",
                    float(settings.get("transcription_timeout", 900)),
                    settings.get("transcription_model_dir") or suno_iterate.DEFAULT_TRANSCRIPTION_MODEL_DIR,
                    bool(settings.get("transcription_vad_filter")),
                )
                report_paths.append(report_path)
                candidate.update(candidate_report_summary(report_path, settings["score_mode"]))
                candidate["analysis_status"] = "complete"
                update_iteration(job_id, iteration, candidates=candidates)
                add_log(job_id, f"Candidate {candidate['index']} score: {candidate['score']:.3f}.")

            best = suno_iterate.choose_best(report_paths, settings["score_mode"])
            reached = best["score"] >= float(settings["threshold"])
            update_iteration(job_id, iteration, status="complete", best=best, reached_threshold=reached)
            patch_job(job_id, best=best, status="complete" if reached else "running")
            add_log(job_id, f"Iteration {iteration} best score: {best['score']:.3f}.")
            if reached:
                patch_job(job_id, status="complete", completed_at=now())
                add_log(job_id, "Threshold reached.")
                return
            revision_notes = suno_iterate.revision_notes_from_axes(best["axes"], float(settings["threshold"]))

        patch_job(job_id, status="stopped", stop_reason="max_iterations", completed_at=now())
        add_log(job_id, "Stopped because max_iterations was reached.")
    except Exception as exc:
        patch_job(job_id, status="error", error=str(exc), traceback=traceback.format_exc(), completed_at=now())
        add_log(job_id, f"Error: {exc}")


def update_iteration(job_id: str, iteration: int, **updates: Any) -> None:
    with STATE_LOCK:
        job = load_job(job_id)
        for item in job.get("iterations", []):
            if item.get("iteration") == iteration:
                item.update(updates)
                break
        save_job(job)


def safe_project_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    path.relative_to(ROOT.resolve())
    return path


def promote_candidate(job_id: str, iteration_number: int, candidate_index: int, overwrite: bool = False) -> dict[str, Any]:
    with STATE_LOCK:
        job = load_job(job_id)
        if not job:
            raise FileNotFoundError(f"job not found: {job_id}")
        source_text = safe_project_path(job["song_text"])
        album_dir = source_text.parent
        audio_dir = album_dir / "audio"
        destination = audio_dir / f"{source_text.stem}.mp3"

        selected_iteration = None
        selected_candidate = None
        for iteration in job.get("iterations", []):
            if int(iteration.get("iteration", -1)) != iteration_number:
                continue
            selected_iteration = iteration
            for candidate in iteration.get("candidates", []):
                if int(candidate.get("index", -1)) == candidate_index:
                    selected_candidate = candidate
                    break
        if selected_iteration is None or selected_candidate is None:
            raise FileNotFoundError(f"candidate {candidate_index} in iteration {iteration_number} was not found")
        audio_path = selected_candidate.get("audio_path")
        if not audio_path:
            raise ValueError("selected candidate has no audio file")
        source_audio = safe_project_path(audio_path)
        if not source_audio.exists():
            raise FileNotFoundError(f"candidate audio does not exist: {source_audio}")
        iteration_text_path = selected_iteration.get("text_path")
        if iteration_text_path:
            source_iteration_text = safe_project_path(iteration_text_path)
        else:
            text_matches = sorted((job_dir(job_id) / f"iteration_{iteration_number:02d}").glob("*.txt"))
            if not text_matches:
                raise ValueError("selected iteration has no text file")
            source_iteration_text = text_matches[0]
        if not source_iteration_text.exists():
            raise FileNotFoundError(f"iteration text does not exist: {source_iteration_text}")
        winner_text = source_iteration_text.read_text(encoding="utf-8")
        if destination.exists() and not overwrite:
            return {
                "ok": False,
                "exists": True,
                "destination": destination.relative_to(ROOT).as_posix(),
                "message": f"{destination.relative_to(ROOT).as_posix()} already exists",
            }

        audio_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists() and overwrite:
            destination.unlink()
        shutil.move(str(source_audio), str(destination))
        source_text.write_text(winner_text, encoding="utf-8")

        for iteration in job.get("iterations", []):
            for candidate in iteration.get("candidates", []):
                candidate["promoted"] = False
        selected_candidate["promoted"] = True
        selected_candidate["promoted_at"] = now()
        selected_candidate["album_audio_path"] = destination.relative_to(ROOT).as_posix()
        selected_candidate["album_text_path"] = source_text.relative_to(ROOT).as_posix()
        selected_candidate["audio_path"] = destination.relative_to(ROOT).as_posix()
        job["winner"] = {
            "iteration": iteration_number,
            "candidate_index": candidate_index,
            "album_audio_path": destination.relative_to(ROOT).as_posix(),
            "album_text_path": source_text.relative_to(ROOT).as_posix(),
            "iteration_text_path": source_iteration_text.relative_to(ROOT).as_posix(),
            "promoted_at": selected_candidate["promoted_at"],
        }
        job["status"] = "promoted"
        save_job(job)
        add_log(
            job_id,
            f"Promoted candidate {candidate_index} to {destination.relative_to(ROOT).as_posix()} and updated {source_text.relative_to(ROOT).as_posix()}.",
        )
        return {"ok": True, "job": enrich_job(job), "destination": destination.relative_to(ROOT).as_posix()}


class Handler(BaseHTTPRequestHandler):
    server_version = "SunoPipeline/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{now()}] {self.address_string()} {fmt % args}")

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    def remote_dashboard_denied(self, path: str) -> bool:
        if ALLOW_REMOTE_DASHBOARD or path.startswith("/api/suno/callback/") or path.startswith("/api/suno/wav-callback/"):
            return False
        host = self.client_address[0]
        return host not in {"127.0.0.1", "::1", "localhost"}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if self.remote_dashboard_denied(path):
            self.send_json({"error": "remote dashboard access is disabled"}, HTTPStatus.FORBIDDEN)
            return
        if path == "/":
            self.send_text(FRONTEND_PATH.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/api/config":
            env = {**suno_iterate.load_env(ROOT / ".env"), **os.environ}
            self.send_json({"public_base_url": env.get("SUNO_PUBLIC_BASE_URL", "")})
        elif path == "/api/songs":
            self.send_json({"songs": list_song_files()})
        elif path == "/api/text-analysis":
            rel = query.get("path", [""])[0]
            try:
                self.send_json(text_analysis(rel))
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif path == "/api/statistics":
            self.send_json(analysis_statistics())
        elif path == "/api/albums":
            self.send_json({"albums": album_catalog()})
        elif path == "/api/album":
            album_name = query.get("name", [""])[0]
            try:
                self.send_json(album_detail(album_name))
            except FileNotFoundError:
                self.send_json({"error": "album not found"}, HTTPStatus.NOT_FOUND)
        elif path == "/api/jobs":
            jobs = [enrich_job(read_json(path / "job.json", {})) for path in sorted(OUTPUT_ROOT.glob("*")) if (path / "job.json").exists()]
            self.send_json({"jobs": sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)})
        elif path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = load_job(job_id)
            if not job:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(enrich_job(job))
        elif path == "/media":
            rel = query.get("path", [""])[0]
            try:
                target = (ROOT / rel).resolve()
                target.relative_to(ROOT.resolve())
                if not target.exists() or target.is_dir():
                    raise FileNotFoundError(rel)
                content_type = "audio/mpeg" if target.suffix.lower() == ".mp3" else "application/octet-stream"
                if target.suffix.lower() == ".wav":
                    content_type = "audio/wav"
                elif target.suffix.lower() in {".jpg", ".jpeg"}:
                    content_type = "image/jpeg"
                elif target.suffix.lower() == ".png":
                    content_type = "image/png"
                elif target.suffix.lower() == ".json":
                    content_type = "application/json"
                data = target.read_bytes()
                range_header = self.headers.get("Range")
                if range_header and range_header.startswith("bytes="):
                    start_raw, _, end_raw = range_header.removeprefix("bytes=").partition("-")
                    start = int(start_raw or 0)
                    end = int(end_raw) if end_raw else len(data) - 1
                    end = min(end, len(data) - 1)
                    chunk = data[start : end + 1]
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_json({"error": "media not found"}, HTTPStatus.NOT_FOUND)
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if self.remote_dashboard_denied(path):
            self.send_json({"error": "remote dashboard access is disabled"}, HTTPStatus.FORBIDDEN)
            return
        if path == "/api/jobs":
            try:
                data = self.read_body()
                song_text = Path(data["song_text"])
                if song_text.is_absolute() or ".." in song_text.parts:
                    raise ValueError("song_text must be a relative project path")
                if not (ROOT / song_text).exists():
                    raise ValueError(f"song_text does not exist: {song_text}")
                job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suno_iterate.slugify((ROOT / song_text).stem)}"
                settings = {
                    "env_file": data.get("env_file", ".env"),
                    "public_base_url": data.get("public_base_url", ""),
                    "threshold": float(data.get("threshold", 8.0)),
                    "score_mode": data.get("score_mode", "mean_axes"),
                    "max_iterations": int(data.get("max_iterations", 3)),
                    "max_api_calls": int(data.get("max_api_calls", 1)),
                    "live": bool(data.get("live")),
                    "style_weight": bounded_float(data.get("style_weight"), 0.75),
                    "weirdness_constraint": bounded_float(data.get("weirdness_constraint"), 0.75),
                    "vocal_gender": normalize_vocal_gender(data.get("vocal_gender")),
                    "framework": data.get("framework") or None,
                    "ollama_model": data.get("ollama_model") or suno_iterate.DEFAULT_OLLAMA_MODEL,
                    "ollama_url": data.get("ollama_url") or "http://localhost:11434",
                    "ollama_timeout": float(data.get("ollama_timeout", 240)),
                    "transcription_backend": data.get("transcription_backend") or suno_iterate.DEFAULT_TRANSCRIPTION_BACKEND,
                    "transcription_model": data.get("transcription_model") or suno_iterate.DEFAULT_TRANSCRIPTION_MODEL,
                    "transcription_language": data.get("transcription_language") or "en",
                    "transcription_device": data.get("transcription_device") or "auto",
                    "transcription_compute_type": data.get("transcription_compute_type") or "default",
                    "transcription_timeout": float(data.get("transcription_timeout", 900)),
                    "transcription_model_dir": data.get("transcription_model_dir") or suno_iterate.DEFAULT_TRANSCRIPTION_MODEL_DIR,
                    "transcription_vad_filter": bool(data.get("transcription_vad_filter")),
                    "beat_this_gpu": str(data.get("beat_this_gpu") or suno_iterate.DEFAULT_BEAT_THIS_GPU),
                    "poll_seconds": float(data.get("poll_seconds", 15)),
                    "task_timeout": float(data.get("task_timeout", 900)),
                    "request_timeout": float(data.get("request_timeout", 60)),
                }
                if settings["live"] and not settings["public_base_url"]:
                    raise ValueError("public_base_url is required for live jobs so Suno can reach the callback.")
                job = {
                    "id": job_id,
                    "song_text": song_text.as_posix(),
                    "status": "queued",
                    "settings": settings,
                    "created_at": now(),
                    "updated_at": now(),
                    "iterations": [],
                    "log": [],
                }
                save_job(job)
                thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
                WORKERS[job_id] = thread
                thread.start()
                self.send_json(enrich_job(load_job(job_id)), HTTPStatus.CREATED)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif path.startswith("/api/jobs/") and path.endswith("/promote"):
            try:
                parts = path.strip("/").split("/")
                job_id = parts[2]
                data = self.read_body()
                result = promote_candidate(
                    job_id,
                    int(data.get("iteration", 1)),
                    int(data["candidate_index"]),
                    bool(data.get("overwrite", False)),
                )
                status = HTTPStatus.CONFLICT if result.get("exists") else HTTPStatus.OK
                self.send_json(result, status)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif path.startswith("/api/jobs/") and path.endswith("/wav"):
            try:
                parts = path.strip("/").split("/")
                job_id = parts[2]
                data = self.read_body()
                result = initiate_wav_conversion(
                    job_id,
                    int(data.get("iteration", 1)),
                    int(data["candidate_index"]),
                )
                self.send_json(result)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif path.startswith("/api/suno/callback/"):
            parts = path.strip("/").split("/")
            try:
                token = parts[3]
                if token != CALLBACK_TOKEN:
                    self.send_json({"error": "invalid callback token"}, HTTPStatus.FORBIDDEN)
                    return
                job_id = parts[4]
                iteration = int(parts[5]) if len(parts) > 5 else 1
                payload = self.read_body()
                iter_dir = job_dir(job_id) / f"iteration_{iteration:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                callback_index = len(list(iter_dir.glob("callback_*.json"))) + 1
                write_json(iter_dir / f"callback_{callback_index:03d}.json", payload)
                write_json(iter_dir / "callback_latest.json", payload)
                data = payload.get("data") or {}
                add_log(job_id, f"Received Suno callback type={data.get('callbackType')} code={payload.get('code')}.")
                event = CALLBACK_EVENTS.setdefault((job_id, iteration), threading.Event())
                event.set()
                self.send_json({"status": "received"})
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif path.startswith("/api/suno/wav-callback/"):
            parts = path.strip("/").split("/")
            try:
                token = parts[3]
                if token != CALLBACK_TOKEN:
                    self.send_json({"error": "invalid callback token"}, HTTPStatus.FORBIDDEN)
                    return
                job_id = urllib.parse.unquote(parts[4])
                iteration = int(parts[5])
                candidate_index = int(parts[6].removesuffix("wavGenerated"))
                payload = self.read_body()
                iter_dir = job_dir(job_id) / f"iteration_{iteration:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                callback_index = len(list(iter_dir.glob(f"wav_candidate_{candidate_index:02d}_callback_*.json"))) + 1
                write_json(iter_dir / f"wav_candidate_{candidate_index:02d}_callback_{callback_index:03d}.json", payload)
                write_json(iter_dir / f"wav_candidate_{candidate_index:02d}_callback_latest.json", payload)

                data = payload.get("data") or {}
                wav_url = wav_result_url(payload)
                if int(payload.get("code", 0) or 0) == 200 and wav_url:
                    save_candidate_wav_updates(
                        job_id,
                        iteration,
                        candidate_index,
                        {
                            "status": "downloading",
                            "completed_at": now(),
                            "callback": payload,
                            "task_id": data.get("task_id") or data.get("taskId"),
                            "audio_wav_url": wav_url,
                            "message": payload.get("msg"),
                        },
                    )
                    download_wav_to_album_async(job_id, iteration, candidate_index, wav_url, payload)
                else:
                    save_candidate_wav_updates(
                        job_id,
                        iteration,
                        candidate_index,
                        {
                            "status": "error",
                            "completed_at": now(),
                            "callback": payload,
                            "task_id": data.get("task_id") or data.get("taskId"),
                            "audio_wav_url": wav_url,
                            "message": payload.get("msg"),
                        },
                    )
                add_log(job_id, f"Received WAV callback for iteration {iteration} candidate {candidate_index}: {payload.get('msg')}.")
                event = WAV_CALLBACK_EVENTS.setdefault((job_id, iteration, candidate_index), threading.Event())
                event.set()
                self.send_json({"status": "received"})
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Suno generation dashboard and callback endpoint.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Env file containing SUNO_CALLBACK_TOKEN.")
    parser.add_argument("--callback-token", help="Secret token embedded in callback URLs.")
    parser.add_argument("--allow-remote-dashboard", action="store_true", help="Allow non-local clients to use the dashboard and job API.")
    return parser.parse_args()


def main() -> int:
    global ALLOW_REMOTE_DASHBOARD, CALLBACK_TOKEN
    args = parse_args()
    env = {**suno_iterate.load_env(ROOT / args.env_file), **os.environ}
    CALLBACK_TOKEN = args.callback_token or env.get("SUNO_CALLBACK_TOKEN") or suno_iterate.slugify(os.urandom(16).hex())
    ALLOW_REMOTE_DASHBOARD = args.allow_remote_dashboard
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Suno dashboard: http://127.0.0.1:{args.port}")
    print(f"Callback path template: /api/suno/callback/{CALLBACK_TOKEN}/<job_id>/<iteration>")
    if not ALLOW_REMOTE_DASHBOARD:
        print("Remote dashboard access is disabled; remote callers can only use the callback route.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
