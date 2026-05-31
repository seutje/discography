#!/usr/bin/env python3
"""Local web UI and callback receiver for the Suno iteration pipeline."""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import suno_iterate


ROOT = Path.cwd()
OUTPUT_ROOT = ROOT / "suno-runs" / "web"
FRONTEND_PATH = Path(__file__).with_name("suno_frontend.html")
STATE_LOCK = threading.RLock()
CALLBACK_EVENTS: dict[tuple[str, int], threading.Event] = {}
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
    return {
        "score": round(score, 3),
        "axes": axes,
        "core_metrics": scoring.get("core_metrics", {}),
        "rung_estimate": scoring.get("rung_estimate"),
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
        api_calls = 0
        patch_job(job_id, status="running", started_at=now())
        add_log(job_id, "Job started.")

        for iteration in range(1, int(settings["max_iterations"]) + 1):
            iter_dir = job_dir(job_id) / f"iteration_{iteration:02d}"
            callback_url = f"{settings['public_base_url'].rstrip('/')}/api/suno/callback/{CALLBACK_TOKEN}/{job_id}/{iteration}"
            payload = suno_iterate.build_payload(spec, revision_notes, callback_url)
            text_path = suno_iterate.write_iteration_inputs(iter_dir, spec, payload, revision_notes)

            with STATE_LOCK:
                job = load_job(job_id)
                job.setdefault("iterations", []).append(
                    {
                        "iteration": iteration,
                        "status": "prepared",
                        "payload_path": (iter_dir / "payload.json").relative_to(ROOT).as_posix(),
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
                        "audio_path": audio_path.relative_to(ROOT).as_posix(),
                        "image_path": image_path.relative_to(ROOT).as_posix() if image_path else None,
                        "analysis_status": "pending",
                    }
                )
            update_iteration(job_id, iteration, status="analyzing", candidates=candidates)
            add_log(job_id, f"Downloaded {len(candidates)} candidate(s); starting analysis.")

            report_paths = []
            analysis_dir = job_dir(job_id) / "analysis"
            framework = ROOT / settings["framework"] if settings.get("framework") else None
            for candidate in candidates:
                audio_path = ROOT / candidate["audio_path"]
                add_log(job_id, f"Analyzing candidate {candidate['index']}: {candidate['title']}.")
                report_path = suno_iterate.run_analyzer(
                    audio_path,
                    text_path,
                    analysis_dir,
                    framework,
                    settings.get("ollama_model") or None,
                    settings["ollama_url"],
                    float(settings["ollama_timeout"]),
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
        if ALLOW_REMOTE_DASHBOARD or path.startswith("/api/suno/callback/"):
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
        elif path == "/api/config":
            env = {**suno_iterate.load_env(ROOT / ".env"), **os.environ}
            self.send_json({"public_base_url": env.get("SUNO_PUBLIC_BASE_URL", "")})
        elif path == "/api/songs":
            self.send_json({"songs": list_song_files()})
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
                if target.suffix.lower() in {".jpg", ".jpeg"}:
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
                    "framework": data.get("framework") or None,
                    "ollama_model": data.get("ollama_model") or "",
                    "ollama_url": data.get("ollama_url") or "http://localhost:11434",
                    "ollama_timeout": float(data.get("ollama_timeout", 240)),
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
