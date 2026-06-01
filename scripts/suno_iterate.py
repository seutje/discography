#!/usr/bin/env python3
"""Iteratively generate Suno tracks and grade them with the local analyzer."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SUNO_BASE_URL = "https://api.sunoapi.org"
SUNO_USER_AGENT = "discography-suno-pipeline/1.0 curl-client"
STYLE_FIELDS = ("GENRE", "MOOD", "TEMPO", "KEY", "VOCALS", "PRODUCTION")
STYLE_LIMIT = 1000
PROMPT_LIMIT = 5000
TITLE_LIMIT = 100
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_BEAT_THIS_GPU = "-1"
AXIS_KEYS = (
    "SC_structural_coherence",
    "MI_motivic_integration",
    "BP_beauty_spatial_poise",
    "EG_evolving_grammar",
    "CD_carry_depth",
)


@dataclass
class TrackSpec:
    source: Path
    title: str
    metadata: dict[str, str]
    lyrics: str
    raw: str


def slugify(value: str, fallback: str = "suno-run") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or fallback


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 15)].rstrip() + "\n[TRUNCATED]"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_track_text(path: Path) -> TrackSpec:
    raw = path.read_text(encoding="utf-8")
    metadata = {key.upper(): value.strip() for key, value in re.findall(r"\[([A-Za-z_ -]+):\s*(.*?)\]", raw, flags=re.S)}
    lyrics_match = re.search(r"\[LYRICS\]\s*(.*)", raw, flags=re.S | re.I)
    if lyrics_match:
        lyrics = lyrics_match.group(1).strip()
    else:
        lyrics = re.sub(r"^\s*(?:\[[A-Za-z_ -]+:\s*.*?\]\s*)+", "", raw, flags=re.S).strip()
    title = metadata.get("TITLE") or path.stem
    return TrackSpec(source=path, title=title, metadata=metadata, lyrics=lyrics, raw=raw)


def render_style_prompt(spec: TrackSpec) -> str:
    lines = []
    for field in STYLE_FIELDS:
        value = spec.metadata.get(field)
        if value:
            lines.append(f"{field}: {value}")
    return "\n".join(lines).strip()


def build_style(spec: TrackSpec, revision_notes: list[str] | None = None) -> str:
    return truncate(render_style_prompt(spec), STYLE_LIMIT)


def render_track_text(spec: TrackSpec) -> str:
    tag_lines = [f"[TITLE: {spec.title}]"]
    for field in STYLE_FIELDS:
        value = spec.metadata.get(field)
        if value:
            tag_lines.append(f"[{field}: {value}]")
    return "\n\n".join(tag_lines) + "\n\n[LYRICS]\n" + f"{spec.lyrics.strip()}\n"


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


def verify_suno_spec(spec: TrackSpec) -> list[str]:
    style = render_style_prompt(spec)
    errors = []
    if not spec.title.strip():
        errors.append("TITLE is empty.")
    if not spec.lyrics.strip():
        errors.append("LYRICS are empty.")
    if not style.strip():
        errors.append("Style prompt is empty.")
    if len(spec.title) > TITLE_LIMIT:
        errors.append(f"TITLE is {len(spec.title)} characters; limit is {TITLE_LIMIT}.")
    if len(style) > STYLE_LIMIT:
        errors.append(f"Style prompt is {len(style)} characters; limit is {STYLE_LIMIT}.")
    if len(spec.lyrics) > PROMPT_LIMIT:
        errors.append(f"LYRICS are {len(spec.lyrics)} characters; limit is {PROMPT_LIMIT}.")
    return errors


def spec_from_llm_revision(source: TrackSpec, raw: dict[str, Any]) -> TrackSpec:
    metadata = dict(source.metadata)
    raw_metadata = raw.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}
    raw_metadata = {str(key).upper(): value for key, value in raw_metadata.items()}
    title = str(raw.get("title") or source.title).strip()
    metadata["TITLE"] = title
    for field in STYLE_FIELDS:
        value = raw_metadata.get(field) or raw.get(field.lower()) or raw.get(field)
        if value is not None:
            metadata[field] = str(value).strip()
    lyrics = str(raw.get("lyrics") or "").strip()
    revised = TrackSpec(source=source.source, title=title, metadata=metadata, lyrics=lyrics, raw="")
    revised.raw = render_track_text(revised)
    return revised


def call_ollama_json(
    model: str,
    ollama_url: str,
    messages: list[dict[str, str]],
    timeout: float,
    num_ctx: int = 16384,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": num_ctx,
        },
    }
    request = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    content = (response_body.get("message") or {}).get("content", "")
    parsed = extract_json_object(content)
    parsed["_ollama"] = {
        "model": model,
        "url": ollama_url,
        "created_at": response_body.get("created_at"),
        "eval_count": response_body.get("eval_count"),
        "eval_duration": response_body.get("eval_duration"),
    }
    return parsed


def revise_spec_with_feedback(
    spec: TrackSpec,
    revision_notes: list[str],
    model: str,
    ollama_url: str,
    timeout: float,
    num_ctx: int = 16384,
) -> tuple[TrackSpec, dict[str, Any]]:
    if not revision_notes:
        return spec, {"changed": False, "verification_errors": verify_suno_spec(spec)}
    if not model:
        raise RuntimeError("An Ollama model is required to revise lyrics/style from feedback.")

    base_payload = {
        "task": "Revise this Suno song source for the next generation iteration.",
        "feedback_to_apply": revision_notes,
        "current_title": spec.title,
        "current_style_prompt": render_style_prompt(spec),
        "current_metadata": {field: spec.metadata.get(field, "") for field in STYLE_FIELDS},
        "current_lyrics": spec.lyrics,
        "character_limits": {
            "title": TITLE_LIMIT,
            "style_prompt_after_metadata_rendering": STYLE_LIMIT,
            "lyrics": PROMPT_LIMIT,
        },
        "style_prompt_rendering": "The style prompt will be rendered from GENRE, MOOD, TEMPO, KEY, VOCALS, and PRODUCTION only.",
        "rules": [
            "Return valid JSON only.",
            "Apply the feedback by changing the lyrics and/or the style metadata fields.",
            "Do not paste the feedback or revision notes into the lyrics or style fields.",
            "Keep the song title recognizable unless a shorter title is needed for the character limit.",
            "Preserve the song's core concept and voice while making the requested improvements.",
            "The returned lyrics must be ready to send directly as Suno's prompt.",
            "The rendered style prompt and lyrics must both fit within the listed character limits.",
        ],
        "required_json_shape": {
            "title": "string",
            "metadata": {field: "string" for field in STYLE_FIELDS},
            "lyrics": "string",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You revise song source text for Suno custom-mode generation. "
                "You receive current lyrics, the current style prompt, and concrete feedback. "
                "Return only the revised source fields as JSON."
            ),
        },
        {"role": "user", "content": json.dumps(base_payload, ensure_ascii=False)},
    ]

    attempts: list[dict[str, Any]] = []
    last_errors: list[str] = []
    revised = spec
    for attempt in range(1, 3):
        raw = call_ollama_json(model, ollama_url, messages, timeout, num_ctx=num_ctx)
        revised = spec_from_llm_revision(spec, raw)
        last_errors = verify_suno_spec(revised)
        attempts.append(
            {
                "attempt": attempt,
                "verification_errors": last_errors,
                "style_chars": len(render_style_prompt(revised)),
                "lyrics_chars": len(revised.lyrics),
                "title_chars": len(revised.title),
                "ollama": raw.get("_ollama", {}),
            }
        )
        if not last_errors:
            return revised, {"changed": True, "attempts": attempts}
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({key: value for key, value in raw.items() if key != "_ollama"}, ensure_ascii=False),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Repair the JSON so it passes verification before Suno submission.",
                        "verification_errors": last_errors,
                        "character_limits": {
                            "title": TITLE_LIMIT,
                            "style_prompt_after_metadata_rendering": STYLE_LIMIT,
                            "lyrics": PROMPT_LIMIT,
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )
    raise RuntimeError(f"LLM revision did not pass Suno input verification: {'; '.join(last_errors)}")


def build_payload(
    spec: TrackSpec,
    revision_notes: list[str],
    callback_url: str,
    style_weight: float = 0.75,
    weirdness_constraint: float = 0.75,
    vocal_gender: str | None = None,
) -> dict[str, Any]:
    payload = {
        "prompt": truncate(spec.lyrics, PROMPT_LIMIT),
        "style": build_style(spec, revision_notes),
        "title": truncate(spec.title, TITLE_LIMIT),
        "customMode": True,
        "instrumental": False,
        "model": "V5_5",
        "styleWeight": style_weight,
        "weirdnessConstraint": weirdness_constraint,
    }
    if vocal_gender:
        payload["vocalGender"] = vocal_gender
    if callback_url:
        payload["callBackUrl"] = callback_url
    return payload


def write_iteration_inputs(iter_dir: Path, spec: TrackSpec, payload: dict[str, Any], revision_notes: list[str]) -> Path:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (iter_dir / "revision_notes.json").write_text(json.dumps(revision_notes, indent=2), encoding="utf-8")
    payload_spec = TrackSpec(
        source=spec.source,
        title=payload["title"],
        metadata={**spec.metadata, "TITLE": payload["title"]},
        lyrics=payload["prompt"],
        raw="",
    )
    generated_text = render_track_text(payload_spec)
    text_path = iter_dir / f"{slugify(payload['title'], 'track')}.txt"
    text_path.write_text(generated_text, encoding="utf-8")
    return text_path


def suno_request_with_curl(method: str, path: str, api_key: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not installed")
    body = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    cmd = [
        curl,
        "--silent",
        "--show-error",
        "--location",
        "--max-time",
        str(timeout),
        "--request",
        method,
        f"{SUNO_BASE_URL}{path}",
        "--header",
        f"Authorization: Bearer {api_key}",
        "--header",
        "Content-Type: application/json",
        "--header",
        "Accept: application/json",
        "--header",
        f"User-Agent: {SUNO_USER_AGENT}",
        "--write-out",
        "\n%{http_code}",
    ]
    if body is not None:
        cmd.extend(["--data-binary", "@-"])
    completed = subprocess.run(cmd, input=body, text=True, capture_output=True)
    output = completed.stdout or ""
    if "\n" not in output:
        raise RuntimeError(f"Suno curl request failed: {completed.stderr.strip() or output.strip()}")
    response_body, status_raw = output.rsplit("\n", 1)
    try:
        status = int(status_raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"Suno curl request returned invalid status: {status_raw}") from exc
    if completed.returncode and not response_body:
        raise RuntimeError(f"Suno curl request failed: {completed.stderr.strip()}")
    if status >= 400:
        raise RuntimeError(f"Suno HTTP {status}: {response_body}")
    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Suno returned non-JSON response: {response_body[:500]}") from exc


def suno_request_with_urllib(method: str, path: str, api_key: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{SUNO_BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": SUNO_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Suno HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Suno request failed: {exc}") from exc


def suno_request(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """Call Suno with curl by default to avoid Python urllib Cloudflare fingerprint blocks."""
    backend = os.environ.get("SUNO_HTTP_BACKEND", "curl").strip().lower()
    if backend == "urllib":
        return suno_request_with_urllib(method, path, api_key, payload, timeout)
    try:
        return suno_request_with_curl(method, path, api_key, payload, timeout)
    except RuntimeError as exc:
        if "curl is not installed" in str(exc):
            return suno_request_with_urllib(method, path, api_key, payload, timeout)
        raise


def submit_generation(payload: dict[str, Any], api_key: str, timeout: float) -> str:
    response = suno_request("POST", "/api/v1/generate", api_key, payload=payload, timeout=timeout)
    if response.get("code") not in (0, 200, "0", "200", None):
        raise RuntimeError(f"Suno generate rejected request: {response}")
    data = response.get("data") or {}
    task_id = data.get("taskId") or response.get("taskId")
    if not task_id:
        raise RuntimeError(f"Suno response did not include taskId: {response}")
    return str(task_id)


def poll_generation(task_id: str, api_key: str, poll_seconds: float, timeout_seconds: float, request_timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    query = urllib.parse.urlencode({"taskId": task_id})
    last_response: dict[str, Any] = {}
    while time.time() < deadline:
        last_response = suno_request("GET", f"/api/v1/generate/record-info?{query}", api_key, timeout=request_timeout)
        data = last_response.get("data") or {}
        status = str(data.get("status") or last_response.get("status") or "").upper()
        if status in {"SUCCESS", "COMPLETE", "COMPLETED"}:
            return last_response
        if "FAILED" in status or "ERROR" in status:
            raise RuntimeError(f"Suno task failed with status {status}: {last_response}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Suno task {task_id} did not finish within {timeout_seconds:.0f}s; last response: {last_response}")


def extract_audio_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    data = record.get("data") or {}
    response = data.get("response") or {}
    candidates = response.get("sunoData") or data.get("sunoData") or response.get("data") or data.get("data") or []
    if isinstance(candidates, dict):
        candidates = [candidates]
    return [item for item in candidates if isinstance(item, dict)]


def download_file(url: str, path: Path, timeout: float) -> None:
    curl = shutil.which("curl")
    if curl:
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--max-time",
            str(max(timeout, 120)),
            "--output",
            str(path),
            url,
        ]
        if path.exists() and path.stat().st_size > 0:
            cmd[1:1] = ["--continue-at", "-"]
        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.returncode == 0:
            return
        raise RuntimeError(completed.stderr.strip() or f"curl download failed with exit code {completed.returncode}")

    request = urllib.request.Request(url, headers={"User-Agent": "discography-suno-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        path.write_bytes(response.read())


def download_audio(record: dict[str, Any], iter_dir: Path, timeout: float) -> list[Path]:
    audio_paths: list[Path] = []
    for index, item in enumerate(extract_audio_items(record), start=1):
        url = (
            item.get("audioUrl")
            or item.get("audio_url")
            or item.get("streamAudioUrl")
            or item.get("stream_audio_url")
            or item.get("sourceAudioUrl")
            or item.get("source_audio_url")
        )
        if not url:
            continue
        parsed_path = urllib.parse.urlparse(url).path
        ext = Path(parsed_path).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            ext = ".mp3"
        title = item.get("title") or f"candidate_{index:02d}"
        audio_path = iter_dir / f"{index:02d}_{slugify(str(title), f'candidate_{index:02d}')}{ext}"
        download_file(url, audio_path, timeout=timeout)
        audio_paths.append(audio_path)
    return audio_paths


def analyzer_stem(audio_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", audio_path.with_suffix("").as_posix()).strip("_")


def run_beat_this(audio_path: Path, output_dir: Path, gpu: str | None = DEFAULT_BEAT_THIS_GPU) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f"{audio_path.stem}.beats"
    if expected.exists():
        return expected

    cmd = ["beat_this", str(audio_path), "-o", str(output_dir)]
    if gpu:
        cmd.extend(["--gpu", str(gpu)])
    try:
        completed = subprocess.run(cmd, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("beat_this is not installed or is not on PATH.") from exc
    if completed.returncode:
        raise RuntimeError(f"beat_this failed for {audio_path}")
    if expected.exists():
        return expected

    matches = sorted(output_dir.rglob(f"{audio_path.stem}.beats"))
    if matches:
        return matches[0]
    raise RuntimeError(f"beat_this did not create a .beats file for {audio_path}")


def run_analyzer(
    audio_path: Path,
    text_path: Path,
    output_dir: Path,
    framework: Path | None,
    ollama_model: str | None,
    ollama_url: str,
    ollama_timeout: float,
    beat_file: Path | None = None,
) -> Path:
    analyzer = Path(__file__).with_name("analyze_track.py")
    cmd = [
        sys.executable,
        str(analyzer),
        str(audio_path),
        "--text",
        str(text_path),
        "--output-dir",
        str(output_dir),
    ]
    if framework:
        cmd.extend(["--framework", str(framework)])
    if beat_file:
        cmd.extend(["--beat-file", str(beat_file)])
    if ollama_model:
        cmd.extend(["--ollama-model", ollama_model, "--ollama-url", ollama_url, "--ollama-timeout", str(ollama_timeout)])
    completed = subprocess.run(cmd, text=True)
    if completed.returncode:
        raise RuntimeError(f"Analyzer failed for {audio_path}")
    return output_dir / f"{analyzer_stem(audio_path)}.analysis.json"


def score_report(path: Path, mode: str) -> tuple[float, dict[str, float], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    llm = report.get("llm_adjusted_scoring") or {}
    scoring = llm if llm.get("available") else report["framework_scoring"]
    axes = {key: float(value) for key, value in (scoring.get("axes") or {}).items() if key in AXIS_KEYS}
    core = scoring.get("core_metrics") or {}
    if mode == "min_axis":
        value = min(axes.values()) if axes else 0.0
    elif mode == "cdpd":
        value = float(core.get("CDPD", 0.0)) * 10.0
    elif mode == "nge":
        value = float(core.get("NGE", 0.0)) * 10.0
    else:
        value = sum(axes.values()) / len(axes) if axes else 0.0
    return value, axes, report


def choose_best(reports: list[Path], score_mode: str) -> dict[str, Any]:
    scored = []
    for path in reports:
        value, axes, report = score_report(path, score_mode)
        scored.append({"score": round(value, 3), "axes": axes, "report": str(path), "track": report["audio"]["path"]})
    if not scored:
        raise RuntimeError("No analysis reports were produced.")
    return max(scored, key=lambda item: item["score"])


def revision_notes_from_axes(axes: dict[str, float], threshold: float) -> list[str]:
    notes: list[str] = []
    shortfalls = sorted(((threshold - value, key, value) for key, value in axes.items() if value < threshold), reverse=True)
    templates = {
        "SC_structural_coherence": "Tighten section architecture: clearer intro, verse, hook, bridge, and outro boundaries with intentional transitions.",
        "MI_motivic_integration": "Make the main rhythmic, melodic, or vocal motif recur in transformed forms across sections.",
        "BP_beauty_spatial_poise": "Improve mix beauty and spatial poise: cleaner low end, controlled brightness, stronger depth, and fewer harsh collisions.",
        "EG_evolving_grammar": "Evolve the song grammar: add section-specific rule changes, transformed returns, grid deviations, and purposeful texture shifts.",
        "CD_carry_depth": "Increase emotional carry: sharpen the central image, intensify vocal commitment, and make the final section reframe earlier material.",
    }
    for _, key, value in shortfalls[:3]:
        notes.append(f"{templates[key]} Current {key} is {value:.2f}/10.")
    if not notes:
        notes.append("Preserve the strongest material while adding one surprising but coherent late-song development.")
    return notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively create Suno songs and grade them with the local framework analyzer.")
    parser.add_argument("song_text", type=Path, help="Seed song text file using TITLE/GENRE/MOOD/TEMPO/KEY/VOCALS/PRODUCTION/LYRICS.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="File containing SUNO_API_KEY and optional SUNO_CALLBACK_URL.")
    parser.add_argument("--output-root", type=Path, default=Path("suno-runs"), help="Directory for payloads, downloads, and iteration summaries.")
    parser.add_argument("--threshold", type=float, default=8.0, help="Stop once the selected score reaches this 0-10 threshold.")
    parser.add_argument("--score-mode", choices=("mean_axes", "min_axis", "cdpd", "nge"), default="mean_axes", help="Quality score used for stopping.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum generate/analyze/revise iterations.")
    parser.add_argument("--max-api-calls", type=int, default=1, help="Maximum live Suno generation calls in this run.")
    parser.add_argument("--live", action="store_true", help="Actually submit to Suno. Without this flag, only write dry-run payloads.")
    parser.add_argument("--callback-url", help="Suno callback URL. Defaults to SUNO_CALLBACK_URL from env file or environment.")
    parser.add_argument("--style-weight", type=float, default=0.75, help="Suno styleWeight control, from 0.00 to 1.00.")
    parser.add_argument("--weirdness-constraint", type=float, default=0.75, help="Suno weirdnessConstraint control, from 0.00 to 1.00.")
    parser.add_argument("--vocal-gender", choices=("m", "f"), help="Optional Suno vocalGender control.")
    parser.add_argument("--framework", type=Path, help="Optional analyzer framework override.")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Local Ollama model passed to analyze_track.py for adjusted grading.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL for analyzer grading.")
    parser.add_argument("--ollama-timeout", type=float, default=240.0, help="Analyzer Ollama timeout per generated candidate.")
    parser.add_argument("--beat-this-gpu", default=DEFAULT_BEAT_THIS_GPU, help="GPU argument passed to beat_this; use -1 for CPU.")
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Seconds between Suno task polls.")
    parser.add_argument("--task-timeout", type=float, default=900.0, help="Maximum seconds to wait for one Suno task.")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="HTTP timeout for Suno requests and audio downloads.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = {**load_env(args.env_file), **os.environ}
    api_key = env.get("SUNO_API_KEY", "")
    callback_url = args.callback_url or env.get("SUNO_CALLBACK_URL", "")
    spec = parse_track_text(args.song_text)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(spec.title)}"
    run_dir = args.output_root / run_id
    analysis_dir = run_dir / "analysis"
    revision_notes: list[str] = []
    working_spec = spec
    api_calls = 0
    history: list[dict[str, Any]] = []

    if args.live and not api_key:
        raise SystemExit("SUNO_API_KEY is required for --live runs.")
    if args.live and not callback_url:
        raise SystemExit("Suno requires callBackUrl for live generation. Set SUNO_CALLBACK_URL in .env or pass --callback-url.")

    for iteration in range(1, args.max_iterations + 1):
        iter_dir = run_dir / f"iteration_{iteration:02d}"
        revision_result: dict[str, Any] | None = None
        if revision_notes:
            working_spec, revision_result = revise_spec_with_feedback(
                working_spec,
                revision_notes,
                args.ollama_model,
                args.ollama_url,
                args.ollama_timeout,
            )
        payload = build_payload(
            working_spec,
            revision_notes,
            callback_url,
            args.style_weight,
            args.weirdness_constraint,
            args.vocal_gender,
        )
        verification_errors = verify_suno_spec(working_spec)
        if verification_errors:
            raise RuntimeError(f"Iteration {iteration} did not pass Suno input verification: {'; '.join(verification_errors)}")
        text_path = write_iteration_inputs(iter_dir, working_spec, payload, revision_notes)
        if revision_result:
            (iter_dir / "text_revision.json").write_text(json.dumps(revision_result, indent=2, ensure_ascii=False), encoding="utf-8")

        if not args.live:
            history.append(
                {
                    "iteration": iteration,
                    "mode": "dry-run",
                    "payload": str(iter_dir / "payload.json"),
                    "text": str(text_path),
                    "revision_notes": revision_notes,
                }
            )
            break
        if api_calls >= args.max_api_calls:
            history.append({"iteration": iteration, "stopped": "max_api_calls", "max_api_calls": args.max_api_calls})
            break

        api_calls += 1
        task_id = submit_generation(payload, api_key, timeout=args.request_timeout)
        (iter_dir / "task.json").write_text(json.dumps({"taskId": task_id}, indent=2), encoding="utf-8")
        record = poll_generation(task_id, api_key, args.poll_seconds, args.task_timeout, args.request_timeout)
        (iter_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        audio_paths = download_audio(record, iter_dir, timeout=args.request_timeout)
        if not audio_paths:
            raise RuntimeError(f"Suno task {task_id} did not expose downloadable audio URLs.")

        beat_dir = iter_dir / "beat_this"
        reports = []
        for audio_path in audio_paths:
            beat_file = run_beat_this(audio_path, beat_dir, args.beat_this_gpu)
            reports.append(
                run_analyzer(
                    audio_path,
                    text_path,
                    analysis_dir,
                    args.framework,
                    args.ollama_model,
                    args.ollama_url,
                    args.ollama_timeout,
                    beat_file,
                )
            )
        best = choose_best(reports, args.score_mode)
        reached = best["score"] >= args.threshold
        history.append(
            {
                "iteration": iteration,
                "taskId": task_id,
                "audio": [str(path) for path in audio_paths],
                "text": str(text_path),
                "best": best,
                "threshold": args.threshold,
                "reached_threshold": reached,
            }
        )
        if reached:
            break
        revision_notes = revision_notes_from_axes(best["axes"], args.threshold)

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if history and history[-1].get("mode") == "dry-run":
        print("Dry run only; no Suno API calls were made. Re-run with --live to submit.")
    elif history:
        best = history[-1].get("best")
        if best:
            print(f"Best score: {best['score']:.3f} via {best['track']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
