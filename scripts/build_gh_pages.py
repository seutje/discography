#!/usr/bin/env python3
"""Build the static GitHub Pages dashboard for analyzed songs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "gh-pages"
SOURCE_DATA = SOURCE_DIR / "data" / "catalog.json"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
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
REPORT_SCAN_IGNORED_DIRS = IGNORED_DIRS - {"analysis-output"}


def read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def parse_track_text(path: Path) -> dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")
    return {key.upper(): value.strip() for key, value in re.findall(r"\[([A-Za-z_ -]+):\s*(.*?)\]", raw, flags=re.S)}


def track_title(path: Path) -> str:
    metadata = parse_track_text(path)
    return metadata.get("TITLE") or path.stem.split(" - ", 1)[-1]


def track_number_from_name(name: str) -> int | None:
    prefix = Path(name).stem.split(" - ", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def album_from_audio_path(audio_path: str) -> str:
    parts = Path(audio_path).parts
    if "audio" in parts:
        index = parts.index("audio")
        if index > 0:
            return parts[index - 1]
    return parts[0] if len(parts) > 1 else "Unsorted"


def audio_path_for_text(text_path: Path) -> Path | None:
    audio_dir = text_path.parent / "audio"
    for suffix in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{text_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def slugify(value: str, fallback: str = "report") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or fallback


def url_for_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part) for part in Path(path).as_posix().split("/"))


def report_asset_path(report_path: str) -> str:
    return f"reports/{slugify(Path(report_path).with_suffix('').as_posix())}.json"


def summarize_analysis_report(report_path: Path, report: dict[str, Any]) -> dict[str, Any] | None:
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
    rung = scoring.get("rung_estimate") or {}
    report_rel = rel(report_path)
    return {
        "report_path": report_rel,
        "report_url": report_asset_path(report_rel),
        "source_type": "batch",
        "generated_at": report.get("generated_at"),
        "title": text.get("title") or Path(audio_path).stem or report_path.stem,
        "album": album_from_audio_path(audio_path),
        "track_number": track_number_from_name(audio_path),
        "audio_path": audio_path,
        "audio_url": f"media/{url_for_path(audio_path)}" if audio_path else "",
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


def analysis_report_paths() -> list[Path]:
    paths = []
    for report_path in ROOT.rglob("*.analysis.json"):
        if any(part in REPORT_SCAN_IGNORED_DIRS or part == "suno-runs" for part in report_path.parts):
            continue
        paths.append(report_path)
    return sorted(paths)


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


def build_album_catalog(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_audio = {
        str(record.get("audio_path") or "").lower(): record
        for record in records
        if record.get("audio_path")
    }
    records_by_album_title: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("album") or "").lower(), str(record.get("title") or "").lower())
        records_by_album_title.setdefault(key, record)

    albums = []
    album_dirs = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and path.name not in IGNORED_DIRS),
        key=lambda path: path.name.lower(),
    )
    for album_dir in album_dirs:
        text_paths = sorted(album_dir.glob("*.txt"), key=lambda path: (track_number_from_name(path.name) or 9999, path.name.lower()))
        if not text_paths:
            continue
        tracks = []
        album_records = []
        for index, text_path in enumerate(text_paths, start=1):
            title = track_title(text_path)
            audio_path = audio_path_for_text(text_path)
            audio_rel = rel(audio_path) if audio_path else ""
            record = records_by_audio.get(audio_rel.lower()) or records_by_album_title.get((album_dir.name.lower(), title.lower()))
            if not record:
                continue
            album_records.append(record)
            tracks.append(
                {
                    "index": index,
                    "track_number": track_number_from_name(text_path.name) or index,
                    "title": title,
                    "text_path": rel(text_path),
                    "audio_path": audio_rel,
                    "audio_url": f"media/{url_for_path(audio_rel)}" if audio_rel else "",
                    "analysis": record,
                }
            )
        if tracks:
            albums.append(
                {
                    "name": album_dir.name,
                    "track_count": len(tracks),
                    "tracks": tracks,
                    "stats": album_stats(album_records, len(tracks)),
                }
            )
    return albums


def build_catalog_from_reports() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for report_path in analysis_report_paths():
        report = read_json(report_path, {})
        if not isinstance(report, dict):
            continue
        summary = summarize_analysis_report(report_path, report)
        if not summary:
            continue
        records.append(summary)
        reports[summary["report_path"]] = report

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "statistics": {"records": records, "summary": {"count": len(records), **album_stats(records)}},
        "albums": build_album_catalog(records),
        "reports": reports,
    }


def load_source_catalog() -> dict[str, Any]:
    report_paths = analysis_report_paths()
    if report_paths:
        return build_catalog_from_reports()
    catalog = read_json(SOURCE_DATA)
    if isinstance(catalog, dict):
        return catalog
    raise FileNotFoundError("No analysis reports found and gh-pages/data/catalog.json does not exist.")


def strip_embedded_reports(catalog: dict[str, Any]) -> dict[str, Any]:
    result = dict(catalog)
    result.pop("reports", None)
    return result


def report_payloads(catalog: dict[str, Any]) -> dict[str, Any]:
    reports = catalog.get("reports") or {}
    if isinstance(reports, dict):
        return reports
    return {}


def collect_audio_paths(catalog: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for record in (catalog.get("statistics") or {}).get("records") or []:
        if record.get("audio_path"):
            paths.add(str(record["audio_path"]))
    for album in catalog.get("albums") or []:
        for track in album.get("tracks") or []:
            if track.get("audio_path"):
                paths.add(str(track["audio_path"]))
    return paths


def copy_site_assets(catalog: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(SOURCE_DIR / "index.html", output_dir / "index.html")
    write_json(output_dir / "data" / "catalog.json", strip_embedded_reports(catalog))

    for report_path, payload in report_payloads(catalog).items():
        write_json(output_dir / report_asset_path(report_path), payload)

    for audio_path in sorted(collect_audio_paths(catalog)):
        source = ROOT / audio_path
        if not source.exists() or not source.is_file():
            continue
        target = output_dir / "media" / audio_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the static GitHub Pages dashboard.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_site", help="Directory for the deployable site.")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh gh-pages/data/catalog.json from local analysis reports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_source_catalog()
    if args.refresh_data:
        write_json(SOURCE_DATA, catalog)
    copy_site_assets(catalog, args.output_dir)
    records = (catalog.get("statistics") or {}).get("records") or []
    print(f"Built {args.output_dir} with {len(records)} analyzed song(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
