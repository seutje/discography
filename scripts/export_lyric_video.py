#!/usr/bin/env python3
"""Render a square MP4 lyric video from a timed-lyrics JSON asset.

The renderer mirrors the GitHub Pages player: a compact audio-reactive field,
line-timed lyrics, and a bold centered current line. Frames are generated with
Pillow and piped directly into ffmpeg for H.264/AAC MP4 output.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "video-exports"
DESIGN_SIZE = 720
DEFAULT_SIZE = 2160
DEFAULT_FPS = 30
DEFAULT_SAMPLE_RATE = 22050


THEMES = {
    "dark": {
        "bg0": "#151713",
        "bg1": "#20241f",
        "panel": "#20241f",
        "chrome": "#1a1d18",
        "ink": "#f5f7f1",
        "muted": "#aab2a6",
        "line": "#3b4239",
        "accent": "#006994",
        "warm": "#800020",
        "shadow": "#050604",
    },
    "light": {
        "bg0": "#f5f6f1",
        "bg1": "#fbfcf8",
        "panel": "#ffffff",
        "chrome": "#eef2eb",
        "ink": "#20231f",
        "muted": "#686e65",
        "line": "#d9ddd2",
        "accent": "#006994",
        "warm": "#800020",
        "shadow": "#ffffff",
    },
}


@dataclass(frozen=True)
class Fonts:
    current: ImageFont.FreeTypeFont
    previous: ImageFont.FreeTypeFont
    meta: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont


@dataclass(frozen=True)
class VisualPoint:
    x: float
    y: float
    phase: float
    radius: float
    hue_role: str


def parse_hex(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid color: {value}")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def mix(left: tuple[int, int, int], right: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(left[index] * (1 - amount) + right[index] * amount) for index in range(3))


def hsla(hue: float, saturation: float, lightness: float, alpha: float) -> tuple[int, int, int, int]:
    red, green, blue = colorsys.hls_to_rgb((hue % 360) / 360, lightness / 100, saturation / 100)
    return (
        round(red * 255),
        round(green * 255),
        round(blue * 255),
        round(max(0.0, min(1.0, alpha)) * 255),
    )


def relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in " ._-" else "_" for char in value).strip(" .") or "track"


def resolve_audio_path(payload: dict[str, Any], lyrics_path: Path, override: Path | None) -> Path:
    if override:
        return override.expanduser().resolve()
    raw = payload.get("audio_path")
    if not raw:
        raise SystemExit("Timed-lyrics JSON does not include audio_path; pass --audio.")
    candidates = [
        ROOT / str(raw),
        lyrics_path.parent / str(raw),
        Path(str(raw)).expanduser(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit(f"Audio file not found for audio_path={raw!r}; pass --audio.")


def default_output_path(payload: dict[str, Any]) -> Path:
    album = safe_filename(str(payload.get("album") or "Album"))
    title = safe_filename(str(payload.get("title") or "Track"))
    return DEFAULT_OUTPUT_ROOT / album / f"{title}.mp4"


def ffprobe_duration(audio_path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


@lru_cache(maxsize=128)
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)


def make_fonts(font_path: str | None, scale: float) -> Fonts:
    return Fonts(
        current=load_font(font_path, round(50 * scale)),
        previous=load_font(font_path, round(24 * scale)),
        meta=load_font(font_path, round(20 * scale)),
        small=load_font(font_path, round(17 * scale)),
    )


def make_base_image(size: int, theme: dict[str, str]) -> Image.Image:
    top = parse_hex(theme["bg0"])
    bottom = parse_hex(theme["bg1"])
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        amount = y / max(1, size - 1)
        color = mix(top, bottom, amount)
        for x in range(size):
            pixels[x, y] = color
    return image


def add_radial(
    overlay: Image.Image,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    draw = ImageDraw.Draw(overlay, "RGBA")
    cx, cy = center
    for step in range(8, 0, -1):
        step_radius = radius * step / 8
        step_alpha = round(255 * alpha * (1 - step / 9) ** 1.35)
        draw.ellipse(
            (cx - step_radius, cy - step_radius, cx + step_radius, cy + step_radius),
            fill=(*color, step_alpha),
        )


def darken(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return mix(color, (0, 0, 0), amount)


def lighten(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return mix(color, (255, 255, 255), amount)


@lru_cache(maxsize=768)
def sphere_sprite(
    radius_key: int,
    color: tuple[int, int, int],
    opacity_key: int,
) -> Image.Image:
    radius = max(1.0, radius_key / 10)
    opacity = max(0.0, min(1.0, opacity_key / 100))
    pad = math.ceil(radius * 4.3)
    side = pad * 2 + 1
    center = pad

    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    dx = (xx - center) / radius
    dy = (yy - center) / radius
    distance = np.sqrt(dx * dx + dy * dy)

    rgba = np.zeros((side, side, 4), dtype=np.float32)

    shadow_dx = radius * 0.48
    shadow_dy = radius * 0.58
    shadow_distance = np.sqrt(((xx - center - shadow_dx) / (radius * 1.16)) ** 2 + ((yy - center - shadow_dy) / (radius * 1.04)) ** 2)
    shadow_alpha = np.clip(1 - shadow_distance, 0, 1) ** 1.8 * opacity * 0.25
    rgba[..., 3] = np.maximum(rgba[..., 3], shadow_alpha)

    glow = np.exp(-((distance / 2.35) ** 2)) * opacity * 0.46
    rgba[..., :3] = np.array(color, dtype=np.float32)
    rgba[..., 3] = np.maximum(rgba[..., 3], glow)

    mask = distance <= 1.0
    if np.any(mask):
        nx = dx[mask]
        ny = dy[mask]
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        normals = np.stack([nx, ny, nz], axis=1)
        light = np.array([-0.48, -0.62, 0.86], dtype=np.float32)
        light = light / np.linalg.norm(light)
        view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ndotl = np.clip(normals @ light, 0, 1)
        half_vec = light + view
        half_vec = half_vec / np.linalg.norm(half_vec)
        specular = np.clip(normals @ half_vec, 0, 1) ** 18
        rim = np.clip(1 - nz, 0, 1) ** 1.7

        base = np.array(color, dtype=np.float32)
        shade = 0.38 + ndotl * 0.74 - rim * 0.18
        sphere_rgb = base[None, :] * shade[:, None]
        sphere_rgb += np.array([255, 255, 255], dtype=np.float32)[None, :] * (specular[:, None] * 0.30)
        sphere_rgb += base[None, :] * (np.clip(0.9 - distance[mask], 0, 1)[:, None] * 0.07)
        sphere_rgb = np.clip(sphere_rgb, 0, 255)

        edge_alpha = np.clip((1.0 - distance[mask]) * 6.0, 0, 1)
        sphere_alpha = opacity * edge_alpha
        existing_alpha = rgba[..., 3][mask]
        out_alpha = sphere_alpha + existing_alpha * (1 - sphere_alpha)
        existing_rgb = rgba[..., :3][mask]
        rgba[..., :3][mask] = (
            sphere_rgb * sphere_alpha[:, None] + existing_rgb * existing_alpha[:, None] * (1 - sphere_alpha[:, None])
        ) / np.maximum(out_alpha[:, None], 0.001)
        rgba[..., 3][mask] = out_alpha

    rgba[..., :3] = np.clip(rgba[..., :3], 0, 255)
    rgba[..., 3] = np.clip(rgba[..., 3] * 255, 0, 255)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def draw_glowing_sphere(
    frame: Image.Image,
    x: float,
    y: float,
    radius: float,
    base_color: tuple[int, int, int],
    opacity: float,
) -> None:
    radius_key = max(10, round(radius * 10))
    opacity_key = max(20, min(75, round(opacity * 100)))
    sprite = sphere_sprite(radius_key, base_color, opacity_key)
    frame.alpha_composite(sprite, (round(x - sprite.width / 2), round(y - sprite.height / 2)))


def make_visual_points(size: int) -> list[VisualPoint]:
    rng = random.Random(1147)
    scale = size / DESIGN_SIZE
    columns = max(18, round(18 * scale))
    rows = max(5, round(5 * scale))
    points: list[VisualPoint] = []
    for index in range(columns * rows):
        column = index % columns
        row = index // columns
        base_x = (column + 0.5) * size / columns
        base_y = (row + 0.65) * size / (rows + 0.4)
        points.append(
            VisualPoint(
                x=base_x + rng.uniform(-6, 6) * scale,
                y=base_y + rng.uniform(-6, 6) * scale,
                phase=index * 0.73 + rng.uniform(-0.25, 0.25),
                radius=(1.0 + (index % 7) / 7 * 2.35) * scale,
                hue_role="warm" if index % 4 == 0 else "accent",
            )
        )
    return points


@lru_cache(maxsize=8)
def vignette_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    margin = round(size * 0.08)
    draw.ellipse((-margin, -margin, size + margin, size + margin), fill=210)
    blurred = mask.filter(ImageFilter.GaussianBlur(round(size * 0.08)))
    return Image.eval(blurred, lambda value: 210 - value)


@lru_cache(maxsize=16)
def vignette_layer(size: int, alpha: int) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.putalpha(vignette_mask(size).point(lambda value: round(value * alpha / 255)))
    return layer


def line_index_at(lines: list[dict[str, Any]], current_time: float) -> int:
    if not lines:
        return -1
    low = 0
    high = len(lines) - 1
    candidate = -1
    while low <= high:
        middle = (low + high) // 2
        start = float(lines[middle].get("start", 0))
        if start <= current_time:
            candidate = middle
            low = middle + 1
        else:
            high = middle - 1
    if candidate < 0:
        return -1

    end = float(lines[candidate].get("end", lines[candidate].get("start", 0)))
    if current_time <= end + 0.35:
        return candidate
    next_start = lines[candidate + 1].get("start") if candidate + 1 < len(lines) else None
    if next_start is not None and float(next_start) - end <= 4 and current_time < float(next_start):
        return candidate
    return -1


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if text_size(draw, trial, font)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)

    split_lines: list[str] = []
    for line in lines:
        if text_size(draw, line, font)[0] <= max_width:
            split_lines.append(line)
            continue
        current = ""
        for char in line:
            trial = f"{current}{char}"
            if current and text_size(draw, trial, font)[0] > max_width:
                split_lines.append(current)
                current = char
            else:
                current = trial
        if current:
            split_lines.append(current)
    return split_lines


@lru_cache(maxsize=2048)
def fit_font(font_path: str | None, text: str, base_size: int, min_size: int, max_width: int, max_lines: int) -> ImageFont.FreeTypeFont:
    probe = Image.new("RGB", (max_width, 200))
    draw = ImageDraw.Draw(probe)
    for size in range(base_size, min_size - 1, -2):
        font = load_font(font_path, size)
        lines = wrap_text(draw, text, font, max_width)
        if len(lines) <= max_lines and all(text_size(draw, line, font)[0] <= max_width for line in lines):
            return font
    return load_font(font_path, min_size)


@lru_cache(maxsize=2048)
def cached_wrap_text(font_path: str | None, font_size: int, text: str, max_width: int) -> tuple[str, ...]:
    probe = Image.new("RGB", (max_width, 200))
    draw = ImageDraw.Draw(probe)
    font = load_font(font_path, font_size)
    return tuple(wrap_text(draw, text, font, max_width))


def draw_centered_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    center_y: int,
    fill: tuple[int, int, int, int],
    shadow: tuple[int, int, int, int],
    max_width: int,
    line_gap: int,
) -> None:
    if not lines:
        return
    canvas_width = draw.im.size[0]
    heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(heights) + line_gap * (len(lines) - 1)
    y = center_y - total_height // 2
    for line, height in zip(lines, heights):
        width, _ = text_size(draw, line, font)
        x = (canvas_width - width) // 2
        for dx, dy in ((0, 4), (0, 2), (2, 2), (-2, 2)):
            draw.text((x + dx, y + dy), line, font=font, fill=shadow)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_gap


def draw_centered_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    font_path: str | None,
    y: int,
    max_width: int,
    theme: dict[str, str],
) -> None:
    for line in cached_wrap_text(font_path, font.size, text, max_width)[:2]:
        width, height = text_size(draw, line, font)
        x = (draw.im.size[0] - width) // 2
        draw.text((x, y + 2), line, font=font, fill=(*parse_hex(theme["shadow"]), 160))
        draw.text((x, y), line, font=font, fill=(*parse_hex(theme["muted"]), 230))
        y += height + 4


def load_audio_features(audio_path: Path, start: float, duration: float, fps: int, sample_rate: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise SystemExit("librosa is required for audio-reactive export. Install requirements.txt first.") from exc

    y, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        offset=max(0.0, start),
        duration=max(0.1, duration),
    )
    if y.size == 0:
        raise SystemExit(f"No audio samples could be loaded from {audio_path}")

    hop_length = max(1, round(sr / fps))
    frame_count = max(1, math.ceil(duration * fps))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True)[0]
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length, center=True)).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            return np.zeros_like(rms)
        return spectrum[mask].mean(axis=0)

    raw = np.vstack([rms, band(20, 250), band(250, 4000), band(4000, sr / 2)])
    normalized = np.zeros((4, frame_count), dtype=np.float32)
    for index, values in enumerate(raw):
        if values.size < frame_count:
            values = np.pad(values, (0, frame_count - values.size), mode="edge")
        values = values[:frame_count]
        scale = float(np.percentile(values, 95)) or float(values.max()) or 1.0
        values = np.clip(values / scale, 0, 1.4)
        smoothed = np.empty(frame_count, dtype=np.float32)
        previous = 0.08
        for frame_index, value in enumerate(values):
            previous = previous * 0.82 + float(value) * 0.18
            smoothed[frame_index] = min(1.0, previous)
        normalized[index] = smoothed
    return normalized.T


def render_frame(
    base: Image.Image,
    theme: dict[str, str],
    points: list[VisualPoint],
    lines: list[dict[str, Any]],
    payload: dict[str, Any],
    font_path: str | None,
    features: np.ndarray,
    frame_index: int,
    fps: int,
    start_time: float,
    reactive: bool,
) -> bytes:
    size = base.size[0]
    scale = size / DESIGN_SIZE
    t = start_time + frame_index / fps
    rms, bass, mid, high = (float(value) for value in features[min(frame_index, len(features) - 1)])
    energy = min(1.0, rms * 0.4 + bass * 0.35 + mid * 0.25 + high * 0.2)
    frame = base.convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))

    accent = parse_hex(theme["accent"])
    warm = parse_hex(theme["warm"])
    add_radial(overlay, (size * 0.18, size * 0.24), size * (0.34 + bass * 0.1), accent, 0.17 + energy * 0.1)
    add_radial(overlay, (size * 0.78, size * 0.16), size * (0.30 + high * 0.08), warm, 0.14 + energy * 0.1)
    frame.alpha_composite(overlay)

    draw = ImageDraw.Draw(frame, "RGBA")
    if reactive:
        for point in points:
            phase = t * (0.45 + high * 2.2) + point.phase
            drift = (4 + energy * 28) * scale
            x = point.x + math.sin(phase) * drift * (0.35 + bass)
            y = point.y + math.cos(phase * 0.8) * drift * (0.3 + mid)
            point_amplitude = min(1.0, energy * 0.72 + (bass if point.hue_role == "warm" else mid) * 0.28)
            radius = point.radius + (point_amplitude * 3.05 + bass * 0.7) * scale
            opacity = 0.20 + point_amplitude * 0.55
            base_color = warm if point.hue_role == "warm" else accent
            draw_glowing_sphere(frame, x, y, radius, base_color, opacity)

        wave_points: list[tuple[float, float]] = []
        amplitude = (9 + mid * 36) * scale
        line_color = (*parse_hex(theme["line"]), round(95 + energy * 85))
        for x in range(0, size + 3, 3):
            y = size * 0.56 + math.sin(t * 1.3 + x * 0.025) * amplitude
            y += math.sin(t * 2.1 + x * 0.011) * high * 16
            wave_points.append((x, y))
        draw.line(wave_points, fill=line_color, width=max(2, round(size / 360)))

    fonts = make_fonts(font_path, scale)
    max_width = round(size * 0.84)
    current_index = line_index_at(lines, t)
    current = str(lines[current_index].get("text") or "") if current_index >= 0 else ""
    previous = str(lines[current_index - 1].get("text") or "") if current_index > 0 else ""
    next_line = str(lines[current_index + 1].get("text") or "") if 0 <= current_index < len(lines) - 1 else ""

    title = str(payload.get("title") or "Untitled")
    draw_centered_label(draw, title, fonts.small, font_path, round(size * 0.075), round(size * 0.76), theme)
    draw_centered_label(
        draw,
        "ART.ficial.IGNORANCE",
        fonts.small,
        font_path,
        round(size * 0.925),
        round(size * 0.76),
        theme,
    )

    if previous:
        previous_lines = cached_wrap_text(font_path, fonts.previous.size, previous, max_width)[:2]
        draw_centered_text_block(
            draw,
            previous_lines,
            fonts.previous,
            round(size * 0.34),
            (*parse_hex(theme["muted"]), 185),
            (*parse_hex(theme["shadow"]), 130),
            max_width,
            round(size * 0.012),
        )

    if current:
        base_font_size = round(52 * scale)
        min_font_size = round(34 * scale)
        current_font = fit_font(font_path, current, base_font_size, min_font_size, max_width, 3)
        current_lines = cached_wrap_text(font_path, current_font.size, current, max_width)
        draw_centered_text_block(
            draw,
            current_lines,
            current_font,
            round(size * 0.52),
            (*parse_hex(theme["ink"]), 255),
            (*parse_hex(theme["shadow"]), 210),
            max_width,
            round(size * 0.018),
        )


    if next_line:
        next_lines = cached_wrap_text(font_path, fonts.previous.size, next_line, max_width)[:2]
        draw_centered_text_block(
            draw,
            next_lines,
            fonts.previous,
            round(size * 0.76),
            (*parse_hex(theme["muted"]), 175),
            (*parse_hex(theme["shadow"]), 120),
            max_width,
            round(size * 0.012),
        )

    vignette_alpha = 85 if theme is THEMES["dark"] else 25
    frame.alpha_composite(vignette_layer(size, vignette_alpha))
    return frame.convert("RGB").tobytes()


def build_ffmpeg_command(args: argparse.Namespace, audio_path: Path, duration: float, output_path: Path) -> list[str]:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to export MP4 files.")

    command = [
        "ffmpeg",
        "-y" if args.overwrite else "-n",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.size}x{args.size}",
        "-r",
        str(args.fps),
        "-i",
        "-",
    ]
    if args.start > 0:
        command.extend(["-ss", f"{args.start:.3f}"])
    command.extend(
        [
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{duration:.3f}",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-maxrate",
            args.maxrate,
            "-bufsize",
            args.bufsize,
            "-c:a",
            "aac",
            "-b:a",
            args.audio_bitrate,
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a timed-lyrics JSON file as a square 4K MP4 lyric video.")
    parser.add_argument("lyrics_json", type=Path, help="Path to a gh-pages/data/lyrics/*.json timed-lyrics file.")
    parser.add_argument("--audio", type=Path, help="Override the audio path in the timed-lyrics JSON.")
    parser.add_argument("--output", type=Path, help=f"Output MP4 path. Defaults under {relative_to_root(DEFAULT_OUTPUT_ROOT)}.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Square video size in pixels. Default: 2160.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame rate. Default: 30.")
    parser.add_argument("--theme", choices=sorted(THEMES), default="dark", help="Video palette. Default: dark.")
    parser.add_argument("--font", help="Optional .ttf/.otf font path. Defaults to a bold system sans font.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds, useful for previews.")
    parser.add_argument("--duration", type=float, help="Limit duration in seconds, useful for previews.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio analysis sample rate. Default: 22050.")
    parser.add_argument("--crf", type=int, default=22, help="x264 CRF quality. Lower is larger/better. Default: 22.")
    parser.add_argument("--preset", default="medium", help="x264 preset. Default: medium.")
    parser.add_argument("--maxrate", default="12000k", help="Constrained video max bitrate. Default: 12000k.")
    parser.add_argument("--bufsize", default="24000k", help="x264 VBV buffer size. Default: 24000k.")
    parser.add_argument("--audio-bitrate", default="160k", help="AAC audio bitrate. Default: 160k.")
    parser.add_argument("--reactive", action="store_true", help="Render the audio-reactive center spheres and wave line.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lyrics_path = args.lyrics_json.expanduser().resolve()
    if not lyrics_path.exists():
        raise SystemExit(f"Timed-lyrics JSON not found: {lyrics_path}")
    if args.size <= 0 or args.fps <= 0:
        raise SystemExit("--size and --fps must be positive.")
    if args.start < 0:
        raise SystemExit("--start must be non-negative.")

    payload = read_json(lyrics_path)
    lines = payload.get("lines") or []
    if not lines:
        raise SystemExit(f"No lyric lines found in {relative_to_root(lyrics_path)}")

    audio_path = resolve_audio_path(payload, lyrics_path, args.audio)
    probed_duration = ffprobe_duration(audio_path)
    json_duration = float(payload.get("duration_seconds") or 0)
    source_duration = probed_duration or json_duration
    if source_duration <= 0:
        raise SystemExit("Could not determine audio duration.")
    available_duration = max(0.1, source_duration - args.start)
    duration = min(available_duration, args.duration) if args.duration else available_duration
    if duration <= 0:
        raise SystemExit("Nothing to render after applying --start/--duration.")

    output_path = (args.output.expanduser().resolve() if args.output else default_output_path(payload).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Timed lyrics: {relative_to_root(lyrics_path)}", file=sys.stderr)
    print(f"Audio: {relative_to_root(audio_path)}", file=sys.stderr)
    print(f"Output: {relative_to_root(output_path)}", file=sys.stderr)
    print(f"Render: {args.size}x{args.size} @ {args.fps}fps for {duration:.2f}s", file=sys.stderr)

    features = load_audio_features(audio_path, args.start, duration, args.fps, args.sample_rate)
    base = make_base_image(args.size, THEMES[args.theme])
    points = make_visual_points(args.size)
    command = build_ffmpeg_command(args, audio_path, duration, output_path)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None

    frame_count = max(1, math.ceil(duration * args.fps))
    try:
        for frame_index in range(frame_count):
            if frame_index and frame_index % (args.fps * 10) == 0:
                print(f"Rendered {frame_index / args.fps:.0f}s / {duration:.0f}s", file=sys.stderr)
            process.stdin.write(
                render_frame(
                    base,
                    THEMES[args.theme],
                    points,
                    lines,
                    payload,
                    args.font,
                    features,
                    frame_index,
                    args.fps,
                    args.start,
                    args.reactive,
                )
            )
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode:
        raise SystemExit(f"ffmpeg failed with exit code {returncode}:\n{stderr.strip()}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {relative_to_root(output_path)} ({size_mb:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
