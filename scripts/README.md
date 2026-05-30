# Discography Analyzer Scripts

These scripts use the project `.venv` packages plus `ffmpeg`/`ffprobe` to extract audio features, parse the companion lyrics/production notes, choose a matching framework from `analyzer/`, and write heuristic Markdown/JSON reports.

Activate the environment first:

```bash
. .venv/bin/activate
```

Analyze one track:

```bash
python scripts/analyze_track.py "stderr/audio/01 - You Follow.mp3"
```

Print the Markdown report instead of only writing files:

```bash
python scripts/analyze_track.py "stderr/audio/01 - You Follow.mp3" --stdout
```

Force a framework:

```bash
python scripts/analyze_track.py "Residual Instabilities/audio/01 - Around Your Center.mp3" --framework "analyzer/Sorelian.txt"
```

Analyze a whole tree:

```bash
python scripts/analyze_collection.py .
```

Outputs are written under `analysis-output/` by default:

- `*.analysis.md`: human-readable report.
- `*.analysis.json`: raw measurements and scoring data.

The scoring is intentionally conservative and heuristic. The scripts measure duration, tempo, key estimate, loudness, clipping, dynamics, spectral shape, onset density, repetition proxy, rough section boundaries, and lyric/text features. They cannot truly hear compositional intent, so final rung placement should be treated as a first-pass assistant for close listening, not an authoritative grade.
