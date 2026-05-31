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

Use a precomputed Beat This! beat/downbeat grid:

```bash
beat_this "stderr/audio" -o "analysis-output/beat-this/stderr" --gpu -1
python scripts/analyze_track.py "stderr/audio/01 - You Follow.mp3" --beat-file "analysis-output/beat-this/stderr/01 - You Follow.beats"
```

Analyze a whole tree:

```bash
python scripts/analyze_collection.py .
```

Analyze a whole album with precomputed Beat This! files:

```bash
python scripts/analyze_collection.py "stderr" --beat-dir "analysis-output/beat-this/stderr"
```

Apply an optional Ollama LLM scoring adjustment:

```bash
python scripts/analyze_track.py "stderr/audio/01 - You Follow.mp3" \
  --beat-file "analysis-output/beat-this/stderr/01 - You Follow.beats" \
  --ollama-model qwen3:8b
```

Batch mode passes the same model through to each track:

```bash
python scripts/analyze_collection.py "stderr" \
  --beat-dir "analysis-output/beat-this/stderr" \
  --ollama-model qwen3:8b
```

The LLM does not analyze raw audio. It receives the measured features, lyrics/production notes, framework excerpt, and Python base scoring, then returns bounded score adjustments plus rationale. Axis changes are limited to `--llm-max-axis-delta` from the Python score by default.

## Suno Iteration Pipeline

`scripts/suno_iterate.py` prepares Suno custom-mode `V5_5` generation requests from the existing song text format:

- `GENRE`, `MOOD`, `TEMPO`, `KEY`, `VOCALS`, and `PRODUCTION` become the Suno `style` prompt.
- Text after `[LYRICS]` becomes the regular Suno `prompt`.
- `TITLE` becomes the Suno title.
- The script enforces Suno's current limits: 1000 characters for `style`, 5000 for `prompt`, and 100 for `title`.

Dry-run payload generation is the default and makes no Suno API calls:

```bash
python scripts/suno_iterate.py "Net Worthless/01 - Main Character Morning.txt" --threshold 8.2
```

Live runs require `SUNO_API_KEY` and `SUNO_CALLBACK_URL` in `.env`, or `--callback-url` on the command line. By default, live mode allows only one Suno generation call per run to keep testing cheap:

```bash
python scripts/suno_iterate.py "Net Worthless/01 - Main Character Morning.txt" \
  --live \
  --threshold 8.2 \
  --max-iterations 3 \
  --max-api-calls 1 \
  --ollama-model qwen3:8b
```

For each live iteration, the script submits `POST /api/v1/generate`, polls `GET /api/v1/generate/record-info?taskId=...`, downloads returned candidates, grades them with `analyze_track.py`, selects the best candidate, and appends targeted revision notes to the next style prompt when the quality threshold is not met.

Run the local dashboard and callback receiver:

```bash
python scripts/suno_server.py --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765/`. The dashboard can start dry-run jobs, start live Suno jobs, show iteration state, and play downloaded candidates after callbacks or polling return generated audio. Live dashboard jobs use callback URLs shaped like:

```text
https://your-public-host.example/api/suno/callback/<token>/<job_id>/<iteration>
```

The callback token comes from `SUNO_CALLBACK_TOKEN` in `.env` or `--callback-token`. By default, non-local clients are only allowed to POST to the callback route; the dashboard and job API stay local unless `--allow-remote-dashboard` is explicitly passed.

For a callback-only Caddy reverse proxy, install Caddy and run the deploy helper with the public hostname that points to this machine:

```bash
sudo apt install caddy
./deploy/install_suno_proxy.sh suno.example.com
```

For a temporary raw public IP, use HTTP:

```bash
./deploy/install_suno_proxy.sh 81.82.156.228
```

Then forward these router ports to this machine's LAN IP:

```text
Temporary raw-IP HTTP:
TCP 80 -> 192.168.0.181:80

Hostname HTTPS:
TCP 80  -> 192.168.0.181:80
TCP 443 -> 192.168.0.181:443
```

Use `http://81.82.156.228` as the dashboard's Public Base URL for the temporary IP setup, or `https://suno.example.com` once a hostname is configured. The proxy only forwards `/api/suno/callback/*` to the local dashboard worker; every other public path returns 404.

Outputs are written under `analysis-output/` by default:

- `*.analysis.md`: human-readable report.
- `*.analysis.json`: raw measurements and scoring data.

The EG score uses an `evolving_grammar` evidence block in the JSON and report. It compares section-level harmony, timbre, rhythm, texture, and beat-grid behavior; looks for transformed returns instead of exact repeats; detects local grid and rule changes; and scores production/vocal notes for intentional arrangement transformations.

The scoring is intentionally conservative and heuristic. The scripts measure duration, tempo, key estimate, loudness, clipping, dynamics, spectral shape, onset density, repetition proxy, rough section boundaries, optional `beat_this` beat/downbeat stability, evolving-grammar proxies, and lyric/text features. They cannot truly hear compositional intent, so final rung placement should be treated as a first-pass assistant for close listening, not an authoritative grade.
