# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up virtual environment (first time)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a single sync cycle (for cron/systemd/manual execution)
python main.py

# Run all tests
pytest test_pipeline_logic.py -v

# Run a single test class
pytest test_pipeline_logic.py::TestSplitSegments -v
pytest test_pipeline_logic.py::TestStateStore -v

# One-shot sync (no loop, 48h lookback) — requires valid config.yaml and tokens
python debug_sync.py

# Extract Garmin device identity from a real wellness FIT file
python identity_grabber.py path/to/WELLNESS.fit

# Docker (cron-based, runs every 2 hours)
docker compose up -d
docker compose logs -f

# System cron (outside Docker) — add to crontab
# 0 */2 * * * cd /path/to/fitbit2garmin && python main.py >> logs/cron.log 2>&1
```

## Architecture

This service syncs Fitbit intraday heart rate and step data to Garmin Connect by encoding valid binary Garmin Monitoring (type 9) FIT files and uploading them via the device sync endpoint — making them appear as if from a real Garmin device.

### Data flow

```
Fitbit API (1-min HR + steps)
  → UTC-normalised points (fitbit_client.py)
  → Recency buffer (hold back last N minutes for real device to sync first)
  → StateStore watermark filter (skip already-uploaded)
  → Garmin coverage check (skip minutes already on Garmin, 403 → pass-through)
  → split_segments (split on >5-min gaps)
  → build_monitoring_fit per segment (fit_engine.py)
  → upload_fit_for_window (garmin_client.py)
  → StateStore.save_last_uploaded
```

### Key files

| File | Role |
|------|------|
| `main.py` | Single sync cycle orchestration: `StateStore`, `split_segments`, `run_sync`. Runs once and exits (looping controlled by cron/systemd) |
| `fit_engine.py` | Binary FIT encoder — builds header, definition/data records, CRC checksums |
| `fitbit_client.py` | OAuth2 auth, intraday fetch, wall-clock → UTC conversion |
| `garmin_client.py` | garth session mgmt, wellness coverage check, FIT upload |
| `identity_grabber.py` | Extracts manufacturer/product/serial/software_version from a real FIT file |
| `debug_sync.py` | Single-shot runner for manual testing |
| `crontab` | Cron schedule for Docker container (runs every 2 hours) |

### Configuration

Copy `config.yaml.example` → `config.yaml` (git-ignored). Key settings under `sync:`:
- `lookback_hours` — how far back to fetch from Fitbit (default 4)
- `recency_minutes` — hold-back buffer so a real Garmin device can sync first (default 60)
- `hooks.on_success` / `hooks.on_failure` — optional shell commands (e.g. healthchecks.io)

Environment variables:
- `CONFIG_FILE` — path to config.yaml (default: `data/config.yaml`)
- `GARTH_HOME` — garth session cache directory (default: `~/.garth`; Docker uses `/app/data/garth`)

### Execution model

**main.py now runs a single sync cycle and exits.** Looping is controlled externally:

- **Docker**: Cron (in the container, defined in `crontab`) runs `main.py` on a schedule. Default: every 2 hours.
- **System cron** (outside Docker): Add an entry to your crontab to run `python main.py` at your desired frequency.
- **systemd**: Use a systemd timer to trigger the script periodically.

To customize the Docker schedule, edit `crontab` before building the image, or override `CMD` in `docker-compose.yaml` to use systemd instead of cron.

### Deduplication layers (in order)

1. **Recency buffer** — drops points newer than `recency_minutes` ago
2. **StateStore watermark** — `data/state.json` persists `last_uploaded_ts`; skips points ≤ watermark
3. **Garmin coverage check** — queries wellness API; skips minutes already present. Falls back to pass-through on 403 (StateStore is then the sole guard)

### FIT file details

- File type 9 (Monitoring), protocol v1.0, profile v20.49
- Messages: `file_id` → `device_info` → `monitoring_info` → one `monitoring` record per minute
- `heart_rate` as uint8; steps encoded as `cycles` (steps × 2) as uint32 cumulative per segment
- `local_timestamp` = Garmin epoch timestamp + UTC offset seconds (handles DST)
- Garmin epoch: 1989-12-31 00:00:00 UTC

### Token / auth files

- Fitbit OAuth2 token: `data/token.json`
- Garmin garth session: `data/garth/oauth1_token.json` + `oauth2_token.json`
- First-time Fitbit auth requires running interactively (`fitbit.authorize()`)
- Garmin re-authenticates automatically if cached session is invalid
