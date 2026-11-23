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
pytest test_google_health_client.py test_pipeline_logic.py test_sleep.py -v

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

This service syncs Fitbit intraday heart rate and step data (read via the Google Health API) to Garmin Connect by encoding valid binary Garmin Monitoring (type 9) FIT files and uploading them via the device sync endpoint — making them appear as if from a real Garmin device.

### Data flow

```
Google Health API (:rollUp, 60s windows, reconciled)
  → 1-min UTC points (google_health_client.py)
  → Recency buffer (hold back last N minutes for real device to sync first)
  → Garmin coverage check (skip minutes with a Garmin HR sample within ±2 min; local dates)
  → split at StateStore watermark:
      new (after watermark) → split_segments → build_monitoring_fit → upload → save_last_uploaded
      old (≤ watermark, still missing on Garmin) → backfill runs ≥ backfill_min_minutes
          → steps stripped if Garmin already has steps there
          → time_created shifted 1–3 s earlier per attempt (avoids 409 duplicate)
          → at most backfill_max_attempts per run (state.json backfill_attempts)
```

Garmin behaviour found 2026-11-22 (see diagnose_hr.py on branch claude/hr-sleep-diagnostics):
- A monitoring file whose cycles never exceed 0 is discarded ("valid type, but contained
  no content"), HR included. fit_engine adds one walking cycle to zero-step segments.
- `device.serial_number` must be a device registered to the account (else 419 "Wellness
  device is not active"), so uploads share an identity with the user's real watch, whose
  own syncs appear to wipe previously uploaded HR. Backfill re-sends those minutes.

### Sleep (sleep_sync.py, sleep_score.py)

Runs after the heart-rate sync when `sleep.enabled`. For each Fitbit Air sleep session
(platform FITBIT only — Google Health also holds Garmin's own sleep relayed through Apple
Health / Health Connect, which must never be uploaded or calibrated on), it asks Garmin
about the local wake date once (state: `sleep.json` in the state directory):
- no Garmin sleep → upload (within `upload_days`, after `holdback_hours`)
- a watch night → calibration pair (Fitbit minutes asleep + sleeping HR vs Garmin's score); refit
- Garmin window equal to the Fitbit window, or in `uploaded` → our own upload, ignored

Sleep FIT layout (type 49, copied from a real FR255S file): file_id, file_creator,
device_info, sleep_data_info (273), sleep start/stop events (74), sleep_assessment (346),
sleep_level (275) stamped at each stage's END (1 awake, 2 light, 3 deep, 4 REM).
Garmin stores nothing without real values in 346; it shows overall_sleep_score as is.
The score model is linear in hours short of / over Garmin's sleep need and sleeping HR,
shrunk toward `sleep_score.DEFAULT_WEIGHTS` (fitted on 200 nights, ~4 points MAE).

### Key files

| File | Role |
|------|------|
| `main.py` | Single sync cycle orchestration: `StateStore`, `split_segments`, `run_sync`. Runs once and exits (looping controlled by cron/systemd) |
| `fit_engine.py` | Binary FIT encoder — builds header, definition/data records, CRC checksums |
| `google_health_client.py` | Google OAuth2, rollUp fetch (60s windows), account timezone, cumulative steps |
| `garmin_client.py` | garth session mgmt, wellness coverage check, FIT upload |
| `sleep_sync.py` / `sleep_score.py` | Sleep upload and the self-calibrating sleep-score model |
| `identity_grabber.py` | Extracts manufacturer/product/serial/software_version from a real FIT file |
| `debug_sync.py` | Single-shot runner for manual testing |
| `crontab` | Cron schedule for Docker container (runs every 2 hours) |

### Configuration

Copy `config.yaml.example` → `config.yaml` (git-ignored). Key settings under `sync:`:
- `lookback_hours` — how far back to fetch from Google Health (default 4; the fetch covers `backfill_hours` if larger)
- `backfill_hours` / `backfill_min_minutes` / `backfill_max_attempts` — re-send gaps Garmin lost (defaults 48 / 10 / 3)
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
2. **Garmin coverage check** — queries wellness API; skips minutes already present (±2 min). Falls back to pass-through on error
3. **StateStore watermark** — `last_uploaded_ts` in state.json; points ≤ watermark are only re-sent as backfill, capped by `backfill_attempts`

### FIT file details

- File type 9 (Monitoring), protocol v1.0, profile v20.49
- Messages: `file_id` → `device_info` → `monitoring_info` → one `monitoring` record per minute
- `heart_rate` as uint8; steps encoded as `cycles` (steps × 2) as uint32 cumulative per segment
- `local_timestamp` = Garmin epoch timestamp + UTC offset seconds (handles DST)
- Garmin epoch: 1989-12-31 00:00:00 UTC

### Token / auth files

- Google Health OAuth2 token: `data/google_health_token.json`
- Garmin garth session: `data/garth/oauth1_token.json` + `oauth2_token.json`
- First-time auth requires running interactively (`GoogleHealthClient.authorize()`)
- Garmin re-authenticates automatically if cached session is invalid
