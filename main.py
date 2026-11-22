"""
main.py
Fitbit2Garmin orchestration loop.
Pulls Google Health intraday data, encodes a Garmin Monitoring FIT file, and uploads it.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import yaml

from google_health_client import GoogleHealthClient
from fit_engine import build_monitoring_fit
from garmin_client import GarminClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fitbit2garmin")

CONFIG_FILE = os.environ.get("CONFIG_FILE", os.path.join("data", "config.yaml"))
STATE_FILE = os.path.join(os.environ.get("STATE_DIRECTORY", "data"), "state.json")


class StateStore:
    """Persist sync state: the last uploaded data point and backfill attempts."""

    def __init__(self, path: str):
        self.path = path

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def _store(self, key: str, value):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = self._load()
        data[key] = value
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
        log.debug("State saved: %s=%s", key, value)

    def load_last_uploaded(self) -> datetime | None:
        raw = self._load().get("last_uploaded_ts")
        try:
            return datetime.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            return None

    def save_last_uploaded(self, dt: datetime):
        self._store("last_uploaded_ts", dt.isoformat())

    def load_backfill_attempts(self) -> dict[str, int]:
        raw = self._load().get("backfill_attempts")
        return dict(raw) if isinstance(raw, dict) else {}

    def save_backfill_attempts(self, attempts: dict[str, int]):
        self._store("backfill_attempts", attempts)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        log.error("Config file not found: %s", path)
        log.error("Copy config.yaml.example to config.yaml and fill in your credentials.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def run_hook(command: str | None):
    """Run a shell command as a hook."""
    if not command:
        return
    log.info("Running hook: %s", command)
    try:
        # Run in a subshell, ignore exit code to avoid crashing the sync loop
        subprocess.run(command, shell=True, check=False)
    except Exception as e:
        log.error("Hook execution failed: %s", e)


def split_segments(points: list[dict], gap_minutes: int = 5) -> list[list[dict]]:
    """
    Split a list of per-minute points into contiguous segments.
    A new segment begins whenever two consecutive points are more than
    `gap_minutes` apart (e.g. the tracker was off-wrist or data was missing).
    """
    if not points:
        return []
    segments = []
    current = [points[0]]
    for prev, pt in zip(points, points[1:]):
        delta = pt["datetime"] - prev["datetime"]
        if delta > timedelta(minutes=gap_minutes):
            segments.append(current)
            current = []
        current.append(pt)
    segments.append(current)
    return segments


def select_backfill_runs(points: list[dict], min_minutes: int = 10) -> list[list[dict]]:
    """
    From already-uploaded-but-missing points, pick runs of heart rate (2-minute holes
    allowed) spanning at least `min_minutes`, plus any steps-only points inside them.
    Shorter holes are left alone so Garmin's 2-minute sampling doesn't cause churn.
    """
    hr_points = [p for p in points if p.get("heart_rate")]
    runs = []
    for run in split_segments(hr_points, gap_minutes=2):
        if run[-1]["datetime"] - run[0]["datetime"] >= timedelta(minutes=min_minutes - 1):
            lo, hi = run[0]["datetime"], run[-1]["datetime"]
            runs.append([p for p in points if lo <= p["datetime"] <= hi])
    return runs


def _upload_segment(cfg, garmin, seg, utc_offset_seconds, time_created_offset=0):
    device = cfg["device"]
    fit_bytes = build_monitoring_fit(
        points=seg,
        manufacturer=device["manufacturer"],
        product_id=device["product_id"],
        serial_number=device["serial_number"],
        software_version=device.get("software_version", 331),
        utc_offset_seconds=utc_offset_seconds,
        time_created_offset=time_created_offset,
    )
    return garmin.upload_fit_for_window(fit_bytes, seg[0]["datetime"])


def run_sync(cfg: dict, health: GoogleHealthClient, garmin: GarminClient, state: StateStore):
    """
    Execute one full sync cycle.

    New data (after the watermark) is uploaded as before. Older minutes that Garmin is
    missing (e.g. wiped by a later sync from the real watch, or rejected) are re-sent
    as backfill, at most `backfill_max_attempts` times per run.
    """
    log.info("=== Fitbit2Garmin cycle starting ===")

    sync_cfg = cfg["sync"]
    lookback = sync_cfg.get("lookback_hours", 4)
    backfill_hours = sync_cfg.get("backfill_hours", 48)
    backfill_min_minutes = sync_cfg.get("backfill_min_minutes", 10)
    backfill_max_attempts = sync_cfg.get("backfill_max_attempts", 3)
    recency_minutes = sync_cfg.get("recency_minutes", 60)

    # 1. Pull Google Health data. Fetching the backfill window as well means data
    #    that reached Google Health late (tracker synced hours later) is still seen.
    health.ensure_authorized()
    points, utc_offset_seconds = health.get_combined_intraday(lookback_hours=max(lookback, backfill_hours))

    if not points:
        log.warning("No Google Health data returned for the last %dh — skipping.", max(lookback, backfill_hours))
        return

    # 2. Apply recency buffer: hold back data newer than recency_minutes.
    #    This gives the real Garmin device time to sync its own data first,
    #    so the coverage check below can see it and skip those minutes.
    recency_cutoff = datetime.now(timezone.utc) - timedelta(minutes=recency_minutes)
    before = len(points)
    points = [p for p in points if p["datetime"] <= recency_cutoff]
    held_back = before - len(points)
    if held_back:
        log.info("Holding back %d point(s) newer than %d min ago.", held_back, recency_minutes)

    if not points:
        log.info("All Google Health data is within the recency buffer — nothing to upload yet.")
        return

    # 3. Split into new data (after the watermark) and already-uploaded data, and drop
    #    minutes Garmin already has (from any source — real device or a previous upload).
    #    New data matches Garmin's samples exactly; Garmin's sample at the end of the
    #    previous upload would otherwise hide the first new minutes. Backfill allows
    #    +-2 min because Garmin reports heart rate about every 2 minutes.
    last_uploaded = state.load_last_uploaded()
    coverage = garmin.get_coverage(points, utc_offset_seconds)
    new_points = [p for p in points if last_uploaded is None or p["datetime"] > last_uploaded]
    old_points = [p for p in points if last_uploaded is not None and p["datetime"] <= last_uploaded]
    if coverage is None:
        # Without Garmin's view every old minute would look missing; never backfill blind.
        log.warning("Garmin coverage unavailable — uploading new data only, no backfill this cycle.")
        old_points = []
    else:
        before = len(new_points)
        new_points = [p for p in new_points if not garmin.is_covered(p["datetime"], coverage)]
        if before - len(new_points):
            log.info("Skipping %d new point(s) already present on Garmin Connect.", before - len(new_points))
        old_points = [p for p in old_points if not garmin.is_covered(p["datetime"], coverage, 2)]

    # 5. New data: one FIT file per contiguous segment, advancing the watermark after each.
    segments = split_segments(new_points)
    log.info("%d new segment(s) to upload.", len(segments))
    for i, seg in enumerate(segments, 1):
        log.info(
            "Segment %d/%d: %s–%s (%d points)",
            i, len(segments), seg[0]["datetime"].strftime("%H:%M"), seg[-1]["datetime"].strftime("%H:%M"), len(seg),
        )
        result = _upload_segment(cfg, garmin, seg, utc_offset_seconds)
        log.info("Upload complete: %s", result)
        state.save_last_uploaded(seg[-1]["datetime"])

    # 6. Backfill: minutes uploaded before but missing on Garmin now.
    backfill_cutoff = datetime.now(timezone.utc) - timedelta(hours=backfill_hours)
    attempts = {
        k: v for k, v in state.load_backfill_attempts().items()
        if datetime.fromisoformat(k) >= backfill_cutoff
    }
    runs = select_backfill_runs(old_points, min_minutes=backfill_min_minutes)
    if runs:
        log.info("%d backfill run(s): minutes uploaded earlier that Garmin no longer has.", len(runs))
    for run in runs:
        for seg in split_segments(run):
            key = seg[0]["datetime"].isoformat()
            tries = attempts.get(key, 0)
            label = f"{seg[0]['datetime']:%m-%d %H:%M}–{seg[-1]['datetime']:%H:%M} ({len(seg)} points)"
            if tries >= backfill_max_attempts:
                log.info("Backfill %s: gave up after %d attempt(s).", label, tries)
                continue
            # Steps were sent the first time; only re-send them if Garmin has none here.
            if garmin.steps_in_range(seg[0]["datetime"], seg[-1]["datetime"], utc_offset_seconds) > 0:
                seg = [{**p, "steps_delta": 0, "cumulative_steps": 0} for p in seg]
            log.info("Backfill %s, attempt %d.", label, tries + 1)
            # Shift time_created a few seconds earlier per attempt so Garmin doesn't
            # reject it as a duplicate of the original upload.
            attempts[key] = tries + 1
            state.save_backfill_attempts(attempts)
            result = _upload_segment(cfg, garmin, seg, utc_offset_seconds, time_created_offset=tries + 1)
            log.info("Backfill upload complete: %s", result)
    state.save_backfill_attempts(attempts)

    log.info("=== Fitbit2Garmin cycle done ===")


def main():
    cfg = load_config(CONFIG_FILE)
    state = StateStore(STATE_FILE)

    health = GoogleHealthClient(
        client_id=cfg["google_health"]["client_id"],
        client_secret=cfg["google_health"]["client_secret"],
        default_timezone=cfg["google_health"].get("timezone", "UTC"),
    )
    garmin = GarminClient(
        email=os.environ.get("GARMIN_EMAIL", cfg["garmin"]["email"]),
        password=os.environ.get("GARMIN_PASSWORD", cfg["garmin"]["password"]),
    )
    garmin.connect()

    log.info("Fitbit2Garmin starting single sync cycle.")

    hooks = cfg.get("sync", {}).get("hooks", {})

    try:
        run_sync(cfg, health, garmin, state)
        run_hook(hooks.get("on_success"))
    except Exception as e:
        log.exception("Sync cycle failed: %s", e)
        run_hook(hooks.get("on_failure"))
        sys.exit(1)


if __name__ == "__main__":
    main()
