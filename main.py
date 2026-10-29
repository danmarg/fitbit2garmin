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
from fit_engine import build_monitoring_fit, build_sleep_fit
from garmin_client import GarminClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fitbit2garmin")

CONFIG_FILE = os.environ.get("CONFIG_FILE", os.path.join("data", "config.yaml"))
STATE_FILE = os.path.join(os.environ.get("STATE_DIRECTORY"), "state.json")


class StateStore:
    """Persist the timestamp of the last successfully uploaded data point."""

    def __init__(self, path: str):
        self.path = path

    def load_last_uploaded(self) -> datetime | None:
        try:
            with open(self.path) as f:
                raw = json.load(f).get("last_uploaded_ts")
            if raw:
                return datetime.fromisoformat(raw)
        except (FileNotFoundError, KeyError, ValueError):
            pass
        return None

    def save_last_uploaded(self, dt: datetime):
        self._save("last_uploaded_ts", dt)

    def load_last_sleep_end(self) -> datetime | None:
        try:
            with open(self.path) as f:
                raw = json.load(f).get("last_sleep_end_ts")
            if raw:
                return datetime.fromisoformat(raw)
        except (FileNotFoundError, KeyError, ValueError):
            pass
        return None

    def save_last_sleep_end(self, dt: datetime):
        self._save("last_sleep_end_ts", dt)

    def _save(self, key: str, dt: datetime):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {}
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError):
            pass
        data[key] = dt.isoformat()
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
        log.debug("State saved: %s=%s", key, dt.isoformat())


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


def run_sync(cfg: dict, health: GoogleHealthClient, garmin: GarminClient, state: StateStore):
    """Execute one full sync cycle."""
    log.info("=== Fitbit2Garmin cycle starting ===")

    lookback = cfg["sync"].get("lookback_hours", 4)
    recency_minutes = cfg["sync"].get("recency_minutes", 60)
    device = cfg["device"]

    # 1. Pull Google Health data
    health.ensure_authorized()
    points, utc_offset_seconds = health.get_combined_intraday(lookback_hours=lookback)

    if not points:
        log.warning("No Google Health data returned for the last %dh — skipping.", lookback)
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

    # 3. Drop points we already uploaded (StateStore watermark).
    #    Acts as fallback deduplication when the wellness API is unavailable.
    last_uploaded = state.load_last_uploaded()
    if last_uploaded is not None:
        before = len(points)
        points = [p for p in points if p["datetime"] > last_uploaded]
        log.info(
            "Skipping %d already-uploaded point(s) (last uploaded: %s).",
            before - len(points),
            last_uploaded.strftime("%H:%M:%S UTC"),
        )

    if not points:
        log.info("No new Google Health data since last upload — nothing to do.")
        return

    # 4. Skip any minute already populated on Garmin (from any source —
    #    real device or a previous fitbit2garmin upload).  This prevents overwriting
    #    data that beat us to Garmin, while the recency buffer makes it likely
    #    that real-device data arrived first for recent minutes.
    points = garmin.filter_covered_points(points)

    if not points:
        log.info("All candidate points are already covered on Garmin — nothing to upload.")
        return

    # 5. Split into contiguous segments
    segments = split_segments(points)
    log.info("%d contiguous segment(s) to upload.", len(segments))

    # 6. Build + upload one FIT file per segment
    for i, seg in enumerate(segments, 1):
        window_start = seg[0]["datetime"]
        window_end = seg[-1]["datetime"]
        log.info(
            "Segment %d/%d: %s–%s (%d points)",
            i, len(segments),
            window_start.strftime("%H:%M"),
            window_end.strftime("%H:%M"),
            len(seg),
        )
        fit_bytes = build_monitoring_fit(
            points=seg,
            manufacturer=device["manufacturer"],
            product_id=device["product_id"],
            serial_number=device["serial_number"],
            software_version=device.get("software_version", 331),
            utc_offset_seconds=utc_offset_seconds,
        )
        result = garmin.upload_fit_for_window(fit_bytes, window_start)
        log.info("Upload complete: %s", result)
        # 7. Advance the watermark immediately after each successful upload so
        #    that if a later segment fails, already-uploaded segments are skipped
        #    by the watermark on the next cycle (not just by the coverage check).
        state.save_last_uploaded(window_end)

    log.info("=== Fitbit2Garmin cycle done ===")


def run_sleep_sync(cfg: dict, health: GoogleHealthClient, garmin: GarminClient, state: StateStore):
    """
    Experimental: upload Google Health sleep sessions as sleep_level FIT records.

    Off unless sync.sleep.enabled is true. A night is skipped if Garmin already has
    sleep for that date (e.g. from a real watch), is still inside the recency buffer,
    or ended at or before the sleep watermark.
    """
    sleep_cfg = cfg["sync"].get("sleep") or {}
    if not sleep_cfg.get("enabled"):
        return
    device = cfg["device"]
    recency_cutoff = datetime.now(timezone.utc) - timedelta(minutes=cfg["sync"].get("recency_minutes", 60))
    last_end = state.load_last_sleep_end()

    sessions = health.get_sleep_sessions(
        lookback_hours=sleep_cfg.get("lookback_hours", 36),
        min_minutes=sleep_cfg.get("min_minutes", 180),
    )
    for s in sessions:
        if last_end is not None and s["end"] <= last_end:
            continue
        if s["end"] > recency_cutoff:
            log.info("Sleep session ending %s is inside the recency buffer — later.", s["end"].isoformat())
            continue
        wake_date = (s["end"] + timedelta(seconds=s["utc_offset_seconds"])).strftime("%Y-%m-%d")
        if garmin.has_sleep(wake_date):
            log.info("Garmin already has sleep for %s — not uploading.", wake_date)
            state.save_last_sleep_end(s["end"])
            continue
        fit_bytes = build_sleep_fit(
            stages=s["stages"],
            manufacturer=device["manufacturer"],
            product_id=device["product_id"],
            serial_number=device["serial_number"],
            software_version=device.get("software_version", 331),
            utc_offset_seconds=s["utc_offset_seconds"],
        )
        result = garmin.upload_fit_for_window(fit_bytes, s["start"])
        log.info("Sleep upload for %s complete: %s", wake_date, result)
        state.save_last_sleep_end(s["end"])


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
        try:
            run_sleep_sync(cfg, health, garmin, state)
        except Exception:
            # Experimental feature: never fail the heart-rate sync because of it.
            log.exception("Sleep sync failed (continuing)")
        run_hook(hooks.get("on_success"))
    except Exception as e:
        log.exception("Sync cycle failed: %s", e)
        run_hook(hooks.get("on_failure"))
        sys.exit(1)


if __name__ == "__main__":
    main()
