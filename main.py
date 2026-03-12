"""
main.py
Shadow Sync orchestration loop.
Pulls Fitbit intraday data, encodes a Garmin Monitoring FIT file, and uploads it.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import yaml

from fitbit_client import FitbitClient
from fit_engine import build_monitoring_fit
from garmin_client import GarminClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("shadow_sync")

CONFIG_FILE = os.environ.get("CONFIG_FILE", os.path.join("data", "config.yaml"))
STATE_FILE = os.path.join(os.path.dirname(CONFIG_FILE), "state.json")


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
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {}
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError):
            pass
        data["last_uploaded_ts"] = dt.isoformat()
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
        log.debug("State saved: last_uploaded_ts=%s", dt.isoformat())


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
    `gap_minutes` apart (e.g. Fitbit was off-wrist or data was missing).
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


def run_sync(cfg: dict, fitbit: FitbitClient, garmin: GarminClient, state: StateStore):
    """Execute one full sync cycle."""
    log.info("=== Shadow Sync cycle starting ===")

    lookback = cfg["sync"].get("lookback_hours", 4)
    recency_minutes = cfg["sync"].get("recency_minutes", 60)
    device = cfg["device"]

    # 1. Pull Fitbit data
    fitbit.ensure_authorized()
    points, utc_offset_seconds = fitbit.get_combined_intraday(lookback_hours=lookback)

    if not points:
        log.warning("No Fitbit data returned for the last %dh — skipping.", lookback)
        return

    # Build a steps-since-local-midnight map from the FULL dataset before any filtering.
    # This lets us correctly offset the cumulative step counter in each FIT segment so
    # that Garmin sees cycles counting up from 0 at midnight — matching real device
    # behaviour — rather than restarting from 0 at the start of each uploaded window.
    local_offset = timedelta(seconds=utc_offset_seconds)
    _date_running: dict = {}
    steps_since_midnight: dict[datetime, int] = {}
    for pt in points:                           # points is already sorted ascending
        local_date = (pt["datetime"] + local_offset).date()
        _date_running[local_date] = _date_running.get(local_date, 0) + pt.get("steps_delta", 0)
        steps_since_midnight[pt["datetime"]] = _date_running[local_date]

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
        log.info("All Fitbit data is within the recency buffer — nothing to upload yet.")
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
        log.info("No new Fitbit data since last upload — nothing to do.")
        return

    # 4. Skip any minute already populated on Garmin (from any source —
    #    real device or a previous shadow upload).  This prevents overwriting
    #    data that beat us to Garmin, while the recency buffer makes it likely
    #    that real-device data arrived first for recent minutes.
    points = garmin.filter_covered_points(points)

    if not points:
        log.info("All candidate points are already covered on Garmin — nothing to upload.")
        return

    # 5. Split into contiguous segments
    segments = split_segments(points)
    log.info("%d contiguous segment(s) to upload.", len(segments))

    last_point_uploaded: datetime | None = None

    # 6. Build + upload one FIT file per segment
    for i, seg in enumerate(segments, 1):
        # Assign cumulative steps using the full-dataset midnight-relative map so that
        # cycles in the FIT file count up from 0 at local midnight (matching real device
        # behaviour).  When the segment crosses midnight the counter resets automatically
        # because the map is keyed by local date.  For points not in the map (shouldn't
        # happen in practice) fall back to the point's existing cumulative_steps value.
        for pt in seg:
            pt["cumulative_steps"] = steps_since_midnight.get(pt["datetime"],
                                                               pt.get("cumulative_steps", 0))

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
        last_point_uploaded = window_end

    # 7. Persist the watermark so the next cycle skips these points.
    if last_point_uploaded is not None:
        state.save_last_uploaded(last_point_uploaded)

    log.info("=== Shadow Sync cycle done ===")


def main():
    cfg = load_config(CONFIG_FILE)
    state = StateStore(STATE_FILE)

    fitbit = FitbitClient(
        client_id=cfg["fitbit"]["client_id"],
        client_secret=cfg["fitbit"]["client_secret"],
    )
    garmin = GarminClient(
        email=cfg["garmin"]["email"],
        password=cfg["garmin"]["password"],
    )
    garmin.connect()

    interval_minutes = cfg["sync"].get("interval_minutes", 120)
    interval_seconds = interval_minutes * 60

    log.info("Shadow Sync started. Interval: %d minutes.", interval_minutes)

    hooks = cfg.get("sync", {}).get("hooks", {})

    while True:
        try:
            run_sync(cfg, fitbit, garmin, state)
            run_hook(hooks.get("on_success"))
        except Exception as e:
            log.exception("Sync cycle failed: %s", e)
            run_hook(hooks.get("on_failure"))

        log.info("Sleeping %d minutes until next sync…", interval_minutes)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
