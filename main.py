"""
main.py
Fitbit2Garmin orchestration loop.
Pulls Fitbit intraday data, encodes a Garmin Monitoring FIT file, and uploads it.
"""

import json
import logging
import os
import subprocess
import sys
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
log = logging.getLogger("fitbit2garmin")

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
    log.info("=== Fitbit2Garmin cycle starting ===")

    lookback = cfg["sync"].get("lookback_hours", 4)
    recency_minutes = cfg["sync"].get("recency_minutes", 60)
    device = cfg["device"]

    # 1. Pull Fitbit data
    fitbit.ensure_authorized()
    points, utc_offset_seconds = fitbit.get_combined_intraday(lookback_hours=lookback)

    if not points:
        log.warning("No Fitbit data returned for the last %dh — skipping.", lookback)
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

    log.info("Fitbit2Garmin starting single sync cycle.")

    hooks = cfg.get("sync", {}).get("hooks", {})

    try:
        run_sync(cfg, fitbit, garmin, state)
        run_hook(hooks.get("on_success"))
    except Exception as e:
        log.exception("Sync cycle failed: %s", e)
        run_hook(hooks.get("on_failure"))
        sys.exit(1)


if __name__ == "__main__":
    main()
