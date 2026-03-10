"""
main.py
Shadow Sync orchestration loop.
Pulls Fitbit intraday data, encodes a Garmin Monitoring FIT file, and uploads it.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

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

CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.yaml")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        log.error("Config file not found: %s", path)
        log.error("Copy config.yaml.example to config.yaml and fill in your credentials.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def split_segments(points: list[dict], gap_minutes: int = 5) -> list[list[dict]]:
    """
    Split a list of per-minute points into contiguous segments.
    A new segment begins whenever two consecutive points are more than
    `gap_minutes` apart (i.e. the Forerunner window punched a hole in the data).
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


def run_sync(cfg: dict, fitbit: FitbitClient, garmin: GarminClient):
    """Execute one full sync cycle."""
    log.info("=== Shadow Sync cycle starting ===")

    lookback = cfg["sync"].get("lookback_hours", 4)
    device = cfg["device"]

    # 1. Pull Fitbit data
    fitbit.ensure_authorized()
    points = fitbit.get_combined_intraday(lookback_hours=lookback)

    if not points:
        log.warning("No Fitbit data returned for the last %dh — skipping.", lookback)
        return

    # 2. Remove minutes already covered by a real Forerunner sync
    points = garmin.filter_points_to_gaps(points)

    if not points:
        log.info("All Fitbit data is covered by real Forerunner data — nothing to upload.")
        return

    # 3. Split into contiguous segments (Forerunner windows may have punched holes)
    segments = split_segments(points)
    log.info("%d contiguous segment(s) to upload after gap removal.", len(segments))

    # 4. Build + upload one FIT file per segment
    for i, seg in enumerate(segments, 1):
        # Reset cumulative steps for this segment to start at 0
        current_cumulative = 0
        for pt in seg:
            current_cumulative += pt.get("steps_delta", 0)
            pt["cumulative_steps"] = current_cumulative

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
        )
        result = garmin.upload_fit_for_window(fit_bytes, window_start)
        log.info("Upload complete: %s", result)

    log.info("=== Shadow Sync cycle done ===")


def main():
    cfg = load_config(CONFIG_FILE)

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

    while True:
        try:
            run_sync(cfg, fitbit, garmin)
        except Exception as e:
            log.exception("Sync cycle failed: %s", e)

        log.info("Sleeping %d minutes until next sync…", interval_minutes)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
