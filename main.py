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
from sleep_sync import SleepStore, run_sleep_sync

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

    def load_backfill_attempts(self) -> dict[datetime, int]:
        """{UTC minute: attempts}. Entries that don't parse are ignored, never fatal."""
        raw = self._load().get("backfill_attempts")
        out: dict[datetime, int] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            try:
                dt = datetime.fromisoformat(k)
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if isinstance(v, int) and not isinstance(v, bool):
                out[dt] = v
        return out

    def save_backfill_attempts(self, attempts: dict[datetime, int]):
        self._store("backfill_attempts", {k.isoformat(): v for k, v in sorted(attempts.items())})



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
    `gap_minutes` apart (e.g. the tracker was off-wrist or data was missing), or
    their UTC offsets differ (a DST change), since a FIT file has one local offset.
    """
    if not points:
        return []
    segments = []
    current = [points[0]]
    for prev, pt in zip(points, points[1:]):
        delta = pt["datetime"] - prev["datetime"]
        if delta > timedelta(minutes=gap_minutes) or pt.get("utc_offset_seconds") != prev.get("utc_offset_seconds"):
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
    """Build and upload one segment; its own UTC offset wins over the current one."""
    utc_offset_seconds = seg[0].get("utc_offset_seconds", utc_offset_seconds)
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
    # Garmin may not show an upload in its daily views for a while; don't treat the
    # last cycle's minutes as missing yet, or they (and their steps) would be sent twice.
    backfill_settle_minutes = sync_cfg.get("backfill_settle_minutes", 90)
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
    #    Garmin reports heart rate about every 2 minutes, so a minute counts as covered
    #    when a Garmin sample is within 2 minutes of it.
    now = datetime.now(timezone.utc)
    last_uploaded = state.load_last_uploaded()
    if last_uploaded is None:
        # First run: upload only the lookback window as new data. Later cycles fill
        # holes Garmin has anywhere in the backfill window (before the watermark),
        # including minutes from before this tool was installed.
        first_cutoff = now - timedelta(hours=lookback)
        points = [p for p in points if p["datetime"] >= first_cutoff]
    new_points = [p for p in points if last_uploaded is None or p["datetime"] > last_uploaded]
    # Backfill candidates use the same window as the attempt counts below (otherwise a
    # minute whose count was pruned would be retried forever), and must have settled.
    backfill_cutoff = now - timedelta(hours=backfill_hours)
    settled = None if last_uploaded is None else last_uploaded - timedelta(minutes=backfill_settle_minutes)
    old_points = [p for p in points if settled is not None and backfill_cutoff <= p["datetime"] <= settled]

    by_date = garmin.get_coverage_by_date(points, utc_offset_seconds)
    if by_date is None:
        # Without Garmin's view every old minute would look missing; never backfill blind.
        log.warning("Garmin coverage unavailable — uploading new data only, no backfill this cycle.")
        old_points = []
    else:
        coverage = set().union(*by_date.values()) if by_date else set()
        # For new data, ignore Garmin samples at or before the watermark: those are the
        # tail of our previous upload and would otherwise hide the first new minutes.
        recent = coverage if last_uploaded is None else {m for m in coverage if m > last_uploaded}
        before = len(new_points)
        new_points = [p for p in new_points if not garmin.is_covered(p["datetime"], recent, 2)]
        if before - len(new_points):
            log.info("Skipping %d new point(s) already present on Garmin Connect.", before - len(new_points))
        # Backfill only dates where Garmin returned some heart rate: an empty day may be
        # a bad response, and re-sending a whole day on that basis is not worth the risk.
        empty = {d for d, mins in by_date.items() if not mins}
        if empty:
            log.info("Garmin has no heart rate for %s — not backfilling those date(s).", ", ".join(sorted(empty)))
        old_points = [
            p for p in old_points
            if garmin.local_date(p, utc_offset_seconds) not in empty
            and not garmin.is_covered(p["datetime"], coverage, 2)
        ]

    # 4. New data: one FIT file per contiguous segment, advancing the watermark after each.
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

    # 5. Backfill: minutes uploaded before but missing on Garmin now. Attempts are
    #    counted per minute, so a gap that shrinks or shifts keeps its count.
    attempts = {m: n for m, n in state.load_backfill_attempts().items() if m >= backfill_cutoff}
    runs = select_backfill_runs(old_points, min_minutes=backfill_min_minutes)
    if runs:
        log.info("%d backfill run(s): minutes uploaded earlier that Garmin no longer has.", len(runs))
    for run in runs:
        remaining = [p for p in run if attempts.get(p["datetime"], 0) < backfill_max_attempts]
        if len(remaining) < len(run):
            log.info("Backfill: %d minute(s) from %s given up after %d attempt(s).",
                     len(run) - len(remaining), f"{run[0]['datetime']:%m-%d %H:%M}", backfill_max_attempts)
        for seg in split_segments(remaining):
            tries = max(attempts.get(p["datetime"], 0) for p in seg)
            label = f"{seg[0]['datetime']:%m-%d %H:%M}–{seg[-1]['datetime']:%H:%M} ({len(seg)} points)"
            seg_offset = seg[0].get("utc_offset_seconds", utc_offset_seconds)
            # Steps were sent the first time; only re-send them if Garmin has none here.
            if garmin.steps_in_range(seg[0]["datetime"], seg[-1]["datetime"], seg_offset) > 0:
                seg = [{**p, "steps_delta": 0, "cumulative_steps": 0} for p in seg]
            log.info("Backfill %s, attempt %d.", label, tries + 1)
            for p in seg:
                attempts[p["datetime"]] = attempts.get(p["datetime"], 0) + 1
            state.save_backfill_attempts(attempts)
            # Shift time_created a few seconds earlier per attempt so Garmin doesn't
            # reject it as a duplicate of an earlier upload starting at the same minute
            # (kept under a minute so it can't collide with the minute before).
            try:
                result = _upload_segment(cfg, garmin, seg, utc_offset_seconds, time_created_offset=tries % 59 + 1)
                log.info("Backfill upload complete: %s", result)
            except Exception as e:
                # Backfill is best effort: new data is already uploaded; try the rest.
                log.warning("Backfill upload of %s failed: %s", label, e)
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

    failed = False
    try:
        run_sync(cfg, health, garmin, state)
    except Exception as e:
        log.exception("Sync cycle failed: %s", e)
        failed = True
    # Sleep runs even if the heart-rate sync failed, and vice versa.
    try:
        run_sleep_sync(cfg, health, garmin, SleepStore())
    except Exception as e:
        log.exception("Sleep sync failed: %s", e)
        failed = True

    if failed:
        run_hook(hooks.get("on_failure"))
        sys.exit(1)
    run_hook(hooks.get("on_success"))


if __name__ == "__main__":
    main()
