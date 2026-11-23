"""
garmin_client.py
Wraps the garminconnect library to authenticate with Garmin Connect and upload
binary FIT files via the device sync endpoint.
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from garminconnect import Garmin

log = logging.getLogger(__name__)

# Default to ~/.garminconnect for consistency with the library's default token store
GARMIN_HOME = os.environ.get("STATE_DIRECTORY", os.path.expanduser("~/.garminconnect"))


def _parse_gmt(value: str) -> datetime:
    """Garmin's "2026-11-22T23:00:00.0" (fraction optional), as UTC; works before Python 3.11."""
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in value else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


class GarminClient:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self._client: Garmin | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def connect(self):
        """Authenticate (reuse cached session if available)."""
        # Ensure directory exists before using it
        os.makedirs(GARMIN_HOME, exist_ok=True)

        self._client = Garmin(self.email, self.password)

        try:
            log.info("Attempting to resume Garmin session from %s", GARMIN_HOME)
            self._client.login(GARMIN_HOME)
            log.info("Garmin authentication successful (resumed or new login).")
        except Exception as e:
            log.error("Garmin authentication failed: %s", e)
            raise

    def _ensure_connected(self):
        if self._client is None:
            self.connect()

    # ------------------------------------------------------------------
    # Covered-minute check
    # ------------------------------------------------------------------

    def get_covered_minutes(self, date: str) -> set[datetime] | None:
        """
        Return the set of UTC minute-datetimes that already have HR data on
        Garmin Connect for the given local date (YYYY-MM-DD), or None if the
        wellness API could not be read.
        """
        self._ensure_connected()
        try:
            resp = self._client.get_heart_rates(date)
            values = resp.get("heartRateValues") or []
            covered = set()
            for v in values:
                if v[1] is not None:  # skip null/gap entries
                    # heartRateValues timestamps are in ms since Unix epoch
                    dt = datetime.fromtimestamp(v[0] / 1000.0, tz=timezone.utc)
                    # Truncate to the minute so we match the source data's 1-min resolution
                    covered.add(dt.replace(second=0, microsecond=0))
            log.info("Garmin already has %d covered minute(s) for %s", len(covered), date)
            return covered
        except Exception as e:
            log.warning("Could not fetch existing Garmin HR data for %s: %s", date, e)
            return None

    @staticmethod
    def local_date(point: dict, utc_offset_seconds: int = 0) -> str:
        """The point's local calendar date, using its own offset when it carries one."""
        offset = point.get("utc_offset_seconds", utc_offset_seconds)
        return (point["datetime"] + timedelta(seconds=offset)).strftime("%Y-%m-%d")

    def get_coverage_by_date(
        self, points: list[dict], utc_offset_seconds: int = 0
    ) -> dict[str, set[datetime]] | None:
        """
        {local date: minutes with HR on Garmin} for every local date the points touch,
        or None if any date could not be read. Garmin's daily heart rate is per local date.
        """
        by_date: dict[str, set[datetime]] = {}
        for date in sorted({self.local_date(p, utc_offset_seconds) for p in points}):
            day = self.get_covered_minutes(date)
            if day is None:
                return None
            by_date[date] = day
        return by_date

    @staticmethod
    def is_covered(dt: datetime, covered: set[datetime], tolerance_minutes: int = 0) -> bool:
        m = dt.replace(second=0, microsecond=0)
        return any(m + timedelta(minutes=k) in covered
                   for k in range(-tolerance_minutes, tolerance_minutes + 1))

    def steps_in_range(self, start: datetime, end: datetime, utc_offset_seconds: int = 0) -> int:
        """
        Steps Garmin already has in the 15-minute buckets overlapping [start, end].
        On error, returns 1 so callers assume steps exist and don't send them twice.
        """
        self._ensure_connected()
        offset = timedelta(seconds=utc_offset_seconds)
        dates = {(start + offset).strftime("%Y-%m-%d"), (end + offset).strftime("%Y-%m-%d")}
        total = 0
        try:
            for date in sorted(dates):
                for b in self._client.get_steps_data(date) or []:
                    b_start = _parse_gmt(b["startGMT"])
                    if start - timedelta(minutes=15) < b_start <= end:
                        total += b.get("steps") or 0
        except Exception as e:
            log.warning("Could not fetch Garmin steps for %s–%s (%s) — assuming present.", start, end, e)
            return 1
        return total

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def get_sleep_summary(self, date: str) -> dict | None:
        """
        Garmin's sleep for the night ending on local `date` (YYYY-MM-DD):
        {"start", "end" (UTC datetimes), "score" (int | None), "need" (minutes | None),
         "stress" (average sleep stress | None), "next_need" (need for the following night),
         "asleep" (minutes), "hr" (average sleeping heart rate | None)},
        or None if Garmin has no sleep that night. Errors propagate to the caller.
        """
        self._ensure_connected()
        data = self._client.get_sleep_data(date) or {}
        dto = data.get("dailySleepDTO") or {}
        if not (dto.get("sleepTimeSeconds") and dto.get("sleepStartTimestampGMT") and dto.get("sleepEndTimestampGMT")):
            return None  # no sleep, or an incomplete record: treat both as nothing there
        ts = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return {
            "start": ts(dto["sleepStartTimestampGMT"]),
            "end": ts(dto["sleepEndTimestampGMT"]),
            "score": ((dto.get("sleepScores") or {}).get("overall") or {}).get("value"),
            "need": (dto.get("sleepNeed") or {}).get("actual"),
            "stress": dto.get("avgSleepStress"),
            "next_need": (dto.get("nextSleepNeed") or {}).get("actual"),
            "asleep": dto["sleepTimeSeconds"] / 60,
            "hr": dto.get("avgHeartRate"),
        }

    def upload_fit(self, fit_bytes: bytes, filename: str = "monitoring.fit") -> dict:
        """
        Upload a raw FIT file to Garmin Connect via the binary upload endpoint.

        Returns the JSON response from Garmin.
        """
        self._ensure_connected()

        _MAX_ATTEMPTS = 3

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            log.info("Uploading %d-byte FIT file to Garmin (%s)…", len(fit_bytes), filename)
            try:
                # The upload_activity method in garminconnect expects a file path.
                # However, our FIT files are generated in-memory.
                # Since garminconnect doesn't seem to expose a direct binary upload
                # for activities in its high-level API easily, we use the internal
                # client's post method which is what upload_activity uses under the hood.
                files = {"file": (filename, io.BytesIO(fit_bytes), "application/octet-stream")}
                # Use the same endpoint as garth did
                resp = self._client.client.post("connectapi", "/upload-service/upload", files=files, api=True)
                log.info("Garmin upload response: %s", resp)
                if "contained no content" in str(resp):
                    log.warning("Garmin stored nothing from %s (\"contained no content\").", filename)
                return resp
            except Exception as e:
                if str(e).startswith("API Error 409") or "Duplicate Wellness File" in str(e):
                    # Garmin already has this file; retrying can never succeed.
                    log.warning("Garmin reports a duplicate wellness file — treating as already uploaded: %s", e)
                    return {"status": "duplicate"}
                if attempt < _MAX_ATTEMPTS:
                    delay = 5 * attempt  # 5 s, 10 s
                    log.warning(
                        "Garmin upload attempt %d/%d failed (%s) — retrying in %ds…",
                        attempt, _MAX_ATTEMPTS, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    def upload_fit_for_window(
        self,
        fit_bytes: bytes,
        window_start: datetime,
    ) -> dict:
        # Include upload-time seconds in the filename so retries never collide with
        # a previously uploaded file for the same window (Garmin returns 409 on
        # filename re-use even when the content has changed).
        date_str = window_start.strftime("%Y-%m-%d")
        upload_ts = datetime.now(timezone.utc).strftime("%H%M%S")
        filename = f"fitbit2garmin_{date_str}_{window_start.strftime('%H%M')}_{upload_ts}.fit"
        return self.upload_fit(fit_bytes, filename=filename)
