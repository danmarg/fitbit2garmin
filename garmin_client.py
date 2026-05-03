"""
garmin_client.py
Wraps the garminconnect library to authenticate with Garmin Connect and upload
binary FIT files via the device sync endpoint.
"""

import io
import logging
import os
import time
from datetime import datetime, timezone

from garminconnect import Garmin

log = logging.getLogger(__name__)

# Default to ~/.garminconnect for consistency with the library's default token store
GARMIN_HOME = os.environ.get("STATE_DIRECTORY", os.path.expanduser("~/.garminconnect"))


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

    def get_covered_minutes(self, date: str) -> set[datetime]:
        """
        Return the set of UTC minute-datetimes that already have HR data on
        Garmin Connect for the given date (YYYY-MM-DD).
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
                    # Truncate to the minute so we match FitbitClient's 1-min resolution
                    covered.add(dt.replace(second=0, microsecond=0))
            log.info("Garmin already has %d covered minute(s) for %s", len(covered), date)
            return covered
        except Exception as e:
            log.info(
                "Could not fetch existing Garmin HR data for %s "
                "(wellness API unavailable, skipping coverage check): %s",
                date, e,
            )
            return set()

    def filter_covered_points(self, points: list[dict]) -> list[dict]:
        """
        Remove any points whose minute already has HR data on Garmin Connect.
        """
        if not points:
            return []

        dates_needed = {p["datetime"].strftime("%Y-%m-%d") for p in points}
        covered: set[datetime] = set()
        for date in dates_needed:
            covered |= self.get_covered_minutes(date)

        if not covered:
            return points

        filtered = [
            p for p in points
            if p["datetime"].replace(second=0, microsecond=0) not in covered
        ]
        removed = len(points) - len(filtered)
        if removed:
            log.info("Skipping %d point(s) already present on Garmin Connect.", removed)
        return filtered

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

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
                return resp
            except Exception as e:
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
