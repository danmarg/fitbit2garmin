"""
garmin_client.py
Wraps the garth library to authenticate with Garmin Connect and upload
binary FIT files via the device sync endpoint.
"""

import io
import logging
import os
from datetime import datetime, timezone

import garth

log = logging.getLogger(__name__)

GARTH_HOME = os.environ.get("GARTH_HOME", os.path.expanduser("~/.garth"))


class GarminClient:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self._client: garth.Client | None = None
        self._display_name: str | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def connect(self):
        """Authenticate (reuse cached session if available)."""
        garth.configure(domain="garmin.com")

        # Only attempt resume if the directory exists AND contains tokens.
        # This prevents FileNotFoundError if the user created an empty volume mount.
        if os.path.isdir(GARTH_HOME) and os.path.isfile(os.path.join(GARTH_HOME, "oauth1_token.json")):
            try:
                garth.resume(GARTH_HOME)
                garth.client.username  # trigger validation
                log.info("Resumed existing Garmin session from %s", GARTH_HOME)
                self._client = garth.client
                self._display_name = self._fetch_display_name()
                return
            except Exception as e:
                log.info("Cached session invalid or incomplete (%s), re-authenticating…", e)

        log.info("Logging into Garmin Connect...")
        garth.login(self.email, self.password)

        # Ensure directory exists before saving
        os.makedirs(GARTH_HOME, exist_ok=True)
        garth.save(GARTH_HOME)
        self._client = garth.client
        self._display_name = self._fetch_display_name()
        log.info("Garmin authentication successful. Session saved to %s", GARTH_HOME)

    def _fetch_display_name(self) -> str:
        """Fetch the UUID display name required by the wellness API."""
        resp = self._client.connectapi("/userprofile-service/socialProfile")
        return resp["displayName"]

    def _ensure_connected(self):
        if self._client is None:
            self.connect()

    # ------------------------------------------------------------------
    # Covered-minute check
    # ------------------------------------------------------------------

    def get_covered_minutes(self, date: str) -> set[datetime]:
        """
        Return the set of UTC minute-datetimes that already have HR data on
        Garmin Connect for the given date (YYYY-MM-DD), regardless of whether
        that data came from a real device or a previous shadow-sync upload.

        Used to avoid uploading Fitbit data over a minute that is already
        populated (by any source).  Falls back to an empty set on 403/error
        so callers can continue safely without this check.
        """
        self._ensure_connected()
        try:
            url = f"/wellness-service/wellness/dailyHeartRate/{self._display_name}?date={date}"
            resp = self._client.connectapi(url)
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

        This prevents uploading Fitbit data over any minute that's already
        populated — whether from a real device sync or a previous shadow-sync
        upload.  When the wellness API is unavailable (403), all points pass
        through so the StateStore watermark acts as the fallback guard.
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

        upload_url = "/upload-service/upload"
        files = {
            "file": (filename, io.BytesIO(fit_bytes), "application/octet-stream")
        }

        log.info("Uploading %d-byte FIT file to Garmin (%s)…", len(fit_bytes), filename)
        resp = self._client.connectapi(
            upload_url,
            method="POST",
            files=files,
        )
        log.info("Garmin upload response: %s", resp)
        return resp

    def upload_fit_for_window(
        self,
        fit_bytes: bytes,
        window_start: datetime,
    ) -> dict:
        date_str = window_start.strftime("%Y-%m-%d")
        filename = f"shadow_sync_{date_str}_{window_start.strftime('%H%M')}.fit"
        return self.upload_fit(fit_bytes, filename=filename)
