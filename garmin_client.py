"""
garmin_client.py
Wraps the garth library to authenticate with Garmin Connect and upload
binary FIT files via the device sync endpoint.
"""

import io
import logging
import os
from datetime import datetime, timedelta, timezone

import garth

log = logging.getLogger(__name__)

GARTH_HOME = os.environ.get("GARTH_HOME", os.path.expanduser("~/.garth"))


class GarminClient:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self._client: garth.Client | None = None

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
                return
            except Exception as e:
                log.info("Cached session invalid or incomplete (%s), re-authenticating…", e)

        log.info("Logging into Garmin Connect...")
        garth.login(self.email, self.password)
        
        # Ensure directory exists before saving
        os.makedirs(GARTH_HOME, exist_ok=True)
        garth.save(GARTH_HOME)
        self._client = garth.client
        log.info("Garmin authentication successful. Session saved to %s", GARTH_HOME)

    def _ensure_connected(self):
        if self._client is None:
            self.connect()

    # ------------------------------------------------------------------
    # Conflict check
    # ------------------------------------------------------------------

    GARMIN_EPOCH = 631065600  # 1989-12-31 00:00:00 UTC

    def get_real_device_windows(self, date: str) -> list[tuple[datetime, datetime]]:
        """
        Fetch the time ranges for `date` (YYYY-MM-DD) that already have real
        device HR data on Garmin Connect.

        The wellness HR endpoint returns heartRateValues as a list of
        [garmin_timestamp_ms, bpm] pairs. Consecutive readings from the real
        device form contiguous windows; we return those windows so callers can
        check overlap against the Fitbit window they want to upload.

        Returns a list of (start, end) datetime pairs (UTC), possibly empty.
        """
        self._ensure_connected()
        try:
            url = f"/wellness-service/wellness/dailyHeartRate/{date}"
            resp = self._client.connectapi(url)
            values = resp.get("heartRateValues") or []
            if not values:
                return []

            # values is [[epoch_ms, bpm], ...]; bpm can be None for gaps
            timestamps = sorted(
                v[0] / 1000.0  # ms → seconds
                for v in values
                if v[1] is not None  # skip null/gap entries
            )
            if not timestamps:
                return []

            # Group consecutive timestamps into windows (gap > 5 min = new window)
            GAP = 5 * 60
            windows = []
            seg_start = timestamps[0]
            seg_end = timestamps[0]
            for ts in timestamps[1:]:
                if ts - seg_end <= GAP:
                    seg_end = ts
                else:
                    windows.append((
                        datetime.fromtimestamp(seg_start, tz=timezone.utc),
                        datetime.fromtimestamp(seg_end, tz=timezone.utc),
                    ))
                    seg_start = seg_end = ts
            windows.append((
                datetime.fromtimestamp(seg_start, tz=timezone.utc),
                datetime.fromtimestamp(seg_end, tz=timezone.utc),
            ))

            log.info(
                "Garmin real-device windows for %s: %s",
                date,
                [(s.strftime("%H:%M"), e.strftime("%H:%M")) for s, e in windows],
            )
            return windows

        except Exception as e:
            log.warning("Could not fetch existing Garmin data: %s", e)
            return []

    def filter_points_to_gaps(self, points: list[dict]) -> list[dict]:
        """
        Remove any Fitbit data points that fall inside a real Forerunner window
        already on Garmin Connect.  What remains are only the minutes where the
        Forerunner was NOT syncing, which is what we want to fill in.

        Points spanning midnight are handled correctly because we query each
        affected date separately.
        """
        if not points:
            return []

        # Fetch real-device windows for every date touched by the points list.
        dates_needed = {p["datetime"].strftime("%Y-%m-%d") for p in points}
        real_windows: list[tuple[datetime, datetime]] = []
        for date in dates_needed:
            real_windows.extend(self.get_real_device_windows(date))

        if not real_windows:
            return points  # nothing to subtract

        def in_real_window(dt: datetime) -> bool:
            aware = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            for real_start, real_end in real_windows:
                if real_start <= aware <= real_end:
                    return True
            return False

        filtered = [p for p in points if not in_real_window(p["datetime"])]
        removed = len(points) - len(filtered)
        if removed:
            log.info(
                "Filtered out %d Fitbit point(s) covered by real Forerunner data.",
                removed,
            )
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
