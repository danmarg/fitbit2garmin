"""
garmin_client.py
Wraps the garth library to authenticate with Garmin Connect and upload
binary FIT files via the device sync endpoint.
"""

import io
import logging
import os
from datetime import datetime

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
