"""
google_health_client.py
Handles OAuth2 authentication and data fetching from the Google Health API
(https://developers.google.com/health), the successor to the Fitbit Web API.
Token state is persisted to google_health_token.json and refreshed automatically.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dateutil import parser as dtparser

log = logging.getLogger(__name__)

TOKEN_FILE = os.environ.get(
    "GOOGLE_HEALTH_TOKEN_FILE",
    os.path.join(
        os.path.dirname(os.environ.get("CONFIG_FILE", os.path.join("data", "config.yaml"))),
        "google_health_token.json",
    ),
)
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://health.googleapis.com/v4/users/me/dataTypes"  # {type}/dataPoints:rollUp
REDIRECT_URI = "http://localhost:8080/"
SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.settings.readonly",
]
SETTINGS_URL = "https://health.googleapis.com/v4/users/me/settings"
PAGE_SIZE = 10000


def _parse_time(value: str) -> datetime:
    """Parse an RFC 3339 timestamp into an aware UTC datetime."""
    dt = dtparser.isoparse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


class GoogleHealthClient:
    def __init__(self, client_id: str, client_secret: str, default_timezone: str = "UTC"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.default_timezone = default_timezone
        self.http = requests.Session()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _load_token(self) -> dict | None:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                return json.load(f) or None
        return None

    def _save_token(self, token: dict):
        """Save the token to disk, merging with existing data to preserve the refresh_token."""
        existing = self._load_token() or {}
        existing.update(token)
        os.makedirs(os.path.dirname(TOKEN_FILE) or ".", exist_ok=True)
        # The token file holds a long-lived refresh_token; keep it owner-only.
        fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2)
        os.chmod(TOKEN_FILE, 0o600)
        log.debug("Token saved to %s", TOKEN_FILE)

    def authorize(self):
        """Run the OAuth2 authorization flow interactively (first-time setup)."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            # offline + consent guarantees Google returns a refresh_token
            "access_type": "offline",
            "prompt": "consent",
        }
        print(f"\nOpen this URL in your browser:\n  {AUTH_URL}?{urlencode(params)}\n")
        redirect_response = input("Paste the full redirect URL here: ").strip()
        code = parse_qs(urlparse(redirect_response).query).get("code", [None])[0]
        if not code:
            raise ValueError("No 'code' parameter found in the pasted redirect URL.")
        resp = self.http.post(
            TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
        )
        if not resp.ok:
            log.error("Google token exchange failed: %d %s", resp.status_code, resp.text)
            resp.raise_for_status()
        token = resp.json()
        token["expires_at"] = time.time() + token["expires_in"]
        self._save_token(token)
        print("Authorization successful. Token saved.")

    def _refresh_token(self):
        """Exchange the refresh_token for a new access token."""
        token = self._load_token()
        if not token or not token.get("refresh_token"):
            raise RuntimeError(
                f"No refresh token in {TOKEN_FILE} — re-run the interactive authorization (README step 4)."
            )
        _TRANSIENT = {429, 500, 502, 503, 504}
        _MAX_ATTEMPTS = 3
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            resp = self.http.post(
                TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": token["refresh_token"],
                    "grant_type": "refresh_token",
                },
            )
            if resp.status_code == 200:
                break
            if resp.status_code in _TRANSIENT and attempt < _MAX_ATTEMPTS:
                delay = 5 * attempt  # 5 s, 10 s
                log.warning(
                    "Google token refresh attempt %d/%d failed (%d) — retrying in %ds…",
                    attempt, _MAX_ATTEMPTS, resp.status_code, delay,
                )
                time.sleep(delay)
            else:
                log.error("Google token refresh failed: %d %s", resp.status_code, resp.text)
                resp.raise_for_status()
        new_token = resp.json()
        new_token["expires_at"] = time.time() + new_token["expires_in"]
        self._save_token(new_token)
        log.info("Google access token refreshed successfully.")

    def _ensure_token_fresh(self):
        """Proactively refresh the token if it is expired or about to expire."""
        token = self._load_token()
        if token and time.time() > token.get("expires_at", 0) - 60:
            log.info("Google token expired or expiring soon — refreshing…")
            self._refresh_token()

    def ensure_authorized(self):
        """Authorize if no token exists yet, otherwise refresh if expired."""
        if not os.path.exists(TOKEN_FILE):
            if not sys.stdin.isatty():
                raise RuntimeError(
                    f"No Google Health token at {TOKEN_FILE} and no terminal to authorize with — "
                    "run the interactive authorization once (README step 4)."
                )
            self.authorize()
        else:
            self._ensure_token_fresh()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        token = self._load_token()
        return {"Authorization": f"Bearer {token['access_token']}"}

    def get_user_timezone(self) -> str:
        """Fetch the user's IANA timezone (e.g., 'America/New_York') from their account settings."""
        resp = self.http.get(SETTINGS_URL, headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json().get("timeZone") or self.default_timezone

    def _roll_up_minutes(self, data_type: str, start: datetime, end: datetime) -> list[dict]:
        """
        POST users/me/dataTypes/{data_type}/dataPoints:rollUp with 60s windows over
        [start, end). The API reconciles overlapping sources and drops off-wrist
        periods; each result is {"startTime", "endTime", <dataType camelCase>: {...}}.
        """
        body = {
            "range": {
                "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "windowSize": "60s",
            "pageSize": PAGE_SIZE,
        }
        points: list[dict] = []
        while True:
            resp = self.http.post(
                f"{API_BASE}/{data_type}/dataPoints:rollUp", headers=self._auth_headers(), json=body
            )
            if not resp.ok:
                log.error("Google Health %s rollUp failed: %d %s", data_type, resp.status_code, resp.text)
                resp.raise_for_status()
            data = resp.json()
            points.extend(data.get("rollupDataPoints", []))
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            body["pageToken"] = next_token
        log.info("Google Health %s: %d minute window(s) since %s", data_type, len(points), start.isoformat())
        return points

    @staticmethod
    def parse_heart_rate(rollups: list[dict]) -> dict[datetime, int]:
        """Map each 60s heart-rate rollup to {UTC minute: mean bpm}."""
        out: dict[datetime, int] = {}
        for r in rollups:
            avg = (r.get("heartRate") or {}).get("beatsPerMinuteAvg")
            if r.get("startTime") is None or avg is None:
                continue
            out[_floor_minute(_parse_time(r["startTime"]))] = round(float(avg))
        return out

    @staticmethod
    def parse_steps(rollups: list[dict]) -> dict[datetime, int]:
        """Map each 60s steps rollup to {UTC minute: steps}. countSum is an int64 string."""
        out: dict[datetime, int] = {}
        for r in rollups:
            count = (r.get("steps") or {}).get("countSum")
            if r.get("startTime") is None or count is None:
                continue
            out[_floor_minute(_parse_time(r["startTime"]))] = int(count)
        return out

    def get_combined_intraday(self, lookback_hours: int = 4) -> tuple[list[dict], int]:
        """
        Fetch HR + Steps for the last `lookback_hours` hours, merged by minute,
        in UTC, with day boundaries taken from the user's Google Health timezone.

        Returns a tuple of:
          - list of dicts:
              [{"datetime": datetime (UTC aware), "heart_rate": int, "steps_delta": int, "cumulative_steps": int}, ...]
              sorted ascending by time.
          - utc_offset_seconds: the user's current UTC offset in whole seconds (e.g. 3600 for UTC+1)
        """
        import pytz

        try:
            tz_name = self.get_user_timezone()
            log.info("Google Health user timezone: %s", tz_name)
            local_tz = pytz.timezone(tz_name)
        except Exception as e:
            log.warning("Could not fetch timezone, using %r: %s", self.default_timezone, e)
            try:
                local_tz = pytz.timezone(self.default_timezone)
            except Exception:
                local_tz = pytz.utc

        now_utc = datetime.now(timezone.utc)
        cutoff_utc = _floor_minute(now_utc - timedelta(hours=lookback_hours))
        end_utc = _floor_minute(now_utc) + timedelta(minutes=1)

        # cumulative_steps is steps-since-local-midnight, so steps must be fetched from the
        # start of the cutoff's local day.
        cutoff_local = cutoff_utc.astimezone(local_tz)
        start_of_day_local = cutoff_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_day_utc = start_of_day_local.astimezone(timezone.utc)

        hr_by_minute = self.parse_heart_rate(self._roll_up_minutes("heart-rate", cutoff_utc, end_utc))
        steps_by_minute = self.parse_steps(self._roll_up_minutes("steps", start_of_day_utc, end_utc))

        merged = []
        cumulative_steps = 0
        current_local_date = None

        for dt in sorted(set(hr_by_minute) | set(steps_by_minute)):
            local_date = dt.astimezone(local_tz).date()
            if local_date != current_local_date:
                current_local_date = local_date
                cumulative_steps = 0  # Reset at local midnight

            steps_delta = steps_by_minute.get(dt, 0)
            cumulative_steps += steps_delta

            if dt < cutoff_utc or dt > now_utc:
                continue

            merged.append(
                {
                    "datetime": dt,
                    "heart_rate": hr_by_minute.get(dt, 0),
                    "steps_delta": steps_delta,
                    "cumulative_steps": cumulative_steps,
                }
            )

        log.info("Got %d merged intraday points for last %dh", len(merged), lookback_hours)
        # Current UTC offset for the user's timezone (handles DST automatically)
        utc_offset = now_utc.astimezone(local_tz).utcoffset()
        utc_offset_seconds = int(utc_offset.total_seconds()) if utc_offset is not None else 0
        log.info("UTC offset for FIT local_timestamp: %+ds (%+.1fh)", utc_offset_seconds, utc_offset_seconds / 3600)
        return merged, utc_offset_seconds
