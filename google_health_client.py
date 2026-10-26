"""
google_health_client.py
Handles OAuth2 authentication and data fetching from the Google Health API
(https://developers.google.com/health), the successor to the Fitbit Web API.
Token state is persisted to google_health_token.json and refreshed automatically.
"""

import json
import logging
import os
import time
from collections import defaultdict
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
API_BASE = "https://health.googleapis.com/v4/users/me/dataTypes"
REDIRECT_URI = "http://localhost:8080/"
SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
]
PAGE_SIZE = 10000


def _parse_time(value: str) -> datetime:
    """Parse an RFC 3339 timestamp into an aware UTC datetime."""
    dt = dtparser.isoparse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_offset(value) -> int | None:
    """Parse a UTC offset such as '3600s' (or a bare number) into whole seconds."""
    if value is None:
        return None
    try:
        return int(float(str(value).rstrip("s")))
    except ValueError:
        return None


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
        with open(TOKEN_FILE, "w") as f:
            json.dump(existing, f, indent=2)
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
            log.warning("No refresh token available — re-authorization required.")
            return
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
            self.authorize()
        else:
            self._ensure_token_fresh()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _list_data_points(self, data_type: str, time_field: str, start: datetime) -> list[dict]:
        """
        Page through users/me/dataTypes/{data_type}/dataPoints for points at or
        after `start`. `time_field` is the snake_case filter path for the point's
        start time (e.g. "steps.interval.start_time").
        """
        token = self._load_token()
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        params = {
            "filter": f'{time_field} >= "{start.strftime("%Y-%m-%dT%H:%M:%SZ")}"',
            "pageSize": PAGE_SIZE,
        }
        points: list[dict] = []
        while True:
            resp = self.http.get(f"{API_BASE}/{data_type}/dataPoints", headers=headers, params=params)
            if not resp.ok:
                log.error("Google Health %s list failed: %d %s", data_type, resp.status_code, resp.text)
                resp.raise_for_status()
            body = resp.json()
            points.extend(body.get("dataPoints", []))
            next_token = body.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token
        log.info("Google Health %s: %d data point(s) since %s", data_type, len(points), start.isoformat())
        return points

    @staticmethod
    def parse_heart_rate(data_points: list[dict]) -> tuple[dict[datetime, int], dict[datetime, int]]:
        """
        Collapse point-in-time HR samples into one value per UTC minute (mean bpm).

        Returns (bpm_by_minute, utc_offset_by_minute).
        """
        samples: dict[datetime, list[int]] = defaultdict(list)
        offsets: dict[datetime, int] = {}
        for dp in data_points:
            hr = dp.get("heartRate") or {}
            sample_time = hr.get("sampleTime") or {}
            raw_time = sample_time.get("physicalTime")
            bpm = hr.get("beatsPerMinute")
            if raw_time is None or bpm is None:
                continue
            minute = _floor_minute(_parse_time(raw_time))
            samples[minute].append(int(bpm))
            offset = _parse_offset(sample_time.get("utcOffset"))
            if offset is not None:
                offsets[minute] = offset
        return {m: round(sum(v) / len(v)) for m, v in samples.items()}, offsets

    @staticmethod
    def parse_steps(data_points: list[dict]) -> tuple[dict[datetime, int], dict[datetime, int]]:
        """
        Spread step intervals across the UTC minutes they cover.

        Returns (steps_by_minute, utc_offset_by_minute).
        """
        steps: dict[datetime, int] = defaultdict(int)
        offsets: dict[datetime, int] = {}
        for dp in data_points:
            interval = (dp.get("steps") or {}).get("interval") or {}
            raw_start = interval.get("startTime")
            raw_end = interval.get("endTime")
            count = (dp.get("steps") or {}).get("count")
            if raw_start is None or count is None:
                continue
            start = _parse_time(raw_start)
            end = _parse_time(raw_end) if raw_end else start
            first = _floor_minute(start)
            # Minutes touched by [start, end); a zero-length interval counts as one minute.
            n_minutes = max(1, int((_floor_minute(end - timedelta(microseconds=1)) - first).total_seconds() // 60) + 1) \
                if end > start else 1
            total = int(count)
            base, remainder = divmod(total, n_minutes)
            offset = _parse_offset(interval.get("startUtcOffset"))
            for i in range(n_minutes):
                minute = first + timedelta(minutes=i)
                steps[minute] += base + (1 if i < remainder else 0)
                if offset is not None:
                    offsets[minute] = offset
        return dict(steps), offsets

    def _fallback_offset_seconds(self, at: datetime) -> int:
        import pytz

        try:
            tz = pytz.timezone(self.default_timezone)
        except pytz.UnknownTimeZoneError:
            log.warning("Unknown timezone %r — using UTC.", self.default_timezone)
            tz = pytz.utc
        offset = at.astimezone(tz).utcoffset()
        return int(offset.total_seconds()) if offset is not None else 0

    def get_combined_intraday(self, lookback_hours: int = 4) -> tuple[list[dict], int]:
        """
        Fetch HR + Steps for the last `lookback_hours` hours, merged by minute, in UTC.

        Returns a tuple of:
          - list of dicts:
              [{"datetime": datetime (UTC aware), "heart_rate": int, "steps_delta": int, "cumulative_steps": int}, ...]
              sorted ascending by time.
          - utc_offset_seconds: the user's current UTC offset in whole seconds (e.g. 3600 for UTC+1)
        """
        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(hours=lookback_hours)

        # cumulative_steps is steps-since-local-midnight, so steps must be fetched from the
        # start of the cutoff's local day. That is at most 24h before the cutoff.
        steps_start = cutoff_utc - timedelta(hours=24)

        hr_by_minute, hr_offsets = self.parse_heart_rate(
            self._list_data_points("heart-rate", "heart_rate.sample_time.physical_time", cutoff_utc)
        )
        steps_by_minute, steps_offsets = self.parse_steps(
            self._list_data_points("steps", "steps.interval.start_time", steps_start)
        )

        offsets = {**steps_offsets, **hr_offsets}
        if offsets:
            utc_offset_seconds = offsets[max(offsets)]  # most recent observed offset
        else:
            utc_offset_seconds = self._fallback_offset_seconds(now_utc)

        def offset_at(dt: datetime) -> int:
            return offsets.get(dt, utc_offset_seconds)

        all_times = sorted(set(hr_by_minute) | set(steps_by_minute))
        merged = []
        cumulative_steps = 0
        current_local_date = None

        for dt in all_times:
            local_date = (dt + timedelta(seconds=offset_at(dt))).date()
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
        log.info("UTC offset for FIT local_timestamp: %+ds (%+.1fh)", utc_offset_seconds, utc_offset_seconds / 3600)
        return merged, utc_offset_seconds
