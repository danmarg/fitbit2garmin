"""
fitbit_client.py
Handles OAuth2 authentication and intraday data fetching from the Fitbit Web API.
Token state is persisted to token.json and refreshed automatically.
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta, timezone

import base64
import requests
from requests_oauthlib import OAuth2Session
from requests_oauthlib.compliance_fixes import fitbit_compliance_fix

log = logging.getLogger(__name__)

TOKEN_FILE = os.environ.get(
    "FITBIT_TOKEN_FILE",
    os.path.join(os.path.dirname(os.environ.get("CONFIG_FILE", os.path.join("data", "config.yaml"))), "token.json"),
)
AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
API_BASE = "https://api.fitbit.com/1/user/-"


class FitbitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = self._init_session()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _init_session(self) -> OAuth2Session:
        token = self._load_token()
        session = OAuth2Session(
            client_id=self.client_id,
            redirect_uri="http://localhost:8080/",
            scope=["heartrate", "activity", "profile"],
            token=token,
            # We handle refresh manually in _ensure_token_fresh to ensure
            # the correct Basic Auth headers are sent, which auto_refresh
            # often gets wrong for Fitbit.
            token_updater=self._save_token,
        )
        fitbit_compliance_fix(session)
        return session

    def _load_token(self) -> dict | None:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                token = json.load(f)
                if token:
                    log.debug("Loaded token from %s. Fields: %s", TOKEN_FILE, list(token.keys()))
                return token
        return None

    def _save_token(self, token: dict):
        """Save the token to disk, merging with existing data to preserve refresh_tokens."""
        existing = self._load_token() or {}
        existing.update(token)
        with open(TOKEN_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        log.debug("Token saved to %s", TOKEN_FILE)

    def authorize(self):
        """Run the OAuth2 authorization flow interactively (first-time setup)."""
        auth_url, _ = self.session.authorization_url(AUTH_URL)
        print(f"\nOpen this URL in your browser:\n  {auth_url}\n")
        redirect_response = input("Paste the full redirect URL here: ").strip()
        # Allow http://localhost redirect URIs during the local OAuth flow.
        # requests-oauthlib rejects http:// by default; this flag disables that check.
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
        token = self.session.fetch_token(
            TOKEN_URL,
            authorization_response=redirect_response,
            client_secret=self.client_secret,
        )
        self._save_token(token)
        print("Authorization successful. Token saved.")

    def _refresh_token(self):
        """Refresh the Fitbit access token using Basic auth (required by Fitbit API).

        requests-oauthlib's built-in auto_refresh sends credentials as form data,
        but Fitbit requires them in the Authorization header as Basic base64(id:secret).
        This method handles the refresh correctly and re-initialises the session.
        """
        token = self._load_token()
        if not token or not token.get("refresh_token"):
            log.warning("No refresh token available — re-authorization required.")
            return
        creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        _TRANSIENT = {429, 500, 502, 503, 504}
        _MAX_ATTEMPTS = 3
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            resp = requests.post(
                TOKEN_URL,
                headers={
                    "Authorization": f"Basic {creds}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "refresh_token", "refresh_token": token["refresh_token"]},
            )
            if resp.status_code == 200:
                break
            if resp.status_code in _TRANSIENT and attempt < _MAX_ATTEMPTS:
                delay = 5 * attempt  # 5 s, 10 s
                log.warning(
                    "Fitbit token refresh attempt %d/%d failed (%d) — retrying in %ds…",
                    attempt, _MAX_ATTEMPTS, resp.status_code, delay,
                )
                time.sleep(delay)
            else:
                log.error("Fitbit token refresh failed: %d %s", resp.status_code, resp.text)
                resp.raise_for_status()
        new_token = resp.json()
        new_token["expires_at"] = time.time() + new_token["expires_in"]
        self._save_token(new_token)
        # Re-init session so subsequent requests use the new access token.
        self.session = self._init_session()
        log.info("Fitbit access token refreshed successfully.")

    def _ensure_token_fresh(self):
        """Proactively refresh the token if it is expired or about to expire."""
        token = self._load_token()
        if token and time.time() > token.get("expires_at", 0) - 60:
            log.info("Fitbit token expired or expiring soon — refreshing…")
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

    def get_intraday_heart_rate(self, date: str, detail_level: str = "1min") -> list[dict]:
        """
        Fetch intraday heart-rate data.

        Returns a list of dicts: [{"time": "HH:MM:SS", "value": bpm}, ...]
        """
        url = f"{API_BASE}/activities/heart/date/{date}/1d/{detail_level}.json"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        dataset = data.get("activities-heart-intraday", {}).get("dataset", [])
        log.info("Fitbit HR dataset size for %s: %d", date, len(dataset))
        return dataset

    def get_intraday_steps(self, date: str, detail_level: str = "1min") -> list[dict]:
        """
        Fetch intraday steps data.

        Returns a list of dicts: [{"time": "HH:MM:SS", "value": steps_in_minute}, ...]
        """
        url = f"{API_BASE}/activities/steps/date/{date}/1d/{detail_level}.json"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        dataset = data.get("activities-steps-intraday", {}).get("dataset", [])
        log.info("Fitbit Steps dataset size for %s: %d", date, len(dataset))
        return dataset

    def get_user_timezone(self) -> str:
        """Fetch the user's timezone string (e.g., 'America/New_York') from their profile."""
        url = "https://api.fitbit.com/1/user/-/profile.json"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json().get("user", {}).get("timezone", "UTC")

    def get_combined_intraday(self, lookback_hours: int = 4) -> tuple[list[dict], int]:
        """
        Fetch HR + Steps for the last `lookback_hours` hours, merged by minute,
        converted to UTC based on the user's Fitbit profile timezone.

        Returns a tuple of:
          - list of dicts:
              [{"datetime": datetime (UTC aware), "heart_rate": int, "steps_delta": int, "cumulative_steps": int}, ...]
              sorted ascending by time.
          - utc_offset_seconds: the user's current UTC offset in whole seconds (e.g. 3600 for UTC+1)
        """
        import pytz

        try:
            tz_name = self.get_user_timezone()
            log.info("Fitbit user timezone: %s", tz_name)
            local_tz = pytz.timezone(tz_name)
        except Exception as e:
            log.warning("Could not fetch timezone, defaulting to UTC: %s", e)
            local_tz = pytz.utc

        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(hours=lookback_hours)

        # Generate all dates (in local time) that might contain our UTC window
        now_local = now_utc.astimezone(local_tz)
        cutoff_local = cutoff_utc.astimezone(local_tz)

        # To ensure cumulative_steps reflects steps-since-midnight, we must fetch 
        # from the start of the earliest day in our lookback window.
        start_of_day_local = cutoff_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_day_utc = start_of_day_local.astimezone(timezone.utc)

        dates_needed = []
        curr_d = start_of_day_local.date()
        while curr_d <= now_local.date():
            dates_needed.append(curr_d)
            curr_d += timedelta(days=1)

        hr_by_minute: dict[datetime, int] = {}
        steps_by_minute: dict[datetime, int] = {}

        for date in sorted(dates_needed):
            date_str = date.strftime("%Y-%m-%d")
            log.info("Fetching Fitbit intraday data for %s", date_str)

            try:
                hr_data = self.get_intraday_heart_rate(date_str)
                for point in hr_data:
                    # Fitbit returns time in local wall-clock
                    naive_dt = datetime.strptime(f"{date_str} {point['time']}", "%Y-%m-%d %H:%M:%S")
                    # Localize and convert to UTC
                    utc_dt = local_tz.localize(naive_dt).astimezone(timezone.utc)
                    hr_by_minute[utc_dt] = point["value"]
            except Exception as e:
                log.warning("Could not fetch HR data for %s: %s", date_str, e)

            try:
                steps_data = self.get_intraday_steps(date_str)
                for point in steps_data:
                    naive_dt = datetime.strptime(f"{date_str} {point['time']}", "%Y-%m-%d %H:%M:%S")
                    utc_dt = local_tz.localize(naive_dt).astimezone(timezone.utc)
                    steps_by_minute[utc_dt] = point["value"]
            except Exception as e:
                log.warning("Could not fetch steps data for %s: %s", date_str, e)

        # Merge and calculate cumulative steps from the start of the earliest day
        all_times = sorted(set(hr_by_minute) | set(steps_by_minute))
        merged = []
        cumulative_steps = 0
        current_local_date = None
        
        for dt in all_times:
            # We must process ALL points from start_of_day_utc to compute cumulative_steps
            # but only append to merged if they are within our lookback window.
            if dt < start_of_day_utc:
                continue
                
            local_dt = dt.astimezone(local_tz)
            if local_dt.date() != current_local_date:
                current_local_date = local_dt.date()
                cumulative_steps = 0 # Reset at local midnight
                
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
        # Compute the current UTC offset for the user's timezone (handles DST automatically)
        utc_offset = now_local.utcoffset()
        utc_offset_seconds = int(utc_offset.total_seconds()) if utc_offset is not None else 0
        log.info("UTC offset for FIT local_timestamp: %+ds (%+.1fh)", utc_offset_seconds, utc_offset_seconds / 3600)
        return merged, utc_offset_seconds
