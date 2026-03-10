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

import requests
from requests_oauthlib import OAuth2Session
from requests_oauthlib.compliance_fixes import fitbit_compliance_fix

log = logging.getLogger(__name__)

TOKEN_FILE = os.environ.get("FITBIT_TOKEN_FILE", os.path.join(os.path.dirname(os.environ.get("CONFIG_FILE", "config.yaml")), "token.json"))
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
            auto_refresh_url=TOKEN_URL,
            auto_refresh_kwargs={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            token_updater=self._save_token,
        )
        fitbit_compliance_fix(session)
        return session

    def _load_token(self) -> dict | None:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                return json.load(f)
        return None

    def _save_token(self, token: dict):
        with open(TOKEN_FILE, "w") as f:
            json.dump(token, f, indent=2)
        log.debug("Token saved to %s", TOKEN_FILE)

    def authorize(self):
        """Run the OAuth2 authorization flow interactively (first-time setup)."""
        auth_url, _ = self.session.authorization_url(AUTH_URL)
        print(f"\nOpen this URL in your browser:\n  {auth_url}\n")
        redirect_response = input("Paste the full redirect URL here: ").strip()
        token = self.session.fetch_token(
            TOKEN_URL,
            authorization_response=redirect_response,
            client_secret=self.client_secret,
        )
        self._save_token(token)
        print("Authorization successful. Token saved.")

    def ensure_authorized(self):
        """Authorize if no token exists yet."""
        if not os.path.exists(TOKEN_FILE):
            self.authorize()

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

    def get_combined_intraday(self, lookback_hours: int = 4) -> list[dict]:
        """
        Fetch HR + Steps for the last `lookback_hours` hours, merged by minute,
        converted to UTC based on the user's Fitbit profile timezone.

        Returns a list of dicts:
            [{"datetime": datetime (UTC aware), "heart_rate": int, "steps_delta": int, "cumulative_steps": int}, ...]
        sorted ascending by time.
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
        # We look at 'now' and 'cutoff' in the user's local time.
        now_local = now_utc.astimezone(local_tz)
        cutoff_local = cutoff_utc.astimezone(local_tz)

        dates_needed = []
        curr_d = cutoff_local.date()
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

        # Merge and filter to the lookback window (UTC)
        all_times = sorted(set(hr_by_minute) | set(steps_by_minute))
        merged = []
        cumulative_steps = 0
        for dt in all_times:
            if dt < cutoff_utc or dt > now_utc:
                continue
            steps_delta = steps_by_minute.get(dt, 0)
            cumulative_steps += steps_delta
            merged.append(
                {
                    "datetime": dt,
                    "heart_rate": hr_by_minute.get(dt, 0),
                    "steps_delta": steps_delta,
                    "cumulative_steps": cumulative_steps,
                }
            )

        log.info("Got %d merged intraday points for last %dh", len(merged), lookback_hours)
        return merged
