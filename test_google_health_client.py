import unittest
import pytz  # import the real module before test_pipeline_logic stubs it
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from google_health_client import GoogleHealthClient, _parse_time


def hr_roll(ts, avg):
    return {"startTime": ts, "endTime": ts, "heartRate": {"beatsPerMinuteAvg": avg, "beatsPerMinuteMax": avg, "beatsPerMinuteMin": avg}}


def step_roll(ts, count):
    return {"startTime": ts, "endTime": ts, "steps": {"countSum": count}}


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestParsing(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(_parse_time("2026-03-11T10:00:05.250Z"), datetime(2026, 3, 11, 10, 0, 5, 250000, tzinfo=timezone.utc))

    def test_heart_rate_rollup(self):
        out = GoogleHealthClient.parse_heart_rate([hr_roll("2026-03-11T10:00:00Z", 64.6), {"startTime": "2026-03-11T10:01:00Z"}])
        self.assertEqual(out, {datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc): 65})

    def test_steps_rollup_int64_string(self):
        out = GoogleHealthClient.parse_steps([step_roll("2026-03-11T10:00:00Z", "72")])
        self.assertEqual(out, {datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc): 72})


class TestCombined(unittest.TestCase):
    def test_cumulative_resets_at_local_midnight(self):
        client = GoogleHealthClient("id", "secret")
        tz = pytz.timezone("Europe/Berlin")
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        midnight = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        if midnight >= now:
            midnight -= timedelta(days=1)
        before, after = midnight - timedelta(minutes=1), midnight
        steps = [step_roll(iso(before), "5"), step_roll(iso(after), "7")]
        hr = [hr_roll(iso(after), 60), hr_roll(iso(now - timedelta(minutes=1)), 61)]

        def fake_roll(data_type, start, end):
            return hr if data_type == "heart-rate" else steps

        with patch.object(client, "get_user_timezone", return_value="Europe/Berlin"), \
             patch.object(client, "_roll_up_minutes", side_effect=fake_roll):
            points, offset = client.get_combined_intraday(lookback_hours=24)
        self.assertIn(offset, (3600, 7200))
        by_dt = {p["datetime"]: p for p in points}
        self.assertEqual(by_dt[after]["cumulative_steps"], 7)  # reset at local midnight, not 12
        self.assertEqual(by_dt[after]["heart_rate"], 60)
        self.assertEqual(points, sorted(points, key=lambda p: p["datetime"]))

    def test_rollup_request_shape_and_paging(self):
        client = GoogleHealthClient("id", "secret")
        pages = [{"rollupDataPoints": [step_roll("2026-03-11T10:00:00Z", "1")], "nextPageToken": "t"},
                 {"rollupDataPoints": [step_roll("2026-03-11T10:01:00Z", "2")]}]
        calls = []

        class R:
            ok = True
            def __init__(self, d): self.d = d
            def json(self): return self.d

        def fake_post(url, headers=None, json=None):
            calls.append((url, dict(json)))
            return R(pages[len(calls) - 1])

        start = datetime(2026, 3, 11, 10, tzinfo=timezone.utc)
        with patch.object(client, "_auth_headers", return_value={}), patch.object(client.http, "post", side_effect=fake_post):
            out = client._roll_up_minutes("steps", start, start + timedelta(minutes=2))
        self.assertEqual(len(out), 2)
        self.assertTrue(calls[0][0].endswith("/steps/dataPoints:rollUp"))
        self.assertEqual(calls[0][1]["windowSize"], "60s")
        self.assertEqual(calls[0][1]["range"]["startTime"], "2026-03-11T10:00:00Z")
        self.assertEqual(calls[1][1]["pageToken"], "t")


if __name__ == "__main__":
    unittest.main()
