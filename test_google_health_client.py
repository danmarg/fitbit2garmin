import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from google_health_client import GoogleHealthClient, _parse_offset, _parse_time


def hr_point(ts, bpm, offset="3600s"):
    return {"heartRate": {"sampleTime": {"physicalTime": ts, "utcOffset": offset}, "beatsPerMinute": bpm}}


def step_point(start, end, count, offset="3600s"):
    return {"steps": {"interval": {"startTime": start, "endTime": end, "startUtcOffset": offset}, "count": count}}


class TestParsing(unittest.TestCase):
    def test_parse_helpers(self):
        self.assertEqual(_parse_offset("3600s"), 3600)
        self.assertEqual(_parse_offset("-18000s"), -18000)
        self.assertIsNone(_parse_offset(None))
        self.assertEqual(_parse_time("2026-03-11T10:00:05.250Z"), datetime(2026, 3, 11, 10, 0, 5, 250000, tzinfo=timezone.utc))

    def test_heart_rate_mean_per_minute(self):
        bpm, offs = GoogleHealthClient.parse_heart_rate([
            hr_point("2026-03-11T10:00:05Z", "60"),   # string int64
            hr_point("2026-03-11T10:00:35Z", 70),     # bare number
            hr_point("2026-03-11T10:01:05Z", "80"),
        ])
        m = datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(bpm[m], 65)
        self.assertEqual(bpm[m + timedelta(minutes=1)], 80)
        self.assertEqual(offs[m], 3600)

    def test_steps_single_and_multi_minute(self):
        steps, _ = GoogleHealthClient.parse_steps([
            step_point("2026-03-11T10:00:00Z", "2026-03-11T10:01:00Z", "12"),
            step_point("2026-03-11T10:05:00Z", "2026-03-11T10:08:00Z", "10"),
        ])
        m = datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(steps[m], 12)
        spread = [steps[m + timedelta(minutes=5 + i)] for i in range(3)]
        self.assertEqual(sum(spread), 10)
        self.assertEqual(spread, [4, 3, 3])


class TestCombined(unittest.TestCase):
    def test_cumulative_resets_at_local_midnight(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        client = GoogleHealthClient("id", "secret")
        offset = 3600
        # Place one step-minute before and one after the next local midnight relative to `now`.
        local_now = now + timedelta(seconds=offset)
        midnight_utc = local_now.replace(hour=0, minute=0) - timedelta(seconds=offset)
        before = midnight_utc - timedelta(minutes=1)
        after = midnight_utc
        if after > now:  # midnight is in the future: use the most recent past midnight
            before -= timedelta(days=1)
            after -= timedelta(days=1)
        iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
        steps = [step_point(iso(t), iso(t + timedelta(minutes=1)), str(n)) for t, n in ((before, 5), (after, 7))]
        hr = [hr_point(iso(after), 60), hr_point(iso(now - timedelta(minutes=1)), 61)]
        with patch.object(client, "_list_data_points", side_effect=lambda dt, *_: hr if dt == "heart-rate" else steps):
            points, utc_offset = client.get_combined_intraday(lookback_hours=24)
        self.assertEqual(utc_offset, offset)
        by_dt = {p["datetime"]: p for p in points}
        self.assertEqual(by_dt[after]["cumulative_steps"], 7)  # reset, not 12
        self.assertEqual(by_dt[after]["heart_rate"], 60)
        self.assertEqual(points, sorted(points, key=lambda p: p["datetime"]))


if __name__ == "__main__":
    unittest.main()
