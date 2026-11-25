
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# Stub out heavy runtime dependencies so tests run without a full install.
for _mod in ("garth", "requests_oauthlib", "requests_oauthlib.compliance_fixes",
             "yaml", "pytz"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

try:
    import fitparse
    HAS_FITPARSE = True
except ImportError:
    HAS_FITPARSE = False

from fit_engine import build_monitoring_fit, GARMIN_EPOCH

@unittest.skipUnless(HAS_FITPARSE, "fitparse not installed")
class TestFitGeneration(unittest.TestCase):
    def test_build_and_parse_monitoring_fit(self):
        # 1. Setup Mock Data (2 hours of data)
        start_dt = datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
        mock_points = []
        for i in range(120):
            dt = start_dt + timedelta(minutes=i)
            mock_points.append({
                "datetime": dt,
                "heart_rate": 70 + (i % 10),
                "steps_delta": 10,
                "cumulative_steps": (i + 1) * 10
            })

        # 2. Spoof IDs
        manufacturer = 1
        product_id = 3993
        serial_number = 3445298263

        # 3. Build FIT file
        fit_bytes = build_monitoring_fit(
            points=mock_points,
            manufacturer=manufacturer,
            product_id=product_id,
            serial_number=serial_number
        )

        # 4. Save to disk
        with open("test_output.fit", "wb") as f:
            f.write(fit_bytes)

        # 5. Parse back and Validate
        fitfile = fitparse.FitFile("test_output.fit")
        
        # Verify File ID
        file_id = next(fitfile.get_messages("file_id"))
        # Some fitparse versions map 4 to 'activity' instead of 'monitoring'
        self.assertIn(file_id.get_value("type"), ["monitoring", "monitoring_a", "monitoring_b", "monitoring_daily", "activity"])
        self.assertEqual(file_id.get_value("product"), product_id)

        # Verify Monitoring Messages: 2 records per point (activity + HR)
        monitoring_msgs = list(fitfile.get_messages("monitoring"))
        self.assertEqual(len(monitoring_msgs), 240)

        # Activity records have steps + activity_type, no heart_rate.
        # HR records have heart_rate only — Garmin ignores HR when activity_type is
        # in the same record, so they must be separate.
        activity_msgs = [m for m in monitoring_msgs if m.get_value("activity_type") is not None]
        hr_msgs = [m for m in monitoring_msgs if m.get_value("heart_rate") is not None]
        self.assertEqual(len(activity_msgs), 120)
        self.assertEqual(len(hr_msgs), 120)

        first_activity = activity_msgs[0]
        print("\nDebug: Fields found in first activity monitoring message:")
        for field in first_activity:
            print(f"  {field.name} ({field.def_num}): {field.value}")

        # 10 steps in the first minute; cycles are cumulative within the segment.
        self.assertEqual(first_activity.get_value("steps"), 10)
        self.assertEqual(activity_msgs[-1].get_value("steps"), 1200)
        self.assertEqual(first_activity.get_value("activity_type"), "walking")
        self.assertIsNone(first_activity.get_value("heart_rate"))
        self.assertEqual(hr_msgs[0].get_value("heart_rate"), 70)

        print(f"\nSUCCESS: Engine logic verified. FIT file correctly encoded with {len(monitoring_msgs)} points.")

@unittest.skipUnless(HAS_FITPARSE, "fitparse not installed")
class TestFileCreatedTime(unittest.TestCase):
    def test_time_created_is_first_data_point(self):
        base = datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
        pts = [{"datetime": base + timedelta(minutes=i), "heart_rate": 60, "steps_delta": 0, "cumulative_steps": 0}
               for i in range(5)]
        fit_bytes = build_monitoring_fit(points=pts, manufacturer=1, product_id=3993, serial_number=1)
        file_id = next(fitparse.FitFile(io.BytesIO(fit_bytes)).get_messages("file_id"))
        self.assertEqual(file_id.get_value("time_created").replace(tzinfo=timezone.utc), base)


@unittest.skipUnless(HAS_FITPARSE, "fitparse not installed")
class TestMissingHeartRate(unittest.TestCase):
    def test_no_hr_record_for_minutes_without_heart_rate(self):
        base = datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
        pts = [
            {"datetime": base, "heart_rate": 70, "steps_delta": 0, "cumulative_steps": 0},
            {"datetime": base + timedelta(minutes=1), "heart_rate": None, "steps_delta": 5, "cumulative_steps": 5},
            {"datetime": base + timedelta(minutes=2), "heart_rate": 0, "steps_delta": 0, "cumulative_steps": 5},
        ]
        fit_bytes = build_monitoring_fit(points=pts, manufacturer=1, product_id=3993, serial_number=1)
        msgs = list(fitparse.FitFile(io.BytesIO(fit_bytes)).get_messages("monitoring"))
        hr = [m.get_value("heart_rate") for m in msgs if m.get_value("heart_rate") is not None]
        self.assertEqual(hr, [70])


@unittest.skipUnless(HAS_FITPARSE, "fitparse not installed")
class TestLocalTimestamp(unittest.TestCase):
    """Verify that the FIT monitoring_info local_timestamp reflects the UTC offset."""

    def _parse_monitoring_info(self, fit_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as f:
            f.write(fit_bytes)
            name = f.name
        try:
            fitfile = fitparse.FitFile(name)
            return next(fitfile.get_messages("monitoring_info"))
        finally:
            os.unlink(name)

    def _make_points(self, n=1):
        start = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
        return [{"datetime": start + timedelta(minutes=i),
                 "heart_rate": 70, "steps_delta": 5, "cumulative_steps": (i+1)*5}
                for i in range(n)]

    def test_local_timestamp_utc(self):
        """With offset=0 local_timestamp == timestamp."""
        points = self._make_points()
        fit_bytes = build_monitoring_fit(points, 1, 3993, 12345, utc_offset_seconds=0)
        msg = self._parse_monitoring_info(fit_bytes)
        self.assertEqual(msg.get_value("timestamp"), msg.get_value("local_timestamp"))

    def test_local_timestamp_positive_offset(self):
        """UTC+1 (3600 s) → local_timestamp is 3600 s ahead of timestamp."""
        points = self._make_points()
        fit_bytes = build_monitoring_fit(points, 1, 3993, 12345, utc_offset_seconds=3600)
        msg = self._parse_monitoring_info(fit_bytes)
        ts = msg.get_value("timestamp")
        local_ts = msg.get_value("local_timestamp")
        self.assertEqual((local_ts - ts).total_seconds(), 3600)

    def test_local_timestamp_negative_offset(self):
        """UTC-5 (-18000 s) → local_timestamp is 18000 s behind timestamp."""
        points = self._make_points()
        fit_bytes = build_monitoring_fit(points, 1, 3993, 12345, utc_offset_seconds=-18000)
        msg = self._parse_monitoring_info(fit_bytes)
        ts = msg.get_value("timestamp")
        local_ts = msg.get_value("local_timestamp")
        self.assertEqual((local_ts - ts).total_seconds(), -18000)


    def test_monitoring_info_activity_type(self):
        """Verify monitoring_info has activity_type=generic (0)."""
        points = self._make_points()
        fit_bytes = build_monitoring_fit(points, 1, 3993, 12345, utc_offset_seconds=0)
        msg = self._parse_monitoring_info(fit_bytes)
        self.assertEqual(msg.get_value("activity_type"), "generic")

class TestSplitSegments(unittest.TestCase):
    """Unit tests for the contiguous-segment splitter in main.py."""

    @staticmethod
    def _pts(minutes):
        base = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
        return [{"datetime": base + timedelta(minutes=m)} for m in minutes]

    def setUp(self):
        # Import here so the module-level CONFIG_FILE default doesn't matter
        from main import split_segments
        self.split = split_segments

    def test_empty(self):
        self.assertEqual(self.split([]), [])

    def test_single_point(self):
        pts = self._pts([0])
        self.assertEqual(self.split(pts), [pts])

    def test_contiguous(self):
        pts = self._pts(range(10))
        segs = self.split(pts)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0], pts)

    def test_local_midnight_splits(self):
        # 22:50-23:09 UTC at UTC+1 crosses local midnight at 23:00 UTC.
        base = datetime(2026, 11, 21, 22, 50, tzinfo=timezone.utc)
        pts = [{"datetime": base + timedelta(minutes=m), "utc_offset_seconds": 3600} for m in range(20)]
        segs = self.split(pts)
        self.assertEqual([len(s) for s in segs], [10, 10])
        self.assertEqual(segs[1][0]["datetime"], datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc))

    def test_gap_splits(self):
        # 0-4 contiguous, then a 10-minute gap, then 15-19
        pts = self._pts(list(range(5)) + list(range(15, 20)))
        segs = self.split(pts)
        self.assertEqual(len(segs), 2)
        self.assertEqual(len(segs[0]), 5)
        self.assertEqual(len(segs[1]), 5)

    def test_gap_exactly_at_threshold_is_split(self):
        # gap_minutes default is 5; a delta of 6 should split
        pts = self._pts([0, 6])
        segs = self.split(pts)
        self.assertEqual(len(segs), 2)

    def test_gap_at_threshold_is_not_split(self):
        # delta of exactly 5 minutes should stay in one segment
        pts = self._pts([0, 5])
        segs = self.split(pts)
        self.assertEqual(len(segs), 1)

    def test_utc_offset_change_splits(self):
        # Clocks go back: one FIT file can only carry one local offset.
        pts = self._pts(range(6))
        for i, p in enumerate(pts):
            p["utc_offset_seconds"] = 7200 if i < 3 else 3600
        self.assertEqual([len(s) for s in self.split(pts)], [3, 3])


class TestStateStore(unittest.TestCase):
    """Unit tests for the StateStore watermark persistence."""

    def setUp(self):
        from main import StateStore
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "state.json")
        self.store = StateStore(self.path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_returns_none_when_missing(self):
        self.assertIsNone(self.store.load_last_uploaded())

    def test_roundtrip(self):
        dt = datetime(2026, 3, 11, 12, 30, 0, tzinfo=timezone.utc)
        self.store.save_last_uploaded(dt)
        loaded = self.store.load_last_uploaded()
        self.assertEqual(loaded, dt)

    def test_save_preserves_existing_keys(self):
        """save_last_uploaded should not clobber unrelated keys in state.json."""
        with open(self.path, "w") as f:
            json.dump({"other_key": "hello"}, f)
        dt = datetime(2026, 3, 11, 14, 0, 0, tzinfo=timezone.utc)
        self.store.save_last_uploaded(dt)
        with open(self.path) as f:
            data = json.load(f)
        self.assertEqual(data["other_key"], "hello")
        self.assertIn("last_uploaded_ts", data)

    def test_load_handles_corrupt_file(self):
        with open(self.path, "w") as f:
            f.write("not json{{{")
        self.assertIsNone(self.store.load_last_uploaded())

    def test_creates_parent_directory(self):
        nested_path = os.path.join(self.tmp, "subdir", "state.json")
        from main import StateStore
        store = StateStore(nested_path)
        dt = datetime(2026, 3, 11, 9, 0, 0, tzinfo=timezone.utc)
        store.save_last_uploaded(dt)
        self.assertTrue(os.path.exists(nested_path))


class TestCoverage(unittest.TestCase):
    """GarminClient.get_coverage_by_date / is_covered, as run_sync uses them (mocked API)."""

    def setUp(self):
        from garmin_client import GarminClient
        self.GarminClient = GarminClient
        self.client = GarminClient.__new__(GarminClient)
        self.client._client = MagicMock()

    def test_queries_each_local_date_once(self):
        day1 = datetime(2026, 3, 11, 23, 58, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 3, 12, 0, 2, 0, tzinfo=timezone.utc)
        with patch.object(self.client, "get_covered_minutes", return_value=set()) as gcm:
            by_date = self.client.get_coverage_by_date([{"datetime": day1}, {"datetime": day2}])
        self.assertEqual({c.args[0] for c in gcm.call_args_list}, {"2026-03-11", "2026-03-12"})
        self.assertEqual(set(by_date), {"2026-03-11", "2026-03-12"})

    def test_dates_are_local(self):
        # 23:30 UTC is already the next day in UTC+1.
        pts = [{"datetime": datetime(2026, 3, 11, 23, 30, tzinfo=timezone.utc)}]
        with patch.object(self.client, "get_covered_minutes", return_value=set()) as gcm:
            self.client.get_coverage_by_date(pts, utc_offset_seconds=3600)
        self.assertEqual([c.args[0] for c in gcm.call_args_list], ["2026-03-12"])

    def test_none_if_any_date_unreadable(self):
        with patch.object(self.client, "get_covered_minutes", return_value=None):
            self.assertIsNone(self.client.get_coverage_by_date([{"datetime": datetime(2026, 3, 11, tzinfo=timezone.utc)}]))

    def test_minute_near_a_garmin_sample_is_covered(self):
        # Garmin samples every 2 minutes; minutes 0..6 around samples at 0 and 4 are covered.
        base = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc)
        covered = {base, base + timedelta(minutes=4)}
        missing = [m for m in range(10) if not self.GarminClient.is_covered(base + timedelta(minutes=m), covered, 2)]
        self.assertEqual(missing, [7, 8, 9])
        self.assertFalse(self.GarminClient.is_covered(base + timedelta(minutes=1), covered, 0))

    def test_parse_gmt(self):
        from garmin_client import _parse_gmt
        expected = datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc)
        self.assertEqual((_parse_gmt("2026-11-22T23:00:00.0"), _parse_gmt("2026-11-22T23:00:00")), (expected, expected))


class TestZeroStepSegment(unittest.TestCase):
    """Garmin stores nothing from a file whose cycles stay at zero."""

    def _cycles(self, pts):
        msgs = fitparse.FitFile(io.BytesIO(build_monitoring_fit(pts, 1, 3993, 1))).get_messages("monitoring")
        return [(m.get("cycles").raw_value, m.get_value("activity_type")) for m in msgs if m.get("cycles")]

    def test_zero_step_segment_gets_one_walking_cycle(self):
        base = datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc)
        pts = [{"datetime": base + timedelta(minutes=i), "heart_rate": 55, "steps_delta": 0, "cumulative_steps": 0}
               for i in range(3)]
        cycles = self._cycles(pts)
        self.assertEqual([c for c, _ in cycles], [1, 1, 1])
        self.assertEqual(cycles[0][1], "walking")

    def test_segment_with_steps_is_unchanged(self):
        base = datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc)
        pts = [{"datetime": base, "heart_rate": 55, "steps_delta": 0, "cumulative_steps": 100},
               {"datetime": base + timedelta(minutes=1), "heart_rate": 56, "steps_delta": 4, "cumulative_steps": 104}]
        self.assertEqual([c for c, _ in self._cycles(pts)], [0, 4])

    def test_cycles_do_not_reset_at_local_midnight(self):
        base = datetime(2026, 3, 10, 22, 58, tzinfo=timezone.utc)
        # cumulative_steps (since local midnight) resets between the 2nd and 3rd minute.
        pts = [{"datetime": base, "heart_rate": 55, "steps_delta": 3, "cumulative_steps": 900},
               {"datetime": base + timedelta(minutes=1), "heart_rate": 55, "steps_delta": 2, "cumulative_steps": 902},
               {"datetime": base + timedelta(minutes=2), "heart_rate": 55, "steps_delta": 4, "cumulative_steps": 4}]
        self.assertEqual([c for c, _ in self._cycles(pts)], [3, 5, 9])

    def test_time_created_offset(self):
        base = datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc)
        pts = [{"datetime": base, "heart_rate": 55, "steps_delta": 0, "cumulative_steps": 0}]
        fit_bytes = build_monitoring_fit(pts, 1, 3993, 1, time_created_offset=2)
        file_id = next(fitparse.FitFile(io.BytesIO(fit_bytes)).get_messages("file_id"))
        self.assertEqual(file_id.get_value("time_created").replace(tzinfo=timezone.utc), base - timedelta(seconds=2))


class TestRunSyncBackfill(unittest.TestCase):
    def setUp(self):
        import main
        self.main = main
        self.tmp = tempfile.mkdtemp()
        self.state = main.StateStore(os.path.join(self.tmp, "state.json"))
        self.cfg = {"sync": {"recency_minutes": 60, "backfill_hours": 48},
                    "device": {"manufacturer": 1, "product_id": 3993, "serial_number": 1}}
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.old_start = now - timedelta(hours=6)
        # 30 old minutes (before the watermark) and 10 new ones (after it, outside recency).
        self.points = [{"datetime": self.old_start + timedelta(minutes=i), "heart_rate": 60,
                        "steps_delta": 2, "cumulative_steps": 2 * (i + 1)} for i in range(30)]
        self.new_start = now - timedelta(hours=3)
        self.points += [{"datetime": self.new_start + timedelta(minutes=i), "heart_rate": 70,
                         "steps_delta": 0, "cumulative_steps": 60} for i in range(10)]
        self.state.save_last_uploaded(now - timedelta(hours=4))
        self.health = MagicMock()
        self.health.get_combined_intraday.return_value = (self.points, 3600)
        from garmin_client import GarminClient
        self.garmin = MagicMock()
        # One Garmin sample on each date, far from every point, so no date counts as empty.
        far = datetime(2000, 1, 1, tzinfo=timezone.utc)
        self.garmin.get_coverage_by_date.side_effect = lambda pts, off: {
            GarminClient.local_date(p, off): {far} for p in pts
        }
        self.garmin.is_covered.side_effect = GarminClient.is_covered
        self.garmin.local_date.side_effect = GarminClient.local_date
        self.garmin.steps_in_range.return_value = 0
        self.uploaded = []
        self.main.build_monitoring_fit = self._capture

    def tearDown(self):
        from fit_engine import build_monitoring_fit
        self.main.build_monitoring_fit = build_monitoring_fit

    def _capture(self, points, **kw):
        self.uploaded.append((points, kw.get("time_created_offset", 0)))
        return b"fit"

    def _run(self):
        self.main.run_sync(self.cfg, self.health, self.garmin, self.state)

    def test_new_then_backfill(self):
        self._run()
        self.assertEqual(len(self.uploaded), 2)
        new, backfill = self.uploaded
        self.assertEqual(new[0][0]["datetime"], self.new_start)
        self.assertEqual(new[1], 0)
        self.assertEqual(backfill[0][0]["datetime"], self.old_start)
        self.assertEqual(backfill[1], 1)  # time_created shifted off the original upload
        self.assertEqual(sum(p["steps_delta"] for p in backfill[0]), 60)  # Garmin had no steps there
        self.assertEqual(self.state.load_last_uploaded(), self.new_start + timedelta(minutes=9))

    def test_backfill_strips_steps_garmin_already_has(self):
        self.garmin.steps_in_range.return_value = 25
        self._run()
        self.assertEqual(sum(p["steps_delta"] for p in self.uploaded[1][0]), 0)

    def test_backfill_gives_up_after_max_attempts(self):
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)  # backfill only
        for expected_offset in (1, 2, 3):
            self.uploaded.clear()
            self._run()
            self.assertEqual([off for _, off in self.uploaded], [expected_offset])
        self.uploaded.clear()
        self._run()
        self.assertEqual(self.uploaded, [])

    def test_no_backfill_when_garmin_coverage_fails(self):
        self.garmin.get_coverage_by_date.side_effect = None
        self.garmin.get_coverage_by_date.return_value = None
        self._run()
        self.assertEqual(len(self.uploaded), 1)  # new data only
        self.assertEqual(self.uploaded[0][0][0]["datetime"], self.new_start)

    def _coverage(self, minutes):
        from garmin_client import GarminClient
        self.garmin.get_coverage_by_date.side_effect = lambda pts, off: {
            GarminClient.local_date(p, off): set(minutes) for p in pts
        }

    def test_no_backfill_for_a_date_garmin_returns_empty(self):
        self._coverage([])
        self._run()
        self.assertEqual(len(self.uploaded), 1)  # new data only

    def test_tail_of_previous_upload_does_not_hide_new_minutes(self):
        # Garmin's sample at the watermark is ours; the next minutes are still new.
        watermark = self.new_start - timedelta(minutes=1)
        self.state.save_last_uploaded(watermark)
        self._coverage([watermark, datetime(2000, 1, 1, tzinfo=timezone.utc)])
        self.health.get_combined_intraday.return_value = (self.points[30:], 3600)
        self._run()
        self.assertEqual(self.uploaded[0][0][0]["datetime"], self.new_start)

    def test_new_minutes_between_real_watch_samples_are_skipped(self):
        # The real watch has samples every 2 min across the new window.
        self._coverage([self.new_start + timedelta(minutes=m) for m in range(0, 10, 2)])
        self.health.get_combined_intraday.return_value = (self.points[30:], 3600)
        self._run()
        self.assertEqual(self.uploaded, [])

    def test_first_run_uploads_only_the_lookback_window(self):
        os.remove(self.state.path)
        self.cfg["sync"]["lookback_hours"] = 4
        self._run()
        self.assertEqual(len(self.uploaded), 1)
        self.assertEqual(self.uploaded[0][0][0]["datetime"], self.new_start)  # the 6h-old run is not sent

    def test_attempt_cap_holds_when_the_gap_start_moves(self):
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)
        offsets = []
        for drop in range(0, 4):
            # Each cycle a little more of the gap's start has been restored on Garmin.
            self.health.get_combined_intraday.return_value = (self.points[drop * 2:30], 3600)
            self.uploaded.clear()
            self._run()
            offsets.append([off for _, off in self.uploaded])
        self.assertEqual(offsets, [[1], [2], [3], []])

    def test_corrupt_backfill_state_is_ignored(self):
        with open(self.state.path, "w") as f:
            json.dump({"last_uploaded_ts": (self.new_start - timedelta(hours=1)).isoformat(),
                       "backfill_attempts": {"not-a-date": 1, self.old_start.isoformat(): "x"}}, f)
        self._run()  # must not raise
        self.assertEqual(len(self.uploaded), 2)

    def test_segment_uses_its_own_utc_offset(self):
        seen = []
        self.main.build_monitoring_fit = lambda points, **kw: seen.append(kw["utc_offset_seconds"]) or b"fit"
        for p in self.points[:30]:
            p["utc_offset_seconds"] = 7200  # before a DST change
        self._run()
        self.assertEqual(sorted(seen), [3600, 7200])

    def test_backfill_limited_to_backfill_window_when_lookback_is_longer(self):
        # lookback 72h > backfill 5h: the 6h-old gap is outside the backfill window, so it
        # is never retried (its attempt count would be pruned each cycle).
        self.cfg["sync"].update({"lookback_hours": 72, "backfill_hours": 5})
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)
        for _ in range(4):
            self._run()
        self.assertEqual(self.uploaded, [])

    def test_backfill_hours_zero_disables_backfill(self):
        self.cfg["sync"]["backfill_hours"] = 0
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)
        self._run()
        self.assertEqual(self.uploaded, [])

    def test_recent_uploads_not_backfilled_before_they_settle(self):
        # The watermark is just after the old run, so Garmin may not show it yet.
        self.state.save_last_uploaded(self.old_start + timedelta(minutes=60))
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)
        self._run()
        self.assertEqual(self.uploaded, [])

    def test_failed_backfill_upload_does_not_stop_the_rest(self):
        second = [{**p, "datetime": p["datetime"] - timedelta(hours=1)} for p in self.points[:30]]
        self.health.get_combined_intraday.return_value = (second + self.points[:30], 3600)
        calls = []

        def flaky(points, **kw):
            calls.append(points[0]["datetime"])
            if len(calls) == 1:
                raise RuntimeError("Garmin said no")
            return b"fit"
        self.main.build_monitoring_fit = flaky
        self._run()  # must not raise
        self.assertEqual(calls, [second[0]["datetime"], self.old_start])

    def test_time_created_shift_stays_within_its_seconds(self):
        self.cfg["sync"]["backfill_max_attempts"] = 200
        self.health.get_combined_intraday.return_value = (self.points[:30], 3600)
        attempts = {p["datetime"]: 14 for p in self.points[:30]}
        self.state.save_backfill_attempts(attempts)
        self._run()
        self.state.save_backfill_attempts({p["datetime"]: 15 for p in self.points[:30]})
        self._run()
        self.assertEqual([off for _, off in self.uploaded], [15, 1])

    def test_short_holes_are_not_backfilled(self):
        self.health.get_combined_intraday.return_value = (self.points[:5] + self.points[30:], 3600)
        self._run()
        self.assertEqual(len(self.uploaded), 1)  # only the new segment


class TestUploadDuplicate(unittest.TestCase):
    """A 409 'Duplicate Wellness File' means Garmin already has the data."""

    def setUp(self):
        from garmin_client import GarminClient
        self.client = GarminClient.__new__(GarminClient)
        self.client._client = MagicMock()

    def test_409_is_not_retried_and_reports_duplicate(self):
        post = self.client._client.client.post
        post.side_effect = Exception("API Error 409 - {'code': 409, 'content': 'Duplicate Wellness File.'}")
        with patch("time.sleep") as sleep:
            result = self.client.upload_fit(b"fit")
        self.assertEqual(result, {"status": "duplicate"})
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_unrelated_error_containing_409_is_not_a_duplicate(self):
        post = self.client._client.client.post
        post.side_effect = Exception("API Error 502 - upstream request id 4095 failed")
        with patch("time.sleep"):
            with self.assertRaises(Exception):
                self.client.upload_fit(b"fit")
        self.assertEqual(post.call_count, 3)

    def test_other_errors_still_retry_then_raise(self):
        post = self.client._client.client.post
        post.side_effect = Exception("API Error 500")
        with patch("time.sleep"):
            with self.assertRaises(Exception):
                self.client.upload_fit(b"fit")
        self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
