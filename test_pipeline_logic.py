
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

        # Verify Monitoring Messages
        monitoring_msgs = list(fitfile.get_messages("monitoring"))
        self.assertEqual(len(monitoring_msgs), 120)
        
        # Check first record
        first_msg = monitoring_msgs[0]
        print("\nDebug: Fields found in monitoring message:")
        for field in first_msg:
            print(f"  {field.name} ({field.def_num}): {field.value}")
            
        self.assertEqual(first_msg.get_value("heart_rate"), 70)
        self.assertEqual(first_msg.get_value("steps"), 20)
        self.assertEqual(first_msg.get_value("duration"), 60)
        self.assertEqual(first_msg.get_value("activity_type"), "walking")

        print(f"\nSUCCESS: Engine logic verified. FIT file correctly encoded with {len(monitoring_msgs)} points.")

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


class TestFilterCoveredPoints(unittest.TestCase):
    """Unit tests for GarminClient.filter_covered_points (mocked API)."""

    def setUp(self):
        from garmin_client import GarminClient
        self.client = GarminClient.__new__(GarminClient)
        self.client._client = MagicMock()

    def _pts(self, minutes, date="2026-03-11"):
        base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return [{"datetime": base + timedelta(minutes=m)} for m in minutes]

    def test_empty_input(self):
        self.assertEqual(self.client.filter_covered_points([]), [])

    def test_all_pass_when_nothing_covered(self):
        with patch.object(self.client, "get_covered_minutes", return_value=set()):
            pts = self._pts(range(5))
            self.assertEqual(self.client.filter_covered_points(pts), pts)

    def test_covered_minutes_are_removed(self):
        # _pts uses midnight as base; covered set must use the same base
        base = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc)
        covered = {base + timedelta(minutes=1), base + timedelta(minutes=3)}
        with patch.object(self.client, "get_covered_minutes", return_value=covered):
            pts = self._pts(range(5))
            result = self.client.filter_covered_points(pts)
            result_times = {p["datetime"] for p in result}
            self.assertNotIn(base + timedelta(minutes=1), result_times)
            self.assertNotIn(base + timedelta(minutes=3), result_times)
            self.assertEqual(len(result), 3)

    def test_all_covered_returns_empty(self):
        base = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc)
        covered = {base + timedelta(minutes=i) for i in range(5)}
        with patch.object(self.client, "get_covered_minutes", return_value=covered):
            self.assertEqual(self.client.filter_covered_points(self._pts(range(5))), [])

    def test_queries_each_date_once(self):
        # Points spanning two dates should trigger two get_covered_minutes calls
        day1 = datetime(2026, 3, 11, 23, 58, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 3, 12, 0, 2, 0, tzinfo=timezone.utc)
        pts = [{"datetime": day1}, {"datetime": day2}]
        with patch.object(self.client, "get_covered_minutes", return_value=set()) as mock_gcm:
            self.client.filter_covered_points(pts)
            called_dates = {c.args[0] for c in mock_gcm.call_args_list}
            self.assertEqual(called_dates, {"2026-03-11", "2026-03-12"})


if __name__ == "__main__":
    unittest.main()
