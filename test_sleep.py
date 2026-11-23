import io
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

os.environ.setdefault("STATE_DIRECTORY", tempfile.mkdtemp())

import fitparse

import sleep_score
from fit_engine import build_sleep_fit, to_garmin_ts, _sleep_level_records
from google_health_client import GoogleHealthClient
from sleep_sync import SleepStore, run_sleep_sync, wake_date

T = datetime(2026, 11, 21, 21, 0, tzinfo=timezone.utc)


def stage(a, b, kind, t=T):
    return {"start": t + timedelta(minutes=a), "end": t + timedelta(minutes=b), "type": kind}


def session(t=T, offset=3600, hours=8):
    n = hours * 60
    stages = [stage(0, n // 2, "LIGHT", t), stage(n // 2, n // 2 + 60, "DEEP", t), stage(n // 2 + 60, n - 10, "REM", t),
              stage(n - 10, n, "AWAKE", t)]
    return {"id": f"s{t:%m%d}", "start": t, "end": t + timedelta(minutes=n), "utc_offset_seconds": offset, "stages": stages}


class TestSleepFit(unittest.TestCase):
    def test_layout_and_stage_end_semantics(self):
        stages = [stage(0, 60, "LIGHT"), stage(60, 90, "DEEP"), stage(90, 95, "WEIRD"), stage(95, 100, "REM")]
        raw = build_sleep_fit(stages, 1, 3993, 3445298263, assessment={"overall_sleep_score": 71},
                              utc_offset_seconds=3600)
        msgs = list(fitparse.FitFile(io.BytesIO(raw)).get_messages())
        self.assertEqual([m.mesg_num for m in msgs[:7]], [0, 49, 23, 273, 21, 21, 346])
        fields = lambda m: {d.def_num: d.raw_value for d in m}
        file_id = fields(msgs[0])
        end = to_garmin_ts(T + timedelta(minutes=100))
        self.assertEqual((file_id[0], file_id[4]), (49, end + 60))
        events = [fields(m) for m in msgs if m.mesg_num == 21]
        self.assertEqual([(e[253], e[3], e[0], e[1]) for e in events],
                         [(to_garmin_ts(T), to_garmin_ts(T) + 3600, 74, 0), (end, end + 3600, 74, 1)])
        self.assertEqual(fields(msgs[6])[6], 71)  # overall_sleep_score
        self.assertIsNone(fields(msgs[6])[15])  # stress not given -> invalid
        ts = lambda mins: to_garmin_ts(T + timedelta(minutes=mins))
        levels = [fields(m) for m in msgs if m.mesg_num == 275]
        # each record is the stage that ENDS at its timestamp; the unknown stage becomes awake
        self.assertEqual([(l[253], l[0]) for l in levels], [(ts(60), 2), (ts(90), 3), (ts(95), 1), (ts(100), 4)])

    def test_overlaps_clipped_and_equal_levels_merged(self):
        start, end, levels = _sleep_level_records([stage(0, 30, "LIGHT"), stage(20, 40, "LIGHT"), stage(35, 50, "DEEP")])
        ts = lambda mins: to_garmin_ts(T + timedelta(minutes=mins))
        self.assertEqual((start, end, levels), (ts(0), ts(50), [(ts(40), 2), (ts(50), 3)]))

    def test_no_usable_stages_raises(self):
        with self.assertRaises(ValueError):
            build_sleep_fit([stage(0, 5, "WEIRD")], 1, 3993, 1, assessment={})


class TestSleepScore(unittest.TestCase):
    def test_fit_recovers_weights_from_enough_nights(self):
        true = [80.0, -6.0, -3.0, -1.0]
        pairs = []
        for i in range(120):
            asleep, hr = 300 + (i * 7) % 260, 50 + (i * 3) % 20
            score = sum(w * x for w, x in zip(true, sleep_score.features(asleep, 470, hr)))
            pairs.append({"date": "2026-10-01", "asleep": asleep, "need": 470, "hr": hr, "score": score})
        w = sleep_score.fit(pairs, prior=[0, 0, 0, 0], lam=0.001, today=date(2026, 10, 1))
        for a, b in zip(w, true):
            self.assertAlmostEqual(a, b, places=2)

    def test_few_nights_stay_near_the_prior(self):
        pairs = [{"date": "2026-11-01", "asleep": 470, "need": 470, "hr": 60, "score": 20}]
        w = sleep_score.fit(pairs, today=date(2026, 11, 2))
        self.assertGreater(sleep_score.predict(w, 470, 470, 60), 60)

    def test_predict_clamped(self):
        self.assertEqual(sleep_score.predict([500, 0, 0, 0], 1, 1, 60), 100)
        self.assertEqual(sleep_score.predict([-5, 0, 0, 0], 1, 1, 60), 0)

    def test_stress_fit_recovers_weights(self):
        true = [20.0, 0.6, -0.5]
        pairs = [{"date": "2026-10-01", "hr": 50 + i % 20, "hrv": 15 + (i * 7) % 30,
                  "stress": sum(w * x for w, x in zip(true, sleep_score.stress_features(50 + i % 20, 15 + (i * 7) % 30)))}
                 for i in range(100)]
        pairs.append({"date": "2026-10-01", "hr": 60, "hrv": None, "stress": 99})  # incomplete: ignored
        w = sleep_score.fit_stress(pairs, prior=[0, 0, 0], lam=0.001, today=date(2026, 10, 1))
        for a, b in zip(w, true):
            self.assertAlmostEqual(a, b, places=2)

    def test_predict_stress(self):
        self.assertIsNone(sleep_score.predict_stress(sleep_score.DEFAULT_STRESS_WEIGHTS, None, 30))
        low = sleep_score.predict_stress(sleep_score.DEFAULT_STRESS_WEIGHTS, 52, 40)
        high = sleep_score.predict_stress(sleep_score.DEFAULT_STRESS_WEIGHTS, 66, 15)
        self.assertLess(low, high)
        self.assertEqual(sleep_score.predict_stress([500, 0, 0], 60, 25), 100)

    def test_assessment_fields(self):
        a = sleep_score.assessment({"DEEP": 90, "LIGHT": 250, "REM": 100, "AWAKE": 12, "AWAKE_COUNT": 2}, 470, 77)
        self.assertEqual((a["overall_sleep_score"], a["awakenings_count"], a["deep_sleep_score"]), (77, 2, 100))
        self.assertLess(a["sleep_duration_score"], 100)
        self.assertTrue(all(0 <= v <= 255 for v in a.values()))
        self.assertNotIn("average_stress_during_sleep", a)
        self.assertEqual(sleep_score.assessment({"LIGHT": 400}, 470, 70, 18.37)["average_stress_during_sleep"], 1837)


class TestSleepSource(unittest.TestCase):
    def test_only_fitbit_platform_kept(self):
        def dp(platform, start):
            return {"name": platform, "dataSource": {"platform": platform}, "sleep": {
                "interval": {"startTime": start, "endTime": "2026-11-22T06:00:00Z", "endUtcOffset": "3600s"},
                "stages": [{"startTime": start, "endTime": "2026-11-22T06:00:00Z", "type": "LIGHT"}]}}
        pts = [dp("FITBIT", "2026-11-21T22:00:00Z"), dp("HEALTH_KIT", "2026-11-21T22:00:00Z")]
        self.assertEqual([s["id"] for s in GoogleHealthClient.parse_sleep_sessions(pts)], ["FITBIT"])
        self.assertEqual(len(GoogleHealthClient.parse_sleep_sessions(pts, platform=None)), 2)


class FakeHealth:
    def __init__(self, sessions, hr=58.0, hrv=None, hrv_fails=False):
        self.sessions, self.hr, self.since = sessions, hr, []
        self.hrv, self.hrv_fails = hrv or {}, hrv_fails

    def get_daily_hrv(self, since):
        if self.hrv_fails:
            raise RuntimeError("no hrv")
        return {d: v for d, v in self.hrv.items() if d >= since.isoformat()}

    def get_sleep_sessions(self, since, min_minutes=180):
        self.since.append(since)
        return [s for s in self.sessions if s["end"] >= since]

    def get_sleeping_heart_rate(self, start, end):
        return self.hr


class FakeGarmin:
    def __init__(self, nights=None, fail=False):
        self.nights, self.fail, self.uploads, self.lookups = nights or {}, fail, [], []

    def get_sleep_summary(self, d):
        self.lookups.append(d)
        if self.fail:
            raise RuntimeError("boom")
        return self.nights.get(d)

    def upload_fit_for_window(self, fit_bytes, start):
        self.uploads.append((start, fit_bytes))
        return {"ok": True}


CFG = {"sleep": {"enabled": True, "holdback_hours": 3, "upload_days": 2},
       "device": {"manufacturer": 1, "product_id": 3993, "serial_number": 3445298263}}


class TestRunSleepSync(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "sleep.json")
        self.store = SleepStore(self.path)
        self.now = T + timedelta(hours=14)  # 6h after the 8h night ends

    def test_disabled_does_nothing(self):
        g = FakeGarmin()
        run_sleep_sync({"sleep": {}}, FakeHealth([session()]), g, self.store, now=self.now)
        self.assertEqual(g.lookups, [])

    def test_uploads_night_garmin_lacks_once(self):
        s, g = session(), FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([s], hr=61, hrv={wake_date(s): 21.5}), g, self.store, now=self.now)
        self.assertEqual(len(g.uploads), 1)
        a = next(m for m in fitparse.FitFile(io.BytesIO(g.uploads[0][1])).get_messages() if m.mesg_num == 346)
        expected = sleep_score.predict_stress(sleep_score.DEFAULT_STRESS_WEIGHTS, 61, 21.5)
        self.assertEqual({d.def_num: d.raw_value for d in a}[15], round(expected * 100))
        self.assertEqual(SleepStore(self.path).data["uploaded"], {wake_date(s): s["start"].isoformat()})
        run_sleep_sync(CFG, FakeHealth([s]), g, SleepStore(self.path), now=self.now + timedelta(hours=1))
        self.assertEqual(len(g.uploads), 1)

    def test_hrv_failure_still_uploads(self):
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([session()], hrv_fails=True), g, self.store, now=self.now)
        self.assertEqual(len(g.uploads), 1)

    def test_holdback(self):
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([session()]), g, self.store, now=T + timedelta(hours=9))
        self.assertEqual((g.lookups, g.uploads), ([], []))

    def test_watch_night_becomes_calibration_pair(self):
        s = session()
        d = wake_date(s)
        g = FakeGarmin({d: {"start": s["start"] + timedelta(minutes=7, seconds=13), "end": s["end"] - timedelta(seconds=41),
                            "score": 81, "need": 450, "stress": 17.0}})
        run_sleep_sync(CFG, FakeHealth([s], hr=55.5, hrv={d: 31.2}), g, self.store, now=self.now)
        data = SleepStore(self.path).data
        self.assertEqual(g.uploads, [])
        self.assertEqual(data["pairs"][d], {"asleep": 470.0, "need": 450, "hr": 55.5, "hrv": 31.2, "score": 81,
                                            "stress": 17.0})
        self.assertNotEqual(data["stress_weights"], sleep_score.DEFAULT_STRESS_WEIGHTS)
        self.assertEqual(data["need"], 450)
        self.assertNotEqual(data["weights"], sleep_score.DEFAULT_WEIGHTS)

    def test_need_comes_from_the_latest_night(self):
        old, new = session(T - timedelta(days=3)), session()
        night = lambda s, need: {"start": s["start"] + timedelta(seconds=7), "end": s["end"], "score": 70, "need": need}
        g = FakeGarmin({wake_date(old): night(old, 540), wake_date(new): night(new, 470)})
        run_sleep_sync({**CFG, "sleep": {**CFG["sleep"], "calibration_days": 10}}, FakeHealth([old, new]), g,
                       self.store, now=self.now)
        self.assertEqual(self.store.data["need"], 470)

    def test_own_upload_never_used_for_calibration(self):
        s = session()
        d = wake_date(s)
        # Same window to the second -> one of ours (e.g. uploaded before the store existed)
        g = FakeGarmin({d: {"start": s["start"], "end": s["end"], "score": 65, "need": 470}})
        run_sleep_sync(CFG, FakeHealth([s]), g, self.store, now=self.now)
        self.assertEqual((self.store.data["pairs"], self.store.data["seen"][d]), ({}, "ours"))

    def test_old_night_without_garmin_sleep_not_uploaded(self):
        old = session(T - timedelta(days=5))
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([old]), g, self.store, now=self.now)
        self.assertEqual(g.uploads, [])
        self.assertEqual(self.store.data["seen"][wake_date(old)], "none")

    def test_lookup_error_keeps_bootstrap_open(self):
        run_sleep_sync(CFG, FakeHealth([session()]), FakeGarmin(fail=True), self.store, now=self.now)
        self.assertFalse(self.store.data["bootstrapped"])
        self.assertEqual(self.store.data["seen"], {})

    def test_bootstrap_spreads_over_runs_then_scans_recent_only(self):
        cfg = {**CFG, "sleep": {**CFG["sleep"], "max_garmin_lookups": 2}}
        sessions = [session(T - timedelta(days=i)) for i in range(5)]
        h, g = FakeHealth(sessions), FakeGarmin()
        run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertFalse(self.store.data["bootstrapped"])
        for _ in range(3):
            run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertTrue(self.store.data["bootstrapped"])
        self.assertEqual(len(g.lookups), 5)
        run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertGreater(h.since[-1], self.now - timedelta(days=4))


if __name__ == "__main__":
    unittest.main()
