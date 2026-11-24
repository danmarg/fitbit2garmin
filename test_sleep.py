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
from sleep_sync import SleepStore, asleep_minutes, main_sessions, merge_split_nights, recorded_minutes, sleeping_heart_rate, run_sleep_sync, stage_minutes, wake_date

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
        true = [80.0, -4.0, -7.0, -3.0, -1.0]
        pairs = []
        for i in range(160):
            asleep, hr = 100 + (i * 37) % 460, 50 + (i * 3) % 20
            score = sum(w * x for w, x in zip(true, sleep_score.features(asleep, 470, hr)))
            pairs.append({"date": "2026-10-01", "asleep": asleep, "need": 470, "hr": hr, "score": score})
        w = sleep_score.fit(pairs, prior=[0] * 5, lam=0.001, today=date(2026, 10, 1))
        for a, b in zip(w, true):
            self.assertAlmostEqual(a, b, places=2)

    def test_few_nights_stay_near_the_prior(self):
        pairs = [{"date": "2026-11-01", "asleep": 470, "need": 470, "hr": 60, "score": 20}]
        w = sleep_score.fit(pairs, today=date(2026, 11, 2))
        self.assertGreater(sleep_score.predict(w, 470, 470, 60), 60)

    def test_predict_clamped(self):
        self.assertEqual(sleep_score.predict([500, 0, 0, 0, 0], 1, 1, 60), 100)
        self.assertEqual(sleep_score.predict([-5, 0, 0, 0, 0], 1, 1, 60), 0)

    def test_generic_duration_curve_is_sane(self):
        p = lambda mins: sleep_score.predict(sleep_score.GENERIC_WEIGHTS, mins, 470, 56)
        self.assertLess(p(120), 45)                    # Garmin scored a 2 h night 29
        self.assertLess(p(120), p(240) - 15)           # steeper once far short
        self.assertGreater(p(600), p(470) - 10)        # oversleep penalty is capped
        self.assertEqual(p(600), p(900))

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
        self.assertIsNone(sleep_score.predict_stress(sleep_score.GENERIC_STRESS_WEIGHTS, None, 30))
        low = sleep_score.predict_stress(sleep_score.GENERIC_STRESS_WEIGHTS, 52, 40)
        high = sleep_score.predict_stress(sleep_score.GENERIC_STRESS_WEIGHTS, 66, 15)
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
    def __init__(self, sessions, hr=58.0, hrv=None, hrv_fails=False, hr_fails=False, sleep_error=None):
        self.sessions, self.hr, self.since = sessions, hr, []
        self.hrv, self.hrv_fails, self.hr_fails, self.sleep_error = hrv or {}, hrv_fails, hr_fails, sleep_error
        self.authorized = 0

    def ensure_authorized(self):
        self.authorized += 1

    def get_daily_hrv(self, since):
        if self.hrv_fails:
            raise RuntimeError("no hrv")
        return {d: v for d, v in self.hrv.items() if d >= since.isoformat()}

    def get_sleep_sessions(self, since, min_minutes=180):
        if self.sleep_error:
            raise self.sleep_error
        self.since.append(since)
        return [s for s in self.sessions if s["end"] >= since]

    def get_sleeping_heart_rate(self, start, end):
        if self.hr_fails:
            raise RuntimeError("rollup 500")
        return self.hr


class FakeGarmin:
    def __init__(self, nights=None, fail=False, fail_dates=(), upload_fails=False):
        self.nights, self.fail, self.uploads, self.lookups = nights or {}, fail, [], []
        self.fail_dates, self.upload_fails = set(fail_dates), upload_fails

    def get_sleep_summary(self, d):
        self.lookups.append(d)
        if self.fail or d in self.fail_dates:
            raise RuntimeError("boom")
        return self.nights.get(d)

    def upload_fit_for_window(self, fit_bytes, start):
        if self.upload_fails:
            raise RuntimeError("502")
        self.uploads.append((start, fit_bytes))
        return {"ok": True}


CFG = {"sleep": {"enabled": True, "holdback_hours": 3, "upload_days": 2, "calibration_days": 1},
       "device": {"manufacturer": 1, "product_id": 3993, "serial_number": 3445298263}}


class TestRunSleepSync(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "sleep.json")
        self.store = SleepStore(self.path)
        self.now = T + timedelta(hours=14)  # 6h after the 8h night ends

    def test_failed_upload_is_retried_then_given_up(self):
        s = session()
        h, d = FakeHealth([s], hrv={wake_date(session()): 30}), wake_date(s)
        g = FakeGarmin(upload_fails=True)
        run_sleep_sync(CFG, h, g, self.store, now=self.now)
        self.assertEqual(self.store.data["uploaded"], {})
        self.assertNotIn(d, self.store.data["seen"])
        g.upload_fails = False
        run_sleep_sync(CFG, h, g, self.store, now=self.now)
        self.assertEqual(len(g.uploads), 1)
        self.assertEqual(self.store.data["seen"][d], "uploaded")
        self.assertNotIn(d, self.store.data["failures"])
        # A night that keeps failing is given up on after MAX_FAILURES.
        other = SleepStore(os.path.join(tempfile.mkdtemp(), "sleep.json"))
        g2 = FakeGarmin(upload_fails=True)
        for _ in range(4):
            run_sleep_sync(CFG, h, g2, other, now=self.now)
        self.assertEqual(other.data["seen"][d], "failed")

    def test_one_failing_date_does_not_block_the_others(self):
        sessions = [session(T - timedelta(days=i)) for i in range(3)]
        bad = wake_date(sessions[0])
        g = FakeGarmin(fail_dates={bad})
        cfg = {**CFG, "sleep": {**CFG["sleep"], "calibration_days": 4, "upload_days": 0}}
        h = FakeHealth(sessions)
        for _ in range(3):
            run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertEqual(self.store.data["seen"][bad], "failed")
        self.assertTrue(all(self.store.data["seen"].get(wake_date(x)) == "none" for x in sessions[1:]))
        run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertTrue(self.store.data["bootstrapped"])

    def test_heart_rate_error_keeps_the_rest_of_the_run(self):
        s = session()
        d = wake_date(s)
        g = FakeGarmin({d: {"start": s["start"] + timedelta(seconds=30), "end": s["end"], "score": 70, "need": 470,
                            "asleep": 460.0, "hr": 55}})
        run_sleep_sync(CFG, FakeHealth([s], hr_fails=True), g, self.store, now=self.now)
        saved = SleepStore(self.path).data
        self.assertIn(d, saved["garmin_nights"])  # the run's work was saved
        self.assertNotIn(d, saved["pairs"])
        self.assertNotIn(d, saved["seen"])  # retried next run

    def test_split_night_is_uploaded_whole(self):
        # Fitbit split the night: 21:00-00:30, awake 2h, 02:30-06:00.
        first = session(hours=3.5)
        second = session(T + timedelta(hours=5.5), hours=3.5)
        nap = session(T + timedelta(hours=16), hours=1)
        merged = merge_split_nights([second, first, nap])
        self.assertEqual(len(merged), 2)
        night = merged[0]
        self.assertEqual((night["start"], night["end"]), (first["start"], second["end"]))
        self.assertEqual(stage_minutes(night)["AWAKE"], 10 + 120 + 10)
        self.assertEqual(asleep_minutes(night), asleep_minutes(first) + asleep_minutes(second))
        g = FakeGarmin()
        now = second["end"] + timedelta(hours=4)
        run_sleep_sync(CFG, FakeHealth([first, second, nap], hrv={wake_date(second): 30}), g, self.store, now=now)
        self.assertEqual([start for start, _ in g.uploads], [first["start"]])

    def test_naps_near_the_night_stay_separate(self):
        night = session(hours=8)                                  # 21:00-05:00
        nap = session(T + timedelta(hours=9), hours=0.75)         # 06:00-06:45, 1h later
        self.assertEqual(len(merge_split_nights([night, nap])), 2)
        # Two short naps 2h apart don't add up to a night either.
        a, b = session(T, hours=0.75), session(T + timedelta(hours=2.75), hours=0.75)
        self.assertEqual(len(merge_split_nights([a, b])), 2)

    def test_overlapping_sessions_are_not_counted_twice(self):
        whole = session(hours=8)
        inner = session(T + timedelta(hours=2), hours=4)
        merged = merge_split_nights([whole, inner])
        self.assertEqual(len(merged), 1)
        self.assertEqual(asleep_minutes(merged[0]), asleep_minutes(whole))
        self.assertEqual(recorded_minutes(merged[0]), 480)

    def test_heart_rate_and_length_leave_out_the_gap(self):
        first, second = session(hours=3.5), session(T + timedelta(hours=5.5), hours=3.5)
        night = merge_split_nights([first, second])[0]
        self.assertEqual(recorded_minutes(night), 420)

        class HR:
            calls = []
            def get_sleeping_heart_rate(self, a, b):
                self.calls.append((a, b))
                return 50.0 if a == first["start"] else 60.0
        h = HR()
        self.assertEqual(sleeping_heart_rate(h, night), 55.0)
        self.assertEqual(h.calls, [(first["start"], first["end"]), (second["start"], second["end"])])

    def test_short_fragment_between_halves_is_taken_in(self):
        a = session(hours=3)                                      # 21:00-00:00
        f0 = T + timedelta(hours=3.5)                             # 00:30-01:00
        frag = {**session(f0, hours=1), "end": f0 + timedelta(minutes=30), "stages": [stage(0, 30, "LIGHT", f0)]}
        c = session(T + timedelta(hours=4.5), hours=3)            # 01:30-04:30
        merged = merge_split_nights([a, frag, c])
        self.assertEqual(len(merged), 1)
        self.assertEqual(recorded_minutes(merged[0]), 180 + 30 + 180)
        stages = merged[0]["stages"]
        self.assertTrue(all(x["end"] <= y["start"] for x, y in zip(stages, stages[1:])))

    def test_main_session_is_chosen_by_recorded_time(self):
        split = merge_split_nights([session(hours=2), session(T + timedelta(hours=4.4), hours=2)])[0]
        whole = session(T + timedelta(hours=0.5), hours=6, offset=3600)
        self.assertEqual(main_sessions([split, whole])[wake_date(whole)]["start"], whole["start"])

    def test_night_waking_in_the_small_hours_waits_for_its_second_half(self):
        first, second = session(hours=3.5), session(T + timedelta(hours=5.5), hours=3.5)  # wake 01:30 local
        h, g = FakeHealth([first], hrv={wake_date(first): 30}), FakeGarmin()
        run_sleep_sync(CFG, h, g, self.store, now=first["end"] + timedelta(hours=3, minutes=10))
        self.assertEqual(g.uploads, [])
        h.sessions = [first, second]
        run_sleep_sync(CFG, h, g, self.store, now=second["end"] + timedelta(hours=3, minutes=10))
        self.assertEqual([start for start, _ in g.uploads], [first["start"]])
        # With no second part it still goes up once max_wait has passed.
        other = SleepStore(os.path.join(tempfile.mkdtemp(), "sleep.json"))
        g2 = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([first], hrv={wake_date(first): 30}), g2, other,
                       now=first["end"] + timedelta(hours=12, minutes=10))
        self.assertEqual(len(g2.uploads), 1)

    def test_naps_of_an_hour_or_more_stay_off_the_night(self):
        def nap(t, minutes):
            return {**session(t, hours=1), "end": t + timedelta(minutes=minutes), "stages": [stage(0, minutes, "LIGHT", t)]}
        # 17:00-18:30 local nap, bed 20:30-05:30 local (UTC+1): gap centred at 19:30 local.
        evening = nap(T - timedelta(hours=4), 90)
        night = session(T - timedelta(hours=0.5), hours=9)
        morning = nap(night["end"] + timedelta(hours=1.5), 90)     # 07:00 local
        self.assertEqual(len(merge_split_nights([evening, night, morning])), 3)

    def test_fragments_are_not_matched_against_merged_nights(self):
        # Two separate split nights a day apart, each with a fragment in its gap.
        n1 = [session(hours=3), session(T + timedelta(hours=5), hours=3)]
        t2 = T + timedelta(days=1)
        n2 = [session(t2, hours=3), session(t2 + timedelta(hours=5), hours=3)]
        frag = {**session(T + timedelta(hours=3.5), hours=1), "end": T + timedelta(hours=4),
                "stages": [stage(0, 30, "LIGHT", T + timedelta(hours=3.5))]}
        merged = merge_split_nights(n1 + n2 + [frag])
        self.assertEqual(len(merged), 2)
        self.assertEqual([len(m["parts"]) for m in merged], [3, 2])

    def test_small_hours_wait_ends_at_ten(self):
        first = session(hours=3.5)                                  # wakes 01:30 local
        h = FakeHealth([first], hrv={wake_date(first): 30})
        g = FakeGarmin()
        nine = datetime(2026, 11, 22, 8, 0, tzinfo=timezone.utc)    # 09:00 local
        run_sleep_sync(CFG, h, g, self.store, now=nine)
        self.assertEqual(g.uploads, [])
        run_sleep_sync(CFG, h, g, self.store, now=nine + timedelta(hours=1, minutes=5))
        self.assertEqual(len(g.uploads), 1)

    def test_refreshes_google_token(self):
        h = FakeHealth([])
        run_sleep_sync(CFG, h, FakeGarmin(), self.store, now=self.now)
        self.assertEqual(h.authorized, 1)

    def test_missing_sleep_scope_skips_sleep_without_failing(self):
        import requests
        resp = requests.Response()
        resp.status_code = 403
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([], sleep_error=requests.HTTPError(response=resp)), g, self.store, now=self.now)
        self.assertEqual(g.lookups, [])

    def test_each_night_uses_its_own_need(self):
        older, newer = session(T - timedelta(days=1)), session()
        self.store.data["needs"] = {wake_date(older): 520, wake_date(newer): 440}
        self.store.data["need"], self.store.data["need_date"] = 440, wake_date(newer)
        self.assertEqual(self.store.need_for(wake_date(older)), 520)
        self.assertEqual(self.store.need_for("2000-01-01"), 440)  # nothing earlier: latest
        later = (date.fromisoformat(wake_date(newer)) + timedelta(days=3)).isoformat()
        self.assertEqual(self.store.need_for(later), 440)

    def test_recent_empty_dates_are_provisional(self):
        # No Fitbit night and nothing on Garmin yet: yesterday/today are asked about again.
        g = FakeGarmin()
        cfg = {**CFG, "sleep": {**CFG["sleep"], "calibration_days": 5}}
        run_sleep_sync(cfg, FakeHealth([]), g, self.store, now=self.now)
        today = self.now.date()
        self.assertNotIn(today.isoformat(), self.store.data["seen"])
        self.assertNotIn((today - timedelta(days=1)).isoformat(), self.store.data["seen"])
        self.assertEqual(self.store.data["seen"][(today - timedelta(days=2)).isoformat()], "none")

    def test_own_upload_recognised_by_uploaded_stage_window(self):
        s = session()
        s["stages"] = [stage(1, 400, "LIGHT"), stage(400, 470, "DEEP"), stage(470, 480, "WEIRD")]
        d = wake_date(s)
        g = FakeGarmin({d: {"start": T + timedelta(minutes=1), "end": T + timedelta(minutes=470), "score": 65,
                            "need": 470, "asleep": 469.0, "hr": 58}})
        run_sleep_sync(CFG, FakeHealth([s]), g, self.store, now=self.now)
        self.assertEqual(self.store.data["seen"][d], "ours")
        self.assertEqual(self.store.data["garmin_nights"], {})

    def test_disabled_does_nothing(self):
        g = FakeGarmin()
        run_sleep_sync({"sleep": {}}, FakeHealth([session()]), g, self.store, now=self.now)
        self.assertEqual(g.lookups, [])

    def test_uploads_night_garmin_lacks_once(self):
        s, g = session(), FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([s], hr=61, hrv={wake_date(s): 21.5}), g, self.store, now=self.now)
        prev = (date.fromisoformat(wake_date(s)) - timedelta(days=1)).isoformat()
        self.assertEqual(g.lookups[-1], prev)  # the previous night, for Garmin's sleep need
        self.assertEqual(len(g.uploads), 1)
        a = next(m for m in fitparse.FitFile(io.BytesIO(g.uploads[0][1])).get_messages() if m.mesg_num == 346)
        expected = sleep_score.predict_stress(sleep_score.GENERIC_STRESS_WEIGHTS, 61, 21.5)
        self.assertEqual({d.def_num: d.raw_value for d in a}[15], round(expected * 100))
        self.assertEqual(SleepStore(self.path).data["uploaded"], {wake_date(s): s["start"].isoformat()})
        run_sleep_sync(CFG, FakeHealth([s]), g, SleepStore(self.path), now=self.now + timedelta(hours=1))
        self.assertEqual(len(g.uploads), 1)

    def test_waits_for_hrv_then_uploads_after_max_wait(self):
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([session()], hrv_fails=True), g, self.store, now=self.now)
        self.assertEqual(g.uploads, [])
        self.assertNotIn(wake_date(session()), self.store.data["seen"])  # still pending
        run_sleep_sync(CFG, FakeHealth([session()], hrv_fails=True), g, self.store, now=T + timedelta(hours=21))
        self.assertEqual(len(g.uploads), 1)

    def test_waits_until_fitbit_has_processed_the_night(self):
        s = {**session(), "processed": False}
        g = FakeGarmin()
        run_sleep_sync(CFG, FakeHealth([s], hrv={wake_date(s): 30}), g, self.store, now=self.now)
        self.assertEqual(g.uploads, [])
        s["processed"] = True
        run_sleep_sync(CFG, FakeHealth([s], hrv={wake_date(s): 30}), g, self.store, now=self.now)
        self.assertEqual(len(g.uploads), 1)

    def test_uses_garmins_need_for_the_night_from_the_previous_night(self):
        s = session()
        d = wake_date(s)
        prev = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        prev_night = {"start": T - timedelta(days=1), "end": T - timedelta(hours=16), "score": 70,
                      "need": 470, "next_need": 520}
        g = FakeGarmin({prev: prev_night})
        run_sleep_sync(CFG, FakeHealth([s], hr=58, hrv={d: 30}), g, self.store, now=self.now)
        self.assertEqual((self.store.data["need"], self.store.data["need_date"]), (520, d))
        a = next(m for m in fitparse.FitFile(io.BytesIO(g.uploads[0][1])).get_messages() if m.mesg_num == 346)
        expected = sleep_score.predict(sleep_score.GENERIC_WEIGHTS, 470, 520, 58)
        self.assertEqual({x.def_num: x.raw_value for x in a}[6], expected)

    def test_live_error_recorded_after_bootstrap(self):
        s = session()
        d = wake_date(s)
        self.store.data["bootstrapped"] = True
        g = FakeGarmin({d: {"start": s["start"] + timedelta(seconds=30), "end": s["end"], "score": 60, "need": 470,
                            "stress": 30.0}})
        run_sleep_sync(CFG, FakeHealth([s], hr=58, hrv={d: 25.0}), g, self.store, now=self.now)
        pair = self.store.data["pairs"][d]
        self.assertEqual(pair["pred_score"], sleep_score.predict(sleep_score.GENERIC_WEIGHTS, 470, 470, 58))
        self.assertIsNotNone(pair["pred_stress"])
        self.assertIn("live error over 1", self.store.live_error())

    def test_holdback(self):
        g = FakeGarmin()
        s = session()
        run_sleep_sync(CFG, FakeHealth([s]), g, self.store, now=T + timedelta(hours=9))
        self.assertEqual(g.uploads, [])
        self.assertNotIn(wake_date(s), self.store.data["seen"])  # decided after the holdback

    def test_watch_night_becomes_calibration_pair(self):
        s = session()
        d = wake_date(s)
        g = FakeGarmin({d: {"start": s["start"] + timedelta(minutes=7, seconds=13), "end": s["end"] - timedelta(seconds=41),
                            "score": 81, "need": 450, "stress": 17.0, "asleep": 455.0, "hr": 54}})
        run_sleep_sync(CFG, FakeHealth([s], hr=55.5, hrv={d: 31.2}), g, self.store, now=self.now)
        data = SleepStore(self.path).data
        self.assertEqual(g.uploads, [])
        self.assertEqual(data["pairs"][d], {"asleep": 470.0, "need": 450, "hr": 55.5, "hrv": 31.2, "score": 81,
                                            "stress": 17.0})
        self.assertNotEqual(data["stress_weights"], sleep_score.GENERIC_STRESS_WEIGHTS)
        self.assertEqual(data["garmin_nights"][d], {"asleep": 455.0, "need": 450, "hr": 54, "score": 81})
        self.assertEqual(data["need"], 450)
        self.assertNotEqual(data["weights"], sleep_score.GENERIC_WEIGHTS)

    def test_watch_night_without_fitbit_still_teaches_the_baseline(self):
        d = (self.now - timedelta(days=1)).date().isoformat()
        night = {"start": T - timedelta(days=1), "end": T - timedelta(hours=16), "score": 40, "need": 470,
                 "stress": 25.0, "asleep": 150.0, "hr": 60}
        g = FakeGarmin({d: night})
        run_sleep_sync(CFG, FakeHealth([]), g, self.store, now=self.now)
        self.assertIn(d, self.store.data["garmin_nights"])
        self.assertEqual(self.store.data["pairs"], {})
        self.assertNotEqual(self.store.weights, sleep_score.GENERIC_WEIGHTS)
        self.assertEqual(self.store.stress_weights, sleep_score.GENERIC_STRESS_WEIGHTS)

    def test_need_comes_from_the_latest_night(self):
        old, new = session(T - timedelta(days=3)), session()
        night = lambda s, need: {"start": s["start"] + timedelta(seconds=7), "end": s["end"], "score": 70, "need": need}
        g = FakeGarmin({wake_date(old): night(old, 540), wake_date(new): night(new, 470)})
        run_sleep_sync({**CFG, "sleep": {**CFG["sleep"], "calibration_days": 4}}, FakeHealth([old, new]), g,
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
        run_sleep_sync({**CFG, "sleep": {**CFG["sleep"], "calibration_days": 10}}, FakeHealth([old]), g, self.store,
                       now=self.now)
        self.assertEqual(g.uploads, [])
        self.assertEqual(self.store.data["seen"][wake_date(old)], "none")

    def test_lookup_error_keeps_bootstrap_open(self):
        run_sleep_sync(CFG, FakeHealth([session()]), FakeGarmin(fail=True), self.store, now=self.now)
        self.assertFalse(self.store.data["bootstrapped"])
        self.assertEqual(self.store.data["seen"], {})

    def test_bootstrap_spreads_over_runs_then_scans_recent_only(self):
        cfg = {**CFG, "sleep": {**CFG["sleep"], "max_garmin_lookups": 2, "calibration_days": 5}}
        sessions = [session(T - timedelta(days=i)) for i in range(5)]
        h, g = FakeHealth(sessions, hrv={wake_date(x): 30 for x in sessions}), FakeGarmin()
        run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertFalse(self.store.data["bootstrapped"])
        for _ in range(3):
            run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertTrue(self.store.data["bootstrapped"])
        self.assertLessEqual({wake_date(x) for x in sessions}, set(self.store.data["seen"]))
        run_sleep_sync(cfg, h, g, self.store, now=self.now)
        self.assertGreater(h.since[-1], self.now - timedelta(days=4))


if __name__ == "__main__":
    unittest.main()


class TestGarminSleepSummary(unittest.TestCase):
    def _client(self, dto):
        from unittest.mock import MagicMock
        from garmin_client import GarminClient
        c = GarminClient.__new__(GarminClient)
        c._client = MagicMock()
        c._client.get_sleep_data.return_value = {"dailySleepDTO": dto}
        c._ensure_connected = lambda: None
        return c

    def test_incomplete_record_is_treated_as_no_sleep(self):
        dto = {"sleepTimeSeconds": 25000, "sleepStartTimestampGMT": 1795294980000, "sleepEndTimestampGMT": None}
        self.assertIsNone(self._client(dto).get_sleep_summary("2026-11-22"))

    def test_summary_fields(self):
        dto = {"sleepTimeSeconds": 28200, "sleepStartTimestampGMT": 1795294980000, "sleepEndTimestampGMT": 1795328100000,
               "sleepScores": {"overall": {"value": 65}}, "sleepNeed": {"actual": 470}, "nextSleepNeed": {"actual": 480},
               "avgSleepStress": 16.0, "avgHeartRate": 59.0}
        g = self._client(dto).get_sleep_summary("2026-11-22")
        self.assertEqual((g["score"], g["need"], g["next_need"], g["stress"], g["asleep"], g["hr"]),
                         (65, 470, 480, 16.0, 470.0, 59.0))
