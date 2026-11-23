"""
sleep_sync.py
Uploads Fitbit sleep to Garmin Connect, with a sleep score and sleep stress learned
from the nights the user slept with their own Garmin watch.

Each run asks Garmin once about each local date (the night ending that morning):
  - no Garmin sleep, and a Fitbit night → upload it (if recent enough)
  - a night from the watch → calibration data:
      garmin_nights: Garmin's own minutes asleep / heart rate / need vs its score
                     (every watch night, Fitbit or not — the personal baseline)
      pairs:         Fitbit's minutes / sleeping HR / HRV vs Garmin's score and stress
                     (nights both devices recorded — the Fitbit-specific fit)
    then both models are refitted (see sleep_score.py)
  - a night we uploaded → ignored (never calibrate on our own output)

A night counts as ours if it is in the uploaded list, or if Garmin's sleep window equals
the Fitbit window to the second (our uploads copy it exactly; the watch's never do).

A Fitbit night is uploaded only once Fitbit has finished processing it and its HRV has
arrived, or after `max_wait_hours` past waking at the latest. Watch nights found after
the initial scan also record what the models predicted before refitting
(pred_score / pred_stress), and the running error is logged.

The store (sleep.json in the state directory) holds:
  seen:          {date: "uploaded" | "watch" | "ours" | "none"}
  uploaded:      {date: fitbit start ISO}
  garmin_nights: {date: {"asleep", "need", "hr", "score"}}
  pairs:         {date: {"asleep", "need", "hr", "hrv", "score", "stress", ["pred_score", "pred_stress"]}}
  weights / stress_weights: the learned models (None until nights are found)
  need / need_date: Garmin's sleep need (minutes) for the latest night it is known for
  bootstrapped:  whether the whole calibration window has been scanned once
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import sleep_score
from fit_engine import build_sleep_fit

log = logging.getLogger("fitbit2garmin.sleep")

SLEEP_FILE = os.path.join(os.environ.get("STATE_DIRECTORY", "data"), "sleep.json")


class SleepStore:
    def __init__(self, path: str = SLEEP_FILE):
        self.path = path
        self.data = {"seen": {}, "uploaded": {}, "garmin_nights": {}, "pairs": {}, "weights": None,
                     "stress_weights": None, "need": None, "bootstrapped": False}
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            log.warning("Could not read %s (%s) — starting a fresh sleep store.", path, e)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    @staticmethod
    def _valid(w, like):
        return w if isinstance(w, list) and len(w) == len(like) else like

    @property
    def weights(self) -> list[float]:
        return self._valid(self.data.get("weights"), sleep_score.GENERIC_WEIGHTS)

    @property
    def stress_weights(self) -> list[float]:
        return self._valid(self.data.get("stress_weights"), sleep_score.GENERIC_STRESS_WEIGHTS)

    def remember_need(self, night: str, minutes):
        """Keep Garmin's sleep need for the latest night it is known for."""
        if minutes and night >= (self.data.get("need_date") or ""):
            self.data["need"], self.data["need_date"] = minutes, night

    def live_error(self) -> str | None:
        rows = [p for p in self.data["pairs"].values() if p.get("pred_score") is not None]
        if not rows:
            return None
        score = sum(abs(p["pred_score"] - p["score"]) for p in rows) / len(rows)
        st = [p for p in rows if p.get("pred_stress") is not None and p.get("stress") is not None]
        stress = sum(abs(p["pred_stress"] - p["stress"]) for p in st) / len(st) if st else None
        return (f"live error over {len(rows)} watch night(s): score {score:.1f}"
                + ("" if stress is None else f", stress {stress:.1f}"))

    def refit(self, today):
        """Personal baseline from all watch nights, then the Fitbit fit shrunk toward it."""
        rows = lambda key: [{"date": d, **p} for d, p in self.data[key].items()]
        garmin_nights, pairs = rows("garmin_nights"), rows("pairs")
        baseline = sleep_score.fit(garmin_nights, prior=sleep_score.GENERIC_WEIGHTS, today=today)
        self.data["weights"] = sleep_score.fit(pairs, prior=baseline, today=today)
        self.data["stress_weights"] = sleep_score.fit_stress(pairs, prior=sleep_score.GENERIC_STRESS_WEIGHTS,
                                                             today=today)
        fmt = lambda names, ws: ", ".join(f"{n}={w:.2f}" for n, w in zip(names, ws))
        log.info("Sleep models refitted (%d watch night(s), %d also recorded by Fitbit): score %s; stress %s",
                 len(garmin_nights), len(pairs), fmt(sleep_score.FEATURE_NAMES, self.data["weights"]),
                 fmt(sleep_score.STRESS_FEATURE_NAMES, self.data["stress_weights"]))


def wake_date(session: dict) -> str:
    return (session["end"] + timedelta(seconds=session["utc_offset_seconds"])).date().isoformat()


def stage_minutes(session: dict) -> dict:
    out = {"AWAKE_COUNT": 0}
    for st in session["stages"]:
        out[st["type"]] = out.get(st["type"], 0) + (st["end"] - st["start"]).total_seconds() / 60
        if st["type"] == "AWAKE":
            out["AWAKE_COUNT"] += 1
    return out


def asleep_minutes(session: dict) -> float:
    m = stage_minutes(session)
    return sum(m.get(k, 0) for k in ("DEEP", "LIGHT", "REM"))


def main_sessions(sessions: list[dict]) -> dict[str, dict]:
    """The longest session per wake date (naps are already dropped by min_minutes)."""
    best: dict[str, dict] = {}
    for s in sessions:
        d = wake_date(s)
        if d not in best or (s["end"] - s["start"]) > (best[d]["end"] - best[d]["start"]):
            best[d] = s
    return best


def _next_day(d: str) -> str:
    return (datetime.fromisoformat(d) + timedelta(days=1)).date().isoformat()


def _prev_day(d: str) -> str:
    return (datetime.fromisoformat(d) - timedelta(days=1)).date().isoformat()


def _same_window(garmin_night: dict, session: dict) -> bool:
    return (abs((garmin_night["start"] - session["start"]).total_seconds()) <= 1
            and abs((garmin_night["end"] - session["end"]).total_seconds()) <= 1)


def run_sleep_sync(cfg: dict, health, garmin, store: SleepStore, now: datetime | None = None):
    scfg = cfg.get("sleep") or {}
    if not scfg.get("enabled", False):
        return
    now = now or datetime.now(timezone.utc)
    holdback = timedelta(hours=scfg.get("holdback_hours", 3))
    max_wait = timedelta(hours=scfg.get("max_wait_hours", 12))
    upload_days = scfg.get("upload_days", 2)
    calibration_days = scfg.get("calibration_days", 365)
    lookups_per_run = scfg.get("max_garmin_lookups", 60)
    device = cfg["device"]

    # Scan the whole calibration window until it has been covered once, then only recent nights.
    window_days = calibration_days if not store.data["bootstrapped"] else max(upload_days, 3)
    sessions = health.get_sleep_sessions(now - timedelta(days=window_days),
                                         min_minutes=scfg.get("min_minutes", 180))
    nights = main_sessions([s for s in sessions if s["end"] <= now - holdback])
    try:
        hrv_by_date = health.get_daily_hrv((now - timedelta(days=window_days + 1)).date())
    except Exception as e:
        log.warning("Fitbit HRV unavailable (%s) — stress uses the default HRV this run.", e)
        hrv_by_date = {}
    recent = now - timedelta(days=upload_days)
    seen = store.data["seen"]
    # Every date in the window (watch nights matter even without a Fitbit night), except
    # today unless Fitbit already has its night. "none" Fitbit nights become eligible
    # again if upload_days is raised to cover them.
    first = (now - timedelta(days=window_days)).date()
    dates = {(first + timedelta(days=i)).isoformat() for i in range((now.date() - first).days)} | set(nights)
    pending = sorted((d for d in dates if d not in seen
                      or (seen[d] == "none" and d in nights and nights[d]["end"] >= recent)), reverse=True)

    lookups = 0
    complete = True
    learned = False
    to_upload = []
    for d in pending:
        if lookups >= lookups_per_run:
            complete = False
            break
        s = nights.get(d)
        lookups += 1
        try:
            g = garmin.get_sleep_summary(d)
        except Exception as e:
            log.warning("Garmin sleep lookup for %s failed (%s) — will retry next run.", d, e)
            complete = False
            break
        if g:
            store.remember_need(d, g.get("need"))
            store.remember_need(_next_day(d), g.get("next_need"))
        if g is None:
            if s is None or s["end"] < recent:
                seen[d] = "none"
            elif (s.get("processed", True) and d in hrv_by_date) or now - s["end"] >= max_wait:
                to_upload.append(d)
            else:
                log.info("Sleep for %s: waiting for Fitbit to finish processing the night%s.", d,
                         "" if d in hrv_by_date else " and its HRV")
        elif d in store.data["uploaded"] or (s is not None and _same_window(g, s)):
            seen[d] = "ours"
        else:
            seen[d] = "watch"
            if g.get("score") is None:
                continue  # watch night without a score: nothing to learn
            need = g.get("need") or sleep_score.DEFAULT_NEED_MINUTES
            if g.get("hr"):
                store.data["garmin_nights"][d] = {"asleep": round(g["asleep"], 1), "need": need, "hr": g["hr"],
                                                  "score": g["score"]}
                learned = True
            if s is not None:
                hr = health.get_sleeping_heart_rate(s["start"], s["end"])
                pair = {"asleep": round(asleep_minutes(s), 1), "need": need,
                        "hr": None if hr is None else round(hr, 2), "hrv": hrv_by_date.get(d),
                        "score": g["score"], "stress": g.get("stress")}
                if store.data["bootstrapped"]:
                    # What the current models would have said, before learning from this night.
                    pair["pred_score"] = sleep_score.predict(store.weights, pair["asleep"], need, hr)
                    ps = sleep_score.predict_stress(store.stress_weights, hr, pair["hrv"])
                    pair["pred_stress"] = None if ps is None else round(ps, 1)
                    log.info("Watch night %s: Garmin score %s (model said %s), stress %s (model said %s).",
                             d, g["score"], pair["pred_score"], g.get("stress"), pair["pred_stress"])
                store.data["pairs"][d] = pair
                learned = True
    if complete and not store.data["bootstrapped"]:
        store.data["bootstrapped"] = True
        log.info("Sleep calibration: initial scan done, %d watch night(s) found, %d also recorded by Fitbit.",
                 len(store.data["garmin_nights"]), len(store.data["pairs"]))
    if learned:
        store.refit(now.date())
        if store.live_error():
            log.info("Sleep models: %s.", store.live_error())
    store.save()

    for d in sorted(to_upload):
        s = nights[d]
        # Garmin works out the next night's sleep need even after a night we uploaded.
        try:
            prev = garmin.get_sleep_summary(_prev_day(d))
        except Exception as e:
            log.warning("Garmin sleep lookup for %s failed (%s) — using the last known sleep need.", _prev_day(d), e)
            prev = None
        if prev:
            store.remember_need(_prev_day(d), prev.get("need"))
            store.remember_need(d, prev.get("next_need"))
        need = store.data.get("need") or sleep_score.DEFAULT_NEED_MINUTES
        minutes = stage_minutes(s)
        hr = health.get_sleeping_heart_rate(s["start"], s["end"])
        score = sleep_score.predict(store.weights, asleep_minutes(s), need, hr)
        stress = sleep_score.predict_stress(store.stress_weights, hr, hrv_by_date.get(d))
        fit_bytes = build_sleep_fit(
            s["stages"], device["manufacturer"], device["product_id"], device["serial_number"],
            assessment=sleep_score.assessment(minutes, need, score, stress),
            software_version=device.get("software_version", 331),
            utc_offset_seconds=s["utc_offset_seconds"],
        )
        log.info("Uploading sleep for %s (%s–%s UTC, %.0f min asleep, sleeping HR %s, HRV %s ms; "
                 "estimated score %d, stress %s).",
                 d, f"{s['start']:%H:%M}", f"{s['end']:%H:%M}", asleep_minutes(s),
                 "n/a" if hr is None else f"{hr:.0f}", hrv_by_date.get(d, "n/a"), score,
                 "n/a" if stress is None else f"{stress:.0f}")
        # Record first: a retry would only be rejected as a duplicate, and a night that
        # Garmin silently drops must not be uploaded again every hour.
        store.data["uploaded"][d] = s["start"].isoformat()
        store.data["seen"][d] = "uploaded"
        store.save()
        result = garmin.upload_fit_for_window(fit_bytes, s["start"])
        log.info("Sleep upload complete: %s", result)
