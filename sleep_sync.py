"""
sleep_sync.py
Uploads Fitbit sleep to Garmin Connect, with a sleep score calibrated against the
nights the user slept with their own Garmin watch.

Each run looks at Fitbit (Fitbit Air) sleep sessions and asks Garmin about each
night's local wake date, once:
  - Garmin has no sleep → upload the Fitbit night (if recent enough), score from the model
  - Garmin has a night from the watch → store it as a calibration pair, refit the model
  - Garmin has a night we uploaded → ignore it (never calibrate on our own output)

A night counts as ours if it is in the uploaded list, or if Garmin's sleep window equals
the Fitbit window to the second (our uploads copy it exactly; the watch's never do).

The store (sleep.json in the state directory) holds:
  seen:     {wake_date: "uploaded" | "watch" | "ours" | "none"}
  uploaded: {wake_date: fitbit start ISO}
  pairs:    {wake_date: {"asleep", "need", "hr", "hrv", "score", "stress"}}
  weights / stress_weights: current score and stress models; need / need_date: Garmin's sleep need (minutes) on the latest night seen
  bootstrapped: whether the full calibration window has been scanned once
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
        self.data = {"seen": {}, "uploaded": {}, "pairs": {}, "weights": None, "stress_weights": None,
                     "need": None, "bootstrapped": False}
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

    @property
    def weights(self) -> list[float]:
        w = self.data.get("weights")
        return w if isinstance(w, list) and len(w) == len(sleep_score.DEFAULT_WEIGHTS) else sleep_score.DEFAULT_WEIGHTS

    @property
    def stress_weights(self) -> list[float]:
        w = self.data.get("stress_weights")
        ok = isinstance(w, list) and len(w) == len(sleep_score.DEFAULT_STRESS_WEIGHTS)
        return w if ok else sleep_score.DEFAULT_STRESS_WEIGHTS

    def refit(self, today):
        pairs = [{"date": d, **p} for d, p in self.data["pairs"].items()]
        self.data["weights"] = sleep_score.fit(pairs, today=today)
        self.data["stress_weights"] = sleep_score.fit_stress(pairs, today=today)
        fmt = lambda names, ws: ", ".join(f"{n}={w:.2f}" for n, w in zip(names, ws))
        log.info("Sleep models refitted on %d watch night(s): score %s; stress %s", len(pairs),
                 fmt(sleep_score.FEATURE_NAMES, self.data["weights"]),
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


def _same_window(garmin_night: dict, session: dict) -> bool:
    return (abs((garmin_night["start"] - session["start"]).total_seconds()) <= 1
            and abs((garmin_night["end"] - session["end"]).total_seconds()) <= 1)


def run_sleep_sync(cfg: dict, health, garmin, store: SleepStore, now: datetime | None = None):
    scfg = cfg.get("sleep") or {}
    if not scfg.get("enabled", False):
        return
    now = now or datetime.now(timezone.utc)
    holdback = timedelta(hours=scfg.get("holdback_hours", 3))
    upload_days = scfg.get("upload_days", 2)
    calibration_days = scfg.get("calibration_days", 365)
    lookups_per_run = scfg.get("max_garmin_lookups", 40)
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
    # "none" nights become eligible again if upload_days is raised to cover them.
    pending = sorted((d for d in nights if d not in seen or (seen[d] == "none" and nights[d]["end"] >= recent)),
                     reverse=True)

    lookups = 0
    complete = True
    pairs_added = False
    to_upload = []
    for d in pending:
        if lookups >= lookups_per_run:
            complete = False
            break
        s = nights[d]
        lookups += 1
        try:
            g = garmin.get_sleep_summary(d)
        except Exception as e:
            log.warning("Garmin sleep lookup for %s failed (%s) — will retry next run.", d, e)
            complete = False
            break
        if g and g.get("need") and d >= (store.data.get("need_date") or ""):
            store.data["need"], store.data["need_date"] = g["need"], d  # keep the latest night's
        if g is None:
            if s["end"] >= recent:
                to_upload.append(d)
            else:
                store.data["seen"][d] = "none"
        elif d in store.data["uploaded"] or _same_window(g, s):
            store.data["seen"][d] = "ours"
        elif g.get("score") is None:
            store.data["seen"][d] = "watch"  # watch night without a score: nothing to learn
        else:
            hr = health.get_sleeping_heart_rate(s["start"], s["end"])
            store.data["pairs"][d] = {"asleep": round(asleep_minutes(s), 1),
                                      "need": g.get("need") or sleep_score.DEFAULT_NEED_MINUTES,
                                      "hr": None if hr is None else round(hr, 2), "hrv": hrv_by_date.get(d),
                                      "score": g["score"], "stress": g.get("stress")}
            store.data["seen"][d] = "watch"
            pairs_added = True
    if complete and not store.data["bootstrapped"]:
        store.data["bootstrapped"] = True
        log.info("Sleep calibration: initial scan done, %d watch night(s) found.", len(store.data["pairs"]))
    if pairs_added:
        store.refit(now.date())
    store.save()

    need = store.data.get("need") or sleep_score.DEFAULT_NEED_MINUTES
    for d in sorted(to_upload):
        s = nights[d]
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
