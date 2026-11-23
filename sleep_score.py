"""
sleep_score.py
Estimates the Garmin sleep score for a Fitbit night, calibrated against nights the
user slept with their Garmin watch.

Garmin's formula is not public, and Fitbit's stage split differs from Garmin's too
much to help (tested on 200 nights both devices recorded, Mar–Oct 2026). What carries
over is how long you slept compared with Garmin's sleep need, and the average sleeping
heart rate (standing in for Garmin's HRV-based stress). A small linear model on those
predicted Garmin's score within about 4 points on average.

Garmin's average sleep stress is derived from HRV; the Fitbit Air's nightly HRV plus
sleeping heart rate predicted it within about 1.2 stress points (198 nights).

Both models are refitted from the stored watch nights ("pairs") whenever one is added,
shrunk toward their defaults so a handful of nights cannot swing them far.
"""

from datetime import date

FEATURE_NAMES = ["intercept", "hours_short_of_need", "hours_over_need", "sleep_hr_minus_60"]

# Fitted 2026-11-23 from the 200 nights in the module docstring (fit with lam=0.01,
# zero prior); 5-fold cross-validated error 4.1 points, 90% of nights within 10.
DEFAULT_WEIGHTS = [74.54, -4.17, -14.96, -1.35]

STRESS_FEATURE_NAMES = ["intercept", "sleep_hr_minus_60", "hrv_minus_25"]

# Fitted 2026-11-23 on 198 nights (Garmin avgSleepStress vs Fitbit Air sleeping HR and
# daily HRV); cross-validated error 1.2 stress points, 90% within 2.5.
DEFAULT_STRESS_WEIGHTS = [22.30, 0.565, -0.539]
DEFAULT_HRV_MS = 25.0

DEFAULT_NEED_MINUTES = 480
RIDGE_LAMBDA = 10.0      # pull toward the prior, in units of "nights"
HALF_LIFE_DAYS = 365     # older watch nights count less


def features(asleep_minutes: float, need_minutes: float, sleep_hr: float | None) -> list[float]:
    hr = 60.0 if sleep_hr is None else sleep_hr
    return [
        1.0,
        max(0.0, need_minutes - asleep_minutes) / 60,
        max(0.0, asleep_minutes - need_minutes) / 60,
        hr - 60.0,
    ]


def stress_features(sleep_hr: float | None, hrv_ms: float | None) -> list[float]:
    hr = 60.0 if sleep_hr is None else sleep_hr
    hrv = DEFAULT_HRV_MS if hrv_ms is None else hrv_ms
    return [1.0, hr - 60.0, hrv - DEFAULT_HRV_MS]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems here are 4x4)."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for c in range(n):
        p = max(range(c, n), key=lambda k: abs(a[k][c]))
        a[c], a[p] = a[p], a[c]
        b[c], b[p] = b[p], b[c]
        for k in range(n):
            if k != c and a[c][c]:
                f = a[k][c] / a[c][c]
                a[k] = [x - f * y for x, y in zip(a[k], a[c])]
                b[k] -= f * b[c]
    return [b[i] / a[i][i] if a[i][i] else 0.0 for i in range(n)]


def _fit(rows: list[tuple[list[float], float, str]], prior: list[float], lam: float, today: date) -> list[float]:
    """Recency-weighted ridge regression toward `prior`. rows: (features, target, "YYYY-MM-DD")."""
    n = len(prior)
    xtx = [[0.0] * n for _ in range(n)]
    xty = [0.0] * n
    for x, y, day in rows:
        age = max(0, (today - date.fromisoformat(day)).days)
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        for i in range(n):
            xty[i] += w * x[i] * y
            for j in range(n):
                xtx[i][j] += w * x[i] * x[j]
    for i in range(n):
        xtx[i][i] += lam
        xty[i] += lam * prior[i]
    return _solve(xtx, xty)


def fit(pairs: list[dict], prior: list[float] | None = None, lam: float = RIDGE_LAMBDA,
        today: date | None = None) -> list[float]:
    """
    Sleep-score model.
    pairs: [{"asleep": min, "need": min, "hr": bpm | None, "score": int, "date": "YYYY-MM-DD"}, ...]
    """
    rows = [(features(p["asleep"], p["need"], p.get("hr")), p["score"], p["date"]) for p in pairs]
    return _fit(rows, DEFAULT_WEIGHTS if prior is None else prior, lam, today or date.today())


def fit_stress(pairs: list[dict], prior: list[float] | None = None, lam: float = RIDGE_LAMBDA,
               today: date | None = None) -> list[float]:
    """Sleep-stress model; uses pairs that have Garmin's stress, sleeping HR and Fitbit HRV."""
    rows = [(stress_features(p["hr"], p["hrv"]), p["stress"], p["date"]) for p in pairs
            if p.get("stress") is not None and p.get("hr") is not None and p.get("hrv") is not None]
    return _fit(rows, DEFAULT_STRESS_WEIGHTS if prior is None else prior, lam, today or date.today())


def predict(weights: list[float], asleep_minutes: float, need_minutes: float, sleep_hr: float | None) -> int:
    x = features(asleep_minutes, need_minutes, sleep_hr)
    return int(round(max(0.0, min(100.0, sum(w * v for w, v in zip(weights, x))))))


def predict_stress(weights: list[float], sleep_hr: float | None, hrv_ms: float | None) -> float | None:
    """Garmin-style average sleep stress (0-100), or None without a sleeping heart rate."""
    if sleep_hr is None:
        return None
    return max(0.0, min(100.0, sum(w * v for w, v in zip(weights, stress_features(sleep_hr, hrv_ms)))))


# Garmin's ideal ranges (percent of time asleep), as its API reports them.
STAGE_RANGES = {"DEEP": (16, 33), "LIGHT": (30, 64), "REM": (21, 31)}


def _range_score(value: float, lo: float, hi: float) -> int:
    dist = lo - value if value < lo else value - hi if value > hi else 0
    return int(max(0, min(100, round(100 - 3 * dist))))


def assessment(stage_minutes: dict, need_minutes: float, overall: int, stress: float | None = None) -> dict:
    """
    Values for the FIT sleep_assessment message. Only the overall score is shown by
    Garmin; the rest are filled so the message looks like the watch's.
    """
    asleep = sum(stage_minutes.get(k, 0) for k in ("DEEP", "LIGHT", "REM")) or 1
    awake = stage_minutes.get("AWAKE", 0)
    short_h = max(0.0, need_minutes - asleep) / 60
    out = {name: overall for name in (
        "combined_awake_score", "awakenings_count_score", "sleep_quality_score", "sleep_recovery_score",
        "sleep_restlessness_score", "interruptions_score")}
    out.update({
        "overall_sleep_score": overall,
        "sleep_duration_score": int(max(0, min(100, round(100 - 25 * short_h)))),
        "awake_time_score": int(max(0, min(100, round(100 - awake)))),
        "awakenings_count": int(min(254, stage_minutes.get("AWAKE_COUNT", 0))),
    })
    if stress is not None:
        out["average_stress_during_sleep"] = int(round(stress * 100))
    for stage, key in (("DEEP", "deep_sleep_score"), ("LIGHT", "light_sleep_score"), ("REM", "rem_sleep_score")):
        out[key] = _range_score(100 * stage_minutes.get(stage, 0) / asleep, *STAGE_RANGES[stage])
    return out
