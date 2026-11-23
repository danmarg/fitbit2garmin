"""
sleep_score.py
Estimates the Garmin sleep score for a Fitbit night, calibrated against nights the
user slept with their Garmin watch.

Garmin's formula is not public, and Fitbit's stage split differs from Garmin's too
much to help (tested on 200 nights both devices recorded, Mar–Oct 2026). What carries
over is how long you slept compared with Garmin's sleep need, and the average sleeping
heart rate (standing in for Garmin's HRV-based stress). A small linear model on those
predicted Garmin's score within about 4 points on average.

The model is refitted from the stored watch nights ("pairs") whenever one is added,
shrunk toward DEFAULT_WEIGHTS so a handful of nights cannot swing it far.
"""

from datetime import date

FEATURE_NAMES = ["intercept", "hours_short_of_need", "hours_over_need", "sleep_hr_minus_60"]

# Fitted 2026-11-23 from the 200 nights in the module docstring (fit with lam=0.01,
# zero prior); 5-fold cross-validated error 4.1 points, 90% of nights within 10.
DEFAULT_WEIGHTS = [74.54, -4.17, -14.96, -1.35]

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


def fit(pairs: list[dict], prior: list[float] | None = None, lam: float = RIDGE_LAMBDA,
        today: date | None = None) -> list[float]:
    """
    Weighted ridge regression toward `prior`.
    pairs: [{"asleep": min, "need": min, "hr": bpm | None, "score": int, "date": "YYYY-MM-DD"}, ...]
    """
    prior = DEFAULT_WEIGHTS if prior is None else prior
    n = len(prior)
    today = today or date.today()
    xtx = [[0.0] * n for _ in range(n)]
    xty = [0.0] * n
    for p in pairs:
        x = features(p["asleep"], p["need"], p.get("hr"))
        age = max(0, (today - date.fromisoformat(p["date"])).days)
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        for i in range(n):
            xty[i] += w * x[i] * p["score"]
            for j in range(n):
                xtx[i][j] += w * x[i] * x[j]
    for i in range(n):
        xtx[i][i] += lam
        xty[i] += lam * prior[i]
    return _solve(xtx, xty)


def predict(weights: list[float], asleep_minutes: float, need_minutes: float, sleep_hr: float | None) -> int:
    x = features(asleep_minutes, need_minutes, sleep_hr)
    return int(round(max(0.0, min(100.0, sum(w * v for w, v in zip(weights, x))))))


# Garmin's ideal ranges (percent of time asleep), as its API reports them.
STAGE_RANGES = {"DEEP": (16, 33), "LIGHT": (30, 64), "REM": (21, 31)}


def _range_score(value: float, lo: float, hi: float) -> int:
    dist = lo - value if value < lo else value - hi if value > hi else 0
    return int(max(0, min(100, round(100 - 3 * dist))))


def assessment(stage_minutes: dict, need_minutes: float, overall: int) -> dict:
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
    for stage, key in (("DEEP", "deep_sleep_score"), ("LIGHT", "light_sleep_score"), ("REM", "rem_sleep_score")):
        out[key] = _range_score(100 * stage_minutes.get(stage, 0) / asleep, *STAGE_RANGES[stage])
    return out
