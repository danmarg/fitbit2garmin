# Fitbit2Garmin

Pulls Fitbit intraday heart rate (via the Google Health API) and step data and injects it into Garmin Connect as `monitoring_b` FIT files, impersonating your registered Garmin device. This populates 24/7 HR data in Garmin Connect, which feeds Training Load, Suggested Workouts, and Physio TrueUp calculations.

## How It Works

1. Fetches intraday HR + steps from the Google Health API (aggregated to 1-minute resolution)
2. Encodes a valid Garmin `monitoring_b` FIT file using the exact binary format produced by your device's firmware
3. Uploads to Garmin Connect via the device sync endpoint, skipping time windows already covered by real device data

---

## Prerequisites

- A Fitbit device linked to a Google account (Fitbit accounts are migrating to Google accounts)
- A Garmin Connect account with an active wearable (used to clone the device identity)
- Docker (for deployment) or Python 3.11+

---

## Setup

### 1. Google Health API OAuth client

The legacy Fitbit Web API shuts down on 2026-11-02; this project uses the [Google Health API](https://developers.google.com/health).

1. In [Google Cloud Console](https://console.cloud.google.com), create a project and enable the **Google Health API**
2. Configure the OAuth consent screen and add yourself as a **test user** (note: Google expires refresh tokens after 7 days for apps left in *Testing* mode — publish the app to "In production" for personal use to avoid re-authorizing weekly)
3. Create an **OAuth client ID** with a redirect URI of `http://localhost:8080/`
4. The app requests these read scopes: `googlehealth.activity_and_fitness.readonly` (steps), `googlehealth.health_metrics_and_measurements.readonly` (heart rate) and `googlehealth.settings.readonly` (timezone)
5. Note your **Client ID** and **Client Secret**

Existing Fitbit tokens do not carry over; you must authorize again (§4).

### 2. Extract Your Garmin Device Identity

Your config must match the device that Garmin has registered as your primary wellness tracker. Use a **wellness** FIT file (not an activity file) — download one from Garmin Connect:

```
Garmin Connect → Health Stats → Download (select a daily wellness export)
```

Then extract the identity:

```bash
python identity_grabber.py path/to/WELLNESS.fit
```

Note the `product_id`, `serial_number`, and `software_version` values.

### 3. Configure

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml`:

```yaml
google_health:
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"

garmin:
  email: "your@email.com"
  password: "yourpassword"

device:
  manufacturer: 1
  product_id: 4063        # from identity_grabber.py on a wellness file
  serial_number: 3439974151
  software_version: 331   # firmware version * 100

sync:
  lookback_hours: 4
  interval_minutes: 120
```

### 4. Authorize Google Health

Run the authorization flow once to generate `google_health_token.json`:

```bash
python -c "
import yaml
from google_health_client import GoogleHealthClient
cfg = yaml.safe_load(open('config.yaml'))
GoogleHealthClient(cfg['google_health']['client_id'], cfg['google_health']['client_secret']).authorize()
"
```

Open the printed URL in your browser, approve access, then paste the redirect URL back.

---

## Sleep (experimental)

Off by default. When `sync.sleep.enabled` is `true`, each cycle also sends your Google Health sleep sessions to Garmin as sleep-stage records (awake, light, deep, REM, one per minute).

- It needs the `googlehealth.sleep.readonly` scope. Tokens created before this feature don't have it, so run the authorization step (§4) again.
- A night is **skipped if Garmin already has sleep for that date**, for example from a real watch, so it never replaces your watch's data. Sessions shorter than `min_minutes` (naps) are ignored, and sleep that ended within `recency_minutes` waits.
- Whether Garmin Connect stores and displays imported sleep, and how its sleep score and recovery numbers react to it, is untested. Try it on a night without watch data first.
- HRV is not imported.

---

## Running Locally

```bash
pip install -r requirements.txt
python main.py
```

For a single debug run (48h lookback):

```bash
python debug_sync.py
```

For tests:

```bash
pytest test_pipeline_logic.py -v
```

---

## Docker Deployment

The container runs the sync loop continuously (`interval_minutes` from config.yaml). Auth state is persisted via a mounted data directory.

### Build

```bash
docker build -t fitbit2garmin .
```

### Prepare the data directory

```bash
mkdir -p /opt/fitbit2garmin/data
cp config.yaml /opt/fitbit2garmin/data/config.yaml
cp google_health_token.json /opt/fitbit2garmin/data/google_health_token.json   # from the auth step above

# Copy your existing Garmin session (usually at ~/.garminconnect)
cp -r ~/.garminconnect /opt/fitbit2garmin/data/garminconnect
```

### Run

```bash
docker run -d \
  --name fitbit2garmin \
  --restart unless-stopped \
  -v /opt/fitbit2garmin/data:/app/data \
  -e CONFIG_FILE=/app/data/config.yaml \
  -e STATE_DIRECTORY=/app/data/\
  fitbit2garmin
```

View logs:

```bash
docker logs -f fitbit2garmin
```

### Host Cron (alternative to the built-in loop)

If you prefer cron over the container's sleep loop, set `interval_minutes` very high (e.g. 9999) and add a host cron job:

```bash
crontab -e
```

```cron
0 */2 * * * docker run --rm \
  -v /opt/fitbit2garmin/data:/app/data \
  -e CONFIG_FILE=/app/data/config.yaml \
  -e STATE_DIRECTORY=/app/data/ \
  fitbit2garmin python debug_sync.py >> /var/log/fitbit2garmin.log 2>&1
```

---

## Token Refresh

- **Google Health**: the access token auto-refreshes using the stored refresh token and is saved back to `google_health_token.json`.
- **Garmin**: the session is cached in `STATE_DIRECTORY` and refreshed on expiry.

If the Google Health token ever expires completely, re-run the authorization step in Setup §4.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestration loop |
| `fit_engine.py` | Encodes binary `monitoring_b` FIT files |
| `google_health_client.py` | Fetches HR + steps from the Google Health API |
| `garmin_client.py` | Uploads FIT files to Garmin Connect |
| `identity_grabber.py` | Extracts device identity from a `.fit` file |
| `debug_sync.py` | Single-shot sync for testing |
| `config.yaml.example` | Configuration template |
