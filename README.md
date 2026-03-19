# Fitbit2Garmin

Pulls Fitbit intraday heart rate and step data and injects it into Garmin Connect as `monitoring_b` FIT files, impersonating your registered Garmin device. This populates 24/7 HR data in Garmin Connect, which feeds Training Load, Suggested Workouts, and Physio TrueUp calculations.

## How It Works

1. Fetches intraday HR + steps from the Fitbit API (1-minute resolution)
2. Encodes a valid Garmin `monitoring_b` FIT file using the exact binary format produced by your device's firmware
3. Uploads to Garmin Connect via the device sync endpoint, skipping time windows already covered by real device data

---

## Prerequisites

- A Fitbit account with a synced device
- A Garmin Connect account with an active wearable (used to clone the device identity)
- Docker (for deployment) or Python 3.11+

---

## Setup

### 1. Fitbit App (Personal access required)

Intraday data requires a **Personal** app type.

1. Go to [dev.fitbit.com](https://dev.fitbit.com) → Manage → Register an App
2. Set **OAuth 2.0 Application Type** to **Personal**
3. Set **Redirect URL** to `http://localhost:8080/`
4. Note your **Client ID** and **Client Secret**

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
fitbit:
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

### 4. Authorize Fitbit

Run the authorization flow once to generate `token.json`:

```bash
python -c "
import yaml
from fitbit_client import FitbitClient
cfg = yaml.safe_load(open('config.yaml'))
FitbitClient(cfg['fitbit']['client_id'], cfg['fitbit']['client_secret']).authorize()
"
```

Open the printed URL in your browser, approve access, then paste the redirect URL back. A `token.json` will be saved.

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
cp token.json  /opt/fitbit2garmin/data/token.json   # from the auth step above

# Copy your existing Garmin session (from garth login, usually at ~/.garth)
cp -r ~/.garth /opt/fitbit2garmin/data/garth
```

### Run

```bash
docker run -d \
  --name fitbit2garmin \
  --restart unless-stopped \
  -v /opt/fitbit2garmin/data:/app/data \
  -e CONFIG_FILE=/app/data/config.yaml \
  -e GARTH_HOME=/app/data/garth \
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
  -e GARTH_HOME=/app/data/garth \
  fitbit2garmin python debug_sync.py >> /var/log/fitbit2garmin.log 2>&1
```

---

## Token Refresh

- **Fitbit**: tokens auto-refresh via `requests-oauthlib`. The updated token is saved back to `token.json` automatically.
- **Garmin**: the `garth` session is cached in `GARTH_HOME` and refreshed on expiry.

If the Fitbit token ever expires completely, re-run the authorization step in Setup §4.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestration loop |
| `fit_engine.py` | Encodes binary `monitoring_b` FIT files |
| `fitbit_client.py` | Fetches intraday HR + steps from Fitbit API |
| `garmin_client.py` | Uploads FIT files to Garmin Connect |
| `identity_grabber.py` | Extracts device identity from a `.fit` file |
| `debug_sync.py` | Single-shot sync for testing |
| `config.yaml.example` | Configuration template |
