# Fitbit2Garmin (Fitbit to Garmin)

A specialized synchronization tool that bridges the gap between Fitbit and Garmin ecosystems by injecting high-resolution Fitbit intraday data (heart rate and steps) into Garmin Connect.

## Project Overview

**Purpose:**
Fitbit2Garmin pulls 1-minute resolution heart rate and step data from the Fitbit Web API and encodes it into Garmin `monitoring_b` (Type 9) FIT files. By impersonating a registered Garmin device, this data is accepted by Garmin Connect as native "wellness" data, which in turn populates metrics like **Training Load**, **Daily Suggested Workouts**, and **Physio TrueUp**—features usually reserved for data recorded directly on Garmin wearables.

**Core Technologies:**
- **Python 3.11+**: The primary runtime.
- **garth**: Handles Garmin Connect authentication, session persistence, and binary file uploads.
- **fitbit**: Manages OAuth2 flows and intraday time-series data retrieval from Fitbit.
- **garmin-fit-sdk**: Used for low-level encoding of binary FIT files to ensure protocol compliance and valid CRCs.
- **Docker**: Provides a containerized environment with persistent volume support for state and credentials.

**Key Components:**
- `main.py`: The orchestration loop that manages sync cycles, state persistence, and error handling.
- `fit_engine.py`: A custom FIT encoder that builds valid `monitoring_b` files using Garmin's official SDK definitions.
- `fitbit_client.py`: Handles Fitbit API interactions, including token refresh and timezone-aware data fetching.
- `garmin_client.py`: Manages Garmin API interactions and implements a coverage check to avoid overwriting real device data.
- `identity_grabber.py`: A utility to extract device identity (manufacturer, product ID, serial number) from an existing Garmin FIT file.

---

## Building and Running

### Prerequisites
- Python 3.11 or higher.
- A Fitbit "Personal" app registration (for intraday API access).
- A Garmin Connect account with a primary wellness tracker.

### Local Installation
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configuration:**
    - Copy `config.yaml.example` to `config.yaml`.
    - Populate with your Fitbit API credentials and Garmin account details.
    - Use `identity_grabber.py` on a real Garmin wellness FIT file to find your `product_id` and `serial_number`.
3.  **Fitbit Authorization:**
    - Run the interactive authorization flow:
      ```bash
      python -c "import yaml; from fitbit_client import FitbitClient; cfg = yaml.safe_load(open('config.yaml')); FitbitClient(cfg['fitbit']['client_id'], cfg['fitbit']['client_secret']).authorize()"
      ```
4.  **Start Syncing:**
    ```bash
    python main.py
    ```

### Running with Docker
1.  **Build Image:**
    ```bash
    docker build -t fitbit2garmin .
    ```
2.  **Run Container:**
    ```bash
    docker run -d \
      --name fitbit2garmin \
      --restart unless-stopped \
      -v /path/to/data:/app/data \
      -e CONFIG_FILE=/app/data/config.yaml \
      -e GARTH_HOME=/app/data/garth \
      fitbit2garmin
    ```

### Testing
Run the suite using `pytest`:
```bash
pytest test_pipeline_logic.py -v
```

---

## Development Conventions

- **Surgical Logic**: Every modification to `fit_engine.py` must maintain strict compatibility with the Garmin FIT profile (Type 9).
- **State Management**: The sync loop relies on `data/state.json` to track the last successfully uploaded timestamp. This prevents redundant API calls and duplicate uploads.
- **Collision Avoidance**:
  - **Recency Buffer**: Data newer than 60 minutes (configurable) is held back to allow real Garmin devices to sync first.
  - **Coverage Check**: Before uploading, the tool queries Garmin's wellness API to check for existing data points, skipping any minutes already populated.
- **Logging**: Use the `fitbit2garmin` logger. All sync cycles, uploads, and errors should be logged with appropriate levels (`INFO`, `WARNING`, `ERROR`).
- **FIT File Identity**: The filename of uploaded FIT files includes the window start time and a unique timestamp to prevent 409 Conflict errors from Garmin on retries.
