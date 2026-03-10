
import logging
import os
import sys
import yaml
from datetime import datetime, timezone

# Add current dir to sys.path just in case
sys.path.append(os.getcwd())

from fitbit_client import FitbitClient
from garmin_client import GarminClient
from main import run_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("debug_sync")

def main():
    if not os.path.exists("config.yaml"):
        log.error("config.yaml not found!")
        return

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    log.info("Starting single debug sync cycle...")
    
    fitbit = FitbitClient(
        client_id=cfg["fitbit"]["client_id"],
        client_secret=cfg["fitbit"]["client_secret"],
    )
    
    garmin = GarminClient(
        email=cfg["garmin"]["email"],
        password=cfg["garmin"]["password"],
    )
    
    try:
        log.info("Connecting to Garmin...")
        garmin.connect()
        
        log.info("Running sync (testing with 48h lookback)...")
        # Increase lookback for debug
        cfg["sync"]["lookback_hours"] = 48
        run_sync(cfg, fitbit, garmin)
        log.info("Debug sync cycle finished.")
    except Exception as e:
        log.exception("Debug sync failed: %s", e)

if __name__ == "__main__":
    main()
