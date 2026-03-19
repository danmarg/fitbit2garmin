#!/bin/bash
# install_cron.sh - Install Fitbit2Garmin to system crontab
# Usage: ./install_cron.sh [--frequency "0 */2 * * *"]
# Default frequency: every hour (0 * * * *)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="${PROJECT_DIR}/.venv"
FREQUENCY="${1:-0 * * * *}"
LOG_DIR="${PROJECT_DIR}/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Generate cron entry
CRON_ENTRY="$FREQUENCY cd '$PROJECT_DIR' && '$VENV_DIR/bin/python' main.py >> '$LOG_DIR/cron.log' 2>&1"

echo "Installing Fitbit2Garmin to crontab..."
echo "Schedule: $FREQUENCY"
echo "Project: $PROJECT_DIR"
echo "Venv: $VENV_DIR"
echo "Logs: $LOG_DIR/cron.log"
echo ""

# Add to crontab if not already present
if (crontab -l 2>/dev/null || true) | grep -F "python' main.py" > /dev/null 2>&1; then
    echo "⚠️  Fitbit2Garmin already appears to be in crontab. Skipping..."
    echo "To update the frequency, remove the old entry and run this script again:"
    echo "  crontab -e"
    exit 0
fi

# Install new cron entry
(crontab -l 2>/dev/null || echo "") | (cat; echo "$CRON_ENTRY") | crontab -

echo "✓ Fitbit2Garmin installed to crontab"
echo ""
echo "To view the cron job:"
echo "  crontab -l | grep main.py"
echo ""
echo "To remove it later:"
echo "  crontab -e  # and delete the line"
