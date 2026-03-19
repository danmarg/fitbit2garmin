FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py ./
COPY config.yaml.example ./
COPY crontab /tmp/crontab

# Install crontab
RUN crontab /tmp/crontab && rm /tmp/crontab

# Volume for persisted auth tokens and config
VOLUME ["/app/data"]

ENV CONFIG_FILE=/app/data/config.yaml

# Run cron in foreground
CMD ["cron", "-f"]
