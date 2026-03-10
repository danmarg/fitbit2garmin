FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py ./
COPY config.yaml.example ./

# Volume for persisted auth tokens and config
VOLUME ["/app/data"]

ENV CONFIG_FILE=/app/data/config.yaml

CMD ["python", "main.py"]
