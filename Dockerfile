# Neuro Fabric / agent image

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Basic system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir for the agent code
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py worker_sizing.py ./
# 🔴 This was missing before: ship the ops package into the image
COPY ops ./ops

# Make "python" and "python3" behave the same
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python || true

# Non-root user
RUN useradd -u 10001 -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Default command: run the agent
CMD ["python", "/app/app.py"]
