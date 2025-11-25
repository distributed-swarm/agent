# Agent image for Neuro Fabric / distributed-swarm

FROM python:3.11-slim-bookworm

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Basic OS deps (curl so you can debug, plus build tools if pip ever needs them)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Work in /app
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt ./requirements.txt

RUN if [ -s requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY app.py worker_sizing.py ./

# Make "python" and "python3" behave the same inside the container
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python || true

# Create non-root user
RUN useradd -u 10001 -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Default command: run the agent
CMD ["python", "/app/app.py"]
