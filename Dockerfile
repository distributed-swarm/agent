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

# Python deps
COPY requirements.txt ./requirements.txt

RUN if [ -s requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# Agent code
COPY app.py worker_sizing.py ./ 
COPY ops ./ops
RUN python -c "import ops; print('ops import OK')"

# Make "python" and "python3" behave the same
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python || true

# Non-root user
RUN useradd -u 10001 -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Default command: run the agent
CMD ["python", "/app/app.py"]
