# Neuro Fabric / agent image
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Basic system deps (keep minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && if [ -s requirements.txt ]; then python -m pip install -r requirements.txt; fi

# Agent code
COPY app.py worker_sizing.py ./
COPY ops ./ops

# Sanity: ensure ops import works at build time
RUN python -c "import ops; print('ops import OK')"

# Non-root user
RUN useradd -u 10001 -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "/app/app.py"]
