# Minimal base
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTROLLER_URL=http://controller:8080

# tools for healthcheck; keep it small
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# workdir
WORKDIR /app

# deps first for layer cache
COPY requirements.txt ./requirements.txt
RUN if [ -s requirements.txt ]; then \
      pip install --no-cache-dir --upgrade pip && \
      pip install --no-cache-dir -r requirements.txt ; \
    fi

# app code – IMPORTANT: include worker_sizing.py
COPY app.py worker_sizing.py ./

# non-root user after deps are installed
RUN useradd -u 10001 -ms /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# healthcheck respects CONTROLLER_URL (shell form so env is expanded)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD sh -c 'curl -fsS "$CONTROLLER_URL/healthz" || exit 1'

# run
CMD ["python", "/app/app.py"]
