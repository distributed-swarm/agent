# Minimal base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# tools we use in healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# non-root user
RUN useradd -u 10001 -ms /bin/bash appuser

# Workdir
WORKDIR /app
RUN chown -R appuser:appuser /app
USER appuser

# Dependencies (optional but safe if we use requests)
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN [ -f requirements.txt ] && pip install --no-cache-dir -r requirements.txt || true

# Your code
COPY --chown=appuser:appuser app.py /app/app.py

# Healthcheck: confirm controller reachable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://controller:8080/healthz || exit 1

# Run the agent
CMD ["python", "/app/app.py"]

