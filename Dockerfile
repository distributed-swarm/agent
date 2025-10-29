# Minimal, boring, and reliable base
FROM python:3.11-slim

# Avoid interactive prompts and keep layer noise low
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add curl for internal checks and controller comms
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user
RUN adduser --disabled-password --gecos '' appuser

# OCI labels help traceability in GHCR
LABEL org.opencontainers.image.title="distributed-swarm Agent" \
      org.opencontainers.image.description="Node runner; executes jobs and reports metrics" \
      org.opencontainers.image.source="https://github.com/distributed-swarm/Agent"

# Workdir owned by our non-root user
WORKDIR /app
RUN chown -R appuser:appuser /app
USER appuser

# If you later have requirements.txt, uncomment these:
# COPY --chown=appuser:appuser requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt

# Lightweight healthcheck: confirm controller is reachable from this agent
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://controller:8080/healthz || exit 1

# Stub command so the container proves it's alive
# Replace this with your real entrypoint when ready
CMD ["python", "app.py"]
