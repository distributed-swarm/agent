# Minimal, boring, and reliable base
FROM python:3.11-slim

# Avoid interactive prompts and keep layer noise low
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create an unprivileged user
RUN useradd -u 10001 -ms /bin/bash appuser

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

# Stub command so the container proves it's alive
# Replace this with your real entrypoint when ready
CMD ["python","-c","print('agent stub up', flush=True); import time; time.sleep(10**9)"]
