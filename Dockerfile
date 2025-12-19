# Use a lightweight Python image
FROM python:3.11-slim

# Prevent python from buffering stdout (helps with logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if your agent does complex image proc
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# Copy any other modules your agent needs
# COPY modules/ ./modules/

# Default connection to the controller (matches your previous docker-compose context)
ENV CONTROLLER_URL="http://controller:8080"
ENV AGENT_NAME="agent-docker-1"

CMD ["python", "app.py"]
