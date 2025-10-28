# app.py — agent heartbeat routine

import os, time, json, socket, requests

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
SLEEP_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

def post_heartbeat():
    url = f"{CONTROLLER_URL}/heartbeat"
    payload = {
        "agent_name": AGENT_NAME,
        "ip": socket.gethostbyname(socket.gethostname()),
    }
    try:
        r = requests.post(url, json=payload, timeout=5)
        print(f"[heartbeat] {AGENT_NAME} → {url}  status={r.status_code}")
    except Exception as e:
        print(f"[heartbeat] failed: {e}")

def main():
    print(f"agent {AGENT_NAME} started; controller={CONTROLLER_URL}")
    while True:
        post_heartbeat()
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
