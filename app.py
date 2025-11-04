# app.py — heartbeat + simple task worker (stable long-poll + backoff)
import os, time, socket, random, signal, json
import requests
from typing import Optional

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))   # heartbeat cadence
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "0"))         # sleep after each task (ms allowed via env too)
WAIT_MS        = int(os.getenv("TASK_WAIT_MS", "2000"))       # server long-poll wait
TIMEOUT_SEC    = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))    # request timeout

# graceful shutdown
_running = True
def _stop(*_):
    global _running
    _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# single Session keeps TCP alive and reduces overhead
session = requests.Session()
session.headers.update({"X-Agent": AGENT_NAME})

def log(msg):
    print(f"[agent:{AGENT_NAME}] {msg}", flush=True)

def heartbeat():
    try:
        r = session.get(f"{CONTROLLER_URL}/healthz", timeout=TIMEOUT_SEC)
        log(f"heartbeat -> {r.status_code}")
    except Exception as e:
        log(f"heartbeat failed: {e}")

def fetch_task() -> Optional[dict]:
    try:
        r = session.get(
            f"{CONTROLLER_URL}/task",
            params={"agent": AGENT_NAME, "wait_ms": WAIT_MS},
            timeout=TIMEOUT_SEC,
        )
        if r.status_code == 204:
            return None
        if r.status_code != 200:
            log(f"task fetch unexpected {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        log(f"task fetch error: {e}")
        return None

def run_task(task: dict) -> dict:
    tid = task.get("id", "tsk-unknown")
    op  = task.get("op")
    payload = task.get("payload", "")

    # normalize payload to str for hashing
    if not isinstance(payload, str):
        try:
            payload = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload = str(payload)

    start = time.time()
    ok, output, err = False, None, None

    try:
        if op == "sha256":
            import hashlib
            output = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            ok = True
        else:
            err = f"unsupported op: {op}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    duration_ms = int((time.time() - start) * 1000)

    res = {
        "id": tid,
        "agent": AGENT_NAME,
        "ok": ok,
        "output": output if ok else None,
        "duration_ms": duration_ms,
        "error": None if ok else err,
    }
    return res

def post_result(res: dict):
    try:
        session.post(f"{CONTROLLER_URL}/result", json=res, timeout=TIMEOUT_SEC)
        log(f"result -> ok={res['ok']} ms={res['duration_ms']}")
    except Exception as e:
        log(f"result post error: {e}")

def main():
    log("starting")
    last_hb = 0.0
    backoff = 0.25  # seconds, grows on errors, shrinks on success

    while _running:
        now = time.time()
        if now - last_hb >= HEARTBEAT_SEC:
            heartbeat()
            last_hb = now

        task = fetch_task()
        if task is None:
            # no work, small jitter to avoid stampedes
            time.sleep(random.uniform(0.02, 0.06))
            continue

        res = run_task(task)
        post_result(res)

        # gentle pacing after each task
        if TASK_INTERVAL > 0:
            time.sleep(TASK_INTERVAL / 1000.0)

        # success path reduces backoff
        backoff = max(0.1, backoff * 0.5)

    log("stopping")

if __name__ == "__main__":
    main()

