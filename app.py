# app.py — heartbeat + simple task worker (sha256 + map_tokenize)

import os, time, socket, random, signal, json
import requests
from typing import Optional

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))   # heartbeat cadence
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "0"))         # sleep after each task (ms)
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

# ------------------ operation handlers ------------------

def handle_sha256(payload: str) -> dict:
    import hashlib
    return {"ok": True, "output": hashlib.sha256(payload.encode("utf-8")).hexdigest()}

def handle_map_tokenize(task: dict) -> dict:
    """
    Expect task payload like:
      {"source": {"file": "C:/path/to/demo.jsonl"}}
    Each JSONL line should contain a {"text": "..."} field.
    """
    try:
        src = (((task.get("payload") or {}).get("source") or {}).get("file"))
        if not src:
            return {"ok": False, "error": "missing payload.source.file"}

        records = 0
        total_tokens = 0

        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                txt = str(obj.get("text", ""))
                # simple whitespace tokenization
                total_tokens += len(txt.split())
                records += 1

        avg_tokens = (total_tokens / records) if records else 0.0
        return {"ok": True, "output": {
            "records": records,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens
        }}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ------------------ task runner ------------------

def run_task(task: dict) -> dict:
    tid = task.get("id", "tsk-unknown")
    op  = task.get("op")
    payload = task.get("payload", "")

    # normalize payload to str for hashing path only
    payload_str = payload
    if not isinstance(payload_str, str):
        try:
            payload_str = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload_str = str(payload)

    start = time.time()
    ok, output, err = False, None, None

    try:
        if op == "sha256":
            r = handle_sha256(payload_str)
            ok, output, err = r.get("ok", False), r.get("output"), r.get("error")
        elif op == "map_tokenize":
            r = handle_map_tokenize(task)
            ok, output, err = r.get("ok", False), r.get("output"), r.get("error")
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
