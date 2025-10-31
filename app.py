# app.py — heartbeat + simple task worker
import os, time, socket, requests, hashlib
import time, random

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
SLEEP_SECONDS  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "0"))

def post_heartbeat():
    url = f"{CONTROLLER_URL}/healthz"
    try:
        r = requests.get(url, timeout=5)
        print(f"[heartbeat] {AGENT_NAME} -> {url}  status={r.status_code}")
    except Exception as e:
        print(f"[heartbeat] failed: {e}")

def do_task():
    turl = f"{CONTROLLER_URL}/task"
    try:
        # long-poll: controller will wait up to ~2s before returning 204
        r = requests.get(turl, params={"agent": AGENT_NAME}, timeout=5)
    except Exception as e:
        print(f"[task] fetch error: {e}")
        return

    if r.status_code == 204:
        # no work right now; tiny jitter so multiple agents don't all re-ask at once
        time.sleep(random.uniform(0.01, 0.05))
        return

    if r.status_code != 200:
        print(f"[task] unexpected {r.status_code}")
        return

    task = r.json()
    op   = task.get("op")
    payload = task.get("payload")
    tid  = task.get("id", "tsk-unknown")

    start = time.time()
    ok, output = False, None

    try:
        if op == "sha256":
            import hashlib
            output = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            ok = True
        else:
            output = f"unsupported op: {op}"
            ok = False
    except Exception as e:
        output = f"error: {e}"
        ok = False

    duration_ms = int((time.time() - start) * 1000)

    try:
        res = {
            "id": tid,
            "agent": AGENT_NAME,
            "ok": ok,
            "output": output,
            "duration_ms": duration_ms,
            "error": None if ok else output,
        }
        requests.post(f"{CONTROLLER_URL}/result", json=res, timeout=5)
        print(f"[task] result -> {res.get('ok')}  ms={duration_ms}")
    except Exception as e:
        print(f"[task] post error: {e}")
