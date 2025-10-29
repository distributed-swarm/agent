# app.py — heartbeat + simple task worker
import os, time, socket, requests, hashlib

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
SLEEP_SECONDS  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "5"))

def post_heartbeat():
    url = f"{CONTROLLER_URL}/healthz"
    try:
        r = requests.get(url, timeout=5)
        print(f"[heartbeat] {AGENT_NAME} -> {url}  status={r.status_code}")
    except Exception as e:
        print(f"[heartbeat] failed: {e}")

def do_task():
    # pull a task
    turl = f"{CONTROLLER_URL}/task"
    try:
        r = requests.get(turl, params={"agent": AGENT_NAME}, timeout=5)
        if r.status_code == 204:
            return  # no task, normal
        r.raise_for_status()
        task = r.json()
    except Exception as e:
        print(f"[task] fetch error: {e}")
        return

    tid = task.get("id", "unknown")
    op = task.get("op")
    payload = task.get("payload", "")
    started = time.time()

    ok, output, err = True, None, None
    try:
        if op == "sha256":
            output = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        else:
            ok, err = False, f"unsupported op: {op}"
    except Exception as e:
        ok, err = False, str(e)

    duration_ms = int((time.time() - started) * 1000)
    print(f"[task] {tid} op={op} ok={ok} ms={duration_ms}")

    # post result
    try:
        rurl = f"{CONTROLLER_URL}/result"
        r = requests.post(rurl, json={
            "id": tid, "agent": AGENT_NAME, "ok": ok,
            "output": output, "duration_ms": duration_ms, "error": err
        }, timeout=5)
        print(f"[task] result -> {r.status_code}")
    except Exception as e:
        print(f"[task] post error: {e}")

def main():
    print(f"agent {AGENT_NAME} started; controller={CONTROLLER_URL}")
    t0 = 0
    while True:
        now = time.time()
        if int(now - t0) >= SLEEP_SECONDS:
            post_heartbeat()
            t0 = now
        do_task()
        time.sleep(TASK_INTERVAL)

if __name__ == "__main__":
    main()

