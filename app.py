# agent/app.py
from __future__ import annotations

import json
import os
import signal
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests


AGENT_NAME = os.getenv("AGENT_NAME", "agent-unknown").strip() or "agent-unknown"
CTRL = os.getenv("CONTROLLER_URL", "http://localhost:8080").rstrip("/")
API_PREFIX = os.getenv("API_PREFIX", "").strip().rstrip("/")  # usually ""

# Networking / pacing
LEASE_BACKOFF_SEC = float(os.getenv("LEASE_BACKOFF_SEC", "0.05"))
ERROR_BACKOFF_SEC = float(os.getenv("ERROR_BACKOFF_SEC", "1.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))

# Toggle tracing without changing behavior
DEBUG_TRACE = os.getenv("DEBUG_TRACE", "0") == "1"

# Lease params
MAX_TASKS = int(os.getenv("MAX_TASKS", "1"))
LEASE_TIMEOUT_MS = int(os.getenv("LEASE_TIMEOUT_MS", "10000"))

# Ops loading behavior
OPS_VERBOSE = os.getenv("OPS_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}
OPS_STRICT = os.getenv("OPS_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}

# If you want to override module list without changing code:
# OPS_MODULES="echo,map_tokenize,gpu_warmup"
OPS_MODULES_ENV = os.getenv("OPS_MODULES", "").strip()

# Global registry the agent actually executes
OPS: Dict[str, Any] = {}
CAPABILITIES_LIST: List[str] = []

_stop = False


def _url(path: str) -> str:
    # path should start with "/"
    return f"{CTRL}{API_PREFIX}{path}"

def _normalize_lease(body: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize controller lease responses into a canonical task dict.

    Accepts:
      - {"lease_id","job_epoch","job": {...}}
      - {"lease_id","job_epoch","task": {...}}
      - {"lease_id","job_epoch","jobs": [ {...} ]}
      - {"lease_id","job_epoch","tasks":[ {...} ]}

    Returns:
      {"job_id","job_epoch","lease_id","op","payload"} or None
    """
    if not body or not isinstance(body, dict):
        return None

    lease_id = body.get("lease_id")
    job_epoch = body.get("job_epoch")

    job = None
    if isinstance(body.get("job"), dict):
        job = body["job"]
    elif isinstance(body.get("task"), dict):
        job = body["task"]
    elif isinstance(body.get("jobs"), list) and body["jobs"]:
        job = body["jobs"][0]
    elif isinstance(body.get("tasks"), list) and body["tasks"]:
        job = body["tasks"][0]

    if not isinstance(job, dict):
        return None

    job_id = job.get("id") or job.get("job_id")
    op = job.get("op")
    payload = job.get("payload", {})

    if not job_id or not op or not lease_id or job_epoch is None:
        return None

    return {
        "job_id": job_id,
        "job_epoch": job_epoch,
        "lease_id": lease_id,
        "op": op,
        "payload": payload if isinstance(payload, dict) else {},
    }

def _log(msg: str) -> None:
    print(msg, flush=True)


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = _url(path)
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code == 204:
            if DEBUG_TRACE:
                _log(f"[trace] POST {url} -> 204 in {time.time()-t0:.3f}s")
            return 204, None

        # best-effort JSON, fallback to text
        try:
            body: Any = r.json()
        except Exception:
            body = r.text

        if DEBUG_TRACE:
            _log(f"[trace] POST {url} -> {r.status_code} in {time.time()-t0:.3f}s")
        return r.status_code, body
    except Exception as e:
        if DEBUG_TRACE:
            _log(f"[trace] POST {url} -> EXC {type(e).__name__}: {e} in {time.time()-t0:.3f}s")
        return 0, f"{type(e).__name__}: {e}"

def _load_ops() -> None:
    """
    Load ops modules via ops.load_modules() and use ops.OPS as the registry.

    Important: Do NOT manually call module-local registration functions.
    If we do, we summon the ancient circular-import demon and nobody sleeps.
    """
    global OPS, CAPABILITIES_LIST

    try:
        import ops as ops_pkg

        modules: Optional[List[str]] = None
        if OPS_MODULES_ENV:
            # "a,b,c" -> ["a","b","c"]
            modules = [x.strip() for x in OPS_MODULES_ENV.split(",") if x.strip()]

        ops_pkg.load_modules(modules=modules, verbose=OPS_VERBOSE, strict=OPS_STRICT)
        OPS = dict(getattr(ops_pkg, "OPS", {}) or {})
        CAPABILITIES_LIST = sorted(list(OPS.keys()))
        return

    except Exception as e:
        _log(f"[agent-v1] ops load failed: {type(e).__name__}: {e}")
        _log(traceback.format_exc())
        OPS = {}
        CAPABILITIES_LIST = []


def _heartbeat() -> None:
    """
    Optional but useful: keep last_seen fresh on controller.
    Controller endpoints vary across builds; tolerate failure silently.
    """
    payload = {
        "name": AGENT_NAME,
        "labels": {},
        "worker_profile": {},
        "capabilities": {"ops": CAPABILITIES_LIST},
        "metrics": {},
    }

    code, body = _post_json("/v1/agents/heartbeat", payload)
    if code in (200, 204):
        return
    # Don't spam if controller doesn't implement it; still run leases.
    if code not in (0, 404):
        _log(f"[agent-v1] heartbeat unexpected HTTP {code}: {body}")


def _task_job_id(task: Dict[str, Any]) -> str:
    return str(task.get("id") or task.get("job_id") or "")


def _extract_task_fields(task: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    job_id = _task_job_id(task)
    op = str(task.get("op") or "")
    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be object/dict, got {type(payload).__name__}")
    if not job_id:
        raise ValueError("missing job_id")
    if not op:
        raise ValueError("missing op")
    return job_id, op, payload


def _lease_once() -> Optional[Tuple[str, int, Dict[str, Any]]]:
    payload = {
        "agent": AGENT_NAME,
        "capabilities": {"ops": CAPABILITIES_LIST},
        "max_tasks": MAX_TASKS,
        "timeout_ms": LEASE_TIMEOUT_MS,
    }

    code, body = _post_json("/v1/leases", payload)
    if code == 204:
        return None
    if code != 200 or not isinstance(body, dict):
        return None

    lease_id = str(body.get("lease_id") or "")

    # job_epoch may live at top-level OR inside the job/task
    job_epoch = int(
        body.get("job_epoch")
        or (body.get("task") or {}).get("job_epoch")
        or (body.get("job") or {}).get("job_epoch")
        or 0
    )

    task = None

    if isinstance(body.get("task"), dict):
        task = body["task"]
    elif isinstance(body.get("job"), dict):
        task = body["job"]
    elif isinstance(body.get("tasks"), list) and body["tasks"]:
        t0 = body["tasks"][0]
        task = t0 if isinstance(t0, dict) else None
    elif isinstance(body.get("jobs"), list) and body["jobs"]:
        j0 = body["jobs"][0]
        task = j0 if isinstance(j0, dict) else None

    job_id = str((task or {}).get("id") or (task or {}).get("job_id") or "")
    op = str((task or {}).get("op") or "")

    # If controller returned a lease wrapper but no runnable job,
    # treat this as idle (NOT a lease).
    if not lease_id or job_epoch <= 0 or not job_id or not op:
        if DEBUG_TRACE and lease_id:
            _log(f"[trace] NO_TASK lease_id={lease_id} job_epoch={job_epoch}")
        return None

    if DEBUG_TRACE and lease_id and isinstance(task, dict):
        _log(
            f"[trace] LEASED lease_id={lease_id} job_id={task.get('id') or task.get('job_id')} "
            f"job_epoch={job_epoch} op={task.get('op')}"
        )

    if lease_id and isinstance(task, dict):
        return lease_id, job_epoch, task
    return None


def _run_task(op: str, payload: Dict[str, Any]) -> Any:
    fn = OPS.get(op)
    if not fn:
        raise KeyError(f"op not supported: {op}")
    # op functions are expected to be pure-ish and return JSON-serializable output
    return fn(payload)


def _post_result(lease_id: str, job_id: str, job_epoch: int, status: str, result: Any, error: Any) -> None:
    payload = {
        "lease_id": lease_id,
        "job_id": job_id,
        "job_epoch": job_epoch,
        "status": status,
        "result": result,
        "error": error,
    }
    code, body = _post_json("/v1/results", payload)
    if code in (200, 204):
        return
    raise RuntimeError(f"results post failed HTTP {code}: {body}")


def _handle_sigterm(signum: int, frame: Any) -> None:
    global _stop
    _stop = True


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _load_ops()

    _log(
        f"[agent-v1] starting v1-only controller='{CTRL}' api_prefix='{API_PREFIX}' "
        f"agent='{AGENT_NAME}' ops={CAPABILITIES_LIST}"
    )

    last_hb = 0.0

    while not _stop:
        try:
            now = time.time()
            if now - last_hb > 5.0:
                _heartbeat()
                last_hb = now

            leased = _lease_once()
            if not leased:
                time.sleep(LEASE_BACKOFF_SEC)
                continue

            lease_id, job_epoch, task = leased

            try:
                job_id, op, payload = _extract_task_fields(task)
                if DEBUG_TRACE:
                    _log(f"[trace] START job_id={job_id} op={op} lease_id={lease_id} job_epoch={job_epoch}")
            except Exception as e:
                job_id = _task_job_id(task)
                err = {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc(), "task": task}
                if job_id:
                    _post_result(lease_id, job_id, job_epoch, "failed", None, err)
                continue

            try:
                t0 = time.time()
                out = _run_task(op, payload)
                if DEBUG_TRACE:
                    _log(f"[trace] FINISH job_id={job_id} op={op} status=succeeded op_secs={time.time()-t0:.3f}")
                _post_result(lease_id, job_id, job_epoch, "succeeded", out, None)
                if DEBUG_TRACE:
                    _log(
                        f"[trace] RESULT_SENT_ATTEMPT job_id={job_id} lease_id={lease_id} "
                        f"job_epoch={job_epoch} status=succeeded"
                    )
            except Exception as e:
                if DEBUG_TRACE:
                    _log("[trace] OP EXCEPTION:\n" + traceback.format_exc())
                err = {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc()}
                try:
                    _post_result(lease_id, job_id, job_epoch, "failed", None, err)
                    if DEBUG_TRACE:
                        _log(
                            f"[trace] RESULT_SENT_ATTEMPT job_id={job_id} lease_id={lease_id} "
                            f"job_epoch={job_epoch} status=failed"
                        )
                except Exception as post_e:
                    _log(f"[agent-v1] post result error: {post_e}")
                    _log(f"[agent-v1] original error: {e}")

        except Exception as e:
            _log(f"[agent-v1] loop error: {e}")
            time.sleep(ERROR_BACKOFF_SEC)

    _log("[agent-v1] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
