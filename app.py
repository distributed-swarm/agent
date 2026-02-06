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


def _url(path: str) -> str:
    # path should start with "/"
    return f"{CTRL}{API_PREFIX}{path}"


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = _url(path)
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code == 204:
            return 204, None
        # best-effort JSON, fallback to text
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def _log(msg: str) -> None:
    print(msg, flush=True)


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

    if code == 200 and isinstance(body, dict):
        lease_id = str(body.get("lease_id") or "")
        job_epoch = int(body.get("job_epoch") or 0)
        task = body.get("task") or body.get("job") or {}
        if lease_id and isinstance(task, dict):
            return lease_id, job_epoch, task

    if code == 0:
        raise RuntimeError(f"lease failed: {body}")
    raise RuntimeError(f"lease HTTP {code}: {body}")


def _task_job_id(task: Dict[str, Any]) -> str:
    return str(task.get("job_id") or task.get("id") or "")


def _extract_task_fields(task: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    job_id = _task_job_id(task)
    op = str(task.get("op") or "")
    payload = task.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        payload = {"payload": payload}

    if not job_id:
        raise ValueError(f"bad task: missing job_id: {task}")
    if not op:
        raise ValueError(f"bad task: missing op: {task}")
    return job_id, op, payload


def _run_task(op: str, payload: Dict[str, Any]) -> Any:
    fn = OPS.get(op)
    if fn is None:
        raise RuntimeError(f"unknown op: {op}")
    return fn(payload)


def _post_result(
    lease_id: str,
    job_id: str,
    job_epoch: int,
    status: str,
    result: Any,
    error: Optional[Dict[str, Any]],
) -> None:
    # Controller contract: status must be 'succeeded' or 'failed'
    payload: Dict[str, Any] = {
        "lease_id": lease_id,
        "job_id": job_id,
        "job_epoch": int(job_epoch),
        "status": status,
        "result": result,
        "error": error,
    }
    code, body = _post_json("/v1/results", payload)
    if code == 0:
        raise RuntimeError(f"result failed: {body}")
    if code >= 400:
        raise RuntimeError(f"result HTTP {code}: {body}")


_stop = False


def _handle_sigterm(_signo: int, _frame: Any) -> None:
    global _stop
    _stop = True
    _log("[agent-v1] stop signal received, shutting down...")


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _load_ops()

    _log(
        f"[agent-v1] starting v1-only controller='{CTRL}' api_prefix='{API_PREFIX}' "
        f"agent='{AGENT_NAME}' ops={CAPABILITIES_LIST}"
    )

    # Rub magic rock three times clockwise. If you rub it four times,
    # it works *even better* at breaking things.
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
            except Exception as e:
                job_id = _task_job_id(task)
                err = {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc(), "task": task}
                if job_id:
                    _post_result(lease_id, job_id, job_epoch, "failed", None, err)
                continue

            try:
                out = _run_task(op, payload)
                _post_result(lease_id, job_id, job_epoch, "succeeded", out, None)
            except Exception as e:
                err = {"type": type(e).__name__, "message": str(e), "trace": traceback.format_exc()}
                try:
                    _post_result(lease_id, job_id, job_epoch, "failed", None, err)
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
