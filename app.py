#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------- config ----------------

CTRL = os.environ.get("CONTROLLER_URL", "http://localhost:8080").rstrip("/")
AGENT_NAME = os.environ.get("AGENT_NAME", "agent-1")
API_PREFIX = os.environ.get("API_PREFIX", "")  # keep empty for v1-only

HTTP_TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "10"))
LEASE_TIMEOUT_MS = int(os.environ.get("LEASE_TIMEOUT_MS", "2000"))
ERROR_BACKOFF_SEC = float(os.environ.get("ERROR_BACKOFF_SEC", "0.5"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "1"))

# Optional identity/metadata
BASE_LABELS: Dict[str, str] = {}
WORKER_PROFILE: Dict[str, Any] = {}

# ---------------- ops loading ----------------

OPS_MODULES = [
    # CPU-ish ops
    "ops.echo",
    "ops.map_tokenize",
    "ops.map_summarize",
    "ops.csv_shard",
    "ops.map_classify",
    "ops.risk_accumulate",
    "ops.sat_verify",
    "ops.fibonacci",
    "ops.prime_factor",
    "ops.subset_sum",
    "ops.map_image_gen",
    # GPU ops (may or may not exist; import guarded below)
    "ops.gpu_state",
    "ops.gpu_probe",
    "ops.gpu_vram_stats",
    "ops.gpu_warmup",
    "ops.gpu_model_load",
    "ops.gpu_embed_image",
    "ops.gpu_detect",
    "ops.gpu_overlay",
    "ops.gpu_image_decode",
    "ops.gpu_image_encode",
    "ops.gpu_video_encode",
]

OPS: Dict[str, Any] = {}
CAPABILITIES_LIST: List[str] = []

# ---------------- http helpers ----------------

_session = requests.Session()

_last_err: Dict[str, float] = {}


def _log_err_ratelimited(key: str, msg: str, every_sec: float = 1.0) -> None:
    now = time.time()
    prev = _last_err.get(key, 0.0)
    if now - prev >= every_sec:
        _last_err[key] = now
        print(msg, file=sys.stderr, flush=True)


def _post_json(path: str, payload: Any) -> Tuple[int, Any]:
    url = CTRL.rstrip("/") + path
    try:
        resp = _session.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct and resp.content:
            return resp.status_code, resp.json()
        return resp.status_code, resp.text
    except Exception as e:
        _log_err_ratelimited("post:" + path, f"[agent-v1] POST {url} failed: {e}")
        time.sleep(ERROR_BACKOFF_SEC)
        return 0, f"{type(e).__name__}: {e}"


def _get_json(path: str) -> Tuple[int, Any]:
    url = CTRL.rstrip("/") + path
    try:
        resp = _session.get(url, timeout=HTTP_TIMEOUT_SEC)
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct and resp.content:
            return resp.status_code, resp.json()
        return resp.status_code, resp.text
    except Exception as e:
        _log_err_ratelimited("get:" + path, f"[agent-v1] GET {url} failed: {e}")
        time.sleep(ERROR_BACKOFF_SEC)
        return 0, f"{type(e).__name__}: {e}"


# ---------------- metrics (minimal) ----------------


def _collect_metrics() -> Dict[str, Any]:
    return {"ts": time.time()}


# ---------------- v1 lease / result ----------------

# Cache epochs to avoid repeated GETs if the controller omits job_epoch in lease payload.
_epoch_cache: Dict[str, int] = {}


def _get_job_epoch(job_id: str) -> Optional[int]:
    if job_id in _epoch_cache:
        return _epoch_cache[job_id]

    # 1 retry (helps during controller restart / brief race)
    for _ in range(2):
        code, body = _get_json(f"/v1/jobs/{job_id}")
        if code == 200 and isinstance(body, dict):
            je = body.get("job_epoch")
            try:
                val = int(je)
                _epoch_cache[job_id] = val
                return val
            except Exception:
                return None
        time.sleep(0.05)
    return None


def _lease_once() -> Optional[Tuple[str, int, Dict[str, Any]]]:
    payload = {
        "agent": AGENT_NAME,
        # dict form is fine; controller normalizes list|dict
        "capabilities": {"ops": CAPABILITIES_LIST},
        "max_tasks": MAX_TASKS,
        "timeout_ms": LEASE_TIMEOUT_MS,
        "labels": BASE_LABELS,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics(),
    }
    code, body = _post_json("/v1/leases", payload)

    if code == 204:
        return None
    if code == 0:
        raise RuntimeError(f"lease failed: {body}")
    if code >= 400:
        raise RuntimeError(f"lease HTTP {code}: {body}")
    if not isinstance(body, dict):
        raise RuntimeError(f"lease body not dict: {body!r}")

    lease_id = str(body.get("lease_id") or body.get("id") or "")
    if not lease_id:
        raise RuntimeError(f"missing lease_id in: {body}")

    task: Optional[Dict[str, Any]] = None
    if isinstance(body.get("task"), dict):
        task = body["task"]
    elif isinstance(body.get("tasks"), list) and body["tasks"]:
        if isinstance(body["tasks"][0], dict):
            task = body["tasks"][0]

    if not task:
        return None

    # job_epoch MUST be present for /v1/results
    job_epoch = task.get("job_epoch") or body.get("job_epoch")

    if job_epoch is None:
        job_id = str(task.get("job_id") or task.get("id") or "")
        if job_id:
            job_epoch = _get_job_epoch(job_id)

    if job_epoch is None:
        # At this point we cannot safely post a result (controller will 422).
        raise RuntimeError(f"missing job_epoch in lease response: {body}")

    try:
        job_epoch_i = int(job_epoch)
    except Exception:
        raise RuntimeError(f"invalid job_epoch={job_epoch!r} in lease response: {body}")

    return lease_id, job_epoch_i, task


def _post_result(
    lease_id: str,
    job_id: str,
    job_epoch: int,
    status: str,
    result: Any = None,
    error: Any = None,
) -> None:
    payload: Dict[str, Any] = {
        "lease_id": lease_id,
        "job_id": job_id,
        "job_epoch": int(job_epoch),  # REQUIRED by controller
        "status": status,
        "result": result,
        "error": error,
    }
    code, body = _post_json("/v1/results", payload)
    if code == 0:
        raise RuntimeError(f"result failed: {body}")
    if code >= 400:
        raise RuntimeError(f"result HTTP {code}: {body}")


# ---------------- task execution ----------------


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


# ---------------- main loop ----------------

_stop = False


def _handle_sigterm(_signo: int, _frame: Any) -> None:
    global _stop
    _stop = True
    print("[agent-v1] stop signal received, shutting down...", flush=True)


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print(
        f"[agent-v1] starting v1-only controller='{CTRL}' api_prefix='{API_PREFIX}' "
        f"agent='{AGENT_NAME}' ops={CAPABILITIES_LIST}",
        flush=True,
    )

    while not _stop:
        try:
            leased = _lease_once()
            if not leased:
                time.sleep(0.05)
                continue

            lease_id, job_epoch, task = leased

            # IMPORTANT: if task is malformed, still close the lease with a FAILED result if we have job_id.
            try:
                job_id, op, payload = _extract_task_fields(task)
            except Exception as e:
                job_id = _task_job_id(task)
                err = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "trace": traceback.format_exc(),
                    "task": task,
                }
                if job_id:
                    try:
                        _post_result(lease_id, job_id, job_epoch, "failed", None, err)
                    except Exception as post_e:
                        _log_err_ratelimited(
                            "post_result_bad_task",
                            f"[agent-v1] post result error (bad task): {post_e}",
                        )
                else:
                    _log_err_ratelimited(
                        "bad_task_no_jobid",
                        f"[agent-v1] bad task with no job_id; cannot post result: {e}; task={task}",
                    )
                continue

            # Normal execution path
            try:
                out = _run_task(op, payload)
                _post_result(lease_id, job_id, job_epoch, "succeeded", out, None)
            except Exception as e:
                err = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "trace": traceback.format_exc(),
                }
                try:
                    _post_result(lease_id, job_id, job_epoch, "failed", None, err)
                except Exception as post_e:
                    _log_err_ratelimited(
                        "post_result",
                        f"[agent-v1] post result error: {post_e}",
                    )

        except Exception as e:
            _log_err_ratelimited("loop", f"[agent-v1] loop error: {e}")
            time.sleep(ERROR_BACKOFF_SEC)

    print("[agent-v1] stopped", flush=True)
    return 0


def _try_import_ops() -> None:
    global OPS, CAPABILITIES_LIST

    for modname in OPS_MODULES:
        try:
            print(f"[ops] importing {modname}", flush=True)
            mod = importlib.import_module(modname)
        except Exception:
            continue

        # Convention: module exports OPS = {"op_name": callable, ...}
        if hasattr(mod, "OPS") and isinstance(mod.OPS, dict):
            for k, v in mod.OPS.items():
                if callable(v):
                    OPS[k] = v

    CAPABILITIES_LIST = sorted(list(OPS.keys()))


if __name__ == "__main__":
    _try_import_ops()
    raise SystemExit(main())
