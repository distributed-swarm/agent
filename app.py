import os
import time
import socket
import signal
import threading
from typing import Optional, List, Dict, Any

import requests
import torch
from transformers import AutoTokenizer
from worker_sizing import build_worker_profile
from ops import get_op, list_ops

# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))

AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")
TASKS_RAW = os.getenv("TASKS", "")

HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "assemblyai/distilbert-base-uncased-sst2",
)

# ---------------- globals ----------------

_running = True
_session = requests.Session()

_worker_profile: Dict[str, Any] = {}
_base_labels: Dict[str, Any] = {}

# tokenizer for legacy map_tokenize
_tokenizer_lock = threading.Lock()
_tokenizer: Optional[AutoTokenizer] = None


# ---------------- signal handling ----------------


def _stop(*_args: Any) -> None:
    global _running
    _running = False
    print(f"[{AGENT_NAME}] received stop signal, shutting down loop soon.", flush=True)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# ---------------- helpers: labels & tasks ----------------


def _parse_agent_labels(raw: str) -> Dict[str, Any]:
    """
    Parse AGENT_LABELS env into a dict.

    Examples:
      "gpu,zone=lab,disk=ssd"
      "zone=home ops=map_tokenize,map_classify"
    """
    labels: Dict[str, Any] = {}
    if not raw:
        return labels

    for part in raw.replace(",", " ").split():
        if "=" in part:
            k, v = part.split("=", 1)
            labels[k.strip()] = v.strip()
        else:
            labels[part.strip()] = True
    return labels


def _parse_tasks(raw: str) -> List[str]:
    """
    Parse TASKS env into a list, e.g. "map_tokenize,map_classify".
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


# ---------------- helpers: GPU + worker profile ----------------


def _detect_gpu() -> Dict[str, Any]:
    gpu_present = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_present else 0
    vram_gb = 0.0

    if gpu_present:
        try:
            props = torch.cuda.get_device_properties(0)
            vram_bytes = getattr(props, "total_memory", 0)
            vram_gb = round(float(vram_bytes) / (1024 ** 3), 2)
        except Exception as e:
            print(f"[{AGENT_NAME}] GPU detection error: {e}", flush=True)

    return {
        "gpu_present": gpu_present,
        "gpu_count": gpu_count,
        "vram_gb": vram_gb,
    }


def _build_worker_profile() -> Dict[str, Any]:
    base = build_worker_profile()
    gpu_info = _detect_gpu()
    base.setdefault("gpu", {}).update(gpu_info)
    return base


def _build_base_labels() -> Dict[str, Any]:
    labels = _parse_agent_labels(AGENT_LABELS_RAW)

    gpu_info = _worker_profile.get("gpu", {})
    labels.setdefault("gpu_present", gpu_info.get("gpu_present", False))
    labels.setdefault("gpu_count", gpu_info.get("gpu_count", 0))
    labels.setdefault("vram_gb", gpu_info.get("vram_gb", 0.0))

    tasks = _parse_tasks(TASKS_RAW)
    if not tasks:
        # If TASKS isn't set, default to whatever ops the registry knows
        tasks = list_ops()

    labels["agent_name"] = AGENT_NAME
    labels["tasks"] = tasks
    labels["ops"] = list_ops()
    return labels


# ---------------- HTTP helpers ----------------


def _post(path: str, json_body: Dict[str, Any]) -> Optional[requests.Response]:
    url = f"{CONTROLLER_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        resp = _session.post(url, json=json_body, timeout=HTTP_TIMEOUT_SEC)
        return resp
    except Exception as e:
        print(f"[{AGENT_NAME}] HTTP POST {url} failed: {e}", flush=True)
        return None


# ---------------- heartbeat ----------------


def _heartbeat_payload() -> Dict[str, Any]:
    return {
        "name": AGENT_NAME,
        "labels": _base_labels,
        "worker": _worker_profile,
    }


def _heartbeat_loop() -> None:
    """
    Periodically tell the controller that this agent exists and is healthy.
    """
    while _running:
        payload = _heartbeat_payload()
        resp = _post("/heartbeat", payload)
        if resp is not None and resp.status_code != 200:
            try:
                text = resp.text
            except Exception:
                text = "<no body>"
            print(
                f"[{AGENT_NAME}] heartbeat status={resp.status_code} body={text}",
                flush=True,
            )
        time.sleep(HEARTBEAT_SEC)


# ---------------- leasing + completion ----------------


def _lease_task() -> Optional[Dict[str, Any]]:
    """
    Ask controller for a task.

    Expected response JSON shape:
      {
        "id": "...",
        "op": "map_classify",
        "payload": {...}
      }

    204 or empty JSON means "nothing to do right now".
    """
    req_body = {
        "agent": AGENT_NAME,
        "wait_ms": TASK_WAIT_MS,
    }
    resp = _post("/lease", req_body)
    if resp is None:
        return None

    if resp.status_code == 204:
        return None

    if not resp.content:
        return None

    try:
        data = resp.json()
    except Exception as e:
        print(f"[{AGENT_NAME}] bad JSON from /lease: {e}", flush=True)
        return None

    if not isinstance(data, dict):
        return None

    if "op" not in data:
        return None

    return data


def _complete_task(task: Dict[str, Any], ok: bool, result: Dict[str, Any]) -> None:
    """
    Report task completion back to the controller.
    """
    payload = {
        "id": task.get("id"),
        "agent": AGENT_NAME,
        "ok": ok,
        "result": result,
    }
    resp = _post("/complete", payload)
    if resp is not None and resp.status_code != 200:
        try:
            text = resp.text
        except Exception:
            text = "<no body>"
        print(
            f"[{AGENT_NAME}] complete status={resp.status_code} body={text}",
            flush=True,
        )


# ---------------- legacy built-in ops ----------------


def _init_tokenizer() -> None:
    global _tokenizer
    if _tokenizer is not None:
        return
    with _tokenizer_lock:
        if _tokenizer is not None:
            return
        print(f"[{AGENT_NAME}] loading HF tokenizer '{HF_MODEL_NAME}'", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)


def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy HF-based tokenizer op.

    Keeps old map_tokenize behavior alive while we migrate to ops/.
    """
    _init_tokenizer()
    assert _tokenizer is not None

    text = str(payload.get("text", ""))
    if not text:
        return {"tokens": [], "ids": []}

    enc = _tokenizer(
        text,
        add_special_tokens=False,
    )
    input_ids = enc.get("input_ids", [])
    tokens = _tokenizer.convert_ids_to_tokens(input_ids)

    return {
        "tokens": tokens,
        "ids": input_ids,
    }


# ---------------- op dispatch ----------------


def execute_op(op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a job to the appropriate handler.

    Priority:
    1. ops registry (Base Agent v2 plugins)
    2. Legacy built-in handlers (v1 behavior)
    """
    # First try plugin registry
    handler = get_op(op)
    if handler is not None:
        return handler(payload)

    # Fallback to legacy built-ins
    if op == "map_tokenize":
        return op_map_tokenize(payload)

    # Unknown op: just echo payload
    return {"op": op, "payload": payload}


# ---------------- main task loop ----------------


def _task_loop() -> None:
    print(f"[{AGENT_NAME}] starting task loop", flush=True)
    while _running:
        task = _lease_task()
        if task is None:
            # Nothing right now; let the controller breathe
            time.sleep(TASK_WAIT_MS / 1000.0)
            continue

        op = task.get("op")
        payload = task.get("payload", {}) or {}

        start = time.time()
        ok = True
        result: Dict[str, Any]

        try:
            result = execute_op(op, payload)
        except Exception as e:
            ok = False
            result = {
                "error": str(e),
                "op": op,
            }
            print(f"[{AGENT_NAME}] error executing op={op}: {e}", flush=True)

        elapsed_ms = int((time.time() - start) * 1000)
        result.setdefault("elapsed_ms", elapsed_ms)

        _complete_task(task, ok, result)


# ---------------- main entrypoint ----------------


def main() -> None:
    global _worker_profile, _base_labels

    print(f"[{AGENT_NAME}] starting agent v2", flush=True)

    _worker_profile = _build_worker_profile()
    _base_labels = _build_base_labels()

    print(f"[{AGENT_NAME}] worker profile: {_worker_profile}", flush=True)
    print(f"[{AGENT_NAME}] base labels: {_base_labels}", flush=True)

    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()

    try:
        _task_loop()
    finally:
        print(f"[{AGENT_NAME}] shutting down.", flush=True)


if __name__ == "__main__":
    main()
