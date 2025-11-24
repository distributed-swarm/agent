import os
import time
import socket
import signal
import threading
from typing import Optional, List, Dict, Any

import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from worker_sizing import build_worker_profile

# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "assemblyai/distilbert-base-uncased-sst2")

_running = True

# ---------------- signal handling ----------------


def _stop(*_args: Any) -> None:
    global _running
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------------- GPU detection ----------------


def detect_gpu() -> Dict[str, Any]:
    gpu_present = False
    gpu_count = 0
    vram_gb = 0.0
    devices: List[Dict[str, Any]] = []

    try:
        if torch.cuda.is_available():
            gpu_present = True
            gpu_count = torch.cuda.device_count()
            max_mem = 0
            for idx in range(gpu_count):
                props = torch.cuda.get_device_properties(idx)
                mem_bytes = int(props.total_memory)
                max_mem = max(max_mem, mem_bytes)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": mem_bytes,
                    }
                )
            if max_mem > 0:
                vram_gb = round(max_mem / (1024 ** 3), 2)
    except Exception as e:
        print(f"[{AGENT_NAME}] GPU detection failed: {e}", flush=True)

    info: Dict[str, Any] = {
        "gpu_present": gpu_present,
        "gpu_count": gpu_count,
        "vram_gb": vram_gb,
        "devices": devices,
    }
    print(f"[{AGENT_NAME}] GPU info: {info}", flush=True)
    return info


# ---------------- labels & capabilities ----------------


def _parse_labels(raw: str) -> Dict[str, str]:
    """
    Parse AGENT_LABELS like "role=gpu,zone=lab" into a dict.
    """
    labels: Dict[str, str] = {}
    if not raw:
        return labels
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        labels[k.strip()] = v.strip()
    return labels


GPU_INFO = detect_gpu()
WORKER_PROFILE = build_worker_profile()

BASE_LABELS: Dict[str, Any] = _parse_labels(AGENT_LABELS_RAW)
BASE_LABELS.update(
    {
        "gpu_present": str(GPU_INFO["gpu_present"]).lower(),
        "gpu_count": str(GPU_INFO["gpu_count"]),
        "vram_gb": str(GPU_INFO["vram_gb"]),
    }
)

BASE_CAPABILITIES: Dict[str, Any] = dict(WORKER_PROFILE)
BASE_CAPABILITIES.update(
    {
        "gpu_present": GPU_INFO["gpu_present"],
        "gpu_count": GPU_INFO["gpu_count"],
        "vram_gb": GPU_INFO["vram_gb"],
        "gpu_devices": GPU_INFO["devices"],
    }
)

# Choose default device for ops
_DEFAULT_DEVICE = "cuda" if GPU_INFO["gpu_present"] else "cpu"
print(f"[{AGENT_NAME}] Using device: {_DEFAULT_DEVICE}", flush=True)

# ---------------- HF model (lazy init) ----------------

_model_lock = threading.Lock()
_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_model_device: str = _DEFAULT_DEVICE


def _init_model() -> None:
    global _model, _tokenizer, _model_device
    if _model is not None and _tokenizer is not None:
        return
    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return
        model_name = HF_MODEL_NAME
        device = "cuda" if GPU_INFO["gpu_present"] and torch.cuda.is_available() else "cpu"
        print(f"[{AGENT_NAME}] Loading HF model '{model_name}' on {device}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _model = model
        _tokenizer = tokenizer
        _model_device = device


# ---------------- HTTP helpers ----------------


def _post_json(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None
    except Exception as e:
        print(f"[{AGENT_NAME}] POST {url} failed: {e}", flush=True)
        return None


def _get_json(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None
    except Exception as e:
        print(f"[{AGENT_NAME}] GET {url} failed: {e}", flush=True)
        return None


# ---------------- ops ----------------


def op_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple CPU op kept for compatibility with existing jobs.
    """
    text = str(payload.get("text", ""))
    tokens = text.split()
    return {
        "tokens": tokens,
        "count": len(tokens),
    }


def op_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sentiment classification via HF SST2 model.
    """
    _init_model()
    assert _model is not None
    assert _tokenizer is not None

    text = str(payload.get("text", ""))
    if not text:
        return {"label": "NEUTRAL", "scores": []}

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(_model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    label_idx = int(torch.argmax(probs).item())
    label = _model.config.id2label.get(label_idx, str(label_idx))
    scores = probs.cpu().tolist()

    return {
        "label": label,
        "scores": scores,
        "device": _model_device,
    }


def execute_op(op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if op == "map_classify":
        return op_map_classify(payload)
    if op == "map_tokenize":
        return op_map_tokenize(payload)
    # Unknown op: just echo payload
    return {"op": op, "payload": payload}


# ---------------- heartbeats ----------------


def _heartbeat_loop() -> None:
    """
    Periodically tell the controller we're alive and refresh labels/capabilities.
    """
    while _running:
        payload = {
            "agent": AGENT_NAME,
            "labels": BASE_LABELS,
            "capabilities": BASE_CAPABILITIES,
        }
        _post_json("/agents/heartbeat", payload)
        time.sleep(max(1, HEARTBEAT_SEC))


# ---------------- task loop ----------------


def _register_once() -> None:
    payload = {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": BASE_CAPABILITIES,
    }
    resp = _post_json("/agents/register", payload)
    print(f"[{AGENT_NAME}] register response: {resp}", flush=True)


def _run_task(job: Dict[str, Any]) -> None:
    job_id = job.get("id")
    op = job.get("op") or job.get("type") or "unknown"
    payload = job.get("payload") or {}

    start = time.time()
    ok = True
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    try:
        result = execute_op(op, payload)
    except Exception as e:
        ok = False
        error = f"{type(e).__name__}: {e}"
        print(f"[{AGENT_NAME}] Error running job {job_id}: {error}", flush=True)

    duration_ms = (time.time() - start) * 1000.0

    result_payload = {
        "id": job_id,
        "agent": AGENT_NAME,
        "ok": ok,
        "result": result,
        "error": error,
        "op": op,
        "duration_ms": duration_ms,
    }
    _post_json("/result", result_payload)


def _task_loop() -> None:
    while _running:
        job = _get_json("/task", {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS})
        if job is None:
            # no task or error; small sleep to avoid tight loop on errors
            time.sleep(0.2)
            continue
        _run_task(job)


# ---------------- main ----------------


def main() -> None:
    print(f"[{AGENT_NAME}] starting. controller={CONTROLLER_URL}", flush=True)
    print(f"[{AGENT_NAME}] labels={BASE_LABELS}", flush=True)
    print(f"[{AGENT_NAME}] capabilities={BASE_CAPABILITIES}", flush=True)

    _register_once()

    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()

    try:
        _task_loop()
    finally:
        print(f"[{AGENT_NAME}] shutting down.", flush=True)


if __name__ == "__main__":
    main()
