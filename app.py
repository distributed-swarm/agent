import os
import time
import socket
import signal
import threading
from typing import Optional, List, Dict, Any

import requests
import torch
import warnings
from worker_sizing import build_worker_profile

# ---------------- warning cleanup ----------------

# Shut up torch.cuda UserWarnings (including the Tesla M10 spam)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda",
)

# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True


def _stop(*_args):
    global _running
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------------- helpers: labels & GPU detection ----------------


def _parse_labels(raw: str) -> Dict[str, Any]:
    labels: Dict[str, Any] = {}
    tags: List[str] = []

    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            labels[k.strip()] = v.strip()
        else:
            tags.append(part)

    if tags:
        labels["tags"] = tags
    return labels


def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPUs using /dev/nvidia* and torch.cuda.

    Returns:
        {
          "gpu_present": bool,
          "gpu_count": int,
          "vram_gb": float | None,
          "devices": [
             {"index": int, "name": str, "total_memory_bytes": int},
             ...
          ]
        }
    """
    info: Dict[str, Any] = {
        "gpu_present": False,
        "gpu_count": 0,
        "vram_gb": None,
        "devices": [],
    }

    # 1) /dev/nvidia* device nodes (works even if torch is CPU-only)
    try:
        import glob

        gpu_indices: List[int] = []
        for path in glob.glob("/dev/nvidia*"):
            base = os.path.basename(path)
            suffix = base.replace("nvidia", "", 1)
            if suffix.isdigit():
                gpu_indices.append(int(suffix))

        if gpu_indices:
            info["gpu_present"] = True
            info["gpu_count"] = len(gpu_indices)
    except Exception as e:
        info["dev_nodes_error"] = f"{type(e).__name__}: {e}"

    # 2) torch.cuda details (if usable)
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices: List[Dict[str, Any]] = []
            vram_list_gb: List[float] = []

            for idx in range(device_count):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
                vram_list_gb.append(props.total_memory / (1024 ** 3))

            if device_count > 0:
                info["gpu_present"] = True
                info["gpu_count"] = device_count
                info["devices"] = devices
                info["vram_gb"] = round(max(vram_list_gb), 2)

            # Explicitly set device 0 for our work
            torch.cuda.set_device(0)
            print(f"[{AGENT_NAME}] Using device: cuda:{torch.cuda.current_device()}")
    except Exception as e:
        info["torch_cuda_error"] = f"{type(e).__name__}: {e}"

    print(f"[{AGENT_NAME}] GPU info: {info}")
    return info


# ---------------- worker profile + labels ----------------

try:
    base_profile = build_worker_profile()
except Exception as e:
    base_profile = {"error": f"{type(e).__name__}: {e}"}

if not isinstance(base_profile, dict):
    base_profile = {}

GPU_INFO = detect_gpu()
base_profile["gpu"] = GPU_INFO
WORKER_PROFILE = base_profile

gpu_names: Optional[List[str]] = None
if GPU_INFO.get("devices"):
    gpu_names = [d.get("name") for d in GPU_INFO["devices"]]

BASE_LABELS: Dict[str, Any] = {
    "gpu_present": bool(GPU_INFO.get("gpu_present")),
    "gpu_count": int(GPU_INFO.get("gpu_count") or 0),
    "gpu_vram_gb": GPU_INFO.get("vram_gb"),
    "gpu_names": gpu_names,
    "worker_profile": WORKER_PROFILE,
}
BASE_LABELS.update(_parse_labels(AGENT_LABELS_RAW))

CAPABILITIES: Dict[str, Any] = {
    "ops": ["map_tokenize", "map_classify"],
}

# ---------------- HTTP helpers ----------------


def _url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return CONTROLLER_URL.rstrip("/") + path


def _post(path: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(_url(path), json=payload, timeout=HTTP_TIMEOUT_SEC)


def _get(path: str, params: Dict[str, Any]) -> requests.Response:
    return requests.get(_url(path), params=params, timeout=HTTP_TIMEOUT_SEC)


# ---------------- registration / heartbeat ----------------


def register_once() -> None:
    body = {
        "agent": AGENT_NAME,
        "labels": BASE_LABELS,
        "capabilities": CAPABILITIES,
    }
    try:
        resp = _post("/agents/register", body)
        resp.raise_for_status()
        print(f"[agent {AGENT_NAME}] registered -> {resp.json()}")
    except Exception as e:
        print(f"[agent {AGENT_NAME}] register failed: {e}")


def heartbeat_loop() -> None:
    while _running:
        body = {
            "agent": AGENT_NAME,
            "labels": BASE_LABELS,
            "capabilities": CAPABILITIES,
        }
        try:
            resp = _post("/agents/heartbeat", body)
            if resp.ok:
                data = resp.json()
                ts = data.get("time")
                print(f"[agent {AGENT_NAME}] heartbeat ok at {ts}")
            else:
                print(f"[agent {AGENT_NAME}] heartbeat HTTP {resp.status_code}")
        except Exception as e:
            print(f"[agent {AGENT_NAME}] heartbeat error: {e}")
        time.sleep(HEARTBEAT_SEC)


# ---------------- HF sentiment (lazy init) ----------------

_SENTIMENT_LOCK = threading.Lock()
_SENTIMENT_PIPELINE = None


def get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE

    from transformers import pipeline

    with _SENTIMENT_LOCK:
        if _SENTIMENT_PIPELINE is None:
            print(f"[agent {AGENT_NAME}] loading sentiment pipeline (torch={torch.__version__})...")
            _SENTIMENT_PIPELINE = pipeline(
                "sentiment-analysis",
                model="assemblyai/distilbert-base-uncased-sst2",
            )
            print(f"[agent {AGENT_NAME}] sentiment pipeline loaded")
    return _SENTIMENT_PIPELINE


# ---------------- op handlers ----------------


def handle_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    tokens = text.split()
    return {"text": text, "token_count": len(tokens), "tokens": tokens}


def handle_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    if not text:
        return {"text": text, "sentiment": [], "warning": "empty text"}

    pipe = get_sentiment_pipeline()
    out = pipe(text)
    return {"text": text, "sentiment": out}


# ---------------- result posting ----------------


def post_result(
    job_id: str,
    op: str,
    ok: bool,
    result: Optional[Dict[str, Any]],
    error: Optional[str],
    duration_ms: float,
) -> None:
    body = {
        "id": job_id,
        "agent": AGENT_NAME,
        "ok": ok,
        "result": result,
        "error": error,
        "op": op,
        "duration_ms": duration_ms,
    }
    try:
        resp = _post("/result", body)
        if not resp.ok:
            print(f"[agent {AGENT_NAME}] result post HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[agent {AGENT_NAME}] result post error: {e}")


# ---------------- task loop ----------------


def task_loop() -> None:
    import traceback

    while _running:
        try:
            resp = _get("/task", {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS})
        except Exception as e:
            print(f"[agent {AGENT_NAME}] task poll error: {e}")
            time.sleep(3.0)
            continue

        if resp.status_code == 204:
            continue

        if not resp.ok:
            print(f"[agent {AGENT_NAME}] task poll HTTP {resp.status_code}: {resp.text}")
            time.sleep(1.0)
            continue

        try:
            job = resp.json()
        except Exception as e:
            print(f"[agent {AGENT_NAME}] bad JSON from controller: {e}")
            time.sleep(1.0)
            continue

        job_id = job.get("id")
        op = job.get("op")
        payload = job.get("payload") or {}

        if not job_id or not op:
            print(f"[agent {AGENT_NAME}] malformed job: {job}")
            time.sleep(1.0)
            continue

        start = time.time()
        ok = True
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

        try:
            if op == "map_tokenize":
                result = handle_map_tokenize(payload)
            elif op == "map_classify":
                result = handle_map_classify(payload)
            else:
                ok = False
                error = f"unknown op: {op}"
        except Exception as e:
            print(f"[agent {AGENT_NAME}] ERROR while handling job {job_id} ({op}):")
            traceback.print_exc()
            ok = False
            error = f"{type(e).__name__}: {e}"

        duration_ms = (time.time() - start) * 1000.0
        post_result(job_id, op, ok, result, error, duration_ms)


# ---------------- main ----------------


def main() -> None:
    print(
        f"[agent {AGENT_NAME}] starting. controller={CONTROLLER_URL}, "
        f"labels={BASE_LABELS}, capabilities={CAPABILITIES}"
    )
    print(f"[agent {AGENT_NAME}] torch version: {torch.__version__}")
    register_once()

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    task_loop()
    print(f"[agent {AGENT_NAME}] shutting down.")


if __name__ == "__main__":
    main()
