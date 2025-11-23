# app.py - agent worker: heartbeat + task loop (map_tokenize, map_classify with HF)

import os
import time
import socket
import random
import signal
import json
import hashlib
import shutil
import subprocess
import threading
from typing import Optional, List, Dict, Any

import requests
import torch  # make sure torch is explicitly imported in this module

from worker_sizing import build_worker_profile  # worker sizing for CPU/GPU

# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_WAIT_MS = int(os.getenv("TASK_WAIT_MS", "2000"))  # ms to sleep after each task poll
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))  # request timeout
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")  # e.g. "lab-basement,host=dl-rig-1"

_running = True


def _stop(*_args):
    global _running
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------------- labels / capabilities ----------------


def _parse_labels(raw: str) -> Dict[str, Any]:
    """
    Turn "a,b,c=1" into {"tags":["a","b"],"c":"1"} without being precious about it.
    """
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


try:
    WORKER_PROFILE = build_worker_profile()
except Exception as e:
    WORKER_PROFILE = {"error": f"{type(e).__name__}: {e}"}

BASE_LABELS: Dict[str, Any] = {
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
    return requests.post(
        _url(path), json=payload, timeout=HTTP_TIMEOUT_SEC
    )


def _get(path: str, params: Dict[str, Any]) -> requests.Response:
    return requests.get(
        _url(path), params=params, timeout=HTTP_TIMEOUT_SEC
    )


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


# ---------------- HuggingFace sentiment (lazy init) ----------------

_SENTIMENT_LOCK = threading.Lock()
_SENTIMENT_PIPELINE = None


def get_sentiment_pipeline():
    """
    Lazily create the HF sentiment pipeline once per process.

    Uses assemblyai/distilbert-base-uncased-sst2 for sentiment.
    """
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE

    from transformers import pipeline  # import here so we see any issues cleanly

    with _SENTIMENT_LOCK:
        if _SENTIMENT_PIPELINE is None:
            print(f"[agent] loading sentiment pipeline (torch={torch.__version__})...")
            try:
                _SENTIMENT_PIPELINE = pipeline(
                    "sentiment-analysis",
                    model="assemblyai/distilbert-base-uncased-sst2",
                )
                print("[agent] sentiment pipeline loaded")
            except Exception as e:
                import traceback

                print("[agent] ERROR loading sentiment pipeline:")
                traceback.print_exc()
                raise
    return _SENTIMENT_PIPELINE


# ---------------- op handlers ----------------


def handle_map_tokenize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    tokens = text.split()
    return {
        "text": text,
        "token_count": len(tokens),
        "tokens": tokens,
    }


def handle_map_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    if not text:
        return {"text": text, "sentiment": [], "warning": "empty text"}

    pipe = get_sentiment_pipeline()
    out = pipe(text)
    return {
        "text": text,
        "sentiment": out,
    }


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
            print(
                f"[agent {AGENT_NAME}] result post HTTP {resp.status_code}: {resp.text}"
            )
    except Exception as e:
        print(f"[agent {AGENT_NAME}] result post error: {e}")


# ---------------- task loop ----------------


def task_loop() -> None:
    while _running:
        try:
            resp = _get(
                "/task",
                {"agent": AGENT_NAME, "wait_ms": TASK_WAIT_MS},
            )
        except Exception as e:
            print(f"[agent {AGENT_NAME}] task poll error: {e}")
            time.sleep(3.0)
            continue

        if resp.status_code == 204:
            # no work
            continue

        if not resp.ok:
            print(
                f"[agent {AGENT_NAME}] task poll HTTP {resp.status_code}: {resp.text}"
            )
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
            import traceback

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
