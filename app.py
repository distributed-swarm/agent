# app.py — heartbeat + simple task worker (sha256 + map_tokenize) + GPU/labels/worker advertise

import os, time, socket, random, signal, json, hashlib, shutil, subprocess
from typing import Optional, List, Dict, Any, Tuple
import requests

from worker_sizing import build_worker_profile  # NEW: worker sizing for CPU/GPU

# ------------------ config ------------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "0"))           # ms to sleep after each task
WAIT_MS        = int(os.getenv("TASK_WAIT_MS", "2000"))         # server long-poll wait
TIMEOUT_SEC    = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))      # request timeout
AGENT_LABELS   = os.getenv("AGENT_LABELS", "")                  # e.g. "lab=basement,host=dell-rtx3060,owner=jack"

# ------------------ lifecycle ------------------

_running = True
def _stop(*_):
    global _running
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

session = requests.Session()
session.headers.update({"X-Agent": AGENT_NAME})
session.headers.update({"X-Tasks": os.getenv("TASKS", "")})

def log(msg: str) -> None:
    print(f"[agent:{AGENT_NAME}] {msg}", flush=True)

# ------------------ labels + capability helpers ------------------

def _parse_labels(raw: str) -> Dict[str, str]:
    """
    Accepts comma-separated labels:
      "lab=basement,host=dell-rtx3060,owner=jack" -> {"lab":"basement","host":"dell-rtx3060","owner":"jack"}
    Whitespaces are trimmed. Bare words become {"tag":"word"} entries with incremental index.
    """
    out: Dict[str, str] = {}
    if not raw:
        return out
    idx = 0
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
        else:
            out[f"tag{idx}"] = p
            idx += 1
    return out

_LABELS = _parse_labels(AGENT_LABELS)

# Optional psutil for CPU/RAM if present
try:
    import psutil
except Exception:
    psutil = None

def _cpu_mem_caps() -> Dict[str, Any]:
    cores = os.cpu_count() or 1
    cpu_load = None
    ram_total_mb = None
    ram_used_mb = None
    if psutil:
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_total_mb = int(vm.total / (1024 * 1024))
            ram_used_mb = int((vm.total - vm.available) / (1024 * 1024))
        except Exception:
            pass
    return {
        "cpu": {"cores": cores, "load_pct": cpu_load},
        "memory": {"ram_total_mb": ram_total_mb, "ram_used_mb": ram_used_mb},
    }

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def _probe_gpu_with_nvidia_smi() -> Optional[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return None
    # Query first GPU line; if multiple, we summarize basics and aggregate VRAM
    fmt = "--query-gpu=name,driver_version,memory.total,compute_cap,utilization.gpu --format=csv,noheader,nounits"
    code, out, err = _run(["nvidia-smi", fmt])
    if code != 0 or not out:
        return None
    try:
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        names = []
        total_vram_mb = 0
        util_samples = []
        sm = None
        driver = None
        for ln in lines:
            # name,driver_version,memory.total,compute_cap,utilization.gpu
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 5:
                continue
            name, driver_version, mem_total_mb, compute_cap, util = parts[:5]
            names.append(name)
            driver = driver_version
            try:
                total_vram_mb += int(mem_total_mb)
            except Exception:
                pass
            sm = compute_cap
            try:
                util_samples.append(int(util))
            except Exception:
                pass
        gpu_name = names[0] if names else "NVIDIA GPU"
        avg_util = sum(util_samples) // max(1, len(util_samples)) if util_samples else None
        return {
            "present": True,
            "name": gpu_name if len(names) == 1 else f"{gpu_name} (+{len(names)-1})",
            "count": len(names),
            "driver_version": driver,
            "cuda_version": None,  # filled by torch probe if available
            "sm": sm,
            "vram_mb": total_vram_mb or None,
            "util_pct": avg_util,
            "fp16": None,
            "bf16": None,
        }
    except Exception:
        return None

def _probe_gpu_with_torch(gpu_caps: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        import torch
    except Exception:
        return gpu_caps
    try:
        has = torch.cuda.is_available()
        if not has:
            return gpu_caps or {"present": False}
        # ensure structure exists
        if gpu_caps is None:
            gpu_caps = {"present": True}
        gpu_caps["cuda_version"] = getattr(torch.version, "cuda", None)
        # precision hints
        try:
            major, minor = torch.cuda.get_device_capability(0)
            gpu_caps["sm"] = f"{major}.{minor}"
        except Exception:
            pass
        # fp16/bf16 support (heuristic)
        gpu_caps["fp16"] = True
        try:
            # bf16 widely supported on Ampere+; torch has dtype check
            x = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
            gpu_caps["bf16"] = bool(x.dtype == torch.bfloat16)
        except Exception:
            gpu_caps["bf16"] = False
        # VRAM (device 0 only; if multi-GPU, nvidia-smi sum already captured)
        try:
            props = torch.cuda.get_device_properties(0)
            vram_mb = int(props.total_memory / (1024 * 1024))
            gpu_caps["vram_mb"] = gpu_caps.get("vram_mb") or vram_mb
            gpu_caps["name"] = gpu_caps.get("name") or props.name
            gpu_caps["count"] = gpu_caps.get("count") or torch.cuda.device_count()
        except Exception:
            pass
        return gpu_caps
    except Exception:
        return gpu_caps

_last_caps: Optional[Dict[str, Any]] = None
_last_caps_ts: float = 0.0

def _collect_capabilities(force: bool = False) -> Dict[str, Any]:
    """
    Collect hardware + labels + worker sizing into one capabilities snapshot.
    """
    global _last_caps, _last_caps_ts
    now = time.time()
    if not force and _last_caps and (now - _last_caps_ts) < max(10, HEARTBEAT_SEC):
        return _last_caps

    gpu_caps = _probe_gpu_with_nvidia_smi()
    gpu_caps = _probe_gpu_with_torch(gpu_caps)

    if gpu_caps is None:
        gpu_caps = {"present": False}

    cpu_mem = _cpu_mem_caps()
    total_cores = os.cpu_count() or 1
    vram_mb = gpu_caps.get("vram_mb")
    vram_gb = int(vram_mb / 1024) if isinstance(vram_mb, int) and vram_mb > 0 else None

    worker_profile = build_worker_profile(
        total_cores=total_cores,
        gpu_present=bool(gpu_caps.get("present")),
        gpu_count=int(gpu_caps.get("count") or 0),
        vram_gb=vram_gb,
    )

    caps: Dict[str, Any] = {
        "gpu": gpu_caps,
        **cpu_mem,
        "labels": _LABELS,
        "agent_name": AGENT_NAME,
        "hostname": socket.gethostname(),
        "api_version": "v1",
        "workers": worker_profile,
    }
    _last_caps, _last_caps_ts = caps, now
    return caps

# ------------------ http helpers ------------------

def heartbeat():
    # Old health check (kept for backward compat)
    try:
        r = session.get(f"{CONTROLLER_URL}/healthz", timeout=TIMEOUT_SEC)
        log(f"heartbeat -> {r.status_code}")
    except Exception as e:
        log(f"heartbeat failed: {e}")

    # New capabilities heartbeat (best-effort; ignore if controller not updated yet)
    caps = _collect_capabilities()
    payload = {
        "agent": AGENT_NAME,
        "labels": caps.get("labels", {}),
        "capabilities": {
            "gpu": caps.get("gpu", {}),
            "cpu": caps.get("cpu", {}),
            "memory": caps.get("memory", {}),
            "workers": caps.get("workers", {}),
        },
        "timestamp": int(time.time()),
    }
    # Try /api/agents/heartbeat, fall back to /api/agents/register on first run
    try:
        session.post(f"{CONTROLLER_URL}/api/agents/heartbeat", json=payload, timeout=TIMEOUT_SEC)
    except Exception:
        # one-time soft register; ignore failure
        try:
            session.post(f"{CONTROLLER_URL}/api/agents/register", json=payload, timeout=TIMEOUT_SEC)
        except Exception:
            pass

def fetch_task() -> Optional[dict]:
    try:
        r = session.get(
            f"{CONTROLLER_URL}/api/task",
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

def post_result(res: dict):
    try:
        session.post(f"{CONTROLLER_URL}/api/result", json=res, timeout=TIMEOUT_SEC)
        log(f"result -> ok={res['ok']} ms={res['duration_ms']}")
    except Exception as e:
        log(f"result post error: {e}")

# ------------------ tokenizer utilities ------------------

# Try fast tokenization with tiktoken; fall back to whitespace tokens.
try:
    import tiktoken
    _enc_cache: Dict[str, Any] = {}
    def _get_enc(name: str):
        if name not in _enc_cache:
            _enc_cache[name] = tiktoken.get_encoding(name)
        return _enc_cache[name]
    def encode(text: str, enc_name: str) -> List[int]:
        return _get_enc(enc_name).encode(text)
    def decode(tokens: List[int], enc_name: str) -> str:
        return _get_enc(enc_name).decode(tokens)
    DEFAULT_ENCODING = "cl100k_base"
except Exception:
    tiktoken = None
    def encode(text: str, enc_name: str) -> List[str]:
        # crude but predictable fallback
        return text.split()
    def decode(tokens: List[str], enc_name: str) -> str:
        return " ".join(tokens)
    DEFAULT_ENCODING = "fallback_ws"

# ------------------ ops ------------------

def handle_sha256(payload: Any) -> dict:
    if isinstance(payload, (dict, list)):
        try:
            payload = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload = str(payload)
    elif not isinstance(payload, str):
        payload = str(payload)
    return {
        "ok": True,
        "output": hashlib.sha256(payload.encode("utf-8")).hexdigest()
    }

def _read_text_from_payload(payload: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Accepts any of:
      - {"text": "..."}                                  -> returns that text
      - {"input_path": "/path/file.txt"}                 -> reads entire file
      - {"source": {"file": "/path/file.jsonl"}}         -> reads JSONL with {"text": "..."} per line
      - plain string -> treated as text
    Returns (text, meta)
    """
    meta: Dict[str, Any] = {}
    if isinstance(payload, str):
        return payload, meta

    if not isinstance(payload, dict):
        return str(payload), meta

    # Case 1: explicit text
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"], meta

    # Case 2: single file path
    path = payload.get("input_path") or ((payload.get("source") or {}).get("file"))
    if path:
        meta["source"] = path
        if path.lower().endswith(".jsonl"):
            texts: List[str] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("text")
                        if isinstance(t, str):
                            texts.append(t)
                    except Exception:
                        # skip malformed lines
                        continue
            return "\n".join(texts), meta
        # otherwise read whole file
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), meta

    # Last resort: stringify
    return json.dumps(payload, ensure_ascii=False), meta

def handle_map_tokenize(payload: Any) -> dict:
    """
    Params can include:
      - text OR input_path OR source.file (JSONL with {"text": ...})
      - chunk_size (default 4096), overlap (default 128), encoding (default cl100k_base)
    """
    # payload may contain both the text/file and the params; allow flat dict
    if isinstance(payload, dict):
        chunk_size = int(payload.get("chunk_size", 4096))
        overlap    = int(payload.get("overlap", 128))
        enc_name   = payload.get("encoding", DEFAULT_ENCODING)
    else:
        chunk_size, overlap, enc_name = 4096, 128, DEFAULT_ENCODING

    text, meta = _read_text_from_payload(payload)
    if not isinstance(text, str) or not text:
        return {"ok": False, "error": "no text to tokenize"}

    # safety: overlap must be less than chunk_size
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    toks = encode(text, enc_name)
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(toks):
        window = toks[i:i+chunk_size]
        chunks.append({
            "idx": len(chunks),
            "start_token": i,
            "end_token": i + len(window),
            "text": decode(window, enc_name)
        })
        if len(window) < chunk_size:
            break
        i += max(1, chunk_size - overlap)

    return {
        "ok": True,
        "output": {
            "meta": meta,
            "encoding": enc_name,
            "total_tokens": len(toks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            # keep payload small but useful
            "samples": [
                {"idx": c["idx"], "preview": c["text"][:160]}
                for c in chunks[:3]
            ]
        }
    }

# ------------------ task runner ------------------

def run_task(task: dict) -> dict:
    tid     = task.get("id", "tsk-unknown")
    op      = task.get("op") or task.get("type")   # tolerate either field
    payload = task.get("payload") or task.get("params") or ""

    log(f"received task: {op}")
    start = time.time()
    ok, output, err = False, None, None

    try:
        if op == "sha256":
            r = handle_sha256(payload)
        elif op == "map_tokenize":
            r = handle_map_tokenize(payload)
        else:
            r = {"ok": False, "error": f"unsupported op: {op}"}

        ok, output, err = r.get("ok", False), r.get("output"), r.get("error")
    except Exception as e:
        ok, output, err = False, None, f"{type(e).__name__}: {e}"

    duration_ms = int((time.time() - start) * 1000)

    res = {
        "id": tid,
        "agent": AGENT_NAME,
        "ok": ok,
        "output": output if ok else None,
        "duration_ms": duration_ms,
        "error": None if ok else err,
        "op": op,
        "trace": "",
        "_raw": {"id": tid, "agent": AGENT_NAME, "ok": ok, "duration_ms": duration_ms, "error": err},
    }

    if ok:
        log("completed task -> ok=True")
    else:
        log(f"completed task -> ok=False error={err}")

    return res

# ------------------ main loop ------------------

def main():
    log("starting")
    # send an initial capabilities register (best-effort)
    try:
        session.post(
            f"{CONTROLLER_URL}/api/agents/register",
            json={
                "agent": AGENT_NAME,
                "labels": _LABELS,
                "capabilities": _collect_capabilities(force=True),
                "timestamp": int(time.time()),
            },
            timeout=TIMEOUT_SEC,
        )
    except Exception:
        pass

    last_hb = 0.0

    while _running:
        now = time.time()
        if now - last_hb >= HEARTBEAT_SEC:
            heartbeat()
            last_hb = now

        task = fetch_task()
        if task is None:
            time.sleep(random.uniform(0.02, 0.06))  # tiny jitter
            continue

        res = run_task(task)
        post_result(res)

        if TASK_INTERVAL > 0:
            time.sleep(TASK_INTERVAL / 1000.0)

    log("stopping")

if __name__ == "__main__":
    main()
