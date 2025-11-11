# app.py — heartbeat + simple task worker (sha256 + map_tokenize)

import os, time, socket, random, signal, json, hashlib
from typing import Optional, List, Dict, Any, Tuple
import requests

# ------------------ config ------------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://controller:8080")
AGENT_NAME     = os.getenv("AGENT_NAME", socket.gethostname())
HEARTBEAT_SEC  = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
TASK_INTERVAL  = int(os.getenv("TASK_INTERVAL", "0"))           # ms to sleep after each task
WAIT_MS        = int(os.getenv("TASK_WAIT_MS", "2000"))         # server long-poll wait
TIMEOUT_SEC    = float(os.getenv("HTTP_TIMEOUT_SEC", "6"))      # request timeout

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

# ------------------ http helpers ------------------

def heartbeat():
    try:
        r = session.get(f"{CONTROLLER_URL}/healthz", timeout=TIMEOUT_SEC)
        log(f"heartbeat -> {r.status_code}")
    except Exception as e:
        log(f"heartbeat failed: {e}")

def fetch_task() -> Optional[dict]:
    try:
        r = session.get(
            f"{CONTROLLER_URL}/task",
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
        session.post(f"{CONTROLLER_URL}/result", json=res, timeout=TIMEOUT_SEC)
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
