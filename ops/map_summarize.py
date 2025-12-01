import threading
from typing import Any, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from . import register_op  # same decorator used by map_classify


DEFAULT_MODEL_NAME = "t5-small"

# Global cache so each worker doesn't re-load models for every task
_MODEL_LOCK = threading.Lock()
_PIPELINES: Dict[str, Any] = {}  # key: f"{model_name}:{device}"


def _get_worker_safe_limits(worker_profile: Dict[str, Any]) -> Dict[str, int]:
    """
    Decide safe token limits based on worker_profile.
    Keeps weak CPUs from stalling and small GPUs from OOM'ing.
    """
    gpu_info = worker_profile.get("gpu", {}) or {}
    cpu_info = worker_profile.get("cpu", {}) or {}

    gpu_present = bool(gpu_info.get("gpu_present"))
    vram_gb = gpu_info.get("vram_gb")

    # Default conservative limits
    max_input_tokens = 512
    max_new_tokens = 96

    if gpu_present and vram_gb:
        # Heavier workloads for bigger VRAM
        if vram_gb >= 24:
            max_input_tokens = 1024
            max_new_tokens = 160
        elif vram_gb >= 12:
            max_input_tokens = 768
            max_new_tokens = 128
        else:
            max_input_tokens = 512
            max_new_tokens = 96
    else:
        # CPU workers: scale with core count
        total_cores = cpu_info.get("total_cores", 4)
        if total_cores >= 16:
            max_input_tokens = 512
            max_new_tokens = 96
        elif total_cores >= 8:
            max_input_tokens = 448
            max_new_tokens = 80
        else:
            max_input_tokens = 384
            max_new_tokens = 64

    return {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
    }


def _resolve_limits(params: Dict[str, Any], worker_profile: Dict[str, Any]) -> Dict[str, int]:
    """
    Combine user params with worker-safe bounds.
    Clamps user values so they can't exceed what the worker can safely handle.
    """
    safe = _get_worker_safe_limits(worker_profile)

    user_max_input = params.get("max_input_tokens")
    user_max_new = params.get("max_new_tokens")
    user_min_new = params.get("min_new_tokens")

    # Clamp user values to safe limits
    if user_max_input is not None:
        max_input_tokens = min(int(user_max_input), safe["max_input_tokens"])
    else:
        max_input_tokens = safe["max_input_tokens"]

    if user_max_new is not None:
        max_new_tokens = min(int(user_max_new), safe["max_new_tokens"])
    else:
        max_new_tokens = safe["max_new_tokens"]

    # Min new tokens: sensible default based on max_new_tokens
    if user_min_new is not None:
        min_new_tokens = max(4, min(int(user_min_new), max_new_tokens // 2))
    else:
        # default ~1/4 of max_new_tokens, but at least 8
        min_new_tokens = max(8, max_new_tokens // 4)

    return {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
    }


def _get_summarizer(model_name: str, device: str, worker_logger=None):
    """
    Lazy-init a summarization pipeline per (model_name, device).
    device is 'cpu' or 'cuda:0', 'cuda:1', etc.
    """
    key = f"{model_name}:{device}"

    if key in _PIPELINES:
        return _PIPELINES[key]

    with _MODEL_LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]

        if worker_logger:
            worker_logger.info(f"[map_summarize] loading model={model_name} on device={device}")

        # Hugging Face pipeline device index: -1 for CPU, 0/1/... for GPUs
        if device.startswith("cuda"):
            try:
                device_index = int(device.split(":")[1])
            except Exception:
                device_index = 0
        else:
            device_index = -1

        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device_index,
        )

        _PIPELINES[key] = summarizer
        return summarizer


@register_op("map_summarize")
def map_summarize(task: Dict[str, Any], worker_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task format (from controller):

    {
      "op": "map_summarize",
      "items": [
        {"id": "doc-1", "text": "..."},
        {"id": "doc-2", "text": "..."}
      ],
      "params": {
        "model_name": "t5-small",
        "max_input_tokens": 512,
        "max_new_tokens": 96,
        "min_new_tokens": 16,
        "batch_size": 4
      }
    }

    Returns:

    {
      "results": [
        {
          "id": "doc-1",
          "summary": "...",
          "raw": { ... optional debug ... }
        },
        ...
      ]
    }
    """

    items: List[Dict[str, Any]] = task.get("items") or []
    params: Dict[str, Any] = task.get("params") or {}

    if not items:
        return {"results": []}

    worker_profile = worker_ctx.get("worker_profile", {}) or {}
    device = worker_ctx.get("device", "cpu")
    logger = worker_ctx.get("logger")

    model_name = params.get("model_name") or DEFAULT_MODEL_NAME
    limits = _resolve_limits(params, worker_profile)

    max_input_tokens = limits["max_input_tokens"]
    max_new_tokens = limits["max_new_tokens"]
    min_new_tokens = limits["min_new_tokens"]

    batch_size = params.get("batch_size")
    if batch_size is None:
        # Slightly bigger batches for GPUs
        gpu_present = bool((worker_profile.get("gpu") or {}).get("gpu_present"))
        if gpu_present:
            batch_size = 8
        else:
            batch_size = 4
    else:
        batch_size = int(batch_size)

    summarizer = _get_summarizer(model_name, device, worker_logger=logger)

    texts = [it.get("text", "") for it in items]
    ids = [it.get("id") for it in items]

    results: List[Dict[str, Any]] = []

    # Process in batches so we don't overload any single worker
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        try:
            batch_outputs = summarizer(
                batch_texts,
                max_length=max_new_tokens,
                min_length=min_new_tokens,
                truncation=True,
            )
        except Exception as e:
            if logger:
                logger.exception(f"[map_summarize] summarization batch failed: {e}")
            # Attach an error entry for each item in this batch
            error_str = str(e)
            for bid in batch_ids:
                results.append(
                    {
                        "id": bid,
                        "summary": None,
                        "error": error_str,
                    }
                )
            continue

        for bid, out in zip(batch_ids, batch_outputs):
            summary_text = out.get("summary_text", "").strip()
            results.append(
                {
                    "id": bid,
                    "summary": summary_text,
                    "raw": {
                        "model_name": model_name,
                        "max_input_tokens": max_input_tokens,
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": min_new_tokens,
                        "batch_size": batch_size,
                    },
                }
            )

    return {"results": results}
