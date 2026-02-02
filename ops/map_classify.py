from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import register_op
from .wrapper import OpError  # structured errors -> wrapper produces envelope


OP_NAME = "map_classify"

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "assemblyai/distilbert-base-uncased-sst2")

# Thresholds (env configurable; safe defaults)
# If score >= POS_THRESH -> POSITIVE
# If score >= NEG_THRESH -> NEGATIVE
# Else -> NEUTRAL
_POS_THRESH = float(os.getenv("MAP_CLASSIFY_POS_THRESH", "0.60"))
_NEG_THRESH = float(os.getenv("MAP_CLASSIFY_NEG_THRESH", "0.60"))

_FORCE_CPU = os.getenv("MAP_CLASSIFY_FORCE_CPU", "0").strip().lower() in ("1", "true", "yes")

_model_lock = threading.Lock()
_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_model_device: str = "cpu"


def _select_device() -> str:
    if _FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_model() -> None:
    global _model, _tokenizer, _model_device
    if _model is not None and _tokenizer is not None:
        return

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return

        device = _select_device()
        print(f"[map_classify] Loading HF model '{HF_MODEL_NAME}' on {device}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        model.to(device)
        model.eval()

        _model = model
        _tokenizer = tokenizer
        _model_device = device


def _infer_label(score_pos: float, score_neg: float, pos_thresh: float, neg_thresh: float) -> str:
    if score_pos >= pos_thresh:
        return "POSITIVE"
    if score_neg >= neg_thresh:
        return "NEGATIVE"
    return "NEUTRAL"


def _get_pos_neg_scores(probs: torch.Tensor, id2label: Dict[int, str]) -> Tuple[float, float, List[float]]:
    """
    Returns (pos_score, neg_score, scores_list) regardless of model label naming.
    Assumes a 2-class classifier (SST2 style).
    """
    scores = probs.detach().cpu().tolist()

    label0 = str(id2label.get(0, "0")).upper()
    label1 = str(id2label.get(1, "1")).upper()

    s0 = float(probs[0].item()) if probs.numel() > 0 else 0.0
    s1 = float(probs[1].item()) if probs.numel() > 1 else 0.0

    if "POS" in label0 and "NEG" in label1:
        pos, neg = s0, s1
    elif "NEG" in label0 and "POS" in label1:
        neg, pos = s0, s1
    else:
        # Fallback: assume index 1 is positive (SST2 convention)
        neg, pos = s0, s1

    return pos, neg, scores


def _empty_raw_result(pos_thresh: float, neg_thresh: float) -> Dict[str, Any]:
    # RAW result only (wrapper adds ok/result/metrics)
    return {
        "label": "NEUTRAL",
        "score": 0.0,
        "scores": [],
        "device": _model_device,
        "thresholds": {"pos": float(pos_thresh), "neg": float(neg_thresh)},
        "model": HF_MODEL_NAME,
    }


def _require_dict(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise OpError("INVALID_ARGUMENT", "payload must be an object", retryable=False)
    return payload


def _classify_one_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scalar classify -> RAW result dict.
    """
    _init_model()
    assert _model is not None
    assert _tokenizer is not None

    text = str(payload.get("text", "")).strip()

    # Optional per-request override
    pos_thresh = float(payload.get("pos_thresh", _POS_THRESH))
    neg_thresh = float(payload.get("neg_thresh", _NEG_THRESH))
    max_length = int(payload.get("max_length", 256))

    if not text:
        return _empty_raw_result(pos_thresh=pos_thresh, neg_thresh=neg_thresh)

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(_model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pos_score, neg_score, scores = _get_pos_neg_scores(probs, _model.config.id2label)
    label = _infer_label(pos_score, neg_score, pos_thresh, neg_thresh)

    # score = confidence of chosen label (or best non-neutral)
    score = pos_score if label == "POSITIVE" else neg_score if label == "NEGATIVE" else max(pos_score, neg_score)

    return {
        "label": label,
        "score": float(score),
        "scores": scores,
        "device": _model_device,
        "thresholds": {"pos": float(pos_thresh), "neg": float(neg_thresh)},
        "model": HF_MODEL_NAME,
    }


def _classify_batch_raw(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    True batched inference (tokenize once, one forward pass) -> list of RAW results.
    """
    _init_model()
    assert _model is not None
    assert _tokenizer is not None

    texts: List[str] = []
    pos_threshes: List[float] = []
    neg_threshes: List[float] = []
    max_lengths: List[int] = []

    for it in items:
        texts.append(str(it.get("text", "")).strip())
        pos_threshes.append(float(it.get("pos_thresh", _POS_THRESH)))
        neg_threshes.append(float(it.get("neg_thresh", _NEG_THRESH)))
        max_lengths.append(int(it.get("max_length", 256)))

    nonempty_idx: List[int] = [i for i, t in enumerate(texts) if t]
    if not nonempty_idx:
        return [_empty_raw_result(pos_threshes[i], neg_threshes[i]) for i in range(len(items))]

    nonempty_texts = [texts[i] for i in nonempty_idx]
    max_length = max(max_lengths[i] for i in nonempty_idx)

    enc = _tokenizer(
        nonempty_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    enc = {k: v.to(_model_device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = _model(**enc)
        probs_b = torch.softmax(outputs.logits, dim=-1)  # [B,2]

    results: List[Dict[str, Any]] = [None] * len(items)  # type: ignore

    for i, t in enumerate(texts):
        if not t:
            results[i] = _empty_raw_result(pos_threshes[i], neg_threshes[i])

    for j, i in enumerate(nonempty_idx):
        probs = probs_b[j]
        pos_score, neg_score, scores = _get_pos_neg_scores(probs, _model.config.id2label)

        pos_thresh = pos_threshes[i]
        neg_thresh = neg_threshes[i]

        label = _infer_label(pos_score, neg_score, pos_thresh, neg_thresh)
        score = pos_score if label == "POSITIVE" else neg_score if label == "NEGATIVE" else max(pos_score, neg_score)

        results[i] = {
            "label": label,
            "score": float(score),
            "scores": scores,
            "device": _model_device,
            "thresholds": {"pos": float(pos_thresh), "neg": float(neg_thresh)},
            "model": HF_MODEL_NAME,
        }

    return results


@register_op(OP_NAME)
def map_classify_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAW-only op.

    Scalar payload:
      {"text":"..."} (+ optional pos_thresh/neg_thresh/max_length)
    Returns RAW dict.

    Batched payload:
      {"items":[ {"text":"..."}, ... ]}
    Returns {"items":[RAW dict, ...]}  (wrapper unwraps to list result)
    """
    payload = _require_dict(payload)

    if isinstance(payload.get("items"), list):
        raw_items = payload["items"]
        dict_items: List[Dict[str, Any]] = []
        for it in raw_items:
            dict_items.append(it if isinstance(it, dict) else {"text": str(it)})
        return {"items": _classify_batch_raw(dict_items)}

    return _classify_one_raw(payload)


# Wrapper fast-path: pass {"items":[...]} through in one call for true batching.
map_classify_op.OP_SPEC = {"accepts_batch": True, "batch_key": "items", "result_key": "items"}
