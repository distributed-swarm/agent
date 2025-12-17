# ops/map_classify.py
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from . import register_op


OP_NAME = "map_classify"

# Model name (same default you used)
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "assemblyai/distilbert-base-uncased-sst2")

# Thresholds (env configurable; safe defaults)
# If score >= POS_THRESH -> POSITIVE
# If score >= NEG_THRESH -> NEGATIVE
# Else -> NEUTRAL
_POS_THRESH = float(os.getenv("MAP_CLASSIFY_POS_THRESH", "0.60"))
_NEG_THRESH = float(os.getenv("MAP_CLASSIFY_NEG_THRESH", "0.60"))

# Optional: if you want to force CPU even when CUDA exists
_FORCE_CPU = os.getenv("MAP_CLASSIFY_FORCE_CPU", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

_model_lock = threading.Lock()
_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_model_device: str = "cpu"


def _select_device() -> str:
    if _FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_model() -> None:
    """
    Lazily load the HF sentiment model on CPU or GPU.
    """
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
    # Default mapping for SST2-style (0=NEGATIVE, 1=POSITIVE), but be defensive.
    label0 = str(id2label.get(0, "0")).upper()
    label1 = str(id2label.get(1, "1")).upper()

    # probs is shape [num_labels]
    s0 = float(probs[0].item()) if probs.numel() > 0 else 0.0
    s1 = float(probs[1].item()) if probs.numel() > 1 else 0.0

    # Try to infer which is POS/NEG from id2label
    if "POS" in label0 and "NEG" in label1:
        pos, neg = s0, s1
    elif "NEG" in label0 and "POS" in label1:
        neg, pos = s0, s1
    else:
        # Fallback: assume index 1 is positive (SST2 convention)
        neg, pos = s0, s1

    return pos, neg, scores


@register_op(OP_NAME)
def map_classify_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sentiment classification via HF SST2 model.

    Payload supported:
      - {"text": "..."}
      - Optional overrides:
          {"text": "...", "pos_thresh": 0.65, "neg_thresh": 0.65, "max_length": 256}
    """
    _init_model()
    assert _model is not None
    assert _tokenizer is not None

    text = str(payload.get("text", "")).strip()
    if not text:
        return {
            "ok": True,
            "label": "NEUTRAL",
            "score": 0.0,
            "scores": [],
            "device": _model_device,
            "thresholds": {"pos": _POS_THRESH, "neg": _NEG_THRESH},
            "model": HF_MODEL_NAME,
        }

    # Allow per-request override (optional)
    pos_thresh = float(payload.get("pos_thresh", _POS_THRESH))
    neg_thresh = float(payload.get("neg_thresh", _NEG_THRESH))
    max_length = int(payload.get("max_length", 256))

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
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pos_score, neg_score, scores = _get_pos_neg_scores(probs, _model.config.id2label)

    label = _infer_label(pos_score=pos_score, score_neg=neg_score, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
    # "score" = confidence of chosen label
    score = pos_score if label == "POSITIVE" else neg_score if label == "NEGATIVE" else max(pos_score, neg_score)

    return {
        "ok": True,
        "label": label,
        "score": float(score),
        "scores": scores,  # raw softmax probs (model order)
        "device": _model_device,
        "thresholds": {"pos": float(pos_thresh), "neg": float(neg_thresh)},
        "model": HF_MODEL_NAME,
    }
