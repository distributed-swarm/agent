import os
import threading
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


OP_NAME = "map_classify"

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "assemblyai/distilbert-base-uncased-sst2")

_model_lock = threading.Lock()
_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_model_device: str = "cpu"


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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[map_classify] Loading HF model '{HF_MODEL_NAME}' on {device}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        model.to(device)
        model.eval()

        _model = model
        _tokenizer = tokenizer
        _model_device = device


def describe() -> Dict[str, Any]:
    """
    Basic descriptor for this op.

    Later we can enrich this with things like:
    - max_batch size
    - hints about needing GPU, etc.
    """
    return {
        "op": OP_NAME,
        "max_batch": 1,
        "labels": {
            "needs_gpu": True,
        },
    }


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sentiment classification via HF SST2 model.

    This is a standalone op implementation; app.py will call this
    via the ops registry.
    """
    _init_model()
    assert _model is not None
    assert _tokenizer is not None

    text = str(payload.get("text", ""))
    if not text:
        return {"label": "NEUTRAL", "scores": [], "device": _model_device}

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
