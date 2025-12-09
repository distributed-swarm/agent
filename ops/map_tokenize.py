# ops/map_tokenize.py

from typing import Dict, Any, List

from . import register_op  # adjust import if your registry name/path is slightly different


@register_op("map_tokenize")
def map_tokenize_op(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple tokenizer op.

    Expected payload shape (you can tweak this):
      {
        "text": "some string",
        "task_id": "optional",
        "meta": {...}  # optional
      }

    Returns:
      {
        "ok": True,
        "tokens": [...],
        "length": int
      }
    """

    text = payload.get("text", "")
    if not isinstance(text, str):
        return {
            "ok": False,
            "error": "payload.text must be a string"
        }

    # Naive whitespace tokenizer for now; swap in HF/BPE later if you want
    tokens: List[str] = text.split()

    return {
        "ok": True,
        "tokens": tokens,
        "length": len(tokens),
    }
