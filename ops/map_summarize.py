# ops/map_summarize.py
from typing import Any, Dict, List, Optional

from . import register_op


def _summarize_placeholder(text: str, max_len: int = 200) -> str:
    s = text.strip()
    if len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


@register_op("map_summarize")
def handle(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map-style summarization handler (placeholder truncation).

    Accepts either:
      - {"items": [ {id,text|document|body}, ... ]}
      - {"text": "..."}  (single convenience)

    Returns:
      {"ok": true, "items": [{"id": "...", "summary": "..."}, ...]}
    or:
      {"ok": false, "error": "..."}
    """
    if payload is None:
        payload = {}

    # Convenience: single text payload
    if "text" in payload and payload.get("text") is not None and "items" not in payload:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "map_summarize: payload.text must be a non-empty string"}
        return {"ok": True, "items": [{"id": payload.get("id"), "summary": _summarize_placeholder(text)}]}

    items = payload.get("items") or []
    if not isinstance(items, list):
        return {"ok": False, "error": "map_summarize: payload.items must be a list"}

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        item_id = None
        text = None

        if isinstance(item, dict):
            item_id = item.get("id")
            text = item.get("text") or item.get("document") or item.get("body")
        else:
            text = str(item)

        if not isinstance(text, str) or not text.strip():
            return {
                "ok": False,
                "error": f"map_summarize: item[{idx}] missing non-empty text in text/document/body",
            }

        results.append({"id": item_id, "summary": _summarize_placeholder(text)})

    return {"ok": True, "items": results}
