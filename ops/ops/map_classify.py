from typing import Any, Dict


OP_NAME = "map_classify"


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


def handle(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder handler for map_classify.

    For now this just proves the wiring shape.
    In a later step we'll move the real HF sentiment logic here.
    """
    payload = task.get("payload", {})
    text = payload.get("text", "")

    return {
        "ok": True,
        "op": OP_NAME,
        "input_text": text,
        "note": "map_classify handler placeholder (Base Agent v2 wiring step)",
    }
