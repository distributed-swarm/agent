# ops/map_summarize.py

def handle(payload):
    """
    Map-style summarization handler.

    Expected payload shape:
      {
        "items": [
          { "id": "demo-1", "text": "..." },
          { "id": "demo-2", "text": "..." },
          ...
        ],
        "params": { ... }   # optional
      }
    """
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise ValueError("Expected 'items' to be a list for map_summarize")

    results = []

    for item in items:
        text = None
        item_id = None

        if isinstance(item, dict):
            item_id = item.get("id")
            text = item.get("text") or item.get("document") or item.get("body")
        else:
            # Fallback: treat item itself as text
            text = str(item)

        if not isinstance(text, str) or not text.strip():
            raise ValueError("No text string provided in 'text'/'document'/'body'.")

        summary = text.strip()
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."

        results.append({
            "id": item_id,
            "summary": summary,
        })

    return {"items": results}
