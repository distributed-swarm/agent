# agent/ops/csv_shard.py

import csv
from typing import Any, Dict, List

from . import register_op


def _read_csv_shard(source_uri: str, start_row: int, shard_size: int) -> List[Dict[str, Any]]:
    """
    Read a slice of rows from a CSV after the header.
    start_row = 0 means first data row.
    """
    rows: List[Dict[str, Any]] = []
    stop_row = start_row + shard_size

    with open(source_uri, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_row:
                continue
            if idx >= stop_row:
                break
            rows.append(row)

    return rows


@register_op("read_csv_shard")
def op_read_csv_shard(task_or_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic CSV shard op.

    Accepts either:
      A) payload dict directly (agent current behavior)
         {
           "dataset_id": "...",
           "source_uri": "...",
           "start_row": 0,
           "shard_size": 100,
           "mode": "rows" | "count"
         }

      B) full task dict containing "payload"
         { "payload": { ... } }
    """
    # If we're handed a full task, extract payload; otherwise treat the input as payload.
    payload = task_or_payload.get("payload") if "payload" in task_or_payload else task_or_payload

    dataset_id = payload.get("dataset_id", "unknown_dataset")
    source_uri = payload["source_uri"]  # required
    start_row = int(payload.get("start_row", 0))
    shard_size = int(payload.get("shard_size", 100))
    mode = payload.get("mode", "rows")

    rows = _read_csv_shard(source_uri, start_row, shard_size)

    if mode == "count":
        return {
            "dataset_id": dataset_id,
            "mode": "count",
            "start_row": start_row,
            "row_count": len(rows),
        }

    return {
        "dataset_id": dataset_id,
        "mode": "rows",
        "start_row": start_row,
        "row_count": len(rows),
        "rows": rows,
    }
