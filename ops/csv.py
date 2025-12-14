import csv
from typing import Any, Dict, List


def read_csv_shard(source_uri: str, start_row: int, shard_size: int) -> List[Dict[str, Any]]:
    """
    Read a slice of rows from a CSV (after the header).
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


def op_read_csv_shard(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic CSV shard op.

    Expected payload:
      {
        "dataset_id": "inc5000_2019",
        "source_uri": "/path/to/file.csv",
        "start_row": 0,
        "shard_size": 100,
        "mode": "rows" | "count"
      }

    mode = "rows": return the raw rows.
    mode = "count": return just a count for this shard.
    """
    payload = task.get("payload", {})

    dataset_id = payload.get("dataset_id", "unknown_dataset")
    source_uri = payload["source_uri"]          # required
    start_row = int(payload.get("start_row", 0))
    shard_size = int(payload.get("shard_size", 100))
    mode = payload.get("mode", "rows")

    rows = read_csv_shard(source_uri, start_row, shard_size)

    if mode == "count":
        return {
            "dataset_id": dataset_id,
            "mode": "count",
            "start_row": start_row,
            "row_count": len(rows),
        }

    # default: raw rows
    return {
        "dataset_id": dataset_id,
        "mode": "rows",
        "start_row": start_row,
        "row_count": len(rows),
        "rows": rows,
    }
