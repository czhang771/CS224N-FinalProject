"""JSONL read/write helpers."""

import json
import os
from typing import Any


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict[str, Any], path: str) -> None:
    """Append a single record to a JSONL file (creates file if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_pubids(path: str) -> set[str]:
    """Return set of pubids already written to the output file."""
    if not os.path.exists(path):
        return set()
    records = load_jsonl(path)
    return {str(r["pubid"]) for r in records if "pubid" in r}
