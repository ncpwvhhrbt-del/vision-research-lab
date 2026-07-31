"""Append-only JSON Lines experiment history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def append_history(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one epoch record without rewriting earlier evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return destination
