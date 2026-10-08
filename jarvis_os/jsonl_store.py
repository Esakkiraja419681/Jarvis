"""jsonl_store - small append-only JSONL persistence helpers.

Phase 5 state lives in newline-delimited JSON files under ``data/``. Every
record must carry a stable ``id`` plus a ``ts`` (ISO timestamp), written with
an atomic append (single ``os.write`` on a file opened ``"ab"``) so a power
loss can only truncate the final line, never tear one in half.

Reading is additive and tolerant: ``replay_records`` walks the file once and
rebuilds a last-wins ``{id: record}`` map, skipping blank or corrupt lines
instead of raising. There is no index, no ordering guarantee and no history
beyond the ledger itself - these files are state snapshots, not archives.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

RECORD_KEYS = ("id", "ts")
_CORRUPT = "corrupt"  # value kept in the map under a reserved id on failure
_YEAR = 946684800       # 2000-01-01T00:00:00Z - the oldest ts Jarvis emits


def new_id(prefix: str) -> str:
    """Return a URL-safe unique id like ``content_``/``lead_``/``order_/alert_`` + hex."""
    return f"{prefix}_{uuid.uuid4().hex}"


def now_iso() -> str:
    """Current UTC time as ISO 8601, second precision."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append ``record`` to the JSONL file atomically, creating parents."""
    if "id" not in record or "ts" not in record:
        raise ValueError(f"JSONL record must carry {RECORD_KEYS}: {record}")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=True)
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line.encode("utf-8") + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def replay_records(path: Path) -> dict[str, dict[str, Any]]:
    """Rebuild a last-wins ``{id: record}`` map from a JSONL file.

    Returns ``{}`` for a missing file and never raises on malformed lines -
    the file is a state snapshot and a torn final line must not block boot.
    """
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError:
                records[_CORRUPT] = {"id": _CORRUPT, "ts": now_iso(),
                                     "error": "invalid json line"}
                continue
            if not isinstance(record, dict) or "id" not in record:
                records[_CORRUPT] = {"id": _CORRUPT, "ts": now_iso(),
                                     "error": "non-dict / id-less record"}
                continue
            records[str(record["id"])] = record
    return records


def record_count(path: Path) -> int:
    """Number of intact records in the file (corrupt lines count separately)."""
    records = replay_records(path)
    return 0 if _CORRUPT in records else len(records)