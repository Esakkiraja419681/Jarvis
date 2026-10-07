"""MemoryIndex - persistent local vector search over SQLite.

A second, query-oriented view over the same knowledge ``PersistentMemory``
stores append-only. The memory's JSON side-store grows forever; this index
keeps one row per distinct text (deduped by content hash) in a WAL-mode
SQLite file so semantic search stays a single indexed query instead of a
full scan of a multi-megabyte JSON array.

Design contract
---------------
* **Fully local.** Default embedder is ``local_embed`` (hashing, 192-d,
  deterministic, offline). The kernel swaps in the Ollama-backed embedder
  when available, but a missing Ollama never degrades to a cloud service.
* **Dedup by content.** ``id`` is the SHA-256 of ``kind|text``; re-adding
  the same fact is a no-op (``INSERT OR IGNORE``).
* **Syncable, not authoritative.** ``sync_from(memory)`` mirrors the
  memory's vector entries in. Memory stays the source of truth; the index
  can always be rebuilt from it.
* **Never writes config.** The path comes from ``_DEFAULTS``/config get.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import Any, Iterable

from .memory import cosine, local_embed

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS vectors (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL DEFAULT 'fact',
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vec_kind ON vectors(kind);
CREATE INDEX IF NOT EXISTS idx_vec_kind_text ON vectors(kind, text);
"""


def _pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def content_id(kind: str, text: str) -> str:
    digest = hashlib.sha256(f"{kind}|{text}".encode("utf-8")).hexdigest()
    return f"vec_{digest[:24]}"


class MemoryIndex:
    """SQLite-backed vector index with a pluggable local embedder."""

    def __init__(self, sqlite_path: str | Path, top_k: int = 5) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.top_k = int(top_k)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._embedder = local_embed
        self.inserted = 0
        self.duplicates = 0

    # ------------------------------------------------------------ lifecycle
    def open(self) -> "MemoryIndex":
        with self._lock:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.sqlite_path),
                                         check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def set_embedder(self, fn) -> None:
        self._embedder = fn

    # ----------------------------------------------------------------- write
    @staticmethod
    def _content_exists(conn: sqlite3.Connection, kind: str, text: str) -> bool:
        # Id-based INSERT OR IGNORE only dedups the same id. remember_fact
        # writes under content_id while sync_from mirrors under the memory's
        # own mem_ id, so without this the same fact lands twice under two ids.
        row = conn.execute(
            "SELECT 1 FROM vectors WHERE kind = ? AND text = ? LIMIT 1",
            (kind, text)).fetchone()
        return row is not None

    def add(self, text: str, kind: str = "fact",
            created_at: float | None = None) -> str | None:
        """Embed and store one text. Returns the id, or None when empty."""
        text = str(text or "").strip()
        if not text:
            return None
        row_id = content_id(kind, text)
        vector = list(self._embedder(text))
        with self._lock:
            conn = self._require()
            if self._content_exists(conn, str(kind), text):
                self.duplicates += 1
                return row_id
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO vectors (id, kind, text, vector, created_at)"
                " VALUES (?,?,?,?,?)",
                (row_id, str(kind), text, _pack(vector),
                 float(created_at or time.time())),
            )
            conn.commit()
            if conn.total_changes > before:
                self.inserted += 1
            else:
                self.duplicates += 1
        return row_id

    def add_record(self, entry: dict[str, Any]) -> str | None:
        """Store a pre-built vector entry (``sync_from`` fast path)."""
        text = str(entry.get("text") or "").strip()
        if not text:
            return None
        kind = str(entry.get("kind") or "fact")
        row_id = str(entry.get("id") or content_id(kind, text))
        vector = entry.get("vector")
        if not isinstance(vector, list) or not vector:
            vector = self._embedder(text)
        with self._lock:
            conn = self._require()
            if self._content_exists(conn, kind, text):
                self.duplicates += 1
                return row_id
            before = conn.total_changes
            conn.execute(
                "INSERT OR IGNORE INTO vectors (id, kind, text, vector, created_at)"
                " VALUES (?,?,?,?,?)",
                (row_id, kind, text, _pack([float(v) for v in vector]),
                 float(entry.get("created_at") or time.time())),
            )
            conn.commit()
            if conn.total_changes > before:
                self.inserted += 1
            else:
                self.duplicates += 1
        return row_id

    def sync_from(self, memory: Any) -> int:
        """Mirror the memory's vector entries into the index. Idempotent."""
        entries = getattr(memory, "_vectors", None) or []
        synced = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            before = self.inserted
            self.add_record(entry)
            if self.inserted > before:
                synced += 1
        return synced

    # ----------------------------------------------------------------- read
    def count(self) -> int:
        with self._lock:
            row = self._require().execute("SELECT COUNT(*) FROM vectors").fetchone()
            return int(row[0]) if row else 0

    def search(self, query: str, limit: int | None = None,
               kind: str | None = None) -> list[dict[str, Any]]:
        """Top-k cosine matches, newest first on ties."""
        query = str(query or "").strip()
        limit = int(limit or self.top_k)
        if not query:
            return []
        qvec = list(self._embedder(query))
        with self._lock:
            conn = self._require()
            if kind:
                rows = conn.execute(
                    "SELECT id, kind, text, vector, created_at FROM vectors"
                    " WHERE kind = ? ORDER BY created_at DESC", (kind,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kind, text, vector, created_at FROM vectors"
                    " ORDER BY created_at DESC").fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            sim = cosine(qvec, _unpack(row["vector"]))
            scored.append((sim, row))
        scored.sort(key=lambda pair: (pair[0], float(pair[1]["created_at"])),
                    reverse=True)
        return [
            {"id": row["id"], "kind": row["kind"], "text": row["text"],
             "score": round(float(score), 4),
             "created_at": float(row["created_at"])}
            for score, row in scored[:limit] if score > 0.0
        ]

    def context(self, query: str, budget_chars: int = 2000,
                limit: int | None = None) -> str:
        """Assemble the best matches into one context block within budget."""
        parts: list[str] = []
        used = 0
        for hit in self.search(query, limit=limit):
            text = str(hit["text"]).strip()
            if not text:
                continue
            if used + len(text) + 2 > budget_chars:
                remaining = budget_chars - used - 2
                if remaining > 40:
                    parts.append(text[:remaining] + "...")
                break
            parts.append(text)
            used += len(text) + 2
        return "\n".join(parts)

    # ---------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.sqlite_path),
            "rows": self.count(),
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "top_k": self.top_k,
            "embedder": getattr(self._embedder, "__name__", "custom"),
        }

    def kinds(self) -> list[str]:
        with self._lock:
            rows = self._require().execute(
                "SELECT DISTINCT kind FROM vectors ORDER BY kind").fetchall()
        return [str(r[0]) for r in rows]
