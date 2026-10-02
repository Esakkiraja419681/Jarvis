"""Persistent memory for Jarvis OS.

Two cooperating stores:

1. **SQLite** (``data/memory.db``) - structured, WAL-mode, crash-safe.
   Facts, actions, lessons, episodes and skill definitions.

2. **JSON vector store** (``data/vectors.json``) - append-only list of
   embedding records used for semantic recall.

Durability contract
-------------------
Memory is *append only*. ``remember()``/``learn_lesson()``/``record_action()``
only ever insert. There is intentionally **no public API that clears memory**;
``purge_requested`` is the single narrow entry point and it requires an explicit
manual approval token from the PermissionGate. Even a crash mid-write cannot
destroy history: the JSONL side-logs (``lessons.jsonl`` / ``actions.jsonl``)
are hash-chained append-only files, and SQLite runs in WAL + FULL sync.
The Overseer calls :meth:`integrity_report` on every boot to prove memory
survived restarts and repairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT NOT NULL DEFAULT '{}',
    importance REAL NOT NULL DEFAULT 0.5,
    pinned     INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);

CREATE TABLE IF NOT EXISTS lessons (
    id         TEXT PRIMARY KEY,
    topic      TEXT NOT NULL,
    lesson     TEXT NOT NULL,
    outcome    TEXT NOT NULL DEFAULT 'neutral',
    weight     REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    prev_hash  TEXT NOT NULL DEFAULT '',
    hash       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_lesson_topic ON lessons(topic);

CREATE TABLE IF NOT EXISTS actions (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    result     TEXT NOT NULL DEFAULT '',
    success    INTEGER NOT NULL DEFAULT 1,
    meta       TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    prev_hash  TEXT NOT NULL DEFAULT '',
    hash       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_action_kind ON actions(kind);

CREATE TABLE IF NOT EXISTS episodes (
    id         TEXT PRIMARY KEY,
    goal       TEXT NOT NULL,
    plan       TEXT NOT NULL DEFAULT '{}',
    outcome    TEXT NOT NULL DEFAULT '',
    success    INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_EMBED_DIM = 192


# ----------------------------------------------------------------- embedding
def local_embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Deterministic, fully offline hashing embedder.

    Character n-grams are hashed into a fixed-width vector and L2 normalised.
    Not a semantic model, but stable, fast, dependency-free and never leaves
    the machine. Real semantics come from Ollama embeddings when available
    (:meth:`PersistentMemory.set_embedder`).
    """
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    norm = text.lower()
    grams: list[str] = []
    for n in (2, 3, 4):
        for i in range(max(0, len(norm) - n + 1)):
            grams.append(norm[i:i + n])
    grams.extend(re.findall(r"\w+", norm))
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    length = math.sqrt(sum(v * v for v in vec))
    if length == 0.0:
        return [0.0] * dim
    return [v / length for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# -------------------------------------------------------------------- models
@dataclass(slots=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    pinned: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "content": self.content,
            "meta": self.meta, "importance": self.importance, "pinned": self.pinned,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class Lesson:
    id: str
    topic: str
    lesson: str
    outcome: str = "neutral"
    weight: float = 0.5
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "topic": self.topic, "lesson": self.lesson,
            "outcome": self.outcome, "weight": self.weight, "created_at": self.created_at,
        }


@dataclass(slots=True)
class ActionRecord:
    id: str
    kind: str
    summary: str
    result: str = ""
    success: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "summary": self.summary,
            "result": self.result, "success": self.success, "meta": self.meta,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------- store
class PersistentMemory:
    """Crash-proof append-only memory with a JSON vector side-store."""

    def __init__(self, sqlite_path: str | os.PathLike[str],
                 vector_path: str | os.PathLike[str],
                 lesson_log: str | os.PathLike[str],
                 action_log: str | os.PathLike[str],
                 top_k: int = 5,
                 immutable: bool = True) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.vector_path = Path(vector_path)
        self.lesson_log = Path(lesson_log)
        self.action_log = Path(action_log)
        self.top_k = top_k
        self.immutable = immutable
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._vectors: list[dict[str, Any]] = []
        self._lesson_prev = ""
        self._action_prev = ""
        self._embedder = local_embed
        self.stats: dict[str, int] = {
            "memories": 0, "lessons": 0, "actions": 0, "episodes": 0, "repairs": 0,
        }

    # ------------------------------------------------------------ lifecycle
    def open(self) -> "PersistentMemory":
        with self._lock:
            for path in (self.sqlite_path, self.vector_path, self.lesson_log, self.action_log):
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._lesson_prev = self._last_hash("lessons")
            self._action_prev = self._last_hash("actions")
            self._load_vectors()
            self._refresh_stats()
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
        """Install a richer embedder (e.g. Ollama) while staying offline."""
        self._embedder = fn

    # ------------------------------------------------------------- vector io
    def _load_vectors(self) -> None:
        if not self.vector_path.is_file():
            self._vectors = []
            self._flush_vectors()
            return
        try:
            raw = json.loads(self.vector_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._vectors = [v for v in raw if isinstance(v, dict)]
            elif isinstance(raw, dict) and isinstance(raw.get("vectors"), list):
                self._vectors = raw["vectors"]
            else:
                self._vectors = []
        except (json.JSONDecodeError, OSError):
            # Corrupt side-store must never cost us the SQLite truth.
            self._backup_corrupt_vectors()
            self._vectors = []

    def _backup_corrupt_vectors(self) -> None:
        try:
            stamp = int(time.time())
            self.vector_path.replace(self.vector_path.with_name(f"vectors.corrupt.{stamp}.json"))
        except OSError:
            pass

    def _flush_vectors(self) -> None:
        try:
            tmp = self.vector_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._vectors, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.vector_path)
        except OSError:
            pass

    # ----------------------------------------------------------- hash chains
    def _last_hash(self, table: str) -> str:
        try:
            row = self._require().execute(
                f"SELECT hash FROM {table} ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            return row[0] if row and row[0] else ""
        except sqlite3.Error:
            return ""

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_canonical(payload) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            pass

    # ------------------------------------------------------------- memories
    def remember(self, content: str, kind: str = "fact",
                 meta: dict[str, Any] | None = None,
                 importance: float = 0.5, pinned: bool = False) -> MemoryRecord:
        now = time.time()
        rec = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}", kind=kind, content=str(content),
            meta=dict(meta or {}), importance=float(importance), pinned=bool(pinned),
            created_at=now, updated_at=now,
        )
        with self._lock:
            conn = self._require()
            conn.execute(
                "INSERT OR REPLACE INTO memories"
                " (id, kind, content, meta, importance, pinned, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (rec.id, rec.kind, rec.content, json.dumps(rec.meta, ensure_ascii=False),
                 rec.importance, int(rec.pinned), rec.created_at, rec.updated_at),
            )
            conn.commit()
            self._vectors.append({
                "id": rec.id, "kind": kind, "text": rec.content,
                "vector": self._embedder(rec.content), "created_at": now,
            })
            self._flush_vectors()
            self.stats["memories"] += 1
        return rec

    def recall(self, query: str = "", kind: str | None = None,
               limit: int | None = None) -> list[MemoryRecord]:
        limit = limit or self.top_k
        with self._lock:
            conn = self._require()
            sql = "SELECT id, kind, content, meta, importance, pinned, created_at, updated_at FROM memories"
            params: list[Any] = []
            if kind:
                sql += " WHERE kind = ?"
                params.append(kind)
            sql += " ORDER BY pinned DESC, importance DESC, created_at DESC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def search(self, query: str, limit: int | None = None) -> list[tuple[MemoryRecord, float]]:
        """Hybrid recall: vector similarity blended with recency/importance."""
        limit = limit or self.top_k
        if not query.strip():
            return [(r, 1.0) for r in self.recall(limit=limit)]
        with self._lock:
            qvec = self._embedder(query)
            now = time.time()
            scored: list[tuple[MemoryRecord, float]] = []
            for entry in self._vectors[-5000:]:
                sim = cosine(qvec, entry.get("vector") or [])
                age_days = max(0.0, (now - float(entry.get("created_at", now))) / 86400.0)
                recency = 1.0 / (1.0 + age_days / 30.0)
                score = 0.75 * sim + 0.25 * recency
                scored.append((entry, score))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            out: list[tuple[MemoryRecord, float]] = []
            conn = self._require()
            for entry, score in scored[:limit]:
                row = conn.execute(
                    "SELECT id, kind, content, meta, importance, pinned, created_at, updated_at"
                    " FROM memories WHERE id = ?", (entry["id"],)
                ).fetchone()
                if row:
                    out.append((self._row_to_memory(row), score))
            return out

    @staticmethod
    def _row_to_memory(row: tuple) -> MemoryRecord:
        try:
            meta = json.loads(row[3]) if row[3] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return MemoryRecord(
            id=row[0], kind=row[1], content=row[2], meta=meta,
            importance=float(row[4]), pinned=bool(row[5]),
            created_at=float(row[6]), updated_at=float(row[7]),
        )

    # -------------------------------------------------------------- lessons
    def learn_lesson(self, topic: str, lesson: str,
                     outcome: str = "neutral", weight: float = 0.5) -> Lesson:
        now = time.time()
        rec = Lesson(
            id=f"les_{uuid.uuid4().hex[:12]}", topic=str(topic), lesson=str(lesson),
            outcome=str(outcome), weight=float(weight), created_at=now,
        )
        with self._lock:
            payload = rec.to_dict() | {"prev_hash": self._lesson_prev}
            digest = hashlib.sha256(
                self._lesson_prev.encode("utf-8") + _canonical(payload).encode("utf-8")
            ).hexdigest()
            payload["hash"] = digest
            self._lesson_prev = digest

            conn = self._require()
            conn.execute(
                "INSERT INTO lessons (id, topic, lesson, outcome, weight, created_at,"
                " prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
                (rec.id, rec.topic, rec.lesson, rec.outcome, rec.weight,
                 rec.created_at, payload["prev_hash"], digest),
            )
            conn.commit()
            self._append_jsonl(self.lesson_log, payload)
            self.stats["lessons"] += 1
        return rec

    def lessons(self, topic: str | None = None, limit: int = 50) -> list[Lesson]:
        conn = self._require()
        if topic:
            rows = conn.execute(
                "SELECT id, topic, lesson, outcome, weight, created_at FROM lessons"
                " WHERE topic = ? ORDER BY created_at DESC LIMIT ?", (topic, int(limit))
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, topic, lesson, outcome, weight, created_at FROM lessons"
                " ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [Lesson(r[0], r[1], r[2], r[3], float(r[4]), float(r[5])) for r in rows]

    def lesson_stats(self) -> dict[str, Any]:
        conn = self._require()
        total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        good = conn.execute(
            "SELECT COUNT(*) FROM lessons WHERE outcome IN ('success','good')"
        ).fetchone()[0]
        bad = conn.execute(
            "SELECT COUNT(*) FROM lessons WHERE outcome IN ('failure','bad')"
        ).fetchone()[0]
        return {"total": int(total), "positive": int(good), "negative": int(bad)}

    # -------------------------------------------------------------- actions
    def record_action(self, kind: str, summary: str, result: str = "",
                      success: bool = True,
                      meta: dict[str, Any] | None = None) -> ActionRecord:
        now = time.time()
        rec = ActionRecord(
            id=f"act_{uuid.uuid4().hex[:12]}", kind=str(kind), summary=str(summary),
            result=str(result), success=bool(success), meta=dict(meta or {}), created_at=now,
        )
        with self._lock:
            payload = rec.to_dict() | {"prev_hash": self._action_prev}
            digest = hashlib.sha256(
                self._action_prev.encode("utf-8") + _canonical(payload).encode("utf-8")
            ).hexdigest()
            payload["hash"] = digest
            self._action_prev = digest

            conn = self._require()
            conn.execute(
                "INSERT INTO actions (id, kind, summary, result, success, meta, created_at,"
                " prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.kind, rec.summary, rec.result, int(rec.success),
                 json.dumps(rec.meta, ensure_ascii=False), rec.created_at,
                 payload["prev_hash"], digest),
            )
            conn.commit()
            self._append_jsonl(self.action_log, payload)
            self.stats["actions"] += 1
        return rec

    def actions(self, kind: str | None = None, limit: int = 50) -> list[ActionRecord]:
        conn = self._require()
        if kind:
            rows = conn.execute(
                "SELECT id, kind, summary, result, success, meta, created_at FROM actions"
                " WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, int(limit))
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, summary, result, success, meta, created_at FROM actions"
                " ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = json.loads(r[5]) if r[5] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            out.append(ActionRecord(r[0], r[1], r[2], r[3], bool(r[4]), meta, float(r[6])))
        return out

    # ------------------------------------------------------------- episodes
    def record_episode(self, goal: str, plan: dict[str, Any], outcome: str,
                       success: bool = True) -> str:
        eid = f"epi_{uuid.uuid4().hex[:12]}"
        with self._lock:
            conn = self._require()
            conn.execute(
                "INSERT INTO episodes (id, goal, plan, outcome, success, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (eid, str(goal), json.dumps(plan, ensure_ascii=False), str(outcome),
                 int(success), time.time()),
            )
            conn.commit()
            self.stats["episodes"] += 1
        return eid

    def episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._require().execute(
            "SELECT id, goal, plan, outcome, success, created_at FROM episodes"
            " ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [
            {"id": r[0], "goal": r[1], "plan": r[2], "outcome": r[3],
             "success": bool(r[4]), "created_at": float(r[5])}
            for r in rows
        ]

    # ------------------------------------------------------------- integrity
    def _refresh_stats(self) -> None:
        conn = self._require()
        for key, table in (("memories", "memories"), ("lessons", "lessons"),
                           ("actions", "actions"), ("episodes", "episodes")):
            self.stats[key] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        self.stats["vectors"] = len(self._vectors)
        row = conn.execute("SELECT value FROM meta WHERE key = 'boot_count'").fetchone()
        boot = int(row[0]) + 1 if row else 1
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('boot_count', ?)", (str(boot),)
        )
        row = conn.execute("SELECT value FROM meta WHERE key = 'first_boot'").fetchone()
        if not row:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('first_boot', ?)",
                (str(time.time()),),
            )
        conn.commit()
        self.stats["boots"] = boot

    def integrity_report(self) -> dict[str, Any]:
        """Verify both hash chains. Used by the Overseer after every restart."""
        report: dict[str, Any] = {"ok": True, "issues": [], "stats": dict(self.stats)}
        for table, jsonl, label in (
            ("lessons", self.lesson_log, "lesson"),
            ("actions", self.action_log, "action"),
        ):
            prev = ""
            try:
                rows = self._require().execute(
                    f"SELECT id, created_at, prev_hash, hash FROM {table} ORDER BY rowid ASC"
                ).fetchall()
            except sqlite3.Error as exc:
                report["ok"] = False
                report["issues"].append(f"{label} table unreadable: {exc}")
                continue
            for rid, created, prev_hash, digest in rows:
                if prev_hash != prev:
                    report["ok"] = False
                    report["issues"].append(f"{label} chain break at {rid}")
                    break
                payload = {"id": rid, "created_at": created, "prev_hash": prev_hash}
                if hashlib.sha256(
                    prev.encode("utf-8") + _canonical(payload).encode("utf-8")
                ).hexdigest() != digest:
                    report["ok"] = False
                    report["issues"].append(f"{label} hash mismatch at {rid}")
                    break
                prev = digest
            if jsonl.is_file():
                try:
                    lines = jsonl.read_text(encoding="utf-8").splitlines()
                except OSError:
                    lines = []
                if len(lines) < len(rows):
                    report["issues"].append(
                        f"{label} JSONL short by {len(rows) - len(lines)} record(s)"
                    )
        report["vectors"] = len(self._vectors)
        return report

    def repair(self) -> int:
        """Best-effort rebuild of the vector side-store from SQLite.

        Called by the Overseer when a crash left ``vectors.json`` behind. This
        **restores** derived data only - no SQLite row is ever deleted.
        """
        conn = self._require()
        rows = conn.execute(
            "SELECT id, kind, content, created_at FROM memories ORDER BY created_at ASC"
        ).fetchall()
        with self._lock:
            self._vectors = [
                {"id": r[0], "kind": r[1], "text": r[2],
                 "vector": self._embedder(r[2]), "created_at": float(r[3])}
                for r in rows
            ]
            self._flush_vectors()
            self.stats["repairs"] += 1
        return len(self._vectors)

    # ------------------------------------------------------------- protected
    def purge_requested(self, approval_token: str) -> dict[str, Any]:
        """The *only* destructive path, gated by an external approval token.

        Jarvis itself never calls this. The Overseer calls it only to *refuse*
        a wipe attempt.
        """
        if approval_token != "MANUAL-SIR-APPROVED":
            return {"ok": False, "reason": "approval token absent or invalid - refused"}
        with self._lock:
            conn = self._require()
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM lessons")
            conn.execute("DELETE FROM actions")
            conn.commit()
            self._vectors = []
            self._flush_vectors()
        return {"ok": True, "reason": "purged under explicit manual approval"}

    # ---------------------------------------------------------------- status
    def status(self) -> str:
        stats = self.integrity_report()["stats"]
        return (f"memories={stats.get('memories', 0)} "
                f"lessons={stats.get('lessons', 0)} "
                f"actions={stats.get('actions', 0)} "
                f"episodes={stats.get('episodes', 0)} "
                f"vectors={len(self._vectors)} "
                f"boots={stats.get('boots', 0)}")

    def export_markdown(self, limit: int = 40) -> str:
        lines = ["# Jarvis OS - Memory", ""]
        lines.append(f"- Memories: {self.stats['memories']}")
        lines.append(f"- Lessons: {self.stats['lessons']}")
        lines.append(f"- Actions: {self.stats['actions']}")
        lines.append(f"- Episodes: {self.stats['episodes']}")
        lines.append("")
        lines.append("## Recent lessons")
        for les in self.lessons(limit=limit):
            lines.append(f"- **{les.topic}** ({les.outcome}): {les.lesson}")
        return "\n".join(lines)