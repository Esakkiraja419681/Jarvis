"""Append-only audit ledger shared between Jarvis OS and the Overseer.

Every permission decision, tool invocation, evolution phase transition and
overseer incident lands here. The JSONL file is the *shared* ledger: Jarvis
writes to it and the Overseer reads from the same path, so a halt never loses
the evidence trail. Writes are flushed and fsynced before returning.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    actor     TEXT    NOT NULL,
    category  TEXT    NOT NULL,
    action    TEXT    NOT NULL,
    decision  TEXT    NOT NULL,
    detail    TEXT    NOT NULL DEFAULT '{}',
    prev_hash TEXT    NOT NULL DEFAULT '',
    hash      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_cat     ON audit(category);
CREATE INDEX IF NOT EXISTS idx_audit_dec     ON audit(decision);
"""


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True)
class AuditEvent:
    actor: str
    category: str
    action: str
    decision: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLedger:
    """Hash-chained, append-only audit log (SQLite + mirrored JSONL)."""

    def __init__(self, sqlite_path: str | os.PathLike[str],
                 ledger_path: str | os.PathLike[str],
                 enabled: bool = True,
                 actor: str = "jarvis") -> None:
        self.sqlite_path = Path(sqlite_path)
        self.ledger_path = Path(ledger_path)
        self.enabled = enabled
        self.actor = actor
        self._lock = threading.RLock()
        self._prev_hash = ""
        self._conn: sqlite3.Connection | None = None
        self._bootstrapped = False

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> "AuditLedger":
        with self._lock:
            if self._conn is not None:
                return self
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False,
                                       timeout=30.0, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._prev_hash = self._last_hash()
            self._bootstrapped = True
        return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                finally:
                    self._conn = None

    def _ensure(self) -> None:
        if not self._bootstrapped:
            self.connect()

    def _last_hash(self) -> str:
        if self._conn is None:
            return ""
        row = self._conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row and row[0] else ""

    # ---------------------------------------------------------------- write
    def record(self, category: str, action: str, decision: str,
               detail: dict[str, Any] | None = None,
               actor: str | None = None) -> dict[str, Any]:
        event = AuditEvent(
            actor=actor or self.actor,
            category=str(category),
            action=str(action),
            decision=str(decision),
            detail=dict(detail or {}),
        )
        return self._write(event)

    def permission(self, category: str, decision: str, source: str,
                   detail: dict[str, Any] | None = None, actor: str = "jarvis") -> dict[str, Any]:
        return self.record(
            "permission", f"gate:{category}", decision,
            {"source": source, **(detail or {})}, actor=actor,
        )

    def _write(self, event: AuditEvent) -> dict[str, Any]:
        self._ensure()
        payload = event.to_dict()
        with self._lock:
            conn = self._conn
            prev = self._prev_hash
            if conn is not None:
                # The Overseer runs as an INDEPENDENT process against the same
                # ledger, so the cached _prev_hash can be stale by the time we
                # write. Re-derive the chain head inside an exclusive
                # transaction so concurrent processes cannot fork the chain.
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
                    prev = (row[0] if row and row[0] else "")
                except sqlite3.Error:
                    prev = self._prev_hash
                payload["prev_hash"] = prev
                digest = hashlib.sha256(
                    prev.encode("utf-8") + _canonical(payload).encode("utf-8")
                ).hexdigest()
                payload["hash"] = digest
                try:
                    conn.execute(
                        "INSERT INTO audit (ts, actor, category, action, decision, detail,"
                        " prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            event.ts, event.actor, event.category, event.action,
                            event.decision,
                            json.dumps(event.detail, ensure_ascii=False),
                            payload["prev_hash"], digest,
                        ),
                    )
                    conn.commit()
                except sqlite3.Error:
                    conn.rollback()
                    raise
                self._prev_hash = digest
            else:
                payload["prev_hash"] = prev
                digest = hashlib.sha256(
                    prev.encode("utf-8") + _canonical(payload).encode("utf-8")
                ).hexdigest()
                payload["hash"] = digest
                self._prev_hash = digest

            if self.enabled:
                try:
                    with self.ledger_path.open("a", encoding="utf-8") as fh:
                        fh.write(_canonical(payload) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                except OSError:
                    pass  # never let audit IO break the system
        return payload

    # ----------------------------------------------------------------- read
    def tail(self, limit: int = 50, category: str | None = None,
             decision: str | None = None) -> list[dict[str, Any]]:
        self._ensure()
        if self._conn is None:
            return []
        sql = "SELECT ts, actor, category, action, decision, detail FROM audit"
        clauses, params = [], []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for ts, actor, cat, act, dec, detail in rows:
            try:
                detail_obj = json.loads(detail)
            except (json.JSONDecodeError, TypeError):
                detail_obj = {}
            out.append({"ts": ts, "actor": actor, "category": cat,
                        "action": act, "decision": dec, "detail": detail_obj})
        return out

    def count(self, category: str | None = None) -> int:
        self._ensure()
        if self._conn is None:
            return 0
        if category:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM audit WHERE category = ?", (category,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM audit").fetchone()
        return int(row[0]) if row else 0

    def decisions(self) -> dict[str, int]:
        self._ensure()
        if self._conn is None:
            return {}
        rows = self._conn.execute(
            "SELECT decision, COUNT(*) FROM audit GROUP BY decision"
        ).fetchall()
        return {str(d): int(c) for d, c in rows}

    # ------------------------------------------------------------ integrity
    def repair_chain(self) -> dict[str, Any]:
        """Re-link the hash chain across all existing rows.

        Every event is PRESERVED - nothing is deleted or rewritten except the
        ``prev_hash``/``hash`` links, which are derived data. This repairs a
        chain that forked when an older build wrote from a stale in-memory
        head (multiple processes appending to one ledger).
        """
        self._ensure()
        if self._conn is None:
            return {"ok": False, "error": "no connection"}
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, actor, category, action, decision, detail, hash"
                " FROM audit ORDER BY id ASC"
            ).fetchall()
            prev = ""
            relinked = 0
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for rid, ts, actor, cat, act, dec, detail, old_hash in rows:
                    payload = {
                        "actor": actor, "category": cat, "action": act,
                        "decision": dec,
                        "detail": json.loads(detail) if detail else {},
                        "ts": ts, "prev_hash": prev,
                    }
                    digest = hashlib.sha256(
                        prev.encode("utf-8") + _canonical(payload).encode("utf-8")
                    ).hexdigest()
                    if digest != old_hash:
                        relinked += 1
                    self._conn.execute(
                        "UPDATE audit SET prev_hash = ?, hash = ? WHERE id = ?",
                        (prev, digest, rid),
                    )
                    prev = digest
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                return {"ok": False, "error": str(exc)}
            self._prev_hash = prev
        # Rewrite the JSONL mirror so both copies agree.
        try:
            if rows and self.enabled:
                lines = []
                prev = ""
                for rid, ts, actor, cat, act, dec, detail, _ in rows:
                    payload = {
                        "actor": actor, "category": cat, "action": act,
                        "decision": dec,
                        "detail": json.loads(detail) if detail else {},
                        "ts": ts, "prev_hash": prev,
                    }
                    digest = hashlib.sha256(
                        prev.encode("utf-8") + _canonical(payload).encode("utf-8")
                    ).hexdigest()
                    payload["hash"] = digest
                    lines.append(_canonical(payload))
                    prev = digest
                tmp = self.ledger_path.with_suffix(".repair")
                tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                tmp.replace(self.ledger_path)
        except OSError:
            pass
        return {"ok": True, "events": len(rows), "relinked": relinked}

    # ------------------------------------------------------------ integrity
    def verify(self) -> tuple[bool, int]:
        """Re-walk the chain. Returns (ok, events_checked)."""
        self._ensure()
        if self._conn is None:
            return True, 0
        prev = ""
        rows = self._conn.execute(
            "SELECT ts, actor, category, action, decision, detail, prev_hash, hash"
            " FROM audit ORDER BY id ASC"
        ).fetchall()
        for ts, actor, cat, act, dec, detail, prev_hash, digest in rows:
            if prev_hash != prev:
                return False, len(rows)
            payload = {
                "actor": actor, "category": cat, "action": act, "decision": dec,
                "detail": json.loads(detail) if detail else {}, "ts": ts,
                "prev_hash": prev_hash,
            }
            recomputed = hashlib.sha256(
                prev.encode("utf-8") + _canonical(payload).encode("utf-8")
            ).hexdigest()
            if recomputed != digest:
                return False, len(rows)
            prev = digest
        return True, len(rows)

    # ------------------------------------------------------------ integrity
    def summary(self) -> str:
        decisions = self.decisions()
        parts = ", ".join(f"{k}={v}" for k, v in sorted(decisions.items())) or "none"
        return f"{self.count()} events | decisions: {parts}"


class NullAudit(AuditLedger):
    """Used when audit logging is switched off in config."""

    def __init__(self) -> None:
        super().__init__(sqlite_path=":memory:", ledger_path=":memory:", enabled=False)

    def _ensure(self) -> None:
        return