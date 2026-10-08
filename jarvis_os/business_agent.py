"""BusinessAgent - offline multi-channel pipeline automation.

Runs the operator's content pipeline and CRM entirely on the local machine:
drafting, human review and (approval-gated) publishing of posts, plus lead
capture, touch tracking, stage moves and a funnel report. Every mutation is
appended to the JSONL ledger and audited, mirroring the rest of Jarvis OS.

Content publishing routes through the ``business_publish`` permission
category, so a real shell always prompts Sir before anything goes out. When
the kernel has no gate attached (a headless auto-pilot constructed directly,
or a pre-boot config probe), publishing degrades to *allowed* - there is
nothing live to protect yet. State lives under ``data/`` and is ignored by
``MEMORY_GLOBS``, so a self-update snapshot never re-commits it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import jsonl_store

CHANNELS = ("blog", "email", "social")
PIPELINE_STAGES = ("draft", "approved", "rejected", "published")
CRM_STAGES = ("new", "contacted", "qualified", "proposal", "won", "lost")

_GATE_CATEGORY = "business_publish"
_GATE_ACTION = "content_publish"

_TEMPLATES = {
    "blog": "{topic}: our working notes on practical implementation so far. "
            "We prefer a small, testable step to a grand design, and we keep "
            "the receipts in the ledger.",
    "email": "Dear {audience}.\n\nOur latest notes on {topic}, kept tight and "
             "actionable. Reply if you would like the worked details.",
    "social": "Thread on {topic} - proof over promises, notes over noise.",
}

_STAGE_COUNT = "current stage counts across the funnel"


def _render(record: dict[str, Any]) -> str:
    """Compact single-line renderer for one state record."""
    kind = record.get("kind", "record")
    key = record.get("id", "?")
    parts = [k for k in ("channel", "topic", "stage") if record.get(k)]
    extra = ""
    if record.get("touches"):
        extra = f" touches={len(record['touches'])}"
    if record.get("score") is not None:
        extra += f" score={record['score']}"
    where = f" [{', '.join(parts)}]" if parts else ""
    return f"{kind} {key}{where}{extra}"


def _render_lead(record: dict[str, Any]) -> str:
    parts = [record.get("stage", "?"), record.get("name", "?")]
    if record.get("company"):
        parts.append(f"@{record['company']}")
    if record.get("score") is not None:
        parts.append(f"score={record['score']}")
    return " | ".join(str(p) for p in parts)


class BusinessAgent:
    """Content pipeline and CRM, both persisted as JSONL state files."""

    # ------------------------------------------------------------- lifecycle
    def __init__(self, kernel: Any, config: Any | None = None) -> None:
        self.kernel = kernel
        self.config = config if config is not None else kernel.config
        self.pipeline_path = Path(
            str(self.config.get("business.pipeline_path", "data/content_pipeline.jsonl")))
        self.crm_path = Path(
            str(self.config.get("business.crm_path", "data/crm_contacts.jsonl")))
        self.ledger_path = Path(
            str(self.config.get("business.ledger_path", "data/business_ledger.jsonl")))

    def _record(self, action: str, result: str, detail: dict[str, Any]) -> None:
        jsonl_store.append_record(self.ledger_path, {
            "id": jsonl_store.new_id("action"),
            "ts": jsonl_store.now_iso(),
            "kind": "action",
            "actor": "BusinessAgent",
            "action": action,
            "result": result,
            "detail": detail,
        })
        audit = getattr(self.kernel, "audit", None)
        if audit is not None:
            audit.record(action, "business", "ok" if result == "ok" else "denied", detail)

    def _authorize(self, category: str, action: str) -> tuple[bool, str]:
        """Ask the live gate; a missing gate (headless) trusts the caller."""
        gate = getattr(self.kernel, "gate", None)
        if gate is None:
            return True, "allowed - no gate attached (headless)"
        try:
            granted = gate.approve(category, f"publish content: {action}",
                                   {"action": action, "category": category})
        except Exception:  # noqa: BLE001 - a failing gate must fail closed
            return False, "gate error - denied"
        if not granted:
            return False, "denied by operator"
        return True, "approved live"

    # ------------------------------------------------------- content pipeline
    def content_draft(self, channel: str, topic: str, audience: str = "") -> dict[str, Any]:
        channel = channel.lower().strip()
        if channel not in CHANNELS:
            return {"error": f"unknown channel {channel!r}; choose {CHANNELS}"}
        record = {
            "id": jsonl_store.new_id("content"),
            "ts": jsonl_store.now_iso(),
            "kind": "content",
            "channel": channel,
            "topic": topic.strip(),
            "stage": "draft",
            "body": _TEMPLATES[channel].format(topic=topic.strip(),
                                               audience=audience or "the working group"),
        }
        jsonl_store.append_record(self.pipeline_path, record)
        self._record("content_draft", "ok", {"id": record["id"], "channel": channel})
        return record

    def content_review(self, content_id: str, verdict: str) -> dict[str, Any]:
        verdict = verdict.lower().strip()
        if verdict not in ("approved", "rejected"):
            return {"error": "verdict must be approved or rejected"}
        records = jsonl_store.replay_records(self.pipeline_path)
        record = records.get(content_id)
        if record is None or record.get("kind") != "content":
            return {"error": f"no content record {content_id}"}
        record["stage"] = verdict
        record["reviewed_at"] = jsonl_store.now_iso()
        jsonl_store.append_record(self.pipeline_path, record)
        self._record("content_review", "ok",
                     {"id": content_id, "verdict": verdict})
        return record

    def content_publish(self, content_id: str) -> dict[str, Any]:
        records = jsonl_store.replay_records(self.pipeline_path)
        record = records.get(content_id)
        if record is None or record.get("kind") != "content":
            return {"error": f"no content record {content_id}"}
        if record.get("stage") != "approved":
            return {"error": f"{content_id} is {record.get('stage')}; approve it first"}
        allowed, how = self._authorize(_GATE_CATEGORY, _GATE_ACTION)
        if not allowed:
            self._record("content_publish", "denied", {"id": content_id})
            return {"error": f"publish {content_id} {how}"}
        record["stage"] = "published"
        record["published_at"] = jsonl_store.now_iso()
        jsonl_store.append_record(self.pipeline_path, record)
        self._record("content_publish", "ok", {"id": content_id})
        return record

    def content_status(self, content_id: str) -> dict[str, Any] | None:
        return jsonl_store.replay_records(self.pipeline_path).get(content_id)

    def content_list(self, stage: str = "", channel: str = "") -> list[dict[str, Any]]:
        out = []
        for record in jsonl_store.replay_records(self.pipeline_path).values():
            if record.get("kind") != "content":
                continue
            if stage and record.get("stage") != stage:
                continue
            if channel and record.get("channel") != channel:
                continue
            out.append(record)
        return sorted(out, key=lambda r: str(r.get("ts", "")))

    # -------------------------------------------------------------------- CRM
    def lead_add(self, name: str, company: str = "", email: str = "",
                 source: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            return {"error": "a lead needs a name"}
        record = {
            "id": jsonl_store.new_id("lead"),
            "ts": jsonl_store.now_iso(),
            "kind": "lead",
            "name": name,
            "company": company.strip(),
            "email": email.strip(),
            "source": source.strip() or "manual",
            "stage": "new",
            "score": 0,
            "touches": [],
        }
        jsonl_store.append_record(self.crm_path, record)
        self._record("lead_add", "ok", {"id": record["id"]})
        return record

    def lead_add_touch(self, lead_id: str, method: str, note: str = "") -> dict[str, Any]:
        records = jsonl_store.replay_records(self.crm_path)
        record = records.get(lead_id)
        if record is None or record.get("kind") != "lead":
            return {"error": f"no lead record {lead_id}"}
        record.setdefault("touches", []).append({
            "ts": jsonl_store.now_iso(),
            "method": method.strip() or "note",
            "note": note.strip(),
        })
        jsonl_store.append_record(self.crm_path, record)
        self._record("lead_touch", "ok", {"id": lead_id, "method": method.strip()})
        return record

    def lead_move_stage(self, lead_id: str, stage: str) -> dict[str, Any]:
        stage = stage.lower().strip()
        if stage not in CRM_STAGES:
            return {"error": f"unknown stage {stage!r}; choose {CRM_STAGES}"}
        records = jsonl_store.replay_records(self.crm_path)
        record = records.get(lead_id)
        if record is None or record.get("kind") != "lead":
            return {"error": f"no lead record {lead_id}"}
        record["stage"] = stage
        record.setdefault("touches", []).append({
            "ts": jsonl_store.now_iso(), "method": "stage", "note": f"moved to {stage}",
        })
        jsonl_store.append_record(self.crm_path, record)
        self._record("lead_move_stage", "ok", {"id": lead_id, "stage": stage})
        return record

    def lead_set_score(self, lead_id: str, score: int) -> dict[str, Any]:
        try:
            score = int(score)
        except (TypeError, ValueError):
            return {"error": "score must be an integer"}
        score = max(0, min(100, score))
        records = jsonl_store.replay_records(self.crm_path)
        record = records.get(lead_id)
        if record is None or record.get("kind") != "lead":
            return {"error": f"no lead record {lead_id}"}
        record["score"] = score
        jsonl_store.append_record(self.crm_path, record)
        self._record("lead_set_score", "ok", {"id": lead_id, "score": score})
        return record

    def lead_get(self, lead_id: str) -> dict[str, Any] | None:
        return jsonl_store.replay_records(self.crm_path).get(lead_id)

    def leads(self, stage: str = "", min_score: int | None = None) -> list[dict[str, Any]]:
        out = []
        for record in jsonl_store.replay_records(self.crm_path).values():
            if record.get("kind") != "lead":
                continue
            if stage and record.get("stage") != stage:
                continue
            if min_score is not None and int(record.get("score", 0)) < int(min_score):
                continue
            out.append(record)
        return sorted(out, key=lambda r: str(r.get("ts", "")))

    def leads_summary(self) -> dict[str, Any]:
        all_leads = self.leads()
        won = [r for r in all_leads if r.get("stage") == "won"]
        closed = [r for r in all_leads if r.get("stage") in ("won", "lost")]
        return {
            "total": len(all_leads),
            "per_stage": {s: sum(1 for r in all_leads if r.get("stage") == s)
                          for s in CRM_STAGES},
            "won": len(won),
            "lost": len(closed) - len(won),
            "win_rate": round(100 * len(won) / len(closed), 1) if closed else 0.0,
        }

    def funnel_report(self) -> str:
        summary = self.leads_summary()
        lines = [f"funnel: {_STAGE_COUNT}"]
        for stage in CRM_STAGES:
            lines.append(f"  {stage:<9} {summary['per_stage'][stage]}")
        lines.append(f"  won={summary['won']} lost={summary['lost']} "
                     f"win_rate={summary['win_rate']}%")
        return "\n".join(lines)

    # ---------------------------------------------------------------- status
    def status(self) -> str:
        credits = jsonl_store.replay_records(self.pipeline_path)
        drafts = sum(1 for r in credits.values()
                     if r.get("kind") == "content" and r.get("stage") == "draft")
        published = sum(1 for r in credits.values()
                        if r.get("kind") == "content" and r.get("stage") == "published")
        summary = self.leads_summary()
        return (f"business: content={len(credits)} (draft={drafts} published={published}) "
                f"leads={summary['total']} funnel={summary['per_stage']}")