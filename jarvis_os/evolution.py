"""SelfEvolution - the propose -> permission -> sandbox -> apply -> rollback engine.

Every phase can be independently enabled or disabled from ``config.json`` under
``evolution.phases``. Disabling any phase halts the pipeline at that point, so
the operator can, for example, let Jarvis propose and test changes but never
apply them.

Hard invariants
---------------
* No file on disk changes before ``request_permission`` returned **approved**
  from a live manual prompt.
* ``sandbox_test`` must pass (when enabled) before ``apply`` runs.
* Any failure in test or apply triggers automatic rollback.
* One cycle changes at most ``evolution.max_changes_per_cycle`` files.
* Every phase transition is written to the shared audit ledger.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PHASES = ("propose", "request_permission", "sandbox_test", "apply", "rollback")

PROMPT = """You are Jarvis's self-evolution engine. Analyse the runtime evidence and
propose at most {limit} concrete, minimal improvement(s) to this codebase.

Hard rules:
- Touch ONLY files under jarvis_os/.
- Never modify permissions.py or overseer.py (protected).
- Never modify, truncate or delete anything under data/ (memory is sacred).
- Changes must be small, surgical and syntactically valid Python.
- Never introduce a network call to any external AI or cloud API.

Return raw JSON only:
{{"insights": ["what the evidence shows"],
 "changes": [{{"file":"jarvis_os/tools.py","reason":"...",
              "old":"exact existing snippet","new":"replacement snippet"}}],
 "tests": [{{"name":"...","code":"python code asserting the fix"}}],
 "risk": "low|medium|high"}}"""


@dataclass(slots=True)
class ProposedChange:
    file: str
    reason: str = ""
    old: str = ""
    new: str = ""

    @property
    def key(self) -> str:
        return f"{self.file}::{hash((self.old, self.new)) & 0xFFFFFF:06x}"

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "reason": self.reason,
                "old": self.old[:200], "new": self.new[:200]}


@dataclass(slots=True)
class EvolutionCycle:
    cycle_id: str
    started_at: float = field(default_factory=time.time)
    phases_run: list[str] = field(default_factory=list)
    phases_skipped: dict[str, str] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    changes: list[ProposedChange] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    risk: str = "unknown"
    approved: bool | None = None
    test_results: list[dict[str, Any]] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    rolled_back: bool = False
    status: str = "running"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id, "status": self.status, "risk": self.risk,
            "started_at": self.started_at, "phases_run": self.phases_run,
            "phases_skipped": self.phases_skipped, "insights": self.insights,
            "changes": [c.to_dict() for c in self.changes], "tests": self.tests,
            "approved": self.approved, "test_results": self.test_results,
            "applied": self.applied, "rolled_back": self.rolled_back, "note": self.note,
        }


class SelfEvolution:
    PROTECTED = ("jarvis_os/permissions.py", "jarvis_os/overseer.py",
                 "jarvis_os/audit.py", "jarvis_os/memory.py")

    def __init__(self, config, llm, memory, gate, self_update, sandbox,
                 audit=None, root: str | Path | None = None) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.gate = gate
        self.self_update = self_update
        self.sandbox = sandbox
        self.audit = audit
        self.root = Path(root or self_update.root if self_update else Path(".")).resolve()
        self.cycles: list[EvolutionCycle] = []
        self._counter = 0

    # ---------------------------------------------------------------- phases
    def phase_enabled(self, name: str) -> bool:
        return bool(self.config.get(f"evolution.phases.{name}", True))

    def phases_report(self) -> str:
        rows = [f"  {name:<20} {'enabled' if self.phase_enabled(name) else 'DISABLED'}"
                for name in PHASES]
        return "Evolution phases:\n" + "\n".join(rows)

    # ----------------------------------------------------------------- cycle
    def run_cycle(self, focus: str = "", report_only: bool = False) -> EvolutionCycle:
        self._counter += 1
        cycle = EvolutionCycle(cycle_id=f"evo_{time.strftime('%Y%m%d-%H%M%S')}_{self._counter}")
        self.cycles.append(cycle)
        self.cycles = self.cycles[-40]

        self._log(cycle, "cycle_start", {"focus": focus, "dry_run": report_only})

        # --- 1. propose ---------------------------------------------------
        if not self.phase_enabled("propose"):
            return self._halt(cycle, "propose", "phase disabled in config")
        cycle.phases_run.append("propose")
        self._propose(cycle, focus)
        if not cycle.changes:
            return self._halt(cycle, "propose", "no viable change proposed")

        # --- 2. request_permission ---------------------------------------
        if not self.phase_enabled("request_permission"):
            return self._halt(cycle, "request_permission", "phase disabled in config")
        cycle.phases_run.append("request_permission")
        if report_only:
            cycle.phases_skipped["request_permission"] = "dry-run requested"
            cycle.status = "dry-run"
            cycle.note = self._describe(cycle)
            self._log(cycle, "dry_run", {"changes": len(cycle.changes)})
            return cycle
        cycle.approved = self._ask(cycle)
        if not cycle.approved:
            cycle.status = "denied"
            cycle.note = "Operator denied this evolution."
            self._log(cycle, "denied", {"changes": len(cycle.changes)})
            return cycle

        # --- 3. sandbox_test ---------------------------------------------
        if self.phase_enabled("sandbox_test"):
            cycle.phases_run.append("sandbox_test")
            ok = self._sandbox_test(cycle)
            if not ok:
                if self.config.get("evolution.auto_rollback_on_failure", True) and \
                        self.phase_enabled("rollback"):
                    cycle.rolled_back = True
                    self.self_update.rollback_to(None, "evolution sandbox test failed")
                cycle.status = "test_failed"
                cycle.note = "Sandbox test failed - change not applied."
                self._log(cycle, "test_failed", {"results": cycle.test_results})
                return cycle
        elif self.config.get("evolution.sandbox_test_required", True):
            return self._halt(cycle, "sandbox_test",
                              "tests required but the sandbox_test phase is disabled")
        else:
            cycle.phases_skipped["sandbox_test"] = "disabled and not required"

        # --- 4. apply ----------------------------------------------------
        if not self.phase_enabled("apply"):
            cycle.phases_skipped["apply"] = "phase disabled in config"
            cycle.status = "not_applied"
            cycle.note = "Change approved and tested but 'apply' is disabled."
            self._log(cycle, "not_applied", {})
            return cycle
        cycle.phases_run.append("apply")
        self._apply(cycle)
        cycle.status = "applied" if cycle.applied else "apply_failed"
        self._log(cycle, cycle.status, {"applied": cycle.applied})

        if cycle.applied and self.memory is not None:
            self.memory.learn_lesson(
                topic="self_evolution",
                lesson=(f"Evolution cycle {cycle.cycle_id} applied "
                        f"{len(cycle.applied)} change(s) with manual approval."),
                outcome="success", weight=0.8,
            )
        return cycle

    # -------------------------------------------------------------- internals
    def _halt(self, cycle: EvolutionCycle, phase: str, reason: str) -> EvolutionCycle:
        cycle.phases_skipped[phase] = reason
        cycle.status = "halted"
        cycle.note = reason
        self._log(cycle, "halted", {"phase": phase, "reason": reason})
        return cycle

    def _propose(self, cycle: EvolutionCycle, focus: str) -> None:
        limit = int(self.config.get("evolution.max_changes_per_cycle", 1))
        evidence = self._gather_evidence(focus)
        reply = self.llm.ask_json(
            PROMPT.format(limit=limit) + "\n\nRuntime evidence:\n" + evidence,
            persona="reviewer",
        )
        if not reply:
            cycle.note = "planner produced no parseable proposal"
            return
        cycle.insights = [str(i)[:300] for i in (reply.get("insights") or [])][:8]
        cycle.risk = str(reply.get("risk", "unknown"))
        raw_tests = reply.get("tests") or []
        cycle.tests = [t for t in raw_tests if isinstance(t, dict)][:3]
        for item in (reply.get("changes") or [])[:limit]:
            if not isinstance(item, dict):
                continue
            change = ProposedChange(
                file=str(item.get("file", "")).strip().replace("\\", "/"),
                reason=str(item.get("reason", ""))[:400],
                old=str(item.get("old", "")),
                new=str(item.get("new", "")),
            )
            verdict = self._validate(change)
            if verdict == "ok":
                cycle.changes.append(change)
            else:
                cycle.insights.append(f"rejected change to {change.file}: {verdict}")

    def _validate(self, change: ProposedChange) -> str:
        if not change.file:
            return "missing file"
        if not change.file.startswith("jarvis_os/"):
            return "outside the jarvis_os package"
        if change.file in self.PROTECTED:
            return "protected module"
        if change.file.startswith("data/") or "/data/" in change.file:
            return "memory paths are immutable"
        if not (self.root / change.file).is_file():
            return "file does not exist"
        if not change.old or not change.new:
            return "incomplete patch"
        if change.old == change.new:
            return "no-op patch"
        try:
            content = (self.root / change.file).read_text(encoding="utf-8")
        except OSError as exc:
            return f"unreadable: {exc}"
        if change.old not in content:
            return "patch context not present in file"
        try:
            ast.parse(content.replace(change.old, change.new, 1), filename=change.file)
        except SyntaxError as exc:
            return f"patched file would not parse: {exc.msg}"
        return "ok"

    def _gather_evidence(self, focus: str) -> str:
        lines: list[str] = []
        if focus:
            lines.append(f"Operator focus: {focus}")
        if self.memory is not None:
            for lesson in self.memory.lessons(limit=8):
                lines.append(f"lesson[{lesson.outcome}] {lesson.topic}: {lesson.lesson[:160]}")
            stats = self.memory.lesson_stats()
            lines.append(f"lesson stats: {json.dumps(stats)}")
            for action in self.memory.actions(limit=8):
                lines.append(f"action[{'ok' if action.success else 'fail'}] "
                             f"{action.kind}: {action.summary[:140]}")
        for entry in self._recent_audit():
            lines.append(f"audit[{entry['decision']}] {entry['action']}: "
                         f"{json.dumps(entry['detail'], ensure_ascii=False)[:140]}")
        return "\n".join(lines) or "(no evidence recorded yet)"

    def _recent_audit(self) -> list[dict[str, Any]]:
        if not self.audit:
            return []
        return [e for e in self.audit.tail(limit=25, decision="failed")] + \
               [e for e in self.audit.tail(limit=25, decision="denied")]

    def _ask(self, cycle: EvolutionCycle) -> bool:
        detail_lines = [
            "Self-evolution request (nothing has been modified yet):",
        ]
        for change in cycle.changes:
            detail_lines.append(f"  file: {change.file}")
            detail_lines.append(f"  why : {change.reason or 'unspecified'}")
            detail_lines.append(f"  risk: {cycle.risk}")
        if cycle.tests:
            detail_lines.append(f"  sandbox tests queued: {len(cycle.tests)}")
        return bool(self.gate.approve(
            "self_evolution", "\n".join(detail_lines),
            detail={"cycle": cycle.cycle_id,
                    "files": [c.file for c in cycle.changes],
                    "risk": cycle.risk},
        ))

    def _sandbox_test(self, cycle: EvolutionCycle) -> bool:
        if not cycle.tests:
            cycle.test_results.append({"name": "syntax", "ok": True,
                                       "detail": "no tests proposed; syntax validated"})
            return True
        passed = True
        for index, test in enumerate(cycle.tests):
            code = str(test.get("code", ""))
            name = str(test.get("name", f"test_{index}"))
            if not code.strip():
                continue
            result = self.sandbox.run(code, timeout=float(
                self.config.get("evolution.sandbox_timeout_seconds", 30)),
                require_permission=False)
            entry = {"name": name, "ok": result.ok, "returncode": result.returncode,
                     "stdout": result.stdout[-600:], "stderr": result.stderr[-600:]}
            cycle.test_results.append(entry)
            passed = passed and result.ok
        return passed

    def _apply(self, cycle: EvolutionCycle) -> None:
        for change in cycle.changes:
            result = self.self_update.write_file(
                change.file,
                (self.root / change.file).read_text(encoding="utf-8").replace(
                    change.old, change.new, 1),
                require_permission=False,  # already approved at phase 2
            )
            if result.ok:
                cycle.applied.append(change.file)
            else:
                cycle.note = result.message
                if self.config.get("evolution.auto_rollback_on_failure", True) and \
                        self.phase_enabled("rollback"):
                    cycle.rolled_back = True
                    self.self_update.rollback_to(None, f"apply failed: {result.message}")
                return

    # --------------------------------------------------------------- reporting
    def _describe(self, cycle: EvolutionCycle) -> str:
        lines = [f"Evolution cycle {cycle.cycle_id} ({cycle.status}) - risk={cycle.risk}"]
        for insight in cycle.insights[:6]:
            lines.append(f"  insight: {insight}")
        for change in cycle.changes:
            lines.append(f"  change : {change.file} - {change.reason}")
        for entry in cycle.test_results:
            lines.append(f"  test   : {entry['name']} -> "
                         f"{'PASS' if entry.get('ok') else 'FAIL'}")
        if cycle.phases_skipped:
            for phase, reason in cycle.phases_skipped.items():
                lines.append(f"  skipped: {phase} ({reason})")
        if cycle.note:
            lines.append(f"  note   : {cycle.note}")
        return "\n".join(lines)

    def report(self) -> str:
        if not self.cycles:
            return "No evolution cycle has run yet."
        return "\n\n".join(self._describe(c) for c in reversed(self.cycles[-5:]))

    def last(self) -> EvolutionCycle | None:
        return self.cycles[-1] if self.cycles else None

    def rollback(self, cycle_id: str | None = None) -> str:
        cycle = (self._find(cycle_id) if cycle_id else self.last())
        if cycle is None:
            return "No matching evolution cycle to roll back."
        result = self.self_update.rollback_to(None, f"rollback of {cycle.cycle_id}")
        cycle.rolled_back = True
        cycle.status = "rolled_back"
        return result.message

    def _find(self, cycle_id: str) -> EvolutionCycle | None:
        for cycle in reversed(self.cycles):
            if cycle.cycle_id == cycle_id:
                return cycle
        return None

    def _log(self, cycle: EvolutionCycle, action: str, detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record("evolution", action, "ok",
                              {"cycle": cycle.cycle_id, **detail})
        if self.memory is not None and action in ("test_failed", "denied", "halted"):
            self.memory.learn_lesson(
                topic="self_evolution",
                lesson=f"cycle {cycle.cycle_id} {action}: "
                       f"{detail.get('reason', detail.get('note', ''))}"[:200],
                outcome="failure", weight=0.6,
            )