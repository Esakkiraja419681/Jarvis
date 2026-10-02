"""TaskManager - goal decomposition and cross-agent delegation.

The flow for any goal:

1. **Decompose** - the planner LLM returns an ordered list of subtasks. If Ollama
   is unavailable, a deterministic heuristic planner takes over so the OS never
   hard-fails.
2. **Delegate** - each subtask is routed to the best-matching agent by role
   affinity (explicit ``agent`` field wins, else role keyword match).
3. **Execute** - respecting ``depends_on`` as a DAG; a subtask only runs once its
   dependencies succeed.
4. **Synthesise** - the orchestrator folds the subtask outputs into one answer.

State transitions (``pending -> ready -> running -> done|failed|skipped``) are
audited so the Overseer can spot a stalled or looping planner.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .agents import Agent, AgentTask

STATUSES = ("pending", "ready", "running", "done", "failed", "skipped", "blocked")

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "researcher": ("find", "search", "research", "gather", "collect", "look up",
                   "investigate", "compare", "fetch", "web", "who", "what is"),
    "coder": ("write", "implement", "code", "script", "build", "create", "program",
              "fix", "refactor", "generate code"),
    "reviewer": ("review", "check", "verify", "validate", "audit", "critique",
                 "find bugs", "test"),
    "operator": ("install", "launch", "start", "stop", "restart", "deploy",
                 "configure", "control", "monitor"),
    "planner": ("plan", "schedule", "organise", "organize", "sequence", "roadmap",
                "strategy"),
    "teacher": ("explain", "teach", "how do", "why", "tutorial", "guide", "learn"),
    "guardian": ("delete", "remove", "security", "safe", "risk", "danger", "unsafe"),
}


@dataclass(slots=True)
class DelegationResult:
    goal: str
    tasks: list[AgentTask] = field(default_factory=list)
    output: str = ""
    ok: bool = True
    error: str = ""
    strategy: str = "llm"
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "ok": self.ok, "error": self.error,
            "output": self.output, "strategy": self.strategy,
            "duration_ms": self.duration_ms,
            "tasks": [t.to_dict() for t in self.tasks],
        }


class TaskManager:
    def __init__(self, agents: "AgentRegistry", memory: Any, llm: Any,
                 config: Any, audit: Any, max_subtasks: int = 8) -> None:
        self.agents = agents
        self.memory = memory
        self.llm = llm
        self.config = config
        self.audit = audit
        self.max_subtasks = max_subtasks
        self.history: list[DelegationResult] = []
        self._active: dict[str, AgentTask] = {}
        self.cycles = 0

    # ---------------------------------------------------------- decompose
    def decompose(self, goal: str) -> tuple[list[AgentTask], str]:
        prompt = (
            f"Goal from the operator ({self.config.honorific()}):\n{goal}\n\n"
            f"Available agent roles: {', '.join(sorted(ROLE_KEYWORDS))}.\n"
            "Return JSON: {\"subtasks\": [{\"id\":\"s1\",\"agent\":\"researcher\","
            "\"objective\":\"...\",\"tools\":[\"web_fetch\"],\"depends_on\":[]}]}"
        )
        parsed = self.llm.ask_json(prompt, persona="planner") if self.llm else None
        tasks = self._parse_subtasks(parsed, goal)
        if tasks:
            return tasks, "llm"
        return self._heuristic_subtasks(goal), "heuristic"

    def _parse_subtasks(self, parsed: dict[str, Any] | None, goal: str) -> list[AgentTask]:
        if not parsed:
            return []
        raw = parsed.get("subtasks") or parsed.get("tasks") or parsed.get("plan") or []
        if not isinstance(raw, list):
            return []
        tasks: list[AgentTask] = []
        id_map: dict[str, str] = {}
        for index, item in enumerate(raw[: self.max_subtasks]):
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective") or item.get("description")
                            or item.get("task") or "").strip()
            if not objective:
                continue
            task_id = f"t{uuid.uuid4().hex[:6]}"
            id_map[str(item.get("id") or index)] = task_id
            deps = [str(d) for d in (item.get("depends_on") or item.get("deps") or [])]
            tasks.append(AgentTask(
                objective=objective,
                assigned_agent=str(item.get("agent") or "orchestrator"),
                tools=[str(t) for t in (item.get("tools") or [])],
                depends_on=deps,
                task_id=task_id,
            ))
        for task in tasks:
            task.depends_on = [id_map.get(d, d) for d in task.depends_on]
        if not tasks:
            return []
        # Preserve any explicit final goal step.
        if len(tasks) == 1 and len(goal) > 40:
            tasks.append(AgentTask(
                objective=f"Synthesise a single clear answer for: {goal}",
                assigned_agent="orchestrator", task_id=f"t{uuid.uuid4().hex[:6]}",
                depends_on=[tasks[0].task_id],
            ))
        return tasks

    def _heuristic_subtasks(self, goal: str) -> list[AgentTask]:
        """Deterministic fallback so a missing Ollama never stalls the OS."""
        text = goal.lower()
        tasks: list[AgentTask] = []
        needs_research = any(k in text for k in
                             ("find", "search", "what", "who", "research", "latest",
                              "news", "compare", "investigate"))
        needs_code = any(k in text for k in
                         ("code", "write", "script", "program", "implement", "build",
                          "automate"))
        needs_run = any(k in text for k in
                        ("run", "install", "launch", "start", "execute", "deploy"))
        if needs_research:
            tasks.append(AgentTask("Research the goal using memory and web tools.",
                                   "researcher", ["memory_recall", "web_fetch"]))
        if needs_code:
            prev = tasks[-1].task_id if tasks else []
            tasks.append(AgentTask("Produce working code or configuration for the goal.",
                                   "coder", ["file_write", "code_runner"], prev))
        if needs_run:
            prev = tasks[-1].task_id if tasks else []
            tasks.append(AgentTask("Run/launch the result and verify it works.",
                                   "operator", ["shell_exec"], prev))
        if not tasks:
            tasks.append(AgentTask(goal, "orchestrator", []))
        final_id = f"t{uuid.uuid4().hex[:6]}"
        tasks.append(AgentTask(
            objective=f"Deliver a single, clear answer to the operator for: {goal}",
            assigned_agent="orchestrator", task_id=final_id,
            depends_on=[t.task_id for t in tasks],
        ))
        return tasks

    # -------------------------------------------------------------- routing
    def resolve_agent(self, requested: str, objective: str) -> Agent | None:
        if agent := self.agents.get(requested):
            return agent
        lowered = objective.lower()
        best, best_score = None, 0
        for role, keywords in ROLE_KEYWORDS.items():
            score = sum(1 for k in keywords if k in lowered)
            if score > best_score:
                for candidate in self.agents.find_by_role(role):
                    best, best_score = candidate, score
        if best is not None:
            return best
        return self.agents.get("orchestrator")

    # ------------------------------------------------------------ execution
    def execute(self, goal: str, context: str = "",
                allow_tools: bool = True) -> DelegationResult:
        started = time.time()
        self.cycles += 1
        tasks, strategy = self.decompose(goal)
        for task in tasks:
            self._active[task.task_id] = task
        if self.audit:
            self.audit.record("task", "decompose", "ok",
                              {"goal": goal[:200], "subtasks": len(tasks), "strategy": strategy})

        completed: dict[str, str] = {}
        failed = False
        for task in tasks:
            blockers = [d for d in task.depends_on
                        if d in {t.task_id for t in tasks}
                        and d not in completed]
            if blockers:
                task.status = "blocked"
                if self.audit:
                    self.audit.record("task", f"skip:{task.task_id}", "blocked",
                                      {"waiting_on": blockers})
                continue
            task.status = "running"
            task.started_at = time.time()
            agent = self.resolve_agent(task.assigned_agent, task.objective)
            if agent is None:
                task.status, task.error = "failed", "no agent available"
                failed = True
                continue
            scoped = context or self._context_from(completed)
            result = agent.run(
                task.objective,
                context=scoped,
                allow_tools=allow_tools,
            )
            task.finished_at = time.time()
            if result.get("ok"):
                task.status = "done"
                task.result = result.get("output", "")
                completed[task.task_id] = str(result.get("output", ""))
            else:
                task.status = "failed"
                task.error = str(result.get("error", "unknown"))
                failed = True
            if self.audit:
                self.audit.record("task", f"{task.status}:{task.task_id}",
                                  "ok" if task.status == "done" else "failed",
                                  {"agent": agent.name, "objective": task.objective[:160]})

        output = self._synthesise(goal, completed)
        duration = int((time.time() - started) * 1000)
        result = DelegationResult(
            goal=goal, tasks=tasks, output=output, ok=not failed, strategy=strategy,
            duration_ms=duration,
        )
        if not result.ok and not output:
            result.error = "all delegated subtasks failed"
        self.history.append(result)
        self.history = self.history[-100:]
        for task in tasks:
            self._active.pop(task.task_id, None)
        if self.memory is not None:
            self.memory.record_episode(goal, result.to_dict(), output[:500], result.ok)
        return result

    def _context_from(self, completed: dict[str, str]) -> str:
        if not completed:
            return ""
        parts = [f"Completed work so far:\n{k[-6:]}: {v}" for k, v in completed.items()]
        return "\n\n".join(parts)

    def _synthesise(self, goal: str, completed: dict[str, str]) -> str:
        if not completed:
            return ""
        if len(completed) == 1:
            return next(iter(completed.values()))
        body = "\n\n".join(f"[{k[-6:]}] {v}" for k, v in completed.items())
        prompt = (
            f"Operator goal: {goal}\n\nAgent outputs:\n{body}\n\n"
            f"Write one final answer for the operator ({self.config.honorific()}). "
            "Be concise, concrete, and do not invent facts."
        )
        reply = self.llm.chat(prompt, persona="orchestrator", fallback="")
        if reply.ok and reply.text.strip():
            return reply.text.strip()
        return body

    # ---------------------------------------------------------------- queue
    def submit(self, goal: str, context: str = "") -> DelegationResult:
        """Alias for execute() that reads naturally from the kernel."""
        return self.execute(goal, context=context)

    def pending(self) -> list[AgentTask]:
        return [t for t in self._active.values() if t.status in ("pending", "ready", "running")]

    def status(self) -> str:
        active = self.pending()
        if not active:
            return (f"TaskManager idle | goals processed={self.cycles} "
                    f"| history={len(self.history)}")
        lines = [f"TaskManager: {len(active)} active task(s), cycles={self.cycles}"]
        for task in active[:12]:
            lines.append(f"  [{task.status:>7}] {task.objective[:60]}")
        return "\n".join(lines)

    def last(self) -> DelegationResult | None:
        return self.history[-1] if self.history else None

    def stats(self) -> dict[str, Any]:
        total = sum(t for t in
                    (a.stats["tasks"] for a in self.agents.all()))
        ok = sum(a.stats["success"] for a in self.agents.all())
        return {"goals_processed": self.cycles, "agent_tasks": total,
                "agent_success": ok, "history": len(self.history)}

    def clear_history(self) -> int:
        count = len(self.history)
        self.history.clear()
        return count

    def reset_active(self, reason: str = "manual") -> int:
        """Overseer hook: break a stalled run without touching memory."""
        count = 0
        for task in list(self._active.values()):
            if task.status in ("running", "ready", "pending"):
                task.status = "skipped"
                task.error = f"reset: {reason}"
                count += 1
        self._active.clear()
        if count and self.audit:
            self.audit.record("task", "reset_active", "ok", {"count": count, "reason": reason})
        return count