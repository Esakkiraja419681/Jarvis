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
    # Every tool the plan asked for, and every tool that actually ran. Kept apart
    # because "no subtask crashed" and "the goal got done" are different claims:
    # a plan that named tools but never invoked one has verified nothing, and
    # reporting that as success is how a greeting ends up half-answered while the
    # console says Done.
    planned_tools: list[str] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)

    @property
    def unexecuted_tools(self) -> list[str]:
        """Tools the plan promised that no subtask actually invoked.

        Derived rather than stored: a caller that builds a result by hand must
        not be able to forget to fill it in and thereby declare a half-run plan
        complete.
        """
        return sorted(set(self.planned_tools) - set(self.used_tools))

    @property
    def partial(self) -> bool:
        """True when the plan promised tools and none of them ran."""
        return bool(self.unexecuted_tools)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "ok": self.ok, "error": self.error,
            "output": self.output, "strategy": self.strategy,
            "duration_ms": self.duration_ms,
            "tasks": [t.to_dict() for t in self.tasks],
            "planned_tools": self.planned_tools,
            "used_tools": self.used_tools,
            "unexecuted_tools": self.unexecuted_tools,
            "partial": self.partial,
        }


class TaskManager:
    def __init__(self, agents: "AgentRegistry", memory: Any, llm: Any,
                 config: Any, audit: Any, tools: Any = None,
                 max_subtasks: int = 8) -> None:
        self.agents = agents
        self.memory = memory
        self.llm = llm
        self.config = config
        self.audit = audit
        # The registry of tools that actually exist. The planner prompt has to
        # name real tools: telling a 4B model to pick from a menu it cannot check
        # invites it to invent one, and inventing web_fetch for "say hello" is
        # exactly the failure this whole path exists to prevent.
        self.tools = tools
        self.max_subtasks = max_subtasks
        self.history: list[DelegationResult] = []
        self._active: dict[str, AgentTask] = {}
        self.cycles = 0

    # ---------------------------------------------------------- decompose
    #
    # Goals the model can answer from its own weights. Planning a tool for these
    # can only produce a worse result: "say hello in Tamil and English" was
    # planned as two web_fetch calls, the fetch ran against a search engine for
    # the word "hello", the answer came back in neither language, and the run
    # still reported success. Listing them in the prompt is worth more than any
    # amount of post-hoc filtering, because a 4B planner reliably copies the tool
    # out of the example verbatim.
    SELF_CONTAINED_MARKERS = (
        "hello", "hii", "hey", "hi ", "greetings", "greet", "how are you",
        "who are you", "thank", "thanks", "please", "translate", "translation",
        "meaning of", "in tamil", "in english", "in spanish", "in french",
        "in hindi", "in telugu", "say hi", "goodbye", "bye",
    )
    TRANSLATION_MARKERS = ("translate", "translation", "in tamil", "in english",
                           "in spanish", "in french", "in hindi", "in telugu")

    @classmethod
    def _is_self_contained(cls, goal: str) -> bool:
        """True when no tool is needed to answer ``goal``.

        A conjunction still counts as self-contained when every clause is: "say
        hello in Tamil and English" is two renderings of one sentence the model
        can already write. What disqualifies it is a verb that needs the outside
        world - fetch, install, read, search the disk.
        """
        text = (goal or "").lower()
        if not text.strip():
            return False
        for clause in re.split(r"[,;]| and then | then | also ", text):
            clause = clause.strip()
            if not clause:
                continue
            if not any(marker in clause for marker in cls.SELF_CONTAINED_MARKERS):
                return False
        return True

    @classmethod
    def _strip_bogus_tools(cls, tasks: list[AgentTask], goal: str) -> list[AgentTask]:
        """Drop tools from a plan for a goal that needs none."""
        if not cls._is_self_contained(goal):
            return tasks
        for task in tasks:
            task.tools = []
        return tasks

    def decompose(self, goal: str) -> tuple[list[AgentTask], str]:
        # No tool named in the example. The old one advertised web_fetch, and
        # the planner treated the example as a template to fill in rather than a
        # shape to imitate - so every goal it was shown inherited web_fetch.
        # Naming no tool also makes "leave tools empty unless genuinely needed"
        # the path of least resistance.
        tools_hint = ""
        if self.tools is not None and self.tools.names():
            tools_hint = (f"Tools that genuinely exist and may be useful: "
                          f"{', '.join(sorted(self.tools.names()))}.\n")
        prompt = (
            f"Goal from the operator ({self.config.honorific()}):\n{goal}\n\n"
            f"Available agent roles: {', '.join(sorted(ROLE_KEYWORDS))}.\n"
            f"{tools_hint}"
            "Return JSON: {\"subtasks\": [{\"id\":\"s1\",\"agent\":\"researcher\","
            "\"objective\":\"...\",\"tools\":[],\"depends_on\":[]}]}\n"
            "Rules:\n"
            "- Use the fewest subtasks that can finish the goal.\n"
            "- Put a tool in \"tools\" ONLY if that subtask must read, write, "
            "fetch or execute something. Leave \"tools\": [] otherwise.\n"
            "- Greetings, introductions, thanks, farewells and translation requests "
            "are answered by the model itself. They need no tool, and web_fetch "
            "cannot help with them."
        )
        parsed = self.llm.ask_json(prompt, persona="planner") if self.llm else None
        tasks = self._parse_subtasks(parsed, goal)
        if tasks:
            return self._strip_bogus_tools(tasks, goal), "llm"
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
        used_tools: set[str] = set()
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
            # Tool bookkeeping is gathered whatever the outcome: a tool that ran
            # before the agent then failed still counts as having run, and only
            # the ones that never appear anywhere can be called unexecuted.
            for tool in result.get("tools_used", []) or []:
                used_tools.add(str(tool))
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
        planned = sorted({str(t) for task in tasks for t in task.tools})
        result = DelegationResult(
            goal=goal, tasks=tasks, output=output, ok=not failed, strategy=strategy,
            duration_ms=duration,
            planned_tools=planned,
            used_tools=sorted(used_tools),
        )
        if result.partial and result.ok:
            # Distinct from a failure: nothing crashed, but the plan's own
            # commitments went unverified. Recorded as its own audit status so
            # "the model answered instead of fetching" never again reads as a
            # clean success.
            if self.audit:
                self.audit.record("task", "execute", "partial",
                                  {"goal": goal[:200],
                                   "unexecuted_tools": result.unexecuted_tools})
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