"""GoalPlanner - autonomous goal decomposition and dynamic DAG execution.

The planner turns one operator goal into a directed acyclic graph of steps
and executes it, respecting dependencies, in three deliberate stages:

1. **Decompose** - the LLM returns steps with ids and ``depends_on`` edges.
   When Ollama is unavailable a deterministic heuristic planner takes over so
   the OS never hard-fails. Decompose *sanitises* the model's output: ids are
   remapped, dangling dependencies dropped, the step budget enforced.
2. **Validate** - Kahn's algorithm proves the graph is acyclic, every
   dependency resolves, ids are unique. An invalid plan refuses to run.
3. **Execute** - a dynamic loop that recomputes the ready set each tick, so a
   runner may append steps while the plan is running, and any failure blocks
   its dependents transitively.

Hazard classification is deterministic keyword matching over the step
objective. The LLM may *add* a hazard category it recognises, never remove
one the keywords found - a model that calls a deletion "maintenance" still
hits the PermissionGate. Hazard steps stop for a live manual approval before
the runner ever sees them; a denial marks the step denied and everything
downstream blocked.

Plan state is persisted (``data/plan_state.json``) so the Overseer can verify
state integrity after a crash.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

STEP_STATUSES = ("pending", "ready", "running", "done", "failed",
                 "denied", "blocked", "skipped")
BLOCKING_STATUSES = frozenset({"failed", "denied", "blocked", "skipped"})

HAZARD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "file_delete": ("delete", "remove", "wipe", "format", "erase", "purge",
                    "drop table", "clear out", "trash"),
    "software_install": ("install", "uninstall", "pip install", "apt install",
                         "download and set up"),
    "shell_exec": ("run ", "exec", "shell", "command", "execute", "script"),
    "software_control": ("restart", "stop ", "kill", "reboot", "shutdown",
                         "halt"),
    "cross_device": ("cross-device", "cross device", "other device",
                     "another device", "remote device", "paired device"),
    "self_update": ("self-modify", "self modify", "self-update", "self update",
                    "self-improve", "self improve", "evolve", "upgrade jarvis",
                    "update jarvis"),
}

STATUS_MARKERS = {
    "pending": "[ ]", "ready": "[ ]", "running": "[>]", "done": "[x]",
    "failed": "[F]", "denied": "[D]", "blocked": "[B]", "skipped": "[-]",
}


@dataclass(slots=True)
class PlanStep:
    step_id: str
    objective: str
    depends_on: list[str] = field(default_factory=list)
    hazard: str = ""
    status: str = "pending"
    result: str = ""
    error: str = ""
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.step_id, "objective": self.objective,
            "depends_on": list(self.depends_on), "hazard": self.hazard,
            "status": self.status, "result": self.result, "error": self.error,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        return cls(
            step_id=str(data.get("id") or data.get("step_id") or ""),
            objective=str(data.get("objective", "")),
            depends_on=[str(d) for d in (data.get("depends_on") or [])],
            hazard=str(data.get("hazard", "")),
            status=str(data.get("status", "pending")),
            result=str(data.get("result", "")),
            error=str(data.get("error", "")),
            source=str(data.get("source", "llm")),
        )


@dataclass(slots=True)
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    strategy: str = "heuristic"
    created_at: float = field(default_factory=time.time)

    def add(self, objective: str, depends_on: list[str] | None = None,
            hazard: str = "", source: str = "runner") -> PlanStep:
        """Append a step at runtime; the executor picks it up next tick."""
        step = PlanStep(
            step_id=f"s{uuid.uuid4().hex[:6]}",
            objective=str(objective),
            depends_on=list(depends_on or []),
            hazard=hazard, source=source,
        )
        self.steps.append(step)
        return step

    def get(self, step_id: str) -> PlanStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def ids(self) -> list[str]:
        return [s.step_id for s in self.steps]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for step in self.steps:
            out[step.status] = out.get(step.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "strategy": self.strategy,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        raw = data.get("steps")
        steps = [PlanStep.from_dict(s) for s in raw if isinstance(s, dict)] \
            if isinstance(raw, list) else []
        return cls(
            goal=str(data.get("goal", "")),
            steps=steps,
            strategy=str(data.get("strategy", "heuristic")),
            created_at=float(data.get("created_at", time.time()) or time.time()),
        )


@dataclass(slots=True)
class ExecutionResult:
    goal: str
    strategy: str = "heuristic"
    steps_done: int = 0
    steps_failed: int = 0
    steps_denied: int = 0
    steps_blocked: int = 0
    steps_skipped: int = 0
    ok: bool = False
    error: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "ok": self.ok, "error": self.error,
            "strategy": self.strategy, "duration_ms": self.duration_ms,
            "steps_done": self.steps_done, "steps_failed": self.steps_failed,
            "steps_denied": self.steps_denied,
            "steps_blocked": self.steps_blocked,
            "steps_skipped": self.steps_skipped,
            "outputs": self.outputs,
        }


Runner = Callable[[PlanStep], "str | dict[str, Any] | None"]


class GoalPlanner:
    """Decompose, validate and execute goals as gated, dynamic DAGs."""

    def __init__(self, config, llm=None, gate=None, audit=None,
                 memory=None) -> None:
        self.config = config
        self.llm = llm
        self.gate = gate
        self.audit = audit
        self.memory = memory
        self.max_steps = int(config.get("planner.max_steps", 12))
        self.state_path = config.resolve(
            config.get("planner.state_path", "data/plan_state.json"))
        self.history: list[ExecutionResult] = []
        self.plans = 0

    # -------------------------------------------------------------- hazards
    @classmethod
    def classify_hazard(cls, objective: str, llm_claim: str = "") -> str:
        """Permission category for a step, keywords first, LLM only additive."""
        text = (objective or "").lower()
        for category, keywords in HAZARD_KEYWORDS.items():
            if any(k in text for k in keywords):
                return category
        claim = str(llm_claim or "").strip()
        if claim and claim in HAZARD_KEYWORDS:
            return claim
        return ""

    # ------------------------------------------------------------ decompose
    def decompose(self, goal: str) -> tuple[Plan, str]:
        prompt = (
            f"Goal from the operator ({self.config.honorific()}):\n{goal}\n\n"
            "Return JSON: {\"steps\": [{\"id\":\"s1\",\"objective\":\"...\","
            "\"depends_on\":[],\"hazard\":\"\"}]}\n"
            "Rules:\n"
            "- Use the fewest steps that can finish the goal.\n"
            "- depends_on may only reference ids that appear above.\n"
            "- Steps with no dependencies can run in parallel.\n"
            "- hazard is ONE permission category - file_delete, shell_exec, "
            "software_install, software_control, cross_device, self_update - "
            "set ONLY when the step deletes files, runs shell commands, "
            "installs or controls software, touches another device, or "
            "modifies Jarvis itself. Otherwise leave it empty."
        )
        parsed = self.llm.ask_json(prompt, persona="planner") if self.llm else None
        steps = self._parse_steps(parsed, goal)
        strategy = "llm" if steps else "heuristic"
        if not steps:
            steps = self._heuristic_steps(goal)
        plan = Plan(goal=goal, steps=steps, strategy=strategy)
        self.plans += 1
        if self.audit:
            self.audit.record("plan", "decompose", "ok",
                              {"goal": goal[:200], "steps": len(steps),
                               "strategy": strategy})
        return plan, strategy

    def _parse_steps(self, parsed: dict[str, Any] | None,
                     goal: str) -> list[PlanStep]:
        if not parsed:
            return []
        raw = parsed.get("steps") or parsed.get("subtasks") \
            or parsed.get("tasks") or parsed.get("plan") or []
        if not isinstance(raw, list):
            return []
        items: list[dict[str, Any]] = []
        id_map: dict[str, str] = {}
        for index, item in enumerate(raw[: self.max_steps]):
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective") or item.get("description")
                            or item.get("task") or "").strip()
            if not objective:
                continue
            new_id = f"s{uuid.uuid4().hex[:6]}"
            id_map[str(item.get("id") or index)] = new_id
            items.append({**item, "_objective": objective, "_id": new_id})
        steps: list[PlanStep] = []
        for item in items:
            deps = [str(d) for d in (item.get("depends_on") or item.get("deps") or [])]
            hazard = self.classify_hazard(
                item["_objective"], str(item.get("hazard") or ""))
            steps.append(PlanStep(
                step_id=item["_id"],
                objective=item["_objective"],
                depends_on=[id_map[d] for d in deps if d in id_map],
                hazard=hazard, source="llm",
            ))
        if not steps:
            return []
        if len(steps) == 1 and len(goal) > 40:
            steps.append(PlanStep(
                step_id=f"s{uuid.uuid4().hex[:6]}",
                objective=f"Deliver one clear answer to the operator for: {goal}",
                depends_on=[steps[0].step_id], source="llm",
            ))
        return steps

    def _heuristic_steps(self, goal: str) -> list[PlanStep]:
        """Deterministic fallback so a missing Ollama never stalls the OS."""
        text = goal.lower()
        steps: list[PlanStep] = []
        if any(k in text for k in ("find", "search", "what", "who", "research",
                                   "latest", "news", "compare", "investigate")):
            steps.append(PlanStep(
                step_id=f"s{uuid.uuid4().hex[:6]}",
                objective="Research the goal using memory and local tools.",
                depends_on=[], source="runner"))
        if any(k in text for k in ("code", "write", "script", "program",
                                   "implement", "build", "automate")):
            deps = [steps[-1].step_id] if steps else []
            steps.append(PlanStep(
                step_id=f"s{uuid.uuid4().hex[:6]}",
                objective="Produce working code or configuration for the goal.",
                depends_on=deps, source="runner"))
        if any(k in text for k in ("run", "install", "launch", "execute",
                                   "deploy")):
            deps = [steps[-1].step_id] if steps else []
            steps.append(PlanStep(
                step_id=f"s{uuid.uuid4().hex[:6]}",
                objective="Run the result and verify it works.",
                depends_on=deps, source="runner"))
        if not steps:
            steps.append(PlanStep(
                step_id=f"s{uuid.uuid4().hex[:6]}",
                objective=goal, depends_on=[], source="runner"))
        steps.append(PlanStep(
            step_id=f"s{uuid.uuid4().hex[:6]}",
            objective="Deliver the outcome of this goal to the operator.",
            depends_on=[s.step_id for s in steps], source="runner",
        ))
        # The fallback path must not be a gate bypass: keywords are checked
        # against every heuristic objective exactly as they are for LLM steps.
        for step in steps:
            step.hazard = self.classify_hazard(step.objective)
        return steps

    # ------------------------------------------------------------- validate
    def validate(self, plan: Plan) -> tuple[bool, str]:
        """Prove the DAG is runnable: unique ids, resolvable deps, no cycle."""
        if not plan.steps:
            return False, "plan has no steps"
        ids = [s.step_id for s in plan.steps]
        if any(not i for i in ids):
            return False, "every step needs an id"
        if len(ids) != len(set(ids)):
            return False, "duplicate step ids"
        known = set(ids)
        for step in plan.steps:
            if step.step_id in step.depends_on:
                return False, f"step {step.step_id} depends on itself"
            for dep in step.depends_on:
                if dep not in known:
                    return False, (f"step {step.step_id} depends on unknown "
                                   f"step '{dep}'")
        indegree = {s.step_id: len(set(s.depends_on)) for s in plan.steps}
        ready = [i for i, d in indegree.items() if d == 0]
        seen = 0
        while ready:
            current = ready.pop()
            seen += 1
            for step in plan.steps:
                if current in step.depends_on:
                    indegree[step.step_id] -= 1
                    if indegree[step.step_id] == 0:
                        ready.append(step.step_id)
        if seen != len(plan.steps):
            return False, "dependency cycle detected"
        return True, ""

    # ------------------------------------------------------------- execution
    def execute(self, plan: Plan, runner: Runner | None = None,
                max_steps: int | None = None) -> ExecutionResult:
        started = time.time()
        budget = int(max_steps or self.max_steps)
        ok, reason = self.validate(plan)
        if not ok:
            if self.audit:
                self.audit.record("plan", "validate", "failed",
                                  {"goal": plan.goal[:200], "reason": reason})
            return ExecutionResult(goal=plan.goal, strategy=plan.strategy,
                                   ok=False, error=f"invalid plan: {reason}")
        if runner is None:
            return ExecutionResult(goal=plan.goal, strategy=plan.strategy,
                                   ok=False, error="no runner bound")

        outputs: dict[str, str] = {}
        executed = 0
        while True:
            self._propagate_blocks(plan)
            if executed >= budget:
                for step in plan.steps:
                    if step.status == "pending":
                        step.status = "skipped"
                        step.error = f"step budget {budget} exhausted"
                        self._audit_step(plan, step)
                if self.audit:
                    self.audit.record("plan", "budget", "skipped",
                                      {"goal": plan.goal[:200],
                                       "budget": budget})
                break
            ready = self._ready(plan)
            if not ready:
                for step in plan.steps:
                    if step.status == "pending":
                        step.status = "blocked"
                        step.error = "blocked: dependency never completed"
                        self._audit_step(plan, step)
                break
            step = ready[0]
            step.status = "running"
            if step.hazard and not self._approve(plan, step):
                step.status = "denied"
                step.error = "operator denied this step"
                self._audit_step(plan, step)
                continue
            executed += 1
            self._run_step(plan, step, runner, outputs)

        self.save_state(plan)
        counts = plan.counts()
        errors = [s.error for s in plan.steps
                  if s.status in BLOCKING_STATUSES and s.error]
        result = ExecutionResult(
            goal=plan.goal, strategy=plan.strategy,
            steps_done=counts.get("done", 0),
            steps_failed=counts.get("failed", 0),
            steps_denied=counts.get("denied", 0),
            steps_blocked=counts.get("blocked", 0),
            steps_skipped=counts.get("skipped", 0),
            ok=(counts.get("failed", 0) == 0
                and counts.get("denied", 0) == 0
                and counts.get("blocked", 0) == 0
                and counts.get("skipped", 0) == 0),
            error=errors[0] if errors else "",
            outputs=outputs,
            duration_ms=int((time.time() - started) * 1000),
        )
        self.history.append(result)
        self.history = self.history[-100:]
        if self.audit:
            self.audit.record("plan", "execute",
                              "ok" if result.ok else "failed",
                              {"goal": plan.goal[:200],
                               "strategy": result.strategy,
                               "done": result.steps_done,
                               "failed": result.steps_failed,
                               "denied": result.steps_denied,
                               "blocked": result.steps_blocked,
                               "ms": result.duration_ms})
        if self.memory is not None:
            outcome = result.error or ", ".join(result.outputs.values())[:400]
            self.memory.record_episode(plan.goal, result.to_dict(),
                                       outcome, result.ok)
        return result

    def _ready(self, plan: Plan) -> list[PlanStep]:
        ready: list[PlanStep] = []
        for step in plan.steps:
            if step.status != "pending":
                continue
            deps = [plan.get(d) for d in step.depends_on]
            if all(d is not None and d.status == "done" for d in deps):
                ready.append(step)
        return ready

    def _propagate_blocks(self, plan: Plan) -> None:
        """A failed, denied or blocked dependency blocks everything downstream."""
        changed = True
        while changed:
            changed = False
            for step in plan.steps:
                if step.status != "pending":
                    continue
                deps = [plan.get(d) for d in step.depends_on]
                bad = [d.step_id if d else "?" for d in deps
                       if d is None or d.status in BLOCKING_STATUSES]
                if bad:
                    step.status = "blocked"
                    step.error = f"blocked by: {', '.join(bad)}"
                    self._audit_step(plan, step)
                    changed = True

    def _run_step(self, plan: Plan, step: PlanStep, runner: Runner,
                  outputs: dict[str, str]) -> None:
        try:
            raw = runner(step)
        except Exception as exc:  # noqa: BLE001 - one bad step must not kill the run
            step.status = "failed"
            step.error = f"{type(exc).__name__}: {exc}"[:300]
            self._audit_step(plan, step)
            return
        if isinstance(raw, str):
            step.status, step.result = "done", raw
            outputs[step.step_id] = raw
        elif isinstance(raw, dict):
            if bool(raw.get("ok")):
                step.status = "done"
                step.result = str(raw.get("output", ""))
                outputs[step.step_id] = step.result
            else:
                step.status = "failed"
                step.error = str(raw.get("error") or "runner reported failure")
        else:
            step.status, step.error = "failed", "runner returned nothing"
        if step.status == "failed" and not step.error:
            step.error = "step failed"
        self._audit_step(plan, step)

    def _approve(self, plan: Plan, step: PlanStep) -> bool:
        category = step.hazard
        prompt = (f"Plan step is hazardous ({category}).\n"
                  f"  goal : {plan.goal[:200]}\n"
                  f"  step : {step.objective[:200]}")
        if self.gate is None:
            granted = False
        else:
            granted = bool(self.gate.approve(
                category, prompt,
                detail={"goal": plan.goal[:200], "step": step.step_id,
                        "objective": step.objective[:200]},
                source="planner"))
        if self.audit:
            self.audit.record("plan", f"approve:{step.step_id}",
                              "approved" if granted else "denied",
                              {"category": category,
                               "objective": step.objective[:200]})
        return granted

    def _audit_step(self, plan: Plan | None, step: PlanStep) -> None:
        if self.audit is None:
            return
        self.audit.record("plan", f"{step.status}:{step.step_id}",
                          "ok" if step.status == "done" else step.status,
                          {"objective": step.objective[:160],
                           "hazard": step.hazard,
                           "error": step.error[:200],
                           "goal": (plan.goal[:120] if plan else "")})

    # --------------------------------------------------------------- render
    def render(self, plan: Plan) -> str:
        counts = plan.counts()
        tally = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        lines = [f"Plan: {plan.goal}",
                 f"  {len(plan.steps)} step(s) [{plan.strategy}] | {tally}"]
        for step in plan.steps:
            if step.status == "done":
                line = f"  [x] {step.step_id}  {step.objective[:80]}"
            elif step.hazard and step.status in ("pending", "ready", "running"):
                line = (f"  [!] {step.step_id}  {step.objective[:80]}"
                        f"  (requires manual approval: {step.hazard})")
            else:
                marker = STATUS_MARKERS.get(step.status, "[?]")
                line = f"  {marker} {step.step_id}  {step.objective[:80]}"
                if step.status in BLOCKING_STATUSES and step.error:
                    line += f"  ({step.error[:60]})"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------ persistence
    def save_state(self, plan: Plan) -> bool:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"saved_at": time.time(), **plan.to_dict()}
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.state_path)
            return True
        except OSError:
            return False

    def load_state(self) -> Plan | None:
        if not self.state_path.is_file():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        plan = Plan.from_dict(data)
        return plan if plan.steps else None

    # ---------------------------------------------------------------- status
    def run(self, goal: str, runner: Runner | None = None) -> ExecutionResult:
        plan, _strategy = self.decompose(goal)
        return self.execute(plan, runner=runner)

    def status(self) -> str:
        return (f"GoalPlanner | plans={self.plans} "
                f"max_steps={self.max_steps} "
                f"history={len(self.history)} "
                f"state={'ok' if self.state_path.is_file() else 'absent'}")
