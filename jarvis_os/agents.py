"""Agent - role, tools, memory, planner.

An agent is a thin, auditable cognitive unit. It owns:

* ``role``      - what it is for (drives the system prompt and delegation).
* ``tools``     - the subset of :class:`~jarvis_os.tools.ToolRegistry` it may use.
* ``memory``    - shared persistent memory (scoped view + personal scratchpad).
* ``planner``   - optional :class:`~jarvis_os.taskmanager.TaskManager` for goals
                  it decomposes itself.

Agents can be created at runtime (``Agent.spawn``) and registered with the
TaskManager, which is how ``evolve`` and ``skill add`` grow the system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import ToolSpec

ROLE_LIBRARY: dict[str, str] = {
    "orchestrator": (
        "You are Jarvis's orchestrator, {honorific}'s chief of staff. Split work, "
        "delegate, and synthesise one clear answer. Always address {honorific}."
    ),
    "planner": (
        "You are a strategic planner. Turn goals into ordered, verifiable steps "
        "with explicit success criteria."
    ),
    "researcher": (
        "You are a research specialist. Gather facts from tools and memory, cite "
        "where each fact came from, and never speculate beyond your evidence."
    ),
    "coder": (
        "You are a senior engineer. Write correct, minimal, well-structured code. "
        "State assumptions explicitly."
    ),
    "reviewer": (
        "You are an adversarial reviewer. Hunt for bugs, unsafe behaviour and "
        "unsupported claims. Default to scepticism."
    ),
    "operator": (
        "You are the control operator. You start, stop, configure and inspect "
        "software. Confirm every state change you make."
    ),
    "teacher": (
        "You are a teacher. Explain clearly in simple language with a worked "
        "example, respecting {honorific}'s time."
    ),
    "guardian": (
        "You are a safety guardian. You refuse unsafe actions and explain why."
    ),
}


@dataclass(slots=True)
class AgentTask:
    objective: str
    assigned_agent: str
    tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    task_id: str = field(default_factory=lambda: f"t{uuid.uuid4().hex[:6]}")
    status: str = "pending"
    result: Any = None
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id, "objective": self.objective,
            "agent": self.assigned_agent, "tools": self.tools,
            "depends_on": self.depends_on, "status": self.status,
            "result": _summarise(self.result), "error": self.error,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }


def _summarise(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text[:limit] + ("..." if len(text) > limit else "")


class Agent:
    def __init__(self, name: str, role: str, tools: Any, memory: Any, planner: Any = None,
                 llm: Any = None, config: Any = None, audit: Any = None,
                 system_prompt: str | None = None, max_tool_calls: int = 8) -> None:
        self.id = f"ag_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.planner = planner
        self.llm = llm
        self.config = config
        self.audit = audit
        # A str, never a callable. run() calls self.system_prompt().
        self._system_prompt = system_prompt or ""
        self.max_tool_calls = max_tool_calls
        self.allowed_tools: set[str] = set()
        self.created_at = time.time()
        self.stats = {"tasks": 0, "success": 0, "failure": 0, "tool_calls": 0}
        self.scratchpad: list[dict[str, Any]] = []

    # ------------------------------------------------------------- identity
    @property
    def honorific(self) -> str:
        return self.config.honorific() if self.config else "Sir"

    def system_prompt(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        template = ROLE_LIBRARY.get(self.role, ROLE_LIBRARY["orchestrator"])
        return template.format(honorific=self.honorific)

    def grant_tools(self, *names: str) -> "Agent":
        self.allowed_tools.update(names)
        return self

    def restrict_tools(self, *names: str) -> "Agent":
        self.allowed_tools.difference_update(names)
        return self

    # -------------------------------------------------------------- running
    def run(self, objective: str, context: str = "",
            allow_tools: bool = True) -> dict[str, Any]:
        """Execute one objective, optionally using whitelisted tools."""
        self.stats["tasks"] += 1
        started = time.time()
        if self.audit:
            self.audit.record("agent", f"run:{self.name}", "start",
                              {"role": self.role, "objective": objective[:200]})

        prompt = self._build_prompt(objective, context)
        allowed_specs: list[ToolSpec] = []
        if allow_tools and self.tools is not None and self.llm is not None:
            allowed_specs = [
                s for s in self.tools.specs()
                if not self.allowed_tools or s.name in self.allowed_tools
            ]

        used_tools: list[str] = []
        result_text = ""
        try:
            for _ in range(max(1, self.max_tool_calls)):
                reply = self.llm.chat(
                    prompt, system=self.system_prompt(), tools=allowed_specs or None,
                    history=self.scratchpad[-6:], fallback="",
                )
                if not reply.ok and not reply.text:
                    return self._fail(objective, f"LLM unavailable: {reply.error}")
                calls = (reply.stats or {}).get("tool_calls") or []
                tool_calls = _extract_tool_calls(reply)
                if not tool_calls:
                    result_text = reply.text.strip()
                    break
                self.scratchpad.append({"role": "assistant", "content": reply.text or "[tool_use]"})
                for call in tool_calls[:3]:
                    outcome = self._invoke_tool(call, used_tools)
                    self.scratchpad.append({
                        "role": "user",
                        "content": f"tool `{call['name']}` -> {_summarise(outcome, 800)}",
                    })
                prompt = ("Continue. Use the tool results above to finish the objective."
                          if used_tools else prompt)
        except Exception as exc:  # noqa: BLE001
            return self._fail(objective, f"{type(exc).__name__}: {exc}")

        if not result_text:
            return self._fail(objective, "agent produced no usable output")
        self.stats["success"] += 1
        duration = int((time.time() - started) * 1000)
        if self.audit:
            self.audit.record("agent", f"done:{self.name}", "success",
                              {"ms": duration, "tools_used": used_tools})
        if self.memory is not None:
            self.memory.record_action(
                kind=f"agent:{self.role}", summary=objective[:200],
                result=result_text[:500], success=True,
                meta={"agent": self.name, "tools": used_tools},
            )
        return {
            "ok": True, "agent": self.name, "role": self.role, "output": result_text,
            "tools_used": used_tools, "duration_ms": duration,
        }

    def _build_prompt(self, objective: str, context: str) -> str:
        parts = [f"Objective: {objective}"]
        if context:
            parts.append(f"Context: {context}")
        if self.memory is not None:
            try:
                hits = self.memory.search(objective, limit=3)
                if hits:
                    recalled = "\n".join(
                        f"- ({s:.2f}) {m.content[:160]}" for m, s in hits
                    )
                    parts.append(f"Recalled memory:\n{recalled}")
            except Exception:  # noqa: BLE001
                pass
        if self.allowed_tools and self.tools is not None:
            names = ", ".join(sorted(self.allowed_tools))
            parts.append(f"Tools you may call: {names}")
        parts.append("Respond with the completed work, not a plan.")
        return "\n\n".join(parts)

    def _invoke_tool(self, call: dict[str, Any], used_tools: list[str]) -> Any:
        name = str(call.get("name", ""))
        if not name:
            return {"ok": False, "error": "tool call without a name"}
        if self.allowed_tools and name not in self.allowed_tools:
            return {"ok": False, "error": f"'{name}' is not granted to agent '{self.name}'"}
        if self.tools is None:
            return {"ok": False, "error": "no tool registry"}
        self.stats["tool_calls"] += 1
        used_tools.append(name)
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = _coerce_args(arguments)
        return self.tools.call(name, arguments)

    def _fail(self, objective: str, error: str) -> dict[str, Any]:
        self.stats["failure"] += 1
        if self.audit:
            self.audit.record("agent", f"fail:{self.name}", "failure", {"error": error[:200]})
        if self.memory is not None:
            self.memory.learn_lesson(
                topic=f"agent:{self.role}", lesson=f"failed on '{objective[:80]}': {error}",
                outcome="failure", weight=0.7,
            )
        return {"ok": False, "agent": self.name, "role": self.role, "error": error}

    # ---------------------------------------------------------------- misc
    def status(self) -> str:
        return (f"{self.name} [{self.role}] tasks={self.stats['tasks']} "
                f"ok={self.stats['success']} fail={self.stats['failure']} "
                f"tools={len(self.allowed_tools)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "tools": sorted(self.allowed_tools), "stats": dict(self.stats),
            "created_at": self.created_at,
        }

    # ------------------------------------------------------------ registry
    @staticmethod
    def spawn(name: str, role: str, registry: "AgentRegistry", **kwargs) -> "Agent":
        """Runtime agent creation, then registration in one step."""
        agent = Agent(name=name, role=role, **kwargs)
        registry.add(agent)
        return agent


class AgentRegistry:
    """Holds every live agent; supports runtime registration and removal."""

    def __init__(self, config=None, audit=None) -> None:
        self.config = config
        self.audit = audit
        self._agents: dict[str, Agent] = {}

    def add(self, agent: Agent) -> Agent:
        self._agents[agent.name] = agent
        if self.audit:
            self.audit.record("agent", f"register:{agent.name}", "ok", {"role": agent.role})
        return agent

    def remove(self, name: str) -> bool:
        removed = self._agents.pop(name, None)
        if removed and self.audit:
            self.audit.record("agent", f"remove:{name}", "ok", {})
        return bool(removed)

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def find_by_role(self, role: str) -> list[Agent]:
        return [a for a in self._agents.values() if a.role == role]

    def names(self) -> list[str]:
        return sorted(self._agents)

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def default_team(self, tools: Any, memory: Any, llm: Any, planner: Any,
                     config: Any, audit: Any) -> dict[str, Agent]:
        team = {
            "orchestrator": Agent("orchestrator", "orchestrator", tools, memory, planner,
                                  llm, config, audit).grant_tools(
                "memory_recall", "memory_remember", "ollama_models"),
            "planner": Agent("planner", "planner", tools, memory, planner, llm, config, audit),
            "researcher": Agent("researcher", "researcher", tools, memory, planner, llm,
                                 config, audit).grant_tools(
                "memory_recall", "web_fetch", "file_read", "dir_list"),
            "coder": Agent("coder", "coder", tools, memory, planner, llm, config, audit).grant_tools(
                "file_read", "file_write", "dir_list", "code_runner"),
            "reviewer": Agent("reviewer", "reviewer", tools, memory, planner, llm,
                              config, audit).grant_tools("memory_recall", "file_read"),
            "operator": Agent("operator", "operator", tools, memory, planner, llm,
                              config, audit).grant_tools("shell_exec", "memory_recall"),
            "guardian": Agent("guardian", "guardian", tools, memory, planner, llm, config, audit),
        }
        for agent in team.values():
            self.add(agent)
        return team

    def describe(self) -> str:
        if not self._agents:
            return "no agents registered"
        return "\n".join(f"  {a.status()}" for a in sorted(
            self._agents.values(), key=lambda x: x.name))


def _extract_tool_calls(reply: Any) -> list[dict[str, Any]]:
    """Pull tool calls out of a reply across Ollama response shapes."""
    calls: list[dict[str, Any]] = []
    raw_calls = (getattr(reply, "stats", {}) or {}).get("tool_calls") or []
    for call in raw_calls:
        function = call.get("function", call) if isinstance(call, dict) else {}
        name = function.get("name") if isinstance(function, dict) else None
        if not name:
            continue
        args = function.get("arguments", {})
        calls.append({"name": name, "arguments": args})
    if calls:
        return calls
    text = getattr(reply, "text", "") or ""
    if "tool_code" in text or '"tool_calls"' in text:
        from .llm import extract_json
        parsed = extract_json(text)
        if parsed:
            for call in parsed.get("tool_calls", []) or []:
                if isinstance(call, dict) and call.get("name"):
                    calls.append({"name": call["name"],
                                  "arguments": call.get("arguments", {})})
    return calls


def _coerce_args(raw: str) -> dict[str, Any]:
    import json
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}