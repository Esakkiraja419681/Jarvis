"""JarvisKernel - assembles every subsystem and runs the infinite loop.

Boot order matters and is deliberate:

1. Audit ledger opens first, so nothing can happen unlogged.
2. Memory opens second and is **integrity-checked**, then repaired if a crash
   left the vector store behind. Memory is never reset here, only *recovered*.
3. PermissionGate arms with the Overseer's isolation instructions applied.
4. Self-update initialises git version control and protects memory paths.
5. Only then do agents, task manager, tools and the loop come up.

The single line that runs forever is :meth:`JarvisKernel.cycle` inside
:meth:`JarvisKernel.run_forever`.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .agents import Agent, AgentRegistry
from .audit import AuditLedger
from .config import Config
from .control import ControlBridge
from .daemon import BackgroundDaemon
from .devices import DeviceManager
from .evolution import SelfEvolution
from .interfaces.text import TextCommandInterface
from .interfaces.voice import VoiceCommandInterface
from .language import UniversalLanguageLayer
from .llm import LLMClient, LLMUnavailable
from .memory import PersistentMemory, local_embed
from .overseer import OverseerAgenticOS
from .permissions import PermissionGate
from .sandbox import PythonSandbox, ShellSandbox
from .selfupdate import SelfUpdate
from .software_factory import SoftwareFactory
from .taskmanager import TaskManager
from .tools import ToolRegistry


class JarvisKernel:
    def __init__(self, config: Config | None = None, honorific: str | None = None) -> None:
        self.config = config or Config()
        self.config.root.mkdir(parents=True, exist_ok=True)
        for folder in ("data", "data/sandbox", "data/built", "data/skills", "data/control"):
            self.config.resolve(folder).mkdir(parents=True, exist_ok=True)
        self._honorific = honorific
        self.boot_time = time.time()
        self.cycles = 0
        self.consecutive_errors = 0
        self.llm_errors = 0
        self.last_output = ""
        self.last_goal: str | None = None
        self.queue: list[str] = []
        self.stopped = False
        self.paused = False
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None

        self._build()

    # ------------------------------------------------------------------ boot
    def _build(self) -> None:
        cfg = self.config
        self.honorific = self._honorific or cfg.honorific()

        self.audit = AuditLedger(
            cfg.resolve(cfg.get("audit.sqlite_path", "data/audit.db")),
            cfg.resolve(cfg.get("audit.ledger_path", "data/audit_ledger.jsonl")),
            enabled=cfg.feature("audit_log", True),
        ).connect()
        # The Overseer appends to this ledger from its own process, so a fork
        # must be re-linked at boot. No event is ever discarded.
        audit_ok, _ = self.audit.verify()
        if not audit_ok:
            repaired = self.audit.repair_chain()
            self.audit.record("audit", "chain_repair", "ok",
                              {"relinked": repaired.get("relinked"),
                               "events": repaired.get("events")})

        self.memory = PersistentMemory(
            cfg.resolve(cfg.get("memory.sqlite_path", "data/memory.db")),
            cfg.resolve(cfg.get("memory.vector_path", "data/vectors.json")),
            cfg.resolve(cfg.get("memory.lesson_log", "data/lessons.jsonl")),
            cfg.resolve(cfg.get("memory.action_log", "data/actions.jsonl")),
            top_k=int(cfg.get("memory.retrieval_top_k", 5)),
            immutable=bool(cfg.get("memory.immutable", True)),
        ).open()

        # --- memory recovery (never a reset) ------------------------------
        self.memory_report = self.memory.integrity_report()
        if not self.memory_report.get("ok") or \
                len(self.memory._vectors) < self.memory.stats.get("memories", 0):
            restored = self.memory.repair()
            self.audit.record("memory", "integrity_repair", "ok",
                              {"restored_vectors": restored,
                               "issues": self.memory_report.get("issues", [])[:5]})
        if cfg.get("memory.hash_chain_verify_on_start", True):
            self.audit.record("memory", "boot_check", "ok",
                              {"stats": self.memory.integrity_report().get("stats", {})})

        # --- interfaces & language ----------------------------------------
        self.language = UniversalLanguageLayer(cfg, audit=self.audit)
        self.text_iface = TextCommandInterface(self, cfg)
        self.voice_iface = VoiceCommandInterface(self, cfg)

        # --- the gate -------------------------------------------------------
        self.gate = PermissionGate(
            cfg, audit=self.audit,
            text_input=lambda prompt: self.text_iface._read(prompt),
            voice_input=self.voice_iface.capture_answer,
            voice_active=self.voice_iface.voice_active,
        )
        self.gate.honorific = self.honorific
        self._apply_overseer_control()

        # --- llm (ollama only) ---------------------------------------------
        self.llm = LLMClient(cfg, gate=self.gate, audit=self.audit)
        self.memory.set_embedder(self._embed)

        # --- execution ------------------------------------------------------
        self.sandbox = PythonSandbox(cfg, gate=self.gate, audit=self.audit)
        self.shell = ShellSandbox(cfg, gate=self.gate, audit=self.audit)
        self.tools = ToolRegistry(cfg, gate=self.gate, audit=self.audit,
                                  memory=self.memory, sandbox=self.sandbox,
                                  shell=self.shell, root=cfg.root)
        self.tools.attach_llm(self.llm)

        # --- cognition ------------------------------------------------------
        self.agents = AgentRegistry(cfg, self.audit)
        self.task_manager = TaskManager(self.agents, self.memory, self.llm,
                                        cfg, self.audit)
        self.team = self.agents.default_team(self.tools, self.memory, self.llm,
                                            self.task_manager, cfg, self.audit)

        # --- self-modification ---------------------------------------------
        self.self_update = SelfUpdate(cfg, gate=self.gate, audit=self.audit,
                                      memory=self.memory, root=cfg.root)
        self.self_update.init()
        self.evolve_engine = SelfEvolution(cfg, self.llm, self.memory, self.gate,
                                           self.self_update, self.sandbox,
                                           audit=self.audit, root=cfg.root)

        # --- software -------------------------------------------------------
        self.control = ControlBridge(cfg, gate=self.gate, audit=self.audit)
        self.software = SoftwareFactory(cfg, self.llm, self.memory, self.gate,
                                        self.control, self.sandbox, audit=self.audit)
        if cfg.feature("control_bridge", True) and cfg.feature("rest_api", True):
            self.control.start_rest()

        # --- devices & daemon ------------------------------------------------
        self.devices = DeviceManager(cfg, gate=self.gate, audit=self.audit,
                                     memory=self.memory, control=self.control)
        self.daemon = BackgroundDaemon(cfg, gate=self.gate, audit=self.audit)
        self.daemon.bind_kernel(self)

        # --- overseer --------------------------------------------------------
        self.overseer = OverseerAgenticOS(cfg, audit=self.audit, memory=self.memory,
                                          llm=self.llm, gate=self.gate, root=cfg.root)
        self.overseer.bind_restart(self._overseer_restart)
        if cfg.feature("overseer", True):
            self.overseer.start()

        self._install_signal_handlers()

    def _embed(self, text: str) -> list[float]:
        try:
            vector = self.llm.embed(text)
            return vector or local_embed(text)
        except Exception:  # noqa: BLE001
            return local_embed(text)

    def _apply_overseer_control(self, include_gate: bool = True) -> None:
        """Honour isolation instructions the Overseer left for this boot.

        ``include_gate`` is False when the Overseer's own restart callback runs:
        the restart already remediated the halt, so the permission gate must come
        back up. Boot passes True so an unresolved incident still halts Jarvis.
        """
        path = self.config.resolve("data/overseer_control.json")
        if not path.is_file():
            return
        try:
            control = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        applied: list[str] = []
        if control.get("gate_disarm"):
            if include_gate:
                self.gate.disarm(f"overseer isolation after incident "
                                 f"{control.get('last_incident', '?')}")
                applied.append("gate_disarm")
            else:
                applied.append("gate_rearmed_after_restart")
        if control.get("sandbox_enabled") is False:
            self.config.set("features.sandbox", False)
            applied.append("sandbox_disabled")
        if control.get("tools_disabled"):
            applied.append("tools_disabled")
        if control.get("planner_disabled"):
            applied.append("planner_disabled")
        if control.get("loop_interval_max"):
            self.config.set("loop.interval_seconds",
                            min(int(control["loop_interval_max"]),
                                int(self.config.get("loop.interval_seconds", 5)) * 12))
            applied.append("loop_interval_max")
        if applied:
            self.audit.record("kernel", "overseer_control_applied", "ok",
                              {"controls": applied,
                               "incident": control.get("last_incident", "")})

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.audit.record("kernel", "signal", "ok", {"signal": int(signum)})
            self.stopped = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # ------------------------------------------------------------- heartbeat
    def heartbeat_payload(self) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "pid": os.getpid(),
            "cycles": self.cycles,
            "started_at": self.boot_time,
            "consecutive_errors": self.consecutive_errors,
            "llm_errors": self.llm_errors,
            "last_output": self.last_output[:800],
            "claimed_success": bool(self.last_output),
            "verified_success": self.last_goal is None or self.cycles % 5 == 0,
            "queued": len(self.queue),
            "paused": self.paused,
            "mode": "daemon" if self.config.feature("background_daemon") else "foreground",
            "notes": "",
            "memory": self.memory.status(),
        }

    def _write_heartbeat(self) -> None:
        if not self.config.feature("overseer", True):
            return
        try:
            self.overseer.write_heartbeat(self.heartbeat_payload())
        except OSError:
            pass

    # -------------------------------------------------------------- one cycle
    def cycle(self) -> dict[str, Any]:
        """One iteration of the infinite loop. This is the whole design."""
        self.cycles += 1
        state: dict[str, Any] = {"cycle": self.cycles, "ts": time.time()}
        try:
            if self.config.resolve(self.config.get("loop.stop_file", "data/STOP")).exists():
                self.stopped = True
                state["action"] = "stop-file"
                return state
            if self.config.resolve(self.config.get("loop.pause_file", "data/PAUSE")).exists():
                self.paused = True
                state["action"] = "paused"
                return state
            self.paused = False

            if self.config.feature("control_bridge", True) and self.config.feature("ipc", True):
                results = self.control.poll()
                if results:
                    state["ipc"] = results

            if self.queue:
                goal = self.queue.pop(0)
                state["action"] = "execute"
                state["goal"] = goal
                state["result"] = self.execute(goal, source="loop")
                self.consecutive_errors = 0
            else:
                state["action"] = "idle"

            if self.cycles % 30 == 0:
                state["evolution"] = self.maybe_evolve()
            self._write_heartbeat()
            return state
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            self.consecutive_errors += 1
            self.audit.record("kernel", "cycle_error", "failed",
                              {"error": f"{type(exc).__name__}: {exc}"[:400],
                               "cycle": self.cycles})
            self.memory.learn_lesson(
                topic="kernel", lesson=f"cycle {self.cycles} failed: {exc}"[:200],
                outcome="failure", weight=0.6)
            self._write_heartbeat()
            state["error"] = str(exc)
            return state

    def run_forever(self) -> int:
        """THE INFINITE SINGLE-LINE LOOP."""
        interval = float(self.config.get("loop.interval_seconds", 5))
        backoff = float(self.config.get("loop.backoff_on_error_seconds", 15))
        max_cycles = int(self.config.get("loop.max_cycles", 0))
        self.audit.record("kernel", "loop_start", "ok",
                          {"interval": interval, "pid": os.getpid(),
                           "mode": "daemon" if self.config.feature("background_daemon") else "foreground"})
        try:
            while not self.stopped:
                started = time.time()
                result = self.cycle()
                if self.consecutive_errors:
                    delay = min(backoff * self.consecutive_errors, 120)
                else:
                    delay = interval
                time.sleep(max(0.0, delay - (time.time() - started)))
                if max_cycles and self.cycles >= max_cycles:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    # -------------------------------------------------------------- execution
    def execute(self, goal: str, source: str = "cli", context: str = "") -> str:
        self.last_goal = goal
        english, detection = self.language.process_inbound(goal)
        self.audit.record("kernel", "execute", "start",
                          {"goal": goal[:300], "source": source,
                           "detected_language": detection.language})
        try:
            reply = self.llm.chat(english, persona="orchestrator", fallback="")
            if not reply.ok:
                self.llm_errors += 1
        except LLMUnavailable:
            self.llm_errors += 1

        result = self.task_manager.execute(english, context=context)
        output = result.output or (
            "I could not complete that. Every delegated step failed - see the audit trail."
            if not result.ok else "Done.")
        self.last_output = output
        self.audit.record("kernel", "execute", "ok" if result.ok else "failed",
                          {"goal": goal[:200], "strategy": result.strategy,
                           "tasks": len(result.tasks), "ms": result.duration_ms})
        if not result.ok:
            self.memory.learn_lesson(
                topic="goal", lesson=f"goal '{goal[:80]}' had failing subtasks",
                outcome="failure", weight=0.6)
        return self.language.process_outbound(output)

    def enqueue(self, goal: str) -> int:
        with self._lock:
            self.queue.append(goal)
            return len(self.queue)

    # ------------------------------------------------------------- evolution
    def maybe_evolve(self, focus: str = "") -> str:
        if not self.config.feature("infinite_loop", True):
            return "evolution skipped: infinite loop disabled"
        if not self.config.get("evolution.self_update_enabled", False):
            return ("self-update is disabled (evolution.self_update_enabled = false). "
                    "Turn it on with: config evolution.self_update_enabled=true")
        cycle = self.evolve_engine.run_cycle(focus=focus)
        return self.evolve_engine._describe(cycle)

    def evolve(self, report_only: bool = False) -> str:
        if not self.config.phase("propose", True):
            return ("Evolution is off: the 'propose' phase is disabled in config.\n"
                    + self.evolve_engine.phases_report())
        return self.evolve_engine._describe(self.evolve_engine.run_cycle(report_only=report_only))

    def rollback(self, target: str) -> str:
        if target.startswith("evo"):
            return self.evolve_engine.rollback(target)
        result = self.software.remove(target) if self.software.registry.get(target) else None
        if result:
            return f"Removed '{target}'. Memory untouched."
        return self.self_update.rollback_to(None, f"operator rollback of {target}").message

    def set_phase(self, name: str, enabled: bool) -> None:
        self.config.set_phase(name, enabled)
        self.audit.record("evolution", f"phase:{name}", "ok",
                          {"enabled": enabled, "source": "cli"})

    # ---------------------------------------------------------------- skills
    def skill_add(self, args: list[str]) -> str:
        """Register a new tool (and optionally a new agent) at runtime."""
        if len(args) < 2:
            return ("Usage: skill add <tool_name> <module:function> [agent_role]\n"
                    "  e.g. skill add my_metric jarvis_os.tools:_memory_recall coder")
        name, target = args[0], args[1]
        role = args[2] if len(args) > 2 else "coder"
        if self.gate is not None and not self.gate.approve(
            "skill_add",
            f"Jarvis requests permission to add a new capability at runtime.\n"
            f"  tool : {name}\n  code : {target}\n  agent: {role}",
            detail={"tool": name, "module": target, "agent_role": role},
        ):
            return "permission denied by operator - no skill was added"
        try:
            module_path, _, attr = target.partition(":")
            import importlib
            handler = getattr(importlib.import_module(module_path), attr or "run")
        except Exception as exc:  # noqa: BLE001
            return f"Could not load '{target}': {exc}"
        try:
            self.tools.register(name, handler,
                                description=(handler.__doc__ or f"skill {name}").strip()[:300],
                                permission_category="code_exec", dangerous=True,
                                builtin=False, replace=True)
        except Exception as exc:  # noqa: BLE001
            return f"Could not register tool: {exc}"
        Agent.spawn(name, role, self.agents, tools=self.tools, memory=self.memory,
                    planner=self.task_manager, llm=self.llm, config=self.config,
                    audit=self.audit, max_tool_calls=4).grant_tools(name)
        self.memory.remember(f"Skill '{name}' added from {target}", kind="skill",
                             importance=0.7)
        self.memory.learn_lesson(topic="skill_add",
                                 lesson=f"'{name}' from {target} is available",
                                 outcome="success", weight=0.6)
        return (f"Skill '{name}' registered and agent '{name}' added with role '{role}'.\n"
                f"  tools {len(self.tools.names())} | agents {len(self.agents.names())}")

    def skill_list(self) -> str:
        lines = [f"Skills directory: {self.config.resolve(self.config.get('skills.path', 'data/skills'))}"]
        skill_dir = self.config.resolve(self.config.get("skills.path", "data/skills"))
        files = sorted(p.name for p in skill_dir.glob("*.py")) if skill_dir.is_dir() else []
        lines.append(f"  skill files: {', '.join(files) if files else 'none'}")
        lines.append("")
        lines.append("Runtime-registered tools and agents:")
        lines.append(self.tools.describe())
        lines.append("")
        lines.append(self.agents.describe())
        return "\n".join(lines)

    def learn(self, text: str) -> Any:
        if self.gate is not None and not self.gate.approve(
            "self_learn", f"Record a lesson for permanent memory?\n  {text[:200]}",
            detail={"text": text[:300]},
        ):
            return None
        lesson = self.memory.learn_lesson(topic="operator", lesson=text[:600],
                                          outcome="neutral", weight=0.8)
        self.audit.record("memory", "learn", "ok", {"id": lesson.id})
        return lesson

    # -------------------------------------------------------------- software
    def build(self, spec: str) -> str:
        result = self.software.build(spec)
        if result.get("ok"):
            return str(result.get("message", "Built."))
        detail = result.get("test", {}) or {}
        return (f"Build failed: {result.get('error', 'unknown error')}\n"
                + (f"  stderr: {str(detail.get('stderr', ''))[:300]}" if detail else ""))

    def software_action(self, action: str, args: list[str]) -> str:
        if not args:
            return ("Usage: <command> <program-id> [args...]\n"
                    "  install <id> | launch <id> | control <id> <start|stop|restart|"
                    "status|inspect|logs|configure>\n"
                    + self.software.report())
        program_id, rest = args[0], args[1:]
        if action == "install":
            return _result_text(self.software.install(program_id))
        if action == "launch":
            return _result_text(self.software.launch(program_id, rest))
        sub = rest[0].lower() if rest else "status"
        return _result_text(self.software.control(program_id, sub, rest[1:]))

    # ------------------------------------------------------------- device/daemon
    def device_command(self, args: list[str]) -> str:
        return self.devices.command(args)

    def daemon_command(self, action: str, args: list[str] | None = None) -> str:
        return self.daemon.command(action, args or [])

    # -------------------------------------------------------------- overseer
    def overseer_control(self, action: str) -> str:
        action = action.lower()
        if action == "on":
            self.config.set_feature("overseer", True)
            self.overseer.start()
            return "OverseerAgenticOS is watching."
        if action == "off":
            self.overseer.stop()
            self.config.set_feature("overseer", False)
            return "Overseer stopped. I am unguarded, Sir - I would not advise it."
        if action in ("clear", "reset", "release"):
            # Operator escape hatch: drop any active isolation and re-arm.
            self.overseer.clear_control()
            if self.gate and not self.gate.armed:
                self.gate.arm()
            self.audit.record("overseer", "isolation_cleared", "ok",
                              {"by": "operator"})
            return ("Isolation cleared and the permission gate is armed again, Sir. "
                    "Note: the Overseer will isolate the faulty layer again if it "
                    "detects the same fault again.")
        if action == "scan":
            return self.overseer.scan_report()
        return self.overseer.status()

    def _overseer_restart(self) -> None:
        """Overseer asked for a restart: re-apply isolation, then keep looping.

        A successful restart IS the remediation for the halt, so the permission
        gate is re-armed here. Layer-specific isolation (sandbox off, planner
        off, loop throttling) is re-applied and stays until the Overseer or the
        operator clears it.
        """
        self.audit.record("kernel", "overseer_restart", "ok", {"cycles": self.cycles})
        self.consecutive_errors = 0
        self.cycles = 0
        self.last_goal = None
        if self.gate and not self.gate.armed:
            self.gate.arm()
        self._apply_overseer_control(include_gate=False)

    # ------------------------------------------------------------------ voice
    def set_voice(self, state: str) -> str:
        state = state.lower()
        if state in ("on", "1", "true", "yes", "enable"):
            ok, message = self.voice_iface.enable()
            # _voice_active is a callable consulted on every approval prompt.
            self.gate._voice_active = lambda: self.voice_iface.voice_active
            return message
        if state in ("off", "0", "false", "no", "disable"):
            return self.voice_iface.disable()
        return self.voice_iface.status()

    # ---------------------------------------------------------------- status
    def features_report(self) -> str:
        lines = ["Features:"]
        for name in self.config.feature_names():
            lines.append(f"  {name:<20} {'on' if self.config.feature(name) else 'off'}")
        lines.append("")
        lines.append(self.evolve_engine.phases_report())
        lines.append("")
        lines.append("Permission categories needing manual approval:")
        for category in self.config.get("permissions.categories_requiring_manual_approval", []):
            lines.append(f"  - {category}")
        return "\n".join(lines)

    def status_report(self) -> str:
        uptime = int(time.time() - self.boot_time)
        sections = [
            f"Jarvis OS  |  uptime {uptime}s  |  cycles {self.cycles}  |  "
            f"errors {self.consecutive_errors}  |  queued {len(self.queue)}",
            "",
            f"[memory]    {self.memory.status()}",
            f"[llm]       provider={self.config.get('llm.provider')} "
            f"model={self.config.get('llm.model')} "
            f"local={'yes' if self.llm.available() else 'NO - Ollama not reachable'}",
            f"[permission]{self.gate.report()}",
            f"[audit]     {self.audit.summary()}",
            f"[language]  {self.language.status()}",
            f"[software]  {len(self.software.registry)} program(s) built",
            f"[devices]   {self.devices.report()}",
            f"[daemon]    {self.daemon.status()}",
            f"[evolve]    {self.evolve_engine.phases_report().splitlines()[0] if self.evolve_engine.cycles else 'no cycle yet'}",
            "",
            self.task_manager.status(),
            "",
            self.overseer.status(),
        ]
        return "\n".join(sections)

    # -------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        try:
            self.control.shutdown_all()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.overseer.stop()
        except Exception:  # noqa: BLE001
            pass
        self.audit.record("kernel", "shutdown", "ok",
                          {"cycles": self.cycles,
                           "memory": self.memory.status()})
        self.memory.close()
        self.audit.close()

    # --------------------------------------------------------- thread helpers
    def start_loop_thread(self) -> threading.Thread:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self.run_forever, daemon=True,
                                            name="jarvis-loop")
            self._thread.start()
        return self._thread


def _result_text(result: dict[str, Any]) -> str:
    if result.get("ok"):
        return str(result.get("message") or
                   json.dumps(result.get("program", result), ensure_ascii=False)[:800])
    if result.get("denied"):
        return f"Permission denied by operator, Sir. Nothing was changed."
    return f"Failed: {result.get('error', 'unknown error')}"