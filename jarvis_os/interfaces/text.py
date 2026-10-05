"""TextCommandInterface - the manual text channel.

Both the interactive REPL and the PermissionGate read through this class, so a
single implementation governs how Jarvis talks to the operator: every reply is
translated into the operator's language and prefixed with the honorific.
"""

from __future__ import annotations

import shlex
import sys
import time
from typing import Any, Callable

# Every reply is translated into the operator's language before it is printed,
# and the default Windows console code page (cp1252) cannot represent Tamil.
# Without this the first translated reply raises UnicodeEncodeError partway
# through printing, so the guard goes in before the interface can emit anything.
# The entry point does the same, because this module is also imported directly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        # A captured or non-TextIOBase stream has nothing to reconfigure, and a
        # missing guard must never stop the interface from loading.
        pass

BANNER = r"""
     _                       ____  ____
    | |                     / ___||  _ \
    | | __ _ _ __ __ _  ___  \___ \| |_) |
    | |/ _` | '__/ _` |/ _ \  ___) |  _ <
    | | (_| | | | (_| |  __/ /____/ |_| \_\   OS
    |_|\__,_|_|  \__, |\___||_____/|____/
                 |___/   self-evolving agentic OS
"""


class TextCommandInterface:
    """Manual text commands. Every response is honorific-prefixed."""

    name = "text"

    def __init__(self, kernel: Any, config: Any) -> None:
        self.kernel = kernel
        self.config = config
        self.history: list[str] = []
        self.running = False
        self.prompt = "jarvis> "
        self._echo = True

    # ------------------------------------------------------------- plumbing
    def say(self, text: str, translate: bool = True) -> str:
        rendered = self.kernel.language.process_outbound(text) if translate else \
            self.kernel.language.prefix(text)
        self.history.append(rendered)
        self.history = self.history[-500:]
        if self._echo:
            print(rendered, flush=True)
        return rendered

    def plain(self, text: str = "") -> None:
        """Raw output - no honorific, no translation (tables, JSON, traces)."""
        if text:
            print(text, flush=True)

    def _read(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            raise SystemExit(0)
        except KeyboardInterrupt:
            self.plain("")
            return "quit"

    # ----------------------------------------------------------------- REPL
    def banner(self) -> None:
        self.plain(BANNER)
        self.say(
            f"Jarvis OS online. Ollama is the only permitted AI backend. "
            f"Every self-change needs your manual approval. Type 'help' for commands."
        )

    def start(self) -> int:
        self.running = True
        # Start beating before the banner so the Overseer has a live, owned pid
        # to supervise from the very first moment, not only once a command runs.
        self.kernel.start_heartbeat_refresher()
        self.kernel.set_mode("interactive")
        self.banner()
        try:
            while self.running:
                try:
                    line = self._read(self.prompt)
                except SystemExit:
                    self.say("Shutting down. Memory preserved.")
                    return 0
                self.dispatch(line)
            return 0
        finally:
            # Reached on /exit and on an unhandled error out of dispatch(). A
            # thread left running would keep rewriting data/heartbeat.json after
            # the process is meant to be done.
            self.kernel.stop_heartbeat_refresher()

    def dispatch(self, line: str) -> Any:
        line = (line or "").strip()
        if not line:
            return None
        self.history.append(f"> {line}")
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        command, args = (parts[0].lower() if parts else ""), parts[1:]
        handler = self.COMMANDS.get(command)
        if handler is None:
            self.say(f"Unknown command '{command}'. Type 'help' for the command list.")
            return None
        try:
            return handler(self, args)
        except PermissionError as exc:
            self.say(f"Refused: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.say(f"Command '{command}' failed: {type(exc).__name__}: {exc}")
            if self.kernel.audit:
                self.kernel.audit.record("cli", f"error:{command}", "failed",
                                         {"error": str(exc)[:300]})
        return None

    # ------------------------------------------------------------- commands
    def cmd_help(self, args: list[str]) -> None:
        rows = [
            ("run <goal>", "plan, delegate and execute a goal"),
            ("plan <goal>", "show the plan without executing it"),
            ("status", "kernel, agents, tasks, overseer, memory"),
            ("memory [query]", "recall memory, or dump a summary"),
            ("tools", "list the tool registry"),
            ("evolve", "run the self-evolution phase engine"),
            ("learn <text>", "record a durable lesson"),
            ("skill add <name> <tool>", "register a new tool at runtime"),
            ("skill list", "list registered skills and dynamic tools"),
            ("agent add <name> <role>", "register a new agent at runtime"),
            ("agent list", "list every live agent and its tool grants"),
            ("update <target>", "self-update pipeline with git versioning"),
            ("approve <category>", "set a category to require approval"),
            ("deny <category>", "flip a permission decision in the audit record"),
            ("rollback <id>", "roll back an evolution or software change"),
            ("voice on|off", "enable or disable the voice interface"),
            ("build <spec>", "SoftwareFactory: generate a program"),
            ("install <id>", "install a built program"),
            ("launch <id>", "launch a built program"),
            ("control <id> <action>", "start/stop/configure/inspect"),
            ("stop", "halt Jarvis OS gracefully"),
            ("translate <text>", "detect language and translate both ways"),
            ("overseer [on|off|scan|status]", "watchdog control"),
            ("device [list|add|remove]", "DeviceManager"),
            ("background [start|stop|status]", "BackgroundDaemon"),
            ("config [key[=value]]", "read or change configuration"),
            ("features", "toggle features and evolution phases"),
            ("audit [n]", "recent audit-ledger entries"),
            ("quit / exit", "leave the REPL"),
        ]
        self.plain("Commands:")
        for name, description in rows:
            self.plain(f"  {name:<28} {description}")

    def cmd_run(self, args: list[str]) -> None:
        goal = " ".join(args).strip()
        if not goal:
            self.say("Give me a goal to run.")
            return
        self.say(f"Working on it: {goal}")
        # Branch on the structured result, not on the translated string: the
        # operator has to be told when the plan's tools never ran. execute()'s
        # return value goes through translation and the honorific, so a status
        # word hidden in it cannot be tested for reliably.
        reply, report = self.kernel.execute_report(goal, source="cli")
        if report.partial:
            missing = ", ".join(report.unexecuted_tools)
            self.say(f"PARTIAL. This goal is not fully done: the plan listed "
                     f"{missing}, but no tool actually ran, so that part is "
                     f"unverified.")
        elif not report.ok:
            self.say("INCOMPLETE. Every delegated step failed - see the audit trail.")
        if reply:
            self.say(reply)

    def cmd_plan(self, args: list[str]) -> None:
        goal = " ".join(args).strip()
        if not goal:
            self.say("Give me a goal to plan.")
            return
        tasks, strategy = self.kernel.task_manager.decompose(goal)
        self.plain(f"Plan ({strategy}) for: {goal}")
        for task in tasks:
            deps = f" <- {', '.join(d[-6:] for d in task.depends_on)}" if task.depends_on else ""
            tools = f"  tools={','.join(task.tools)}" if task.tools else ""
            self.plain(f"  [{task.assigned_agent:<13}] {task.objective}{deps}{tools}")

    def cmd_status(self, args: list[str]) -> None:
        self.plain(self.kernel.status_report())

    def cmd_memory(self, args: list[str]) -> None:
        if args:
            hits = self.kernel.memory.search(" ".join(args))
            if not hits:
                self.say("Nothing relevant in memory.")
                return
            for record, score in hits:
                self.plain(f"  [{score:.2f}] ({record.kind}) {record.content[:200]}")
            return
        self.plain(self.kernel.memory.export_markdown())

    def cmd_tools(self, args: list[str]) -> None:
        self.plain(self.kernel.tools.describe())

    def cmd_evolve(self, args: list[str]) -> None:
        self.plain(self.kernel.evolve(report_only="--dry-run" in args))

    def cmd_learn(self, args: list[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.say("Tell me what to remember.")
            return
        lesson = self.kernel.learn(text)
        self.say(f"Recorded as lesson {lesson.id}.")

    def cmd_skill(self, args: list[str]) -> None:
        sub = (args[0].lower() if args else "list")
        if sub == "add":
            self.plain(self.kernel.skill_add(args[1:]))
        elif sub == "list":
            self.plain(self.kernel.skill_list())
        else:
            self.say("Usage: skill add <name> <module:function> | skill list")

    def cmd_agent(self, args: list[str]) -> None:
        sub = (args[0].lower() if args else "list")
        if sub == "add":
            self.plain(self.kernel.agent_add(args[1:]))
        elif sub == "remove":
            self.plain(self.kernel.agent_remove(args[1:]))
        elif sub == "list":
            self.plain(self.kernel.agent_list())
        else:
            self.say("Usage: agent add <name> <role> | agent remove <name> | agent list")

    def cmd_update(self, args: list[str]) -> None:
        self.plain(self.kernel.self_update.describe_target(args[0] if args else None))

    def cmd_approve(self, args: list[str]) -> None:
        if not args:
            self.say("Name the category to gate, e.g. approve shell_exec")
            return
        self.kernel.gate.require(args[0])
        self.say(f"'{args[0]}' now requires manual approval every time.")

    def cmd_deny(self, args: list[str]) -> None:
        if not args:
            self.say("Name the category to record as denied.")
            return
        self.kernel.audit.record("permission", f"gate:{args[0]}", "denied",
                                 {"source": "cli", "operator": "manual"})
        self.say(f"Recorded a manual denial for '{args[0]}'.")

    def cmd_rollback(self, args: list[str]) -> None:
        if not args:
            self.say("Usage: rollback <evolution-id|software-id>")
            return
        self.say(self.kernel.rollback(args[0]))

    def cmd_voice(self, args: list[str]) -> None:
        state = (args[0].lower() if args else "status")
        self.say(self.kernel.set_voice(state))

    def cmd_build(self, args: list[str]) -> None:
        self.say(self.kernel.build(" ".join(args).strip()))

    def cmd_install(self, args: list[str]) -> None:
        self.say(self.kernel.software_action("install", args))

    def cmd_launch(self, args: list[str]) -> None:
        self.say(self.kernel.software_action("launch", args))

    def cmd_control(self, args: list[str]) -> None:
        self.say(self.kernel.software_action("control", args))

    def cmd_stop(self, args: list[str]) -> None:
        self.running = False
        self.say("Stopping. Memory stays intact; I will be here when you call.")

    def cmd_translate(self, args: list[str]) -> None:
        text = " ".join(args).strip()
        if not text:
            self.say("Give me something to translate.")
            return
        detection = self.kernel.language.detect(text)
        english = self.kernel.language.translate(text, "en")
        # process_outbound already applies the honorific, so print it directly
        # instead of prefixing it a second time.
        native = self.kernel.language.process_outbound(english)
        self.plain(f"detected: {detection.name} ({detection.language}, {detection.confidence:.2f})")
        self.plain(f"english:   {english}")
        self.plain(native)

    def cmd_overseer(self, args: list[str]) -> None:
        sub = (args[0].lower() if args else "status")
        if sub == "on":
            self.say(self.kernel.overseer_control("start"))
        elif sub == "off":
            self.say(self.kernel.overseer_control("stop"))
        elif sub == "scan":
            self.plain(self.kernel.overseer_control("scan"))
        elif sub in ("clear", "reset", "release"):
            self.say(self.kernel.overseer_control("clear"))
        else:
            self.plain(self.kernel.overseer_control("status"))

    def cmd_device(self, args: list[str]) -> None:
        self.plain(self.kernel.device_command(args))

    def cmd_background(self, args: list[str]) -> None:
        sub = (args[0].lower() if args else "status")
        self.say(self.kernel.daemon_command(sub, args[1:]))

    def cmd_config(self, args: list[str]) -> None:
        if not args:
            self.plain("current config:\n" + _dump_config(self.kernel.config.as_dict()))
            return
        target = args[0]
        if "=" in target:
            key, _, value = target.partition("=")
            self.kernel.config.set(key.strip(), _coerce(value.strip()))
            self.say(f"Set {key.strip()} = {value.strip()}")
            return
        self.plain(f"{target} = {self.kernel.config.get(target)!r}")

    def cmd_features(self, args: list[str]) -> None:
        if not args:
            self.plain(self.kernel.features_report())
            return
        target = args[0]
        if target.startswith("phase."):
            name, _, state = target.partition("=")
            self.kernel.set_phase(name.split(".", 1)[1], state.strip().lower() in ("1", "on", "true", "yes"))
            self.say(f"Evolution phase '{name.split('.', 1)[1]}' set.")
            return
        value = (args[1].lower() if len(args) > 1 else "toggle")
        enabled = value in ("on", "1", "true", "yes", "enable")
        if value == "toggle":
            enabled = not self.kernel.config.feature(target)
        self.kernel.config.set_feature(target, enabled)
        self.say(f"Feature '{target}' is now {'on' if enabled else 'off'}.")

    def cmd_audit(self, args: list[str]) -> None:
        limit = int(args[0]) if args and args[0].isdigit() else 20
        for entry in self.kernel.audit.tail(limit=limit):
            stamp = time.strftime("%H:%M:%S", time.localtime(entry["ts"]))
            self.plain(f"  {stamp} [{entry['category']}] {entry['action']} -> {entry['decision']}")

    def cmd_quit(self, args: list[str]) -> None:
        self.running = False
        self.say("Goodbye. Memory is safe.")

    COMMANDS: dict[str, Callable] = {}


TextCommandInterface.COMMANDS = {
    "help": TextCommandInterface.cmd_help, "?": TextCommandInterface.cmd_help,
    "run": TextCommandInterface.cmd_run, "do": TextCommandInterface.cmd_run,
    "plan": TextCommandInterface.cmd_plan,
    "status": TextCommandInterface.cmd_status, "stat": TextCommandInterface.cmd_status,
    "memory": TextCommandInterface.cmd_memory, "mem": TextCommandInterface.cmd_memory,
    "tools": TextCommandInterface.cmd_tools,
    "evolve": TextCommandInterface.cmd_evolve,
    "learn": TextCommandInterface.cmd_learn,
    "skill": TextCommandInterface.cmd_skill,
    "agent": TextCommandInterface.cmd_agent, "agents": TextCommandInterface.cmd_agent,
    "update": TextCommandInterface.cmd_update,
    "approve": TextCommandInterface.cmd_approve,
    "deny": TextCommandInterface.cmd_deny,
    "rollback": TextCommandInterface.cmd_rollback,
    "voice": TextCommandInterface.cmd_voice,
    "build": TextCommandInterface.cmd_build,
    "install": TextCommandInterface.cmd_install,
    "launch": TextCommandInterface.cmd_launch,
    "control": TextCommandInterface.cmd_control,
    "stop": TextCommandInterface.cmd_stop,
    "translate": TextCommandInterface.cmd_translate, "tr": TextCommandInterface.cmd_translate,
    "overseer": TextCommandInterface.cmd_overseer,
    "device": TextCommandInterface.cmd_device, "devices": TextCommandInterface.cmd_device,
    "background": TextCommandInterface.cmd_background, "bg": TextCommandInterface.cmd_background,
    "config": TextCommandInterface.cmd_config,
    "features": TextCommandInterface.cmd_features,
    "audit": TextCommandInterface.cmd_audit,
    "quit": TextCommandInterface.cmd_quit, "exit": TextCommandInterface.cmd_quit,
}


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _dump_config(data: dict[str, Any], indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_config(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {value!r}")
    return "\n".join(lines)