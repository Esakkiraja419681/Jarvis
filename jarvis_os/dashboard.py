"""Dashboard - a single-terminal view across the whole OS.

``_lines()`` renders every subsystem as deterministic, dependency-free text,
so ``render_plain()`` works even with nothing extra installed. ``render()``
lifts the same content into a Rich ``Panel`` (box borders, title) when ``rich``
is present and falls straight back to plain text when it is not - the
dashboard is a convenience view, never a boot requirement.

``Dashboard(kernel)`` reaches for ``kernel.business`` / ``kernel.trading``;
when those have not been wired (a fake kernel in a test), ``_agent()`` builds
standalone agents from the passed config instead, so the view stays truthful
about state on disk either way. Every subsection is guarded: a missing
subsystem renders its line, not a traceback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .business_agent import BusinessAgent
from .trading_bridge import TradingBridge


class Dashboard:
    """Render the live OS state as plain text or a Rich panel."""

    def __init__(self, kernel: Any, config: Any | None = None) -> None:
        self.kernel = kernel
        self.config = config if config is not None else getattr(kernel, "config", None)

    def _agent(self, kind: str) -> Any | None:
        if self.config is None:
            return None
        existing = getattr(self.kernel, kind, None)
        if existing is not None:
            return existing
        try:
            if kind == "business":
                return BusinessAgent(self.kernel, self.config)
            return TradingBridge(self.kernel, self.config)
        except Exception:  # noqa: BLE001
            return None

    def _heartbeat(self) -> dict[str, Any] | None:
        overseer = getattr(self.kernel, "overseer", None)
        path = getattr(overseer, "heartbeat_path", None)
        if path is None and self.config is not None:
            path = self.config.resolve(self.config.get(
                "overseer.heartbeat_path", "data/heartbeat.json"))
        try:
            if path is not None and Path(path).is_file():
                return json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return None

    # ---------------------------------------------------------------- content
    def _header_line(self) -> str:
        uptime = int(time.time() - getattr(self.kernel, "boot_time", time.time()))
        return (f"Jarvis OS | uptime {uptime}s | cycles {self.kernel.cycles} "
                f"| errors {self.kernel.consecutive_errors} "
                f"| queued {len(self.kernel.queue)} | mode "
                f"{getattr(self.kernel, '_mode', 'foreground')}")

    def _llm_lines(self) -> list[str]:
        llm = getattr(self.kernel, "llm", None)
        if llm is None:
            return ["llm: (no client)"]
        online = "yes" if (getattr(llm, "available", lambda: True)()) else "NO"
        return [f"llm: provider={self.config.get('llm.provider') if self.config else '?'} "
                f"model={self.config.get('llm.model') if self.config else '?'} "
                f"online={online}"]

    def _memory_lines(self) -> list[str]:
        memory = getattr(self.kernel, "memory", None)
        if memory is None:
            return ["memory: (no store)"]
        return [f"memory: {self._single_line(memory.status())}"]

    def _index_lines(self) -> list[str]:
        index = getattr(self.kernel, "memory_index", None)
        if index is None:
            return ["index: (none)"]
        try:
            rows = index.count()
        except Exception:  # noqa: BLE001
            rows = "?"
        return [f"index: rows={rows}"]

    def _gate_lines(self) -> list[str]:
        gate = getattr(self.kernel, "gate", None)
        if gate is None:
            return ["gate: (none attached)"]
        try:
            return [f"gate: {self._single_line(gate.report())}"]
        except Exception:  # noqa: BLE001
            return ["gate: report unavailable"]

    def _business_lines(self) -> list[str]:
        agent = self._agent("business")
        if agent is None:
            return ["business: (off)"]
        try:
            return [f"business: {agent.status()}", "", agent.funnel_report()]
        except Exception:  # noqa: BLE001
            return ["business: read failed"]

    def _trading_lines(self) -> list[str]:
        bridge = self._agent("trading")
        if bridge is None:
            return ["trading: (off)"]
        try:
            lines = [f"trading: {bridge.status()}"]
            for row in bridge.pnl().get("positions", [])[:3]:
                lines.append(f"  {row['symbol']} {row['qty']} @ ~{row['avg_price']} "
                             f"uPnL ${row['unrealized']:,.2f}")
            return lines
        except Exception:  # noqa: BLE001
            return ["trading: read failed"]

    def _heartbeat_lines(self) -> list[str]:
        beat = self._heartbeat()
        if beat is None:
            return ["heartbeat: none on disk"]
        return [f"heartbeat: pid={beat.get('pid')} mode={beat.get('mode')} "
                f"cycles={beat.get('cycles')}"]

    def _overseer_lines(self) -> list[str]:
        overseer = getattr(self.kernel, "overseer", None)
        if overseer is None:
            return ["overseer: (not attached)"]
        try:
            return [f"overseer: {self._single_line(overseer.status())}"]
        except Exception:  # noqa: BLE001
            return ["overseer: read failed"]

    def _daemon_lines(self) -> list[str]:
        daemon = getattr(self.kernel, "daemon", None)
        if daemon is None:
            return ["daemon: (off)"]
        try:
            return [f"daemon: {self._single_line(daemon.status())}"]
        except Exception:  # noqa: BLE001
            return ["daemon: read failed"]

    def _voice_lines(self) -> list[str]:
        iface = getattr(self.kernel, "voice_iface", None)
        dump = getattr(self.kernel, "voice_daemon", None)
        parts = []
        if iface is not None:
            try:
                parts.append(f"iface={iface.status()}")
            except Exception:  # noqa: BLE001
                pass
        if dump is not None:
            parts.append(f"daemon={'on' if dump.running else 'off'}")
        return ["voice: " + (" ".join(parts) if parts else "(no interface)")]

    @staticmethod
    def _single_line(text: str) -> str:
        return " ".join(str(text).split())

    def _lines(self) -> list[str]:
        return [
            self._header_line(),
            *self._llm_lines(),
            *self._memory_lines(),
            *self._index_lines(),
            *self._gate_lines(),
            "",
            *self._business_lines(),
            "",
            *self._trading_lines(),
            "",
            *self._heartbeat_lines(),
            *self._overseer_lines(),
            *self._daemon_lines(),
            *self._voice_lines(),
        ]

    # ----------------------------------------------------------------- render
    def render_plain(self) -> str:
        return "\n".join(self._lines())

    def render(self) -> str:
        try:
            from io import StringIO

            from rich.console import Console
            from rich.panel import Panel
        except ImportError:
            return self.render_plain()
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=100)
        console.print(Panel("\n".join(self._lines()),
                            title="[bold]Jarvis OS Dashboard[/bold]",
                            border_style="cyan"))
        return buffer.getvalue().rstrip()