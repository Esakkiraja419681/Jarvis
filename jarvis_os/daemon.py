"""BackgroundDaemon - keep Jarvis OS alive when the terminal window closes.

Mechanism
---------
* ``start`` re-launches the kernel in a **detached** process
  (``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` on Windows, ``setsid`` on
  POSIX) with stdin/stdout/stderr detached from any console, then returns
  immediately. The window can be closed and Jarvis keeps running.
* A pidfile plus a heartbeat file let ``status``, ``stop`` and the foreground
  recall find the daemon.
* ``foreground`` (single command recall) attaches a live listener to the daemon
  via the same IPC/REST control surface, so the operator talks to the *running*
  background instance rather than starting a second one.

All of it is gated behind the ``background_daemon`` permission category.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DaemonState:
    pid: int | None = None
    started_at: float | None = None
    mode: str = "foreground"
    entry: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "started_at": self.started_at,
                "mode": self.mode, "entry": self.entry,
                "uptime_seconds": round(time.time() - self.started_at, 1)
                if self.started_at else 0}


class BackgroundDaemon:
    def __init__(self, config, gate=None, audit=None,
                 root: str | os.PathLike[str] | None = None,
                 entry: str | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.root = Path(root or config.root)
        self.entry = Path(entry or (self.root / "jarvis.py")).resolve()
        self.pidfile = self.root / "data" / "jarvis.pid"
        self.statefile = self.root / "data" / "daemon_state.json"
        self.logfile = self.root / "data" / "jarvis_daemon.log"
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)
        self.state = DaemonState()
        self.load()

    # ------------------------------------------------------------- pid state
    def load(self) -> "BackgroundDaemon":
        if self.statefile.is_file():
            try:
                raw = json.loads(self.statefile.read_text(encoding="utf-8"))
                self.state = DaemonState(
                    pid=raw.get("pid"), started_at=raw.get("started_at"),
                    mode=raw.get("mode", "foreground"), entry=raw.get("entry", ""),
                )
            except (json.JSONDecodeError, OSError, ValueError):
                self.state = DaemonState()
        if self.statefile.is_file() and not self.alive():
            self.state.mode = "stopped"
            self.state.pid = None
        return self

    def save(self) -> None:
        self.statefile.write_text(
            json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ query
    def pid(self) -> int | None:
        if self.statefile.is_file() and not self.state.pid:
            try:
                raw = json.loads(self.statefile.read_text(encoding="utf-8"))
                self.state.pid = raw.get("pid")
            except (json.JSONDecodeError, OSError):
                return None
        return self.state.pid

    def alive(self) -> bool:
        pid = self.pid()
        if not pid:
            return False
        try:
            if os.name == "nt":
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
                if not handle:
                    return False
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                return code.value == 259  # STILL_ACTIVE
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError, AttributeError):
            return False

    def status(self) -> str:
        self.load()
        if not self.alive():
            return ("BackgroundDaemon: not running. "
                    "Start it with: background start")
        return (f"BackgroundDaemon: RUNNING (pid {self.state.pid}, "
                f"up {self.state.to_dict()['uptime_seconds']}s, "
                f"mode={self.state.mode}) | log: {self.logfile.name}")

    # ---------------------------------------------------------------- control
    def command(self, action: str, args: list[str] | None = None) -> str:
        action = (action or "status").lower()
        args = args or []
        return {
            "start": self.start, "stop": self.stop, "restart": self.restart,
            "status": lambda _a: self.status(), "foreground": self.recall,
            "foreground": self.recall, "logs": self.logs, "enable": self.enable,
        }.get(action, lambda _a: (
            "Usage: background [start | stop | restart | status | foreground | logs | enable]"))(args)

    def enable(self, args: list[str] | None = None) -> str:
        self.config.set_feature("background_daemon", True)
        return "Background daemon mode enabled in config. Use: background start"

    def start(self, args: list[str] | None = None) -> str:
        self.load()
        if self.alive():
            return (f"Already running as pid {self.state.pid}. "
                    f"Recall it with: background foreground")
        if self.gate is not None and not self.gate.approve(
            "background_daemon",
            "Jarvis requests permission to start itself in the background.\n"
            "It will keep running after you close this window.",
            detail={"entry": str(self.entry)},
        ):
            self._log("start", "denied")
            return "permission denied by operator - background mode not started"
        if not self.entry.is_file():
            return f"entry point not found: {self.entry}"

        self.config.set_feature("background_daemon", True)
        argv = [sys.executable or "python", "-u", str(self.entry), "--daemon"]
        argv.extend(args or [])
        log = open(self.logfile, "a", encoding="utf-8")  # noqa: SIM115 - owned by the daemon
        creation = 0
        if os.name == "nt":
            creation = (getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                        | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
            popen = subprocess.Popen(argv, cwd=str(self.root), stdout=log, stderr=log,
                                     stdin=subprocess.DEVNULL, creationflags=creation,
                                     close_fds=True)
        else:
            popen = subprocess.Popen(argv, cwd=str(self.root), stdout=log, stderr=log,
                                     stdin=subprocess.DEVNULL, start_new_session=True)

        self.state = DaemonState(pid=popen.pid, started_at=time.time(),
                                 mode="background", entry=str(self.entry))
        self.pidfile.write_text(str(popen.pid), encoding="utf-8")
        self.save()
        self._log("start", "ok", {"pid": popen.pid})
        return (f"Jarvis is now running in the background as pid {popen.pid}.\n"
                f"Close this window safely. Bring it back with a single command:\n"
                f"  background foreground\n"
                f"Log: {self.logfile}")

    def stop(self, args: list[str] | None = None) -> str:
        self.load()
        pid = self.pid()
        if not pid or not self.alive():
            return "BackgroundDaemon is not running."
        if self.gate is not None and not self.gate.approve(
            "background_daemon", f"Stop the background Jarvis OS (pid {pid})?",
            detail={"pid": pid},
        ):
            return "permission denied by operator"
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=20, check=False)
            else:
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1.0)
                if self.alive():
                    os.kill(int(pid), signal.SIGKILL)
        except Exception as exc:  # noqa: BLE001
            return f"failed to stop: {exc}"
        self.state.mode, self.state.pid = "stopped", None
        self.pidfile.unlink(missing_ok=True)
        self.save()
        self._log("stop", "ok", {"pid": pid})
        return f"Background Jarvis OS stopped (pid {pid}). Memory is intact."

    def restart(self, args: list[str] | None = None) -> str:
        self.stop()
        time.sleep(0.5)
        return self.start(args)

    def logs(self, args: list[str] | None = None) -> str:
        if not self.logfile.is_file():
            return "no daemon log yet."
        lines = int(args[0]) if args and str(args[0]).isdigit() else 60
        try:
            tail = self.logfile.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"cannot read log: {exc}"
        return "\n".join(tail[-lines:])

    # ---------------------------------------------------------------- recall
    def recall(self, args: list[str] | None = None) -> str:
        """Bring the running background instance back to the foreground."""
        self.load()
        if not self.alive():
            return "BackgroundDaemon is not running. Start it with: background start"
        return self.attach(args or [])

    def attach(self, args: list[str]) -> str:
        """Reattach the foreground to the kernel this process already holds.

        A mid-session recall and a cold ``python jarvis.py --attach`` are the same
        operation, and neither may boot anything: the kernel bound at construction
        is reused, so memory is not re-read from disk and the boot count does not
        move. When a foreground session is already live the interactive loop is
        left strictly alone - calling ``start()`` again from inside it nests a
        second ``input()`` loop within the running one and prints a second banner
        mid-conversation.
        """
        self.load()
        if not self.alive():
            return "BackgroundDaemon is not running. Start it with: background start"
        kernel = getattr(self, "_kernel", None)
        if kernel is None:
            return ("No kernel handle is bound to this process, Sir. Recall from a session "
                    "started with 'python jarvis.py', or attach a cold one with: "
                    "python jarvis.py --attach")
        iface = kernel.text_iface
        stats = getattr(kernel.memory, "stats", None) or {}
        boots = stats.get("boots", 0)
        self._attachments = getattr(self, "_attachments", 0) + 1

        if getattr(iface, "running", False):
            # Already in the foreground: same kernel, same loop, nothing restarted.
            iface.prompt = "jarvis(fg)> "
            return (f"Reattached to the live session, Sir - same kernel, {boots} boot(s), "
                    f"memory untouched, nothing restarted. Background instance pid "
                    f"{self.state.pid} keeps serving the loop.")

        iface.running = True
        iface.banner()
        iface.prompt = "jarvis(fg)> "
        iface.say(f"I am back in the foreground, Sir. Reattached to the live kernel "
                  f"({boots} boot(s)) - nothing was restarted and memory is intact. "
                  f"Background instance pid {self.state.pid} keeps serving the loop.")
        iface.start()
        return "Foreground session ended; the background instance keeps running."

    def bind_kernel(self, kernel: Any) -> None:
        self._kernel = kernel

    def _log(self, action: str, decision: str, detail: dict[str, Any] | None = None) -> None:
        if self.audit:
            self.audit.record("daemon", action, decision, detail or {})