"""ControlBridge - three ways to inspect and control built software.

* **subprocess** - direct process management (start/stop/restart, stdout capture).
* **IPC**        - a local JSON command file that any process can write to and
  Jarvis picks up; a zero-dependency control channel.
* **REST**       - a loopback-only HTTP endpoint (``control <cmd>`` over
  ``http://127.0.0.1:8770/control``), bound to localhost and token-protected.

All three funnel into :meth:`ControlBridge.dispatch`, so there is exactly one
authorisation and audit path regardless of which channel a command arrives on.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ACTIONS = ("start", "stop", "restart", "status", "inspect", "logs", "configure", "signal")


@dataclass(slots=True)
class ManagedProcess:
    program_id: str
    argv: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    pid: int | None = None
    status: str = "built"
    started_at: float | None = None
    stopped_at: float | None = None
    exit_code: int | None = None
    log_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.program_id, "argv": self.argv, "cwd": self.cwd, "pid": self.pid,
            "status": self.status, "started_at": self.started_at,
            "stopped_at": self.stopped_at, "exit_code": self.exit_code,
            "log_path": self.log_path,
        }


class ControlBridge:
    def __init__(self, config, gate=None, audit=None,
                 root: str | os.PathLike[str] | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.root = Path(root or "data/control").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ipc_path = self.root / "command.json"
        self.response_path = self.root / "response.json"
        self.rest_host = str(config.get("control_bridge.rest_host", "127.0.0.1"))
        self.rest_port = int(config.get("control_bridge.rest_port", 8770))
        self.processes: dict[str, ManagedProcess] = {}
        self.token = self._token()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.handler: Callable[[str, list[str]], Any] | None = None
        self.command_count = 0

    # ----------------------------------------------------------------- token
    def _token(self) -> str:
        import secrets
        path = Path(self.config.get("control_bridge.token_file", "data/control.token"))
        if not path.is_absolute():
            path = self.config.root / path
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_hex(24)
        path.write_text(value, encoding="utf-8")
        return value

    # ------------------------------------------------------------- dispatch
    def dispatch(self, program_id: str, action: str,
                 args: list[str] | None = None) -> dict[str, Any]:
        action = str(action).lower()
        self.command_count += 1
        if action not in ACTIONS:
            return {"ok": False, "error": f"unknown action '{action}'. "
                                         f"Valid: {', '.join(ACTIONS)}"}
        if self.gate is not None:
            if not self.gate.approve(
                "software_control",
                f"Jarvis requests permission to {action} software '{program_id}'.",
                detail={"program": program_id, "action": action, "args": args or []},
            ):
                if self.audit:
                    self.audit.record("control", f"deny:{program_id}:{action}", "denied")
                return {"ok": False, "denied": True,
                        "error": "operator denied the control command"}
        handler = {
            "start": self._start, "stop": self._stop, "restart": self._restart,
            "status": self._status, "inspect": self._inspect, "logs": self._logs,
            "configure": self._configure, "signal": self._signal,
        }[action]
        try:
            result = handler(program_id, args or [])
        except KeyError:
            return {"ok": False, "error": f"unknown program '{program_id}'"}
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self.audit:
            self.audit.record("control", f"{action}:{program_id}",
                              "ok" if result.get("ok") else "failed",
                              {"args": args or []})
        return result

    # -------------------------------------------------------------- register
    def register(self, program_id: str, argv: list[str], cwd: str | os.PathLike[str],
                 env: dict[str, str] | None = None) -> ManagedProcess:
        proc = ManagedProcess(
            program_id=program_id, argv=list(argv), cwd=str(cwd),
            env=dict(env or {}), log_path=str(self.root / f"{program_id}.log"),
        )
        self.processes[program_id] = proc
        return proc

    # ------------------------------------------------------------- lifecycle
    def _start(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        if proc.pid and self._alive(proc):
            return {"ok": True, "status": "already_running", "pid": proc.pid}
        argv = proc.argv + args
        env = {**os.environ, **proc.env, "JARVIS_MANAGED": "1", "JARVIS_PROGRAM": program_id}
        log = open(proc.log_path, "a", encoding="utf-8")  # noqa: SIM115 - lifetime = process
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
            getattr(subprocess, "DETACHED_PROCESS", 0)
        try:
            popen = subprocess.Popen(
                argv, cwd=proc.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=creation,
            )
        except (OSError, ValueError) as exc:
            log.close()
            return {"ok": False, "error": f"failed to start: {exc}"}
        proc.pid, proc.status, proc.started_at = popen.popen.pid, "running", time.time()
        proc.stopped_at, proc.exit_code = None, None
        return {"ok": True, "status": "running", "pid": proc.pid,
                "message": f"'{program_id}' started as pid {proc.pid}"}

    def _stop(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        if not proc.pid or not self._alive(proc):
            proc.status = "stopped"
            return {"ok": True, "status": "stopped", "message": "was not running"}
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False)
            else:
                import signal as sigmod
                os.killpg(os.getpgid(proc.pid), sigmod.SIGTERM)
                time.sleep(0.6)
                if self._alive(proc):
                    os.killpg(os.getpgid(proc.pid), sigmod.SIGKILL)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"failed to stop: {exc}"}
        proc.status, proc.stopped_at, proc.pid = "stopped", time.time(), None
        return {"ok": True, "status": "stopped", "message": f"'{program_id}' stopped"}

    def _restart(self, program_id: str, args: list[str]) -> dict[str, Any]:
        self._stop(program_id, args)
        time.sleep(0.4)
        return self._start(program_id, args)

    def _status(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        alive = bool(proc.pid) and self._alive(proc)
        proc.status = "running" if alive else ("stopped" if proc.stopped_at else "built")
        uptime = (time.time() - proc.started_at) if (alive and proc.started_at) else 0.0
        return {"ok": True, "program": proc.to_dict(), "alive": alive,
                "uptime_seconds": round(uptime, 1)}

    def _inspect(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        info: dict[str, Any] = {**proc.to_dict(), "alive": bool(proc.pid) and self._alive(proc)}
        try:
            import psutil  # optional
            if proc.pid:
                p = psutil.Process(proc.pid)
                info["cpu_percent"] = p.cpu_percent(interval=0.1)
                info["memory_mb"] = round(p.memory_info().rss / 1_048_576, 2)
                info["threads"] = p.num_threads()
        except Exception:  # noqa: BLE001
            info["note"] = "install psutil for cpu/memory introspection"
        return {"ok": True, "program": info}

    def _logs(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        lines = int(args[0]) if args and str(args[0]).isdigit() else 100
        path = Path(proc.log_path)
        if not path.is_file():
            return {"ok": True, "logs": [], "note": "no log file yet"}
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "logs": tail, "path": proc.log_path}

    def _configure(self, program_id: str, args: list[str]) -> dict[str, Any]:
        """Write an env override / config file for a program, then restart it."""
        proc = self._require(program_id)
        if not args:
            return {"ok": False, "error": "usage: configure KEY=VALUE [KEY=VALUE ...]"}
        for pair in args:
            key, sep, value = str(pair).partition("=")
            if not sep:
                return {"ok": False, "error": f"bad pair '{pair}' (expected KEY=VALUE)"}
            proc.env[key.strip()] = value.strip()
        env_file = Path(proc.cwd) / ".jarvis.env"
        env_file.write_text("\n".join(f"{k}={v}" for k, v in proc.env.items()) + "\n",
                            encoding="utf-8")
        restarted = None
        if proc.pid:
            restarted = self._restart(program_id, [])
        return {"ok": True, "configured": sorted(proc.env), "env_file": str(env_file),
                "restarted": bool(restarted and restarted.get("ok"))}

    def _signal(self, program_id: str, args: list[str]) -> dict[str, Any]:
        proc = self._require(program_id)
        if not proc.pid or not self._alive(proc):
            return {"ok": False, "error": "not running"}
        import signal as sigmod
        name = (args[0] if args else "SIGTERM").upper().replace("SIG", "")
        sig = getattr(sigmod, name if name.startswith("SIG") else f"SIG{name}", None)
        if sig is None:
            return {"ok": False, "error": f"unknown signal '{name}'"}
        os.kill(proc.pid, sig)
        return {"ok": True, "signalled": name, "pid": proc.pid}

    def _require(self, program_id: str) -> ManagedProcess:
        proc = self.processes.get(program_id)
        if proc is None:
            raise KeyError(program_id)
        return proc

    @staticmethod
    def _alive(proc: ManagedProcess) -> bool:
        if not proc.pid:
            return False
        try:
            if os.name == "nt":
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, proc.pid)
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            os.kill(proc.pid, 0)
            return True
        except (OSError, AttributeError):
            return False

    # ------------------------------------------------------------------ IPC
    def submit(self, program_id: str, action: str, args: list[str] | None = None) -> None:
        """Write a command for the kernel to pick up (cross-process control)."""
        payload = {"program": program_id, "action": action, "args": args or [],
                   "ts": time.time(), "token": self.token}
        tmp = self.ipc_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.ipc_path)

    def poll(self) -> list[dict[str, Any]]:
        """Drain and execute queued IPC commands."""
        if not self.ipc_path.is_file():
            return []
        try:
            payload = json.loads(self.ipc_path.read_text(encoding="utf-8"))
            self.ipc_path.unlink()
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, dict):
            return []
        if payload.get("token") != self.token:
            if self.audit:
                self.audit.record("control", "ipc", "refused", {"reason": "bad token"})
            return [{"ok": False, "error": "bad IPC token"}]
        result = self.dispatch(str(payload.get("program", "")),
                               str(payload.get("action", "status")),
                               list(payload.get("args") or []))
        try:
            self.response_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return [result]

    # ----------------------------------------------------------------- REST
    def start_rest(self) -> str:
        if self._server is not None:
            return f"REST control already listening on {self.rest_host}:{self.rest_port}"
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:  # silence stdlib logging
                return

            def _reply(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/health":
                    self._reply({"ok": True, "service": "jarvis-control",
                                 "programs": list(bridge.processes)})
                    return
                if parsed.path == "/control":
                    query = urllib.parse.parse_qs(parsed.query)
                    token = (query.get("token") or [""])[0]
                    if token != bridge.token:
                        self._reply({"ok": False, "error": "unauthorised"}, 403)
                        return
                    program = (query.get("program") or [""])[0]
                    action = (query.get("action") or ["status"])[0]
                    args = [a for a in (query.get("args") or [])]
                    self._reply(bridge.dispatch(program, action, args))
                    return
                self._reply({"ok": False, "error": "not found"}, 404)

        try:
            server = ThreadingHTTPServer((self.rest_host, self.rest_port), Handler)
        except OSError as exc:
            return f"REST control could not bind {self.rest_host}:{self.rest_port}: {exc}"
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True,
                                        name="jarvis-control-rest")
        self._thread.start()
        if self.audit:
            self.audit.record("control", "rest_start", "ok",
                              {"host": self.rest_host, "port": self.rest_port})
        return f"REST control listening on http://{self.rest_host}:{self.rest_port}/control"

    def stop_rest(self) -> str:
        if self._server is None:
            return "REST control is not running."
        self._server.shutdown()
        self._server.server_close()
        self._server, self._thread = None, None
        return "REST control stopped."

    def rest_url(self, program: str = "", action: str = "status",
                 args: Sequence[str] = ()) -> str:
        query = urllib.parse.urlencode(
            {"token": self.token, "program": program, "action": action,
             "args": list(args), "safe": ""}, doseq=True)
        return f"http://{self.rest_host}:{self.rest_port}/control?{query}"

    # ---------------------------------------------------------------- status
    def status(self) -> str:
        lines = [f"ControlBridge: {len(self.processes)} managed program(s), "
                 f"{self.command_count} commands, ipc={self.ipc_path.name}, "
                 f"rest={'up' if self._server else 'down'}"]
        for proc in self.processes.values():
            lines.append(f"  {proc.program_id:<20} {proc.status:<10} pid={proc.pid}")
        return "\n".join(lines)

    def shutdown_all(self) -> int:
        stopped = 0
        for program_id in list(self.processes):
            if self._status(program_id, []).get("alive"):
                if self.dispatch(program_id, "stop").get("ok"):
                    stopped += 1
        self.stop_rest()
        return stopped