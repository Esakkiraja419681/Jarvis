"""Sandboxed execution for generated code, shell commands and evolution tests.

Two independent mechanisms:

* :class:`PythonSandbox` - runs candidate Python in a *separate process* with a
  scrubbed environment, an isolated temp CWD, wall-clock timeout, an address-space
  cap (POSIX) and best-effort import/OS-call blocking. Nothing the kernel touches
  is importable and no network module is preloaded.
* :class:`ShellSandbox` - command execution with an explicit allowlist, shell
  metacharacter rejection and a timeout.

Everything routes through the PermissionGate before it runs; the sandbox is the
*second* line of defence, not the first.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PRELUDE = r'''
import builtins as _b, sys as _sys, os as _os
_BLOCKED_MODULES = {
    "socket", "socketserver", "http", "urllib", "urllib3", "requests", "ftplib",
    "telnetlib", "smtplib", "asyncio", "ssl", "subprocess", "multiprocessing",
    "ctypes", "pickle", "shutil", "webbrowser", "paramiko", "aiohttp",
}
_REAL_IMPORT = _b.__import__
def _guarded_import(name, *a, **k):
    root = name.split(".")[0]
    if root in _BLOCKED_MODULES:
        raise PermissionError(
            f"sandbox: import of '{root}' is not permitted. Ask {self_honorific()} "
            "to approve elevated execution."
        )
    return _REAL_IMPORT(name, *a, **k)
_b.__import__ = _guarded_import
try:
    _os.environ.clear()
except Exception:
    pass
_os.environ["PATH"] = "/usr/bin:/bin"
_os.environ["HOME"] = _os.getcwd()
_os.environ["SANDBOXED"] = "1"
'''

_SNIPPET = """
def __jarvis_honorific():
    return {honorific!r}
self_honorific = __jarvis_honorific
{SANDBOX_PRELUDE}
"""


@dataclass(slots=True)
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    duration_ms: int = 0
    timed_out: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "stdout": self.stdout, "stderr": self.stderr,
            "returncode": self.returncode, "duration_ms": self.duration_ms,
            "timed_out": self.timed_out, "truncated": self.truncated,
        }


class SandboxError(RuntimeError):
    pass


class PythonSandbox:
    def __init__(self, config, root: str | os.PathLike[str] | None = None,
                 gate=None, audit=None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        sb = config.get_dict("sandbox")
        self.timeout = float(sb.get("timeout_seconds", 30))
        self.memory_mb = int(sb.get("memory_mb", 512))
        self.network = bool(sb.get("network", False))
        self.python = str(sb.get("python_executable", sys.executable or "python"))
        self.root = Path(root) if root else Path(sb.get("workdir", "data/sandbox"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs = 0

    # ------------------------------------------------------------- helpers
    def _preload(self) -> str:
        """Memory caps are POSIX-only; preload a rlimit where supported."""
        if os.name == "nt":
            return ""
        return (
            f"import resource\n"
            f"resource.setrlimit(resource.RLIMIT_AS, "
            f"({self.memory_mb * 1024 * 1024}, {self.memory_mb * 1024 * 1024}))\n"
        )

    def _wrapper(self) -> str:
        honorific = self.config.honorific()
        return _SNIPPET.replace("{honorific!r}", repr(honorific)) \
                        .replace("{SANDBOX_PRELUDE}", PRELUDE + self._preload())

    # ----------------------------------------------------------------- run
    def run(self, code: str, timeout: float | None = None,
            files: dict[str, str] | None = None,
            category: str = "code_exec",
            require_permission: bool = True) -> SandboxResult:
        timeout = float(timeout or self.timeout)
        if require_permission and self.gate is not None:
            if not self.gate.approve(
                category,
                "Jarvis requests permission to execute code in the sandbox.",
                detail={"bytes": len(code), "timeout": timeout},
                source="text",
            ):
                self._log("code", "sandbox.run", "denied", {"bytes": len(code)})
                return SandboxResult(ok=False, stderr="permission denied by operator", returncode=-2)

        # self.root is created once in __init__, but it can disappear before a
        # later call (a crashed run's cleanup, a temp cleaner, AV quarantine).
        # mkdtemp then raises a bare WinError 3 naming a path nobody can act on,
        # so re-create the root here and report it explicitly if that fails too.
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            workdir = Path(tempfile.mkdtemp(prefix="jbx_", dir=str(self.root)))
        except OSError as exc:
            raise SandboxError(
                f"sandbox root unavailable: {self.root} ({exc})"
            ) from exc
        try:
            for name, content in (files or {}).items():
                target = workdir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            (workdir / "_jarvis_main.py").write_text(
                self._wrapper() + "\n\n# ---- user code ----\n" + code,
                encoding="utf-8",
            )
            return self._spawn(workdir, timeout)
        finally:
            try:
                for child in sorted(workdir.rglob("*"), reverse=True):
                    try:
                        child.unlink() if child.is_file() else child.rmdir()
                    except OSError:
                        pass
                workdir.rmdir()
            except OSError:
                pass

    def _spawn(self, workdir: Path, timeout: float) -> SandboxResult:
        started = time.time()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SANDBOXED": "1",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "TEMP": str(workdir),
            "TMP": str(workdir),
            "HOME": str(workdir),
        }
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                [self.python, "-I", "-S", "_jarvis_main.py"],
                cwd=str(workdir), env=env, capture_output=True, text=True,
                timeout=timeout, creationflags=creation,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = f"TIMEOUT after {timeout}s - process killed by sandbox"
            code, timed_out = -9, True
        except FileNotFoundError as exc:
            stdout, stderr, code, timed_out = "", f"python not found: {exc}", -1, False

        truncated = len(stdout) > 20000
        result = SandboxResult(
            ok=(code == 0 and not timed_out),
            stdout=stdout[:20000], stderr=stderr[:8000], returncode=code,
            duration_ms=int((time.time() - started) * 1000), timed_out=timed_out,
            truncated=truncated,
        )
        self.runs += 1
        self._log("code", "sandbox.run", "ok" if result.ok else "failed",
                  {"returncode": code, "ms": result.duration_ms, "timed_out": timed_out})
        return result

    def _log(self, cat: str, action: str, decision: str, detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record(cat, action, decision, detail)


ALLOWED_COMMANDS: dict[str, set[str]] = {
    "safe": {
        "ls", "dir", "pwd", "cd", "echo", "cat", "type", "head", "tail", "wc",
        "date", "time", "whoami", "hostname", "uname", "ver", "python", "python3",
        "pip", "git", "node", "npm", "pytest", "env", "find", "grep", "rg",
        "tree", "stat", "du", "df", "which", "where", "tasklist", "ipconfig",
    },
}

_META = frozenset("|&;<>`$(){}[]*?!#\n\t\\\"'")


class ShellSandbox:
    """Allowlisted shell execution. Metacharacters are rejected outright."""

    def __init__(self, config, gate=None, audit=None,
                 workdir: str | os.PathLike[str] | None = None,
                 allowed: set[str] | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.workdir = Path(workdir or Path(config.get("sandbox.workdir", "data/sandbox")))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.allowed = set(allowed or ALLOWED_COMMANDS["safe"])
        self.history: list[dict[str, Any]] = []

    def validate(self, command: str) -> tuple[bool, str]:
        command = str(command).strip()
        if not command:
            return False, "empty command"
        bad = _META & set(command)
        if bad:
            return False, f"shell metacharacters rejected: {sorted(bad)}"
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return False, f"unparsable command: {exc}"
        if not parts:
            return False, "empty command"
        head = Path(parts[0]).name.lower()
        if head not in self.allowed:
            return False, f"command '{head}' is not on the allowlist"
        return True, "ok"

    def run(self, command: str, timeout: float = 60.0, cwd: str | None = None,
            category: str = "shell_exec",
            require_permission: bool = True) -> SandboxResult:
        valid, reason = self.validate(command)
        if not valid:
            if self.audit:
                self.audit.record("shell", f"reject:{command[:60]}", "rejected", {"reason": reason})
            return SandboxResult(ok=False, stderr=reason, returncode=-2)

        if require_permission and self.gate is not None:
            if not self.gate.approve(
                category,
                f"Jarvis requests permission to run a shell command:\n  {command}",
                detail={"command": command, "cwd": cwd or str(self.workdir)},
            ):
                if self.audit:
                    self.audit.record("shell", f"deny:{command[:60]}", "denied")
                return SandboxResult(ok=False, stderr="permission denied by operator", returncode=-2)

        started = time.time()
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                command, shell=True, cwd=cwd or str(self.workdir),
                capture_output=True, text=True, timeout=timeout,
                creationflags=creation,
            )
            result = SandboxResult(
                ok=proc.returncode == 0, stdout=proc.stdout[:20000],
                stderr=proc.stderr[:8000], returncode=proc.returncode,
                duration_ms=int((time.time() - started) * 1000),
            )
        except subprocess.TimeoutExpired:
            result = SandboxResult(ok=False, stderr=f"TIMEOUT after {timeout}s", returncode=-9)
        except OSError as exc:
            result = SandboxResult(ok=False, stderr=str(exc), returncode=-1)

        self.history.append({"command": command, "ok": result.ok, "ts": time.time()})
        self.history = self.history[-200:]
        if self.audit:
            self.audit.record("shell", f"run:{command[:60]}", "ok" if result.ok else "failed",
                              {"returncode": result.returncode})
        return result