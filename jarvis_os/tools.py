"""ToolRegistry - the capability surface Jarvis can reach for.

Built in: file read/write/delete/list, shell exec, web fetch, code runner,
HTTP service probe, and a few local introspection helpers.

Every tool declares a ``permission_category``. The registry refuses to run a
gated tool without a live :class:`~jarvis_os.permissions.PermissionGate`
approval, and records the outcome in the audit ledger. Tools, subagents and
even new tools can be registered **at runtime** - the registry is intentionally
mutable, which is what the evolution engine and the ``skill add`` command use.
"""

from __future__ import annotations

import importlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .llm import ToolSpec


@dataclass(slots=True)
class RegisteredTool:
    name: str
    description: str
    handler: Callable[..., Any]
    spec: ToolSpec
    permission_category: str = "file_read"
    dangerous: bool = False
    builtin: bool = True
    registered_at: float = field(default_factory=time.time)
    calls: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "permission_category": self.permission_category,
            "dangerous": self.dangerous, "builtin": self.builtin,
            "calls": self.calls, "failures": self.failures,
            "parameters": self.spec.parameters,
        }


class ToolError(RuntimeError):
    pass


_SECRET_KEYS = {"password", "passwd", "secret", "token", "api_key", "apikey",
                "authorization", "auth", "credential", "credentials", "private_key"}


def _redact(arguments: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    """Never let secrets reach the audit ledger."""
    out: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if any(s in str(key).lower() for s in _SECRET_KEYS):
            out[key] = "***redacted***"
        elif isinstance(value, str) and len(value) > limit:
            out[key] = value[:limit] + f"...(+{len(value) - limit} chars)"
        elif isinstance(value, dict):
            out[key] = _redact(value, limit)
        else:
            out[key] = value
    return out


class ToolRegistry:
    def __init__(self, config, gate=None, audit=None, memory=None,
                 sandbox=None, shell=None, root: str | os.PathLike[str] | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.memory = memory
        self.sandbox = sandbox
        self.shell = shell
        self.root = Path(root or ".").resolve()
        self._tools: dict[str, RegisteredTool] = {}
        self._install_builtins()

    # ------------------------------------------------------------- registry
    def register(self, name: str, handler: Callable[..., Any], description: str,
                 parameters: dict[str, Any] | None = None,
                 permission_category: str = "file_read",
                 dangerous: bool = False, builtin: bool = False,
                 replace: bool = False) -> RegisteredTool:
        name = self._validate_name(name)
        if name in self._tools and not replace:
            raise ToolError(f"tool '{name}' already registered; pass replace=True")
        tool = RegisteredTool(
            name=name, description=description, handler=handler,
            spec=ToolSpec(name=name, description=description, parameters=parameters or {}),
            permission_category=permission_category, dangerous=dangerous, builtin=builtin,
        )
        self._tools[name] = tool
        if self.audit:
            self.audit.record("tool", f"register:{name}", "ok",
                              {"category": permission_category, "builtin": builtin})
        return tool

    def register_module_tool(self, module_path: str, attr: str = "run",
                            name: str | None = None) -> RegisteredTool:
        """Dynamic runtime registration from ``pkg.module:function``."""
        mod_path, _, func_name = module_path.partition(":")
        func_name = func_name or attr
        module = importlib.import_module(mod_path)
        handler = getattr(module, func_name)
        return self.register(
            name or f"{mod_path.split('.')[-1]}_{func_name}",
            handler,
            description=(handler.__doc__ or f"runtime tool from {mod_path}").strip()[:300],
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
            permission_category="code_exec", dangerous=True, builtin=False, replace=True,
        )

    def unregister(self, name: str) -> bool:
        tool = self._tools.pop(name, None)
        if tool and self.audit:
            self.audit.record("tool", f"unregister:{name}", "ok", {})
        return bool(tool)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, only_safe: bool = False) -> list[ToolSpec]:
        return [
            t.spec for t in sorted(self._tools.values(), key=lambda x: x.name)
            if not (only_safe and t.dangerous)
        ]

    def to_schema(self) -> list[dict[str, Any]]:
        return [s.to_ollama() for s in self.specs()]

    def describe(self) -> str:
        lines = ["Registered tools:"]
        for tool in sorted(self._tools.values(), key=lambda x: x.name):
            mark = "!" if tool.dangerous else " "
            lines.append(f"  {mark} {tool.name:<24} [{tool.permission_category}] "
                         f"calls={tool.calls} fails={tool.failures} - {tool.description[:60]}")
        return "\n".join(lines)

    @staticmethod
    def _validate_name(name: str) -> str:
        cleaned = str(name).strip()
        if not cleaned or not all(c.isalnum() or c == "_" for c in cleaned):
            raise ToolError(
                f"invalid tool name '{name}' - use letters, digits and underscore only"
            )
        return cleaned

    # -------------------------------------------------------------- execute
    def call(self, name: str, arguments: dict[str, Any] | None = None,
             require_permission: bool = True) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool '{name}'")

        category = tool.permission_category
        if require_permission and self.gate is not None:
            detail = {"tool": name, "arguments": _redact(arguments or {})}
            if not self.gate.approve(
                category, f"Jarvis requests permission to use tool '{name}' ({category}).",
                detail=detail,
            ):
                tool.calls += 1
                tool.failures += 1
                if self.audit:
                    self.audit.record("tool", f"call:{name}", "denied", detail)
                return {"ok": False, "error": "permission denied by operator",
                        "tool": name, "denied": True}

        tool.calls += 1
        started = time.time()
        try:
            result = tool.handler(**(arguments or {}))
        except TypeError as exc:
            tool.failures += 1
            result = {"ok": False, "error": f"bad arguments for '{name}': {exc}"}
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the OS
            tool.failures += 1
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": name}
        if self.audit:
            self.audit.record(
                "tool", f"call:{name}", "ok",
                {"ms": int((time.time() - started) * 1000),
                 "ok": bool(isinstance(result, dict) and result.get("ok", True))},
            )
        if self.memory is not None and isinstance(result, dict) and not result.get("ok", True):
            self.memory.learn_lesson(
                topic=f"tool:{name}", lesson=f"tool '{name}' failed: {result.get('error')}",
                outcome="failure", weight=0.6,
            )
        return result

    # ------------------------------------------------------------- builtins
    def _install_builtins(self) -> None:
        schema_str = {"type": "object",
                      "properties": {"path": {"type": "string"}}, "required": ["path"]}

        self.register("file_read", lambda path: self._file_read(path),
                      "Read a UTF-8 text file (relative paths are resolved inside Jarvis root).",
                      schema_str, permission_category="file_read", builtin=True)

        self.register("file_write", self._file_write,
                      "Create or overwrite a text file.",
                      {"type": "object", "properties": {
                          "path": {"type": "string"}, "content": {"type": "string"},
                          "append": {"type": "boolean"}}, "required": ["path", "content"]},
                      permission_category="file_write", dangerous=True, builtin=True)

        self.register("file_delete", self._file_delete,
                      "Delete a file or directory inside Jarvis root.",
                      {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
                      permission_category="file_delete", dangerous=True, builtin=True)

        self.register("dir_list", self._dir_list,
                      "List a directory inside Jarvis root.",
                      {"type": "object", "properties": {
                          "path": {"type": "string"}, "recursive": {"type": "boolean"}},
                       "required": ["path"]},
                      permission_category="dir_list", builtin=True)

        self.register("shell_exec", self._shell_exec,
                      "Run an allowlisted shell command in the sandbox.",
                      {"type": "object", "properties": {
                          "command": {"type": "string"},
                          "timeout": {"type": "number"}}, "required": ["command"]},
                      permission_category="shell_exec", dangerous=True, builtin=True)

        self.register("web_fetch", self._web_fetch,
                      "Fetch a URL over HTTP(S) and return text (never used to reach AI APIs).",
                      {"type": "object", "properties": {
                          "url": {"type": "string"}, "timeout": {"type": "number"},
                          "max_bytes": {"type": "number"}}, "required": ["url"]},
                      permission_category="web_fetch", dangerous=True, builtin=True)

        self.register("code_runner", self._code_runner,
                      "Execute Python code inside the sandbox and return stdout/stderr.",
                      {"type": "object", "properties": {
                          "code": {"type": "string"}, "timeout": {"type": "number"}},
                       "required": ["code"]},
                      permission_category="code_exec", dangerous=True, builtin=True)

        self.register("ollama_models", self._ollama_models,
                      "List models available in the local Ollama daemon.",
                      {"type": "object", "properties": {}},
                      permission_category="file_read", builtin=True)

        self.register("memory_recall", lambda query="", limit=5: self._memory_recall(query, limit),
                      "Semantic recall from Jarvis persistent memory.",
                      {"type": "object", "properties": {
                          "query": {"type": "string"}, "limit": {"type": "number"}}},
                      permission_category="memory_read", builtin=True)

        self.register("memory_remember", lambda content, kind="fact", importance=0.5: self._memory_remember(content, kind, importance),
                      "Store a durable fact in Jarvis memory.",
                      {"type": "object", "properties": {
                          "content": {"type": "string"}, "kind": {"type": "string"},
                          "importance": {"type": "number"}}, "required": ["content"]},
                      permission_category="self_learn", dangerous=True, builtin=True)

    # -------------------------------------------------------- path safety
    def _resolve(self, raw: str) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolError(
                f"path '{raw}' escapes Jarvis root ({self.root}); refused for safety"
            ) from exc
        return resolved

    # ----------------------------------------------------------- handlers
    def _file_read(self, path: str, max_bytes: int = 400_000) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            return {"ok": False, "error": f"not a file: {path}"}
        size = target.stat().st_size
        if size > max_bytes:
            return {"ok": False, "error": f"file too large ({size} bytes, cap {max_bytes})"}
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(target.relative_to(self.root)),
                "bytes": size, "content": text}

    def _file_write(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if append:
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(content)
            else:
                target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(target.relative_to(self.root)),
                "bytes": len(content.encode("utf-8")), "appended": bool(append)}

    def _file_delete(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"does not exist: {path}"}
        try:
            if target.is_dir():
                for child in sorted(target.rglob("*"), reverse=True):
                    child.unlink() if child.is_file() else child.rmdir()
                target.rmdir()
            else:
                target.unlink()
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "deleted": str(target.relative_to(self.root))}

    def _dir_list(self, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        entries = []
        pattern = "**/*" if recursive else "*"
        for item in sorted(target.glob(pattern)):
            try:
                entries.append({
                    "path": str(item.relative_to(self.root)),
                    "dir": item.is_dir(),
                    "bytes": item.stat().st_size if item.is_file() else 0,
                })
            except OSError:
                continue
            if len(entries) >= 500:
                break
        return {"ok": True, "path": str(target.relative_to(self.root)), "entries": entries}

    def _shell_exec(self, command: str, timeout: float = 60.0) -> dict[str, Any]:
        if self.shell is None:
            return {"ok": False, "error": "shell sandbox not initialised"}
        result = self.shell.run(command, timeout=timeout, require_permission=False)
        return result.to_dict()

    def _web_fetch(self, url: str, timeout: float = 20.0,
                   max_bytes: int = 500_000) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {"ok": False, "error": "only http/https URLs are supported"}
        host = (parsed.hostname or "").lower()
        if self.config.offline_only() and self.config.get("llm.deny_non_local_hosts", True):
            ai_hosts = {"api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
                        "api.cohere.ai", "api.mistral.ai", "api.groq.com", "api.together.xyz"}
            if host in ai_hosts or host.endswith(".openai.com"):
                return {"ok": False,
                        "error": "refused: that host is an external AI API. Jarvis OS requires "
                                 "explicit manual approval for external AI connections."}
        req = urllib.request.Request(url, headers={"User-Agent": "JarvisOS/1.0 (+local)"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(max_bytes)
                ctype = resp.headers.get("Content-Type", "")
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": f"HTTP {exc.code}", "status": exc.code}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": f"fetch failed: {exc}"}
        if ctype.startswith("text/") or "json" in ctype or "xml" in ctype or not ctype:
            try:
                body = raw.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                return {"ok": False, "error": "binary payload", "content_type": ctype}
        else:
            body = f"<{len(raw)} bytes of {ctype or 'binary'}>"
        return {"ok": True, "url": url, "status": status,
                "content_type": ctype, "body": body[:200_000]}

    def _code_runner(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        if self.sandbox is None:
            return {"ok": False, "error": "sandbox not initialised"}
        result = self.sandbox.run(code, timeout=timeout, require_permission=False)
        return result.to_dict()

    def _ollama_models(self) -> dict[str, Any]:
        client = getattr(self, "_llm", None)
        if client is None:
            return {"ok": True, "models": [], "note": "LLM client not attached to registry"}
        try:
            return {"ok": True, "models": client.local_models()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _memory_recall(self, query: str = "", limit: int = 5) -> dict[str, Any]:
        if self.memory is None:
            return {"ok": True, "results": [], "note": "memory not attached"}
        hits = self.memory.search(query, limit=limit)
        return {"ok": True, "results": [
            {"content": m.content, "kind": m.kind, "score": round(s, 4)} for m, s in hits
        ]}

    def _memory_remember(self, content: str, kind: str = "fact",
                         importance: float = 0.5) -> dict[str, Any]:
        if self.memory is None:
            return {"ok": False, "error": "memory not attached"}
        rec = self.memory.remember(content, kind=kind, importance=importance)
        return {"ok": True, "id": rec.id}

    # --------------------------------------------------------- attach deps
    def attach_llm(self, llm) -> "ToolRegistry":
        self._llm = llm
        return self