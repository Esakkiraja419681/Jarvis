"""tool_registry.py - the validation gate every dynamic tool must clear.

Tools that did not ship with Jarvis - built by the evolution engine, written
by a delegated agent, or handed over by the operator - pass through
:func:`validate_and_register` before joining the live :class:`ToolRegistry`:

1. **static scan** - AST denylist (network, process, filesystem and
   desktop-escape modules, ``eval``/``exec``/``open``-class calls, dunder
   access) plus ``compile()``. Nothing executes at this stage.
2. **PermissionGate** - category ``skill_add`` is a MANDATORY manual approval,
   requested *before* the candidate ever runs.
3. **sandbox probe** - the source executes inside :class:`PythonSandbox` with
   benign arguments; a tool that crashes, hangs or returns garbage never
   reaches the registry.
4. **registration** - only then is the handler loaded in-process (its module
   body already ran once, cleanly, under the probe) and registered as a
   ``dangerous`` ``code_exec`` runtime tool. Every stage is audited.

Fail-closed by design: no sandbox means no dynamic loading.
"""

from __future__ import annotations

import ast
import json
from typing import Any

MAX_SOURCE_BYTES = 64_000
PROBE_TIMEOUT_SECONDS = 20.0

# Top-level import roots a dynamic tool may never pull in. Network and
# process escapes mirror the sandbox prelude; filesystem/serialization
# modules block silent out-of-project reads and writes; UI modules block
# calls that must instead go through the gated ``ui_control`` tool.
FORBIDDEN_MODULES = frozenset({
    # network
    "socket", "socketserver", "ssl", "ftplib", "smtplib", "poplib", "imaplib",
    "telnetlib", "urllib", "urllib3", "http", "httpx", "requests", "aiohttp",
    "websocket", "paramiko", "asyncio", "dns", "xmlrpc",
    # process / OS escape
    "os", "sys", "subprocess", "multiprocessing", "ctypes", "signal",
    "shutil", "webbrowser", "importlib", "runpy", "pkgutil", "resource",
    "_thread", "thread", "threading",
    # filesystem / serialization
    "pathlib", "tempfile", "pickle", "marshal", "shelve", "dbm", "sqlite3",
    "zipfile", "tarfile", "gzip", "io",
    # desktop capture/injection - must use the gated ui_control tool
    "pyautogui", "PIL", "mss", "imagegrab", "keyboard", "mouse",
    # Windows internals
    "winreg", "win32api", "win32gui", "win32com", "pythoncom", "wmi",
    # introspection escapes
    "gc", "inspect",
})

# Calls that execute code, touch the filesystem or reach across objects.
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help",
})

ALLOWED_DUNDERS = frozenset({"__name__", "__main__"})

# The probe harness. It runs as the sandbox payload: exec the candidate,
# call its entry point with benign arguments, print a JSON verdict. Tool
# stdout printed before the verdict is tolerated - the parser reads the
# last JSON line that carries an "ok" key.
_HARNESS = '''\
import json
SOURCE = {source!r}
ATTR = {attr!r}
ARGS = {args!r}
ns = {{"__name__": "candidate"}}
try:
    exec(compile(SOURCE, "candidate.py", "exec"), ns)
except BaseException as exc:
    print(json.dumps({{"ok": False, "stage": "import",
                      "error": f"{{type(exc).__name__}}: {{exc}}"}}))
    raise SystemExit(0)
fn = ns.get(ATTR)
if not callable(fn):
    print(json.dumps({{"ok": False, "stage": "call",
                      "error": "missing callable entry point " + repr(ATTR)}}))
    raise SystemExit(0)
try:
    out = fn(**ARGS)
except BaseException as exc:
    print(json.dumps({{"ok": False, "stage": "call",
                      "error": f"{{type(exc).__name__}}: {{exc}}"}}))
    raise SystemExit(0)
if not isinstance(out, dict):
    print(json.dumps({{"ok": False, "stage": "call",
                      "error": f"returned {{type(out).__name__}}, expected dict"}}))
    raise SystemExit(0)
print(json.dumps({{"ok": True, "stage": "call", "returned": out}}))
'''


def _audit(audit, name: str, stage: str, decision: str,
           detail: dict[str, Any] | None = None) -> None:
    if audit:
        audit.record("tool", f"validate:{name}:{stage}", decision, detail or {})


def _fail(stage: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "stage": stage, "error": error, **extra}


def validate_name(name: str) -> str:
    """Return the cleaned name or raise ValueError (letters/digits/underscore)."""
    cleaned = str(name).strip()
    if not cleaned or not all(c.isalnum() or c == "_" for c in cleaned):
        raise ValueError(
            f"invalid tool name '{name}' - use letters, digits and underscore only")
    return cleaned


def static_validate(name: str, source: str) -> dict[str, Any]:
    """Stage 1: name, size, syntax and AST denylist. Nothing executes."""
    try:
        cleaned = validate_name(name)
    except ValueError as exc:
        return _fail("static", str(exc))
    if not str(source).strip():
        return _fail("static", "empty source")
    if len(source.encode("utf-8", "replace")) > MAX_SOURCE_BYTES:
        return _fail("static", f"source exceeds {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source, filename=f"<tool:{cleaned}>")
        compile(source, f"<tool:{cleaned}>", "exec")
    except (SyntaxError, ValueError) as exc:
        return _fail("static", f"does not compile: {type(exc).__name__}: {exc}")
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return _fail("static", "relative imports are not allowed")
            modules = [node.module or ""]
        for mod in modules:
            root = mod.split(".")[0]
            if root in FORBIDDEN_MODULES or mod in FORBIDDEN_MODULES:
                return _fail("static",
                             f"import of '{mod}' is not permitted in dynamic tools")
        if isinstance(node, ast.Call):
            func = node.func
            callee = func.id if isinstance(func, ast.Name) else \
                (func.attr if isinstance(func, ast.Attribute) else "")
            if callee in FORBIDDEN_CALLS:
                return _fail("static", f"call to '{callee}' is not permitted")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_DUNDERS \
                and "__" in node.id:
            return _fail("static", f"dunder access '{node.id}' is not permitted")
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_DUNDERS \
                and "__" in node.attr:
            return _fail("static", f"dunder access '{node.attr}' is not permitted")
    return {"ok": True, "stage": "static", "name": cleaned}


def _parse_probe(stdout: str, stderr: str, timed_out: bool,
                 returncode: int) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and "ok" in payload:
                return payload
    if timed_out:
        return _fail("probe", f"probe timed out after {PROBE_TIMEOUT_SECONDS}s")
    return _fail("probe", f"sandbox exited rc={returncode}: "
                          f"{stderr.strip()[:300] or 'no verdict'}")


def probe_source(sandbox, name: str, source: str, *, attr: str = "run",
                 probe_args: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
    """Stage 3: execute the candidate inside the sandbox with benign arguments.

    The gate has already approved ``skill_add`` by the time this runs, so the
    probe itself does not ask again (``require_permission=False``).
    """
    if sandbox is None:
        return _fail("probe", "sandbox unavailable - refusing to load "
                              "an unvalidated tool")
    harness = _HARNESS.format(source=source, attr=attr,
                              args=probe_args or {})
    try:
        result = sandbox.run(harness, timeout=timeout or PROBE_TIMEOUT_SECONDS,
                             require_permission=False)
    except Exception as exc:  # noqa: BLE001 - sandbox errors are verdicts here
        return _fail("probe", f"sandbox run failed: {type(exc).__name__}: {exc}")
    if result.truncated:
        return _fail("probe", "probe output exceeded the sandbox limit")
    return _parse_probe(result.stdout, result.stderr, result.timed_out,
                        result.returncode)


def validate_and_register(registry, name: str, source: str, description: str,
                          *, parameters: dict[str, Any] | None = None,
                          attr: str = "run",
                          probe_args: dict[str, Any] | None = None,
                          category: str = "code_exec",
                          replace: bool = False) -> dict[str, Any]:
    """Run the full chain (static -> gate -> sandbox probe -> register).

    Returns a verdict dict: ``{"ok": True, "stage": "registered", ...}`` or
    ``{"ok": False, "stage": <failed stage>, "error": ...}``.
    """
    gate = getattr(registry, "gate", None)
    audit = getattr(registry, "audit", None)
    sandbox = getattr(registry, "sandbox", None)

    static = static_validate(name, source)
    if not static["ok"]:
        _audit(audit, str(name), "static", "denied", {"error": static["error"]})
        return static
    cleaned = static["name"]

    if gate is not None and not gate.approve(
        "skill_add",
        "Jarvis requests permission to register a new dynamic tool.\n"
        f"  tool : {cleaned}\n"
        f"  entry: {attr}\n"
        f"  bytes: {len(source.encode('utf-8', 'replace'))}",
        detail={"tool": cleaned, "entry": attr, "category": category},
    ):
        _audit(audit, cleaned, "permission", "denied")
        return _fail("permission", "operator denied the tool")
    _audit(audit, cleaned, "permission", "ok", {"category": category})

    probe = probe_source(sandbox, cleaned, source, attr=attr,
                         probe_args=probe_args)
    if not probe["ok"]:
        _audit(audit, cleaned, "probe", "failed", {"error": probe.get("error", "")})
        return probe
    _audit(audit, cleaned, "probe", "ok", {"returned": probe.get("returned")})

    # The module body already executed once, cleanly, under the probe; load
    # the same source in-process so the registry holds a real callable.
    namespace: dict[str, Any] = {"__name__": f"jarvis_tool_{cleaned}"}
    try:
        exec(compile(source, f"<tool:{cleaned}>", "exec"), namespace)
        handler = namespace.get(attr)
        if not callable(handler):
            return _fail("load", f"missing callable entry point {attr!r}")
    except Exception as exc:  # noqa: BLE001
        return _fail("load", f"{type(exc).__name__}: {exc}")

    try:
        registry.register(
            cleaned, handler,
            description=str(description or f"dynamic tool {cleaned}").strip()[:300],
            parameters=parameters,
            permission_category=category, dangerous=True, builtin=False,
            replace=replace,
        )
    except Exception as exc:  # noqa: BLE001 - e.g. duplicate without replace
        _audit(audit, cleaned, "register", "failed", {"error": str(exc)})
        return _fail("register", str(exc))
    _audit(audit, cleaned, "register", "ok", {"category": category})
    return {"ok": True, "stage": "registered", "name": cleaned,
            "category": category, "returned": probe.get("returned")}
