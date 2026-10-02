"""SoftwareFactory - generate, install, launch, monitor and control any program.

Flow: **spec -> generate -> sandbox-verify -> register -> install -> launch ->
monitor -> control**. :class:`SoftwareRegistry` is the durable ledger of every
program Jarvis has ever built; :class:`~jarvis_os.control.ControlBridge` owns the
process lifecycle.

The factory works for any kind of runnable it can host locally (Python, Node,
shell-free static bundles). Types it cannot host are reported honestly rather
than faked.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILD_PROMPT = """You are Jarvis's software factory. Build a small, complete, runnable
program from the operator's specification.

SPECIFICATION:
{spec}

RULES:
- Output ONE python source file named exactly `{entry}`.
- No network calls, no cloud APIs, no credentials.
- Include a `main()` and `if __name__ == "__main__": main()`.
- Add a short module docstring describing what it does.
- Keep it under {max_lines} lines. No external dependencies unless absolutely required.
- If the spec cannot be satisfied locally, write a program that explains why
  and exits non-zero - never fake success.

Return raw JSON only: {{"name":"short-slug","summary":"one line",
"code":"full source","entry":"{entry}","notes":"how to run it"}}"""


@dataclass(slots=True)
class Program:
    program_id: str
    name: str
    summary: str
    spec: str
    language: str
    entry: str
    path: str
    status: str = "generated"
    built_at: float = field(default_factory=time.time)
    installed: bool = False
    launched_at: float | None = None
    pid: int | None = None
    notes: str = ""
    test: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.program_id, "name": self.name, "summary": self.summary,
            "spec": self.spec, "language": self.language, "entry": self.entry,
            "path": self.path, "status": self.status, "built_at": self.built_at,
            "installed": self.installed, "launched_at": self.launched_at,
            "pid": self.pid, "notes": self.notes, "test": self.test,
            "history": self.history[-10:],
        }


class SoftwareRegistry:
    """Durable, JSON-backed registry of everything Jarvis has built."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._programs: dict[str, Program] = {}
        self.load()

    def load(self) -> "SoftwareRegistry":
        if not self.path.is_file():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self
        for item in raw.get("programs", []) if isinstance(raw, dict) else []:
            if isinstance(item, dict) and item.get("id"):
                known = {f for f in Program.__slots__}
                # to_dict() serialises program_id as "id"; map it back on load
                # so the registry survives a restart.
                fields = dict(item)
                if "id" in fields:
                    fields["program_id"] = fields.pop("id")
                program = Program(**{k: v for k, v in fields.items() if k in known})
                self._programs[program.program_id] = program
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "saved_at": time.time(),
                   "programs": [p.to_dict() for p in self._programs.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, program: Program) -> Program:
        self._programs[program.program_id] = program
        self.save()
        return program

    def get(self, program_id: str) -> Program | None:
        if program_id in self._programs:
            return self._programs[program_id]
        for program in self._programs.values():
            if program.name == program_id:
                return program
        matches = [p for p in self._programs.values() if program_id.lower() in p.name.lower()]
        return matches[-1] if len(matches) == 1 else None

    def all(self) -> list[Program]:
        return sorted(self._programs.values(), key=lambda p: p.built_at, reverse=True)

    def remove(self, program_id: str) -> bool:
        program = self._programs.pop(program_id, None)
        self.save()
        return bool(program)

    def __len__(self) -> int:
        return len(self._programs)


class SoftwareFactory:
    def __init__(self, config, llm, memory, gate, control, sandbox,
                 audit=None, root: str | Path | None = None) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.gate = gate
        self.bridge = control
        self.sandbox = sandbox
        self.audit = audit
        self.root = Path(root or config.resolve("software_factory.build_root"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = SoftwareRegistry(
            config.resolve("software_factory.registry_path"))
        self.runtime = str(config.get("software_factory.runtime", "python"))

    # ----------------------------------------------------------------- build
    def build(self, spec: str, name: str = "", language: str = "") -> dict[str, Any]:
        if not spec.strip():
            return {"ok": False, "error": "empty specification"}
        language = language or str(self.config.get("software_factory.default_language", "python"))
        entry = str(self.config.get("software_factory.default_entry", "main.py"))
        if language != "python":
            entry = {"node": "main.js", "shell": "main.sh"}.get(language, entry)

        if self.gate is not None and not self.gate.approve(
            "software_install",
            f"Jarvis requests permission to generate and install new software.\n"
            f"  spec: {spec[:160]}",
            detail={"spec": spec[:300], "language": language},
        ):
            self._log("build", "denied", {"spec": spec[:160]})
            return {"ok": False, "denied": True, "error": "operator denied the build"}

        generated = self._generate(spec, entry)
        if not generated.get("ok"):
            return generated
        code = str(generated["code"])
        program_id = f"sw_{uuid.uuid4().hex[:8]}"
        program_name = name or str(generated.get("name") or f"program_{program_id[-4:]}")
        program_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", program_name).strip("_").lower() or program_id
        folder = self.root / program_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / entry
        target.write_text(code, encoding="utf-8")
        (folder / "SPEC.md").write_text(
            f"# {program_name}\n\n{spec}\n", encoding="utf-8")
        (folder / "BUILD.json").write_text(
            json.dumps(generated, indent=2, ensure_ascii=False)[:8000], encoding="utf-8")

        program = Program(
            program_id=program_id, name=program_name,
            summary=str(generated.get("summary", ""))[:300], spec=spec,
            language=language, entry=entry, path=str(folder),
            notes=str(generated.get("notes", ""))[:500],
        )
        program.test = self._verify(program, code)
        if not program.test.get("ok"):
            program.status = "failed_verification"
            self.registry.add(program)
            self._log("build", "verify_failed", {"program": program_id,
                                                  "error": program.test.get("stderr", "")[:200]})
            return {"ok": False, "error": "generated code failed sandbox verification",
                    "program": program.to_dict(), "test": program.test}

        program.status = "generated"
        self.registry.add(program)
        if self.memory is not None:
            self.memory.record_action(
                kind="software_build", summary=spec[:200], result=program.summary,
                success=True, meta={"program": program_id, "name": program_name},
            )
        self._log("build", "ok", {"program": program_id, "name": program_name})
        return {"ok": True, "program": program.to_dict(),
                "message": (f"Built '{program_name}' ({program_id}) at {folder}. "
                            f"Verified in sandbox. Say 'install {program_id}' next, Sir.")}

    def _generate(self, spec: str, entry: str) -> dict[str, Any]:
        prompt = BUILD_PROMPT.format(spec=spec, entry=entry, max_lines=250)
        parsed = self.llm.ask_json(prompt, persona="builder")
        if not parsed:
            fallback = self._fallback_program(spec, entry)
            if fallback:
                return fallback
            return {"ok": False, "error": "the local model produced no usable program"}
        code = str(parsed.get("code", ""))
        if "```" in code:
            blocks = re.findall(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
            code = max(blocks, key=len) if blocks else code
        if not code.strip():
            return {"ok": False, "error": "generated program was empty"}
        return {
            "ok": True, "name": parsed.get("name", ""),
            "summary": parsed.get("summary", ""), "code": code,
            "entry": parsed.get("entry", entry), "notes": parsed.get("notes", ""),
        }

    def _fallback_program(self, spec: str, entry: str) -> dict[str, Any] | None:
        """Deterministic template so 'build' always yields a runnable artifact."""
        if "hello" in spec.lower() or not spec.strip():
            return {
                "ok": True, "name": "hello_sir",
                "summary": f"Hello program for: {spec[:60]}",
                "notes": "Generated locally as a template; ask for a richer build.",
                "code": (
                    '"""Hello, Sir."""\n\n\n'
                    f'SPEC = {spec!r}\n\n\n'
                    'def main() -> int:\n'
                    '    print(f"Hello, Sir. Working on: {SPEC}")\n'
                    '    return 0\n\n\n'
                    'if __name__ == "__main__":\n'
                    '    raise SystemExit(main())\n'
                ),
            }
        return None

    def _verify(self, program: Program, code: str) -> dict[str, Any]:
        """Compile + execute the program briefly inside the sandbox."""
        try:
            compile(code, program.entry, "exec")
        except SyntaxError as exc:
            return {"ok": False, "stage": "compile", "error": str(exc)}
        if program.language != "python":
            return {"ok": True, "stage": "compile", "note": "non-python target, syntax-checked only"}
        result = self.sandbox.run(code, timeout=15, require_permission=False)
        return {
            "ok": result.ok, "stage": "execute",
            "stdout": result.stdout[-800:], "stderr": result.stderr[-400:],
            "returncode": result.returncode, "timed_out": result.timed_out,
        }

    # -------------------------------------------------------------- lifecycle
    def install(self, program_id: str) -> dict[str, Any]:
        program = self.registry.get(program_id)
        if program is None:
            return {"ok": False, "error": f"no program matching '{program_id}'"}
        if self.gate is not None and not self.gate.approve(
            "software_install", f"Install and register '{program.name}' ({program_id})?",
            detail={"program": program_id, "path": program.path},
        ):
            return {"ok": False, "denied": True, "error": "operator denied the install"}

        entry_path = Path(program.path) / program.entry
        if not entry_path.is_file():
            return {"ok": False, "error": f"entry file missing: {entry_path}"}

        argv = self._argv(program)
        self.bridge.register(program.program_id, argv, program.path,
                              env={"JARVIS_PROGRAM": program.program_id,
                                   "JARVIS_SIR": self.config.honorific()})
        program.installed = True
        program.status = "installed"
        program.history.append(f"installed at {time.time():.0f}")
        self.registry.save()
        self._log("install", "ok", {"program": program.program_id})
        return {"ok": True, "program": program.to_dict(),
                "message": f"Installed '{program.name}'. Command: launch {program.program_id}"}

    def _argv(self, program: Program) -> list[str]:
        if program.language == "python":
            return [sys.executable or "python", "-u", program.entry]
        if program.language == "node":
            return ["node", program.entry]
        return [str(Path(program.path) / program.entry)]

    def launch(self, program_id: str, args: list[str] | None = None) -> dict[str, Any]:
        program = self.registry.get(program_id)
        if program is None:
            return {"ok": False, "error": f"no program matching '{program_id}'"}
        if not program.installed:
            return {"ok": False,
                    "error": f"not installed yet - run 'install {program.program_id}' first"}
        result = self.bridge.dispatch(program.program_id, "start", args or [])
        if result.get("ok"):
            program.status = "running"
            program.launched_at = time.time()
            program.pid = result.get("pid")
            program.history.append(f"launched pid={result.get('pid')}")
            self.registry.save()
            self._log("launch", "ok", {"program": program.program_id,
                                       "pid": result.get("pid")})
        return result

    def control(self, program_id: str, action: str, args: list[str] | None = None
                ) -> dict[str, Any]:
        program = self.registry.get(program_id)
        if program is None:
            return {"ok": False, "error": f"no program matching '{program_id}'"}
        result = self.bridge.dispatch(program.program_id, action, args or [])
        if result.get("ok"):
            if action in ("stop", "restart"):
                program.status = "stopped" if action == "stop" else "running"
                program.history.append(f"{action} at {time.time():.0f}")
                self.registry.save()
        return result

    def monitor(self, program_id: str | None = None) -> dict[str, Any]:
        programs = [program_id] if program_id else [p.program_id for p in self.registry.all()]
        out = []
        for pid in programs:
            program = self.registry.get(pid)
            if program is None:
                continue
            state = self.bridge.dispatch(pid, "status", [])
            out.append({"id": program.program_id, "name": program.name,
                        "status": program.status, "installed": program.installed,
                        "pid": program.pid, "detail": state})
        alive = sum(1 for r in out if r["detail"].get("alive"))
        return {"ok": True, "programs": out, "running": alive, "total": len(out)}

    def remove(self, program_id: str) -> dict[str, Any]:
        program = self.registry.get(program_id)
        if program is None:
            return {"ok": False, "error": f"no program matching '{program_id}'"}
        self.bridge.dispatch(program.program_id, "stop", [])
        if self.gate is not None and not self.gate.approve(
            "file_delete", f"Delete '{program.name}' and everything in {program.path}?",
            detail={"program": program.program_id},
        ):
            return {"ok": False, "denied": True, "error": "operator denied the deletion"}
        try:
            shutil.rmtree(program.path, ignore_errors=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        self.registry.remove(program.program_id)
        self._log("remove", "ok", {"program": program.program_id})
        return {"ok": True, "removed": program.program_id}

    # ---------------------------------------------------------------- report
    def report(self) -> str:
        if not len(self.registry):
            return "SoftwareFactory: no programs built yet. Say: build <spec>"
        lines = [f"SoftwareRegistry: {len(self.registry)} program(s)"]
        for program in self.registry.all():
            mark = {"running": "*", "installed": "+", "generated": " "}.get(program.status, "?")
            lines.append(f" {mark} {program.program_id:<14} {program.name:<20} "
                         f"{program.status:<12} {program.summary[:44]}")
        return "\n".join(lines)

    def _log(self, action: str, decision: str, detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record("software", action, decision, detail)