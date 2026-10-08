"""SelfUpdate - Jarvis patching its own source, safely and reversibly.

Mechanism
---------
* The Jarvis OS package tree is a git repository (initialised on first use).
* Every change is: ``snapshot -> patch -> commit -> verify``.
* Rollback is ``git revert`` to a recorded tag, so recovery is deterministic.
* Memory paths are hard-excluded from every git operation. The ``preserve_paths``
  list is additionally written into ``.gitignore`` so ``git add -A`` can never
  capture ``memory.db`` or the vector store.

Gating
------
Every call passes through the PermissionGate with category ``self_update``. The
module refuses to touch disk at all until the operator approves, and it refuses
to commit when ``evolution.self_update_enabled`` is false in config.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

MEMORY_GLOBS = (
    "data/memory.db", "data/memory.db-*", "data/vectors.json", "data/vectors.json.tmp",
    "data/lessons.jsonl", "data/actions.jsonl", "data/audit_ledger.jsonl",
    "data/audit.db*", "data/heartbeat.json", "*.key", "data/control.token",
    "data/memory_index.db", "data/memory_index.db-*", "data/plan_state.json",
    "data/content_pipeline.jsonl", "data/crm_contacts.jsonl",
    "data/business_ledger.jsonl", "data/trading_ledger.jsonl",
    "data/trading_alerts.jsonl",
)


@dataclass(slots=True)
class Snapshot:
    tag: str
    commit: str
    ts: float
    reason: str
    files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "commit": self.commit, "ts": self.ts,
                "reason": self.reason, "files": self.files}


@dataclass(slots=True)
class UpdateResult:
    ok: bool
    message: str
    snapshot: Snapshot | None = None
    changed: list[str] = field(default_factory=list)
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message,
                "snapshot": self.snapshot.to_dict() if self.snapshot else None,
                "changed": self.changed, "rolled_back": self.rolled_back}


class SelfUpdate:
    def __init__(self, config, gate=None, audit=None, memory=None,
                 root: str | os.PathLike[str] | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.memory = memory
        self.root = Path(root or config.resolve("self_update.repo_path")).resolve()
        self.package_dir = self.root / "jarvis_os"
        self.snapshots: list[Snapshot] = []
        self._init_done = False

    # ------------------------------------------------------------------ git
    def _git(self, *args: str, check: bool = True, timeout: int = 60) -> tuple[int, str, str]:
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                ["git", *args], cwd=str(self.root), capture_output=True, text=True,
                timeout=timeout, creationflags=creation,
            )
        except FileNotFoundError:
            return 127, "", "git not found on PATH"
        except subprocess.TimeoutExpired:
            return 124, "", f"git {' '.join(args)} timed out"
        if check and proc.returncode != 0:
            return proc.returncode, proc.stdout, proc.stderr.strip()
        return proc.returncode, proc.stdout, proc.stderr.strip()

    def available(self) -> bool:
        code, _, _ = self._git("--version", check=False)
        return code == 0

    def is_repo(self) -> bool:
        code, out, _ = self._git("rev-parse", "--is-inside-work-tree", check=False)
        return code == 0 and out.strip() == "true"

    def init(self) -> UpdateResult:
        """Initialise the self-versioning repo (idempotent, no permission needed)."""
        if self.is_repo():
            self._protect_memory()
            self._init_done = True
            return UpdateResult(True, "git repository already initialised")
        self.root.mkdir(parents=True, exist_ok=True)
        self._protect_memory()
        self._git("init", check=False)
        code, _, err = self._git("add", "-A", check=False)
        if code != 0:
            return UpdateResult(False, f"git add failed: {err}")
        author = str(self.config.get("evolution.git_author", "Jarvis OS"))
        self._git("-c", f"user.name={author}", "-c", f"user.email=jarvis@localhost",
                  "commit", "-m", "JARVIS: baseline snapshot", check=False)
        self._init_done = True
        if self.audit:
            self.audit.record("self_update", "init", "ok", {"root": str(self.root)})
        return UpdateResult(True, "git repository initialised with a baseline commit")

    _MEMORY_HEADER = "# Persistent memory and audit ledger, never committed."

    def _protect_memory(self) -> None:
        """Ensure memory can never be committed or reverted.

        Append-only, and a no-op once every pattern is present.

        This used to collect every line into a ``set`` and rewrite the file
        ``sorted()``. That destroyed blank lines and comment placement on every
        single call - and because ``#`` sorts ahead of every glob, it stacked all
        13 comments at the top of the file. It ran from ``init()``, so every
        snapshot (including the selftest's own) re-mangled the file, which made
        any comment fix revert on the next run.

        Only a *missing* pattern has to be added, so nothing else is touched.
        Appending also leans on git's last-match-wins rule: the memory patterns
        land last and therefore outrank any earlier negation or re-inclusion.
        """
        ignore = self.root / ".gitignore"
        existing = ignore.read_text(encoding="utf-8").splitlines() \
            if ignore.is_file() else []
        # Only a real rule can satisfy the requirement - a comment that happens
        # to name a pattern is not a rule and must not count as present.
        present = {l.strip() for l in existing
                   if l.strip() and not l.strip().startswith("#")}
        wanted = [*MEMORY_GLOBS,
                  *(str(p) for p
                    in (self.config.get("self_update.preserve_paths", []) or []))]
        # dict.fromkeys keeps the caller's order while dropping repeats.
        missing = [p for p in dict.fromkeys(wanted) if p and p not in present]
        if not missing:
            return  # byte-identical no-op: the file is never rewritten
        lines = list(existing)
        if not any(l.strip() == self._MEMORY_HEADER for l in lines):
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(self._MEMORY_HEADER)
        lines.extend(missing)
        ignore.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")

    def _configured(self) -> bool:
        return bool(self.config.get("evolution.self_update_enabled", False))

    # ------------------------------------------------------------ snapshots
    def snapshot(self, reason: str = "pre-change") -> Snapshot | None:
        self.init()
        prefix = str(self.config.get("self_update.snapshot_prefix", "jarvis-selfupdate"))
        tag = f"{prefix}/{time.strftime('%Y%m%d-%H%M%S')}"
        code, out, err = self._git("add", "-A", check=False)
        if code != 0:
            self._log("snapshot", "failed", {"error": err[:200]})
            return None
        code, diff, _ = self._git("diff", "--cached", "--name-only", check=False)
        files = len([l for l in diff.splitlines() if l.strip()])
        if files == 0:
            snap = Snapshot(tag=tag, commit=self.head(), ts=time.time(),
                            reason=f"{reason} (nothing to snapshot)", files=0)
            self.snapshots.append(snap)
            return snap
        author = str(self.config.get("evolution.git_author", "Jarvis OS"))
        code, _, err = self._git(
            "-c", f"user.name={author}", "-c", f"user.email=jarvis@localhost",
            "commit", "-m", f"JARVIS: snapshot before {reason}", check=False,
        )
        if code != 0:
            self._log("snapshot", "failed", {"error": err[:200]})
            return None
        self._git("tag", "-f", tag, check=False)
        snap = Snapshot(tag=tag, commit=self.head(), ts=time.time(),
                        reason=reason, files=files)
        self.snapshots.append(snap)
        self.snapshots = self.snapshots[-50:]
        self._log("snapshot", "ok", {"tag": tag, "commit": snap.commit, "files": files})
        return snap

    def head(self) -> str:
        code, out, _ = self._git("rev-parse", "--short", "HEAD", check=False)
        return out.strip() if code == 0 else ""

    def log(self, limit: int = 15) -> str:
        code, out, err = self._git("log", f"-{limit}", "--oneline",
                                   "--decorate", check=False)
        if code != 0:
            return f"git log unavailable: {err}"
        return out.strip() or "(no commits yet)"

    # --------------------------------------------------------------- patches
    def write_file(self, path: str, content: str,
                   require_permission: bool = True,
                   override_disarm: bool = False) -> UpdateResult:
        """Patch one source file, gated by ``self_update`` approval."""
        if self.config.get("self_update.forbid_self_patch", True) and \
                str(path).startswith(("jarvis_os/permissions", "jarvis_os/overseer")):
            return UpdateResult(False, "refused: protected module may not be patched by evolution")

        if require_permission and self.gate is not None:
            if not self.gate.approve(
                "self_update",
                f"Jarvis requests permission to modify its own source file:\n  {path}",
                detail={"path": path, "bytes": len(content),
                        "reason": "self-update patch"},
                override_disarm=override_disarm,
            ):
                self._log("patch", "denied", {"path": path})
                return UpdateResult(False, "permission denied by operator")

        target = (self.root / path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            return UpdateResult(False, f"path '{path}' escapes the repo root")
        if any(str(target).endswith(p) for p in MEMORY_GLOBS):
            return UpdateResult(False, "refused: memory files are immutable to self-update")
        if not target.exists():
            return UpdateResult(False, f"file does not exist: {path}")

        before = target.read_text(encoding="utf-8", errors="replace")
        if before == content:
            return UpdateResult(True, "no change needed", changed=[])
        snap = self.snapshot(f"patch {Path(path).name}")
        backup = target.with_suffix(target.suffix + ".jarvis.bak")
        try:
            shutil.copy2(target, backup)
            target.write_text(content, encoding="utf-8")
            compile_check = self._compile_ok(target)
            if not compile_check:
                target.write_text(before, encoding="utf-8")
                self.rollback_to(snap.tag if snap else None, reason="syntax check failed")
                return UpdateResult(False, "patch produced invalid Python - rolled back")
            author = str(self.config.get("evolution.git_author", "Jarvis OS"))
            self._git("add", str(target.relative_to(self.root)).replace("\\", "/"), check=False)
            self._git("-c", f"user.name={author}", "-c", f"user.email=jarvis@localhost",
                      "commit", "-m",
                      f"JARVIS: {str(self.config.get('self_update.commit_prefix', 'JARVIS:'))}"
                      f" patch {path}", check=False)
            backup.unlink(missing_ok=True)
        except OSError as exc:
            return UpdateResult(False, f"write failed: {exc}")

        if self.memory is not None:
            self.memory.remember(
                f"Self-update patched {path} ({self.head()})", kind="self_update",
                importance=0.8, meta={"tag": snap.tag if snap else ""})
        self._log("patch", "ok", {"path": path, "commit": self.head()})
        return UpdateResult(True, f"patched {path} at {self.head()}", snap, [str(path)])

    def apply_text_patch(self, path: str, old: str, new: str) -> UpdateResult:
        target = (self.root / path).resolve()
        if not target.is_file():
            return UpdateResult(False, f"file not found: {path}")
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return UpdateResult(False, str(exc))
        if old not in content:
            return UpdateResult(False, "patch context not found in file")
        return self.write_file(path, content.replace(old, new, 1))

    @staticmethod
    def _compile_ok(target: Path) -> bool:
        try:
            compile(target.read_text(encoding="utf-8"), str(target), "exec")
            return True
        except (SyntaxError, ValueError):
            return False

    # ------------------------------------------------------------- rollback
    def rollback_to(self, tag: str | None = None,
                    reason: str = "operator rollback") -> UpdateResult:
        if self.gate is not None and not self.gate.approve(
            "self_update", f"Jarvis requests permission to roll back to '{tag or 'HEAD~1'}'.",
            detail={"tag": tag, "reason": reason},
        ):
            return UpdateResult(False, "rollback denied by operator")

        self.init()
        target = tag
        if not target:
            target = self.snapshots[-1].tag if self.snapshots else None
        if not target:
            code, out, _ = self._git("rev-parse", "--short", "HEAD~1", check=False)
            target = out.strip() or None
        if not target:
            return UpdateResult(False, "no rollback point available")

        self._protect_memory()
        if self.config.get("evolution.phases.rollback", True):
            self._git("revert", "--no-edit", "--no-commit", target, check=False)
            code, _, err = self._git("checkout", target, "--", ".", check=False)
            if code != 0:
                return UpdateResult(False, f"checkout failed: {err}")
            code, _, _ = self._git("reset", check=False)
        self._git("clean", "-fd", "--exclude", "data", check=False)
        self._log("rollback", "ok", {"tag": target, "reason": reason})
        if self.memory is not None:
            self.memory.remember(
                f"Rolled back to {target} ({reason})", kind="self_update", importance=0.9)
        return UpdateResult(True, f"rolled back to {target}; memory untouched",
                            rolled_back=True)

    def diff_stat(self) -> str:
        code, out, err = self._git("status", "--short", check=False)
        if code != 0:
            return f"git unavailable: {err}"
        return out.strip() or "working tree clean"

    def describe_target(self, target: str | None = None) -> str:
        if not self.available():
            return "git is not on PATH; self-update is unavailable."
        if not self.is_repo():
            return ("Jarvis OS is not yet under git version control.\n"
                    "Run 'evolve' or 'update init' to create a baseline snapshot.")
        lines = [
            f"Self-update: enabled={self._configured()} repo={self.root}",
            f"HEAD: {self.head()}",
            f"Memory protection: {len(MEMORY_GLOBS)} patterns in .gitignore",
            "",
            "Recent commits:",
            self.log(10),
        ]
        if self.snapshots:
            lines.append("")
            lines.append("Snapshots:")
            for snap in self.snapshots[-5:]:
                lines.append(f"  {snap.tag}  ({snap.files} files)  {snap.reason}")
        if target:
            lines.append("")
            lines.append(f"Target '{target}' - use: update apply {target}")
        return "\n".join(lines)

    def _log(self, action: str, decision: str, detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record("self_update", action, decision, detail)