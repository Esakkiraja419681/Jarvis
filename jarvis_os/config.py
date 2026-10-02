"""Configuration loader for Jarvis OS.

Every knob the operator can flip lives in ``config.json``. Evolution
phases and features are individually toggleable so nothing changes
itself without a manual switch being thrown.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG_NAME = "config.json"

_DEFAULTS: dict[str, Any] = {
    "honorific": "Sir",
    "preferred_language": "ta",
    "llm": {
        "provider": "ollama",
        "locked_to_local": True,
        "base_url": "http://127.0.0.1:11434",
        "model": "llama3.1:8b",
        "timeout_seconds": 120,
        "temperature": 0.2,
        "allowed_local_hosts": ["127.0.0.1", "localhost", "::1"],
        "deny_non_local_hosts": True,
        "allow_external_providers": False,
    },
    "permissions": {
        "categories_requiring_manual_approval": [
            "self_improve",
            "self_update",
            "self_evolution",
            "self_learn",
            "skill_add",
            "external_ai",
            "cross_device",
            "shell_exec",
            "web_fetch",
            "file_write",
            "file_delete",
            "code_exec",
            "software_install",
            "software_launch",
            "software_control",
            "voice_capture",
            "background_daemon",
        ],
        "auto_allowed": ["file_read", "dir_list", "memory_read", "status"],
        "approval_words": ["yes", "allow", "approve", "approved", "proceed", "ok", "haan"],
        "deny_words": ["no", "deny", "denied", "nope", "stop", "reject", "never"],
        "remember_grants": False,
        "grant_ttl_minutes": 0,
        "timeout_seconds": 300,
    },
    "evolution": {
        "phases": {
            "propose": True,
            "request_permission": True,
            "sandbox_test": True,
            "apply": True,
            "rollback": True,
        },
        "sandbox_test_required": True,
        "sandbox_timeout_seconds": 30,
        "auto_rollback_on_failure": True,
        "max_changes_per_cycle": 1,
        "self_update_enabled": False,
    },
    "features": {
        "voice": False,
        "text_interface": True,
        "language_layer": True,
        "overseer": True,
        "background_daemon": False,
        "device_manager": True,
        "software_factory": True,
        "control_bridge": True,
        "sandbox": True,
        "audit_log": True,
        "infinite_loop": True,
        "rest_api": True,
        "ipc": True,
        "show_reasoning": False,
        "offline_only": True,
    },
    "loop": {
        "interval_seconds": 5,
        "max_cycles": 0,
        "backoff_on_error_seconds": 15,
        "stop_file": "data/STOP",
        "pause_file": "data/PAUSE",
    },
    "overseer": {
        "heartbeat_path": "data/heartbeat.json",
        "interval_seconds": 3,
        "stall_timeout_seconds": 45,
        "halt_on_fault": True,
        "auto_repair": True,
        "auto_restart": True,
        "max_repair_attempts": 3,
    },
    "sandbox": {
        "python_executable": "python",
        "timeout_seconds": 30,
        "memory_mb": 512,
        "network": False,
        "workdir": "data/sandbox",
    },
    "language": {
        "backend": "auto",
        "target_language": "en",
        "offline_only": True,
        "always_prefix_honorific": True,
    },
    "voice": {
        "wake_words": ["jarvis", "sir"],
        "tts_rate": 175,
        "listen_timeout_seconds": 8,
    },
    "software_factory": {
        "build_root": "data/built",
        "registry_path": "data/software_registry.json",
        "default_language": "python",
        "default_entry": "main.py",
        "runtime": "python",
    },
    "control_bridge": {
        "ipc_path": "data/control.sock",
        "rest_host": "127.0.0.1",
        "rest_port": 8770,
    },
    "devices": {
        "registry_path": "data/devices.json",
        "current_device_id": "local",
        "allow_remote_config": False,
    },
    "self_update": {
        "repo_path": ".",
        "branch": "main",
        "auto_snapshot": True,
        "preserve_paths": [
            "data/memory.db",
            "data/vectors.json",
            "data/lessons.jsonl",
            "data/actions.jsonl",
            "data/audit_ledger.jsonl",
            "config.json",
        ],
    },
    "skills": {"path": "data/skills", "auto_load": True},
    "memory": {
        "sqlite_path": "data/memory.db",
        "vector_path": "data/vectors.json",
        "lesson_log": "data/lessons.jsonl",
        "action_log": "data/actions.jsonl",
        "retrieval_top_k": 5,
        "immutable": True,
    },
    "audit": {
        "ledger_path": "data/audit_ledger.jsonl",
        "sqlite_path": "data/audit.db",
        "shared_with_overseer": True,
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Thread-safe, path-resolving configuration store."""

    def __init__(self, path: str | os.PathLike[str] | None = None,
                 root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.path = Path(path) if path else self.root / DEFAULT_CONFIG_NAME
        self._lock = threading.RLock()
        self._data: dict[str, Any] = copy.deepcopy(_DEFAULTS)
        self.reload()

    # ------------------------------------------------------------------ load
    def reload(self) -> "Config":
        with self._lock:
            if self.path.is_file():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        self._data = _deep_merge(_DEFAULTS, raw)
                except (json.JSONDecodeError, OSError):
                    self._data = copy.deepcopy(_DEFAULTS)
            else:
                self._data = copy.deepcopy(_DEFAULTS)
            return self

    def save(self) -> "Config":
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(self.path)
        return self

    # ------------------------------------------------------------------ read
    def get(self, dotted: str, default: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def get_dict(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    def set(self, dotted: str, value: Any, persist: bool = True) -> "Config":
        with self._lock:
            parts = dotted.split(".")
            node = self._data
            for part in parts[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[parts[-1]] = value
            if persist:
                self.save()
        return self

    def unset(self, dotted: str, persist: bool = True) -> "Config":
        with self._lock:
            parts = dotted.split(".")
            node: Any = self._data
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    return self
                node = node[part]
            if isinstance(node, dict):
                node.pop(parts[-1], None)
            if persist:
                self.save()
        return self

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    # --------------------------------------------------------------- helpers
    def resolve(self, dotted_or_path: str) -> Path:
        candidate = Path(dotted_or_path)
        if candidate.is_absolute():
            return candidate
        p = Path(dotted_or_path)
        if p.parts and p.parts[0] in self._data:
            value = self.get(dotted_or_path)
            if isinstance(value, str):
                return (self.root / value).resolve()
        return (self.root / dotted_or_path).resolve()

    def feature(self, name: str, default: bool = False) -> bool:
        value = self.get(f"features.{name}", default)
        return bool(value) if isinstance(value, bool) else default

    def set_feature(self, name: str, enabled: bool) -> "Config":
        return self.set(f"features.{name}", bool(enabled))

    def feature_names(self) -> Iterable[str]:
        return sorted(self.get_dict("features").keys())

    def phase(self, name: str, default: bool = True) -> bool:
        value = self.get(f"evolution.phases.{name}", default)
        return bool(value) if isinstance(value, bool) else default

    def set_phase(self, name: str, enabled: bool) -> "Config":
        return self.set(f"evolution.phases.{name}", bool(enabled))

    def phase_names(self) -> Iterable[str]:
        return sorted(self.get_dict("evolution.phases").keys())

    def honorific(self) -> str:
        return str(self.get("honorific", "Sir"))

    def offline_only(self) -> bool:
        return self.feature("offline_only", True)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Config path={self.path.name} root={self.root}>"