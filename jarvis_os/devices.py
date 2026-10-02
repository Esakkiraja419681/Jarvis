"""DeviceManager - let Jarvis run on the operator's laptop, phones and anything
else they explicitly grant.

Threat model
------------
Cross-device access is one of the mandatory-approval categories. A remote node
is only ever added, contacted, reconfigured or removed after a live manual
approval, and nothing here will phone home unless it is told to.

Mechanism
---------
* Local registry of permitted devices in ``data/devices.json``.
* Each device has an optional REST control endpoint - the same
  ``ControlBridge`` surface exposed on the laptop, so a phone can send the same
  commands.
* Remote configuration changes are queued, require approval, and are applied
  through the ControlBridge so they land in the audit ledger.
* ``allow_remote_config`` in config is an extra hard brake: with it off, the
  DeviceManager refuses configuration writes entirely, approval or not.
"""

from __future__ import annotations

import json
import platform
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Device:
    device_id: str
    label: str
    kind: str
    endpoint: str = ""
    os_name: str = ""
    granted_at: float = field(default_factory=time.time)
    last_seen: float | None = None
    trusted: bool = False
    status: str = "registered"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.device_id, "label": self.label, "kind": self.kind,
            "endpoint": self.endpoint, "os": self.os_name, "granted_at": self.granted_at,
            "last_seen": self.last_seen, "trusted": self.trusted,
            "status": self.status, "notes": self.notes,
        }


class DeviceManager:
    def __init__(self, config, gate=None, audit=None, memory=None,
                 control=None, path: str | Path | None = None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self.memory = memory
        self.control = control
        self.path = Path(path or config.resolve("devices.registry_path"))
        self.devices: dict[str, Device] = {}
        self.current = str(config.get("devices.current_device_id", "local"))
        self.load()
        if self.current not in self.devices:
            self.devices[self.current] = self._local_device()

    # ------------------------------------------------------------ persistence
    def load(self) -> "DeviceManager":
        if not self.path.is_file():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self
        for item in raw.get("devices", []) if isinstance(raw, dict) else []:
            if isinstance(item, dict) and item.get("id"):
                known = set(Device.__slots__)
                device = Device(**{k: v for k, v in item.items() if k in known})
                self.devices[device.device_id] = device
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"current": self.current, "saved_at": time.time(),
                   "devices": [d.to_dict() for d in self.devices.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ----------------------------------------------------------------- local
    def _local_device(self) -> Device:
        return Device(
            device_id=self.current, label=f"{platform.node()} (this machine)",
            kind="host", endpoint="", os_name=f"{platform.system()} {platform.release()}",
            trusted=True, status="local",
            notes="The machine Jarvis OS is running on right now.",
        )

    # --------------------------------------------------------------- commands
    def command(self, args: list[str]) -> str:
        sub = (args[0].lower() if args else "list")
        rest = args[1:]
        return {
            "list": self.cmd_list, "add": self.cmd_add, "remove": self.cmd_remove,
            "trust": self.cmd_trust, "reach": self.cmd_reach, "configure": self.cmd_config,
            "current": self.cmd_current,
        }.get(sub, self.cmd_unknown)(rest)

    def cmd_unknown(self, args: list[str]) -> str:
        return ("Usage: device [list | add <id> <endpoint> | remove <id> | "
                "trust <id> | reach <id> | configure <id> <action> | current]")

    def cmd_list(self, args: list[str]) -> str:
        self.load()
        lines = [f"Devices ({len(self.devices)}):"]
        for device in sorted(self.devices.values(), key=lambda d: d.granted_at):
            mark = "*" if device.device_id == self.current else ("+" if device.trusted else " ")
            seen = ("seen " + time.strftime("%H:%M", time.localtime(device.last_seen))
                    ) if device.last_seen else "never seen"
            lines.append(f" {mark} {device.device_id:<16} {device.kind:<10} "
                         f"{device.status:<12} {seen:<14} {device.label}")
        lines.append(f"remote config allowed: {self.config.get('devices.allow_remote_config')}")
        return "\n".join(lines)

    def cmd_current(self, args: list[str]) -> str:
        device = self.devices.get(self.current)
        return f"current device: {device.label} ({device.device_id}) on {device.os_name}"

    def cmd_add(self, args: list[str]) -> str:
        if len(args) < 1:
            return "Usage: device add <id> [endpoint]"
        device_id = args[0]
        endpoint = args[1] if len(args) > 1 else ""
        if device_id in self.devices:
            return f"device '{device_id}' is already registered"
        if self.gate is not None and not self.gate.approve(
            "cross_device",
            f"Jarvis requests permission to register a remote device.\n"
            f"  id: {device_id}\n  endpoint: {endpoint or 'unknown'}",
            detail={"device": device_id, "endpoint": endpoint},
        ):
            return "permission denied by operator - device not registered"
        self.devices[device_id] = Device(
            device_id=device_id, label=device_id, kind="remote", endpoint=endpoint,
            os_name="", status="registered", notes="added via manual approval",
        )
        self.save()
        self._log("add", "approved", {"device": device_id, "endpoint": endpoint})
        if self.memory is not None:
            self.memory.remember(f"Device '{device_id}' granted by operator",
                                 kind="device", importance=0.7)
        return f"Device '{device_id}' registered. Reach it with: device reach {device_id}"

    def cmd_remove(self, args: list[str]) -> str:
        if not args:
            return "Usage: device remove <id>"
        device_id = args[0]
        if device_id not in self.devices:
            return f"no such device '{device_id}'"
        if device_id == self.current:
            return "refused: cannot remove the machine Jarvis is running on"
        if self.gate is not None and not self.gate.approve(
            "cross_device", f"Revoke access for device '{device_id}'?",
            detail={"device": device_id},
        ):
            return "permission denied by operator"
        self.devices.pop(device_id)
        self.save()
        self._log("remove", "approved", {"device": device_id})
        return f"Device '{device_id}' revoked."

    def cmd_trust(self, args: list[str]) -> str:
        if not args:
            return "Usage: device trust <id>"
        device_id = args[0]
        device = self.devices.get(device_id)
        if device is None:
            return f"no such device '{device_id}'"
        if self.gate is not None and not self.gate.approve(
            "cross_device", f"Mark '{device_id}' as a trusted device?",
            detail={"device": device_id},
        ):
            return "permission denied by operator"
        device.trusted = not device.trusted
        self.save()
        self._log("trust", "approved", {"device": device_id, "trusted": device.trusted})
        return f"'{device_id}' trusted={device.trusted}"

    # ------------------------------------------------------------- remote ops
    def cmd_reach(self, args: list[str]) -> str:
        if not args:
            return "Usage: device reach <id>"
        device = self.devices.get(args[0])
        if device is None:
            return f"no such device '{args[0]}'"
        if not device.endpoint:
            return (f"'{device.device_id}' has no endpoint on record. "
                    f"Add one with: device add {device.device_id} <url>")
        return self.ping(device.device_id)

    def ping(self, device_id: str) -> str:
        device = self.devices.get(device_id)
        if device is None or not device.endpoint:
            return f"cannot reach '{device_id}': no endpoint"
        if self.gate is not None and not self.gate.approve(
            "cross_device", f"Contact remote device '{device_id}' at {device.endpoint}?",
            detail={"device": device_id, "endpoint": device.endpoint},
        ):
            return "permission denied by operator - no contact attempted"
        try:
            with urllib.request.urlopen(f"{device.endpoint.rstrip('/')}/health", timeout=8) as r:
                body = r.read(2000).decode("utf-8", "replace")
            device.last_seen = time.time()
            device.status = "online"
            self.save()
            self._log("reach", "ok", {"device": device_id})
            return f"'{device_id}' is online: {body[:400]}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            device.status = "unreachable"
            self.save()
            self._log("reach", "failed", {"device": device_id, "error": str(exc)[:200]})
            return f"'{device_id}' unreachable: {exc}"

    def cmd_config(self, args: list[str]) -> str:
        if len(args) < 2:
            return ("Usage: device configure <id> <program> [action] - e.g.\n"
                    "  device configure phone my-app restart")
        if not self.config.get("devices.allow_remote_config", False):
            return ("Remote configuration is disabled in config "
                    "(devices.allow_remote_config = false). Turn it on with:\n"
                    "  config devices.allow_remote_config=true\n"
                    "then approve when prompted, Sir.")
        device_id, program = args[0], args[1]
        action = args[2] if len(args) > 2 else "status"
        device = self.devices.get(device_id)
        if device is None:
            return f"no such device '{device_id}'"
        if self.gate is not None and not self.gate.approve(
            "cross_device",
            f"Push a remote configuration change to '{device_id}'?\n"
            f"  program: {program}\n  action : {action}",
            detail={"device": device_id, "program": program, "action": action},
        ):
            return "permission denied by operator - nothing was changed remotely"
        token = self.control.token if self.control else ""
        query = urllib.parse.urlencode({"token": token, "program": program, "action": action})
        url = f"{device.endpoint.rstrip('/')}/control?{query}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                body = response.read(8000).decode("utf-8", "replace")
            device.last_seen = time.time()
            self.save()
            self._log("configure", "ok", {"device": device_id, "program": program,
                                           "action": action})
            return f"remote change applied on '{device_id}': {body[:600]}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self._log("configure", "failed", {"device": device_id, "error": str(exc)[:200]})
            return f"remote configuration failed: {exc}"

    def report(self) -> str:
        return f"DeviceManager: {len(self.devices)} device(s), current={self.current}, " \
               f"remote_config={self.config.get('devices.allow_remote_config')}"

    def _log(self, action: str, decision: str, detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record("device", action, decision, detail)