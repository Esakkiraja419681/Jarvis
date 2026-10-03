"""OverseerAgenticOS - a second, independent Agentic OS that watches the first.

Runs as its own process, in parallel, with its own planner and its own view of
the world. It never trusts Jarvis OS's self-assessment; it reads the shared
audit ledger, the heartbeat file and process state, and it reasons about every
layer independently.

Layer map (used for fault isolation)
------------------------------------
``interface`` - the text/voice front doors, loop guards, runaway detection
``planner``   - decomposition quality, dependency cycles, stalled plans
``memory``    - integrity, attempted wipes, vector-store corruption
``tools``     - error rate, repeated identical failures
``llm``       - availability, hallucination markers, provider escapes
``execution`` - sandbox escapes, unmanaged children, illegal paths
``permission``- violations: any action taken without an approved gate entry

Fault cycle
-----------
detect -> **halt Jarvis OS** -> isolate layer -> diagnose (LLM + heuristics)
-> rewrite -> sandbox-test -> restart with memory intact -> log to shared ledger.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LAYERS = ("interface", "planner", "memory", "tools", "llm", "execution", "permission")

HALLUCINATION_MARKERS = (
    "i have completed", "successfully ran", "i already did that", "as i mentioned",
    "i created the file", "i installed", "the command returned", "task complete",
)

RUNAWAY_MARKERS = ("max_cycles", "runaway", "infinite loop detected", "cycle_budget_exceeded")

VIOLATION_MARKERS = (
    "permission_bypass", "gate_skipped", "unauthorized", "without approval",
    "external_ai_blocked", "auto_granted",
)

DIAGNOSE_PROMPT = """You are the Overseer, an independent watchdog for Jarvis OS.

A fault was detected in layer `{layer}`.
Signal: {signal}
Observed evidence:
{evidence}

Recent audit trail:
{audit}

Return raw JSON only:
{{"fault":"...", "layer":"{layer}", "root_cause":"one sentence",
  "fix_plan":"concrete minimal code change", "target_file":"jarvis_os/....py",
  "confidence":0.0-1.0, "needs_manual_approval":true}}"""


@dataclass(slots=True)
class Incident:
    incident_id: str
    fault: str
    layer: str
    ts: float = field(default_factory=time.time)
    evidence: str = ""
    diagnosis: dict[str, Any] = field(default_factory=dict)
    action_taken: str = ""
    halted: bool = False
    repaired: bool = False
    restarted: bool = False
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.incident_id, "fault": self.fault, "layer": self.layer,
            "ts": self.ts, "evidence": self.evidence[:2000], "diagnosis": self.diagnosis,
            "action_taken": self.action_taken, "halted": self.halted,
            "repaired": self.repaired, "restarted": self.restarted,
            "resolved": self.resolved,
        }


class OverseerAgenticOS:
    def __init__(self, config, audit=None, memory=None, llm=None, gate=None,
                 root: str | os.PathLike[str] | None = None) -> None:
        self.config = config
        self.root = Path(root or config.root).resolve()
        self.audit = audit
        self.memory = memory
        self.llm = llm
        self.gate = gate
        self.heartbeat_path = self.config.resolve(config.get(
            "overseer.heartbeat_path", "data/heartbeat.json"))
        self.stall_timeout = float(config.get("overseer.stall_timeout_seconds", 45))
        self.interval = float(config.get("overseer.interval_seconds", 3))
        self.halt_on_fault = bool(config.get("overseer.halt_on_fault", True))
        self.auto_repair = bool(config.get("overseer.auto_repair", True))
        self.auto_restart = bool(config.get("overseer.auto_restart", True))
        self.max_repairs = int(config.get("overseer.max_repair_attempts", 3))
        self.incidents: list[Incident] = []
        self.enabled = bool(config.feature("overseer", True))
        self.running = False
        self.repair_attempts = 0
        self._counter = 0
        self._jarvis_pid: int | None = None
        self._restart_cb: Callable[[], Any] | None = None
        self._lock = threading.RLock()
        self.last_scan: dict[str, Any] = {}

    # ------------------------------------------------------------- lifecycle
    def bind_restart(self, callback: Callable[[], Any]) -> None:
        self._restart_cb = callback

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        if self.audit:
            self.audit.record("overseer", "start", "ok",
                              {"heartbeat": str(self.heartbeat_path), "pid": os.getpid()},
                              actor="overseer")
        threading.Thread(target=self._loop, daemon=True, name="overseer").start()

    def stop(self) -> None:
        self.running = False
        if self.audit:
            self.audit.record("overseer", "stop", "ok", {}, actor="overseer")

    def _loop(self) -> None:
        while self.running:
            try:
                self.scan()
            except Exception as exc:  # noqa: BLE001 - the overseer must never die
                if self.audit:
                    self.audit.record("overseer", "scan_error", "failed",
                                      {"error": str(exc)[:300]}, actor="overseer")
            time.sleep(self.interval)

    # ------------------------------------------------------------- heartbeat
    def heartbeat(self) -> dict[str, Any]:
        if not self.heartbeat_path.is_file():
            return {}
        try:
            return json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def write_heartbeat(self, payload: dict[str, Any]) -> None:
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.heartbeat_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.heartbeat_path)

    @staticmethod
    def age(payload: dict[str, Any]) -> float:
        try:
            return time.time() - float(payload.get("ts", 0))
        except (TypeError, ValueError):
            return 1e9

    def jarvis_alive(self) -> bool:
        beat = self.heartbeat()
        pid = beat.get("pid") or self._jarvis_pid
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
                return code.value == 259
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError, AttributeError):
            return False

    # ----------------------------------------------------------- diagnostics
    def check_permission_integrity(self) -> list[str]:
        """Any privileged action with no matching approval entry is a violation."""
        if not self.audit:
            return []
        approvals = {
            (e["action"], round(float(e["ts"]), 2))
            for e in self.audit.tail(limit=400, category="permission")
            if e["decision"] == "approved"
        }
        approvals_actions = {a for a, _ in approvals}
        violations: list[str] = []
        for entry in self.audit.tail(limit=400):
            action, decision = entry["action"], entry["decision"]
            category = entry["category"]
            if category in ("permission", "agent", "tool", "control", "shell",
                            "code", "software", "device", "evolution", "llm"):
                if any(marker in str(entry["detail"]).lower() for marker in VIOLATION_MARKERS):
                    violations.append(f"permission:{action}:{decision}")
            if decision == "approved" and not str(action).startswith("gate:"):
                violations.append(f"unclassified approval: {action}")
        return violations[:20]

    def check_hallucination(self, beat: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        last = beat.get("last_output", "")
        if last:
            lowered = str(last).lower()
            hits = [m for m in HALLUCINATION_MARKERS if m in lowered]
            if hits:
                findings.append(f"hallucination markers in last output: {hits}")
        if beat.get("claimed_success") and not beat.get("verified_success"):
            findings.append("unverified success claim in heartbeat")
        return findings

    def check_runaway(self, beat: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        cycle_count = int(beat.get("cycles", 0) or 0)
        uptime = max(1.0, time.time() - float(beat.get("started_at", time.time()) or time.time()))
        rate = cycle_count / uptime
        if rate > 20.0:
            findings.append(f"runaway loop: {rate:.1f} cycles/sec")
        if beat.get("consecutive_errors", 0) >= 8:
            findings.append(f"error storm: {beat['consecutive_errors']} consecutive errors")
        for marker in RUNAWAY_MARKERS:
            if marker in str(beat.get("notes", "")):
                findings.append(f"kernel flagged itself: {marker}")
        return findings

    def check_memory_integrity(self) -> list[str]:
        if self.memory is None:
            return []
        findings: list[str] = []
        report = self.memory.integrity_report()
        if not report.get("ok"):
            findings.extend(report.get("issues", [])[:10])
        findings.extend(f"memory: {issue}" for issue in report.get("issues", [])[:5])
        stats = report.get("stats", {})
        if stats.get("memories", 0) == 0 and stats.get("boots", 1) > 1:
            findings.append("memory_wipe_attempt: zero memories after a restart")
        return findings[:10]

    def check_tool_health(self) -> list[str]:
        if self.audit is None:
            return []
        findings: list[str] = []
        recent = self.audit.tail(limit=300, category="tool")
        counts: dict[str, int] = {}
        for entry in recent:
            if entry["decision"] in ("failed", "denied"):
                counts[entry["action"]] = counts.get(entry["action"], 0) + 1
        for action, count in counts.items():
            if count >= 5:
                findings.append(f"tool layer: '{action}' failing {count}x recently")
        return findings[:10]

    def check_llm_health(self) -> list[str]:
        findings: list[str] = []
        beat = self.heartbeat()
        if beat.get("llm_errors", 0) >= 10:
            findings.append(f"llm layer: {beat['llm_errors']} recent LLM failures")
        provider = str(self.config.get("llm.provider", "ollama"))
        base = str(self.config.get("llm.base_url", ""))
        if provider != "ollama":
            findings.append(f"llm layer: provider drift to '{provider}'")
        if base and not any(h in base for h in ("127.0.0.1", "localhost", "::1")):
            findings.append(f"llm layer: base_url is not local ({base})")
        return findings[:10]

    # ------------------------------------------------------------------ scan
    def scan(self) -> dict[str, Any]:
        beat = self.heartbeat()
        faults: list[dict[str, Any]] = []
        if not beat:
            faults.append({"fault": "no_heartbeat", "layer": "interface",
                           "evidence": "heartbeat file missing or unreadable"})
        else:
            age = self.age(beat)
            if age > self.stall_timeout:
                faults.append({
                    "fault": "stall", "layer": "planner",
                    "evidence": f"heartbeat is {age:.1f}s old "
                                f"(timeout {self.stall_timeout}s); cycles={beat.get('cycles')}",
                })
            for finding in self.check_runaway(beat):
                faults.append({"fault": "runaway_loop", "layer": "interface",
                               "evidence": finding})
            for finding in self.check_hallucination(beat):
                faults.append({"fault": "hallucination", "layer": "llm",
                               "evidence": finding})
            for finding in self.check_llm_health():
                faults.append({"fault": "layer_failure", "layer": "llm",
                               "evidence": finding})

        for finding in self.check_memory_integrity():
            layer = "memory"
            fault = ("memory_wipe_attempt" if "wipe" in finding.lower()
                     else "layer_failure")
            faults.append({"fault": fault, "layer": layer, "evidence": finding})

        for finding in self.check_tool_health():
            faults.append({"fault": "layer_failure", "layer": "tools",
                           "evidence": finding})

        for finding in self.check_permission_integrity():
            faults.append({"fault": "permission_violation", "layer": "permission",
                           "evidence": finding})

        self.last_scan = {"ts": time.time(), "faults": faults,
                          "jarvis_alive": self.jarvis_alive()}
        for fault in faults:
            self.handle_fault(fault["fault"], fault["layer"], fault["evidence"])
        return self.last_scan

    # ----------------------------------------------------------- fault cycle
    def handle_fault(self, fault: str, layer: str, evidence: str) -> Incident:
        self._counter += 1
        incident = Incident(
            incident_id=f"inc_{time.strftime('%Y%m%d-%H%M%S')}_{self._counter}",
            fault=fault, layer=layer, evidence=evidence,
        )
        self.incidents.append(incident)
        self.incidents = self.incidents[-50:]
        if self.audit:
            self.audit.record("overseer", f"fault:{fault}", "detected",
                              {"layer": layer, "evidence": evidence[:400],
                               "incident": incident.incident_id}, actor="overseer")

        # 1. HALT - immediately, before any diagnosis.
        if self.halt_on_fault and fault in ("permission_violation", "runaway_loop",
                                             "memory_wipe_attempt", "unsafe_action"):
            incident.halted = self.halt_jarvis(reason=f"{fault}: {evidence[:120]}")
            incident.action_taken = "halted Jarvis OS" if incident.halted else \
                "halt requested (Jarvis was not running)"

        # 2. ISOLATE the faulty layer.
        self.isolate_layer(layer, incident)

        # 3. DIAGNOSE across layers.
        incident.diagnosis = self.diagnose(fault, layer, evidence)

        # 4. REWRITE + verify, then restart.
        if self.auto_repair and incident.halted:
            incident.repaired = self.repair(incident)
        if self.auto_restart:
            incident.restarted = self.restart_jarvis(reason=f"overseer repair {incident.incident_id}")
        incident.resolved = True
        if self.audit:
            self.audit.record(
                "overseer", f"resolved:{fault}", "ok",
                {"layer": layer, "incident": incident.incident_id,
                 "halted": incident.halted, "repaired": incident.repaired,
                 "restarted": incident.restarted,
                 "root_cause": incident.diagnosis.get("root_cause", "")},
                actor="overseer",
            )
        if self.memory is not None:
            self.memory.learn_lesson(
                topic="overseer",
                lesson=(f"Incident {incident.incident_id}: {fault} in layer {layer}. "
                        f"{incident.diagnosis.get('root_cause', 'undetermined')}"),
                outcome="failure" if not incident.repaired else "success", weight=0.85,
            )
        return incident

    def halt_jarvis(self, reason: str = "overseer fault") -> bool:
        """Terminate the Jarvis process group immediately."""
        beat = self.heartbeat()
        pid = beat.get("pid") or self._jarvis_pid
        if not pid:
            return False
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=20, check=False)
            else:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        except Exception as exc:  # noqa: BLE001
            if self.audit:
                self.audit.record("overseer", "halt", "failed",
                                  {"error": str(exc)[:200]}, actor="overseer")
            return False
        if self.audit:
            self.audit.record("overseer", "halt", "ok",
                              {"pid": pid, "reason": reason}, actor="overseer")
        return True

    def isolate_layer(self, layer: str, incident: Incident) -> dict[str, Any]:
        """Shut down just the suspect layer and report what was frozen."""
        if layer not in LAYERS:
            return {"layer": layer, "isolated": False, "reason": "unknown layer"}
        controls = {
            "permission": lambda: self._write_control({"gate_disarm": True,
                                                       "incident": incident.incident_id}),
            "tools": lambda: self._write_control({"tools_disabled": True}),
            "llm": lambda: self._write_control({"llm_disabled": True}),
            "memory": lambda: self._write_control({"memory_readonly": True}),
            "execution": lambda: self._write_control({"sandbox_enabled": False}),
            "planner": lambda: self._write_control({"planner_disabled": True}),
            "interface": lambda: self._write_control({"loop_interval_max": 60}),
        }
        detail = controls[layer]()
        incident.action_taken += f"; isolated layer '{layer}'"
        if self.audit:
            self.audit.record("overseer", f"isolate:{layer}", "ok",
                              {"incident": incident.incident_id, **detail}, actor="overseer")
        return {"layer": layer, "isolated": True, **detail}

    def _write_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write an isolation instruction the kernel applies on restart."""
        path = self.config.resolve(
            self.config.get("overseer.control_file", "data/overseer_control.json"))
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update({k: v for k, v in payload.items() if k != "incident"})
        existing["last_incident"] = payload.get("incident", "")
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            return {"written": False, "error": str(exc)}
        return {"written": True, "control": existing}

    def diagnose(self, fault: str, layer: str, evidence: str) -> dict[str, Any]:
        audit_tail = ""
        if self.audit:
            lines = [
                f"{e['category']}/{e['action']} -> {e['decision']}"
                for e in self.audit.tail(limit=25)
            ]
            audit_tail = "\n".join(lines)
        prompt = DIAGNOSE_PROMPT.format(layer=layer, signal=fault,
                                        evidence=evidence[:1500], audit=audit_tail or "(empty)")
        parsed = self.llm.ask_json(prompt, persona="overseer") if self.llm else None
        if not parsed or "root_cause" not in parsed:
            parsed = self._heuristic_diagnose(fault, layer, evidence)
        parsed.setdefault("layer", layer)
        parsed.setdefault("fault", fault)
        parsed.setdefault("confidence", 0.3)
        parsed.setdefault("needs_manual_approval", True)
        parsed["cross_layer_trace"] = self._cross_layer_trace(fault, layer)
        if self.audit:
            self.audit.record("overseer", f"diagnose:{layer}", "ok",
                              {"fault": fault, "root_cause": parsed.get("root_cause", "")[:300],
                               "confidence": parsed.get("confidence")}, actor="overseer")
        return parsed

    def _cross_layer_trace(self, fault: str, layer: str) -> dict[str, Any]:
        """Walk every layer and record its health, so the fault is not misfiled."""
        trace: dict[str, Any] = {}
        beat = self.heartbeat()
        for name in LAYERS:
            if name == layer:
                trace[name] = "FAULT"
            elif name == "permission":
                trace[name] = "VIOLATION" if self.check_permission_integrity() else "ok"
            elif name == "memory":
                trace[name] = "ok" if not self.check_memory_integrity() else "degraded"
            elif name == "tools":
                trace[name] = "degraded" if self.check_tool_health() else "ok"
            elif name == "llm":
                trace[name] = "degraded" if self.check_llm_health() else "ok"
            elif name == "interface":
                trace[name] = "ok" if not self.check_runaway(beat) else "FAULT"
            elif name == "planner":
                trace[name] = "FAULT" if self.age(beat) > self.stall_timeout else "ok"
            else:
                trace[name] = "ok"
        return trace

    def _heuristic_diagnose(self, fault: str, layer: str, evidence: str) -> dict[str, Any]:
        catalogue = {
            "permission_violation": ("jarvis_os/permissions.py",
                                     "PermissionGate.approve returned a decision without a "
                                     "live operator answer; the gate is being bypassed."),
            "runaway_loop": ("jarvis_os/kernel.py",
                             "The cycle loop has no backoff or budget guard."),
            "hallucination": ("jarvis_os/kernel.py",
                              "Output is being reported as success without tool verification."),
            "stall": ("jarvis_os/taskmanager.py",
                      "A delegated subtask never resolved, so the planner waited forever."),
            "memory_wipe_attempt": ("jarvis_os/memory.py",
                                    "Something attempted a destructive memory operation."),
            "layer_failure": (f"jarvis_os/{layer}.py", f"Layer '{layer}' reported failure."),
        }
        target, cause = catalogue.get(
            fault, ("jarvis_os/kernel.py", f"Unclassified fault '{fault}' in layer '{layer}'."))
        return {
            "fault": fault, "layer": layer, "root_cause": cause,
            "fix_plan": f"Review {target} and add a guard against: {evidence[:120]}",
            "target_file": target, "confidence": 0.35,
            "needs_manual_approval": True, "source": "heuristic",
        }

    def repair(self, incident: Incident) -> bool:
        """Rewrite the faulty code path and prove the fix in the sandbox."""
        from .sandbox import PythonSandbox
        from .selfupdate import SelfUpdate

        if self.repair_attempts >= self.max_repairs:
            if self.audit:
                self.audit.record("overseer", "repair", "skipped",
                                  {"reason": "repair attempt budget exhausted",
                                   "incident": incident.incident_id}, actor="overseer")
            return False
        # The Overseer is never allowed to rewrite source without a live
        # approval channel. In standalone mode there is no gate, so repairing
        # would silently bypass the operator - refuse instead of escalating.
        if self.gate is None:
            if self.audit:
                self.audit.record("overseer", "repair", "skipped",
                                  {"reason": "no permission gate available - "
                                             "refusing to rewrite source without "
                                             "a live approval channel",
                                   "incident": incident.incident_id}, actor="overseer")
            return False
        self.repair_attempts += 1
        diagnosis = incident.diagnosis or {}
        target = str(diagnosis.get("target_file", ""))
        if not target.startswith("jarvis_os/") or not (self.root / target).is_file():
            if self.audit:
                self.audit.record("overseer", "repair", "skipped",
                                  {"reason": f"no safe target file ('{target}')",
                                   "incident": incident.incident_id}, actor="overseer")
            return False

        patch = self._synthesise_patch(incident)
        if not patch:
            return False
        sandbox = PythonSandbox(self.config, gate=self.gate, audit=self.audit)
        test_code = patch.get("test_code") or "compile(open(r'%s', encoding='utf-8').read(), '%s', 'exec'); print('syntax ok')" % (
            str(self.root / target), target)
        result = sandbox.run(test_code, timeout=20, require_permission=False)
        if not result.ok:
            if self.audit:
                self.audit.record("overseer", "repair", "test_failed",
                                  {"incident": incident.incident_id,
                                   "stderr": result.stderr[:300]}, actor="overseer")
            return False

        updater = SelfUpdate(self.config, gate=self.gate, audit=self.audit,
                             memory=self.memory, root=self.root)
        # The Overseer's own repair is gated too: it asks the operator before it
        # rewrites anything, and refuses to touch protected modules.
        write_result = updater.write_file(target, patch["content"],
                                          require_permission=diagnosis.get(
                                              "needs_manual_approval", True),
                                          # The master switch may be down because
                                          # of THIS incident; the operator must
                                          # still be able to approve the repair.
                                          override_disarm=True)
        if write_result.ok:
            self.repair_attempts = 0
            self._clear_control()
            if self.audit:
                self.audit.record("overseer", "repair", "ok",
                                  {"incident": incident.incident_id, "file": target,
                                   "commit": updater.head()}, actor="overseer")
            return True
        return False

    def _synthesise_patch(self, incident: Incident) -> dict[str, Any] | None:
        diagnosis = incident.diagnosis or {}
        target = str(diagnosis.get("target_file", ""))
        path = self.root / target
        try:
            original = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if target in ("jarvis_os/permissions.py", "jarvis_os/memory.py",
                      "jarvis_os/audit.py", "jarvis_os/overseer.py"):
            # Never rewrite the guard rails themselves.
            return None
        prompt = (
            f"File: {target}\n"
            f"Fault: {incident.fault} in layer {incident.layer}\n"
            f"Root cause: {diagnosis.get('root_cause', 'unknown')}\n"
            f"Required fix: {diagnosis.get('fix_plan', 'harden against the fault')}\n\n"
            f"Current source:\n```python\n{original}\n```\n\n"
            "Return raw JSON only: {\"content\":\"the FULL corrected file source\","
            " \"test_code\":\"a standalone python snippet that imports nothing from the "
            "repo and asserts the fix holds\"}. Preserve all existing behaviour. "
            "Never add network calls to external services."
        )
        parsed = self.llm.ask_json(prompt, persona="overseer") if self.llm else None
        if not parsed or not str(parsed.get("content", "")).strip():
            return None
        content = str(parsed["content"])
        try:
            compile(content, target, "exec")
        except SyntaxError as exc:
            if self.audit:
                self.audit.record("overseer", "repair", "syntax_rejected",
                                  {"error": str(exc)[:200]}, actor="overseer")
            return None
        return {"content": content, "test_code": str(parsed.get("test_code", ""))}

    def _clear_control(self) -> None:
        path = self.config.resolve(
            self.config.get("overseer.control_file", "data/overseer_control.json"))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def clear_control(self) -> bool:
        """Public, operator-driven release of any active layer isolation."""
        path = self.config.resolve(
            self.config.get("overseer.control_file", "data/overseer_control.json"))
        try:
            existed = path.is_file()
            path.unlink(missing_ok=True)
            return existed
        except OSError:
            return False

    # -------------------------------------------------------------- restart
    def restart_jarvis(self, reason: str = "overseer") -> bool:
        """Bring Jarvis OS back with memory fully preserved (never reset)."""
        if self._restart_cb is not None:
            try:
                self._restart_cb()
            except Exception as exc:  # noqa: BLE001
                if self.audit:
                    self.audit.record("overseer", "restart", "failed",
                                      {"error": str(exc)[:300]}, actor="overseer")
                return False
            self._jarvis_pid = None
            if self.audit:
                self.audit.record("overseer", "restart", "ok",
                                  {"reason": reason}, actor="overseer")
            return True
        entry = self.root / "jarvis.py"
        if not entry.is_file():
            return False
        argv = [sys.executable or "python", "-u", str(entry)]
        if self.config.feature("background_daemon"):
            argv.append("--daemon")
        log_path = self.config.resolve("data/jarvis_daemon.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creation = 0
        if os.name == "nt":
            creation = (getattr(subprocess, "DETACHED_PROCESS", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            with log_path.open("a", encoding="utf-8") as log:
                popen = subprocess.Popen(argv, cwd=str(self.root), stdout=log, stderr=log,
                                         stdin=subprocess.DEVNULL, creationflags=creation)
        except OSError as exc:
            if self.audit:
                self.audit.record("overseer", "restart", "failed",
                                  {"error": str(exc)[:200]}, actor="overseer")
            return False
        self._jarvis_pid = popen.pid
        self.repair_attempts = 0
        self._clear_control()
        if self.audit:
            self.audit.record("overseer", "restart", "ok",
                              {"pid": popen.pid, "reason": reason,
                               "memory": "preserved"}, actor="overseer")
        return True

    # ---------------------------------------------------------------- status
    def status(self) -> str:
        beat = self.heartbeat()
        alive = self.jarvis_alive()
        age = self.age(beat) if beat else None
        lines = [
            "OverseerAgenticOS",
            f"  running        : {self.running}",
            f"  jarvis alive   : {alive} (pid {beat.get('pid', '-')})",
            f"  heartbeat age  : {f'{age:.1f}s' if age is not None else 'never'} "
            f"(stall threshold {self.stall_timeout}s)",
            f"  auto_repair    : {self.auto_repair}   auto_restart: {self.auto_restart}",
            f"  incidents      : {len(self.incidents)}",
            f"  repairs budget : {self.repair_attempts}/{self.max_repairs}",
        ]
        if self.incidents:
            lines.append("  recent incidents:")
            for incident in self.incidents[-5:]:
                mark = "resolved" if incident.resolved else "OPEN"
                lines.append(f"    [{mark}] {incident.incident_id} "
                             f"{incident.fault}@{incident.layer} - "
                             f"{incident.diagnosis.get('root_cause', '')[:70]}")
        return "\n".join(lines)

    def incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.incidents[-limit:]]

    def scan_report(self) -> str:
        result = self.scan()
        if not result["faults"]:
            return (f"Scan clean at {time.strftime('%H:%M:%S')} - "
                    f"jarvis_alive={result['jarvis_alive']}, all 7 layers ok.")
        lines = [f"Scan found {len(result['faults'])} issue(s):"]
        for fault in result["faults"][:15]:
            lines.append(f"  - {fault['fault']} @ {fault['layer']}: {fault['evidence'][:120]}")
        return "\n".join(lines)


def run_overseer(config_path: str | None = None) -> int:
    """Standalone entry point: python -m jarvis_os.overseer"""
    from .audit import AuditLedger
    from .config import Config
    from .llm import LLMClient
    from .memory import PersistentMemory

    config = Config(config_path)
    audit = AuditLedger(config.resolve(config.get("audit.sqlite_path", "data/audit.db")),
                        config.resolve(config.get("audit.ledger_path", "data/audit_ledger.jsonl")),
                        enabled=config.feature("audit_log", True), actor="overseer").connect()
    memory = PersistentMemory(
        config.resolve(config.get("memory.sqlite_path", "data/memory.db")),
        config.resolve(config.get("memory.vector_path", "data/vectors.json")),
        config.resolve(config.get("memory.lesson_log", "data/lessons.jsonl")),
        config.resolve(config.get("memory.action_log", "data/actions.jsonl")),
    ).open()
    llm = LLMClient(config)
    overseer = OverseerAgenticOS(config, audit=audit, memory=memory, llm=llm)
    print("OverseerAgenticOS watching. Ctrl-C to stop.")
    overseer.start()
    try:
        while True:
            time.sleep(overseer.interval)
    except KeyboardInterrupt:
        overseer.stop()
        print("Overseer stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_overseer())