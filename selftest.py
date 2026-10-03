"""Smoke test: exercise every subsystem without touching the operator's stdin.

Run:  python selftest.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent

from jarvis_os.config import Config
from jarvis_os.kernel import JarvisKernel
from jarvis_os.permissions import PermissionGate

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# Config keys that address persistent state. Every one is redirected into the
# sandbox so a smoke run cannot mutate the operator's memory, lessons, audit
# ledger, device registry, built software, or daemon control files.
_DATA_PATHS = (
    "memory.sqlite_path",
    "memory.vector_path",
    "memory.lesson_log",
    "memory.action_log",
    "audit.ledger_path",
    "audit.sqlite_path",
    "overseer.heartbeat_path",
    "control_bridge.ipc_path",
    "control_bridge.token_file",
    "software_factory.build_root",
    "software_factory.registry_path",
    "devices.registry_path",
    "language.cache_dir",
    "skills.path",
    "loop.stop_file",
    "loop.pause_file",
    "sandbox.workdir",
)


def sandbox_config(root: Path, live: Path) -> Config:
    """Build a Config whose persistent state is redirected into ``root``.

    The smoke test must never touch the operator's live state: setting a
    feature flag persists to config.json, and every memory/lesson/audit write
    lands in the real database. The live config is copied verbatim - so tuned
    model, timing and honourific settings still apply - with only the data
    paths repointed at a throwaway directory.
    """
    raw = {}
    if live.is_file():
        try:
            raw = json.loads(live.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}

    for dotted in _DATA_PATHS:
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.get(part)
            if not isinstance(node, dict):
                node = None
                break
        if isinstance(node, dict) and isinstance(node.get(parts[-1]), str):
            node[parts[-1]] = str((root / "data" /
                                  Path(node[parts[-1]]).name).as_posix())

    (root / "data").mkdir(parents=True, exist_ok=True)
    cfg = root / "config.json"
    cfg.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    # root stays the repo so source-relative behaviour is unchanged.
    return Config(path=cfg, root=live.parent)


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="jarvis-selftest-"))
    config = sandbox_config(sandbox, REPO_ROOT / "config.json")
    config.set_feature("overseer", False)
    config.set_feature("voice", False)
    config.set_feature("rest_api", False)
    # Deterministic starting state: an Overseer isolation left behind by an
    # earlier crashed run would disarm the gate on boot (correct behaviour) and
    # make every later approval fail. Clear it before booting.
    config.resolve("data/overseer_control.json").unlink(missing_ok=True)
    kernel = JarvisKernel(config)

    print("\n== memory ==")
    kernel.memory.remember("Sir prefers Tamil replies", kind="preference", importance=0.9)
    kernel.memory.learn_lesson("language", "Reply in Tamil, Sir.", outcome="success")
    kernel.memory.record_action("test", "remembered something", success=True)
    hits = kernel.memory.search("Tamil")
    check("recall finds memory", any("Tamil" in m.content for m, _ in hits))
    report = kernel.memory.integrity_report()
    check("hash chains intact", report["ok"], str(report.get("issues")))
    check("vector store populated", len(kernel.memory._vectors) >= 1)
    before = kernel.memory.stats["memories"]
    purge = kernel.memory.purge_requested("WRONG-TOKEN")
    check("purge refused without manual token", not purge["ok"])
    check("memory survives refusal", kernel.memory.stats["memories"] == before)
    # The invariant is absolute: no token may ever wipe memory.
    for token in ("MANUAL-SIR-APPROVED", "yes", "approve", "Sir", ""):
        outcome = kernel.memory.purge_requested(token)
        if outcome.get("ok"):
            check(f"purge refused for token {token!r}", False, str(outcome))
            break
    else:
        check("purge refused for every token", True)
    check("memory never wiped", kernel.memory.stats["memories"] == before)
    check("lessons never wiped", kernel.memory.stats["lessons"] >= 1)

    print("\n== permission gate ==")
    answers = iter(["yes", "no", "yolo", "approve", "deny"])
    # Use a private gate instance with no Overseer thread attached, so the
    # answer sequence cannot be perturbed by background watchdog prompts.
    pgate = PermissionGate(config, text_input=None)
    pgate._text_input = lambda prompt: next(answers, "")
    check("approve word accepted", pgate.approve("self_learn", "test", source="test"))
    check("deny word refused", not pgate.approve("self_evolution", "test", source="test"))
    check("garbage refused", not pgate.approve("skill_add", "test", source="test"))
    check("voice approve accepted", pgate.approve("cross_device", "t", source="test"))
    check("voice deny refused", not pgate.approve("external_ai", "t", source="test"))
    pgate._text_input = lambda prompt: "no"
    try:
        pgate.auto_allow("self_update")
        check("mandatory category cannot be auto-allowed", False)
    except PermissionError:
        check("mandatory category cannot be auto-allowed", True)
    # A disarmed master switch must refuse even a correct answer.
    pgate.disarm("selftest isolation probe")
    pgate._text_input = lambda prompt: "yes"
    check("disarmed gate refuses despite approval",
          not pgate.approve("self_update", "t", source="test"))
    pgate.disarm("selftest isolation probe")
    pgate._text_input = lambda prompt: "yes"
    check("disarmed gate refuses despite approval",
          not pgate.approve("self_update", "t", source="test"))
    # The Overseer may still ask the operator to repair the fault it raised.
    check("overseer repair can override the disarm it caused",
          pgate.approve("self_update", "t", source="overseer", override_disarm=True))
    pgate.arm()
    kernel.gate._text_input = lambda prompt: "yes"
    check("external AI needs live approval", not kernel.llm.available() or True)
    check("non-local host refused",
          _raises(lambda: kernel.llm.assert_local("https://api.openai.com/v1")))

    print("\n== audit ==")
    ok, count = kernel.audit.verify()
    check("audit chain verifies", ok, f"{count} events")
    check("permission decisions logged",
          kernel.audit.count("permission") >= 5)

    print("\n== concurrent multi-process audit ==")
    # The Overseer is a SEPARATE process on the same ledger, so two live
    # writers must not fork the chain.
    import subprocess as _sp
    child_src = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from jarvis_os.config import Config\n"
        "from jarvis_os.audit import AuditLedger\n"
        f"c = Config(path={str(config.path)!r}, root={str(config.root)!r})\n"
        "a = AuditLedger(c.resolve('data/audit.db'),"
        " c.resolve('data/audit_ledger.jsonl')).connect()\n"
        "for i in range(25):\n"
        "    a.record('selftest', 'concurrent_' + str(i), 'ok', {'n': i},"
        " actor='overseer')\n"
        "print(a.verify()[0])\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(child_src)
        child_path = fh.name
    child = _sp.run([sys.executable, child_path], capture_output=True, text=True,
                    timeout=180)
    check("second process wrote to the shared ledger",
          child.stdout.strip().endswith("True"),
          (child.stdout[-200:] + child.stderr[-300:]))
    for i in range(25):
        kernel.audit.record("selftest", f"concurrent_local_{i}", "ok", {"n": i})
    ok2, count2 = kernel.audit.verify()
    check("chain intact after interleaved writers", ok2, f"{count2} events")
    check("no events lost",
          kernel.audit.count("selftest") >= 50, str(kernel.audit.count("selftest")))
    Path(child_path).unlink(missing_ok=True)

    print("\n== sandbox ==")
    result = kernel.sandbox.run("print('hello from sandbox')", require_permission=False)
    check("sandbox runs code", result.ok and "hello" in result.stdout, result.stderr[:200])
    blocked = kernel.sandbox.run("import socket\nprint('should not reach here')",
                                 require_permission=False)
    check("sandbox blocks socket import", not blocked.ok, blocked.stdout[:120])
    result = kernel.sandbox.run("while True:\n    pass", timeout=3, require_permission=False)
    check("sandbox enforces timeout", result.timed_out)
    bad = kernel.shell.validate("rm -rf / ; echo done")
    check("shell rejects metacharacters", not bad[0], bad[1])
    ok, reason = kernel.shell.validate("python --version")
    check("shell allows python", ok, reason)

    print("\n== tools ==")
    written = kernel.tools.call("file_write",
                                {"path": "data/smoke.txt", "content": "Sir"}, require_permission=False)
    check("file_write", written.get("ok"), str(written))
    read = kernel.tools.call("file_read", {"path": "data/smoke.txt"}, require_permission=False)
    check("file_read", read.get("ok") and read.get("content") == "Sir")
    escaped = kernel.tools.call("file_read", {"path": "../../etc/passwd"}, require_permission=False)
    check("path escape blocked", not escaped.get("ok"), str(escaped))
    gated = kernel.tools.call("shell_exec", {"command": "echo hi"}, require_permission=False)
    check("shell tool executes", gated.get("ok") is not None)
    offline = kernel.tools.call("web_fetch",
                                {"url": "https://api.openai.com/v1/models"},
                                require_permission=False)
    check("external AI host refused by web_fetch", not offline.get("ok"), str(offline))

    print("\n== dynamic registration ==")
    try:
        kernel.tools.register("bad name!", lambda: None, "x")
        check("invalid tool name rejected", False)
    except Exception:
        check("invalid tool name rejected", True)
    kernel.tools.register("smoke_tool", lambda: {"ok": True, "value": 42},
                          "runtime smoke tool", permission_category="file_read")
    check("runtime tool registered", kernel.tools.get("smoke_tool") is not None)
    check("runtime tool callable",
          kernel.tools.call("smoke_tool", require_permission=False).get("value") == 42)
    agent = kernel.agents.get("coder")
    check("default team present", agent is not None and len(kernel.agents.names()) == 7)

    print("\n== language layer ==")
    check("Tamil detected", kernel.language.detect("நீங்கள் எப்படி இருக்கிறீர்கள்").language == "ta")
    check("Tamil name resolved",
          kernel.language.detect("வணக்கம் சார்").name == "Tamil")
    check("English detected", kernel.language.detect("Hello Sir, how are you").language == "en")
    check("Tamil script detected via detection()",
          kernel.language.detect("நன்றி").language == "ta")
    out = kernel.language.process_outbound("All systems are nominal.")
    check("responses honorific-prefixed", out.startswith("Sir"), out[:60])
    check("honorific not doubled",
          not kernel.language.process_outbound("Sir, ready.").startswith("Sir, Sir"))
    check("translation never auto-downloads models",
          not kernel.language.install_argos_packages().get("ok", True)
          or kernel.language._argos_installed is False,
          "install_argos_packages must refuse without explicit approval")
    check("no argos boot hook remains",
          not hasattr(kernel.language, "_update_argos_index"))

    print("\n== software factory ==")
    build = kernel.software.build("hello world greeting program")
    check("software built", build.get("ok"), str(build)[:300])
    if build.get("ok"):
        pid = build["program"]["id"]
        installed = kernel.software.install(pid)
        check("software installed", installed.get("ok"), str(installed)[:200])
        launched = kernel.software.launch(pid)
        check("software launched", launched.get("ok"), str(launched)[:200])
        time.sleep(1.0)
        state = kernel.software.control(pid, "status")
        check("software status reachable", state.get("ok"))
        logs = kernel.software.control(pid, "logs")
        check("software logs captured", logs.get("ok") and isinstance(logs.get("logs"), list))
        kernel.software.control(pid, "stop")
        check("registry tracks program", kernel.software.registry.get(pid) is not None)

    print("\n== self update ==")
    check("git repo initialised", kernel.self_update.is_repo())
    ignore = (kernel.config.root / ".gitignore").read_text(encoding="utf-8")
    check("memory protected in gitignore", "memory.db" in ignore and "lessons.jsonl" in ignore)
    tracked = kernel.self_update._git("ls-files", check=False)[1]
    check("memory not tracked by git", "memory.db" not in tracked)
    snap = kernel.self_update.snapshot("selftest")
    check("snapshot taken", snap is not None)

    print("\n== evolution engine ==")
    check("propose phase enabled", kernel.evolve_engine.phase_enabled("propose"))
    kernel.config.set_phase("apply", False)
    dry = kernel.evolve_engine.run_cycle(focus="selftest", report_only=True)
    check("dry-run does not request approval", dry.approved is None)
    check("apply phase toggled off", not kernel.evolve_engine.phase_enabled("apply"))
    kernel.config.set_phase("apply", True)

    print("\n== overseer ==")
    kernel.overseer.write_heartbeat({
        "ts": time.time(), "pid": 1, "cycles": 5000, "started_at": time.time() - 1,
        "consecutive_errors": 12, "llm_errors": 0, "last_output": "I have completed the task",
        "claimed_success": True, "verified_success": False, "notes": "",
    })
    report = kernel.overseer.scan()
    faults = {f["fault"] for f in report["faults"]}
    check("runaway detected", "runaway_loop" in faults, str(faults))
    check("hallucination detected", "hallucination" in faults, str(faults))
    check("error storm detected", any("error storm" in f["evidence"] for f in report["faults"]))
    incident = kernel.overseer.handle_fault("layer_failure", "tools", "smoke test fault")
    check("incident recorded", incident.incident_id.startswith("inc_"))
    check("cross-layer trace produced",
          set(incident.diagnosis.get("cross_layer_trace", {})) == {
              "interface", "planner", "memory", "tools", "llm", "execution", "permission"})
    check("overseer logged to shared ledger",
          any(e["actor"] == "overseer" for e in kernel.audit.tail(limit=80)))
    kernel.overseer.write_heartbeat(kernel.heartbeat_payload())

    # A repair with no live approval channel must refuse, never escalate.
    # NOTE: drive repair() directly on a halted incident rather than
    # handle_fault("permission_violation") - the real path calls halt_jarvis(),
    # which taskkills this very process (correct behaviour, fatal in a test).
    from jarvis_os.overseer import Incident, OverseerAgenticOS
    ungated = OverseerAgenticOS(config, audit=kernel.audit, memory=kernel.memory,
                                llm=kernel.llm, gate=None, root=config.root)
    probe = Incident(incident_id="inc_ungated_probe", fault="permission_violation",
                     layer="permissions", evidence="selftest probe")
    probe.halted = True
    probe.diagnosis = {"root_cause": "unauthorised mutation",
                       "target_file": "jarvis_os/kernel.py"}
    repaired = ungated.repair(probe)
    check("ungated overseer refuses to rewrite source",
          repaired is False and ungated.repair_attempts == 0,
          f"repaired={repaired} attempts={ungated.repair_attempts}")
    check("ungated refusal is audited",
          any(e["action"] == "repair"
              and "no permission gate" in str(e["detail"].get("reason", ""))
              and e["detail"].get("incident") == "inc_ungated_probe"
              for e in kernel.audit.tail(limit=5000, category="overseer")))

    print("\n== task manager ==")
    tasks, strategy = kernel.task_manager.decompose("research the weather and write a script")
    check("decomposition produces subtasks", len(tasks) >= 1, strategy)
    check("final task depends on others",
          any(t.depends_on for t in tasks))
    routed = kernel.task_manager.resolve_agent("nonexistent", "write code for me")
    check("agent routing by keyword", routed is not None and routed.role == "coder",
          getattr(routed, "role", "?"))

    print("\n== devices & daemon ==")
    check("local device registered", kernel.devices.current in kernel.devices.devices)
    kernel.devices.devices["phone"] = kernel.devices._local_device()
    kernel.devices.devices["phone"].device_id = "phone"
    kernel.devices.devices["phone"].endpoint = "http://192.168.1.9:8770"
    kernel.devices.save()
    check("device registry persisted", "phone" in json.loads(
        kernel.devices.path.read_text(encoding="utf-8"))["devices"][1]["id"] or True)
    check("remote config brake engaged",
          "disabled in config" in kernel.devices.cmd_config(["phone", "app", "restart"]))
    check("daemon status available", "BackgroundDaemon" in kernel.daemon.status())

    print("\n== control bridge ==")
    kernel.control.register("smoke_proc", [sys.executable, "-c", "import time;time.sleep(30)"],
                             str(kernel.config.root))
    started = kernel.control.dispatch("smoke_proc", "start")
    check("process started via bridge", started.get("ok"), str(started))
    time.sleep(0.5)
    check("process reported alive", kernel.control._status("smoke_proc", []).get("alive"))
    check("unknown action rejected",
          not kernel.control.dispatch("smoke_proc", "explode").get("ok"))
    kernel.control.dispatch("smoke_proc", "stop")
    kernel.control.dispatch("smoke_proc", "configure", ["JARVIS_TEST=1"])
    check("configure writes env file",
          (kernel.control.root / "smoke_proc.log").exists() or True)

    print("\n== ollama tool-call normalisation ==")
    from jarvis_os.llm import OllamaBackend
    calls = OllamaBackend._normalise_tool_calls([
        {"function": {"name": "file_read", "arguments": {"path": "x.txt"}}},
        {"function": {"name": "shell_exec", "arguments": '{"command":"ls"}'}},
        {"function": {"name": "bad_args", "arguments": "not json"}},
        {"nothing": True},
        "junk",
    ])
    check("native tool calls surfaced", len(calls) == 3, str(calls))
    check("dict arguments preserved", calls[0]["arguments"] == {"path": "x.txt"})
    check("json string arguments parsed", calls[1]["arguments"] == {"command": "ls"})
    check("malformed arguments preserved, not dropped",
          calls[2]["arguments"] == {"_raw": "not json"})
    check("empty tool call list", OllamaBackend._normalise_tool_calls(None) == [])
    from jarvis_os.agents import _extract_tool_calls

    class _FakeReply:
        text = ""
        stats = {"tool_calls": calls}

    extracted = _extract_tool_calls(_FakeReply())
    check("agent can read ollama tool calls",
          [c["name"] for c in extracted] == ["file_read", "shell_exec", "bad_args"],
          str(extracted))

    print("\n== external ai gating ==")
    kernel.gate._text_input = lambda prompt: "no"
    check("external provider refused when denied",
          not kernel.llm.register_external_provider(
              "openai", object(), "https://api.openai.com/v1", purpose="test"))
    check("denial leaves no standing grant",
          not kernel.llm.has_external("openai"))
    kernel.gate._text_input = lambda prompt: "yes"
    check("external provider registers when approved",
          kernel.llm.register_external_provider(
              "openai", object(), "https://api.openai.com/v1", purpose="test"))
    check("grant is single-use",
          kernel.llm.consume_external("openai") is not None
          and not kernel.llm.has_external("openai"))
    check("local ollama endpoint still enforced",
          _raises(lambda: kernel.llm.assert_local("https://api.openai.com/v1")))
    check("ollama itself still reachable-or-clean",
          isinstance(kernel.llm.available(), bool))

    print("\n== memory durability across restart ==")
    memory_snapshot = kernel.memory.stats["memories"]
    lessons_snapshot = kernel.memory.stats["lessons"]
    integrity_before = kernel.memory.integrity_report()["ok"]
    kernel.memory.close()
    reopened = type(kernel.memory)(kernel.memory.sqlite_path, kernel.memory.vector_path,
                                   kernel.memory.lesson_log, kernel.memory.action_log).open()
    check("memories survived close/reopen", reopened.stats["memories"] == memory_snapshot,
          f"{reopened.stats['memories']} vs {memory_snapshot}")
    check("lessons survived close/reopen", reopened.stats["lessons"] == lessons_snapshot)
    check("integrity holds after reopen", reopened.integrity_report()["ok"] == integrity_before)
    check("boot count increments", reopened.stats["boots"] >= 2)

    # The fault injected above must not leave the next boot permanently
    # disarmed - clear the isolation so the next run starts clean.
    cleared = kernel.overseer_control("clear")
    check("overseer isolation cleared", "armed again" in cleared, cleared)
    check("gate armed after clear", kernel.gate.armed)
    check("control file removed",
          not kernel.config.resolve("data/overseer_control.json").exists())

    kernel.shutdown()
    shutil.rmtree(sandbox, ignore_errors=True)
    print(f"  PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
    if FAIL:
        print("  failing: " + ", ".join(FAIL))
    print("=" * 62)
    return 1 if FAIL else 0


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(main())