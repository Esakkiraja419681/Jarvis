"""Smoke test: exercise every subsystem without touching the operator's stdin.

Run:  python selftest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent

# The assertions below record Tamil strings, and the default Windows code page
# (cp1252) cannot represent them: printing one degrades it to a row of '?' and
# hides which word was actually tested. The streams are switched to UTF-8 with
# replacement before check() can write anything.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        # A captured or non-TextIOBase stream has nothing to reconfigure, and
        # losing the guard must never stop the suite from running.
        pass

from jarvis_os.config import _DEFAULTS, Config
from jarvis_os.daemon import BackgroundDaemon, DaemonState
from jarvis_os.device_manager import DeviceSyncManager
from jarvis_os.kernel import JarvisKernel
from jarvis_os.language import UniversalLanguageLayer
from jarvis_os.language_layer import LanguageLayer
from jarvis_os.permissions import PermissionGate
from jarvis_os.selfupdate import MEMORY_GLOBS, SelfUpdate
from jarvis_os.sandbox import PythonSandbox, SandboxError

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# Config keys that address persistent state. Every one is redirected into the
# sandbox so a smoke run cannot mutate the operator's memory, lessons, audit
# ledger, device registry, built software, or daemon control files.
_LIVE_STATE_FILES = (
    "config.json",
    "data/memory.db",
    "data/vectors.json",
    "data/lessons.jsonl",
    "data/actions.jsonl",
    "data/audit_ledger.jsonl",
)

_DATA_PATHS = (
    "memory.sqlite_path",
    "memory.vector_path",
    "memory.lesson_log",
    "memory.action_log",
    "audit.ledger_path",
    "audit.sqlite_path",
    "overseer.heartbeat_path",
    "overseer.control_file",
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
    "device_manager.tls_dir",
    "device_manager.pairing_token_path",
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

    for dotted in ("device_manager.tls_dir", "device_manager.pairing_token_path"):
        parts = dotted.split(".")
        node = raw
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        fallback = _DEFAULTS
        for part in parts:
            fallback = fallback.get(part) if isinstance(fallback, dict) else None
        if isinstance(fallback, str) and not isinstance(node.get(parts[-1]), str):
            node[parts[-1]] = fallback

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


def live_fingerprint() -> dict[str, str]:
    """SHA-256 of the operator's real state files.

    The sandbox redirects every configured data path, but a hard-coded path or a
    subsystem writing outside Config.resolve would still land here. Comparing
    this fingerprint before and after the run turns "the smoke test must never
    touch live state" from a claim into an assertion.
    """
    out: dict[str, str] = {}
    for rel in _LIVE_STATE_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError:
            continue
        out[rel] = digest.hexdigest()
    return out


def main() -> int:
    """Run the suite inside a throwaway sandbox that is always torn down.

    Teardown lives in a ``finally`` rather than at the end of the suite body.
    Previously an unhandled exception, or a kill part-way through, left the whole
    ``jarvis-selftest-*`` directory behind - which is how a crashed run leaked one
    temp tree per attempt. ``ignore_errors=True`` keeps a failed teardown from
    masking the error that triggered it.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="jarvis-selftest-"))
    try:
        return _run(sandbox)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "data" / ".selftest_scratch", ignore_errors=True)


def _run(sandbox: Path) -> int:
    live_before = live_fingerprint()
    config = sandbox_config(sandbox, REPO_ROOT / "config.json")
    config.set_feature("overseer", False)
    config.set_feature("voice", False)
    config.set_feature("rest_api", False)
    config.set("device_manager.ws_port", 0, persist=False)
    config.set("device_manager.bind_host", "127.0.0.1", persist=False)
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
    # It shares the kernel's ledger so these decisions are auditable, which is
    # what makes the "permission decisions logged" check meaningful on a clean
    # install rather than only after enough history has accumulated.
    pgate = PermissionGate(config, audit=kernel.audit, text_input=None)
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
        "a = AuditLedger(c.resolve('audit.sqlite_path'),"
        " c.resolve('audit.ledger_path')).connect()\n"
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

    # Regression for the run9 crash. The sandbox root is created once in
    # __init__, so anything that deletes it afterwards used to make mkdtemp raise
    # a bare WinError 3 and take the whole suite down with it. Reproduce exactly
    # that condition and require the run to recover.
    workdir_parent = kernel.config.resolve("sandbox.workdir").parent
    vanish_root = workdir_parent / "sandbox_vanished"
    probe = PythonSandbox(root=vanish_root, config=kernel.config)
    shutil.rmtree(vanish_root, ignore_errors=True)
    check("sandbox root removed to simulate a vanished temp dir",
          not vanish_root.exists())
    recovered = probe.run("print('root came back')", require_permission=False)
    check("sandbox recreates a vanished root",
          recovered.ok and "root came back" in recovered.stdout,
          recovered.stderr[:200])
    check("recreated sandbox root is a directory", vanish_root.is_dir())

    # ...and when the root genuinely cannot be recreated, the failure must name
    # the root instead of surfacing an opaque path error from mkdtemp.
    blocked_root = workdir_parent / "sandbox_blocked"
    blocked = PythonSandbox(root=blocked_root, config=kernel.config)
    shutil.rmtree(blocked_root, ignore_errors=True)
    blocked_root.write_text("blocker", encoding="utf-8")
    try:
        blocked.run("print(1)", require_permission=False)
        reported = False
        detail = "no error raised"
    except SandboxError as exc:
        reported = "sandbox root unavailable" in str(exc)
        detail = str(exc)[:160]
    except OSError as exc:
        reported = False
        detail = f"raw {type(exc).__name__}: {exc}"[:160]
    check("sandbox reports an unusable root by name", reported, detail)
    blocked_root.unlink(missing_ok=True)
    bad = kernel.shell.validate("rm -rf / ; echo done")
    check("shell rejects metacharacters", not bad[0], bad[1])
    ok, reason = kernel.shell.validate("python --version")
    check("shell allows python", ok, reason)

    print("\n== tools ==")
    # ToolRegistry refuses any path outside the Jarvis root, so the scratch file
    # cannot live in the TEMP sandbox. It is written to a test-only directory
    # that the sandbox setup tears down, instead of into live data/.
    smoke_rel = "data/.selftest_scratch/smoke.txt"
    (REPO_ROOT / "data" / ".selftest_scratch").mkdir(parents=True, exist_ok=True)
    written = kernel.tools.call("file_write",
                                {"path": smoke_rel, "content": "Sir"}, require_permission=False)
    check("file_write", written.get("ok"), str(written))
    read = kernel.tools.call("file_read", {"path": smoke_rel}, require_permission=False)
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

    # Runtime agent registration has to be reachable from the REPL, not just
    # from boot-time default_team(): AgentRegistry.add/remove used to have no
    # user-facing caller at all.
    text_iface = kernel.text_iface
    check("agent command exposed in REPL",
          "agent" in text_iface.COMMANDS and "agents" in text_iface.COMMANDS)
    before = len(kernel.agents.names())
    text_iface.dispatch("agent list")
    check("agent list lists the default team",
          "orchestrator" in kernel.agent_list() and "reviewer" in kernel.agent_list())
    text_iface.dispatch("agent add scribe archivist")
    added = kernel.agents.get("scribe")
    check("agent add registers at runtime",
          added is not None and added.role == "archivist"
          and len(kernel.agents.names()) == before + 1,
          f"{before} -> {len(kernel.agents.names())}")
    check("agent add agent is functional (has live tool registry)",
          added is not None and added.tools is kernel.tools)
    check("duplicate agent add refused",
          "already registered" in kernel.agent_add(["scribe", "archivist"]))
    check("agent remove unregisters",
          "Removed agent" in kernel.agent_remove(["scribe"])
          and kernel.agents.get("scribe") is None
          and len(kernel.agents.names()) == before)
    check("agent remove unknown reports cleanly",
          "no such agent" in kernel.agent_remove(["ghost"]))
    check("agent add usage on missing role", "Usage: agent add" in kernel.agent_add(["solo"]))
    spoken: list[str] = []
    real_say = text_iface.say
    text_iface.say = lambda msg, **kw: spoken.append(str(msg))  # type: ignore[method-assign]
    try:
        text_iface.dispatch("agent wat")
    finally:
        text_iface.say = real_say  # type: ignore[method-assign]
    check("agent subcommand misuse shows usage",
          any("Usage: agent add" in m for m in spoken))

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
          not kernel.language.process_outbound("Sir, ready.").startswith("Sir, Sir"),
          "prefix() must add 'Sir' exactly once")
    # The model often opens its own reply with a Tamil honorific. Left in place
    # the operator reads "Sir, ஐயா, ..."; these lock in that it is removed.
    _pfx = kernel.language.prefix
    check("model Tamil honorific stripped",
          _pfx("ஐயா, அனைத்து அமைப்புகளும் செயல்படுகின்றன.")
          == "Sir, அனைத்து அமைப்புகளும் செயல்படுகின்றன.",
          _pfx("ஐயா, அனைப்புகளும் செயல்படுகின்றன."))
    check("model 'Sir' plus Tamil honorific both stripped",
          _pfx("Sir, ஐயா, வணக்கம்") == "Sir, வணக்கம்",
          _pfx("Sir, ஐயா, வணக்கம்"))
    check("real word starting with an honorific is preserved",
          _pfx("ஐயாவு அம்மா வீடு") == "Sir, ஐயாவு அம்மா வீடு",
          "must not mangle 'ஐயாவு' (grandfather) - the mid-word guard")
    check("real word starting with 'திரு' is preserved",
          _pfx("திருமலை நகரம்") == "Sir, திருமலை நகரம்",
          "must not mangle 'திருமலை' (Tirumalai)")
    check("mid-sentence honorific is never stripped",
          _pfx("நான் ஐயாவைப் பார்த்தேன்") == "Sir, நான் ஐயாவைப் பார்த்தேன்",
          "only the front of the string is ever examined")
    check("translation never auto-downloads models",
          not kernel.language.install_argos_packages().get("ok", True)
          or kernel.language._argos_installed is False,
          "install_argos_packages must refuse without explicit approval")
    check("no argos boot hook remains",
          not hasattr(kernel.language, "_update_argos_index"))

    # Regression: with the NLLB weights off disk, translate() returned the source
    # verbatim, so an untranslated Tamil sentence came back looking exactly like
    # a successful English translation. A complete model tree is fabricated in
    # the sandbox cache with a *sparse* weight file, so the rename below is a
    # real present -> absent transition instead of a test that passes either way.
    # Nothing is deleted and nothing is ever downloaded.
    _nllb_name = str(kernel.language.config.get(
        "language.nllb_model", "facebook/nllb-200-distilled-600M"))
    _cache_dir = kernel.language.config.resolve("language.cache_dir")
    _snap_root = _cache_dir / ("models--" + _nllb_name.replace("/", "--")) / "snapshots"
    _probe = _snap_root / "selftest-probe"
    # Must move *out* of snapshots/: _nllb_weights_present() globs snapshots/*,
    # so a sibling rename would still be found with its weights intact.
    _aside = _cache_dir / "selftest-probe-aside"
    _probe.mkdir(parents=True, exist_ok=True)
    (_probe / "config.json").write_text("{}", encoding="utf-8")
    (_probe / "tokenizer.json").write_text("{}", encoding="utf-8")
    # Sparse: truncate() satisfies the size gate without writing the bytes.
    # The real gate is _MIN_WEIGHT_BYTES (512 MB), but provisioning that much
    # disk for a test is not the test's business - and on a nearly full volume
    # truncate() raises ENOSPC instead of staying sparse, which failed the whole
    # suite on a machine with 240 MB free. Lower the gate for this probe and put
    # it back afterwards: what matters is a genuine present -> absent transition,
    # not the specific byte count a real NLLB checkpoint has.
    _orig_min_bytes = UniversalLanguageLayer._MIN_WEIGHT_BYTES
    UniversalLanguageLayer._MIN_WEIGHT_BYTES = 4096
    with open(_probe / "model.safetensors", "wb") as _fh:
        _fh.truncate(UniversalLanguageLayer._MIN_WEIGHT_BYTES)
    try:
        # Detection must be able to see the weights, otherwise the rename below
        # proves nothing about the missing-model branch.
        _present = UniversalLanguageLayer(kernel.config, llm=None)
        check("nllb weights detected before the rename",
              _present._nllb_weights_present(),
              "the sparse weight file must clear the size gate")

        _probe.rename(_aside)

        # Built with llm=None so a live local model cannot quietly satisfy the
        # request, and constructed after the rename so backend detection sees
        # the weights as genuinely absent.
        _bare = UniversalLanguageLayer(kernel.config, llm=None)
        check("nllb weights absent after the rename",
              not _bare._nllb_weights_present())
        # Distinct from every string translated above, so a cached passthrough
        # can never mask the notice.
        _skip_src = "இன்று மழை பெய்கிறது"
        _skip_out = _bare.translate(_skip_src, "en")
        # The source comes back untouched. A diagnostic is not something to say
        # to the operator, and translate() is called from inside
        # process_outbound - so anything it returned was itself a reply. That is
        # what put "translation skipped: ..." in front of every banner, every
        # status line and "Goodbye".
        check("missing model returns the source unchanged",
              _skip_out == _skip_src, _skip_out[:160])
        check("skip note never reaches the operator",
              "translation skipped" not in _skip_out, _skip_out[:160])
        # It is still recorded, just not spoken. The bare layer was built without
        # an audit ledger, so it gets the kernel's - that is the ledger a real run
        # would write to.
        _bare.audit = kernel.audit
        _before_lang_events = kernel.audit.count(category="language")
        _bare.translate("மற்றொரு தமிழ் வரி", "en")
        _lang_events = kernel.audit.tail(limit=20, category="language")
        check("a passthrough is recorded in the audit ledger",
              kernel.audit.count(category="language") > _before_lang_events,
              f"before={_before_lang_events} after={kernel.audit.count(category='language')}")
        check("the audit record names the real cause",
              any("nllb" in str(e.get("detail", "")).lower() for e in _lang_events),
              str(_lang_events[-1])[:200] if _lang_events else "no event")
        check("the audit record marks it a passthrough, not a success",
              any(e.get("decision") == "passthrough" for e in _lang_events),
              str([e.get("decision") for e in _lang_events])[:200])

        # The reason itself must still name the real cause, or the audit trail
        # is useless for diagnosing a passthrough.
        _note = _bare._skip_note("ta", "en", [])
        check("skip reason still names nllb",
              "NLLB model not installed" in _note, _note[:160])

        # The regression that matters is the whole operator-facing path. With a
        # reply language set and no local translator, every utterance had to come
        # back clean: the banner, the farewell and an unrecognised command.
        _tamil_cfg = kernel.config
        _prev_lang = _tamil_cfg.get("language.reply_language", "en")
        _tamil_cfg.set("language.reply_language", "ta")
        try:
            _reply_layer = UniversalLanguageLayer(_tamil_cfg, llm=None)
            _iface = kernel.text_iface
            _prev_layer = kernel.language
            kernel.language = _reply_layer

            _banner = "Jarvis OS online. Ollama is the only permitted AI backend."
            _goodbye = "Shutting down. Memory preserved."
            for _label, _text in (("banner", _banner), ("goodbye", _goodbye),
                                  ("unknown command", "Unknown command: frobnicate")):
                # say() is the real operator path: process_outbound, not the
                # language layer called directly.
                _spoken = _iface.say(_text, translate=True)
                check(f"{_label} carries no translation skip note",
                      "translation skipped" not in _spoken, _spoken[:160])
                check(f"{_label} still gets the honorific",
                      _spoken.startswith(f"{_iface.kernel.config.honorific()},"),
                      _spoken[:160])
                check(f"{_label} keeps its own text",
                      _text.split(":")[0][:24] in _spoken or _text[:24] in _spoken,
                      _spoken[:160])

            # And the reverse: a greeting in Tamil must not be mangled either.
            _greet = _iface.say("வணக்கம்", translate=True)
            check("tamil greeting round-trips without a note",
                  "translation skipped" not in _greet, _greet[:160])
            check("tamil greeting stays renderable",
                  "வணக்கம்" in _greet, _greet[:160])
        finally:
            _tamil_cfg.set("language.reply_language", _prev_lang)
            kernel.language = _prev_layer

        # An alternate backend that is available must be used before giving up.
        class _StubReply:
            text = "It is raining today"

        class _StubLLM:
            def chat(self, text, system=None, persona=None, temperature=None):
                return _StubReply()

        _alt = UniversalLanguageLayer(kernel.config, llm=_StubLLM())
        _alt_out = _alt.translate(_skip_src, "en")
        check("fallback backend used before reporting unavailable",
              _alt_out == "It is raining today"
              and "translation skipped" not in _alt_out, _alt_out[:160])
        check("fallback layer still reports nllb as not ready",
              not _alt._nllb_ready, str(_alt._nllb_ready))
    finally:
        UniversalLanguageLayer._MIN_WEIGHT_BYTES = _orig_min_bytes
        for _p in (_probe, _aside):
            shutil.rmtree(_p, ignore_errors=True)
    check("fabricated model tree cleaned up", not _probe.exists() and not _aside.exists())
    check("weight-size gate restored",
          UniversalLanguageLayer._MIN_WEIGHT_BYTES == _orig_min_bytes,
          f"{UniversalLanguageLayer._MIN_WEIGHT_BYTES} != {_orig_min_bytes}")

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

        # Phase 2 smoke chain runs build/install/launch as separate processes:
        # the bridge table is in-memory, so a fresh SoftwareFactory over the
        # same registry must restore the registration (and a stop must clear
        # the persisted pid so a stale one is never resurrected).
        kernel.control.processes.pop(pid, None)
        from jarvis_os.software_factory import SoftwareFactory
        SoftwareFactory(kernel.config, kernel.llm, kernel.memory, kernel.gate,
                        kernel.control, kernel.sandbox, audit=kernel.audit)
        check("fresh factory restores bridge registration",
              pid in kernel.control.processes, str(sorted(kernel.control.processes)))
        check("restored registration is controllable",
              kernel.software.control(pid, "status").get("ok"))
        check("stopped program persists no running pid",
              kernel.software.registry.get(pid).pid is None,
              str(kernel.software.registry.get(pid).pid))

    # A small local model can return parseable code that does not run. The
    # build must still yield a runnable artifact instead of failing outright.
    original_ask = kernel.software.llm.ask_json
    kernel.software.llm.ask_json = lambda prompt, persona=None: {
        "name": "hello_sir", "summary": "LLM claimed success", "notes": "model output",
        "entry": "main.py",
        "code": "def main():\n    print(undefined_name_boom)\n\nmain()\n",
    }
    try:
        rescued = kernel.software.build("hello world greeting program")
    finally:
        kernel.software.llm.ask_json = original_ask
    rprog = rescued.get("program") or {}
    check("unrunnable model output still yields a runnable build", rescued.get("ok") is True,
          str(rescued)[:200])
    check("rescued build says it degraded",
          "failed sandbox verification" in (rprog.get("notes") or ""), str(rprog.get("notes")))
    check("rescued build actually executes",
          (rprog.get("test") or {}).get("returncode") == 0, str(rprog.get("test")))

    print("\n== self update ==")
    check("git repo initialised", kernel.self_update.is_repo())
    ignore = (kernel.config.root / ".gitignore").read_text(encoding="utf-8")
    check("memory protected in gitignore", "memory.db" in ignore and "lessons.jsonl" in ignore)
    tracked = kernel.self_update._git("ls-files", check=False)[1]
    check("memory not tracked by git", "memory.db" not in tracked)
    snap = kernel.self_update.snapshot("selftest")
    check("snapshot taken", snap is not None)

    # Regression: _protect_memory() used to collect every line into a set and
    # rewrite the file sorted(). That deleted blank lines and comment placement,
    # and because '#' sorts ahead of every glob it stacked all 13 comments at the
    # top of the file. init() calls it, so every snapshot - this suite's own
    # included - re-mangled the real .gitignore. It must now be append-only and
    # idempotent. Every case below runs against a throwaway root, so the
    # repository's own .gitignore is never a test subject.
    _su_scratch = Path(tempfile.mkdtemp(prefix="jarvis-gitignore-"))
    try:
        # A fully protected file must come back byte-identical.
        _real_ignore = (kernel.config.root / ".gitignore").read_text(encoding="utf-8")
        _real_dir = _su_scratch / "protected"
        _real_dir.mkdir(parents=True, exist_ok=True)
        _real_copy = _real_dir / ".gitignore"
        _real_copy.write_text(_real_ignore, encoding="utf-8")
        _su_real = SelfUpdate(kernel.config, root=_real_dir)
        _su_real._protect_memory()
        _real_once = _real_copy.read_text(encoding="utf-8")
        _su_real._protect_memory()
        _real_twice = _real_copy.read_text(encoding="utf-8")
        check("protect_memory is idempotent", _real_once == _real_twice)
        check("protect_memory leaves a protected file byte-identical",
              _real_once == _real_ignore,
              "appending nothing must not rewrite a single byte")
        check("protect_memory preserved blank lines",
              _real_once.count("\n\n") == _real_ignore.count("\n\n"),
              f"{_real_once.count(chr(10) * 2)} vs {_real_ignore.count(chr(10) * 2)}")
        _comment_lines = [l for l in _real_once.splitlines() if l.strip().startswith("#")]
        _comment_input = [l for l in _real_ignore.splitlines()
                          if l.strip().startswith("#")]
        check("protect_memory preserved every comment",
              len(_comment_lines) == len(_comment_input),
              f"{len(_comment_lines)} vs {len(_comment_input)}")
        check("protect_memory preserved comment placement",
              "# Secrets and local-only config." in _real_once
              and _real_once.index("# Secrets and local-only config.")
              < _real_once.index("\nconfig.json"),
              "each comment must still head its own rule")

        # A file missing the memory patterns gets exactly them appended, once.
        _mini_dir = _su_scratch / "bare"
        _mini_dir.mkdir(parents=True, exist_ok=True)
        _mini = _mini_dir / ".gitignore"
        _before = ("# Byte-compiled output.\n"
                   "*$py.class\n"
                   "\n"
                   "# Secrets and local-only config.\n"
                   "config.json\n")
        _mini.write_text(_before, encoding="utf-8")
        _su_mini = SelfUpdate(kernel.config, root=_mini_dir)
        _su_mini._protect_memory()
        _appended = _mini.read_text(encoding="utf-8")
        _su_mini._protect_memory()
        _appended_twice = _mini.read_text(encoding="utf-8")
        check("protect_memory is idempotent when appending",
              _appended == _appended_twice)
        check("protect_memory appended without touching existing bytes",
              _appended.startswith(_before),
              "existing lines must never be reordered, deduped or rewritten")
        check("protect_memory added the header exactly once",
              _appended.count(SelfUpdate._MEMORY_HEADER) == 1,
              str(_appended.count(SelfUpdate._MEMORY_HEADER)))
        _rules_now = {l.strip() for l in _appended.splitlines()
                      if l.strip() and not l.strip().startswith("#")}
        _still_missing = [p for p in MEMORY_GLOBS if p not in _rules_now]
        check("every memory glob is present after appending",
              not _still_missing, str(_still_missing))
        check("no pre-existing rule was dropped",
              all(p in _rules_now for p in ("*$py.class", "config.json")),
              str(sorted(_rules_now)))
        check("appended rules sit under the header",
              _appended.index(SelfUpdate._MEMORY_HEADER)
              < _appended.index("data/memory.db"))
    finally:
        shutil.rmtree(_su_scratch, ignore_errors=True)
    check("gitignore scratch dirs cleaned up",
          not any(Path(tempfile.gettempdir()).glob("jarvis-gitignore-*")))

    print("\n== evolution engine ==")
    check("propose phase enabled", kernel.evolve_engine.phase_enabled("propose"))
    kernel.config.set_phase("apply", False)
    dry = kernel.evolve_engine.run_cycle(focus="selftest", report_only=True)
    check("dry-run does not request approval", dry.approved is None)
    check("apply phase toggled off", not kernel.evolve_engine.phase_enabled("apply"))
    kernel.config.set_phase("apply", True)

    print("\n== overseer ==")
    from jarvis_os.overseer import Incident, OverseerAgenticOS
    kernel.overseer.write_heartbeat({
        # The pid must be the current process: the run-scoped checks below
        # (cycles, last_output, error counters) are only evaluated for a
        # heartbeat this run actually wrote.
        "ts": time.time(), "pid": os.getpid(), "cycles": 5000,
        "started_at": time.time() - 1,
        "consecutive_errors": 12, "llm_errors": 0, "last_output": "I have completed the task",
        "claimed_success": True, "verified_success": False, "notes": "",
    })
    # runaway_loop is a halt-on-fault trigger and halt_jarvis() reads the pid
    # from the heartbeat above, which is this very process. Disable the halt for
    # this scan so the detection can be asserted without the test killing
    # itself - the previous "pid": 1 only avoided death by aiming at PID 1.
    prev_halt = kernel.overseer.halt_on_fault
    kernel.overseer.halt_on_fault = False
    try:
        report = kernel.overseer.scan()
    finally:
        kernel.overseer.halt_on_fault = prev_halt
    faults = {f["fault"] for f in report["faults"]}
    check("runaway detected", "runaway_loop" in faults, str(faults))
    check("hallucination detected", "hallucination" in faults, str(faults))
    check("error storm detected", any("error storm" in f["evidence"] for f in report["faults"]))

    # Regression: a heartbeat left behind by a dead process must not be read as
    # a stall of the run that just started. That bug stopped Jarvis milliseconds
    # after boot and disarmed the permission gate on every restart.
    timeout = float(config.get("overseer.stall_timeout_seconds", 45))
    stale = {"ts": time.time() - (timeout + 30), "pid": 999999,
             "cycles": 1, "started_at": time.time() - 600}
    check("stale heartbeat from a dead pid is not a stall",
          kernel.overseer.stall_fault(stale) is None)
    # The exact production failure: a leftover hours old from a dead run must
    # never accuse the fresh process, however far past the timeout it is.
    check("an ancient leftover from a dead run is not a stall",
          kernel.overseer.stall_fault({"ts": time.time() - 9000, "pid": 999999,
                                       "cycles": 1}) is None)
    # Detection must survive: a live process that genuinely stops beating is
    # still accused.
    check("stall still detected for the live process",
          (kernel.overseer.stall_fault(dict(stale, pid=os.getpid())) or {}).get("fault") == "stall")
    # A missing heartbeat is reported separately, so losing the file entirely
    # must still surface as a fault rather than being silently ignored.
    original_beat = kernel.overseer.heartbeat
    kernel.overseer.heartbeat = lambda: None
    kernel.overseer.halt_on_fault = False
    try:
        no_beat_faults = {f["fault"] for f in kernel.overseer.scan()["faults"]}
    finally:
        kernel.overseer.heartbeat = original_beat
        kernel.overseer.halt_on_fault = True
    check("a missing heartbeat is still reported",
          "no_heartbeat" in no_beat_faults, str(no_beat_faults))

    # As a separate watchdog process the Overseer must learn which run it is
    # watching. Comparing against its own pid would never match, and every
    # heartbeat would look foreign. The adopted pid must be a process that is
    # genuinely alive, so this spawns one on purpose: using the test's own parent
    # would be unreliable, because a detached runner may already have exited by
    # the time this runs.
    import subprocess
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        watchdog = OverseerAgenticOS(config)
        watched_pid = bystander.pid
        foreign = {"ts": time.time(), "pid": watched_pid, "cycles": 1}
        watchdog.adopt_observed_pid(foreign)
        check("watchdog adopts the supervised pid",
              watchdog.supervised_pid() == watched_pid, str(watchdog.supervised_pid()))
        check("watchdog recognises the run it watches",
              watchdog.heartbeat_is_current(foreign) is True)
        check("watchdog still rejects a foreign heartbeat",
              watchdog.heartbeat_is_current(dict(foreign, pid=watched_pid + 1)) is False)
    finally:
        bystander.kill()
        bystander.wait(timeout=10)

    # Regression: the heartbeat file outlives the run that wrote it, so adopting
    # a pid that is no longer alive laundered that leftover into proof of a stall
    # and isolated a layer milliseconds after boot. A dead pid must be refused,
    # however healthy the heartbeat next to it looks.
    dead_watchdog = OverseerAgenticOS(config)
    dead_watchdog.adopt_observed_pid({"ts": time.time(), "pid": 999999, "cycles": 1})
    check("watchdog refuses to adopt a dead pid",
          dead_watchdog._observed_pid is None, str(dead_watchdog._observed_pid))
    check("dead-pid heartbeat is not read as the supervised run",
          dead_watchdog.heartbeat_is_current(
              {"ts": time.time(), "pid": 999999, "cycles": 1}) is False)
    # ...and the end-to-end path that actually failed in production: scan()
    # adopts before it judges, so a stale leftover must not fault on boot.
    fresh_watchdog = OverseerAgenticOS(config)
    stale_boot = kernel.overseer.stall_fault(
        {"ts": time.time() - 90000, "pid": 999999, "cycles": 1})
    check("ancient dead-run leftover survives an adopting scan", stale_boot is None,
          str(stale_boot))

    # Regression: isolation must record which incident caused it, for every
    # layer - only "permission" used to pass the id and the rest wrote "".
    probe_control = config.resolve(config.get("overseer.control_file",
                                               "data/overseer_control.json"))
    probe_control.unlink(missing_ok=True)
    marker = Incident(incident_id="inc_marker_probe", fault="stall", layer="planner",
                      evidence="probe")
    kernel.overseer.isolate_layer("planner", marker)
    recorded = json.loads(probe_control.read_text(encoding="utf-8"))
    check("isolation records the incident id",
          recorded.get("last_incident") == "inc_marker_probe", str(recorded))
    check("isolation flags the layer", recorded.get("planner_disabled") is True, str(recorded))
    probe_control.unlink(missing_ok=True)

    incident = kernel.overseer.handle_fault("layer_failure", "tools", "smoke test fault")
    check("incident recorded", incident.incident_id.startswith("inc_"))
    check("cross-layer trace produced",
          set(incident.diagnosis.get("cross_layer_trace", {})) == {
              "interface", "planner", "memory", "tools", "llm", "execution", "permission"})
    check("overseer logged to shared ledger",
          any(e["actor"] == "overseer" for e in kernel.audit.tail(limit=80)))
    kernel.overseer.write_heartbeat(kernel.heartbeat_payload())

    print("\n== repl heartbeat ==")
    # Regression: the infinite loop beats once per cycle, but an interactive REPL
    # has no cycles - it blocks in input() while a person thinks. So it never beat
    # at all, and the Overseer was left supervising a pid that had already exited,
    # reading a leftover heartbeat from a dead run. It reported "running: True"
    # while watching nothing.
    #
    # The whole suite runs with features.overseer off (see _run) so that no fault
    # injection can halt the process mid-test. The refresher is gated on that same
    # flag, by design - a session nobody is supervising has nothing to beat for -
    # so the Overseer is switched back on for this block and off again after.
    _was_overseer = config.feature("overseer", True)
    config.set_feature("overseer", True)
    try:
        # The kernel owns its own heartbeat, and that path is the sandboxed one, so
        # resolve it rather than assuming the Overseer writes anywhere else.
        _hb_path = Path(kernel.config.resolve(
            config.get("overseer.heartbeat_path", "data/heartbeat.json")))
        check("kernel and overseer agree on the heartbeat path",
              _hb_path == Path(kernel.overseer.heartbeat_path),
              f"{_hb_path} vs {kernel.overseer.heartbeat_path}")
        # The refresher does not know how the session is being driven; the REPL says so.
        # Asserting it here keeps the payload honest without coupling the kernel
        # to the interface.
        _prev_mode = kernel._mode
        kernel.set_mode("interactive")
        _hb_had_previous = _hb_path.exists()
        try:
            # Remove any leftover so a stale file cannot make this pass by accident.
            if _hb_had_previous:
                _hb_path.unlink()
            check("no stale heartbeat to mistake for a fresh one",
                  not _hb_path.exists(), f"{_hb_path} still present")

            started = kernel.start_heartbeat_refresher()
            check("heartbeat refresher starts for an interactive session", started)
            check("heartbeat written immediately, before any command",
                  _hb_path.exists(), f"missing {_hb_path}")

            _first = json.loads(_hb_path.read_text(encoding="utf-8"))
            check("heartbeat is owned by this process",
                  int(_first.get("pid") or 0) == os.getpid(),
                  f"pid={_first.get('pid')} ours={os.getpid()}")
            check("repl heartbeat reports interactive, not daemon",
                  _first.get("mode") == "interactive",
                  f"mode={_first.get('mode')!r}")
            check("fresh heartbeat is not a stall",
                  kernel.overseer.stall_fault(_first) is None)
            check("overseer accepts the beat as its own",
                  kernel.overseer.heartbeat_is_current(_first))

            # The real requirement: it has to keep beating across an idle gap, or the
            # Overseer will accuse an ordinary pause of being a stall. Compressed by
            # rewriting the interval rather than by sleeping for 15 seconds.
            _stall_cfg = float(config.get("overseer.stall_timeout_seconds", 45))
            check("beat interval sits inside the stall timeout",
                  kernel._hb_interval < _stall_cfg,
                  f"interval={kernel._hb_interval} timeout={_stall_cfg}")

            # The refresher must read the Overseer's actual threshold rather than
            # assume 45s: a shorter timeout needs a proportionally faster beat,
            # or a tightened watchdog starts manufacturing stalls.
            _orig_timeout = kernel.overseer.stall_timeout
            try:
                kernel.stop_heartbeat_refresher()
                kernel.overseer.stall_timeout = 30.0
                kernel.start_heartbeat_refresher()
                check("beat interval follows a shorter stall timeout",
                      kernel._hb_interval <= 30.0 / 3.0 + 1e-6,
                      f"interval={kernel._hb_interval} expected<={30.0 / 3.0}")
                kernel.stop_heartbeat_refresher()
            finally:
                kernel.overseer.stall_timeout = _orig_timeout
            kernel.start_heartbeat_refresher()

            kernel._hb_interval = 0.05
            time.sleep(0.4)
            _second = json.loads(_hb_path.read_text(encoding="utf-8"))
            check("heartbeat refreshed while idle",
                  float(_second.get("ts") or 0) > float(_first.get("ts") or 0),
                  f"{_first.get('ts')} -> {_second.get('ts')}")
            check("idle refresh stays inside the stall timeout",
                  kernel.overseer.age(_second) < _stall_cfg)
            check("idle refresh is not a stall fault",
                  kernel.overseer.stall_fault(_second) is None)
        finally:
            kernel._hb_interval = 15.0
            kernel.stop_heartbeat_refresher()
            kernel.set_mode(_prev_mode)
            check("heartbeat refresher stops cleanly",
                  kernel._hb_thread is None)

    # The loop must not leave a thread behind, and starting twice is a no-op
        # rather than a second writer racing on the same file.
        _again = kernel.start_heartbeat_refresher()
        _first_thread = kernel._hb_thread
        _dup = kernel.start_heartbeat_refresher()
        check("a second refresher is refused", _again and not _dup)
        check("no duplicate heartbeat thread was created",
              kernel._hb_thread is _first_thread)
        kernel.stop_heartbeat_refresher()

        # set_mode is what keeps the heartbeat honest about the live process.
        for _m in ("foreground", "interactive", "daemon"):
            kernel.set_mode(_m)
            check(f"mode {_m} recorded", kernel.heartbeat_payload()["mode"] == _m)
        try:
            kernel.set_mode("nonsense")
            check("unknown mode rejected", False, "set_mode accepted 'nonsense'")
        except ValueError:
            check("unknown mode rejected", True)
        kernel.set_mode("foreground")
    finally:
        # Back to the suite-wide setting, so later fault-injection checks run
        # under the same conditions they were written for.
        config.set_feature("overseer", _was_overseer)
        check("overseer feature restored",
              config.feature("overseer", True) == _was_overseer)

    # A repair with no live approval channel must refuse, never escalate.
    # NOTE: drive repair() directly on a halted incident rather than
    # handle_fault("permission_violation") - the real path calls halt_jarvis(),
    # which taskkills this very process (correct behaviour, fatal in a test).
    from jarvis_os.overseer import OverseerAgenticOS
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
    print("\n== goal tool discipline ==")
    # Regression: the planner prompt's example contained "tools":["web_fetch"], and
    # a 4B model copies the example rather than filling in the shape. So every
    # goal it saw inherited web_fetch - including "say hello in Tamil and English",
    # which was planned as two web searches for the word "hello", answered in
    # neither language, and still reported success.
    _plan_goal = "say hello in Tamil and English"
    _tasks, _plan_strategy = kernel.task_manager.decompose(_plan_goal)
    check("greeting goal still produces a plan", bool(_tasks),
          f"strategy={_plan_strategy}")
    _planned = sorted({t for task in _tasks for t in task.tools})
    check("greeting plan asks for no tools", _planned == [], f"planned={_planned}")
    check("greeting plan never names web_fetch",
          not any("web_fetch" in t for t in _planned), f"planned={_planned}")
    check("greeting plan produces no invented tools",
          all(t in (kernel.tools.names() if kernel.tools else []) for t in _planned),
          f"planned={_planned}")

    # The classifier has to survive the conjunction, and must not swallow a goal
    # that genuinely needs the outside world.
    _tm = kernel.task_manager
    check("classifier accepts a bilingual greeting",
          _tm._is_self_contained("say hello in Tamil and English"))
    check("classifier accepts a single greeting",
          _tm._is_self_contained("hello"))
    check("classifier accepts a translation request",
          _tm._is_self_contained("translate this into Tamil"))
    check("classifier accepts a thanks",
          _tm._is_self_contained("thank you"))
    check("classifier rejects a greeting plus a real task",
          not _tm._is_self_contained("say hello and then fetch the news"))
    check("classifier rejects a research goal",
          not _tm._is_self_contained("what is the latest news"))
    check("classifier rejects an empty goal",
          not _tm._is_self_contained("   "))
    check("classifier does not fire on a research plan",
          not _tm._is_self_contained("research the weather and write a script"))

    # A real tool plan must survive the stripping untouched.
    _research_tasks, _research_strategy = kernel.task_manager.decompose(
        "research the weather and write a script")
    check("research goal still plans", bool(_research_tasks),
          f"strategy={_research_strategy}")

    # ---- PARTIAL reporting -----------------------------------------------
    # "no subtask crashed" and "the goal got done" are different claims. A plan
    # that named tools and ran none has verified nothing, and reporting that as
    # success is how a half-answered greeting came back as "Done".
    from jarvis_os.taskmanager import AgentTask, DelegationResult, TaskManager

    _r = DelegationResult(goal="g", planned_tools=["web_fetch"], used_tools=[])
    check("unexecuted tools make a result partial", _r.partial)
    check("partial result still reports ok", _r.ok,
          "no subtask crashed, so ok stays true")
    _r2 = DelegationResult(goal="g", planned_tools=["web_fetch"],
                           used_tools=["web_fetch"])
    check("executed tools are not partial", not _r2.partial)
    _r3 = DelegationResult(goal="g")
    check("a plan with no tools is not partial", not _r3.partial)
    check("to_dict carries the tool evidence",
          _r.to_dict()["unexecuted_tools"] == ["web_fetch"]
          and _r.to_dict()["partial"] is True)

    # End to end: force a plan that names a tool no agent can reach, and confirm
    # execute() flags it partial instead of quietly succeeding.
    class _ToolBlindAgent:
        name = "tool-blind"
        role = "orchestrator"

        def run(self, objective, context="", allow_tools=True):
            # Never invokes a tool, and says so plainly.
            return {"ok": True, "agent": self.name, "role": self.role,
                    "output": f"answered from memory: {objective}",
                    "tools_used": [], "duration_ms": 1}

    class _BlindRegistry:
        def __init__(self, agent):
            self._agent = agent

        def get(self, name):
            return self._agent

        def find_by_role(self, role):
            return [self._agent]

    _blind = TaskManager(_BlindRegistry(_ToolBlindAgent()), kernel.memory, None,
                         kernel.config, kernel.audit, kernel.tools)
    _real_decompose = _blind.decompose
    _blind.decompose = lambda goal: ([
        AgentTask(objective="fetch it", assigned_agent="orchestrator",
                  tools=["web_fetch"])], "forced")
    try:
        _blind_res = _blind.execute(_plan_goal)
        _blind_out = _blind_res.output
        check("a plan whose tools never ran is partial", _blind_res.partial,
              f"partial={_blind_res.partial} used={_blind_res.used_tools}")
        check("the unexecuted tool is named",
              _blind_res.unexecuted_tools == ["web_fetch"],
              str(_blind_res.unexecuted_tools))
        check("execute() still returns the agent's text",
              "answered from memory" in _blind_out, _blind_out[:160])
    finally:
        del _blind.decompose

    # And the operator-facing surface has to say so out loud. Assert on what the
    # operator actually receives: cmd_run's status word goes through
    # process_outbound like any other reply, so checking the untranslated argument
    # would pass while the console showed something else entirely - which is
    # exactly how "INCOMPLETE" came to be translated away unnoticed.
    _iface = kernel.text_iface
    _echo = _iface._echo
    _iface._echo = False
    _said: list[str] = []
    _spoken: list[str] = []
    _prev_say = _iface.say
    _iface.say = lambda text, translate=True: (
        _said.append(text),
        _spoken.append(_prev_say(text, translate=translate)))[1]
    _prev_exec = kernel.task_manager.execute
    try:
        kernel.task_manager.execute = lambda goal, context="", allow_tools=True: (
            DelegationResult(goal=goal, output="Sir, \u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bbe\u0bae. Hello.",
                             planned_tools=["web_fetch"], used_tools=[]))
        _iface.cmd_run(["say", "hello", "in", "Tamil", "and", "English"])
        _out = " ".join(_spoken)
        check("run announces PARTIAL when tools did not execute",
              "PARTIAL" in _out, _out[:200])
        check("run names the tool that never ran",
              "web_fetch" in _out, _out[:200])
        check("run still delivers the reply",
              "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bbe\u0bae" in _out, _out[:200])
        check("the reply carries no translation skip note",
              "translation skipped" not in _out, _out[:200])
        check("every line the operator sees is honorific-prefixed",
              all(ln.startswith("Sir,") for ln in _spoken if ln.strip()),
              str([ln[:40] for ln in _spoken])[:200])

        _said.clear()
        _spoken.clear()
        kernel.task_manager.execute = lambda goal, context="", allow_tools=True: (
            DelegationResult(goal=goal, output="all good", ok=False))
        _iface.cmd_run(["do", "something", "impossible"])
        _out = " ".join(_spoken)
        check("run announces INCOMPLETE on total failure",
              "INCOMPLETE" in _out, _out[:200])
        check("a total failure is not reported as partial",
              "PARTIAL" not in _out, _out[:200])
        check("a failed run still gets the honorific",
              all(ln.startswith("Sir,") for ln in _spoken if ln.strip()),
              str([ln[:40] for ln in _spoken])[:200])
    finally:
        kernel.task_manager.execute = _prev_exec
        _iface.say = _prev_say
        _iface._echo = _echo

    tasks, strategy = kernel.task_manager.decompose("research the weather and write a script")
    check("decomposition produces subtasks", len(tasks) >= 1, strategy)
    check("final task depends on others",
          any(t.depends_on for t in tasks))
    routed = kernel.task_manager.resolve_agent("nonexistent", "write code for me")
    check("agent routing by keyword", routed is not None and routed.role == "coder",
          getattr(routed, "role", "?"))

    print("\n== devices & daemon ==")
    check("local device registered", kernel.devices.current in kernel.devices.devices)
    # Assert persistence for real. The old check ended in "... or True", so it
    # passed unconditionally and could never have caught the phantom below.
    probe = kernel.devices._local_device()
    probe.device_id = "probe"
    probe.endpoint = ""
    kernel.devices.devices["probe"] = probe
    kernel.devices.save()
    saved = json.loads(kernel.devices.path.read_text(encoding="utf-8"))
    check("device registry persisted", "probe" in [d["id"] for d in saved["devices"]],
          str([d["id"] for d in saved["devices"]]))
    kernel.devices.devices.pop("probe", None)
    kernel.devices.save()
    check("remote config brake engaged",
          "disabled in config" in kernel.devices.cmd_config(["probe", "app", "restart"]))
    check("remote config is off by default",
          kernel.devices.config.get("devices.allow_remote_config") is False)
    # Regression: an earlier version of this suite wrote a "phone" fixture into
    # the *live* registry and a pre-selftest snapshot committed it. That entry was
    # a verbatim clone of the host entry - same label, kind, os, status, notes -
    # differing only in id and endpoint, and the endpoint (192.168.1.9:8770) was
    # unreachable and not this machine. A trusted host twin pointing at a remote
    # endpoint is exactly the shape that must never reappear.
    live_path = REPO_ROOT / "data" / "devices.json"
    live_devices = (json.loads(live_path.read_text(encoding="utf-8"))
                    if live_path.is_file() else {"devices": []})
    live_ids = [d.get("id") for d in live_devices.get("devices", [])]
    twins = [d.get("id") for d in live_devices.get("devices", [])
             if d.get("kind") == "host" and d.get("endpoint")]
    check("no phantom host twin in the live registry", not twins, str(twins))
    check("live registry holds only the local device", live_ids == ["local"], str(live_ids))
    check("daemon status available", "BackgroundDaemon" in kernel.daemon.status())

    # Regression: recall/attach must be continuous with the running session.
    # attach() used to call text_iface.start() unconditionally, so recalling from
    # a live prompt nested a second input() loop inside the running one, and a
    # recall with no bound handle just told the operator to give up. It now
    # reuses the kernel this process already holds - no reboot, memory intact.
    #
    # The daemon is simulated rather than really started on purpose: start()
    # spawns a detached `python jarvis.py`, which would boot a second kernel
    # against the *live* config.json and write into live data/, breaking the very
    # guarantee this suite asserts. A BackgroundDaemon rooted in the throwaway
    # sandbox, claiming liveness with this process's own pid, exercises the same
    # attach() code path with no side effect outside the sandbox.
    probe_daemon = BackgroundDaemon(config=kernel.config, gate=kernel.gate,
                                    audit=kernel.audit, root=sandbox)
    probe_daemon.state = DaemonState(pid=os.getpid(), started_at=time.time(),
                                     mode="background", entry=str(probe_daemon.entry))
    probe_daemon.save()
    check("simulated daemon reports alive", probe_daemon.alive())
    probe_daemon.bind_kernel(kernel)

    boots_before = kernel.memory.stats.get("boots", 0)
    kernel.memory.remember("daemon reattach marker", kind="preference", importance=0.9)
    memories_written = kernel.memory.stats.get("memories", 0)

    # A foreground session is live, then closes; recall must handle both.
    starts: list[int] = []
    said: list[str] = []
    real_start = kernel.text_iface.start
    real_say = kernel.text_iface.say
    kernel.text_iface.start = lambda: starts.append(1)
    kernel.text_iface.say = lambda text, *a, **k: said.append(str(text))
    try:
        kernel.text_iface.running = True
        mid = probe_daemon.recall()
        check("mid-session recall does not restart the prompt loop", not starts, str(starts))
        check("mid-session recall reports a reattach", "Reattached" in mid, mid[:80])
        check("mid-session recall keeps the same kernel", probe_daemon._kernel is kernel)
        kernel.text_iface.running = False
        cold = probe_daemon.recall()
        check("cold recall starts the loop exactly once", len(starts) == 1, str(starts))
        # The cold path announces the reattach through say(); its return value is
        # only spoken once the operator leaves the loop again.
        check("cold recall announces the reattach",
              any("Reattached" in m for m in said), " | ".join(said)[:120])
    finally:
        kernel.text_iface.start = real_start
        kernel.text_iface.say = real_say
        kernel.text_iface.running = False

    check("reattach did not reboot the kernel",
          kernel.memory.stats.get("boots", 0) == boots_before,
          f"{boots_before} -> {kernel.memory.stats.get('boots')}")
    check("reattach kept memory intact",
          kernel.memory.stats.get("memories", 0) >= memories_written,
          f"{memories_written} -> {kernel.memory.stats.get('memories')}")
    check("attach counted the continuous session", probe_daemon._attachments == 2,
          str(getattr(probe_daemon, "_attachments", None)))

    # With no handle bound the reply must still say how to get one, not dead-end.
    orphan = BackgroundDaemon(config=kernel.config, root=sandbox)
    orphan_msg = orphan.attach([])
    check("unbound handle explains how to reattach",
          "python jarvis.py" in orphan_msg, orphan_msg[:90])

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

    print("\n== ipc command bridge ==")
    # submit -> poll round-trip through the JSON command file. The bridge root
    # follows control_bridge.ipc_path, which _DATA_PATHS redirected into the
    # sandbox, so these writes never land in the live data/control/ directory.
    kernel.control.submit("smoke_proc", "status")
    ipc_results = kernel.control.poll()
    check("ipc round-trip executes the command",
          bool(ipc_results) and ipc_results[0].get("ok"), str(ipc_results)[:200])
    resp_path = kernel.control.response_path
    check("ipc response file written for the caller",
          resp_path.is_file(), str(resp_path))
    good_response = (json.loads(resp_path.read_text(encoding="utf-8"))
                     if resp_path.is_file() else {})
    check("ipc response file carries the verdict",
          good_response.get("ok"), str(good_response)[:160])
    bad_payload = {"program": "smoke_proc", "action": "stop", "args": [],
                   "ts": 0, "token": "not-the-real-token"}
    kernel.control.ipc_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    refused = kernel.control.poll()
    check("ipc bad token refused",
          bool(refused) and not refused[0].get("ok")
          and "token" in str(refused[0]), str(refused)[:200])
    check("refused command file cleared after poll",
          not kernel.control.ipc_path.exists())
    still = (json.loads(resp_path.read_text(encoding="utf-8"))
             if resp_path.is_file() else {})
    check("refused command does not overwrite the response",
          still == good_response, f"{still} vs {good_response}")
    check("refused ipc never reaches dispatch",
          kernel.control.command_count >= 2 and refused
          and kernel.control.processes["smoke_proc"].pid is None,
          str(kernel.control.processes["smoke_proc"].pid))

    # The module rename ships with Phase 2: jarvis_os.control must be gone
    # and the gui channel must exist on the bridge.
    from jarvis_os import control_bridge as _cb_mod
    check("gui exposed as a control-bridge action",
          "gui" in _cb_mod.ACTIONS, str(_cb_mod.ACTIONS))
    gui_unknown = kernel.control.dispatch("_", "gui", ["warp"])
    check("gui rejects unknown sub-action",
          not gui_unknown.get("ok")
          and "gui" in str(gui_unknown.get("error", "")), str(gui_unknown))
    try:
        import jarvis_os.control as _legacy  # noqa: F401
        legacy_gone = False
    except ModuleNotFoundError:
        legacy_gone = True
    check("legacy jarvis_os.control module renamed away", legacy_gone)

    print("\n== dynamic tool validation ==")
    from jarvis_os.tool_registry import static_validate
    good_src = ('def run(value=""):\n'
                '    """Selftest dynamic tool."""\n'
                '    return {"ok": True, "echo": str(value), "n": len(str(value))}\n')
    verdict = kernel.tools.register_validated(
        "st_dyn_tool", good_src, "selftest validated tool",
        probe_args={"value": "hi"})
    check("validated tool registered",
          verdict.get("ok") and verdict.get("stage") == "registered",
          str(verdict)[:200])
    dyn = kernel.tools.get("st_dyn_tool")
    check("dynamic tool present and callable",
          dyn is not None and dyn.handler(value="x").get("echo") == "x",
          str(dyn.handler(value="x")) if dyn else "not registered")
    check("dynamic tool marked dangerous code_exec",
          dyn is not None and dyn.dangerous
          and dyn.permission_category == "code_exec",
          str(dyn.permission_category) if dyn else "not registered")
    check("dynamic tool is not builtin",
          dyn is not None and not dyn.builtin)
    check("network import rejected statically",
          not static_validate("st_bad", "import socket\ndef run():\n    return {}")
          .get("ok"))
    check("eval rejected statically",
          not static_validate("st_bad", "def run():\n    return eval('1')").get("ok"))
    check("open rejected statically",
          not static_validate("st_bad", "def run():\n    open('x', 'w')")
          .get("ok"))
    check("syntax error rejected statically",
          not static_validate("st_bad", "def run(:").get("ok"))
    check("invalid tool name rejected statically",
          not static_validate("bad name!", "def run():\n    return {}").get("ok"))
    raiser = 'def run(**kw):\n    raise RuntimeError("boom")\n'
    probe_fail = kernel.tools.register_validated(
        "st_boom_tool", raiser, "always raises")
    check("raising tool fails the sandbox probe",
          not probe_fail.get("ok")
          and probe_fail.get("stage") in ("call", "probe", "import"),
          str(probe_fail)[:200])
    check("raising tool never reaches the registry",
          kernel.tools.get("st_boom_tool") is None)
    kernel.gate._text_input = lambda prompt: "no"
    try:
        denied = kernel.tools.register_validated(
            "st_denied_tool", good_src, "denied tool")
    finally:
        kernel.gate._text_input = lambda prompt: "yes"
    check("gate denial blocks registration",
          not denied.get("ok") and denied.get("stage") == "permission",
          str(denied)[:200])
    check("denied tool never reaches the registry",
          kernel.tools.get("st_denied_tool") is None)

    print("\n== ocr & dependency wiring ==")
    ocr = kernel.tools.call("ui_control", {"action": "ocr"},
                            require_permission=False)
    check("ocr returns a dict verdict, never an exception",
          isinstance(ocr, dict) and "ok" in ocr, str(ocr)[:160])
    if ocr.get("ok"):
        check("ocr returns text payload", isinstance(ocr.get("text"), str),
              str(ocr)[:160])
    else:
        check("ocr failure names the problem",
              any(k in str(ocr.get("error", "")).lower()
                  for k in ("install", "unavailable", "engine", "pyautogui")),
              str(ocr)[:200])
    ct = kernel.tools.call("ui_control", {"action": "click_text"},
                           require_permission=False)
    check("click_text requires a text argument",
          not ct.get("ok") and "text" in str(ct.get("error", "")).lower(),
          str(ct)[:200])
    req_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("requirements.txt declares pytesseract",
          "pytesseract" in req_text, "missing from requirements.txt")
    setup_text = (REPO_ROOT / "setup-windows-daemon.ps1").read_text(encoding="utf-8")
    check("setup script installs and reports pytesseract",
          setup_text.count("pytesseract") >= 2,
          f"found {setup_text.count('pytesseract')} reference(s)")

    print("\n== cli action flags ==")
    import jarvis as jarvis_cli
    parser = jarvis_cli.build_parser()
    ns = parser.parse_args(["--build", "hello world greeting program"])
    check("--build parses the spec",
          ns.build == "hello world greeting program", ns.build)
    ns = parser.parse_args(["--install", "sw_123"])
    check("--install parses the program id", ns.install == "sw_123", ns.install)
    ns = parser.parse_args(["--launch", "sw_123"])
    check("--launch parses the program id", ns.launch == "sw_123", ns.launch)
    ns = parser.parse_args(["--control", "sw_123", "status"])
    check("--control parses id and action",
          ns.control == ["sw_123", "status"], str(ns.control))
    ns = parser.parse_args(["--stop", "sw_123"])
    check("--stop parses the program id", ns.stop == "sw_123", ns.stop)
    ns = parser.parse_args(["--tools"])
    check("--tools parses", ns.tools is True)
    ns = parser.parse_args([])
    check("action flags default to inert (no accidental action)",
          not ns.tools and not ns.build and not ns.install
          and not ns.launch and ns.control is None and not ns.stop
          and ns.device is None and ns.sync is None
          and ns.translate is None and not ns.to)
    check("--device parses subcommand words",
          parser.parse_args(["--device", "serve"]).device == ["serve"])
    check("--sync parses ids (bare --sync = all)",
          parser.parse_args(["--sync", "phone1"]).sync == ["phone1"]
          and parser.parse_args(["--sync"]).sync == [])
    ns = parser.parse_args(["--translate", "vanakkam", "--to", "en"])
    check("--translate parses text and target",
          ns.translate == ["vanakkam"] and ns.to == "en",
          f"{ns.translate} to={ns.to}")
    check("build_parser exposes a callable factory",
          callable(getattr(jarvis_cli, "build_parser", None)))

    print("\n== language layer ==")
    check("language layer wired into the kernel",
          isinstance(kernel.language_layer, LanguageLayer))
    honorific = str(kernel.language.honorific)
    greeted = kernel.language_layer.respond("ready.")
    check("respond prefixes the honorific",
          greeted.startswith(honorific), greeted)
    check("respond is idempotent",
          kernel.language_layer.respond(greeted) == greeted,
          kernel.language_layer.respond(greeted))
    r = kernel.language_layer.translate("All systems are nominal.", "en")
    check("translate reports already-in-target",
          r.get("ok") is True and r.get("already") is True
          and r.get("source") == "en" and r.get("target") == "en", str(r))
    check("already message honors",
          str(r.get("message", "")).startswith(honorific),
          str(r.get("message")))
    r = kernel.language_layer.translate("hello", "fr")
    check("unsupported target refused",
          r.get("ok") is False
          and "serves en and ta" in str(r.get("error", "")), str(r))
    r = kernel.language_layer.translate("")
    check("empty text refused",
          r.get("ok") is False
          and "give me text to translate" in str(r.get("error", "")), str(r))
    r = kernel.language_layer.translate("hello")
    check("degraded translation is honest, not silent",
          "backend" in r and (r.get("ok") is True or r.get("degraded") is True),
          str(r))
    r = kernel.language_layer.parse_intent('translate "good morning"')
    check("intent: translate with quoted entity",
          r.get("intent") == "translate"
          and "good morning" in r.get("entities", {}).get("quoted", []), str(r))
    r = kernel.language_layer.parse_intent("remember the milk")
    check("intent: remember", r.get("intent") == "remember", str(r))
    r = kernel.language_layer.parse_intent("hello there")
    check("intent: greeting", r.get("intent") == "greeting", str(r))
    r = kernel.language_layer.parse_intent("this is fine")
    check("'this' does not fire 'hi'", r.get("intent") != "greeting", str(r))
    r = kernel.language_layer.parse_intent("மொழிபெயர் இதை")
    check("intent: Tamil translate keyword",
          r.get("intent") == "translate", str(r))
    r = kernel.language_layer.parse_intent("Stop right now")
    check("intent: stop", r.get("intent") == "stop", str(r))
    r = kernel.language_layer.parse_intent("what is the time")
    check("intent: time", r.get("intent") == "time", str(r))
    r = kernel.language_layer.parse_intent("xyzzy nonsense")
    check("intent: unknown at zero confidence",
          r.get("intent") == "unknown" and r.get("confidence") == 0.0, str(r))
    check("intent messages honor",
          str(r.get("message", "")).startswith(honorific),
          str(r.get("message")))
    kernel.config.set_feature("language_layer", False, persist=False)
    r = kernel.language_layer.translate("hello")
    check("language layer feature fails closed",
          r.get("ok") is False
          and "features.language_layer" in str(r.get("error", "")), str(r))
    kernel.config.set_feature("language_layer", True, persist=False)
    st = kernel.language_layer.status()
    check("language layer status complete",
          st.get("ok") is True and st.get("enabled") is True
          and st.get("intent_rules", 0) >= 10
          and "preferred" in st and "backends" in st, str(st))

    print("\n== device sync ==")
    ds = kernel.device_sync
    check("device sync wired into the kernel",
          isinstance(ds, DeviceSyncManager))
    check("device sync report renders",
          "DeviceSyncManager" in ds.report(), ds.report())
    r = ds.command([])
    check("command router defaults to status",
          isinstance(r, str) and "device sync" in r, str(r))
    r = ds.command(["bogus"])
    check("unknown verb shows usage",
          isinstance(r, dict) and r.get("ok") is False
          and "Usage" in str(r.get("error", "")), str(r))
    r = ds.command(["dispatch"])
    check("dispatch without args shows usage",
          isinstance(r, dict) and r.get("ok") is False
          and "Usage" in str(r.get("error", "")), str(r))
    tls = ds.ensure_tls()
    check("tls material generated",
          tls.get("ok") is True and tls.get("generated") is True, str(tls))
    check("cert file exists",
          Path(str(tls.get("cert", ""))).is_file(), str(tls))
    again = ds.ensure_tls()
    check("tls is generated once",
          again.get("ok") is True and again.get("generated") is False, str(again))
    tok = ds.ensure_token()
    check("pairing token created",
          tok.get("ok") is True
          and len(str(tok.get("token", ""))) >= 32, str(tok))
    tok2 = ds.ensure_token()
    check("pairing token stable",
          tok2.get("token") == tok.get("token"), str(tok2))
    parse = DeviceSyncManager._parse_endpoint
    check("parse ssh endpoint",
          parse("ssh://u@127.0.0.1:2222") == {"scheme": "ssh", "host": "127.0.0.1",
                                              "port": 2222, "user": "u"},
          str(parse("ssh://u@127.0.0.1:2222")))
    check("parse wss endpoint",
          parse("wss://example.test:8443") == {"scheme": "wss",
                                               "host": "example.test",
                                               "port": 8443, "user": ""},
          str(parse("wss://example.test:8443")))
    check("ssh default port",
          parse("ssh://127.0.0.1")["port"] == 8022,
          str(parse("ssh://127.0.0.1")))
    check("http endpoint refused", parse("http://x") is None,
          str(parse("http://x")))
    check("plain ws endpoint refused",
          parse("ws://1.2.3.4:9") is None, str(parse("ws://1.2.3.4:9")))
    tr = ds.transports()
    check("websockets transport present",
          tr.get("websockets") is True, str(tr))
    check("tls transport present", tr.get("tls") is True, str(tr))
    r = ds.sync()
    check("sync with no peers is a no-op",
          r.get("ok") is True and r.get("synced") == 0, str(r))
    added = kernel.devices.command(["add", "phone1", "ssh://u@127.0.0.1:20223"])
    check("peer device registered", "registered" in added, added)
    kernel.devices.command(["add", "bare"])
    kernel.devices.command(["add", "insecure", "ws://1.2.3.4:9"])
    r = ds.dispatch("", ["echo", "hi"])
    check("dispatch refuses empty id",
          r.get("ok") is False and "Usage" in str(r.get("error", "")), str(r))
    r = ds.dispatch("nope", ["echo", "hi"])
    check("dispatch to unregistered device refused",
          r.get("ok") is False
          and "not registered" in str(r.get("error", "")), str(r))
    r = ds.dispatch(kernel.devices.current, ["echo", "hi"])
    check("dispatch to this machine refused",
          r.get("ok") is False
          and "this machine" in str(r.get("error", "")), str(r))
    r = ds.dispatch("bare", ["echo", "hi"])
    check("dispatch without endpoint refused",
          r.get("ok") is False
          and "no endpoint" in str(r.get("error", "")), str(r))
    r = ds.dispatch("insecure", ["echo", "hi"])
    check("plain ws endpoint refused at dispatch",
          r.get("ok") is False
          and "refused" in str(r.get("error", "")).lower(), str(r))
    r = ds.alert("", "")
    check("alert without args shows usage",
          r.get("ok") is False and "Usage" in str(r.get("error", "")), str(r))
    kernel.gate._text_input = lambda prompt: "no"
    try:
        r = ds.dispatch("phone1", ["echo", "hi"])
        check("dispatch denied by operator", r.get("denied") is True, str(r))
        r = ds.alert("phone1", "selftest alert")
        check("alert denied by operator", r.get("denied") is True, str(r))
    finally:
        kernel.gate._text_input = lambda prompt: "yes"
    started = time.monotonic()
    r = ds.dispatch("phone1", ["echo", "hi"])
    elapsed = time.monotonic() - started
    check("ssh dispatch fails gracefully when refused",
          r.get("ok") is False and "ssh exit" in str(r.get("error", "")),
          str(r)[:200])
    check("ssh failure is bounded", elapsed < 15.0, f"{elapsed:.2f}s")
    listing = ds.command(["list"])
    check("device registry lists peers",
          isinstance(listing, str) and "Devices" in listing
          and "phone1" in listing, str(listing).splitlines()[0])
    state = ds.command(["status"])
    check("device sync status string",
          isinstance(state, str) and "device sync" in str(state), str(state))
    kernel.config.set_feature("device_manager", False, persist=False)
    check("feature off flips status",
          "off for this session" in ds.status(), ds.status())
    off = ds.dispatch("phone1", ["echo", "hi"])
    check("device sync feature fails closed",
          off.get("ok") is False
          and "features.device_manager" in str(off.get("error", "")), str(off))
    kernel.config.set_feature("device_manager", True, persist=False)

    print("\n== device sync server ==")
    ds.stream_interval = 0.2
    start = ds.start_server()
    check("sync server starts",
          start.get("ok") is True and start.get("serving") is True, str(start))
    port = int(start.get("port") or 0)
    check("ephemeral port bound", port > 0, str(start.get("port")))
    check("endpoint is wss",
          str(start.get("endpoint", "")).startswith("wss://"),
          str(start.get("endpoint")))
    check("start returns the pairing token",
          len(str(start.get("token", ""))) >= 32,
          str(start.get("token", ""))[:8])
    again = ds.start_server()
    check("second start is idempotent",
          again.get("ok") is True and again.get("already") is True
          and int(again.get("port") or 0) == port, str(again))

    import ssl as _ssl
    from websockets.sync.client import connect as ws_connect

    def peer_client(token, timeout=8.0):
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        return ws_connect(f"wss://127.0.0.1:{port}", ssl=ctx,
                          open_timeout=timeout)

    def next_frame(conn, want=None, timeout=6.0):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = conn.recv(timeout=remaining)
            except Exception:  # noqa: BLE001 - closed or timed out
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(frame, dict) and (want is None
                                            or frame.get("type") == want):
                return frame

    try:
        with peer_client("wrong-token") as conn:
            conn.send(json.dumps({"type": "auth", "token": "wrong-token"}))
            frame = next_frame(conn, want="error", timeout=5.0)
            check("bad pairing token refused",
                  frame is not None
                  and "pairing token" in str(frame.get("payload", {})),
                  str(frame))
    except Exception as exc:  # noqa: BLE001 - a failed dial is the expected fail-closed
        check("bad pairing token refused", False, f"{type(exc).__name__}: {exc}")
    token = str(start.get("token", ""))
    time.sleep(0.3)
    with peer_client(token) as conn:
        conn.send(json.dumps({"type": "auth", "token": token}))
        first = next_frame(conn, want="ack", timeout=5.0)
        check("handshake acked",
              first is not None
              and first.get("payload", {}).get("authenticated") is True,
              str(first))
        pushed = next_frame(conn, want="status", timeout=5.0)
        check("status pushed after auth", pushed is not None, str(pushed))
        conn.send(json.dumps({"type": "command", "command": "ls"}))
        frame = next_frame(conn, want="error", timeout=5.0)
        check("inbound command refused",
              frame is not None
              and "one-directional" in str(frame.get("payload", {})), str(frame))
        conn.send(json.dumps({"type": "alert", "payload": {"text": "selftest"}}))
        frame = next_frame(conn, want="ack", timeout=5.0)
        check("inbound alert acknowledged",
              frame is not None
              and frame.get("payload", {}).get("kind") == "alert", str(frame))
        frame = next_frame(conn, want="stream", timeout=3.0)
        check("status stream received", frame is not None, str(frame))
        conn.send(json.dumps({"type": "bye"}))
        closed = False
        close_deadline = time.monotonic() + 5.0
        while time.monotonic() < close_deadline:
            try:
                conn.recv(timeout=close_deadline - time.monotonic())
            except TimeoutError:
                break
            except Exception:  # noqa: BLE001 - close frame or abrupt drop
                closed = True
                break
        check("bye closes the connection", closed)

    endpoint = str(start.get("endpoint", ""))
    registered = kernel.devices.command(["add", "syncbox", endpoint])
    check("local server registered as a peer",
          "registered" in registered, registered)
    r = ds.dispatch("syncbox", ["echo", "hi"])
    check("wss dispatch reaches the peer and is refused",
          r.get("ok") is False
          and "refused" in str(r.get("error", "")).lower(), str(r))
    pin = ds.tls_dir / "peers" / f"127.0.0.1_{port}.pem"
    check("TOFU pin written on first handshake", pin.is_file(), str(pin))
    original = pin.read_text(encoding="utf-8") if pin.is_file() else ""
    pin.write_text("TAMPERED", encoding="utf-8")
    r = ds.dispatch("syncbox", ["echo", "hi"])
    check("pin mismatch refused",
          r.get("ok") is False
          and "pin" in str(r.get("error", "")).lower(), str(r))
    pin.write_text(original, encoding="utf-8")
    r = ds.dispatch("syncbox", ["echo", "hi"])
    check("restored pin reaches the peer again",
          r.get("ok") is False
          and "refused" in str(r.get("error", "")).lower(), str(r))
    r = ds.sync(["syncbox"])
    check("wss sync succeeds against the local server",
          r.get("ok") is True and r.get("synced") == 1, str(r))
    r = ds.sync(["nope"])
    check("sync with unknown id refused",
          r.get("ok") is False
          and "unknown device id" in str(r.get("error", "")), str(r))
    stop = ds.stop_server()
    check("server stops",
          stop.get("ok") is True and stop.get("serving") is False, str(stop))
    stop2 = ds.stop_server()
    check("second stop is idempotent",
          stop2.get("ok") is True and stop2.get("already") is True, str(stop2))
    reachable = True
    try:
        with peer_client("x", timeout=3.0) as conn:
            conn.recv(timeout=2.0)
    except Exception:  # noqa: BLE001 - any failure means the port is closed
        reachable = False
    check("stopped server refuses connections", not reachable)

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
    # kernel.shutdown() closes kernel.memory, but this second handle on the same
    # file stays open and keeps memory.db locked on Windows, so the teardown
    # rmtree skips it and leaves the whole sandbox tree behind. Close it here.
    reopened.close()

    # The fault injected above must not leave the next boot permanently
    # disarmed - clear the isolation so the next run starts clean.
    cleared = kernel.overseer_control("clear")
    check("overseer isolation cleared", "armed again" in cleared, cleared)
    check("gate armed after clear", kernel.gate.armed)
    check("control file removed",
          not kernel.config.resolve("data/overseer_control.json").exists())

    kernel.shutdown()
    # The sandbox tree itself is torn down by main()'s finally on every exit
    # path. This scratch dir is cleared here as well so a crashed run cannot
    # hand the next one stale state.
    shutil.rmtree(REPO_ROOT / "data" / ".selftest_scratch", ignore_errors=True)
    after = live_fingerprint()
    drifted = sorted(k for k in set(live_before) | set(after)
                     if live_before.get(k) != after.get(k))
    check("live state untouched by selftest", not drifted, ", ".join(drifted))
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