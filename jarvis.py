#!/usr/bin/env python3
"""Jarvis OS - single entry point.

    python jarvis.py                 # interactive REPL (default)
    python jarvis.py --loop          # headless infinite loop only
    python jarvis.py --daemon        # detached background mode
    python jarvis.py --overseer      # run the Overseer watchdog on its own
    python jarvis.py --goal "..."    # run one goal, then exit
    python jarvis.py --plan "..."    # decompose a goal into a step DAG, print, exit
    python jarvis.py --remember "..."  # store a fact in memory + local index, exit
    python jarvis.py --attach        # recall a running background instance

    python jarvis.py --build "spec"           # build software, then exit
    python jarvis.py --install <id>           # install a built program, then exit
    python jarvis.py --launch <id>            # launch an installed program, then exit
    python jarvis.py --control <id> <action>  # control/status/logs a program
    python jarvis.py --control gui ocr        # desktop OCR via the control bridge
    python jarvis.py --stop <id>              # stop a running program, then exit
    python jarvis.py --tools                  # list the tool registry, then exit

    python jarvis.py --device serve           # serve device sync, then exit
    python jarvis.py --sync [id ...]          # sync one device / all devices, then exit
    python jarvis.py --translate "text" [--to en|ta]  # translate offline, then exit

Every flag is a SESSION override: none of them writes config.json. Jarvis
addresses the operator as "Sir" in every channel and runs exclusively on
local Ollama. No external AI or cloud service is contacted without a live,
manual, per-request approval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Jarvis answers in the operator's language, and Tamil is not representable in
# the default Windows console code page (cp1252). Printing it would raise
# UnicodeEncodeError and take down the reply mid-print, so the streams are
# switched to UTF-8 with replacement before anything can write to them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        # Not a real TextIOBase (a test capture object, or pythonw): nothing to
        # reconfigure, and losing the guard must never stop Jarvis from starting.
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jarvis_os.config import Config          # noqa: E402
from jarvis_os.kernel import JarvisKernel    # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface as a factory, so selftests can parse without booting."""
    parser = argparse.ArgumentParser(
        prog="jarvis", description="Jarvis OS - self-evolving local Agentic OS")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--loop", action="store_true", help="headless infinite loop")
    parser.add_argument("--daemon", action="store_true", help="background/detached mode")
    parser.add_argument("--attach", action="store_true", help="recall a background instance")
    parser.add_argument("--overseer", action="store_true", help="run the Overseer watchdog")
    parser.add_argument("--goal", default="", help="execute a single goal then exit")
    parser.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = forever)")
    parser.add_argument("--no-overseer", action="store_true", help="disable the watchdog")
    parser.add_argument("--no-voice", action="store_true", help="force voice off")
    parser.add_argument("--voice", action="store_true",
                        help="start the background voice daemon for this session only")
    parser.add_argument("--honorific", default=None, help="override the honorific")
    # --- one-shot software / control actions (session-scoped, exit after) ---
    parser.add_argument("--build", default="", metavar="SPEC",
                        help="build software from a one-line spec, then exit")
    parser.add_argument("--install", default="", metavar="PROGRAM-ID",
                        help="install a built program by id, then exit")
    parser.add_argument("--launch", default="", metavar="PROGRAM-ID",
                        help="launch an installed program, then exit")
    parser.add_argument("--control", nargs="+", default=None, metavar="ARGS",
                        help="control a program: <id> [start|stop|restart|status|"
                             "inspect|logs|configure|signal], or 'gui <ocr|"
                             "click_text|type_text>' for the desktop")
    parser.add_argument("--stop", default="", metavar="PROGRAM-ID",
                        help="stop a running program, then exit")
    parser.add_argument("--tools", action="store_true",
                        help="list the registered tool registry, then exit")
    # --- one-shot device sync / translation actions (session-scoped) --------
    parser.add_argument("--device", nargs="+", default=None, metavar="ARGS",
                        help="device sync control: serve|stop|status|scan|sync|"
                             "alert|list|add|remove|trust|reach|configure|"
                             "current, then exit")
    parser.add_argument("--sync", nargs="*", default=None, metavar="ID",
                        help="sync with the given device ids (none = every "
                             "registered device), then exit")
    parser.add_argument("--translate", nargs="+", default=None, metavar="TEXT",
                        help="translate text offline into the target "
                             "language, then exit")
    parser.add_argument("--to", default="", choices=("en", "ta"),
                        help="target language for --translate (default: your "
                             "preferred language)")
    # --- one-shot planner / memory actions (session-scoped, exit after) -----
    parser.add_argument("--plan", default="", metavar="GOAL",
                        help="decompose a goal into a step DAG and print it "
                             "without executing, then exit")
    parser.add_argument("--remember", default="", metavar="TEXT",
                        help="store a fact in memory and the local vector "
                             "index, then exit")
    return parser


def _report(kernel: JarvisKernel, result: object) -> int:
    """Print one action verdict through the text interface; 0 when ok."""
    iface = kernel.text_iface
    if isinstance(result, dict):
        if result.get("ok"):
            message = result.get("message") or json.dumps(
                result.get("program", result), ensure_ascii=False)[:800]
            iface.say(str(message))
            return 0
        if result.get("denied"):
            iface.say("Permission denied by operator, Sir. Nothing was changed.")
            return 1
        iface.say(f"Failed: {result.get('error', 'unknown error')}")
        return 1
    iface.say(str(result))
    return 0


def _run_action(kernel: JarvisKernel, args: argparse.Namespace) -> int:
    """Dispatch one one-shot action flag. The kernel is already booted."""
    if args.tools:
        kernel.text_iface.plain(kernel.tools.describe())
        return 0
    if args.build:
        return _report(kernel, kernel.software.build(args.build))
    if args.install:
        return _report(kernel, kernel.software.install(args.install))
    if args.launch:
        return _report(kernel, kernel.software.launch(args.launch))
    if args.control:
        if args.control[0].lower() == "gui":
            # Desktop commands address the screen, not a registered program.
            return _report(kernel, kernel.control.dispatch("", "gui",
                                                           args.control[1:]))
        program_id, rest = args.control[0], args.control[1:]
        action = rest[0].lower() if rest else "status"
        return _report(kernel, kernel.software.control(program_id, action,
                                                       rest[1:]))
    if args.device is not None:
        result = kernel.device_sync.command(args.device)
        if isinstance(result, dict) and result.get("serving"):
            # serve stays up until the STOP file or Ctrl+C: an exit here would
            # tear the listener down the moment it came up.
            return _serve_until_stopped(kernel, result)
        return _report(kernel, result)
    if args.sync is not None:
        return _report(kernel, kernel.device_sync.sync(args.sync or None))
    if args.translate is not None:
        return _report(kernel, kernel.language_layer.translate(
            " ".join(args.translate), args.to))
    if args.plan:
        plan, _strategy = kernel.planner.decompose(args.plan)
        kernel.text_iface.plain(kernel.planner.render(plan))
        return 0
    if args.remember:
        return _report(kernel, kernel.remember_fact(args.remember))
    return _report(kernel, kernel.software.control(args.stop, "stop"))


def _serve_until_stopped(kernel: JarvisKernel, result: dict) -> int:
    """Serve device sync until data/STOP exists or the operator interrupts."""
    _report(kernel, result)
    stop_file = kernel.config.resolve(
        str(kernel.config.get("loop.stop_file", "data/STOP")))
    try:
        while kernel.device_sync.running:
            if stop_file.exists():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    return _report(kernel, kernel.device_sync.stop_server())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config(args.config)

    # Every flag below is a SESSION override, never a config write: persisting
    # them would let a one-off smoke run permanently rewrite the machine - a
    # single `--cycles 2` would cap the infinite loop forever, and a `--daemon`
    # or `--no-overseer` test would leave the watchdog off on every next boot.
    if args.no_overseer:
        config.set_feature("overseer", False, persist=False)
    if args.no_voice:
        config.set_feature("voice", False, persist=False)
    if args.voice:
        # Session-scoped voice enable: the kernel sees the feature on at boot
        # and starts jarvis-voice, but config.json stays byte-identical.
        config.set_feature("voice", True, persist=False)
    if args.daemon:
        config.set_feature("background_daemon", True, persist=False)
    if args.cycles:
        config.set("loop.max_cycles", int(args.cycles), persist=False)

    # One dispatch, two gates: the feature block below and the action block
    # after boot must agree on what counts as a one-shot action, so the
    # predicate is computed exactly once instead of drifting between them.
    one_shot = bool(
        args.tools or args.build or args.install or args.launch
        or args.control or args.stop
        or args.device is not None or args.sync is not None
        or args.translate is not None
        or args.plan or args.remember)

    if one_shot:
        # Session-scoped enable for the one-shot action flags: the kernel sees
        # the features on at boot; persist=False keeps config.json untouched.
        if args.tools or args.build or args.install or args.launch \
                or args.control or args.stop:
            config.set_feature("software_factory", True, persist=False)
            config.set_feature("control_bridge", True, persist=False)
        if args.device is not None or args.sync is not None:
            config.set_feature("device_manager", True, persist=False)
        if args.translate is not None:
            config.set_feature("language_layer", True, persist=False)

    if args.overseer:
        from jarvis_os.overseer import run_overseer
        return run_overseer(args.config)

    kernel = JarvisKernel(config, honorific=args.honorific)

    if one_shot:
        try:
            return _run_action(kernel, args)
        finally:
            kernel.shutdown()

    if args.goal:
        kernel.text_iface.say(kernel.execute(args.goal, source="cli"))
        kernel.shutdown()
        return 0

    if args.loop or args.daemon:
        if args.daemon:
            kernel.text_iface.plain(
                f"Jarvis OS running in the background (pid {__import__('os').getpid()}).")
            kernel.text_iface.plain("Recall it any time with:  background foreground")
        return kernel.run_forever()

    if args.attach:
        kernel.text_iface.say(kernel.daemon.attach([]))
        return 0

    kernel.text_iface.say(
        f"Everything is up. The Overseer is watching me and memory is safe - "
        f"it has survived {kernel.memory.stats.get('boots', 1)} boots.")
    try:
        return kernel.text_iface.start()
    except KeyboardInterrupt:
        kernel.shutdown()
        return 0
    finally:
        kernel.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())