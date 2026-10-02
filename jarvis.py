#!/usr/bin/env python3
"""Jarvis OS - single entry point.

    python jarvis.py                 # interactive REPL (default)
    python jarvis.py --loop          # headless infinite loop only
    python jarvis.py --daemon        # detached background mode
    python jarvis.py --overseer      # run the Overseer watchdog on its own
    python jarvis.py --goal "..."    # run one goal, then exit
    python jarvis.py --attach        # recall a running background instance

Jarvis addresses the operator as "Sir" in every channel and runs exclusively on
local Ollama. No external AI or cloud service is contacted without a live,
manual, per-request approval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jarvis_os.config import Config          # noqa: E402
from jarvis_os.kernel import JarvisKernel    # noqa: E402


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--honorific", default=None, help="override the honorific")
    args = parser.parse_args(argv)

    config = Config(args.config)

    if args.no_overseer:
        config.set_feature("overseer", False)
    if args.no_voice:
        config.set_feature("voice", False)
    if args.daemon:
        config.set_feature("background_daemon", True)
    if args.cycles:
        config.set("loop.max_cycles", int(args.cycles))

    if args.overseer:
        from jarvis_os.overseer import run_overseer
        return run_overseer(args.config)

    kernel = JarvisKernel(config, honorific=args.honorific)

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