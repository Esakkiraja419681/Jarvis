# Jarvis Agentic OS

A terminal-based, self-improving agent operating system. It runs an infinite
heartbeat loop on a **local Ollama** model, splits your goals across specialist
agents, executes tools inside a sandbox, remembers everything permanently, and
lets a second independent watchdog (the Overseer) halt, isolate and repair it
when something goes wrong.

You are always addressed as **Sir**.

---

## Quick start

```powershell
# one-shot goal
python jarvis.py --goal "summarise the files in this folder"

# persistent session (REPL)
python jarvis.py

# bounded loop for a smoke test
python jarvis.py --loop --cycles 2

# full self-test suite (130 checks)
python selftest.py
```

Ollama must be running locally with at least one model pulled:

```powershell
ollama serve
ollama pull llama3.1
```

The default endpoint is `http://127.0.0.1:11434`. If Ollama is unreachable,
Jarvis says so and every other subsystem keeps working.

---

## Modes

| Command | Purpose |
| --- | --- |
| `python jarvis.py` | Interactive REPL |
| `python jarvis.py --goal "..."` | Decompose and execute one goal, then exit |
| `python jarvis.py --loop` | Infinite heartbeat loop (Ctrl+C to stop) |
| `python jarvis.py --daemon` | Run in the background, recallable via `background attach` |
| `python jarvis.py --overseer` | Run **only** the watchdog process |
| `python jarvis.py --no-voice` | Force the text interface |
| `python jarvis.py --honorific "Madam"` | Change the form of address |

To get a genuinely independent second Overseer process, start it separately:

```powershell
python jarvis.py --overseer
```

---

## Commands

```text
goal <text>          set the goal for the infinite loop
run                  execute one cycle now
ask <text>           direct question
recall <text>        search permanent memory
remember <text>      store something permanently (asks permission)
forget / purge       ALWAYS refused - memory is permanent
learn <text>         record a lesson (asks permission)

build <spec>         generate + install software
software <status|logs|configure|stop|start> <id>
software status      inspect the durable software registry

agent list              list every live agent
agent add <name> <role> register a new agent at runtime
agent remove <name>

device list             list registered devices
device add <id> [url]   register a remote device (asks permission)
device reach <id>       contact a device's /health endpoint (asks permission)
device configure <id> <program> [action]   push a control change (asks permission)
device trust <id>       toggle the trusted flag (asks permission)
device remove <id>      revoke a device (asks permission)
device current          show the machine Jarvis is running on

background <status|start|stop|attach|recall>

overseer <status|scan|on|off|clear>
skill add <name>     load a Python file as a new tool
skill list

config <key> [value] show or change a setting
evolve               run one evolution cycle
rollback <id>        undo an evolution
update check         show the git HEAD
scan                 full capability report
history              recent audit entries
heartbeat            current heartbeat payload
help                 command list
```

---

## The guarantees, and what enforces them

**Nothing happens without permission.** Sensitive operations route through
`jarvis_os/permissions.py`. The operator must answer `yes`/`allow` in text or
`approve` by voice, live, at the moment of the action. There is no token, flag
or config value that pre-authorises a category, and `auto_allow()` raises
`PermissionError` for every mandatory category.

**No external AI without a fresh, explicit approval.** The client is
provider-agnostic but locked to local Ollama by default. Any third-party or
cloud endpoint must be registered one at a time through
`register_external_provider()`, which asks the operator, and the grant is
single-use.

**Memory is permanent.** `MemoryStore.purge_requested()` refuses unconditionally
- there is no correct token to supply. Memories, lessons, actions and the vector
store are append-only. On boot Jarvis verifies the hash chains and repairs the
vector index if it was damaged; it never resets data.

**Nothing runs unsandboxed.** Python execution goes through a subprocess with a
clean preload that blocks `socket`/`ctypes`-style escape hatches and is killed on
timeout. Shell commands are validated against a metacharacter blacklist. File
tools refuse paths outside the project root.

**Everything is auditable.** Every permission decision, tool call, memory write,
software action and Overseer action is appended to a hash-chained SQLite ledger.
Writes use `BEGIN IMMEDIATE`, so concurrent processes interleave safely and
`audit verify` recomputes the whole chain.

**`devices.allow_remote_config` stays `false`, and the default is deliberate.**
`device configure` is refused outright while it is off - before the per-action
`cross_device` approval is even consulted - so remote configuration is a local-only
operation. Do not flip it on to make a device reachable:

* `ControlBridge` binds `control_bridge.rest_host`, which defaults to `127.0.0.1`
  (`jarvis_os/control.py`), and is token-protected. It is not listening on the LAN,
  so no other machine can be configured no matter what this flag says. Enabling it
  cannot make the feature work.
* Because the only reachable control endpoint is the laptop's own loopback one,
  turning the flag on would let Jarvis reconfigure *itself* over HTTP while the
  target nominally looked like a remote device. That is precisely the confusion
  that produced a phantom "phone" entry pointing at the host's own address; the
  entry has been removed rather than repointed.
* The flag is the second brake behind live approval. Defaulting it off keeps a
  whole class of remote writes shut until someone deliberately reopens it.

Treat `false` as the standing default. Only consider `true` when all four hold:
a real second device exists, its endpoint has been verified reachable, the control
bridge is still loopback-bound and token-protected, and the operator opts in for
that session knowing the target is a genuine separate machine.

---

## The Overseer

The watchdog detects faults across the layers (interface, planner, memory, tools,
llm, execution, permission) and can:

1. **Halt** Jarvis immediately on a permission violation, runaway loop, memory
   wipe attempt or unsafe action (`taskkill /T /F` on the process tree).
2. **Isolate** just the faulty layer.
3. **Diagnose** the root cause.
4. **Repair** the code in the sandbox and prove the fix before committing.
5. **Restart** Jarvis with memory fully preserved.

Every repair still requires your explicit approval, even when the Overseer itself
tripped the halt - it is never allowed to rewrite source unattended.

If an incident leaves Jarvis isolated and you want it back immediately:

```powershell
python jarvis.py            # then in the REPL:
overseer clear
```

This drops the isolation and re-arms the permission gate. The gate stays down for
ordinary requests while isolation is active, so an unresolved fault cannot be
ignored by accident.

---

## Offline by default

No model, language pack or index is ever downloaded automatically.

- Translation uses Argos Translate with locally installed packages. Install them
  deliberately with `install_argos_packages(approved=True)`, or ask Jarvis for
  the missing-pair list first.
- Speech-to-text uses SpeechRecognition with a local backend; text-to-speech uses
  `pyttsx3`.
- If the optional packages in `requirements.txt` are absent, Jarvis falls back to
  the text interface and says so rather than failing.

---

## Configuration

`config.json` holds every feature flag. Inspect and change values at runtime:

```text
config llm.model
config features.sandbox false
config evolution.self_update_enabled true
```

The Overseer also writes `data/overseer_control.json` when it isolates a layer;
delete it or run `overseer clear` to release the isolation.

---

## Layout

```text
jarvis.py                    CLI entry point / REPL
selftest.py                  130-check end-to-end suite
config.json                  feature flags and settings
jarvis_os/
  kernel.py                  central wiring, heartbeat, execution, loop
  config.py  audit.py  memory.py  permissions.py
  sandbox.py  tools.py  agents.py  taskmanager.py  llm.py
  language.py  selfupdate.py  evolution.py  overseer.py
  software_factory.py  control.py  devices.py  daemon.py
  interfaces/                text and voice front ends
data/                        runtime state (memory, audit, registries, logs)
```

`data/` holds the memories and is listed in `.gitignore`, so a self-update
snapshot or `git reset --hard` can never destroy them.

---

## Known limits

- `background attach` recreates the command surface rather than reattaching to a
  live terminal multiplexer session.
- The software factory generates and supervises Python programs; wrapping
  arbitrary existing binaries is limited.
- `--overseer` is a separate process you launch; the main kernel also runs an
  in-process watchdog thread, so a hard crash of the kernel needs the standalone
  Overseer already running to notice.
- The honourific is configurable; set it back to `Sir` with
  `--honorific "Sir"` or `config honorific "Sir"`.
- `data/devices.json` registers only this machine. Cross-device control is
  implemented and gated behind live approval, but it has never been exercised
  against real second hardware: the control bridge binds `127.0.0.1` only, so a
  genuinely remote device cannot be reached until that bind is deliberately
  widened. See the `allow_remote_config` decision above.
