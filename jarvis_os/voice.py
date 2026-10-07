"""VoiceDaemon - the always-on background listening thread.

A single daemon thread owns the microphone end to end: a non-blocking
``sounddevice`` InputStream feeds 16 kHz audio into an offline Vosk
recogniser, and only utterances that start with the wake word "Jarvis" are
routed into the OS. Everything is local - no audio ever leaves the machine.

Design rules, in order of importance:

* **Side-effect-free construction.** ``VoiceDaemon(kernel)`` opens nothing,
  imports nothing heavy and starts no thread, so kernels built by the
  selftest (voice off) and text-only runs never touch the microphone.
  :meth:`start` owns every import and every resource acquisition.
* **Graceful degradation.** Missing ``vosk``/``sounddevice`` packages or a
  missing model directory produce a spoken/logged hint to run
  ``setup-windows-daemon.ps1 -Action install`` - never an exception.
* **Echo suppression.** While Jarvis is speaking (pyttsx3), incoming audio
  is dropped so the daemon cannot hear its own voice, wake itself up or
  feed the TTS output back into the recogniser.
* **Wake-word protocol.** "jarvis" alone arms an 8 second window in which
  the next utterance is taken as a command; "jarvis <command>" runs the
  command immediately. Stray speech without the wake word is ignored.

Routing mirrors the REPL: recognised text goes through the language layer
first, then a leading REPL verb (``status``, ``memory`` ...) is dispatched
to the text interface, anything else becomes a goal for the TaskManager.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .interfaces.voice import WAKE_RE


class VoiceDaemon:
    """Background wake-word listener (``jarvis-voice`` thread)."""

    SAMPLE_RATE = 16000          # Hz, what the Vosk small English model expects
    BLOCK_FRAMES = 4000          # callback granularity (~250 ms)
    ARM_WINDOW = 8.0             # seconds a bare "jarvis" keeps listening
    BUFFER_SECONDS = 10          # audio held while a goal is executing
    _BYTE_RATE = SAMPLE_RATE * 2  # int16 mono -> 32 000 bytes/s

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.config = kernel.config
        self.running = False
        self.armed_until = 0.0
        self.heard_count = 0
        self.last_transcript = ""
        self._speaking = False
        self._buffer: list[bytes] = []
        self._buffered_bytes = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._model: Any = None
        self._rec: Any = None
        self._speaker_hook = self._on_speaking

    # ------------------------------------------------------------ lifecycle
    def start(self) -> tuple[bool, str]:
        """Open the microphone and launch the recognition thread."""
        if self.running:
            return True, "Voice daemon already running, Sir."
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except ImportError as exc:
            return False, (f"Voice daemon needs '{exc.name}' - run "
                           "setup-windows-daemon.ps1 -Action install, Sir.")

        model_path = self._model_path()
        if not model_path.is_dir():
            return False, (f"Vosk model not found at {model_path.as_posix()} - run "
                           "setup-windows-daemon.ps1 -Action install, Sir.")
        try:
            import sounddevice as sd
            import vosk

            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(model_path))
            self._rec = vosk.KaldiRecognizer(self._model, float(self.SAMPLE_RATE))
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the boot
            self._model = self._rec = None
            return False, f"Voice daemon failed to load Vosk: {type(exc).__name__}: {exc}, Sir."
        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_FRAMES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - no mic is not fatal
            self._model = self._rec = None
            return False, f"Microphone unavailable: {exc}, Sir."

        self.kernel.voice_iface.add_speaking_listener(self._speaker_hook)
        with self._lock:
            self._buffer.clear()
            self._buffered_bytes = 0
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="jarvis-voice",
                                        daemon=True)
        self._thread.start()
        if self.kernel.audit:
            self.kernel.audit.record("voice", "daemon_start", "ok",
                                     {"model": model_path.name,
                                      "sample_rate": self.SAMPLE_RATE})
        return True, "Voice daemon is listening for 'Jarvis', Sir."

    def stop(self) -> str:
        """Release the microphone and join the recognition thread."""
        was_running = self.running
        self.running = False
        self.armed_until = 0.0
        self.kernel.voice_iface.remove_speaking_listener(self._speaker_hook)
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._model = self._rec = None
        with self._lock:
            self._buffer.clear()
            self._buffered_bytes = 0
        if was_running and self.kernel.audit:
            self.kernel.audit.record("voice", "daemon_stop", "ok", {})
        return "Voice daemon stopped." if was_running else "Voice daemon was not running."

    def status(self) -> str:
        if not self.running:
            return "voice daemon: stopped"
        armed = "armed" if time.time() < self.armed_until else "idle"
        return f"voice daemon: listening ({armed}, heard={self.heard_count})"

    def _model_path(self) -> Path:
        raw = str(self.config.get("voice.vosk_model_path",
                                  "models/vosk-model-small-en-us-0.15"))
        path = Path(raw)
        if not path.is_absolute():
            path = Path(self.config.root) / path
        return path

    # ------------------------------------------------------------- audio in
    def _audio_callback(self, indata: Any, frames: int, time_info: Any,
                        status: Any) -> None:
        """sounddevice callback - keep it allocation-light and never block."""
        if not self.running or self._speaking:
            return
        chunk = bytes(indata)
        with self._lock:
            self._buffer.append(chunk)
            self._buffered_bytes += len(chunk)
            # Bounded ring: while a long goal runs, old audio is dropped
            # rather than growing without limit.
            while self._buffered_bytes > self.BUFFER_SECONDS * self._BYTE_RATE:
                self._buffered_bytes -= len(self._buffer[0])
                self._buffer.pop(0)

    def _on_speaking(self, speaking: bool) -> None:
        """Hook fired by VoiceCommandInterface.say() around TTS playback."""
        self._speaking = speaking
        if speaking:
            with self._lock:
                self._buffer.clear()
                self._buffered_bytes = 0

    # ------------------------------------------------------- recognition loop
    def _loop(self) -> None:
        while self.running:
            with self._lock:
                if not self._buffer:
                    chunk = b""
                else:
                    chunk = b"".join(self._buffer)
                    self._buffer.clear()
                    self._buffered_bytes = 0
            if not chunk:
                time.sleep(0.05)
                continue
            rec = self._rec
            if rec is None:
                break
            try:
                if rec.AcceptWaveform(chunk):
                    text = json.loads(rec.Result()).get("text", "")
                else:
                    # Partial hypotheses only arm early: "Jarvis" said with a
                    # quick follow-up must not lose the wake word to silence.
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial and time.time() >= self.armed_until \
                            and WAKE_RE.search(partial):
                        self.armed_until = time.time() + self.ARM_WINDOW
                    text = ""
            except Exception:  # noqa: BLE001 - a bad chunk must not kill audio
                continue
            if text:
                self._on_final(text)

    # ------------------------------------------------------------- routing
    def _on_final(self, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        self.heard_count += 1
        self.last_transcript = text
        iface = self.kernel.voice_iface
        iface.heard_count += 1
        iface.last_transcript = text
        iface._audit("heard", {"transcript": text[:200], "source": "daemon"})

        now = time.time()
        armed = now < self.armed_until
        if not armed:
            if not WAKE_RE.search(text):
                # Stray speech: not addressed to Jarvis, ignore it.
                if self.kernel.audit:
                    self.kernel.audit.record("voice", "ignored", "ok",
                                             {"transcript": text[:120]})
                return
            command = WAKE_RE.sub("", text).strip(" ,.")
            if not command:
                # Bare "Jarvis" - acknowledge and arm the command window.
                self.armed_until = now + self.ARM_WINDOW
                iface.say("Yes, Sir?")
                return
            self.armed_until = 0.0
            self._route(command)
            return
        # Armed window: everything heard counts, wake word or not.
        self.armed_until = 0.0
        self._route(WAKE_RE.sub("", text).strip(" ,.") or text)

    def _route(self, command: str) -> None:
        """Language layer -> REPL verb dispatch, else TaskManager goal."""
        if self.kernel.audit:
            self.kernel.audit.record("voice", "command", "ok",
                                     {"command": command[:200]})
        try:
            english, _detection = self.kernel.language.process_inbound(command)
        except Exception:  # noqa: BLE001 - the raw words are good enough
            english = command
        english = (english or command).strip()
        if not english:
            return
        first = english.split()[0].lower().strip(",.")
        if first in self.kernel.text_iface.COMMANDS:
            self.kernel.text_iface.dispatch(english)
            return
        self._run_goal(english)

    def _run_goal(self, goal: str) -> None:
        """Same outcome reporting as the REPL 'run' command, spoken."""
        iface = self.kernel.voice_iface
        iface.say(f"Working on it: {goal}")
        try:
            reply, report = self.kernel.execute_report(goal, source="voice")
        except Exception as exc:  # noqa: BLE001
            iface.say(f"Sir, I could not complete that: {type(exc).__name__}: {exc}")
            return
        if report.partial:
            missing = ", ".join(report.unexecuted_tools)
            iface.say(f"PARTIAL. This goal is not fully done: the plan listed "
                      f"{missing}, but no tool actually ran, so that part is "
                      f"unverified.")
        elif not report.ok:
            iface.say("INCOMPLETE. Every delegated step failed - see the "
                      "audit trail.")
        if reply:
            iface.say(reply)
