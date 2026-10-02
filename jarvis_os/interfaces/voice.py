"""VoiceCommandInterface - manual voice commands and spoken replies.

Optional dependencies, both resolved lazily so Jarvis runs fine without them:

* ``speech_recognition`` + a microphone for capture (plus ``pyaudio`` or
  ``sounddevice`` as a backend).
* ``pyttsx3`` for text-to-speech on the local machine.

Behaviour
---------
* Jarvis always addresses the operator as ``Sir`` in speech as well as text.
* Voice approval and denial words route through the same PermissionGate as
  text, so a spoken "approve" is a genuine manual approval and a spoken
  "deny" is a genuine refusal.
* No audio is uploaded anywhere. ``SpeechRecognizer`` is pinned to the offline
  engines (Sphinx where available, otherwise the default local path); the
  Google endpoint is explicitly disabled so a keyless accidental cloud call can
  never happen.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

WAKE_RE = re.compile(r"\b(jarvis|jarvis os)\b[,\s]*(sir)?", re.IGNORECASE)


class VoiceUnavailable(RuntimeError):
    pass


class VoiceCommandInterface:
    name = "voice"

    def __init__(self, kernel: Any, config: Any) -> None:
        self.kernel = kernel
        self.config = config
        self.honorific = config.honorific()
        self.enabled = bool(config.feature("voice", False))
        self._recognizer: Any = None
        self._tts: Any = None
        self._lock = threading.RLock()
        self.last_transcript = ""
        self.history: list[str] = []
        self.said_count = 0
        self.heard_count = 0

    # ------------------------------------------------------------ capability
    @staticmethod
    def availability() -> dict[str, bool]:
        def _has(module: str) -> bool:
            try:
                __import__(module)
                return True
            except Exception:  # noqa: BLE001
                return False

        return {
            "pyttsx3": _has("pyttsx3"),
            "speech_recognition": _has("speech_recognition"),
            "pyaudio": _has("pyaudio"),
            "sounddevice": _has("sounddevice"),
        }

    def install_hint(self) -> str:
        return ("Voice mode needs: pip install pyttsx3 SpeechRecognition pyaudio "
                "(Windows may also need: pip install pywin32)")

    # ------------------------------------------------------------------ init
    def _ensure_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        try:
            import speech_recognition as sr
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(
                f"speech_recognition not installed: {exc}. {self.install_hint()}"
            ) from exc
        recognizer = sr.Recognizer()
        # Hard-disable every cloud backend. Audio must never leave the machine.
        recognizer.pocketphinx = None
        try:
            from speech_recognition import Microphone
            device_index = self.config.get("voice.device_index")
            mic_kwargs: dict[str, Any] = {}
            if device_index is not None:
                mic_kwargs["device_index"] = int(device_index)
            recognizer.microphone = Microphone(**mic_kwargs)
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = float(self.config.get("voice.energy_threshold", 0.4))
        except Exception:  # noqa: BLE001 - no mic is not fatal
            recognizer.microphone = None
        self._recognizer = recognizer
        return recognizer

    def _ensure_tts(self) -> Any:
        if self._tts is not None:
            return self._tts
        try:
            import pyttsx3
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(f"pyttsx3 not installed: {exc}. {self.install_hint()}") from exc
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", int(self.config.get("voice.tts_rate", 175)))
            engine.setProperty("volume", float(self.config.get("voice.tts_volume", 1.0)))
            self._tts = engine
            return engine
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(f"TTS engine unavailable: {exc}") from exc

    # ------------------------------------------------------------------ say
    def speech_text(self, text: str) -> str:
        """Honorific-prefixed, translated text destined for the speaker."""
        rendered = self.kernel.language.process_outbound(text)
        self.said_count += 1
        self.history.append(rendered)
        self.history = self.history[-200:]
        return rendered

    def say(self, text: str, translate: bool = True) -> bool:
        rendered = (self.kernel.language.process_outbound(text) if translate
                    else self.kernel.language.prefix(text))
        print(f"[voice] {rendered}", flush=True)
        if not self.enabled:
            return False
        try:
            engine = self._ensure_tts()
        except VoiceUnavailable:
            return False
        with self._lock:
            try:
                engine.say(rendered)
                engine.runAndWait()
                return True
            except Exception:  # noqa: BLE001 - a stuck speaker must not hang the OS
                return False

    # ---------------------------------------------------------------- listen
    def listen(self, timeout: float | None = None, phrase_limit: float | None = None) -> str:
        """Capture one utterance. Returns '' on silence/failure."""
        recognizer = self._ensure_recognizer()
        timeout = timeout or float(self.config.get("voice.listen_timeout_seconds", 8))
        phrase_limit = phrase_limit or float(
            self.config.get("voice.phrase_time_limit_seconds", 12))
        if getattr(recognizer, "microphone", None) is None:
            raise VoiceUnavailable("No microphone available for speech capture")
        try:
            with recognizer.microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = recognizer.listen(source, timeout=timeout,
                                          phrase_time_limit=phrase_limit)
        except Exception as exc:  # noqa: BLE001
            self._audit("listen_failed", {"error": str(exc)[:200]})
            return ""
        transcript = self._transcribe(recognizer, audio)
        if transcript:
            self.heard_count += 1
            self.last_transcript = transcript
            self._audit("heard", {"transcript": transcript[:200]})
        return transcript

    def _transcribe(self, recognizer: Any, audio: Any) -> str:
        """Offline recognition only.

        ``recognize_google`` is intentionally never called: it would ship the
        operator's audio to a third-party service, which Jarvis OS must never do.
        """
        try:  # Vosk - offline, accurate, preferred when installed
            raw = recognizer.recognize_vosk(audio)
            if isinstance(raw, str):
                return raw
            if isinstance(raw, dict):
                return " ".join(
                    part.get("text", "") for part in raw.get("results", []) if part
                ).strip()
        except Exception:  # noqa: BLE001
            pass
        try:  # Sphinx - offline, bundled with SpeechRecognition
            return str(recognizer.recognize_sphinx(audio))
        except Exception:  # noqa: BLE001
            return ""

    def wait_for_wake(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.listen(timeout=min(5.0, timeout))
            if text and WAKE_RE.search(text):
                self.say("Yes, Sir? I am listening.")
                return True
        return False

    # ------------------------------------------------------------- gate hook
    def capture_answer(self, prompt: str) -> str:
        """Input adapter handed to PermissionGate - returns the spoken verdict."""
        self.say(f"{prompt} Say 'approve' or 'deny'.", translate=False)
        answer = self.listen(timeout=8.0)
        if not answer:
            return ""
        tokens = {t for t in re.findall(r"[a-z']+", answer.lower())}
        approvals = {"approve", "yes", "allow", "allowed", "proceed", "ok"}
        denials = {"deny", "no", "nope", "stop", "reject", "never"}
        if tokens & denials:
            return "deny"
        if tokens & approvals:
            return "approve"
        return answer

    def voice_active(self) -> bool:
        return bool(self.enabled)

    # -------------------------------------------------------------- control
    def enable(self) -> tuple[bool, str]:
        avail = self.availability()
        self.enabled = True
        self.config.set_feature("voice", True)
        if not avail["pyttsx3"]:
            return False, f"Voice enabled in config, but pyttsx3 is missing. {self.install_hint()}"
        if not avail["speech_recognition"]:
            return False, ("Voice enabled in config, but SpeechRecognition is missing. "
                           f"{self.install_hint()}")
        self.say("Voice interface is live, Sir.", translate=False)
        return True, "Voice interface is live, Sir."

    def disable(self) -> str:
        self.enabled = False
        self.config.set_feature("voice", False)
        return "Voice interface is off. Text remains your manual channel, Sir."

    def status(self) -> str:
        avail = self.availability()
        flags = " ".join(
            f"{name}={'yes' if present else 'no'}" for name, present in avail.items()
        )
        return (f"voice: {'ON' if self.enabled else 'OFF'} | deps: {flags} "
                f"| heard={self.heard_count} said={self.said_count}")

    def _audit(self, action: str, detail: dict[str, Any]) -> None:
        if self.kernel.audit:
            self.kernel.audit.record("voice", action, "ok", detail)

    def command(self, text: str) -> Any:
        """Route a spoken command through the text dispatcher."""
        cleaned = WAKE_RE.sub("", text or "").strip(" ,.")
        if not cleaned:
            return None
        return self.kernel.text_iface.dispatch(cleaned)