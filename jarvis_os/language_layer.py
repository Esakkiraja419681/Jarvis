"""Operator-facing language surface: on-demand translation + intent matching.

Threat model
------------
This layer performs no I/O of its own. Every byte it reads was already
reachable through the kernel: ``UniversalLanguageLayer`` never downloads a
model and never talks to a non-local host, so wrapping it here adds no new
network or filesystem surface. The risks this class is built around are
therefore behavioural, not networked:

* **Silent failure.** ``UniversalLanguageLayer.translate`` returns the source
  text unchanged when no backend can serve the pair - that is correct for the
  reply path (see the ``_NOTE_*`` rationale in ``language.py``) but wrong for
  an explicit ``--translate`` request, where the operator must be told the
  text was not translated. Success is therefore decided by comparing the
  output against the input: unchanged output degrades to ``ok: False`` with a
  ``degraded`` flag and a plain-language ``error``.
* **Error/response format drift.** Errors are rendered by ``_report`` as
  ``Failed: {error}`` and exit 1; ``message`` fields are rendered by ``say``
  and exit 0. Errors therefore carry no honorific (it would be doubled by the
  renderer) and every ``message`` is routed through ``respond`` so it opens
  with exactly one honorific.
* **Feature-flag bypass.** ``features.language_layer`` is re-read on every
  call rather than cached at boot, so a session-scoped ``set_feature``
  (``persist=False``) takes effect immediately and fails closed.
* **Intent matching as a code path.** Match rules are a static, regex-bounded
  table: no model call, no eval, no operator input ever reaches an
  interpreter. ASCII keywords are word-boundary anchored so substrings
  ("this" must not fire "hi") and Tamil keywords match as substrings because
  Tamil runs words together without spaces.

Mechanism
---------
``translate`` is a three-gate funnel (feature off -> empty -> unsupported
target) before it ever touches a backend, then delegates the actual work to
``language.translate`` - which owns backend chaining, caching and the audit
ledger - and classifies the result as success (``already``/``translated``)
or graceful degradation. ``parse_intent`` is a single pass over the ordered
``_INTENTS`` table, first match wins with the greeting deliberately last so
"translate this" is a translate request even when it opens with "hello".
``respond`` is the one and only way text leaves this class toward the
operator, and it is intentionally not feature-gated: the honorific is an
invariant of the product, not a translation feature.
"""

from __future__ import annotations

import re
from typing import Any


class LanguageLayer:
    """Translate on demand and match operator input to a bounded intent table."""

    _QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
    _NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

    # (intent, confidence, keywords). First match wins; greeting is last so
    # that "hello, translate this" stays a translate request.
    _INTENTS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
        ("translate", 0.95, ("translate", "மொழிபெயர்")),
        ("sync", 0.90, ("sync", "synchronize", "ஒத்திசை")),
        ("alert", 0.90, ("alert", "alerts", "notify", "warning", "எச்சரிக்கை")),
        ("remember", 0.92, ("remember", "memorize", "note this")),
        ("recall", 0.88, ("recall", "retrieve", "fetch")),
        ("build", 0.90, ("build", "compile")),
        ("launch", 0.88, ("launch", "open program", "start program")),
        ("stop", 0.92, ("stop", "halt", "நிறுத்து")),
        ("learn", 0.85, ("learn", "study", "கற்று")),
        ("help", 0.90, ("help", "commands", "உதவி")),
        ("time", 0.88, ("time", "clock", "date", "நேரம்")),
        ("status", 0.85, ("status", "state", "health", "நிலை")),
        ("greeting", 0.90, ("hello", "hi", "hey", "good morning", "வணக்கம்")),
    )

    def __init__(self, config, language) -> None:
        self.config = config
        self.language = language

    @property
    def _enabled(self) -> bool:
        """Live view of the feature flag: session overrides take effect now."""
        return bool(self.config.feature("language_layer", True))

    def respond(self, text: Any) -> str:
        """Operator-facing text with exactly one honorific prefix."""
        return self.language.prefix(str(text if text is not None else ""))

    def translate(self, text: str = "", target: str = "") -> dict[str, Any]:
        """Translate ``text`` into ``target`` (default: operator's language).

        Returns a result dict. ``message`` is set only when something is
        shown as a normal reply (honorific applied); failures carry a plain
        ``error`` that the CLI renderer prefixes with "Failed: ".
        """
        if not self._enabled:
            return {"ok": False, "error":
                    "language layer is off for this session (features.language_layer)."}
        body = str(text if text is not None else "").strip()
        if not body:
            return {"ok": False, "error":
                    "there is no text to translate - give me text to translate."}
        wanted = str(target or self.language.preferred or "en").strip().lower()
        if wanted not in ("en", "ta"):
            return {"ok": False, "error":
                    f"unsupported target '{wanted}' - this build serves en and ta only."}

        source = self.language.detect(body).language
        backend = self.language.backend_for(source, wanted)
        if source == wanted:
            return {"ok": True, "already": True, "source": source, "target": wanted,
                    "backend": backend,
                    "message": self.respond(
                        f"text is already in {wanted} - nothing to translate.")}

        out = self.language.translate(body, wanted)
        if out == body:
            # Unchanged output means no backend produced a translation. The
            # reason is already in the log and the audit ledger; this is the
            # operator-facing statement that the text was NOT translated.
            if backend == "passthrough":
                error = (f"no offline model installed for {source}->{wanted} "
                         f"- the text is unchanged.")
            else:
                error = (f"backend '{backend}' is available but could not translate "
                         f"{source}->{wanted} right now - the text is unchanged.")
            return {"ok": False, "degraded": True, "source": source, "target": wanted,
                    "backend": backend, "error": error}
        return {"ok": True, "already": False, "source": source, "target": wanted,
                "backend": backend, "translated": out, "message": self.respond(out)}

    @staticmethod
    def _keyword_hit(keyword: str, body: str) -> bool:
        """Whole-word match for ASCII keywords, substring match for Tamil."""
        if keyword.isascii():
            pattern = (rf"(?<![0-9A-Za-z]){re.escape(keyword)}"
                       rf"(?![0-9A-Za-z])")
            return re.search(pattern, body, re.IGNORECASE) is not None
        return keyword in body

    def parse_intent(self, text: str) -> dict[str, Any]:
        """Classify ``text`` against the bounded intent table.

        Always succeeds (``ok: True``): an unknown request is a valid answer,
        reported as intent "unknown" at confidence 0.0.
        """
        body = str(text if text is not None else "").strip()
        quoted = [a or b for a, b in self._QUOTED_RE.findall(body) if a or b]
        numbers = self._NUMBER_RE.findall(body)

        intent, confidence = "unknown", 0.0
        for name, score, keywords in self._INTENTS:
            if any(self._keyword_hit(keyword, body) for keyword in keywords):
                intent, confidence = name, score
                break

        return {"ok": True, "intent": intent, "confidence": confidence,
                "entities": {"quoted": quoted, "numbers": numbers},
                "message": self.respond(
                    f"intent '{intent}' matched (confidence {confidence:.2f}).")}

    def status(self) -> dict[str, Any]:
        """Operator-readable state: flag, language, backends, rule count."""
        language = self.language
        return {"ok": True, "enabled": self._enabled,
                "preferred": str(language.preferred),
                "honorific": str(language.honorific),
                "backends": {"active": str(language.backend),
                             "en->ta": str(language.backend_for("en", "ta")),
                             "ta->en": str(language.backend_for("ta", "en"))},
                "intent_rules": len(self._INTENTS)}
