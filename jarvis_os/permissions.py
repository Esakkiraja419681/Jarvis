"""PermissionGate - the single chokepoint for every sensitive capability.

Nothing in Jarvis OS executes a privileged action without passing through
:meth:`PermissionGate.approve`. The gate covers, by construction:

* self-improvement / self-update / self-evolution / self-learning
* skill addition
* external AI connection
* cross-device access

plus shell exec, web fetch, code exec, file mutation, software
install/launch/control, voice capture and daemon start.

Rules
-----
* Approval must be **live and manual**. Grants are never cached, never written
  to memory, never inferred from a previous "yes", even if the config's
  ``remember_grants`` is flipped on - self-authorisation is exactly the class of
  behaviour the gate exists to prevent.
* Every decision (approve, deny, timeout, escalate, auto-allow, refused) is
  written to the shared audit ledger with a hash chain.
* A denial is final for that request. Callers must not retry in a loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Categories that can *never* be auto-approved, regardless of config drift.
MANDATORY_APPROVAL = frozenset({
    "self_improve",
    "self_update",
    "self_evolution",
    "self_learn",
    "skill_add",
    "external_ai",
    "cross_device",
})

DECISION_ALLOW = "approved"
DECISION_DENY = "denied"
DECISION_TIMEOUT = "timeout"
DECISION_REFUSED = "refused"
DECISION_ESCALATED = "escalated"


@dataclass(slots=True)
class Decision:
    category: str
    decision: str
    reason: str = ""
    source: str = "text"
    ts: float = field(default_factory=time.time)

    @property
    def granted(self) -> bool:
        return self.decision == DECISION_ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category, "decision": self.decision,
            "reason": self.reason, "source": self.source, "ts": self.ts,
        }


class PermissionGate:
    def __init__(self, config, audit=None,
                 text_input: Callable[[str], str] | None = None,
                 voice_input: Callable[[], str] | None = None,
                 voice_active: Callable[[], bool] | None = None) -> None:
        self.config = config
        self.audit = audit
        self.honorific = config.honorific()
        self._text_input = text_input or self._console_input
        self._voice_input = voice_input
        self._voice_active = voice_active or (lambda: False)
        self._lock = threading.RLock()
        self.stats: dict[str, int] = {}
        self.denials: list[dict[str, Any]] = []
        self.last_decision: Decision | None = None
        self._armed = True  # master switch; overseer can disarm on fault

    # ------------------------------------------------------------ vocabulary
    def _approval_words(self) -> set[str]:
        words = self.config.get("permissions.approval_words", []) or []
        words = {str(w).strip().lower() for w in words}
        words.add(str(self.config.get("permissions.voice_approval_word", "approve")).lower())
        return words

    def _deny_words(self) -> set[str]:
        words = self.config.get("permissions.deny_words", []) or []
        words = {str(w).strip().lower() for w in words}
        words.add(str(self.config.get("permissions.voice_deny_word", "deny")).lower())
        return words

    def requires_approval(self, category: str) -> bool:
        if category in MANDATORY_APPROVAL:
            return True
        auto = {str(c) for c in (self.config.get("permissions.auto_allowed", []) or [])}
        if category in auto:
            return False
        gated = {
            str(c) for c in (
                self.config.get("permissions.categories_requiring_manual_approval", []) or []
            )
        }
        return category in gated

    # ---------------------------------------------------------------- master
    def disarm(self, reason: str = "overseer fault") -> None:
        """Master stop: every future approval request is refused."""
        with self._lock:
            self._armed = False
        self._log("permission", "gate:master", DECISION_REFUSED, {"reason": reason})

    def arm(self) -> None:
        with self._lock:
            self._armed = True

    @property
    def armed(self) -> bool:
        return self._armed

    # -------------------------------------------------------------- approval
    def approve(self, category: str, prompt: str, detail: dict[str, Any] | None = None,
                source: str = "text", timeout: float | None = None,
                override_disarm: bool = False) -> bool:
        """Return True only on a live manual approval from the operator.

        ``override_disarm`` exists for exactly one caller: the Overseer asking
        the operator whether it may repair the fault that caused the disarm.
        It does NOT re-arm Jarvis - every other Jarvis request still fails
        while the master switch is down. Without it the Overseer would be
        locked out of fixing the very incident it raised.
        """
        detail = dict(detail or {})
        detail["prompt"] = prompt

        if not self.requires_approval(category):
            self._log("permission", f"gate:{category}", "auto_allowed",
                      {**detail, "source": source})
            return True

        with self._lock:
            armed = self._armed
        if not armed and not override_disarm:
            self._log("permission", f"gate:{category}", DECISION_REFUSED,
                      {**detail, "source": source,
                       "reason": "master switch disarmed by Overseer"})
            return False

        use_voice = bool(self._voice_active()) and self._voice_input is not None
        effective_source = "voice" if use_voice else source
        self._announce(category, prompt, use_voice)
        self._log("permission", f"gate:{category}", "pending",
                  {**detail, "source": effective_source})

        try:
            # _announce() already printed the full prompt; passing it again
            # would echo the whole question twice.
            answer = (self._voice_input() if use_voice else self._text_input(""))
        except (EOFError, KeyboardInterrupt):
            answer = ""
        except Exception as exc:  # noqa: BLE001 - gate must never crash the OS
            self._log("permission", f"gate:{category}", DECISION_REFUSED,
                      {**detail, "error": f"input channel failed: {exc}"})
            return False

        verdict, normalized = self._classify(answer)
        granted = verdict == DECISION_ALLOW
        self.last_decision = Decision(
            category=category, decision=verdict,
            reason=f"operator said '{normalized}'" if normalized else "no usable answer",
            source=effective_source,
        )
        with self._lock:
            self.stats[verdict] = self.stats.get(verdict, 0) + 1
            if not granted:
                self.denials.append({"category": category, "prompt": prompt, "ts": time.time()})
        self._log("permission", f"gate:{category}", verdict,
                  {**detail, "source": effective_source,
                   "operator_reply": normalized, "reason": self.last_decision.reason})
        return granted

    def _classify(self, answer: str | None) -> tuple[str, str]:
        if answer is None:
            return DECISION_TIMEOUT, ""
        text = str(answer).strip().lower().strip(".!,")
        if not text:
            return DECISION_TIMEOUT, ""
        approvals, denials = self._approval_words(), self._deny_words()
        tokens = {t for t in text.replace("/", " ").split() if t}
        if tokens & denials:
            return DECISION_DENY, text
        if tokens & approvals:
            return DECISION_ALLOW, text
        if text in approvals:
            return DECISION_ALLOW, text
        if text in denials:
            return DECISION_DENY, text
        if any(t in denials for t in tokens):
            return DECISION_DENY, text
        if any(t in approvals for t in tokens):
            return DECISION_ALLOW, text
        return DECISION_TIMEOUT, text

    # ------------------------------------------------------------------- ui
    def _announce(self, category: str, prompt: str, voice: bool) -> None:
        line = (f"[PERMISSION] {self.honorific}, manual approval required "
                f"for '{category}'. {prompt}")
        print(f"\n{line}\n  type yes/allow (or no/deny) -> ", end="", flush=True)

    @staticmethod
    def _console_input(prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            return ""

    # ---------------------------------------------------------------- audit
    def _log(self, category: str, action: str, decision: str,
             detail: dict[str, Any]) -> None:
        if self.audit:
            self.audit.record(category, action, decision, detail)

    # ---------------------------------------------------------------- status
    def report(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())) or "no decisions yet"
        armed = "armed" if self.armed else "DISARMED BY OVERSEER"
        return f"gate {armed} | {parts}"

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.audit:
            return []
        return self.audit.tail(limit=limit, category="permission")

    # ------------------------------------------------------- rule management
    def require(self, category: str) -> "PermissionGate":
        """Force a category to always need manual approval."""
        current = [str(c) for c in (
            self.config.get("permissions.categories_requiring_manual_approval", []) or []
        )]
        if category not in current:
            current.append(category)
            self.config.set("permissions.categories_requiring_manual_approval", current)
        auto = [str(c) for c in (self.config.get("permissions.auto_allowed", []) or [])]
        if category in auto:
            self.config.set("permissions.auto_allowed",
                            [c for c in auto if c != category])
        return self

    def auto_allow(self, category: str) -> "PermissionGate":
        """Deliberately downgrade a category to auto-approved.

        Mandatory categories cannot be downgraded - the gate refuses.
        """
        if category in MANDATORY_APPROVAL:
            raise PermissionError(
                f"'{category}' is a mandatory-approval category and can never be "
                "auto-allowed."
            )
        auto = [str(c) for c in (self.config.get("permissions.auto_allowed", []) or [])]
        if category not in auto:
            auto.append(category)
            self.config.set("permissions.auto_allowed", auto)
        gated = [str(c) for c in (
            self.config.get("permissions.categories_requiring_manual_approval", []) or []
        )]
        if category in gated:
            self.config.set("permissions.categories_requiring_manual_approval",
                            [c for c in gated if c != category])
        return self