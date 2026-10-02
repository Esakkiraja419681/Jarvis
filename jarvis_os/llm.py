"""LLM client for Jarvis OS - provider-agnostic interface, Ollama-locked body.

Design contract
---------------
* The public surface (:class:`LLMClient`) is provider-agnostic: any backend can
  be registered, so the rest of the OS never learns which vendor it talks to.
* The **only** backend that is actually wired up is local Ollama. Any attempt to
  reach a non-local host raises :class:`ExternalAccessDenied` *before* a socket
  is opened.
* Connecting an external AI service (OpenAI, Anthropic, Google, an arbitrary
  OpenAI-compatible gateway...) is possible in principle, but it is hard-gated:
  it demands a manual approval from the operator **every single time**, with no
  caching and no remembered grants. That gate lives in :meth:`LLMClient.
  register_external_provider` and is enforced through the injected ``gate``.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]"}


class ExternalAccessDenied(RuntimeError):
    """Raised when a non-local AI endpoint is touched without live approval."""


class LLMUnavailable(RuntimeError):
    """Raised when the local Ollama daemon is not reachable."""


@dataclass(slots=True)
class LLMReply:
    text: str
    model: str
    provider: str
    ok: bool = True
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok and bool(self.text)


@dataclass(slots=True)
class ToolSpec:
    """Function-calling schema (JSON-Schema subset) for a registry tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_ollama(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "You are Jarvis, an autonomous Agentic OS running entirely on the "
        "operator's own machine. Address the operator as 'Sir'. Be precise, "
        "never invent tool results, never claim an action succeeded unless the "
        "tool output confirms it, and always state uncertainty explicitly."
    ),
    "planner": (
        "You are Jarvis's planner. Decompose the operator's goal into a JSON list "
        "of subtasks. Each subtask: {id, agent, objective, tools, depends_on}. "
        "Return raw JSON only. Never exceed 8 subtasks."
    ),
    "overseer": (
        "You are the Overseer, an independent watchdog judge. Given a fault "
        "report and layer diagnostics, return raw JSON only: "
        "{fault, layer, root_cause, fix_plan, confidence, needs_manual_approval}."
    ),
    "reviewer": (
        "You are Jarvis's self-reviewer. Judge whether a proposed change is safe, "
        "minimal and correct. Return raw JSON only: "
        "{verdict, risks, verdict_reason}."
    ),
    "language": "You are a translator. Return only the translated text.",
    "builder": (
        "You are Jarvis's software builder. Emit complete, runnable source code "
        "inside a single fenced code block. No prose outside the block."
    ),
}


class OllamaBackend:
    """Local Ollama over its HTTP API. Loopback only, enforced by the client."""

    name = "ollama"

    def __init__(self, base_url: str, timeout: float = 120.0,
                 temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    # ------------------------------------------------------------ plumbing
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"Ollama unreachable at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, TimeoutError) as exc:
            raise LLMUnavailable(f"Malformed Ollama response: {exc}") from exc

    def _get(self, path: str) -> Any:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"Ollama unreachable at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, TimeoutError) as exc:
            raise LLMUnavailable(f"Malformed Ollama response: {exc}") from exc

    # ---------------------------------------------------------------- api
    def list_models(self) -> list[str]:
        data = self._get("/api/tags")
        if isinstance(data, dict):
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return []

    def chat(self, model: str, messages: list[dict[str, str]],
             tools: list[dict[str, Any]] | None = None,
             temperature: float | None = None,
             fmt: str | None = None) -> LLMReply:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
            },
        }
        if tools:
            payload["tools"] = tools
        if fmt:
            payload["format"] = fmt
        data = self._post("/api/chat", payload)
        message = (data.get("message") or {}) if isinstance(data, dict) else {}
        return LLMReply(
            text=str(message.get("content") or ""),
            model=model,
            provider=self.name,
            stats={
                "prompt_tokens": (data or {}).get("prompt_eval_count"),
                "eval_tokens": (data or {}).get("eval_count"),
                "total_duration_ms": (data or {}).get("total_duration"),
            },
        )

    def embed(self, model: str, text: str) -> list[float]:
        data = self._post("/api/embeddings", {"model": model, "prompt": text})
        if isinstance(data, dict):
            return list(data.get("embedding") or [])
        return []


class LLMClient:
    """Provider-agnostic front door with an Ollama-only body."""

    def __init__(self, config, gate=None, audit=None) -> None:
        self.config = config
        self.gate = gate
        self.audit = audit
        self._lock = threading.RLock()
        self._backends: dict[str, Any] = {}
        self._external: dict[str, dict[str, Any]] = {}
        self._approved_once: set[str] = set()
        self._install_ollama()

    # -------------------------------------------------------------- backend
    def _install_ollama(self) -> None:
        llm = self.config.get_dict("llm")
        backend = OllamaBackend(
            base_url=str(llm.get("base_url", "http://127.0.0.1:11434")),
            timeout=float(llm.get("timeout_seconds", 120)),
            temperature=float(llm.get("temperature", 0.2)),
        )
        self._backends["ollama"] = backend
        self.ollama = backend

    # --------------------------------------------------------------- checks
    def assert_local(self, url: str) -> None:
        """Reject any non-loopback host. No approval can bypass this."""
        host = (urlparse(url).hostname or "").lower()
        allowed = {
            str(h).lower()
            for h in self.config.get("llm.allowed_local_hosts", list(LOCAL_HOSTS))
        } | LOCAL_HOSTS
        if host not in allowed:
            raise ExternalAccessDenied(
                f"Non-local AI endpoint '{host}' is not permitted. Jarvis OS is "
                "locked to local Ollama. External access requires the operator's "
                "explicit manual approval each time."
            )

    def available(self) -> bool:
        try:
            self.assert_local(str(self.config.get("llm.base_url", "")))
            return bool(self.ollama.list_models())
        except (ExternalAccessDenied, LLMUnavailable, Exception):
            return False

    def local_models(self) -> list[str]:
        try:
            return sorted(self.ollama.list_models())
        except LLMUnavailable:
            return []

    def register_external_provider(self, provider: str, adapter: Any,
                                   endpoint: str, purpose: str = "") -> bool:
        """Attach a non-Ollama backend - only with live manual approval.

        ``remember_grants`` is deliberately ignored here: the operator's rule is
        "explicit manual permission each time".
        """
        with self._lock:
            if str(self.config.get("llm.provider", "ollama")) != "ollama":
                return False
            self.assert_local(endpoint)  # raises if endpoint is off-box
            decision = self._manual_approval(
                "external_ai",
                {
                    "provider": provider,
                    "endpoint": endpoint,
                    "purpose": purpose or "unspecified",
                    "note": "This connects Jarvis OS to a third-party AI API. "
                            "Approve only if you intend to.",
                },
            )
            if not decision:
                return False
            self._external[provider] = {"adapter": adapter, "endpoint": endpoint}
            self._approved_once.add(provider)
            if self.audit:
                self.audit.record(
                    "external_ai", f"register:{provider}", "approved",
                    {"endpoint": endpoint, "purpose": purpose},
                )
            return True

    def unregister_external_provider(self, provider: str) -> bool:
        with self._lock:
            removed = self._external.pop(provider, None)
            self._approved_once.discard(provider)
        if removed and self.audit:
            self.audit.record("external_ai", f"unregister:{provider}", "revoked")
        return bool(removed)

    def _manual_approval(self, category: str, detail: dict[str, Any]) -> bool:
        if self.gate is None:
            return False
        return self.gate.approve(
            category,
            detail.get("prompt") or f"Jarvis requests permission: {category}",
            detail=detail,
            source="llm_client",
        )

    # -------------------------------------------------------------- chat api
    def chat(self, prompt: str, system: str | None = None,
             persona: str = "default", tools: list[ToolSpec] | None = None,
             model: str | None = None, temperature: float | None = None,
             fmt: str | None = None, history: list[dict[str, str]] | None = None,
             fallback: str = "") -> LLMReply:
        sys_prompt = system or SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS["default"])
        if self.config.get("honorific") and sys_prompt is SYSTEM_PROMPTS.get("default", ""):
            sys_prompt = f"{sys_prompt} Address the operator as '{self.config.honorific()}'."
        messages = [{"role": "system", "content": sys_prompt}]
        for item in history or []:
            messages.append(item)
        messages.append({"role": "user", "content": prompt})

        chosen = model or self._model_for(persona)
        payload_tools = [t.to_ollama() for t in tools] if tools else None
        try:
            return self.ollama.chat(chosen, messages, payload_tools, temperature, fmt)
        except LLMUnavailable as exc:
            if self.audit:
                self.audit.record("llm", "chat", "unavailable",
                                  {"model": chosen, "error": str(exc), "persona": persona})
            if fallback:
                return LLMReply(text=fallback, model=chosen, provider="fallback",
                                ok=True, error=str(exc), stats={"fallback": True})
            return LLMReply(text="", model=chosen, provider="ollama",
                            ok=False, error=str(exc))

    def _model_for(self, persona: str) -> str:
        llm = self.config.get_dict("llm")
        key = {
            "planner": "planner_model",
            "overseer": "reviewer_model",
            "reviewer": "reviewer_model",
        }.get(persona, "model")
        return str(llm.get(key) or llm.get("model") or "llama3.1:8b")

    def ask_json(self, prompt: str, persona: str = "planner",
                 model: str | None = None) -> dict[str, Any] | None:
        """Ask for structured JSON and salvage the object if the model adds prose."""
        reply = self.chat(
            prompt,
            persona=persona,
            model=model,
            temperature=0.0,
            fmt="json",
            fallback="{}",
        )
        return extract_json(reply.text)

    def embed(self, text: str) -> list[float]:
        """Offline embeddings. Falls back to the local hashing embedder."""
        model = str(self.config.get("llm.embed_model", "nomic-embed-text"))
        try:
            self.assert_local(str(self.config.get("llm.base_url", "")))
            vector = self.ollama.embed(model, text)
            if vector:
                return vector
        except (ExternalAccessDenied, LLMUnavailable):
            pass
        from .memory import local_embed
        return local_embed(text)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON recovery from a model response."""
    if not text:
        return None
    candidate = text.strip()
    fence = candidate.find("```")
    if fence != -1:
        rest = candidate[fence + 3:]
        if rest[:4].lower().startswith("json"):
            rest = rest[4:]
        end = rest.find("```")
        candidate = (rest[:end] if end != -1 else rest).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                parsed = json.loads(candidate[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}
            except json.JSONDecodeError:
                continue
    return None