"""UniversalLanguageLayer - offline language detection and translation.

Policy: **nothing leaves the operator's machine.** Translation backends are
attempted in order and every one of them is local:

1. **Argos Translate** - offline OpenNMT models, preferred when installed.
2. **NLLB-200** via ``transformers`` - offline HuggingFace weights on disk.
3. **Passthrough** - honest degradation: the text is returned unchanged and
   Jarvis says so, rather than pretending it translated.

There is deliberately no network code path here. No Google Translate, no
DeepL, no cloud MT API - the operator's instruction was explicit.

Detection is script/lexicon based (works with zero dependencies) and is refined
with ``langdetect``/``langid`` if those happen to be present.
"""

from __future__ import annotations

import importlib.util
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

# ------------------------------------------------------------------- scripts
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "ta": ((0x0B80, 0x0BFF),),            # Tamil
    "hi": ((0x0900, 0x097F),),            # Devanagari
    "bn": ((0x0980, 0x09FF),),            # Bengali
    "te": ((0x0C00, 0x0C7F),),            # Telugu
    "kn": ((0x0C80, 0x0CFF),),            # Kannada
    "ml": ((0x0D00, 0x0D7F),),            # Malayalam
    "gu": ((0x0A80, 0x0AFF),),            # Gujarati
    "pa": ((0x0A00, 0x0A7F),),            # Gurmukhi
    "or": ((0x0B00, 0x0B7F),),            # Oriya
    "mr": ((0x0900, 0x097F),),            # Marathi (Devanagari)
    "ar": ((0x0600, 0x06FF),),            # Arabic
    "he": ((0x0590, 0x05FF),),            # Hebrew
    "fa": ((0x0600, 0x06FF),),            # Persian
    "ur": ((0x0600, 0x06FF),),            # Urdu
    "ru": ((0x0400, 0x04FF),),            # Cyrillic
    "el": ((0x0370, 0x03FF),),            # Greek
    "th": ((0x0E00, 0x0E7F),),            # Thai
    "ka": ((0x10A0, 0x10FF),),            # Georgian
    "hy": ((0x0530, 0x058F),),            # Armenian
    "zh": ((0x4E00, 0x9FFF),),            # Han
    "ja": ((0x3040, 0x30FF),),            # Kana
    "ko": ((0xAC00, 0xD7AF),),            # Hangul
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "ta": "Tamil", "hi": "Hindi", "bn": "Bengali", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi",
    "or": "Odia", "mr": "Marathi", "ar": "Arabic", "he": "Hebrew", "fa": "Persian",
    "ur": "Urdu", "ru": "Russian", "el": "Greek", "th": "Thai", "ka": "Georgian",
    "hy": "Armenian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
    "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "id": "Indonesian", "tr": "Turkish", "vi": "Vietnamese", "nl": "Dutch",
    "pl": "Polish", "uk": "Ukrainian", "ms": "Malay", "sv": "Swedish",
}

_COMMON_WORDS: dict[str, set[str]] = {
    "en": {"the", "is", "and", "of", "to", "in", "you", "sir", "jarvis", "please",
           "with", "for", "not", "this", "that", "have", "will", "can", "what"},
    "ta": {"அது", "இது", "என்று", "ஆம்", "நீங்கள்", "நான்", "என்ன", "எப்படி",
           "செய்", "ஆண்ட", "பின்", "ஒரு", "உள்ள", "வேண்டும்", "நன்றி", "சார்"},
    "hi": {"और", "का", "की", "है", "में", "से", "यह", "वह", "कि", "कर", "सर",
           "क्या", "कैसे", "मुझे", "देता"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "se", "por", "con", "para"},
    "fr": {"le", "la", "de", "et", "est", "un", "une", "pour", "dans", "avec", "vous"},
    "de": {"der", "die", "und", "ist", "nicht", "sie", "mit", "für", "das", "von"},
}


@dataclass(slots=True)
class Detection:
    language: str
    confidence: float
    method: str = "script"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = LANGUAGE_NAMES.get(self.language, self.language)

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "name": self.name,
                "confidence": round(self.confidence, 3), "method": self.method}


class TranslationUnavailable(RuntimeError):
    """No offline translation backend is installed."""


class UniversalLanguageLayer:
    def __init__(self, config, llm=None, audit=None) -> None:
        self.config = config
        self.llm = llm
        self.audit = audit
        self.target = str(config.get("language.target_language", "en"))
        # Operator reply language. Read from the language section like every
        # other key; an unnamespaced lookup silently fell back to the "ta"
        # default, so config.json could never actually change it.
        self.preferred = str(config.get("language.preferred_language", "ta"))
        self.honorific = config.honorific()
        self._lock = threading.RLock()
        self._argos_installed = False
        self._nllb: dict[str, Any] = {}
        self._cache: dict[tuple[str, str], str] = {}
        self.backend = "none"
        self._detect_backends()

    # ------------------------------------------------------------- backends
    @staticmethod
    def _module_present(name: str) -> bool:
        """Is the module importable *without importing it*?

        Importing ``transformers`` on a cold interpreter costs over a minute
        because it drags in torch. Doing that inside the kernel constructor
        stalls boot long enough for the watchdog to look like a hang, so
        availability is probed through the import system instead.
        """
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def _detect_backends(self) -> None:
        """Probe which offline backends exist. Never imports a heavy backend.

        Backend selection is per language pair, not global: Argos being
        importable says nothing about whether it can translate *this* pair.
        Argos ships no Tamil model at all, so a global "argos" choice silently
        passed Tamil through untranslated. The chain is now resolved in
        ``_chain_for`` against the pairs actually installed on disk.
        """
        wanted = str(self.config.get("language.backend", "auto"))
        self._argos_installed = self._module_present("argostranslate")
        self._pkg = None
        self._argos_translate = None
        self._pairs: set[tuple[str, str]] | None = None
        self._nllb_ready = bool(wanted in ("auto", "nllb")
                                and self._module_present("transformers")
                                and self._nllb_weights_present())
        # The local Ollama model is always a candidate: it is already running
        # for reasoning, it stays on this machine, and it covers languages
        # Argos has no package for.
        self._ollama_ready = bool(wanted in ("auto", "ollama") and self.llm is not None)
        self.backend = "ollama" if self._ollama_ready else (
            "argos" if self._argos_installed else
            "nllb" if self._nllb_ready else "none")

    _WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
    _MIN_WEIGHT_BYTES = 512 * 1024 * 1024

    def _nllb_weights_present(self) -> bool:
        """Are a *complete* set of NLLB weights already on disk?

        ``_nllb_translate`` loads with ``local_files_only=True`` so it can never
        pull weights at runtime. Reporting the backend as ready when only the
        Python packages are installed would advertise a translator that always
        fails.

        Presence of the repository directory is not enough: an interrupted
        download leaves the directory in place while the snapshot is missing
        either the tokenizer or the weights, so the resolved snapshot must
        actually contain config + tokenizer + a plausibly sized weight file.
        """
        name = str(self.config.get("language.nllb_model",
                                   "facebook/nllb-200-distilled-600M"))
        cache = self.config.resolve("language.cache_dir")
        try:
            root = cache / ("models--" + name.replace("/", "--"))
            snapshots = [root / "snapshots"] if (root / "snapshots").is_dir() else []
            snapshots.append(cache / name)
            for base in snapshots:
                candidates = [base] if base.is_dir() and base.name != "snapshots" \
                    else list(base.glob("*"))
                for snap in candidates:
                    if not (snap / "config.json").is_file():
                        continue
                    if not (snap / "tokenizer.json").is_file():
                        continue
                    for weights in self._WEIGHT_FILES:
                        path = snap / weights
                        try:
                            if path.is_file() and path.stat().st_size >= self._MIN_WEIGHT_BYTES:
                                return True
                        except OSError:
                            continue
            return False
        except OSError:
            return False

    def _installed_pairs(self) -> set[tuple[str, str]]:
        """Cached Argos pairs. Probing argostranslate touches disk, not network."""
        if self._pairs is None:
            self._pairs = self._installed_argos_pairs()
        return self._pairs

    def _chain_for(self, source: str, target: str) -> list[str]:
        """Offline backends able to translate ``source`` -> ``target``."""
        chain: list[str] = []
        if self._ensure_argos() and (source, target) in self._installed_pairs():
            chain.append("argos")
        if self._nllb_ready:
            chain.append("nllb")
        if self._ollama_ready:
            chain.append("ollama")
        return chain

    def backend_for(self, source: str, target: str) -> str:
        chain = self._chain_for(source, target)
        return chain[0] if chain else "passthrough"

    def _ensure_argos(self) -> bool:
        """Import Argos on first real use and cache the module handles."""
        if not self._argos_installed:
            return False
        if self._pkg is None:
            try:
                import argostranslate.package as pkg
                from argostranslate import translate as _t

                self._pkg = pkg
                self._argos_translate = _t.translate
            except Exception:  # noqa: BLE001 - Argos is optional
                self._argos_installed = False
                return False
        return True

    def _installed_argos_pairs(self) -> set[tuple[str, str]]:
        """Language pairs already present on disk. Never touches the network."""
        pairs: set[tuple[str, str]] = set()
        if not self._ensure_argos():
            return pairs
        try:
            for pkg_item in self._pkg.get_installed_packages():
                pairs.add((pkg_item.from_code, pkg_item.to_code))
        except Exception:  # noqa: BLE001
            pass
        return pairs

    def install_argos_packages(self, approved: bool = False) -> dict[str, Any]:
        """Download offline translation models. OFF by default, never automatic.

        This is the only method in the language layer that touches the network,
        and it refuses to run unless the operator explicitly approved *this*
        call. Jarvis will not silently fetch anything at boot.
        """
        if not self._ensure_argos():
            return {"ok": False, "error": "argostranslate is not installed"}
        if not approved:
            return {"ok": False, "error":
                    "refused: downloading models requires explicit operator "
                    "approval for this specific request"}
        try:
            self._pkg.update_package_index()
            available = self._pkg.get_available_packages()
            installed = self._installed_argos_pairs()
            installed_targets = {src for src, dst in installed}
            wanted = [p for p in available
                      if p.from_code in {self.preferred, "en", self.target}
                      and (p.from_code, p.to_code) not in installed]
            wanted = [p for p in wanted if p.from_code not in installed_targets] or wanted
            added: list[str] = []
            for pkg_item in wanted:
                try:
                    self._pkg.install_from_path(pkg_item.download())
                    added.append(f"{pkg_item.from_code}->{pkg_item.to_code}")
                except Exception:  # noqa: BLE001
                    continue
            return {"ok": True, "installed": added,
                    "pairs": sorted(f"{a}->{b}" for a, b in self._installed_argos_pairs())}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def missing_pairs(self) -> list[str]:
        """Pairs the operator may want, reported so Jarvis can ask - not fetch."""
        if not self._argos_installed:
            return []
        installed = self._installed_argos_pairs()
        wanted = []
        for src in {self.preferred, "en", self.target}:
            for dst in {self.preferred, "en", self.target}:
                if src != dst and (src, dst) not in installed:
                    wanted.append(f"{src}->{dst}")
        return sorted(set(wanted))

    def available_backends(self) -> list[str]:
        out = []
        if self._argos_installed:
            out.append("argos-translate (offline)")
        if self._nllb_ready:
            out.append("nllb-200 (offline)")
        if self._ollama_ready:
            out.append("ollama-local (offline)")
        if not out:
            out.append("passthrough (no offline model installed)")
        return out

    # ------------------------------------------------------------ detection
    def detect(self, text: str) -> Detection:
        if not text or not text.strip():
            return Detection("en", 0.0, "empty")
        try:  # optional refinement
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0
            best = max(detect_langs(text), key=lambda d: d.prob)
            return Detection(best.lang, float(best.prob), "langdetect")
        except Exception:  # noqa: BLE001
            pass
        return self._detect_by_script(text)

    def _detect_by_script(self, text: str) -> Detection:
        counts: dict[str, int] = {}
        letters = 0
        for ch in text:
            code = ord(ch)
            if not ch.isalpha():
                continue
            letters += 1
            for lang, ranges in SCRIPT_RANGES.items():
                if any(low <= code <= high for low, high in ranges):
                    counts[lang] = counts.get(lang, 0) + 1
                    break
        if counts:
            lang, hits = max(counts.items(), key=lambda kv: kv[1])
            confidence = hits / max(1, letters)
            if lang == "hi":  # disambiguate Hindi/Marathi on lexicon
                words = set(re.findall(r"\w+", text.lower()))
                if words & _COMMON_WORDS.get("mr", set()):
                    lang = "hi"
            return Detection(lang, confidence, "script")

        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return Detection("en", 0.3, "fallback")
        scores = {
            lang: sum(1 for w in words if w in common) / len(words)
            for lang, common in _COMMON_WORDS.items()
        }
        lang, score = max(scores.items(), key=lambda kv: kv[1])
        return Detection(lang if score > 0.05 else "en", score or 0.35, "lexicon")

    def detect_to_english(self, text: str) -> tuple[str, Detection]:
        detection = self.detect(text)
        if detection.language == "en":
            return text, detection
        return self.translate(text, "en"), detection

    # ---------------------------------------------------------- translation
    def translate(self, text: str, target: str | None = None) -> str:
        """Translate offline. Returns ``text`` unchanged if no backend exists."""
        target = target or self.preferred
        if not text or not text.strip():
            return text
        detection = self.detect(text)
        if detection.language == target:
            return text
        key = (text[:512], target)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        translated: str | None = None
        chain = self._chain_for(detection.language, target)
        used = "passthrough"
        for backend in chain:
            try:
                if backend == "argos":
                    translated = self._argos_translate(text, detection.language, target)
                elif backend == "nllb":
                    translated = self._nllb_translate(text, detection.language, target)
                elif backend == "ollama":
                    translated = self._ollama_translate(text, detection.language, target)
            except Exception as exc:  # noqa: BLE001 - never break the reply path
                if self.audit:
                    self.audit.record("language", f"translate:{backend}", "failed",
                                      {"error": str(exc)[:200]})
                translated = None
            if translated:
                used = backend
                break
        if not translated:
            translated = text
            if self.audit:
                self.audit.record("language", "translate", "passthrough",
                                  {"detected": detection.language, "target": target,
                                   "backend": used, "tried": chain})
        with self._lock:
            self._cache[key] = translated
        return translated

    _TRANSLATE_SYSTEM = (
        "You are a translation engine. Translate the user's message into {target}. "
        "Reply with ONLY the translated text: no preamble, no notes, no quotes, "
        "no explanation, and do not address anyone. Preserve every number, "
        "identifier, filename and command exactly as written. If the text is "
        "already in {target}, repeat it unchanged."
    )

    def _ollama_translate(self, text: str, source: str, target: str) -> str | None:
        """Translate with the already-running local Ollama model.

        This is the only offline backend that covers Tamil: Argos publishes no
        ``ta`` package at all. The model is the same local one used for
        reasoning, so nothing leaves the machine and no extra download is
        needed. Temperature is pinned to 0 for faithful translation.
        """
        if not self._ollama_ready or self.llm is None:
            return None
        target_name = LANGUAGE_NAMES.get(target, target)
        reply = self.llm.chat(
            text,
            system=self._TRANSLATE_SYSTEM.format(target=target_name),
            persona="translator",
            temperature=0.0,
        )
        out = (getattr(reply, "text", "") or "").strip()
        if not out:
            return None
        # Models sometimes wrap output in quotes or a fenced block.
        if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'":
            out = out[1:-1].strip()
        if out.startswith("```") and out.endswith("```"):
            out = out[3:-3].strip()
        return out or None

    # NLLB is keyed by ISO 639-3, the language layer detects ISO 639-1.
    _NLLB_CODES: dict[str, str] = {
        "en": "eng", "ta": "tam", "hi": "hin", "bn": "ben", "te": "tel",
        "kn": "kan", "ml": "mal", "gu": "guj", "pa": "pan", "ur": "urd",
        "or": "ory", "mr": "mar", "ar": "arb", "he": "heb", "fa": "pes",
        "de": "deu", "fr": "fra", "es": "spa", "ru": "rus", "zh": "zho",
        "ja": "jpn", "ko": "kor", "pt": "por", "it": "ita", "nl": "nld",
        "tr": "tur", "vi": "vie", "id": "ind", "th": "tha", "pl": "pol",
    }

    @classmethod
    def _iso3(cls, code: str) -> str:
        """ISO 639-1 -> ISO 639-3, which is what NLLB keys its vocabulary on."""
        return cls._NLLB_CODES.get(code, code)

    _LANG_TOKEN = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3}$")

    @classmethod
    def _nllb_lang_ids(cls, tokenizer) -> dict[str, int]:
        """Map ISO 639-3 -> NLLB language-token id, read from the tokenizer.

        NLLB does not use a fixed script suffix: languages that have their own
        script use it (``tam_Taml``, ``hin_Deva``, ``ben_Beng``) while others
        use Latin (``eng_Latn``). Assuming ``_Latn`` silently mistranslates --
        an unknown code resolves to a meaningless id and the model emits the
        wrong language entirely. The tokens are added tokens, so they are only
        reachable through id decoding, not the token->id vocabulary.
        """
        found: dict[str, int] = {}
        top = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if top <= 0:
            return found
        for token_id in range(max(0, top - 1024), top):
            try:
                token = tokenizer.convert_ids_to_tokens(token_id)
            except Exception:  # noqa: BLE001
                continue
            if token and cls._LANG_TOKEN.match(str(token)):
                found[str(token).split("_", 1)[0]] = token_id
        return found

    def _nllb_translate(self, text: str, source: str, target: str) -> str | None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception:  # noqa: BLE001
            return None
        name = str(self.config.get("language.nllb_model", "facebook/nllb-200-distilled-600M"))
        with self._lock:
            if name not in self._nllb:
                cache = self.config.resolve("language.cache_dir")
                cache.mkdir(parents=True, exist_ok=True)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        name, cache_dir=str(cache), local_files_only=True)
                    lang_ids = self._nllb_lang_ids(tokenizer)
                    if not lang_ids:
                        raise RuntimeError("nllb language vocabulary missing")
                    self._nllb[name] = {
                        "tok": tokenizer,
                        "lang_ids": lang_ids,
                        "model": AutoModelForSeq2SeqLM.from_pretrained(
                            name, cache_dir=str(cache), local_files_only=True),
                    }
                    # generation_config carries max_length=200; leaving it set
                    # while also passing max_new_tokens only produces a warning
                    # on every single translation.
                    self._nllb[name]["model"].generation_config.max_length = None
                except Exception:  # noqa: BLE001
                    self._nllb[name] = {}
                    return None
            bundle = self._nllb.get(name) or {}
        if not bundle:
            return None
        try:
            tokenizer, model = bundle["tok"], bundle["model"]
            lang_ids: dict[str, int] = bundle["lang_ids"]
            tgt3 = self._iso3(target)
            if tgt3 not in lang_ids:
                return None
            src3 = self._iso3(source)
            if src3 not in lang_ids:
                return None
            src_code = f"{src3}_Latn"
            tokenizer.src_lang = src_code
            # NLLB is a multilingual model sharing one vocabulary: without an
            # explicit target-language BOS it silently emits whichever language
            # it feels like, so the target must be forced at generate() time.
            forced_bos = lang_ids[tgt3]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=512,
                    num_beams=4,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            return decoded or None
        except Exception as exc:  # noqa: BLE001
            if self.audit:
                self.audit.record("language", "translate:nllb", "failed",
                                  {"error": str(exc)[:200]})
            return None

    # --------------------------------------------------------------- output
    def to_operator_language(self, text: str) -> str:
        """Translate an English response into the operator's preferred language."""
        if not self.config.feature("language_layer", True):
            return self.prefix(text)
        return self.prefix(self.translate(text, self.preferred))

    def prefix(self, text: str) -> str:
        """Every utterance to the operator starts with the honorific."""
        honorific = self.honorific
        body = text or ""
        stripped = body.lstrip()
        if stripped.lower().startswith(honorific.lower()):
            return body
        return f"{honorific}, {stripped}"

    def process_inbound(self, text: str) -> tuple[str, Detection]:
        """Operator input -> English for reasoning + detection result."""
        return self.detect_to_english(text)

    def process_outbound(self, text: str) -> str:
        """Reasoning output -> operator's language, honorific-prefixed."""
        return self.to_operator_language(text)

    def reply(self, text: str) -> str:
        """Answer the operator in their own language, as one round trip.

        Detects the language the operator wrote in and translates it to English
        so all reasoning happens in a single language, asks the local model,
        then translates the answer back into the operator's preferred language
        and prefixes the honorific. This composes the pieces that already exist
        here rather than reimplementing detection or translation, so there is
        one code path to keep correct.

        It never raises. A missing or dead backend degrades to echoing the
        understood English text through the same outbound path, which still
        translates and still carries the honorific - the interface degrades, it
        does not disappear.
        """
        source = (text or "").strip()
        if not source:
            return self.process_outbound("I did not catch that, Sir. Please repeat.")
        english, detection = self.process_inbound(source)
        if not self._ollama_ready or self.llm is None:
            return self.process_outbound(english)
        try:
            answer = self.llm.chat(english, persona="default")
        except Exception as exc:  # noqa: BLE001
            if self.audit:
                self.audit.record("language", "reply", "failed",
                                  {"error": str(exc)[:200],
                                   "detected": detection.language,
                                   "confidence": round(detection.confidence, 3)},
                                  actor="language")
            return self.process_outbound(english)
        out = (getattr(answer, "text", "") or "").strip()
        return self.process_outbound(out or english)

    def wrap_plan(self, steps: Iterable[str], title: str = "Plan") -> str:
        body = "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
        return self.process_outbound(f"{title}:\n{body}")

    def status(self) -> str:
        detection_backend = "script+lexicon"
        try:
            import langdetect  # noqa: F401
            detection_backend = "langdetect+script"
        except Exception:  # noqa: BLE001
            pass
        target_name = LANGUAGE_NAMES.get(self.preferred, self.preferred)
        return (f"language layer: target=en, reply={target_name}"
                f" | translate={self.backend_for('en', self.preferred)}"
                f" | detect={detection_backend} | offline=True")

    def install_hint(self) -> str:
        return (
            "Offline translation models are optional. Install for full bilingual support:\n"
            "  pip install argostranslate\n"
            "  python -c \"import argostranslate.package as p; p.update_package_index();\n"
            "    [p.install_from_path(x.download()) for x in p.get_available_packages()\n"
            "     if x.from_code in ('ta','hi','en') and x.to_code in ('en','ta')]\"\n"
            "Or for NLLB (larger, better quality):\n"
            "  pip install transformers torch"
        )