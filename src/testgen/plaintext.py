from __future__ import annotations

import unicodedata

from testgen.spec import TestSpec

_DEFAULT_GENERATION_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.1-flash-lite-preview",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "la": "Latin",
}


class PlaintextGenerator:
    """Generates novel plaintext passages using a low-cost LLM.

    The default stays on Claude Haiku for backward compatibility, but callers
    can select OpenAI or Gemini when running cross-provider experiments.
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: str = "anthropic",
        model: str | None = None,
    ) -> None:
        from agent.model_provider import canonical_provider, make_model_provider

        self.provider = canonical_provider(provider)
        self.model = model or _DEFAULT_GENERATION_MODELS[self.provider]
        self._api = make_model_provider(
            provider=self.provider,
            api_key=api_key,
            model=self.model,
        )

    def generate(self, spec: TestSpec) -> str:
        """Return normalised uppercase prose matching the spec."""
        prompt = self._build_prompt(spec)
        response = self._api.send(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        raw = "\n".join(
            block.text for block in response.content if block.type == "text"
        )
        return self._normalize(raw)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, spec: TestSpec) -> str:
        lang_name = _LANGUAGE_NAMES.get(spec.language, spec.language.upper())
        topic_hint = f" on the topic of {spec.topic}" if spec.topic != "general" else ""
        return (
            f"Write approximately {spec.approx_length} words of original "
            f"{lang_name} prose{topic_hint}. "
            "The text must NOT be from any famous, well-known, or commonly quoted "
            "work — it must be entirely novel content you are composing now. "
            "Use plain continuous prose with normal vocabulary. "
            "Do not include any headings, titles, explanations, or commentary — "
            "output only the prose itself, nothing else."
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Uppercase, strip diacritics, keep only A-Z words of length >= 2."""
        # NFD decompose → remove combining marks → ASCII letters only
        nfd = unicodedata.normalize("NFD", text)
        base_only = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        words = []
        for token in base_only.upper().split():
            clean = "".join(c for c in token if "A" <= c <= "Z")
            if len(clean) >= 2:
                words.append(clean)
        return " ".join(words)
