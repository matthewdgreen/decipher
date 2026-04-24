from __future__ import annotations

import re
import unicodedata


_NON_ALPHA_RE = re.compile(r"[^a-z]")


def normalize_text(text: str, language: str) -> str:
    """Normalize text into a flat lowercase ``a-z`` stream."""
    lang = (language or "en").strip().lower()
    text = text.lower()
    if lang == "de":
        text = (
            text.replace("ä", "a")
            .replace("ö", "o")
            .replace("ü", "u")
            .replace("ß", "ss")
        )
    elif lang in {"fr", "it", "la"}:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return _NON_ALPHA_RE.sub("", text)
