"""Language-model variant registry.

Removes the hard-coded one-model-per-language rule: every ``models/ngram5_*.bin``
model may declare an optional ``variant`` slug + ``display_label`` in its sidecar
(``*.bin.metadata.json``). This module scans those sidecars and resolves a model
path from an optional variant selection, with a strict precedence that keeps the
default (no env, no variant) resolution byte-identical to the historical
``automated.runner._zenith_native_model_path``.

Resolution precedence (``resolve_language_model``):

  (a) ``DECIPHER_NGRAM_MODEL_<LANG>`` env override -- always wins (back-compat
      pin; returns the path if it exists, else ``None``).
  (b) explicit ``variant`` argument -- exact-slug registry match; a
      :class:`ModelVariantError` (listing available slugs) is raised if absent.
  (c) default -- ``models/ngram5_<lang>.bin`` exactly as before, including the
      English-only proprietary Zenith fallbacks.

The registry never crashes on a missing or malformed sidecar: such a model is
still listed with ``variant=None`` and a filename-derived label.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """One language model discovered in the models directory.

    ``distinct_ngrams``/``chars``/``sha256`` come from the sidecar's
    ``corpus_stats``/``sha256`` and are ``None`` when the sidecar is missing or
    unreadable.
    """

    path: Path
    language: str | None
    variant: str | None
    display_label: str
    distinct_ngrams: int | None
    chars: int | None
    sha256: str | None


class ModelVariantError(ValueError):
    """Raised when an explicit variant slug does not exist for a language."""

    def __init__(self, language: str, requested: str, available: list[str]) -> None:
        self.language = language
        self.requested = requested
        self.available = list(available)
        avail = ", ".join(self.available) if self.available else "(none)"
        super().__init__(
            f"Unknown model variant {requested!r} for language {language!r}. "
            f"Available variants: {avail}."
        )


def _repo_root() -> Path:
    # src/analysis/model_registry.py -> parents[2] == repo root.
    return Path(__file__).resolve().parents[2]


def _models_root(models_dir: str | Path | None) -> Path:
    if models_dir is not None:
        return Path(models_dir).expanduser()
    return _repo_root() / "models"


def _lang_from_filename(bin_path: Path) -> str | None:
    """Best-effort language code from ``ngram5_<lang>[...]`` filenames."""
    stem = bin_path.name
    if stem.endswith(".bin"):
        stem = stem[: -len(".bin")]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "ngram5" and parts[1]:
        return parts[1]
    return None


def _model_info_for(bin_path: Path) -> ModelInfo:
    """Build a :class:`ModelInfo`, tolerating a missing/invalid sidecar."""
    sidecar = bin_path.with_name(bin_path.name + ".metadata.json")
    meta: dict | None = None
    if sidecar.exists():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            meta = None

    if meta is not None:
        language = meta.get("language") or _lang_from_filename(bin_path)
        variant = meta.get("variant")
        display_label = meta.get("display_label") or bin_path.name
        corpus = meta.get("corpus_stats")
        if isinstance(corpus, dict):
            distinct = corpus.get("distinct_seen_5grams")
            chars = corpus.get("normalized_characters")
        else:
            distinct = chars = None
        sha256 = meta.get("sha256")
    else:
        language = _lang_from_filename(bin_path)
        variant = None
        display_label = bin_path.name
        distinct = chars = None
        sha256 = None

    return ModelInfo(
        path=bin_path,
        language=language,
        variant=variant if isinstance(variant, str) else None,
        display_label=str(display_label),
        distinct_ngrams=distinct if isinstance(distinct, int) else None,
        chars=chars if isinstance(chars, int) else None,
        sha256=sha256 if isinstance(sha256, str) else None,
    )


def _scan(models_root: Path) -> list[ModelInfo]:
    if not models_root.exists():
        return []
    return [_model_info_for(p) for p in sorted(models_root.glob("ngram5_*.bin"))]


def list_language_models(
    language: str | None = None,
    models_dir: str | Path | None = None,
) -> list[ModelInfo]:
    """List discovered models, optionally filtered to one language.

    Repo-root anchored (``models/``) with ``expanduser`` on an override. Missing
    or invalid sidecars yield a ``ModelInfo`` with ``variant=None`` and a
    filename-derived label rather than raising.
    """
    infos = _scan(_models_root(models_dir))
    if language is not None:
        lang_key = language.strip().lower()
        infos = [m for m in infos if (m.language or "").lower() == lang_key]
    return infos


def resolve_language_model(
    language: str = "en",
    variant: str | None = None,
    models_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the binary model path for ``language`` under the precedence above.

    With no env override and ``variant=None`` this is byte-identical to the
    historical ``_zenith_native_model_path`` for every language.
    """
    lang_key = (language or "en").strip().lower()

    # (a) env override -- always wins (back-compat pin).
    env_path = os.environ.get(f"DECIPHER_NGRAM_MODEL_{lang_key.upper()}")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None

    models_root = _models_root(models_dir)

    # (b) explicit variant -- exact-slug match within the language.
    if variant is not None:
        for info in _scan(models_root):
            if (info.language or "").lower() == lang_key and info.variant == variant:
                return info.path
        available = sorted(
            {
                info.variant
                for info in _scan(models_root)
                if (info.language or "").lower() == lang_key and info.variant
            }
        )
        raise ModelVariantError(lang_key, variant, available)

    # (c) default -- models/ngram5_<lang>.bin, else the English-only fallbacks.
    candidate = models_root / f"ngram5_{lang_key}.bin"
    if candidate.exists():
        return candidate
    if lang_key != "en":
        return None
    env_path = os.environ.get("DECIPHER_ZENITH_BINARY_MODEL")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None
    repo_root = models_root.parent
    for fallback in (
        repo_root / "other_tools" / "zenith-2026.2" / "zenith-model.array.bin",
        repo_root / "other_tools" / "zenith" / "zenith-model.array.bin",
    ):
        if fallback.exists():
            return fallback
    return None


def active_selection(
    language: str,
    variant: str | None = None,
    models_dir: str | Path | None = None,
) -> dict:
    """Describe which model is active for ``language`` and *why*.

    Returns a model-visible summary (label + variant + source) with NO paths or
    shas. ``source`` is one of ``env`` / ``variant`` / ``default``.
    """
    lang_key = (language or "en").strip().lower()
    if os.environ.get(f"DECIPHER_NGRAM_MODEL_{lang_key.upper()}"):
        return {"source": "env", "variant": None, "label": "environment override"}

    infos = list_language_models(lang_key, models_dir)
    if variant is not None:
        match = next((m for m in infos if m.variant == variant), None)
        return {
            "source": "variant",
            "variant": variant,
            "label": match.display_label if match else None,
        }

    default_name = f"ngram5_{lang_key}.bin"
    match = next((m for m in infos if m.path.name == default_name), None)
    return {
        "source": "default",
        "variant": match.variant if match else None,
        "label": match.display_label if match else default_name,
    }


def format_registry_preflight_line(
    language: str,
    variant: str | None = None,
    models_dir: str | Path | None = None,
) -> str | None:
    """One preflight line: available variants + the active selection.

    Returns ``None`` when the language has no discoverable models (so callers can
    omit the line entirely). No paths/shas are included.
    """
    infos = list_language_models(language, models_dir)
    labelled = [m for m in infos if m.variant]
    if not labelled:
        return None
    active = active_selection(language, variant, models_dir)
    active_variant = active.get("variant")
    parts = []
    for m in labelled:
        marker = " (active)" if m.variant == active_variant else ""
        parts.append(f"{m.variant}{marker}")
    active_desc = (
        f"{active_variant} — {active.get('label')}"
        if active_variant
        else f"{active.get('label')} [{active.get('source')}]"
    )
    return (
        f"Language models for {language}: {', '.join(parts)}. "
        f"Active: {active_desc}. "
        "Agents may switch with act_set_model_variant / inspect via "
        "observe_language_models."
    )
