"""LLM reader library: rank candidate plaintext excerpts by language quality.

Promoted from ``scripts/research/copiale/rank_candidate_texts_with_llm.py`` into
a reusable library (improvement program Phase 4a, item 4.1). The reader asks a
model to judge damaged candidate plaintext excerpts purely as text quality and
return structured per-candidate scores plus a ranking.

Input firewall (HARD RULE)
--------------------------
The prompt the model sees contains ONLY:

* neutral positional candidate ids (``C01``, ``C02`` ...), assigned here so the
  real ``candidate_id`` — which may encode a null mask (``mask:S1,S2``) or an
  edit set (``edits:S3:a->b``) — never reaches the model;
* the candidate text excerpt (truncated to the configured budget);
* the language name.

No ground truth, no solver scores, no filenames/test ids, no mask/edit
provenance ever enters the prompt. :func:`reader_prompt_for` returns the exact
prompt text so tests can leak-check it, and :func:`assert_no_ground_truth_leak`
is a ready-made assertion for that check.

The reader never crashes the caller: parse failures retry once (bounded by
``config.max_calls``) and then fall back to marking every candidate
``unscored``; provider errors are captured on :attr:`ReaderResult.error`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agent.model_provider import (
    AgentModelProvider,
    ModelProviderError,
    ModelResponse,
    ModelUsage,
    TextBlock,
    canonical_provider,
    estimate_provider_cost,
    make_model_provider,
)

DEFAULT_READER_PROVIDER = "openai"
DEFAULT_READER_MODEL = "gpt-5.6-luna"

_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "la": "Latin",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
}

READER_SYSTEM = (
    "You rank damaged candidate plaintext excerpts by text quality. Return only "
    "the requested JSON object. Do not use tools. Do not discuss or infer "
    "metadata you were not shown."
)


@dataclass(frozen=True)
class LLMReaderConfig:
    """Configuration for one reader ranking pass.

    ``max_calls`` bounds the total number of model requests (initial call plus
    retries): the default of 2 permits exactly one retry on a parse failure.
    ``max_candidates`` and ``max_chars_per_candidate`` are the budget caps and
    are applied in :func:`reader_prompt_for`, so their effect is visible in the
    prompt text the tests inspect.
    """

    provider: str = DEFAULT_READER_PROVIDER
    model: str = DEFAULT_READER_MODEL
    max_candidates: int = 12
    max_chars_per_candidate: int = 700
    max_calls: int = 2
    language: str = "de"
    # Reasoning-tier models (e.g. gpt-5.6-luna) spend part of the output budget
    # on reasoning tokens before the JSON, so a tight ceiling truncates the
    # structured answer intermittently. 4000 leaves headroom for a full 12-way
    # rating plus reasoning; parse-retry still covers the rare overflow.
    max_tokens: int = 4000

    def language_name(self) -> str:
        code = (self.language or "").strip().lower()
        return _LANGUAGE_NAMES.get(code, (self.language or "").strip() or "the target language")


@dataclass(frozen=True)
class ReaderCandidateScore:
    """One candidate's reader verdict.

    ``index`` is the 0-based position of the candidate in the (already
    truncated) input list, and ``display_id`` is the neutral id shown to the
    model. ``candidate_id`` is the caller's original id, mapped back so the
    caller can act on the reader's pick without re-deriving positions.
    """

    candidate_id: str
    display_id: str
    index: int
    readability: float | None = None
    coherent_clause: bool | None = None
    rationale: str = ""
    rank: int | None = None
    unscored: bool = False


@dataclass
class ReaderResult:
    """Structured result of a reader ranking pass. Never raised, always returned."""

    provider: str
    model: str
    scores: list[ReaderCandidateScore] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    best_candidate_id: str | None = None
    best_index: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    raw_response: str = ""
    parse_ok: bool = False
    calls: int = 0
    unscored: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "display_id": score.display_id,
                    "index": score.index,
                    "readability": score.readability,
                    "coherent_clause": score.coherent_clause,
                    "rationale": score.rationale,
                    "rank": score.rank,
                    "unscored": score.unscored,
                }
                for score in self.scores
            ],
            "ranking": list(self.ranking),
            "best_candidate_id": self.best_candidate_id,
            "best_index": self.best_index,
            "usage": dict(self.usage),
            "parse_ok": self.parse_ok,
            "calls": self.calls,
            "unscored": list(self.unscored),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Firewall-safe candidate extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PromptCandidate:
    display_id: str
    candidate_id: str
    index: int
    text: str


def _get(candidate: Any, key: str) -> Any:
    """Read ``key`` from a CandidatePacket-like object or a plain dict."""
    if isinstance(candidate, dict):
        return candidate.get(key)
    return getattr(candidate, key, None)


def _candidate_original_id(candidate: Any, index: int) -> str:
    raw = _get(candidate, "candidate_id")
    if raw is None or str(raw).strip() == "":
        return f"cand_{index + 1}"
    return str(raw)


def _normalize_text(text: str) -> str:
    """Collapse newlines/runs of whitespace into single spaces."""
    return " ".join(str(text).replace("\n", " ").split())


def _candidate_text(candidate: Any) -> str:
    """Return the firewall-safe visible text for a candidate.

    Sourcing (spec 4.1): ``text`` or ``preview``; for word-repair packets
    (``text`` is ``None``) use the preview plus ``extras["page_previews"]`` when
    present. Plain dicts additionally fall back to ``decryption`` (the null-mask
    probe-row spelling). Full keys / provenance are never read.
    """
    text = _get(candidate, "text")
    if text:
        return _normalize_text(text)

    preview = _get(candidate, "preview")
    parts: list[str] = []
    if preview:
        parts.append(_normalize_text(preview))

    extras = _get(candidate, "extras")
    if isinstance(extras, dict):
        page_previews = extras.get("page_previews")
        if isinstance(page_previews, (list, tuple)):
            for page in page_previews:
                # Word-repair packets carry page_previews as a list of dicts
                # ({"test_id", "preview", "filtered_length"}). Only the preview
                # text is firewall-safe: str()-ing the whole dict would leak the
                # benchmark test_id (and other provenance) into the prompt.
                if isinstance(page, dict):
                    page_text = page.get("preview")
                    if page_text:
                        parts.append(_normalize_text(str(page_text)))
                elif page:
                    parts.append(_normalize_text(str(page)))
        elif page_previews:
            parts.append(_normalize_text(str(page_previews)))

    if parts:
        return " ".join(part for part in parts if part).strip()

    # Plain-dict fallback: probe rows store the candidate plaintext as
    # ``decryption``. Never used for CandidatePacket objects (packets keep full
    # decryptions out; only text/preview travel).
    if isinstance(candidate, dict):
        decryption = candidate.get("decryption")
        if decryption:
            return _normalize_text(decryption)
    return ""


def _prepare_candidates(
    candidates: Sequence[Any],
    config: LLMReaderConfig,
) -> list[_PromptCandidate]:
    """Apply the budget caps and assign neutral display ids.

    Candidates with no visible text are dropped (they can never be scored from
    text). ``max_candidates`` truncates the list; ``max_chars_per_candidate``
    truncates each excerpt. Both caps are applied here so their effect shows up
    in :func:`reader_prompt_for` output.
    """
    prepared: list[_PromptCandidate] = []
    max_chars = max(0, int(config.max_chars_per_candidate))
    for index, candidate in enumerate(candidates):
        text = _candidate_text(candidate)
        if not text:
            continue
        if max_chars:
            text = text[:max_chars]
        display_id = f"C{len(prepared) + 1:02d}"
        prepared.append(
            _PromptCandidate(
                display_id=display_id,
                candidate_id=_candidate_original_id(candidate, index),
                index=len(prepared),
                text=text,
            )
        )
        if len(prepared) >= max(1, int(config.max_candidates)):
            break
    return prepared


def _build_prompt(prompt_candidates: Sequence[_PromptCandidate], config: LLMReaderConfig) -> str:
    language = config.language_name()
    blocks = [f"{candidate.display_id}:\n{candidate.text}" for candidate in prompt_candidates]
    candidate_block = "\n\n".join(blocks)
    return (
        "You will be given several candidate plaintext excerpts. They are all "
        f"damaged outputs from a cipher-solving process, expected to be {language}.\n\n"
        f"Task: judge each candidate purely as {language} text quality and rank "
        "them. Prefer candidates that read most like coherent prose across whole "
        "phrases and sentences.\n\n"
        "Be careful about the word-island trap: a candidate should not score "
        "highly merely because it contains a few visible real words. A good local "
        "word only counts when the surrounding words, endings, and phrase "
        "structure also make sense. Penalize candidates where a good word is "
        "embedded in nonsense or reads like disconnected fragments.\n\n"
        "Do not assume the text is archaic, dialectal, or from a specific "
        "historical source. Judge it simply as text quality.\n\n"
        "For each candidate return: readability (0-10 integer; 10 = fluent, "
        "0 = pure noise), coherent_clause (true if at least one clause reads as "
        "grammatical prose), and a short rationale (<=140 characters).\n\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "candidates": [\n'
        '    {"id": "C01", "readability": 0, "coherent_clause": false, "rationale": "..."}\n'
        "  ],\n"
        '  "ranking": ["C01", "C02"],\n'
        '  "best": "C01"\n'
        "}\n\n"
        "The ranking lists candidate ids best-first (a vote ordering). Include "
        "every candidate id exactly once in candidates and ranking.\n\n"
        "Candidate texts:\n\n"
        f"{candidate_block}"
    )


def reader_prompt_for(candidates: Sequence[Any], config: LLMReaderConfig) -> str:
    """Return the exact user-message prompt the reader would send.

    Exposed for leak-checking: the returned string is what
    :func:`rank_candidates` puts on the wire (budget caps applied, neutral ids
    only). See :func:`assert_no_ground_truth_leak`.
    """
    return _build_prompt(_prepare_candidates(candidates, config), config)


# ---------------------------------------------------------------------------
# Leak-check helper
# ---------------------------------------------------------------------------


def assert_no_ground_truth_leak(prompt: str, forbidden: Sequence[str]) -> None:
    """Assert none of the ``forbidden`` strings appear in ``prompt``.

    Comparison is case-insensitive and whitespace-insensitive so a leak that is
    re-spaced or re-cased is still caught. Intended for tests: pass ground-truth
    text, solver-score strings, test ids, filenames, and mask/edit labels.
    """
    haystack = "".join(str(prompt).split()).lower()
    for needle in forbidden:
        text = str(needle).strip()
        if not text:
            continue
        collapsed = "".join(text.split()).lower()
        if collapsed and collapsed in haystack:
            raise AssertionError(f"firewall leak: {text!r} appears in reader prompt")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _visible_text(response: ModelResponse) -> str:
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    ).strip()


def _parse_json(text: str) -> Any:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _coerce_readability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(10.0, float(value)))
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
    return None


def _parsed_rows(parsed: Any) -> list[dict[str, Any]] | None:
    if not isinstance(parsed, dict):
        return None
    rows = parsed.get("candidates")
    if not isinstance(rows, list):
        # Accept the legacy "ranking of objects" shape too.
        rows = parsed.get("ranking")
    if not isinstance(rows, list):
        return None
    out = [row for row in rows if isinstance(row, dict)]
    return out or None


def _row_display_id(row: dict[str, Any]) -> str:
    for key in ("id", "candidate_id", "display_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _ranking_from_parsed(parsed: Any, rows: list[dict[str, Any]]) -> list[str]:
    if isinstance(parsed, dict):
        raw = parsed.get("ranking")
        if isinstance(raw, list):
            ordered: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    ordered.append(item)
                elif isinstance(item, dict):
                    ordered.append(_row_display_id(item))
            ordered = [item for item in ordered if item]
            if ordered:
                return ordered
    # Fall back to ordering the scored rows by readability (descending).
    ranked = sorted(
        rows,
        key=lambda row: (
            _coerce_readability(row.get("readability")) is None,
            -(_coerce_readability(row.get("readability")) or 0.0),
        ),
    )
    return [_row_display_id(row) for row in ranked if _row_display_id(row)]


def _valid_parse(parsed: Any, prompt_candidates: Sequence[_PromptCandidate]) -> bool:
    rows = _parsed_rows(parsed)
    if not rows:
        return False
    display_ids = {candidate.display_id for candidate in prompt_candidates}
    matched = {
        _row_display_id(row) for row in rows if _row_display_id(row) in display_ids
    }
    # Require the model to have addressed at least half the candidates by id.
    return len(matched) >= max(1, (len(display_ids) + 1) // 2)


def _build_result_from_parse(
    parsed: Any,
    prompt_candidates: list[_PromptCandidate],
    *,
    provider: str,
    model: str,
    prompt: str,
    raw_response: str,
    usage: ModelUsage,
    cost: float,
    calls: int,
    error: str | None,
) -> ReaderResult:
    by_display = {candidate.display_id: candidate for candidate in prompt_candidates}
    usage_dict = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "estimated_cost_usd": round(cost, 6),
        "calls": calls,
    }

    rows = _parsed_rows(parsed) if _valid_parse(parsed, prompt_candidates) else None
    if rows is None:
        # Retry-exhausted / unparsable: every candidate is unscored.
        scores = [
            ReaderCandidateScore(
                candidate_id=candidate.candidate_id,
                display_id=candidate.display_id,
                index=candidate.index,
                unscored=True,
            )
            for candidate in prompt_candidates
        ]
        return ReaderResult(
            provider=provider,
            model=model,
            scores=scores,
            ranking=[],
            best_candidate_id=None,
            best_index=None,
            usage=usage_dict,
            prompt=prompt,
            raw_response=raw_response,
            parse_ok=False,
            calls=calls,
            unscored=[candidate.candidate_id for candidate in prompt_candidates],
            error=error or "parse_failed",
        )

    row_by_display: dict[str, dict[str, Any]] = {}
    for row in rows:
        display_id = _row_display_id(row)
        if display_id in by_display and display_id not in row_by_display:
            row_by_display[display_id] = row

    display_ranking = _ranking_from_parsed(parsed, rows)
    rank_of: dict[str, int] = {}
    for position, display_id in enumerate(display_ranking, start=1):
        if display_id in by_display and display_id not in rank_of:
            rank_of[display_id] = position

    scores: list[ReaderCandidateScore] = []
    unscored_ids: list[str] = []
    for candidate in prompt_candidates:
        row = row_by_display.get(candidate.display_id)
        if row is None:
            scores.append(
                ReaderCandidateScore(
                    candidate_id=candidate.candidate_id,
                    display_id=candidate.display_id,
                    index=candidate.index,
                    rank=rank_of.get(candidate.display_id),
                    unscored=True,
                )
            )
            unscored_ids.append(candidate.candidate_id)
            continue
        rationale = str(row.get("rationale") or "")[:140]
        scores.append(
            ReaderCandidateScore(
                candidate_id=candidate.candidate_id,
                display_id=candidate.display_id,
                index=candidate.index,
                readability=_coerce_readability(row.get("readability")),
                coherent_clause=_coerce_bool(row.get("coherent_clause")),
                rationale=rationale,
                rank=rank_of.get(candidate.display_id),
            )
        )

    # candidate-id ranking (mapped back from display ids).
    candidate_ranking = [
        by_display[display_id].candidate_id
        for display_id in display_ranking
        if display_id in by_display
    ]

    best_display = ""
    if isinstance(parsed, dict):
        best_display = _best_display(parsed.get("best"))
    if best_display not in by_display:
        best_display = display_ranking[0] if display_ranking else ""
    best_candidate_id = by_display[best_display].candidate_id if best_display in by_display else None
    best_index = by_display[best_display].index if best_display in by_display else None

    return ReaderResult(
        provider=provider,
        model=model,
        scores=scores,
        ranking=candidate_ranking,
        best_candidate_id=best_candidate_id,
        best_index=best_index,
        usage=usage_dict,
        prompt=prompt,
        raw_response=raw_response,
        parse_ok=True,
        calls=calls,
        unscored=unscored_ids,
        error=error,
    )


def _best_display(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _row_display_id(value)
    return ""


def _add_usage(base: ModelUsage, extra: ModelUsage | None) -> ModelUsage:
    if extra is None:
        return base
    return ModelUsage(
        input_tokens=base.input_tokens + int(extra.input_tokens or 0),
        output_tokens=base.output_tokens + int(extra.output_tokens or 0),
        cache_read_input_tokens=base.cache_read_input_tokens
        + int(extra.cache_read_input_tokens or 0),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: Sequence[Any],
    config: LLMReaderConfig,
    *,
    provider: AgentModelProvider | None = None,
    api_key: str | None = None,
    provider_factory: Callable[..., AgentModelProvider] | None = None,
) -> ReaderResult:
    """Rank ``candidates`` by language quality; never raises.

    Pass a ready ``provider`` (unit tests), or leave it ``None`` to build one
    via ``provider_factory`` (default :func:`make_model_provider`) using
    ``api_key``. Provider errors and parse failures are captured on the result
    rather than raised. Parse failures consume one retry (bounded by
    ``config.max_calls``); if still unparsable every candidate is marked
    ``unscored``.
    """
    resolved_provider = canonical_provider(config.provider)
    prompt_candidates = _prepare_candidates(candidates, config)
    prompt = _build_prompt(prompt_candidates, config)

    if not prompt_candidates:
        return ReaderResult(
            provider=resolved_provider,
            model=config.model,
            prompt=prompt,
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "estimated_cost_usd": 0.0,
                "calls": 0,
            },
            error="no_candidates",
        )

    client = provider
    if client is None:
        factory = provider_factory or make_model_provider
        try:
            client = factory(
                provider=resolved_provider,
                api_key=api_key or "",
                model=config.model,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the caller
            return ReaderResult(
                provider=resolved_provider,
                model=config.model,
                scores=[
                    ReaderCandidateScore(
                        candidate_id=candidate.candidate_id,
                        display_id=candidate.display_id,
                        index=candidate.index,
                        unscored=True,
                    )
                    for candidate in prompt_candidates
                ],
                prompt=prompt,
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "calls": 0,
                },
                unscored=[c.candidate_id for c in prompt_candidates],
                error=f"provider_init_failed: {exc}",
            )

    usage_total = ModelUsage()
    calls = 0
    parsed: Any = None
    raw_response = ""
    error: str | None = None
    max_calls = max(1, int(config.max_calls))

    for _attempt in range(max_calls):
        try:
            response = client.send(
                messages=[{"role": "user", "content": prompt}],
                system=READER_SYSTEM,
                max_tokens=config.max_tokens,
            )
        except ModelProviderError as exc:
            error = f"provider_error: {exc}"
            break
        except Exception as exc:  # noqa: BLE001 — never crash the caller
            error = f"provider_error: {exc}"
            break
        calls += 1
        usage_total = _add_usage(usage_total, response.usage)
        raw_response = _visible_text(response)
        parsed = _parse_json(raw_response)
        if _valid_parse(parsed, prompt_candidates):
            error = None
            break
        error = "parse_failed"

    cost = estimate_provider_cost(
        resolved_provider,
        config.model,
        usage_total.input_tokens,
        usage_total.output_tokens,
        usage_total.cache_read_input_tokens,
    )

    return _build_result_from_parse(
        parsed,
        prompt_candidates,
        provider=resolved_provider,
        model=config.model,
        prompt=prompt,
        raw_response=raw_response,
        usage=usage_total,
        cost=cost,
        calls=calls,
        error=error,
    )
