"""All non-LLM triage strategies.

Each strategy is a pure function over the cached structural metrics; none
requires LLM calls.  The decoded_text strategies require the labeler to have
run (they use decoded_text from run_automated).

Strategies implemented
----------------------
  baseline_ngram              Current behavior — rank by original_rank.
  score_only                  Rank by the raw screen_transform_candidates score.
  dict_rate_weighted          score + alpha * dict_rate metric.
  consecutive_dict_words      Bonus for runs of k+ dictionary words in decoded text.
  ic_adjusted                 Penalize candidates whose post-transform IC is far
                              from the reference language IC.
  cluster_dedupe              Deduplicate by token_order_hash, then rank by score.
  mmr_diversity               Maximal Marginal Relevance: balance score vs diversity.
  delta_only                  Rank by delta_vs_identity (how much better than identity).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from triage.strategies.base import Strategy, register
from triage.types import CandidateRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENGLISH_WORDS: set[str] | None = None


def _load_dict(language: str = "en") -> set[str]:
    global _ENGLISH_WORDS
    if language == "en" and _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS
    # Lazy import to avoid circular deps at module load time.
    try:
        from analysis.dictionary import load_word_set, get_dictionary_path
        words = load_word_set(get_dictionary_path(language))
    except Exception:  # noqa: BLE001
        words = set()
    if language == "en":
        _ENGLISH_WORDS = words
    return words


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two metric dicts (sparse vectors)."""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v ** 2 for v in a.values()))
    nb = math.sqrt(sum(v ** 2 for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@register
class BaselineNgramStrategy(Strategy):
    """Rank by original_rank — reproduces the current agent-visible ordering."""

    name = "baseline_ngram"

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        return [c.candidate_id for c in sorted(candidates, key=lambda c: c.original_rank)]


@register
class ScoreOnlyStrategy(Strategy):
    """Rank by the raw structural score from screen_transform_candidates."""

    name = "score_only"

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        return [
            c.candidate_id
            for c in sorted(candidates, key=lambda c: c.score, reverse=True)
        ]


@register
class DeltaOnlyStrategy(Strategy):
    """Rank by delta_vs_identity — how much better than leaving the cipher alone."""

    name = "delta_only"

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        return [
            c.candidate_id
            for c in sorted(candidates, key=lambda c: c.delta_vs_identity, reverse=True)
        ]


@register
class DictRateWeightedStrategy(Strategy):
    """Combine structural score with the dict_rate metric (alpha=0.3)."""

    name = "dict_rate_weighted"
    ALPHA = 0.3

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        def _key(c: CandidateRecord) -> float:
            dr = c.metrics.get("dict_rate", 0.0)
            return c.score + self.ALPHA * dr

        return [c.candidate_id for c in sorted(candidates, key=_key, reverse=True)]


@register
class IcAdjustedStrategy(Strategy):
    """Prefer candidates whose post-transform IC is closer to English reference (0.067)."""

    name = "ic_adjusted"
    REFERENCE_IC = 0.067
    WEIGHT = 2.0

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        def _key(c: CandidateRecord) -> float:
            ic = c.metrics.get("ic", 0.0)
            ic_penalty = abs(ic - self.REFERENCE_IC)
            return c.score - self.WEIGHT * ic_penalty

        return [c.candidate_id for c in sorted(candidates, key=_key, reverse=True)]


@register
class ClusterDedupeStrategy(Strategy):
    """Keep one representative per token_order_hash cluster, then rank by score."""

    name = "cluster_dedupe"

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        # Sort by score descending so each hash's best representative is kept.
        by_score = sorted(candidates, key=lambda c: c.score, reverse=True)
        seen: set[str] = set()
        deduped: list[CandidateRecord] = []
        remainder: list[CandidateRecord] = []
        for c in by_score:
            h = c.token_order_hash or c.candidate_id
            if h not in seen:
                seen.add(h)
                deduped.append(c)
            else:
                remainder.append(c)
        # Deduped representatives first (score order), then duplicates last.
        return [c.candidate_id for c in deduped + remainder]


@register
class MmrDiversityStrategy(Strategy):
    """Maximal Marginal Relevance: balance score and diversity (lambda=0.5).

    Uses the metrics dict as a feature vector for similarity.
    """

    name = "mmr_diversity"
    LAMBDA = 0.5

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        if not candidates:
            return []

        # Normalize scores to [0, 1].
        scores = [c.score for c in candidates]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        norm: dict[str, float] = {
            c.candidate_id: (c.score - min_s) / rng for c in candidates
        }
        metrics: dict[str, dict[str, float]] = {
            c.candidate_id: c.metrics for c in candidates
        }
        remaining = list(candidates)
        ranked: list[str] = []

        while remaining:
            if not ranked:
                # Pick highest normalized score first.
                best = max(remaining, key=lambda c: norm[c.candidate_id])
            else:
                # MMR: max over (lambda * relevance - (1-lambda) * max_sim_to_ranked)
                def _mmr(c: CandidateRecord) -> float:
                    rel = norm[c.candidate_id]
                    max_sim = max(
                        _cosine_sim(metrics[c.candidate_id], metrics[rid])
                        for rid in ranked
                    )
                    return self.LAMBDA * rel - (1 - self.LAMBDA) * max_sim

                best = max(remaining, key=_mmr)

            ranked.append(best.candidate_id)
            remaining.remove(best)

        return ranked


@register
class ConsecutiveDictWordsStrategy(Strategy):
    """Boost candidates whose decoded text has runs of k+ consecutive dict words.

    Requires decoded_text from the labeler.  Falls back to score_only when
    decoded_text is unavailable.
    """

    name = "consecutive_dict_words"
    requires_decoded_text = True
    K = 3        # minimum run length for a bonus

    def _consecutive_run_bonus(self, text: str, word_set: set[str]) -> float:
        words = text.upper().split()
        if not words:
            return 0.0
        max_run = 0
        run = 0
        for w in words:
            if w in word_set:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return float(max(0, max_run - self.K + 1))

    def rank(
        self, candidates: list[CandidateRecord], case_metadata: dict[str, Any]
    ) -> list[str]:
        language = case_metadata.get("language", "en")
        word_set = _load_dict(language)

        def _key(c: CandidateRecord) -> tuple[float, float]:
            bonus = 0.0
            if c.decoded_text and word_set:
                bonus = self._consecutive_run_bonus(c.decoded_text, word_set)
            return (bonus, c.score)

        return [c.candidate_id for c in sorted(candidates, key=_key, reverse=True)]
