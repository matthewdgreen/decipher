"""Population generator for the transform triage evaluation framework.

Generates synthetic transposition+substitution and transposition+homophonic
test cases from cached plaintexts.  Never makes LLM API calls — if a
plaintext is not already in the cache the case is silently skipped.

Usage
-----
    from triage.population import generate_population, PIPELINE_TEMPLATES

    n = generate_population(
        sweep=default_sweep(),
        output_path=Path("eval/artifacts/populations/default.jsonl"),
        cache_dir=Path("testgen_cache"),
    )
    print(f"Generated {n} cases")
"""
from __future__ import annotations

import hashlib
import json
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These imports come from src/ on the PYTHONPATH.
from analysis.transformers import TransformPipeline, apply_transform_pipeline, make_inverse_input_for_pipeline
from testgen.cache import PlaintextCache
from testgen.spec import TestSpec

from triage.types import PopulationEntry


# ---------------------------------------------------------------------------
# Pipeline templates
# ---------------------------------------------------------------------------

def _make_pipeline(template: str, columns: int, rows: int) -> dict[str, Any]:
    """Return a raw pipeline dict for the given template and grid size."""
    base: dict[str, Any] = {"columns": columns, "rows": rows}
    if template == "route_columns_down":
        return {**base, "steps": [{"name": "RouteRead", "data": {"route": "columns_down"}}]}
    if template == "route_columns_up":
        return {**base, "steps": [{"name": "RouteRead", "data": {"route": "columns_up"}}]}
    if template == "route_boustrophedon":
        return {**base, "steps": [{"name": "RouteRead", "data": {"route": "rows_boustrophedon"}}]}
    if template == "route_columns_boustrophedon":
        return {**base, "steps": [{"name": "RouteRead", "data": {"route": "columns_boustrophedon"}}]}
    if template == "route_spiral_cw":
        return {**base, "steps": [{"name": "RouteRead", "data": {"route": "spiral_clockwise"}}]}
    if template == "ndown_1_1":
        return {**base, "steps": [{"name": "NDownMAcross", "data": {
            "rangeStart": 0, "rangeEnd": rows - 1, "down": 1, "across": 1,
        }}]}
    if template == "ndown_2_1":
        return {**base, "steps": [{"name": "NDownMAcross", "data": {
            "rangeStart": 0, "rangeEnd": rows - 1, "down": 2, "across": 1,
        }}]}
    if template == "ndown_3_1":
        return {**base, "steps": [{"name": "NDownMAcross", "data": {
            "rangeStart": 0, "rangeEnd": rows - 1, "down": 3, "across": 1,
        }}]}
    if template == "row_reversals":
        steps = [
            {"name": "Reverse", "data": {
                "rangeStart": r * columns,
                "rangeEnd": min((r + 1) * columns, columns * rows) - 1,
            }}
            for r in range(rows)
        ]
        return {**base, "steps": steps}
    if template == "whole_reverse":
        return {"steps": [{"name": "Reverse", "data": {}}]}
    raise ValueError(f"Unknown pipeline template: {template!r}")


# All named templates — used to validate sweep configs.
PIPELINE_TEMPLATES = [
    "route_columns_down",
    "route_columns_up",
    "route_boustrophedon",
    "route_columns_boustrophedon",
    "route_spiral_cw",
    "ndown_1_1",
    "ndown_2_1",
    "ndown_3_1",
    "row_reversals",
    "whole_reverse",
]


# ---------------------------------------------------------------------------
# Pipeline hash helper  (mirrors transform_search._token_hash)
# ---------------------------------------------------------------------------

def _pipeline_token_order_hash(token_count: int, pipeline_raw: dict[str, Any] | None) -> str | None:
    """Hash of the position-permutation produced by pipeline on range(token_count)."""
    if not pipeline_raw:
        return None
    pipeline = TransformPipeline.from_raw(pipeline_raw)
    if pipeline is None or pipeline.is_empty():
        return None
    order = apply_transform_pipeline(list(range(token_count)), pipeline).tokens
    h = hashlib.sha1()
    h.update(array("I", order).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class SweepEntry:
    language: str
    transform_family: str      # "t_substitution" | "t_homophonic"
    approx_length: int
    columns: int
    pipeline_template: str
    seeds: list[int]
    topic: str = "general"
    frequency_style: str = "normal"


def default_sweep() -> list[SweepEntry]:
    """Return the default evaluation sweep (English, two families, five templates)."""
    seeds = list(range(1, 11))   # 10 seeds per cell
    entries = []
    for transform_family, homophonic in [("t_substitution", False), ("t_homophonic", True)]:
        for approx_length, columns_list in [(200, [7, 9, 11, 13])]:
            for columns in columns_list:
                for template in [
                    "route_columns_down",
                    "route_boustrophedon",
                    "ndown_1_1",
                    "row_reversals",
                    "whole_reverse",
                ]:
                    entries.append(SweepEntry(
                        language="en",
                        transform_family=transform_family,
                        approx_length=approx_length,
                        columns=columns,
                        pipeline_template=template,
                        seeds=seeds,
                        topic="general",
                        frequency_style="normal",
                    ))
    return entries


# ---------------------------------------------------------------------------
# Cipher construction helpers  (parallel to testgen.builder internals)
# ---------------------------------------------------------------------------

def _make_substitution_key(rng: random.Random) -> dict[str, str]:
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _apply_substitution_key(words: list[str], key: dict[str, str]) -> list[str]:
    return ["".join(key.get(ch, ch) for ch in word.upper()) for word in words]


def _make_homophonic_key(rng: random.Random) -> dict[str, list[str]]:
    """Distribute two-character cipher tokens across plaintext letters.

    Each plaintext letter gets at least 1 token; common letters get extras.
    """
    _FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
    base_tokens = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(52)]
    rng.shuffle(base_tokens)
    key: dict[str, list[str]] = {ch: [] for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    # Assign one token per letter first.
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        key[ch].append(base_tokens[i])
    # Distribute the remaining 26 tokens to frequent letters.
    for j, ch in enumerate(_FREQ_ORDER):
        key[ch].append(base_tokens[26 + j])
    return key


def _apply_homophonic_key(
    words: list[str], key: dict[str, list[str]], rng: random.Random
) -> list[list[str]]:
    result = []
    for word in words:
        tokens = [rng.choice(key[ch.upper()]) for ch in word if ch.upper() in key]
        result.append(tokens)
    return result


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

class _NoCachePlaintext(Exception):
    pass


class _CacheMissRejector:
    """Drop-in PlaintextGenerator that errors on any LLM call."""
    def generate(self, spec: TestSpec) -> str:
        raise _NoCachePlaintext(
            f"No cached plaintext for lang={spec.language} "
            f"length={spec.approx_length} topic={spec.topic}"
        )


def _build_case(
    entry: SweepEntry,
    seed: int,
    plaintext: str,
) -> PopulationEntry | None:
    """Build one PopulationEntry from a plaintext string and sweep params."""
    words = plaintext.split()
    if not words:
        return None

    rng = random.Random(seed)

    # Build the pre-transform flat token list first so we know the actual
    # token count.  We need it to set correct grid rows for the pipeline.
    if entry.transform_family == "t_homophonic":
        homo_key = _make_homophonic_key(rng)
        token_words = _apply_homophonic_key(words, homo_key, rng)
        flat_tokens: list[str] = [t for word in token_words for t in word]
        cipher_system = "transposition_homophonic"
    else:
        sub_key = _make_substitution_key(rng)
        cipher_words = _apply_substitution_key(words, sub_key)
        flat_tokens = [ch for word in cipher_words for ch in word]
        cipher_system = "transposition_substitution"

    # Now we know the actual token count — use it for the grid.
    token_count = len(flat_tokens)
    rows = max(2, token_count // entry.columns)

    # Build the pipeline with the correct (columns, rows).
    pipeline_raw = _make_pipeline(entry.pipeline_template, entry.columns, rows)
    pipeline = TransformPipeline.from_raw(pipeline_raw)

    # Apply the pipeline (make_inverse_input_for_pipeline scrambles the tokens
    # so that applying the pipeline in forward direction yields the plaintext
    # token order).
    if pipeline is not None and not pipeline.is_empty():
        scrambled = make_inverse_input_for_pipeline(flat_tokens, pipeline)
    else:
        scrambled = flat_tokens

    canonical = " ".join(scrambled)
    plaintext_out = "".join(words)

    # Compute ground-truth token-order hash from the actual token count.
    gt_hash = _pipeline_token_order_hash(token_count, pipeline_raw)

    family_short = "th" if entry.transform_family == "t_homophonic" else "ts"
    template_short = entry.pipeline_template.replace("_", "")[:12]
    case_id = (
        f"triage_{entry.language}_{entry.approx_length}"
        f"{family_short}_{entry.columns}c_{template_short}_s{seed}"
    )

    return PopulationEntry(
        case_id=case_id,
        language=entry.language,
        transform_family=entry.transform_family,
        columns=entry.columns,
        rows=rows,
        approx_length=entry.approx_length,
        seed=seed,
        homophonic=entry.transform_family == "t_homophonic",
        canonical=canonical,
        plaintext=plaintext_out,
        cipher_system=cipher_system,
        ground_truth_pipeline=pipeline_raw,
        ground_truth_token_order_hash=gt_hash,
    )


def generate_population(
    sweep: list[SweepEntry],
    output_path: Path,
    cache_dir: Path = Path("testgen_cache"),
    *,
    verbose: bool = False,
) -> int:
    """Write population entries to output_path (JSONL), return count written."""
    cache = PlaintextCache(cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    seen_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as fh:
        for sweep_entry in sweep:
            spec = TestSpec(
                language=sweep_entry.language,
                approx_length=sweep_entry.approx_length,
                topic=sweep_entry.topic,
                frequency_style=sweep_entry.frequency_style,
            )
            plaintext = cache.get(spec)
            if plaintext is None:
                if verbose:
                    print(
                        f"  SKIP  no cached plaintext for "
                        f"lang={sweep_entry.language} "
                        f"length={sweep_entry.approx_length} "
                        f"topic={sweep_entry.topic}"
                    )
                skipped += len(sweep_entry.seeds)
                continue

            for seed in sweep_entry.seeds:
                pop_entry = _build_case(sweep_entry, seed, plaintext)
                if pop_entry is None:
                    skipped += 1
                    continue
                if pop_entry.case_id in seen_ids:
                    # Dedup — same plaintext × same params produces same case.
                    skipped += 1
                    continue
                seen_ids.add(pop_entry.case_id)
                fh.write(json.dumps(pop_entry.to_dict(), ensure_ascii=False) + "\n")
                total += 1

    if verbose:
        print(f"Wrote {total} population entries ({skipped} skipped) to {output_path}")
    return total


def load_population(path: Path) -> list[PopulationEntry]:
    """Load all entries from a population JSONL file."""
    entries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(PopulationEntry.from_dict(json.loads(line)))
    return entries
