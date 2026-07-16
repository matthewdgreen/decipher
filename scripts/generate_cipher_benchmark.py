#!/usr/bin/env python3
"""On-demand cipher benchmark generator (Slice C).

Three-step, fully deterministic, offline, LLM-free pipeline:

  1. plaintext selection  — Slice-B ``corpus_library.select()`` picks a MEASURED
     passage per (language, difficulty) window;
  2. key + encipherment   — Slice-A family (via ``testgen.family_registry``)
     samples a key from an injected ``random.Random`` and enciphers the passage;
  3. batch                — for each requested family x language x difficulty,
     emit N examples with graded CONTEXT TIERS (none -> language ->
     era_provenance -> rich), and write a loader-compatible benchmark tree.

Output tree (loads unchanged through ``benchmark.loader.BenchmarkLoader``)::

    <out>/manifest/records.jsonl          # one BenchmarkRecord per case
    <out>/data/<source>/<id>.canonical.txt        # ciphertext (safe to show)
    <out>/ground_truth/<source>/<id>.plaintext.txt   # ground-truth plaintext
    <out>/ground_truth/<source>/<id>.key.json        # key (FIREWALLED)
    <out>/splits/gen_<family>.jsonl       # one split per family
    <out>/splits/all_generated.jsonl      # every generated test

FIREWALL: context tiers live in each record's ``context_layers`` metadata and
NEVER contain the plaintext or the key; keys live only under ``ground_truth/``.
Every context layer is checked at generation time (:func:`_assert_firewall`).

CLI::

    python scripts/generate_cipher_benchmark.py \
        --families all --languages en,de --per-family 3 \
        --difficulty medium --context-tiers none,language,era_provenance,rich \
        --out artifacts/generated_benchmark --seed 20260716

``--dry-run`` prints the plan/counts and writes nothing.  Same ``--seed`` +
inputs -> byte-identical ``manifest/records.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Make ``src`` importable when run as a script (tests set PYTHONPATH=src).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.ic import index_of_coincidence  # noqa: E402
from testgen.corpus_library import (  # noqa: E402
    DEFAULT_LIBRARY_DIR,
    LibraryEmpty,
    PassageRecord,
    load_library,
    select,
)
from testgen.family_registry import (  # noqa: E402
    DIFFICULTIES,
    LETTERS,
    FamilyGenSpec,
    KeyContext,
    all_family_ids,
    family_cross_reference,
    get_spec,
    length_window,
    new_family_ids,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_TIERS: tuple[str, ...] = ("none", "language", "era_provenance", "rich")

# Tier -> benchmark track name. ``none`` uses the historical default track;
# richer tiers are separate tracks (mirroring the existing benchmark's track
# structure) so a runner can select a tier by track.
_TIER_TRACK: dict[str, str] = {
    "none": "transcription2plaintext",
    "language": "transcription2plaintext_language",
    "era_provenance": "transcription2plaintext_era_provenance",
    "rich": "transcription2plaintext_rich",
}

_LANG_NAME: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "la": "Latin",
}

DEFAULT_SEED = 20260716


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GeneratedCase:
    """One generated (family, language, difficulty, idx) case (in-memory)."""

    record_id: str
    source: str
    family_id: str
    language: str
    difficulty: str
    index: int
    canonical_form: str
    canonical_ciphertext: str
    plaintext: str
    key: Any
    key_description: str
    ioc: float | None
    length_chars: int
    token_count: int
    era: str
    provenance: str
    topic: str
    measured: dict[str, float]
    context_layers: dict[str, dict[str, Any]]
    tiers: list[str]


@dataclass
class GenerationResult:
    out_dir: str | None
    seed: int
    families: list[str]
    languages: list[str]
    difficulties: list[str]
    tiers: list[str]
    per_family: int
    dry_run: bool
    cases: list[GeneratedCase] = field(default_factory=list)
    num_records: int = 0
    num_tests: int = 0
    split_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def _case_seed(master_seed: int, family_id: str, language: str, difficulty: str, idx: int) -> int:
    payload = f"{master_seed}|{family_id}|{language}|{difficulty}|{idx}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _dumps(obj: Any) -> str:
    """Deterministic JSON line (sorted keys, UTF-8) for byte-identical output."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Context tiers + firewall
# ---------------------------------------------------------------------------


def _build_context_layers(
    spec: FamilyGenSpec,
    *,
    language: str,
    era: str,
    provenance: str,
    topic: str,
    tiers: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Graded context text per tier.  NEVER contains plaintext or key."""
    lang_name = _LANG_NAME.get(language, language)
    layers: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        if tier == "none":
            text = ""
            plaintext_hint = False
            cipher_hint = False
        elif tier == "language":
            text = f"Hypothesized plaintext language: {lang_name}."
            plaintext_hint = True
            cipher_hint = False
        elif tier == "era_provenance":
            text = (
                f"Hypothesized plaintext language: {lang_name}. "
                f"Era: {era}. Provenance: {provenance}."
            )
            plaintext_hint = True
            cipher_hint = False
        elif tier == "rich":
            inv = spec.inv_family_id or "none"
            text = (
                f"Hypothesized plaintext language: {lang_name}. "
                f"Era: {era}. Provenance: {provenance}. Topic: {topic}. "
                f"Cipher family: {spec.display_name} "
                f"(family_id={spec.family_id}; INV rollup={inv})."
            )
            plaintext_hint = True
            cipher_hint = True
        else:
            raise ValueError(f"unknown context tier: {tier!r}")
        layers[tier] = {
            "tier": tier,
            "label": tier,
            "text": text,
            "contains_solution": False,
            "contains_plaintext_hint": plaintext_hint,
            "contains_cipher_type_hint": cipher_hint,
        }
    return layers


def _assert_firewall(
    context_layers: dict[str, dict[str, Any]],
    *,
    plaintext: str,
    key_description: str,
    record_id: str,
) -> None:
    """Belt-and-suspenders: no context layer may leak plaintext or key.

    Context text is a fixed template that only ever interpolates
    language/era/provenance/topic/family labels — never the key variable — so
    the key cannot structurally appear.  We still assert the full plaintext and
    the human ``key_description`` are absent.  (A raw scalar key such as a
    rail-count ``3`` is deliberately NOT substring-checked: a bare digit is a
    meaningless coincidental match, not a leak.)
    """
    for tier, layer in context_layers.items():
        text = layer.get("text", "")
        if not text:
            continue
        if plaintext and plaintext in text:
            raise AssertionError(f"{record_id}: plaintext leaked into {tier} context")
        if key_description and key_description in text:
            raise AssertionError(f"{record_id}: key_description leaked into {tier} context")
        if layer.get("contains_solution"):
            raise AssertionError(f"{record_id}: {tier} context claims contains_solution")


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def _canonical_ciphertext(ciphertext: str, form: str) -> str:
    """Serialise ciphertext into the canonical transcription string.

    ``letters`` -> space-separated single letters (the no-boundary convention of
    ``builder._format_canonical``).  ``numeric`` / ``tokens`` -> verbatim (the
    family already produces space-separated numbers or an opaque encoded string).
    """
    if form == LETTERS:
        return " ".join(ciphertext)
    return ciphertext


def _ioc_for(ciphertext: str, form: str) -> float | None:
    if form != LETTERS:
        return None
    tokens = [ord(c) - 65 for c in ciphertext if "A" <= c <= "Z"]
    if len(tokens) <= 1:
        return None
    return round(index_of_coincidence(tokens, 26), 6)


def generate_case(
    spec: FamilyGenSpec,
    *,
    language: str,
    difficulty: str,
    index: int,
    master_seed: int,
    library_records: list[PassageRecord],
    tiers: list[str],
) -> GeneratedCase:
    """Run the 3-step pipeline for a single case (deterministic)."""
    import random

    seed = _case_seed(master_seed, spec.family_id, language, difficulty, index)
    rng = random.Random(seed)

    # Step 1 — plaintext selection (MEASURED difficulty window).
    window = length_window(difficulty)
    passage = select(
        records=library_records,
        language=language,
        min_words=window["min_words"],
        max_words=window["max_words"],
        rng=rng,
    )

    # Step 2 — key + encipherment.
    cipher = spec.new_cipher()
    ctx = KeyContext(
        language=language,
        difficulty=difficulty,
        plaintext=passage.text,
        library_records=library_records,
        rng=rng,
        source_hash=passage.content_hash,
    )
    key = spec.make_key(cipher, ctx)
    # Normalise key through JSON so the stored key and the round-trip key are
    # byte-identical (tuples -> lists), and confirm it still enciphers.
    key = json.loads(json.dumps(key))
    ciphertext = cipher.encrypt(passage.text, key)
    plaintext = cipher.decrypt(ciphertext, key)

    # Internal consistency guard: re-enciphering the stored plaintext with the
    # stored key must reproduce the stored ciphertext (a genuine round-trip).
    if cipher.encrypt(plaintext, key) != ciphertext:
        raise AssertionError(
            f"round-trip inconsistency for {spec.family_id} "
            f"({language}/{difficulty}/{index})"
        )

    canonical = _canonical_ciphertext(ciphertext, spec.canonical_form)
    ioc = _ioc_for(ciphertext, spec.canonical_form)
    key_description = cipher.describe_key(key)

    source = f"gen_{spec.family_id}"
    record_id = f"gen_{spec.family_id}_{language}_{difficulty}_{index:03d}"

    context_layers = _build_context_layers(
        spec,
        language=language,
        era=passage.era,
        provenance=passage.provenance,
        topic=passage.topic,
        tiers=tiers,
    )
    _assert_firewall(
        context_layers,
        plaintext=plaintext,
        key_description=key_description,
        record_id=record_id,
    )

    return GeneratedCase(
        record_id=record_id,
        source=source,
        family_id=spec.family_id,
        language=language,
        difficulty=difficulty,
        index=index,
        canonical_form=spec.canonical_form,
        canonical_ciphertext=canonical,
        plaintext=plaintext,
        key=key,
        key_description=key_description,
        ioc=ioc,
        length_chars=len(plaintext),
        token_count=len(canonical.split()),
        era=passage.era,
        provenance=passage.provenance,
        topic=passage.topic,
        measured=dict(passage.measured),
        context_layers=context_layers,
        tiers=list(tiers),
    )


# ---------------------------------------------------------------------------
# Manifest / split records
# ---------------------------------------------------------------------------


def _canonical_rel(case: GeneratedCase) -> str:
    return f"data/{case.source}/{case.record_id}.canonical.txt"


def _plaintext_rel(case: GeneratedCase) -> str:
    return f"ground_truth/{case.source}/{case.record_id}.plaintext.txt"


def _key_rel(case: GeneratedCase) -> str:
    return f"ground_truth/{case.source}/{case.record_id}.key.json"


def _manifest_record(case: GeneratedCase, spec: FamilyGenSpec) -> dict[str, Any]:
    """A BenchmarkRecord-compatible manifest line (+ documented raw extras).

    ``plaintext_file`` points into the firewalled ``ground_truth/`` area; the
    loader legitimately reads it as the target's own ground truth for scoring.
    """
    return {
        # BenchmarkRecord fields (loader reads these).
        "id": case.record_id,
        "source": case.source,
        "cipher_type": [case.family_id],
        "plaintext_language": case.language,
        "transcription_canonical_file": _canonical_rel(case),
        "plaintext_file": _plaintext_rel(case),
        "has_key": True,
        # Documented raw extras.
        "synthetic": True,
        "generator": "cipher_benchmark_generator_slice_c",
        "family_id": case.family_id,
        "inv_family_id": spec.inv_family_id,
        "is_new_family": spec.family_id in set(new_family_ids()),
        "canonical_form": case.canonical_form,
        "word_boundaries": False,
        "difficulty": case.difficulty,
        "example_index": case.index,
        "era": case.era,
        "provenance": case.provenance,
        "topic": case.topic,
        "measured": case.measured,
        "length_chars": case.length_chars,
        "token_count": case.token_count,
        "index_of_coincidence": case.ioc,
        "context_tier_default": "none",
        "context_tiers": case.tiers,
        "context_layers": case.context_layers,
        # Ground-truth pointers (firewalled area; NOT context).
        "key_description": case.key_description,
        "key_file": _key_rel(case),
        "rights_class": "open",
        "status": "solved_verified",
    }


def _test_records(case: GeneratedCase, spec: FamilyGenSpec) -> list[dict[str, Any]]:
    """One BenchmarkTest per requested context tier for this case."""
    tests: list[dict[str, Any]] = []
    lang_name = _LANG_NAME.get(case.language, case.language)
    for tier in case.tiers:
        tests.append(
            {
                "test_id": f"{case.record_id}__{tier}",
                "track": _TIER_TRACK[tier],
                "cipher_system": case.family_id,
                "target_records": [case.record_id],
                "context_records": [],
                "description": (
                    f"{spec.display_name} ({case.family_id}), {lang_name}, "
                    f"difficulty={case.difficulty}, context tier={tier}, "
                    f"canonical form={case.canonical_form}."
                ),
            }
        )
    return tests


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _resolve_families(families_arg: str) -> list[str]:
    if families_arg.strip().lower() == "all":
        return all_family_ids()
    requested = [f.strip() for f in families_arg.split(",") if f.strip()]
    unknown = [f for f in requested if f not in set(all_family_ids())]
    if unknown:
        raise SystemExit(
            f"unknown families: {', '.join(unknown)}\n"
            f"known: {', '.join(all_family_ids())}"
        )
    return requested


def _resolve_list(arg: str, *, allowed: tuple[str, ...], name: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(allowed)
    values = [v.strip() for v in arg.split(",") if v.strip()]
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise SystemExit(f"unknown {name}: {', '.join(unknown)}; allowed: {', '.join(allowed)}")
    return values


def _plan(
    families: list[str], languages: list[str], difficulties: list[str], per_family: int
) -> list[tuple[str, str, str, int]]:
    """Deterministic (family, language, difficulty, idx) plan.

    Languages are intersected with each family's ``applicable`` set (all Slice-A
    families apply to every shipped language today, so nothing is dropped).
    """
    plan: list[tuple[str, str, str, int]] = []
    for family_id in families:
        spec = get_spec(family_id)
        for language in languages:
            if language not in spec.languages:
                continue
            for difficulty in difficulties:
                for idx in range(per_family):
                    plan.append((family_id, language, difficulty, idx))
    return plan


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


def generate(
    *,
    families: list[str],
    languages: list[str],
    difficulties: list[str],
    tiers: list[str],
    per_family: int,
    seed: int,
    out_dir: str | os.PathLike | None,
    dry_run: bool = False,
    library_dir: str | os.PathLike | None = None,
) -> GenerationResult:
    """Run the full generator.  Returns an in-memory summary regardless of I/O."""
    result = GenerationResult(
        out_dir=str(out_dir) if out_dir is not None else None,
        seed=seed,
        families=list(families),
        languages=list(languages),
        difficulties=list(difficulties),
        tiers=list(tiers),
        per_family=per_family,
        dry_run=dry_run,
    )

    # Load each language library once (deterministic, offline).
    lib_dir = Path(library_dir) if library_dir is not None else DEFAULT_LIBRARY_DIR
    libraries: dict[str, list[PassageRecord]] = {}
    for language in languages:
        libraries[language] = load_library(language, library_dir=lib_dir)

    plan = _plan(families, languages, difficulties, per_family)

    cases: list[GeneratedCase] = []
    for family_id, language, difficulty, idx in plan:
        spec = get_spec(family_id)
        records = libraries.get(language, [])
        if not records:
            raise SystemExit(
                f"no plaintext library for language={language!r} "
                f"(looked in {lib_dir}); cannot generate {family_id}"
            )
        try:
            case = generate_case(
                spec,
                language=language,
                difficulty=difficulty,
                index=idx,
                master_seed=seed,
                library_records=records,
                tiers=tiers,
            )
        except LibraryEmpty as exc:
            raise SystemExit(
                f"cannot select plaintext for {family_id} {language}/{difficulty}: {exc}"
            ) from exc
        cases.append(case)

    result.cases = cases
    result.num_records = len(cases)
    result.num_tests = sum(len(c.tiers) for c in cases)

    if dry_run:
        return result

    if out_dir is None:
        raise SystemExit("--out is required unless --dry-run")

    _write_tree(result, Path(out_dir))
    return result


def _write_tree(result: GenerationResult, out: Path) -> None:
    manifest_dir = out / "manifest"
    splits_dir = out / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    per_family_tests: dict[str, list[str]] = {}
    all_tests: list[str] = []

    for case in result.cases:
        spec = get_spec(case.family_id)

        # Data + ground-truth files.
        canon_path = out / _canonical_rel(case)
        pt_path = out / _plaintext_rel(case)
        key_path = out / _key_rel(case)
        canon_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        canon_path.write_text(case.canonical_ciphertext + "\n", encoding="utf-8")
        pt_path.write_text(case.plaintext + "\n", encoding="utf-8")
        key_path.write_text(
            _dumps(
                {
                    "family_id": case.family_id,
                    "key": case.key,
                    "key_description": case.key_description,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        manifest_lines.append(_dumps(_manifest_record(case, spec)))

        for test in _test_records(case, spec):
            line = _dumps(test)
            per_family_tests.setdefault(case.family_id, []).append(line)
            all_tests.append(line)

    (manifest_dir / "records.jsonl").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )

    split_files: list[str] = []
    for family_id, lines in per_family_tests.items():
        path = splits_dir / f"gen_{family_id}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_files.append(str(path))
    (splits_dir / "all_generated.jsonl").write_text(
        "\n".join(all_tests) + ("\n" if all_tests else ""), encoding="utf-8"
    )
    split_files.append(str(splits_dir / "all_generated.jsonl"))

    # README pointer for the firewalled area (documentation only).
    gt_readme = out / "ground_truth" / "README.txt"
    if gt_readme.parent.exists():
        gt_readme.write_text(
            "FIREWALLED ground truth (keys + plaintext). Never surface these as "
            "solver/model context. Context tiers live in each manifest record's "
            "context_layers and contain no plaintext or key.\n",
            encoding="utf-8",
        )

    result.split_files = split_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deterministic, offline cipher benchmark generator (Slice C).",
    )
    p.add_argument(
        "--families", default="all",
        help="'all' or a comma list of family ids (see --list-families).",
    )
    p.add_argument(
        "--languages", default="en",
        help="'all' or a comma list of languages (en,de,fr,it,la).",
    )
    p.add_argument("--per-family", type=int, default=3, help="examples per family x language x difficulty.")
    p.add_argument(
        "--difficulty", default="medium",
        help="'all' or a comma list of {easy,medium,hard}.",
    )
    p.add_argument(
        "--context-tiers", default=",".join(CONTEXT_TIERS),
        help="'all' or a comma list of {none,language,era_provenance,rich}.",
    )
    p.add_argument("--out", default=None, help="output benchmark tree directory.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master RNG seed.")
    p.add_argument("--dry-run", action="store_true", help="print plan/counts, write nothing.")
    p.add_argument("--library-dir", default=None, help="override plaintext library directory.")
    p.add_argument("--list-families", action="store_true", help="list generatable families and exit.")
    return p


def _print_family_list() -> None:
    print("Generatable families (family_id -> INV rollup, form, NEW?):")
    for row in family_cross_reference():
        tag = "NEW" if row["is_new"] else "reuses INV id"
        print(
            f"  {row['family_id']:<26} rollup={str(row['inv_family_id']):<28} "
            f"form={row['canonical_form']:<8} {tag}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list_families:
        _print_family_list()
        return 0

    families = _resolve_families(args.families)
    languages = _resolve_list(args.languages, allowed=("en", "de", "fr", "it", "la"), name="languages")
    difficulties = _resolve_list(args.difficulty, allowed=DIFFICULTIES, name="difficulty")
    tiers = _resolve_list(args.context_tiers, allowed=CONTEXT_TIERS, name="context-tiers")

    result = generate(
        families=families,
        languages=languages,
        difficulties=difficulties,
        tiers=tiers,
        per_family=args.per_family,
        seed=args.seed,
        out_dir=args.out,
        dry_run=args.dry_run,
        library_dir=args.library_dir,
    )

    plan_n = len(families) * len(languages) * len(difficulties) * args.per_family
    print("Cipher benchmark generation plan:")
    print(f"  families      : {len(families)} ({', '.join(families) if len(families) <= 12 else 'all'})")
    print(f"  languages     : {', '.join(languages)}")
    print(f"  difficulties  : {', '.join(difficulties)}")
    print(f"  context tiers : {', '.join(tiers)}")
    print(f"  per-family    : {args.per_family}")
    print(f"  seed          : {args.seed}")
    print(f"  planned cases : {plan_n}")
    print(f"  records       : {result.num_records}")
    print(f"  tests         : {result.num_tests}")
    if args.dry_run:
        print("  DRY RUN: no files written.")
    else:
        print(f"  out           : {result.out_dir}")
        print(f"  split files   : {len(result.split_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
