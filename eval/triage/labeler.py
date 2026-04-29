"""Ground-truth labeler for captured transform-search candidates.

Three label tiers are computed per candidate:

  transform_correct   — did the candidate's pipeline match the ground-truth
                        pipeline (same position-permutation hash)?  Cheap,
                        metadata-only.

  readable_now        — character accuracy of run_automated's best decryption
                        when given this candidate's pipeline.  Requires a
                        downstream solver call.

  rescuable           — readable_now >= RESCUABLE_THRESHOLD.  Derived from
                        readable_now once the solver call has run.

The labeler only runs the expensive solver on a bounded budget per case:
  - all top-<top_rank_budget> candidates by original_rank
  - any candidate where transform_correct is True
  - a random sample of <tail_sample> from remaining candidates

Usage
-----
    from triage.labeler import label_case

    labeled = label_case(case, verbose=True)
    labeled.save(path)
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

# src/ imports
from automated.runner import run_automated
from benchmark.loader import parse_canonical_transcription
from benchmark.scorer import score_decryption

from triage.types import CandidateRecord, CapturedCase, PopulationEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESCUABLE_THRESHOLD = 0.70        # minimum char_accuracy to call a candidate rescuable
DEFAULT_TOP_RANK_BUDGET = 25      # always label these many top-ranked candidates
DEFAULT_TAIL_SAMPLE = 10          # random sample from candidates below budget
DEFAULT_HOMOPHONIC_BUDGET = "screen"


# ---------------------------------------------------------------------------
# transform_correct labeling  (cheap, no solver)
# ---------------------------------------------------------------------------

def apply_transform_correct_labels(case: CapturedCase) -> CapturedCase:
    """Mark candidates whose token-order hash matches the ground-truth pipeline.

    Returns a new CapturedCase with transform_correct filled in for all
    candidates.  Candidates where the case has no ground-truth hash remain
    transform_correct=None.
    """
    entry = case.entry()
    gt_hash = entry.ground_truth_token_order_hash
    if gt_hash is None:
        return case

    new_candidates = []
    for c in case.candidates:
        c2 = deepcopy(c)
        c2.transform_correct = (c.token_order_hash == gt_hash)
        new_candidates.append(c2)

    result = deepcopy(case)
    result.candidates = new_candidates
    return result


# ---------------------------------------------------------------------------
# Solver-based labels  (readable_now, rescuable, decoded_text)
# ---------------------------------------------------------------------------

def _label_one(
    candidate: CandidateRecord,
    entry: PopulationEntry,
    *,
    homophonic_budget: str,
) -> CandidateRecord:
    """Run run_automated on one candidate pipeline and fill solver labels."""
    if candidate.readable_now is not None:
        # Already labeled — skip.
        return candidate

    cipher_text = parse_canonical_transcription(entry.canonical)

    try:
        result = run_automated(
            cipher_text=cipher_text,
            language=entry.language,
            cipher_id=f"triage_label_{candidate.candidate_id}",
            ground_truth=entry.plaintext,
            cipher_system=entry.cipher_system,
            transform_pipeline=candidate.pipeline,
            homophonic_budget=homophonic_budget,
            transform_search="off",
        )
        char_acc = result.char_accuracy
        decoded = result.final_decryption
    except Exception:  # noqa: BLE001
        # Solver crashed on this candidate — treat as unrescuable.
        char_acc = 0.0
        decoded = ""

    c2 = deepcopy(candidate)
    c2.readable_now = round(char_acc, 4)
    c2.rescuable = char_acc >= RESCUABLE_THRESHOLD
    c2.rescuable_char_accuracy = round(char_acc, 4)
    c2.decoded_text = decoded
    return c2


def _select_label_budget(
    candidates: list[CandidateRecord],
    *,
    top_rank_budget: int,
    tail_sample: int,
    rng: random.Random,
) -> set[int]:
    """Return the set of original_rank values to label with the solver."""
    to_label: set[int] = set()

    # Always label top N by original_rank.
    for c in candidates:
        if c.original_rank < top_rank_budget:
            to_label.add(c.original_rank)

    # Always label any already-known transform_correct candidates.
    for c in candidates:
        if c.transform_correct:
            to_label.add(c.original_rank)

    # Random sample from the tail (not yet in budget).
    tail = [c for c in candidates if c.original_rank not in to_label]
    sample_count = min(tail_sample, len(tail))
    for c in rng.sample(tail, sample_count):
        to_label.add(c.original_rank)

    return to_label


def label_case(
    case: CapturedCase,
    *,
    top_rank_budget: int = DEFAULT_TOP_RANK_BUDGET,
    tail_sample: int = DEFAULT_TAIL_SAMPLE,
    homophonic_budget: str = DEFAULT_HOMOPHONIC_BUDGET,
    rescuable_threshold: float = RESCUABLE_THRESHOLD,
    rng_seed: int = 0,
    verbose: bool = False,
) -> CapturedCase:
    """Label a CapturedCase with transform_correct + solver-based labels.

    Returns a new CapturedCase (does not mutate the input).
    """
    # Step 1: cheap structural labels.
    case = apply_transform_correct_labels(case)
    entry = case.entry()

    # Step 2: decide which candidates get the expensive solver label.
    rng = random.Random(rng_seed)
    budget = _select_label_budget(
        case.candidates,
        top_rank_budget=top_rank_budget,
        tail_sample=tail_sample,
        rng=rng,
    )

    if verbose:
        print(
            f"  {case.case_id}: labeling {len(budget)}/{len(case.candidates)} "
            f"candidates (top {top_rank_budget} + {len(budget) - min(top_rank_budget, len(case.candidates))} extra)"
        )

    # Step 3: run solver on budgeted candidates.
    new_candidates = []
    labeled_count = 0
    for c in case.candidates:
        if c.original_rank in budget:
            c2 = _label_one(c, entry, homophonic_budget=homophonic_budget)
            # Override rescuable threshold if different from default.
            if c2.rescuable_char_accuracy is not None:
                c2.rescuable = c2.rescuable_char_accuracy >= rescuable_threshold
            new_candidates.append(c2)
            labeled_count += 1
        else:
            new_candidates.append(deepcopy(c))

    if verbose:
        rescuable = sum(1 for c in new_candidates if c.rescuable)
        print(f"  {case.case_id}: {rescuable} rescuable in labeled set")

    result = deepcopy(case)
    result.candidates = new_candidates
    return result
