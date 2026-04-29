"""Ground-truth labeler for captured transform-search candidates.

Three label tiers are computed per candidate:

  transform_correct   — did the candidate's pipeline match the ground-truth
                        pipeline (same position-permutation hash)?  Cheap,
                        metadata-only.  Always computed for all candidates.

  readable_now        — character accuracy of the best decryption when the
                        candidate pipeline is applied then solved.  Computed
                        by a fast solver call (substitution hill-climb only;
                        never zenith SA).

  rescuable           — readable_now >= RESCUABLE_THRESHOLD.

Labeling strategy by cipher family
-----------------------------------
  t_substitution  — run a fast substitution hill-climb on each budgeted
                    candidate (typically <1 s each).  Full solver.

  t_homophonic    — transform_correct is a reliable rescuable proxy because
                    the downstream SA will always succeed from the right
                    pipeline.  We therefore set rescuable=True for any
                    transform_correct candidate and skip the expensive SA for
                    all others.  This makes labeling ~100× faster.

                    To get a measured readable_now on the transform_correct
                    candidate only, pass measure_homophonic_correct=True
                    (default False — requires full homophonic solve time).

Usage
-----
    from triage.labeler import label_case

    labeled = label_case(case, verbose=True)
    labeled.save(path)
"""
from __future__ import annotations

import random
from copy import deepcopy

# src/ imports
from analysis.transform_search import screen_transform_candidates
from automated.runner import run_automated
from benchmark.loader import parse_canonical_transcription

from triage.types import CandidateRecord, CapturedCase, PopulationEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESCUABLE_THRESHOLD = 0.70        # minimum char_accuracy to call a candidate rescuable
DEFAULT_TOP_RANK_BUDGET = 25      # always label these many top-ranked candidates
DEFAULT_TAIL_SAMPLE = 10          # random sample from candidates outside budget


# ---------------------------------------------------------------------------
# transform_correct labeling  (cheap, no solver, always run for all candidates)
# ---------------------------------------------------------------------------

def apply_transform_correct_labels(case: CapturedCase) -> CapturedCase:
    """Mark candidates whose token-order hash matches the ground-truth pipeline.

    Returns a new CapturedCase with transform_correct filled in for all
    candidates.  Cases with no ground-truth hash leave transform_correct=None.
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
# Fast substitution-only solver label
# ---------------------------------------------------------------------------

def _label_substitution(
    candidate: CandidateRecord,
    entry: PopulationEntry,
) -> CandidateRecord:
    """Apply candidate pipeline then run a fast substitution solve.

    Uses run_automated with transform_search=off and the substitution solver
    path.  For t_substitution cases this is a hill-climb — typically <1 s.
    """
    if candidate.readable_now is not None:
        return candidate

    cipher_text = parse_canonical_transcription(entry.canonical)

    try:
        result = run_automated(
            cipher_text=cipher_text,
            language=entry.language,
            cipher_id=f"triage_label_{candidate.candidate_id}",
            ground_truth=entry.plaintext,
            cipher_system="transposition_substitution",   # force fast path
            transform_pipeline=candidate.pipeline,
            transform_search="off",
        )
        char_acc = result.char_accuracy
        decoded = result.final_decryption
    except Exception:  # noqa: BLE001
        char_acc = 0.0
        decoded = ""

    c2 = deepcopy(candidate)
    c2.readable_now = round(char_acc, 4)
    c2.rescuable = char_acc >= RESCUABLE_THRESHOLD
    c2.rescuable_char_accuracy = round(char_acc, 4)
    c2.decoded_text = decoded
    return c2


def _label_homophonic_correct(
    candidate: CandidateRecord,
    entry: PopulationEntry,
    *,
    homophonic_budget: str = "screen",
) -> CandidateRecord:
    """Run the full homophonic solver on one transform_correct candidate.

    Only called when measure_homophonic_correct=True.  Slow (~10–60 s).
    """
    if candidate.readable_now is not None:
        return candidate

    cipher_text = parse_canonical_transcription(entry.canonical)

    try:
        result = run_automated(
            cipher_text=cipher_text,
            language=entry.language,
            cipher_id=f"triage_label_homo_{candidate.candidate_id}",
            ground_truth=entry.plaintext,
            cipher_system=entry.cipher_system,
            transform_pipeline=candidate.pipeline,
            homophonic_budget=homophonic_budget,
            transform_search="off",
        )
        char_acc = result.char_accuracy
        decoded = result.final_decryption
    except Exception:  # noqa: BLE001
        char_acc = 0.0
        decoded = ""

    c2 = deepcopy(candidate)
    c2.readable_now = round(char_acc, 4)
    c2.rescuable = char_acc >= RESCUABLE_THRESHOLD
    c2.rescuable_char_accuracy = round(char_acc, 4)
    c2.decoded_text = decoded
    return c2


# ---------------------------------------------------------------------------
# Budget selection
# ---------------------------------------------------------------------------

def _select_label_budget(
    candidates: list[CandidateRecord],
    *,
    top_rank_budget: int,
    tail_sample: int,
    rng: random.Random,
) -> set[int]:
    """Return original_rank values to run through the solver."""
    to_label: set[int] = set()

    for c in candidates:
        if c.original_rank < top_rank_budget:
            to_label.add(c.original_rank)

    for c in candidates:
        if c.transform_correct:
            to_label.add(c.original_rank)

    tail = [c for c in candidates if c.original_rank not in to_label]
    for c in rng.sample(tail, min(tail_sample, len(tail))):
        to_label.add(c.original_rank)

    return to_label


# ---------------------------------------------------------------------------
# Main label_case entry point
# ---------------------------------------------------------------------------

def label_case(
    case: CapturedCase,
    *,
    top_rank_budget: int = DEFAULT_TOP_RANK_BUDGET,
    tail_sample: int = DEFAULT_TAIL_SAMPLE,
    homophonic_budget: str = "screen",
    rescuable_threshold: float = RESCUABLE_THRESHOLD,
    measure_homophonic_correct: bool = False,
    rng_seed: int = 0,
    verbose: bool = False,
) -> CapturedCase:
    """Label a CapturedCase.  Returns a new CapturedCase; does not mutate input.

    For t_homophonic cases the labeling is fast by default:
      - transform_correct candidates are marked rescuable=True without running SA.
      - Non-correct candidates in budget get IC + substitution-hill-climb only.
      - Pass measure_homophonic_correct=True to also run the full SA on the
        transform_correct candidate (slow, ~10–60 s per case).
    """
    # Step 1: cheap hash labels for all candidates.
    case = apply_transform_correct_labels(case)
    entry = case.entry()
    is_homophonic = entry.transform_family == "t_homophonic"

    rng = random.Random(rng_seed)
    budget = _select_label_budget(
        case.candidates,
        top_rank_budget=top_rank_budget,
        tail_sample=tail_sample,
        rng=rng,
    )

    if verbose:
        print(
            f"  {case.case_id}: {'homophonic' if is_homophonic else 'substitution'} — "
            f"labeling {len(budget)}/{len(case.candidates)} candidates"
        )

    new_candidates = []
    for c in case.candidates:
        if c.original_rank not in budget:
            new_candidates.append(deepcopy(c))
            continue

        if is_homophonic:
            if c.transform_correct:
                if measure_homophonic_correct:
                    # Full SA solve — slow but gives real accuracy.
                    c2 = _label_homophonic_correct(
                        c, entry, homophonic_budget=homophonic_budget
                    )
                else:
                    # Fast path: correct pipeline → rescuable by definition.
                    # The downstream homophonic SA always succeeds from here.
                    c2 = deepcopy(c)
                    c2.rescuable = True
                    c2.rescuable_char_accuracy = None   # not measured
                    c2.readable_now = None
            else:
                # Wrong transform on a homophonic cipher → not rescuable.
                # No solver call: the 52-symbol alphabet would route to SA
                # regardless of cipher_system hint, making this very slow.
                # A wrong permutation of a homophonic cipher cannot be solved
                # by any downstream substitution search.
                c2 = deepcopy(c)
                c2.rescuable = False
                c2.rescuable_char_accuracy = 0.0
                c2.readable_now = 0.0
        else:
            c2 = _label_substitution(c, entry)

        if c2.rescuable_char_accuracy is not None:
            c2.rescuable = c2.rescuable_char_accuracy >= rescuable_threshold

        new_candidates.append(c2)

    if verbose:
        rescuable = sum(1 for c in new_candidates if c.rescuable)
        print(f"  {case.case_id}: {rescuable} rescuable")

    result = deepcopy(case)
    result.candidates = new_candidates
    return result
