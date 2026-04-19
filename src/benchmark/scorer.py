from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Detailed scoring of agent output vs ground truth."""

    test_id: str
    char_accuracy: float  # fraction of characters correct
    word_accuracy: float  # fraction of words correct
    total_chars: int
    correct_chars: int
    total_words: int
    correct_words: int
    agent_score: float  # agent's own dictionary-based score
    status: str


def normalize_text(text: str) -> str:
    """Normalize text for comparison: uppercase, collapse whitespace,
    strip editorial annotations like [brackets] and <=r>."""
    text = text.upper()
    # Remove editorial annotations: [text], <=text>, (?)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<[=!].*?>", "", text)
    text = re.sub(r"\(\?\)", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def score_decryption(
    test_id: str,
    decrypted: str,
    ground_truth: str,
    agent_score: float,
    status: str,
) -> ScoreResult:
    """Score the agent's decryption against ground truth plaintext.

    For multisym cipher output, the decrypted text may have spaces between
    letters and ' | ' between words. We normalize both sides before comparing.
    """
    # Normalize the decrypted text
    # For multisym output: split on word separator, collapse each word, rejoin
    if " | " in decrypted:
        word_parts = decrypted.split(" | ")
        collapsed_words = [_collapse_spaced_letters(w) for w in word_parts]
        dec_clean = " ".join(collapsed_words)
    else:
        dec_clean = _collapse_spaced_letters(decrypted)
    dec_clean = normalize_text(dec_clean)

    gt_clean = normalize_text(ground_truth)

    # Character-level accuracy (aligned comparison)
    dec_chars = dec_clean.replace(" ", "")
    gt_chars = gt_clean.replace(" ", "")

    min_len = min(len(dec_chars), len(gt_chars))
    max_len = max(len(dec_chars), len(gt_chars))

    if max_len == 0:
        char_accuracy = 1.0
        correct_chars = 0
    else:
        correct = sum(1 for i in range(min_len) if dec_chars[i] == gt_chars[i])
        correct_chars = correct
        char_accuracy = correct / max_len

    # Word-level accuracy
    dec_words = dec_clean.split()
    gt_words = gt_clean.split()

    min_words = min(len(dec_words), len(gt_words))
    max_words = max(len(dec_words), len(gt_words))

    if max_words == 0:
        word_accuracy = 1.0
        correct_words = 0
    else:
        correct_w = sum(1 for i in range(min_words) if dec_words[i] == gt_words[i])
        correct_words = correct_w
        word_accuracy = correct_w / max_words

    return ScoreResult(
        test_id=test_id,
        char_accuracy=char_accuracy,
        word_accuracy=word_accuracy,
        total_chars=max_len,
        correct_chars=correct_chars,
        total_words=max_words,
        correct_words=correct_words,
        agent_score=agent_score,
        status=status,
    )


def _collapse_spaced_letters(text: str) -> str:
    """Collapse sequences like 'Q U E D A M' into 'QUEDAM'.

    A spaced-letter sequence is 3+ single uppercase letters separated by
    single spaces, not part of a longer word.
    """
    # Match sequences of single letters separated by single spaces
    def replacer(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    # Pattern: 3+ single letters (A-Z or ?) separated by single spaces
    return re.sub(r"(?<!\S)([A-Z?] ){2,}[A-Z?](?!\S)", replacer, text)


def score_branch_decryptions(
    test_id: str,
    branches: list[tuple[str, str, int]],  # (branch_name, decryption, mapped_count)
    ground_truth: str,
) -> list[dict]:
    """Score every branch against ground truth. Returns list of dicts with
    branch_name, char_accuracy, word_accuracy, mapped_count."""
    results = []
    for branch_name, decryption, mapped_count in branches:
        sr = score_decryption(test_id, decryption, ground_truth, 0.0, "")
        results.append({
            "branch": branch_name,
            "char_accuracy": sr.char_accuracy,
            "word_accuracy": sr.word_accuracy,
            "mapped_count": mapped_count,
            "correct_chars": sr.correct_chars,
            "total_chars": sr.total_chars,
            "correct_words": sr.correct_words,
            "total_words": sr.total_words,
        })
    results.sort(key=lambda r: r["char_accuracy"], reverse=True)
    return results


def has_word_boundaries(text: str) -> bool:
    """Return True if normalized text contains multiple words (has spaces)."""
    return " " in normalize_text(text)


def format_char_diff(decoded: str, ground_truth: str, context: int = 10) -> str:
    """Show character-level mismatches for no-word-boundary ciphers.

    Finds all positions where decoded and ground truth differ and prints
    each mismatch with surrounding context.
    """
    if " | " in decoded:
        dec_chars = "".join(_collapse_spaced_letters(w) for w in decoded.split(" | "))
    else:
        dec_chars = _collapse_spaced_letters(decoded)
    dec_chars = normalize_text(dec_chars).replace(" ", "")
    gt_chars = normalize_text(ground_truth).replace(" ", "")

    n = min(len(dec_chars), len(gt_chars))
    extra = abs(len(dec_chars) - len(gt_chars))

    # Find mismatch spans (merge nearby mismatches within context distance)
    mismatches = [i for i in range(n) if dec_chars[i] != gt_chars[i]]
    if not mismatches and extra == 0:
        return "  (exact match)"

    lines = []
    total_errors = len(mismatches) + extra
    lines.append(f"  {total_errors} character error(s) out of {max(len(dec_chars), len(gt_chars))}")

    # Group mismatches into spans separated by more than context*2 matching chars
    groups: list[tuple[int, int]] = []
    if mismatches:
        start = mismatches[0]
        end = mismatches[0]
        for pos in mismatches[1:]:
            if pos <= end + context * 2:
                end = pos
            else:
                groups.append((start, end))
                start = pos
                end = pos
        groups.append((start, end))

    for span_start, span_end in groups[:12]:
        lo = max(0, span_start - context)
        hi = min(n, span_end + context + 1)
        dec_seg = dec_chars[lo:hi]
        gt_seg = gt_chars[lo:hi]
        # Mark differing positions with brackets
        dec_marked = ""
        gt_marked = ""
        for i, (d, g) in enumerate(zip(dec_seg, gt_seg)):
            if d != g:
                dec_marked += f"[{d}]"
                gt_marked += f"[{g}]"
            else:
                dec_marked += d
                gt_marked += g
        lines.append(f"  pos {lo+1:>4}-{hi:>4}  dec: {dec_marked}")
        lines.append(f"           gt:  {gt_marked}")

    if len(groups) > 12:
        lines.append(f"  ... ({len(groups) - 12} more error spans not shown)")
    if extra:
        lines.append(f"  length mismatch: decoded={len(dec_chars)}, ground_truth={len(gt_chars)}")

    return "\n".join(lines)


def format_alignment(decoded: str, ground_truth: str, max_words: int = 60) -> str:
    """Return a compact word-by-word alignment table for display.

    Decryption and ground truth are compared position-by-position (same
    logic as score_decryption). Matching words are shown with ✓, mismatches
    with the ground-truth word in brackets.
    """
    if " | " in decoded:
        word_parts = decoded.split(" | ")
        dec_words = [_collapse_spaced_letters(w) for w in word_parts]
    else:
        dec_words = _collapse_spaced_letters(decoded).split()
    dec_words = normalize_text(" ".join(dec_words)).split()
    gt_words = normalize_text(ground_truth).split()

    n = max(len(dec_words), len(gt_words))
    lines = []
    shown = min(n, max_words)
    col_w = max(12, max((len(w) for w in dec_words[:shown]), default=6),
                max((len(w) for w in gt_words[:shown]), default=6))
    col_w = min(col_w, 20)

    header = f"  {'#':>4}  {'Decoded':<{col_w}}  {'Ground truth':<{col_w}}  Match"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for i in range(shown):
        dw = dec_words[i] if i < len(dec_words) else "(missing)"
        gw = gt_words[i] if i < len(gt_words) else "(missing)"
        match = "✓" if dw == gw else f"✗ [{gw}]"
        lines.append(f"  {i:>4}  {dw:<{col_w}}  {gw:<{col_w}}  {match}")

    if n > max_words:
        lines.append(f"  ... ({n - max_words} more words not shown)")

    return "\n".join(lines)


def format_report(scores: list[ScoreResult]) -> str:
    """Format a summary report of benchmark results."""
    if not scores:
        return "No results to report."

    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Test ID':<40} {'Status':<8} {'Char%':>6} {'Word%':>6} {'Agent%':>7}")
    lines.append("-" * 80)

    for s in scores:
        word_str = "  N/A" if s.total_words <= 1 else f"{s.word_accuracy:>5.1%}"
        lines.append(
            f"{s.test_id:<40} {s.status:<8} "
            f"{s.char_accuracy:>5.1%} {word_str} "
            f"{s.agent_score:>6.1%}"
        )

    lines.append("-" * 80)

    # Aggregates
    n = len(scores)
    avg_char = sum(s.char_accuracy for s in scores) / n
    avg_word = sum(s.word_accuracy for s in scores) / n
    avg_agent = sum(s.agent_score for s in scores) / n
    solved = sum(1 for s in scores if s.status == "solved")

    lines.append(
        f"{'AVERAGE':<40} {solved}/{n:<5} "
        f"{avg_char:>5.1%} {avg_word:>5.1%} "
        f"{avg_agent:>6.1%}"
    )
    lines.append("")

    return "\n".join(lines)
