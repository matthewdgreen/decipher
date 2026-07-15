from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping
from typing import Literal


AlignmentOp = Literal["match", "substitute", "insert", "delete"]
DiagnosticUnitKind = Literal[
    "text",
    "known_logogram",
    "unknown_logogram",
    "possible_logogram",
    "unknown_gap",
    "null",
]


@dataclass
class WordAlignmentRow:
    decoded_index: int | None
    ground_truth_index: int | None
    decoded: str | None
    ground_truth: str | None
    op: AlignmentOp


@dataclass
class CharAlignmentRow:
    decoded_index: int | None
    ground_truth_index: int | None
    decoded: str | None
    ground_truth: str | None
    op: AlignmentOp


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


@dataclass
class ReportRow:
    """Lightweight per-test row for the end-of-benchmark report table (F2).

    Built directly from a runner result (``RunResultV2`` / ``AutomatedRunResult``)
    so ``format_report`` does not need agent-only fields (``total_words`` /
    ``agent_score``) that those result types lack. ``word_accuracy`` may be
    ``None`` for no-word-boundary ciphers (rendered as ``N/A``).
    """

    test_id: str
    status: str
    char_accuracy: float
    word_accuracy: float | None = None
    duration: float = 0.0
    cost: float = 0.0


@dataclass(frozen=True)
class DiagnosticPlaintextUnit:
    """Typed post-hoc scoring unit for logogram/null diagnostics.

    This is deliberately separate from the normal benchmark scorer.  It lets
    reports reason about known whole-word symbols, unknown whole-word gaps, and
    nulls without changing the cross-suite character accuracy metric.
    """

    kind: DiagnosticUnitKind
    text: str = ""
    symbol: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class DiagnosticUnknownSpan:
    """A likely ground-truth span explained by an unknown diagnostic unit."""

    unit_index: int
    kind: DiagnosticUnitKind
    symbol: str | None
    decoded_char_offset: int
    ground_truth_start: int | None
    ground_truth_end: int | None
    ground_truth_text: str
    context_before: str
    context_after: str


@dataclass
class DiagnosticScoreResult:
    """Post-hoc diagnostic scoring with typed logogram/null evidence.

    `base_score` is the ordinary shared `ScoreResult` for the rendered text.
    Unknown logograms/gaps are omitted from that rendered text, then annotated
    with likely ground-truth spans from the same edit-aware alignment.  Those
    spans are diagnostic evidence only; they do not add character credit.
    """

    test_id: str
    rendered_text: str
    base_score: ScoreResult
    units: list[DiagnosticPlaintextUnit]
    unknown_spans: list[DiagnosticUnknownSpan]
    known_logogram_count: int
    unknown_logogram_count: int
    null_count: int

    @property
    def inferred_unknown_chars(self) -> int:
        return sum(len(span.ground_truth_text) for span in self.unknown_spans)


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

    # Character-level accuracy. Use edit-aware exact-character alignment so a
    # local insertion/deletion can resynchronize later text. Substitutions and
    # gaps are still errors; only exact matched characters get credit.
    dec_chars = dec_clean.replace(" ", "")
    gt_chars = gt_clean.replace(" ", "")
    char_accuracy, correct_chars, max_len = aligned_char_accuracy(dec_chars, gt_chars)

    # Word-level accuracy. This uses edit-aware exact-word alignment, not a
    # brittle positional zip, so local split/merge/extra-word errors do not
    # cause every subsequent word to score as wrong.
    dec_words = dec_clean.split()
    gt_words = gt_clean.split()
    if len(gt_words) <= 1:
        # Ground truth has no word boundaries (no-boundary cipher).  Any word
        # segmentation the agent adds is useful bookkeeping but cannot be
        # evaluated against an unsegmented ground truth, so word accuracy is
        # not meaningful here.  Return total_words=0 so the report shows N/A.
        word_accuracy, correct_words, max_words = 0.0, 0, 0
    else:
        word_accuracy, correct_words, max_words = aligned_word_accuracy(dec_words, gt_words)

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


def score_decryption_diagnostic(
    test_id: str,
    units: list[DiagnosticPlaintextUnit | Mapping[str, object]],
    ground_truth: str,
    agent_score: float = 0.0,
    status: str = "diagnostic",
    *,
    context: int = 12,
) -> DiagnosticScoreResult:
    """Score typed diagnostic output without changing the default metric.

    Known text and known logograms are rendered into a plaintext string and
    scored with the shared `score_decryption()` path.  Unknown logogram/gap
    units are omitted from the rendered text, then aligned against ground truth
    to infer which deleted plaintext span they may explain.  This is intended
    for post-hoc reports and artifact analyzers; it must not be used inside
    solver selection loops.
    """
    normalized_units = [_coerce_diagnostic_unit(unit) for unit in units]
    rendered_parts: list[str] = []
    unknown_offsets: list[tuple[int, DiagnosticPlaintextUnit]] = []
    known_logograms = 0
    unknown_logograms = 0
    nulls = 0

    for idx, unit in enumerate(normalized_units):
        if unit.kind in {"text", "known_logogram"}:
            if unit.kind == "known_logogram":
                known_logograms += 1
            rendered_parts.append(unit.text)
        elif unit.kind in {"unknown_logogram", "possible_logogram", "unknown_gap"}:
            unknown_logograms += 1
            unknown_offsets.append((idx, unit))
        elif unit.kind == "null":
            nulls += 1
        else:
            raise ValueError(f"unsupported diagnostic unit kind: {unit.kind}")

    rendered_text = "".join(rendered_parts)
    base_score = score_decryption(
        test_id,
        rendered_text,
        ground_truth,
        agent_score,
        status,
    )

    rendered_chars = _scoring_chars(rendered_text)
    gt_chars = _scoring_chars(ground_truth)
    rows = align_char_sequences(rendered_chars, gt_chars)
    unknown_spans = [
        _infer_unknown_span(
            unit_index=idx,
            unit=unit,
            decoded_char_offset=offset,
            rows=rows,
            gt_chars=gt_chars,
            context=context,
        )
        for offset, (idx, unit) in zip(
            _unknown_rendered_offsets(normalized_units),
            unknown_offsets,
        )
    ]

    return DiagnosticScoreResult(
        test_id=test_id,
        rendered_text=rendered_text,
        base_score=base_score,
        units=normalized_units,
        unknown_spans=unknown_spans,
        known_logogram_count=known_logograms,
        unknown_logogram_count=unknown_logograms,
        null_count=nulls,
    )


def _coerce_diagnostic_unit(
    unit: DiagnosticPlaintextUnit | Mapping[str, object],
) -> DiagnosticPlaintextUnit:
    if isinstance(unit, DiagnosticPlaintextUnit):
        return unit
    kind = unit.get("kind")
    if not isinstance(kind, str):
        raise ValueError("diagnostic unit is missing string 'kind'")
    allowed_kinds = {
        "text",
        "known_logogram",
        "unknown_logogram",
        "possible_logogram",
        "unknown_gap",
        "null",
    }
    if kind not in allowed_kinds:
        raise ValueError(f"unsupported diagnostic unit kind: {kind}")
    text = unit.get("text", "")
    symbol = unit.get("symbol")
    confidence = unit.get("confidence")
    if not isinstance(text, str):
        raise ValueError("diagnostic unit 'text' must be a string")
    if symbol is not None and not isinstance(symbol, str):
        raise ValueError("diagnostic unit 'symbol' must be a string when present")
    if confidence is not None and not isinstance(confidence, int | float):
        raise ValueError("diagnostic unit 'confidence' must be numeric when present")
    return DiagnosticPlaintextUnit(
        kind=kind,  # type: ignore[arg-type]
        text=text,
        symbol=symbol,
        confidence=float(confidence) if confidence is not None else None,
    )


def _scoring_chars(text: str) -> str:
    return normalize_text(_collapse_spaced_letters(text)).replace(" ", "")


def _unknown_rendered_offsets(
    units: list[DiagnosticPlaintextUnit],
) -> list[int]:
    offsets: list[int] = []
    rendered_char_offset = 0
    for unit in units:
        if unit.kind in {"text", "known_logogram"}:
            rendered_char_offset += len(_scoring_chars(unit.text))
        elif unit.kind in {"unknown_logogram", "possible_logogram", "unknown_gap"}:
            offsets.append(rendered_char_offset)
    return offsets


def _infer_unknown_span(
    *,
    unit_index: int,
    unit: DiagnosticPlaintextUnit,
    decoded_char_offset: int,
    rows: list[CharAlignmentRow],
    gt_chars: str,
    context: int,
) -> DiagnosticUnknownSpan:
    # Find deleted ground-truth characters that occur exactly at the rendered
    # cursor where this unknown unit was omitted.
    span_indices: list[int] = []
    cursor_seen = False
    for row in rows:
        if row.decoded_index is not None and row.decoded_index < decoded_char_offset:
            continue
        if row.decoded_index is not None and row.decoded_index >= decoded_char_offset:
            if cursor_seen:
                break
            cursor_seen = True
        if row.decoded_index is None and row.ground_truth_index is not None:
            # At offset 0, leading deletions arrive before any decoded row.
            if decoded_char_offset == 0 or cursor_seen:
                span_indices.append(row.ground_truth_index)
                continue
        if cursor_seen and row.decoded_index is not None:
            break

    # For the common middle-of-text case, the loop above sees the next decoded
    # character before the deleted span.  Use the interval between previous and
    # next decoded characters as a second pass.
    if not span_indices:
        before = None
        after = None
        for pos, row in enumerate(rows):
            if row.decoded_index == decoded_char_offset - 1:
                before = pos
            if row.decoded_index == decoded_char_offset and after is None:
                after = pos
        lo = (before + 1) if before is not None else 0
        hi = after if after is not None else len(rows)
        span_indices = [
            row.ground_truth_index
            for row in rows[lo:hi]
            if row.decoded_index is None and row.ground_truth_index is not None
        ]

    if span_indices:
        start = min(span_indices)
        end = max(span_indices) + 1
        span_text = gt_chars[start:end]
    else:
        start = None
        end = None
        span_text = ""

    context_start = max(0, (start if start is not None else 0) - context)
    context_end = min(len(gt_chars), (end if end is not None else 0) + context)
    return DiagnosticUnknownSpan(
        unit_index=unit_index,
        kind=unit.kind,
        symbol=unit.symbol,
        decoded_char_offset=decoded_char_offset,
        ground_truth_start=start,
        ground_truth_end=end,
        ground_truth_text=span_text,
        context_before=gt_chars[context_start:start] if start is not None else "",
        context_after=gt_chars[end:context_end] if end is not None else "",
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


def _decoded_words(decoded: str) -> list[str]:
    if " | " in decoded:
        word_parts = decoded.split(" | ")
        words = [_collapse_spaced_letters(w) for w in word_parts]
        return normalize_text(" ".join(words)).split()
    return normalize_text(_collapse_spaced_letters(decoded)).split()


def _ground_truth_words(ground_truth: str) -> list[str]:
    return normalize_text(ground_truth).split()


def align_char_sequences(
    decoded_chars: str,
    ground_truth_chars: str,
) -> list[CharAlignmentRow]:
    """Globally align decoded and ground-truth character streams.

    This is stricter than "best common subsequence": substitutions are cheaper
    than opening two gaps, so ordinary wrong letters stay wrong letters, but a
    true insertion/deletion can be skipped to recover later exact matches.
    """
    m = len(decoded_chars)
    n = len(ground_truth_chars)
    if m == 0 and n == 0:
        return []

    match_score = 2
    mismatch_score = -1
    gap_score = -1
    scores = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        scores[i][0] = scores[i - 1][0] + gap_score
        back[i][0] = "up"
    for j in range(1, n + 1):
        scores[0][j] = scores[0][j - 1] + gap_score
        back[0][j] = "left"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            exact = decoded_chars[i - 1] == ground_truth_chars[j - 1]
            diag_score = scores[i - 1][j - 1] + (
                match_score if exact else mismatch_score
            )
            up_score = scores[i - 1][j] + gap_score
            left_score = scores[i][j - 1] + gap_score
            candidates: list[tuple[int, int, str]] = []
            if exact:
                candidates.append((diag_score, 4, "diag"))
            if not exact:
                candidates.append((diag_score, 3, "diag"))
            candidates.append((up_score, 2, "up"))
            candidates.append((left_score, 1, "left"))
            best_score, _rank, move = max(candidates)
            scores[i][j] = best_score
            back[i][j] = move

    rows: list[CharAlignmentRow] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            dc = decoded_chars[i - 1]
            gc = ground_truth_chars[j - 1]
            rows.append(CharAlignmentRow(
                decoded_index=i - 1,
                ground_truth_index=j - 1,
                decoded=dc,
                ground_truth=gc,
                op="match" if dc == gc else "substitute",
            ))
            i -= 1
            j -= 1
        elif move == "up":
            rows.append(CharAlignmentRow(
                decoded_index=i - 1,
                ground_truth_index=None,
                decoded=decoded_chars[i - 1],
                ground_truth=None,
                op="insert",
            ))
            i -= 1
        else:
            rows.append(CharAlignmentRow(
                decoded_index=None,
                ground_truth_index=j - 1,
                decoded=None,
                ground_truth=ground_truth_chars[j - 1],
                op="delete",
            ))
            j -= 1
    rows.reverse()
    return rows


def aligned_char_accuracy(
    decoded_chars: str,
    ground_truth_chars: str,
) -> tuple[float, int, int]:
    total_chars = max(len(decoded_chars), len(ground_truth_chars))
    if total_chars == 0:
        return 1.0, 0, 0
    rows = align_char_sequences(decoded_chars, ground_truth_chars)
    correct_chars = sum(1 for row in rows if row.op == "match")
    return correct_chars / total_chars, correct_chars, total_chars


def align_word_sequences(
    decoded_words: list[str],
    ground_truth_words: list[str],
) -> list[WordAlignmentRow]:
    """Globally align decoded words to ground truth with edit-aware resync.

    Exact word matches still define correctness, but local insertions,
    deletions, split words, or merged words can be skipped so later matching
    words line up again.
    """
    m = len(decoded_words)
    n = len(ground_truth_words)
    if m == 0 and n == 0:
        return []

    match_score = 3
    mismatch_score = -2
    gap_score = -1
    scores = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        scores[i][0] = scores[i - 1][0] + gap_score
        back[i][0] = "up"
    for j in range(1, n + 1):
        scores[0][j] = scores[0][j - 1] + gap_score
        back[0][j] = "left"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            exact = decoded_words[i - 1] == ground_truth_words[j - 1]
            diag_score = scores[i - 1][j - 1] + (
                match_score if exact else mismatch_score
            )
            up_score = scores[i - 1][j] + gap_score
            left_score = scores[i][j - 1] + gap_score
            candidates: list[tuple[int, int, str]] = []
            if exact:
                candidates.append((diag_score, 4, "diag"))
            candidates.append((up_score, 3, "up"))
            candidates.append((left_score, 2, "left"))
            if not exact:
                candidates.append((diag_score, 1, "diag"))
            best_score, _rank, move = max(candidates)
            scores[i][j] = best_score
            back[i][j] = move

    rows: list[WordAlignmentRow] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            dw = decoded_words[i - 1]
            gw = ground_truth_words[j - 1]
            rows.append(WordAlignmentRow(
                decoded_index=i - 1,
                ground_truth_index=j - 1,
                decoded=dw,
                ground_truth=gw,
                op="match" if dw == gw else "substitute",
            ))
            i -= 1
            j -= 1
        elif move == "up":
            rows.append(WordAlignmentRow(
                decoded_index=i - 1,
                ground_truth_index=None,
                decoded=decoded_words[i - 1],
                ground_truth=None,
                op="insert",
            ))
            i -= 1
        else:
            rows.append(WordAlignmentRow(
                decoded_index=None,
                ground_truth_index=j - 1,
                decoded=None,
                ground_truth=ground_truth_words[j - 1],
                op="delete",
            ))
            j -= 1
    rows.reverse()
    return rows


def aligned_word_accuracy(
    decoded_words: list[str],
    ground_truth_words: list[str],
) -> tuple[float, int, int]:
    total_words = max(len(decoded_words), len(ground_truth_words))
    if total_words == 0:
        return 1.0, 0, 0
    rows = align_word_sequences(decoded_words, ground_truth_words)
    correct_words = sum(1 for row in rows if row.op == "match")
    return correct_words / total_words, correct_words, total_words


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

    Uses the same edit-aware alignment as score_decryption, then prints
    mismatch/gap spans with surrounding aligned context.
    """
    if " | " in decoded:
        dec_chars = "".join(_collapse_spaced_letters(w) for w in decoded.split(" | "))
    else:
        dec_chars = _collapse_spaced_letters(decoded)
    dec_chars = normalize_text(dec_chars).replace(" ", "")
    gt_chars = normalize_text(ground_truth).replace(" ", "")

    rows = align_char_sequences(dec_chars, gt_chars)
    errors = [i for i, row in enumerate(rows) if row.op != "match"]
    if not errors:
        return "  (exact match)"

    lines = []
    total_chars = max(len(dec_chars), len(gt_chars))
    matches = sum(1 for row in rows if row.op == "match")
    lines.append(
        f"  {len(errors)} character alignment error(s); "
        f"{matches}/{total_chars} exact aligned characters"
    )

    groups: list[tuple[int, int]] = []
    if errors:
        start = errors[0]
        end = errors[0]
        for pos in errors[1:]:
            if pos <= end + context * 2:
                end = pos
            else:
                groups.append((start, end))
                start = pos
                end = pos
        groups.append((start, end))

    for span_start, span_end in groups[:12]:
        lo = max(0, span_start - context)
        hi = min(len(rows), span_end + context + 1)
        dec_marked = ""
        gt_marked = ""
        d_positions = [
            row.decoded_index for row in rows[lo:hi]
            if row.decoded_index is not None
        ]
        g_positions = [
            row.ground_truth_index for row in rows[lo:hi]
            if row.ground_truth_index is not None
        ]
        d_lo = min(d_positions) + 1 if d_positions else "-"
        d_hi = max(d_positions) + 1 if d_positions else "-"
        g_lo = min(g_positions) + 1 if g_positions else "-"
        g_hi = max(g_positions) + 1 if g_positions else "-"
        for row in rows[lo:hi]:
            d = row.decoded or "-"
            g = row.ground_truth or "-"
            if row.op != "match":
                dec_marked += f"[{d}]"
                gt_marked += f"[{g}]"
            else:
                dec_marked += d
                gt_marked += g
        lines.append(f"  dec {d_lo:>4}-{d_hi:<4} gt {g_lo:>4}-{g_hi:<4}  dec: {dec_marked}")
        lines.append(f"                         gt:  {gt_marked}")

    if len(groups) > 12:
        lines.append(f"  ... ({len(groups) - 12} more error spans not shown)")
    if len(dec_chars) != len(gt_chars):
        lines.append(f"  length mismatch: decoded={len(dec_chars)}, ground_truth={len(gt_chars)}")

    return "\n".join(lines)


def format_alignment(decoded: str, ground_truth: str, max_words: int = 60) -> str:
    """Return a compact word-by-word alignment table for display.

    Decryption and ground truth are compared with the same edit-aware word
    alignment used by score_decryption. Matching words are shown with ✓,
    mismatches with the ground-truth word in brackets, and local insertions or
    deletions are shown explicitly so the table can resynchronize.
    """
    dec_words = _decoded_words(decoded)
    gt_words = _ground_truth_words(ground_truth)
    rows = align_word_sequences(dec_words, gt_words)

    n = len(rows)
    lines = []
    shown = min(n, max_words)
    shown_rows = rows[:shown]
    col_w = max(
        12,
        max((len(row.decoded or "") for row in shown_rows), default=6),
        max((len(row.ground_truth or "") for row in shown_rows), default=6),
    )
    col_w = min(col_w, 20)

    header = (
        f"  {'D#':>4}  {'G#':>4}  "
        f"{'Decoded':<{col_w}}  {'Ground truth':<{col_w}}  Match"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for i, row in enumerate(shown_rows):
        dw = row.decoded or "(gap)"
        gw = row.ground_truth or "(gap)"
        if row.op == "match":
            match = "✓"
        elif row.op == "insert":
            match = "↷ decoded extra"
        elif row.op == "delete":
            match = "↶ missing decoded"
        else:
            match = f"✗ [{gw}]"
        d_idx = str(row.decoded_index) if row.decoded_index is not None else "-"
        g_idx = (
            str(row.ground_truth_index)
            if row.ground_truth_index is not None else "-"
        )
        lines.append(f"  {d_idx:>4}  {g_idx:>4}  {dw:<{col_w}}  {gw:<{col_w}}  {match}")

    if n > max_words:
        lines.append(f"  ... ({n - max_words} more alignment row(s) not shown)")

    return "\n".join(lines)


def format_report(rows: list[ReportRow]) -> str:
    """Format the end-of-benchmark per-test report table (F2).

    Takes lightweight :class:`ReportRow` items (built from runner results).
    Columns: Test ID, Status, Char%, Word%, Time, Cost. There is no Agent%
    column (RunResultV2 has no agent self-score). ``solved`` counts both agentic
    ``solved`` and automated ``completed`` statuses.
    """
    if not rows:
        return "No results to report."

    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append("Char% and Word% are comparisons to known ground-truth plaintext.")
    lines.append("")
    lines.append(
        f"{'Test ID':<40} {'Status':<17} {'Char%':>6} {'Word%':>6} "
        f"{'Time':>7} {'Cost':>8}"
    )
    lines.append("-" * 88)

    for r in rows:
        word_str = "   N/A" if r.word_accuracy is None else f"{r.word_accuracy:>5.1%}"
        lines.append(
            f"{r.test_id:<40} {r.status:<17} "
            f"{r.char_accuracy:>5.1%} {word_str} "
            f"{r.duration:>6.1f}s ${r.cost:>7.2f}"
        )

    lines.append("-" * 88)

    # Aggregates
    n = len(rows)
    avg_char = sum(r.char_accuracy for r in rows) / n
    words = [r.word_accuracy for r in rows if r.word_accuracy is not None]
    avg_word = (sum(words) / len(words)) if words else 0.0
    total_time = sum(r.duration for r in rows)
    total_cost = sum(r.cost for r in rows)
    solved = sum(1 for r in rows if r.status in {"solved", "completed"})

    lines.append(
        f"{'AVERAGE':<40} {f'{solved}/{n}':<17} "
        f"{avg_char:>5.1%} {avg_word:>5.1%} "
        f"{total_time:>6.1f}s ${total_cost:>7.2f}"
    )
    lines.append("")

    return "\n".join(lines)
