"""Utilities for homophonic/nomenclator candidate renderings.

The functions in this module are intentionally narrow and ground-truth-free.
They keep track of how a symbol-level homophonic key renders into plaintext,
including null masks and optional whole-word/codeword expansions.  Research
scripts and agent tools can use these views to build recurrence packets without
duplicating token-position logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.cipher_text import CipherText


@dataclass(frozen=True)
class TokenView:
    """One cipher-token occurrence projected into rendered plaintext space."""

    symbol: str
    output_start: int
    output_end: int
    rendered: str
    assignment: str


def render_token_views(
    cipher_text: CipherText,
    *,
    key: dict[int, int],
    mask: tuple[str, ...] = (),
    expansions: dict[str, str] | None = None,
) -> tuple[list[TokenView], str]:
    """Render a homophonic key while preserving per-token output positions.

    `mask` symbols render as nulls. `expansions` may map a symbol to a whole
    plaintext string, which is useful for post-solve codeword/logogram review.
    Key values are interpreted as A-Z plaintext IDs.
    """
    masked = set(mask)
    expansions = {
        str(symbol): _clean_expansion(value)
        for symbol, value in (expansions or {}).items()
        if _clean_expansion(value)
    }
    rendered_parts: list[str] = []
    views: list[TokenView] = []
    output_index = 0
    for token_id in cipher_text.tokens:
        symbol = cipher_text.alphabet.decode([token_id])
        if symbol in expansions:
            rendered = expansions[symbol]
            assignment = rendered
        elif symbol in masked:
            rendered = ""
            assignment = "<null>"
        else:
            value = key.get(token_id)
            if value is None or value < 0 or value > 25:
                rendered = ""
                assignment = "?"
            else:
                rendered = chr(ord("A") + value)
                assignment = rendered
        start = output_index
        if rendered:
            rendered_parts.append(rendered)
            output_index += len(rendered)
        views.append(
            TokenView(
                symbol=symbol,
                output_start=start,
                output_end=output_index,
                rendered=rendered,
                assignment=assignment,
            )
        )
    return views, "".join(rendered_parts)


def render_with_expansions(
    cipher_text: CipherText,
    *,
    key: dict[int, int],
    mask: tuple[str, ...] = (),
    expansions: dict[str, str] | None = None,
) -> tuple[str, list[TokenView]]:
    """Return `(plaintext, token_views)` for optional whole-word expansions."""
    views, rendered = render_token_views(
        cipher_text,
        key=key,
        mask=mask,
        expansions=expansions,
    )
    return rendered, views


def suspicious_symbol_groups(
    views: list[TokenView],
    *,
    include_one_letter_symbols: bool,
    min_occurrences: int,
    max_symbols: int,
) -> list[tuple[str, list[TokenView]]]:
    """Group symbols that may warrant null/codeword recurrence review."""
    grouped: dict[str, list[TokenView]] = {}
    for view in views:
        if view.assignment in {"<null>", "?"} or (
            include_one_letter_symbols and len(view.rendered) == 1
        ):
            grouped.setdefault(view.symbol, []).append(view)
    rows = [
        (symbol, symbol_views)
        for symbol, symbol_views in grouped.items()
        if len(symbol_views) >= min_occurrences
    ]
    rows.sort(
        key=lambda item: (
            0 if item[1][0].assignment in {"<null>", "?"} else 1,
            -len(item[1]),
            item[0],
        )
    )
    return rows[: max(0, max_symbols)]


def symbol_context_packet(
    *,
    symbol: str,
    views: list[TokenView],
    baseline: str,
    context_chars: int,
    recurrence_limit: int,
) -> dict[str, Any]:
    """Build a compact recurrence packet for one reviewed symbol."""
    occurrences = []
    for view in views[: max(1, recurrence_limit)]:
        idx = view.output_start
        left = baseline[max(0, idx - context_chars):idx]
        right = baseline[idx:min(len(baseline), idx + context_chars)]
        marker = f"⟦{symbol}:{view.assignment}⟧"
        occurrences.append({
            "output_index": idx,
            "assignment": view.assignment,
            "context": left + marker + right,
        })
    return {
        "symbol": symbol,
        "assignment": views[0].assignment if views else "",
        "occurrence_count": len(views),
        "signal": (
            "missing_or_unmapped_symbol"
            if views and views[0].assignment in {"<null>", "?"}
            else "one_letter_symbol"
        ),
        "occurrences": occurrences,
    }


def reread_occurrences(
    *,
    symbol: str,
    views: list[TokenView],
    expanded: str,
    context_chars: int,
    recurrence_limit: int,
) -> list[str]:
    """Return recurrence snippets after a symbol expansion has been applied."""
    rows: list[str] = []
    for view in [item for item in views if item.symbol == symbol][: max(1, recurrence_limit)]:
        left = expanded[max(0, view.output_start - context_chars):view.output_start]
        right = expanded[view.output_end:min(len(expanded), view.output_end + context_chars)]
        rows.append(left + f"⟦{view.rendered or view.assignment}⟧" + right)
    return rows


def _clean_expansion(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if "A" <= ch <= "Z")
