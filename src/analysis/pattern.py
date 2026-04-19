from __future__ import annotations

import os
from collections import defaultdict


def word_pattern(tokens: list[int]) -> str:
    """Compute the isomorph pattern of a token sequence.

    Maps each unique token to a sequential integer, producing a
    canonical pattern string like "0.1.2.1.0".
    """
    mapping: dict[int, int] = {}
    pattern_parts: list[str] = []
    next_id = 0
    for t in tokens:
        if t not in mapping:
            mapping[t] = next_id
            next_id += 1
        pattern_parts.append(str(mapping[t]))
    return ".".join(pattern_parts)


def split_into_words(
    tokens: list[int], separator_id: int | None
) -> list[list[int]]:
    """Split a token sequence into words on a separator token.

    If separator_id is None, returns the entire sequence as one word.
    """
    if separator_id is None:
        return [tokens]
    words: list[list[int]] = []
    current: list[int] = []
    for t in tokens:
        if t == separator_id:
            if current:
                words.append(current)
                current = []
        else:
            current.append(t)
    if current:
        words.append(current)
    return words


def find_isomorphs(
    token_words: list[list[int]],
) -> dict[str, list[list[int]]]:
    """Group words by their isomorph pattern."""
    groups: dict[str, list[list[int]]] = defaultdict(list)
    for word in token_words:
        pat = word_pattern(word)
        groups[pat].append(word)
    return dict(groups)


def build_pattern_dictionary(word_list: list[str]) -> dict[str, list[str]]:
    """Build a pattern -> words mapping from a plaintext word list.

    Each word is converted to uppercase and its letter pattern is computed.
    """
    patterns: dict[str, list[str]] = defaultdict(list)
    for word in word_list:
        w = word.upper().strip()
        if not w:
            continue
        token_ids = list(range(len(w)))
        # Build pattern using character mapping
        char_map: dict[str, int] = {}
        parts: list[str] = []
        next_id = 0
        for ch in w:
            if ch not in char_map:
                char_map[ch] = next_id
                next_id += 1
            parts.append(str(char_map[ch]))
        pat = ".".join(parts)
        patterns[pat].append(w)
    return dict(patterns)


def match_pattern(
    pattern: str, pattern_dict: dict[str, list[str]]
) -> list[str]:
    """Find candidate plaintext words matching a ciphertext word pattern."""
    return pattern_dict.get(pattern, [])


def load_word_list(path: str) -> list[str]:
    """Load a word list file (one word per line)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
