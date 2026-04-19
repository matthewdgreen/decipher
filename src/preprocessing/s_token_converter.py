"""S-token to letter normalization for API compatibility.

Historical manuscript transcriptions use S-token sequences (S025 S012 S006...)
as canonical symbol identifiers. This module normalizes them to single-letter
format while preserving the substitution structure, improving API compatibility.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple


def convert_s_tokens_to_letters(text: str) -> Tuple[str, Dict[str, str]]:
    """Normalize S-token format to single letters for API compatibility.

    Args:
        text: Text containing S-tokens like "S025 S012 S006 | S003 S007"

    Returns:
        (converted_text, token_mapping)
        converted_text: Same structure with letters "A B C | D E"
        token_mapping: Dict mapping S025->A, S012->B, etc.
    """
    # Find all S-tokens in order of appearance
    s_tokens = re.findall(r'S\d{3}', text)

    # Get unique tokens preserving order
    unique_tokens = list(dict.fromkeys(s_tokens))

    # Create mapping to letters A, B, C, ..., Z, AA, BB, etc.
    token_mapping = {}
    for i, token in enumerate(unique_tokens):
        if i < 26:
            letter = chr(65 + i)  # A-Z
        else:
            # After Z, use AA, BB, CC, etc.
            base_letter = chr(65 + ((i - 26) % 26))
            letter = base_letter * (2 + (i - 26) // 26)
        token_mapping[token] = letter

    # Replace S-tokens with letters
    converted = text
    for s_token, letter in token_mapping.items():
        converted = converted.replace(s_token, letter)

    return converted, token_mapping


def convert_letters_back_to_s_tokens(text: str, token_mapping: Dict[str, str]) -> str:
    """Convert letter format back to S-tokens (for result interpretation).

    Args:
        text: Text with letters "A B C | D E"
        token_mapping: Original mapping from convert_s_tokens_to_letters

    Returns:
        Text with S-tokens restored
    """
    # Reverse the mapping
    reverse_mapping = {letter: s_token for s_token, letter in token_mapping.items()}

    # Sort by letter length (longest first) to handle AA, BB correctly
    letters_by_length = sorted(reverse_mapping.keys(), key=len, reverse=True)

    result = text
    for letter in letters_by_length:
        s_token = reverse_mapping[letter]
        result = result.replace(letter, s_token)

    return result


def is_s_token_format(text: str) -> bool:
    """Check if text contains S-token format."""
    return bool(re.search(r'S\d{3}', text))


def estimate_normalization_benefit(text: str) -> str:
    """Estimate how much S-token normalization would help API compatibility."""
    if not is_s_token_format(text):
        return "low"

    s_token_count = len(re.findall(r'S\d{3}', text))

    if s_token_count <= 20:
        return "low"
    elif s_token_count <= 100:
        return "medium"
    else:
        return "high"


if __name__ == "__main__":
    # Test with Borg cipher sample
    test_text = "S025 S012 S006 S016 S003 S005 | S003 S007 S012 S019 | S005 S009 S010 S009"

    print("Original:", test_text)
    converted, mapping = convert_s_tokens_to_letters(test_text)
    print("Converted:", converted)
    print("Mapping:", mapping)

    # Test reverse conversion
    restored = convert_letters_back_to_s_tokens(converted, mapping)
    print("Restored:", restored)
    print("Match:", test_text == restored)