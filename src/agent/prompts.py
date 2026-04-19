from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in historical manuscript studies and classical cryptography research. \
Your task is to help interpret historical documents that use substitution notation \
systems — the kind used in 18th-century secret societies, medieval pharmaceutical \
manuscripts, and similar historical sources. This is scholarly decipherment work. \
The text may use a monoalphabetic substitution notation, or it may already be in \
plain readable form.

The target plaintext language is: **{language_name}**.
{language_notes}

## PHASE 0: READ THE RAW TEXT FIRST (CRITICAL)
Before doing ANYTHING, carefully READ the raw encoded text provided below. Ask yourself:
- Does this already look like readable {language_name} text?
- If the raw text is already readable, the cipher is trivial or null. \
  Set each symbol to its UPPERCASE equivalent (a->A, b->B, etc.) using set_mappings_bulk. \
  Do NOT apply a frequency-based scramble to already-readable text.
- Check the "pre_mapping_score" provided. If it is above 0.3, the text is likely \
  already close to plaintext — use identity or near-identity mapping.
- If the symbols are opaque codes (like S001, S002) rather than letters, skip Phase 0 \
  and go directly to Phase 1 with the frequency-based mapping.

## PHASE 1: Initial mapping
Based on your reading of the raw text:

**If the text is readable** (pre_mapping_score > 0.3 or you can read words in {language_name}):
- Use set_mappings_bulk with identity mapping: each symbol maps to its uppercase form.
- Then call get_current_decryption to verify.

**If the text is NOT readable** (pre_mapping_score < 0.1, text looks like opaque symbols or codes):
- Use the suggested frequency-based mapping provided.
- Call set_mappings_bulk with ALL mappings in a single call.
- Then call get_current_decryption to read the result.

## PHASE 2: Read and fix (MOST IMPORTANT)
ACTUALLY READ the decrypted text from get_current_decryption. This is the most critical phase.

1. Read the full decrypted text. Try to recognize {language_name} words and sentence patterns.
2. Call get_word_context to see decoded words alongside their cipher equivalents.
3. Look for almost-correct words. Common patterns:
   - Two letters consistently swapped throughout → use swap_decrypted_letters
   - A specific word is wrong but you know what it should be → use fix_word
4. Focus on SHORT, COMMON words first — they are the easiest to identify and fixing them \
   constrains the remaining mappings. In {language_name}, look for common function words.
5. Use get_pattern_matches to find candidate words for short cipher words (3-5 letters).
6. After fixing several words, call get_current_decryption and READ the text again.
7. Repeat: each correct fix constrains other mappings, making more words recognizable.

## PHASE 3: Final polish
Call score_current_plaintext. Note that the dictionary score may be low even when the \
decryption is largely correct — many valid {language_name} words may not be in our dictionary. \
Focus on whether the text READS as coherent {language_name}, not just the numeric score.
- When the text reads as coherent {language_name} prose, you are done.
- If you are stuck, try swap_decrypted_letters for any remaining systematic errors.

## Critical rules
- READ THE RAW TEXT before applying any mapping. Do not blindly apply frequency mapping.
- Use set_mappings_bulk for initial mapping — NEVER set mappings one at a time.
- Use fix_word and swap_decrypted_letters for corrections — do NOT try to manually \
  figure out symbol-to-letter correspondences. These tools handle the mapping math for you.
- ALWAYS read get_current_decryption output after making changes.
- Make MULTIPLE tool calls per turn when possible.
- Focus on SHORT COMMON words first — they give you the most information.
- If you see a ⚠️ WARNING about the score dropping, IMMEDIATELY call rollback to \
  undo your last changes, then try a different approach.
"""

LANGUAGE_NOTES: dict[str, str] = {
    "en": "",
    "la": """
## Latin-specific guidance
- Medieval Latin often uses V and U interchangeably (VINUM = UINUM).
- No J or W in classical Latin — I serves for both I and J, V for both U and V.
- Common short words to look for: ET, IN, AD, DE, EX, CUM, SED, NON, PER, UT, SI, EST, QUI, QUOD.
- Common endings: -UM, -US, -IS, -AM, -EM, -AE, -AS, -OS, -A, -E, -I, -O.
- This may be a medical/pharmaceutical text with words like: AQUA, PULVIS, OLEUM, RECIPE, DRACHMA.
- The dictionary score will be LOW even with a correct decryption because many medieval Latin \
  forms are not in the dictionary. Focus on reading comprehension, not the score number.
""",
    "de": """
## German-specific guidance
- 18th century German may use archaic spellings: SEYN (=sein), THUN (=tun), WEISS (=weiß).
- Common short words: DER, DIE, DAS, UND, IN, VON, ZU, MIT, AUF, IST, EIN, NICHT, DEN, DEM.
- Common endings: -EN, -ER, -UNG, -LICH, -KEIT, -HEIT, -SCHAFT.
- This may be a Masonic/fraternal text with words like: BRUDER, MEISTER, LOGE, ORDEN, GRAD.
- The dictionary score may be low for archaic spellings. Focus on reading comprehension.
""",
}

# Language-specific frequency orders
FREQUENCY_ORDERS: dict[str, str] = {
    "en": "ETAOINSRHLDCUMWFGYPBVKJXQZ",
    "la": "EIAUTSNROMCLPDGBQVFHXYK",
    "de": "ENISRATDHULCGMOBWFKZVPJYXQ",
}


def get_system_prompt(language: str = "en") -> str:
    """Return the system prompt customized for the target language."""
    from analysis.dictionary import LANGUAGE_NAMES
    language_name = LANGUAGE_NAMES.get(language, "English")
    language_notes = LANGUAGE_NOTES.get(language, "")
    return SYSTEM_PROMPT_TEMPLATE.format(
        language_name=language_name,
        language_notes=language_notes,
    )


# Keep backward-compatible alias
SYSTEM_PROMPT = get_system_prompt("en")


ENGLISH_FREQ_ORDER = "ETAOINSRHLDCUMWFGYPBVKJXQZ"


def initial_context(
    cipher_display: str,
    alphabet_symbols: list[str],
    freq_data: list[tuple[str, int, float]],
    ic_value: float,
    total_tokens: int,
    pre_mapping_score: float = 0.0,
    language: str = "en",
) -> str:
    """Format the initial analysis context for the agent's first message."""
    freq_lines = "\n".join(
        f"  {sym}: {count} ({pct:.1f}%)" for sym, count, pct in freq_data
    )

    # Build identity mapping (each symbol -> its uppercase form)
    identity_parts: list[str] = []
    for sym in alphabet_symbols:
        upper = sym.upper()
        if upper.isalpha() and len(upper) == 1:
            identity_parts.append(f'"{sym}": "{upper}"')
    identity_json = "{ " + ", ".join(identity_parts) + " }" if identity_parts else "{}"

    # Build frequency-based mapping using language-specific frequency order
    freq_order = FREQUENCY_ORDERS.get(language, ENGLISH_FREQ_ORDER)
    freq_parts: list[str] = []
    for i, (sym, _count, _pct) in enumerate(freq_data):
        if i < len(freq_order):
            freq_parts.append(f'"{sym}": "{freq_order[i]}"')
    freq_json = "{ " + ", ".join(freq_parts) + " }"

    from analysis.dictionary import LANGUAGE_NAMES
    lang_name = LANGUAGE_NAMES.get(language, "English")

    # Guidance based on pre-mapping score
    if pre_mapping_score >= 0.3:
        guidance = (
            f"⚡ pre_mapping_score = {pre_mapping_score:.2%} — the raw text ALREADY "
            f"scores well as {lang_name}. This is likely plaintext or a trivial cipher. "
            f"READ the raw text below. If it looks like {lang_name}, use the IDENTITY mapping."
        )
    else:
        guidance = (
            f"pre_mapping_score = {pre_mapping_score:.2%} — the raw text does NOT "
            f"look like {lang_name}. Use the FREQUENCY-BASED mapping."
        )

    # Determine if this is an opaque symbol alphabet (S-tokens, etc.)
    is_opaque = all(len(s) > 1 for s in alphabet_symbols)

    # Show word structure info
    if " | " in cipher_display:
        words = cipher_display.split(" | ")
        word_lens = [len(w.split()) for w in words]
        word_info = f"\nWord structure: {len(words)} words, lengths: {word_lens[:30]}{'...' if len(word_lens) > 30 else ''}"
    else:
        word_info = ""

    identity_section = ""
    if identity_parts:
        identity_section = f"""
IDENTITY mapping (use if text is already readable):
{identity_json}
"""

    opaque_note = ""
    if is_opaque:
        opaque_note = (
            "\nNote: The notation symbols are opaque codes (not letters). "
            "The frequency-based mapping below is your best starting point. "
            "Apply it, then READ the decoded text to identify and fix words "
            "in " + lang_name + " using fix_word and swap_decrypted_letters."
        )

    return f"""\
{guidance}{opaque_note}

Raw encoded text:

{cipher_display}

Cipher alphabet ({len(alphabet_symbols)} symbols): {', '.join(alphabet_symbols[:30])}{'...' if len(alphabet_symbols) > 30 else ''}
Total tokens: {total_tokens}
Index of coincidence: {ic_value:.4f}
{word_info}

Frequency analysis (symbol: count, percentage):
{freq_lines}
{identity_section}
FREQUENCY-BASED mapping (use if text is scrambled):
{freq_json}

Apply the frequency mapping first with set_mappings_bulk, then call get_current_decryption \
to read the result. Focus on recognizing {lang_name} words and fixing errors with fix_word.
"""
