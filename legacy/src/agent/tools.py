from __future__ import annotations

import json
from typing import Any

from agent.state import AgentState
from analysis import dictionary, frequency, ic, pattern
from models.session import Session

# --- Tool definitions for Claude API ---

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_frequency_analysis",
        "description": "Get frequency analysis of the encoded text. Returns token counts and percentages, sorted by frequency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ngram": {
                    "type": "string",
                    "enum": ["mono", "bigram", "trigram"],
                    "description": "Type of frequency analysis to perform.",
                }
            },
            "required": ["ngram"],
        },
    },
    {
        "name": "get_ic",
        "description": "Get the index of coincidence of the encoded text. English text ≈ 0.0667, random ≈ 0.0385.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_pattern_matches",
        "description": "Get candidate plaintext words for a encoded text word based on its letter pattern (isomorph). Only works if encoded text contains word separators (spaces).",
        "input_schema": {
            "type": "object",
            "properties": {
                "word_index": {
                    "type": "integer",
                    "description": "Index of the encoded text word (0-based) to find pattern matches for.",
                }
            },
            "required": ["word_index"],
        },
    },
    {
        "name": "set_mapping",
        "description": "Map a single encoded text symbol to a plaintext letter. For setting multiple mappings at once, use set_mappings_bulk instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cipher_symbol": {
                    "type": "string",
                    "description": "The encoded text symbol to map.",
                },
                "plain_letter": {
                    "type": "string",
                    "description": "The plaintext letter (A-Z or space) to map it to.",
                },
            },
            "required": ["cipher_symbol", "plain_letter"],
        },
    },
    {
        "name": "set_mappings_bulk",
        "description": "Set multiple symbol-to-letter mappings in one call. Use this for initial frequency-based mapping to set all letters at once. Much faster than calling set_mapping repeatedly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "object",
                    "description": "Object mapping encoded symbols to plaintext letters, e.g. {\"x\": \"E\", \"q\": \"T\", \"m\": \"A\"}",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["mappings"],
        },
    },
    {
        "name": "clear_mapping",
        "description": "Remove the mapping for a encoded text symbol, marking it as unknown again.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cipher_symbol": {
                    "type": "string",
                    "description": "The encoded text symbol to unmap.",
                }
            },
            "required": ["cipher_symbol"],
        },
    },
    {
        "name": "get_current_decryption",
        "description": "Get the current partial decryption of the encoded text using the current key. Unmapped symbols show as '?'.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "score_current_plaintext",
        "description": "Score the current partial decryption against an English dictionary. Returns 0.0-1.0 (fraction of words found). Only meaningful when most symbols are mapped.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_remaining_unmapped",
        "description": "Get lists of unmapped encoded text symbols and unused plaintext letters.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "fix_word",
        "description": "Tell the system that a decoded word is wrong and what it SHOULD be. The system will automatically determine and update the correct symbol-to-letter mappings. This is the PREFERRED way to fix errors — just say what the word should be.",
        "input_schema": {
            "type": "object",
            "properties": {
                "word_index": {
                    "type": "integer",
                    "description": "Index of the word to fix (from get_word_context or get_unrecognized_words).",
                },
                "correct_word": {
                    "type": "string",
                    "description": "What the word SHOULD decode to, e.g. 'THIS' or 'WHOLE'.",
                },
            },
            "required": ["word_index", "correct_word"],
        },
    },
    {
        "name": "swap_decrypted_letters",
        "description": "In the current decryption output, everywhere you see letter A, make it show letter B instead, and vice versa. Works on the OUTPUT letters you see, not encoded symbols. Example: if you see 'THAO' and want 'THIS', you might swap O and S — every O in the output becomes S and every S becomes O.",
        "input_schema": {
            "type": "object",
            "properties": {
                "letter_1": {
                    "type": "string",
                    "description": "First output letter to swap (what you SEE in the decryption).",
                },
                "letter_2": {
                    "type": "string",
                    "description": "Second output letter to swap.",
                },
            },
            "required": ["letter_1", "letter_2"],
        },
    },
    {
        "name": "get_word_context",
        "description": "Get a section of the current decryption showing decoded words with their corresponding encoded words side-by-side. Useful for reading the text and spotting errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_word": {
                    "type": "integer",
                    "description": "Starting word index (0-based).",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of words to show (default 20, max 50).",
                },
            },
            "required": ["start_word"],
        },
    },
    {
        "name": "get_unrecognized_words",
        "description": "List decoded words that are NOT found in the English dictionary. These are likely errors from wrong mappings. Shows each bad word with its index and surrounding context.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "rollback",
        "description": "Undo recent changes by restoring the key to a previous checkpoint. Use this when the score has dropped or changes made things worse. With no arguments, rolls back to the last checkpoint (before the most recent set of changes).",
        "input_schema": {
            "type": "object",
            "properties": {
                "checkpoint_index": {
                    "type": "integer",
                    "description": "Which checkpoint to restore (0 = first, -1 = most recent, -2 = second most recent). Default: -1.",
                },
            },
        },
    },
    {
        "name": "identify_cipher_symbol",
        "description": "Given a position in the decrypted text where a '?' appears or a letter seems wrong, identify which encoded symbol is at that position so you can fix its mapping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "word_index": {
                    "type": "integer",
                    "description": "Word index (0-based) containing the suspect character.",
                },
                "char_index": {
                    "type": "integer",
                    "description": "Character index within the word (0-based).",
                },
            },
            "required": ["word_index", "char_index"],
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against a Session and analysis functions."""

    def __init__(
        self,
        session: Session,
        word_set: set[str],
        pattern_dict: dict[str, list[str]],
        agent_state: AgentState | None = None,
    ) -> None:
        self.session = session
        self.word_set = word_set
        self.pattern_dict = pattern_dict
        self.agent_state = agent_state
        self._ct_words: list[list[int]] | None = None

    def _get_ct_words(self) -> list[list[int]]:
        """Get encoded text split into words (cached)."""
        if self._ct_words is None and self.session.cipher_text is not None:
            self._ct_words = self.session.cipher_text.words
        return self._ct_words or []

    def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call and return a JSON string result."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_get_frequency_analysis(self, args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        ngram = args.get("ngram", "mono")
        alpha = ct.alphabet
        if ngram == "mono":
            counts = frequency.sorted_frequency(ct.tokens)
            return [
                {"symbol": alpha.symbol_for(tid), "count": c,
                 "pct": round(c / len(ct.tokens) * 100, 2)}
                for tid, c in counts
            ]
        elif ngram == "bigram":
            bg = frequency.bigram_frequency(ct.tokens)
            sorted_bg = sorted(bg.items(), key=lambda x: x[1], reverse=True)[:30]
            return [
                {"bigram": f"{alpha.symbol_for(a)}{alpha.symbol_for(b)}",
                 "count": c}
                for (a, b), c in sorted_bg
            ]
        elif ngram == "trigram":
            tg = frequency.trigram_frequency(ct.tokens)
            sorted_tg = sorted(tg.items(), key=lambda x: x[1], reverse=True)[:20]
            return [
                {"trigram": f"{alpha.symbol_for(a)}{alpha.symbol_for(b)}{alpha.symbol_for(c)}",
                 "count": cnt}
                for (a, b, c), cnt in sorted_tg
            ]
        return {"error": f"Unknown ngram type: {ngram}"}

    def _tool_get_ic(self, _args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        val = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)
        return {
            "ic": round(val, 4),
            "english_expected": ic.ENGLISH_IC,
            "random_expected": round(ic.random_ic(ct.alphabet.size), 4),
            "likely_monoalphabetic": ic.is_likely_monoalphabetic(val, ct.alphabet.size),
        }

    def _tool_get_pattern_matches(self, args: dict) -> Any:
        word_index = args["word_index"]
        words = self._get_ct_words()
        if word_index < 0 or word_index >= len(words):
            return {"error": f"Word index {word_index} out of range (0-{len(words)-1})"}
        word_tokens = words[word_index]
        pat = pattern.word_pattern(word_tokens)
        ct = self.session.cipher_text
        word_display = " ".join(
            ct.alphabet.symbol_for(t) for t in word_tokens
        ) if ct else ""
        matches = pattern.match_pattern(pat, self.pattern_dict)
        return {
            "word": word_display,
            "pattern": pat,
            "candidates": matches[:50],
            "total_candidates": len(matches),
        }

    def _tool_set_mapping(self, args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        cipher_sym = args["cipher_symbol"]
        plain_letter = args["plain_letter"].upper()
        if not ct.alphabet.has_symbol(cipher_sym):
            return {"error": f"Unknown encoded symbol: {cipher_sym}"}
        if not self.session.plaintext_alphabet.has_symbol(plain_letter):
            return {"error": f"Unknown plaintext letter: {plain_letter}"}
        ct_id = ct.alphabet.id_for(cipher_sym)
        pt_id = self.session.plaintext_alphabet.id_for(plain_letter)
        self.session.set_mapping(ct_id, pt_id)
        return {"status": "ok", "mapping": f"{cipher_sym} -> {plain_letter}"}

    def _tool_set_mappings_bulk(self, args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        mappings = args.get("mappings", {})
        results = []
        errors = []
        for cipher_sym, plain_letter in mappings.items():
            plain_letter = plain_letter.upper()
            if not ct.alphabet.has_symbol(cipher_sym):
                errors.append(f"Unknown encoded symbol: {cipher_sym}")
                continue
            if not self.session.plaintext_alphabet.has_symbol(plain_letter):
                errors.append(f"Unknown plaintext letter: {plain_letter}")
                continue
            ct_id = ct.alphabet.id_for(cipher_sym)
            pt_id = self.session.plaintext_alphabet.id_for(plain_letter)
            self.session.set_mapping(ct_id, pt_id)
            results.append(f"{cipher_sym} -> {plain_letter}")
        return {
            "status": "ok",
            "mappings_set": len(results),
            "details": results,
            "errors": errors if errors else None,
        }

    def _tool_clear_mapping(self, args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        cipher_sym = args["cipher_symbol"]
        if not ct.alphabet.has_symbol(cipher_sym):
            return {"error": f"Unknown encoded symbol: {cipher_sym}"}
        ct_id = ct.alphabet.id_for(cipher_sym)
        self.session.clear_mapping(ct_id)
        return {"status": "ok", "cleared": cipher_sym}

    def _tool_get_current_decryption(self, _args: dict) -> Any:
        decrypted = self.session.apply_key()
        unmapped_count = len(self.session.unmapped_cipher_ids)
        return {
            "decryption": decrypted,
            "mapped_count": self.session.mapped_count,
            "unmapped_count": unmapped_count,
            "complete": self.session.is_complete,
            "hint": "Read this text carefully. Look for almost-correct words and fix them."
            if self.session.mapped_count > 0 else
            "No mappings yet. Start with frequency-based mapping.",
        }

    def _tool_score_current_plaintext(self, _args: dict) -> Any:
        decrypted = self.session.apply_key()
        score = dictionary.score_plaintext(decrypted, self.word_set)
        words = decrypted.split()
        bad_words = [
            w for w in words
            if (w.upper() not in self.word_set and len(w) > 1) or "?" in w
        ]
        return {
            "score": round(score, 4),
            "total_words": len(words),
            "unrecognized_count": len(bad_words),
            "unrecognized_sample": bad_words[:15],
            "decryption": decrypted,
            "mapped_count": self.session.mapped_count,
            "complete": self.session.is_complete,
        }

    def _tool_get_remaining_unmapped(self, _args: dict) -> Any:
        ct = self.session.cipher_text
        if ct is None:
            return {"error": "No encoded text loaded"}
        unmapped_ct = [
            ct.alphabet.symbol_for(i) for i in self.session.unmapped_cipher_ids
        ]
        unmapped_pt = [
            self.session.plaintext_alphabet.symbol_for(i)
            for i in self.session.unmapped_plain_ids
        ]
        return {
            "unmapped_cipher_symbols": unmapped_ct,
            "unused_plain_letters": unmapped_pt,
        }

    def _decode_word(self, word_tokens: list[int]) -> str:
        """Decode a single word's tokens through the current key."""
        key = self.session.key
        pt_alpha = self.session.plaintext_alphabet
        parts = []
        for t in word_tokens:
            if t in key:
                parts.append(pt_alpha.symbol_for(key[t]))
            else:
                parts.append("?")
        return "".join(parts)

    def _cipher_word_display(self, word_tokens: list[int]) -> str:
        """Display a word in encoded symbols."""
        ct = self.session.cipher_text
        if ct is None:
            return ""
        return "".join(ct.alphabet.symbol_for(t) for t in word_tokens)

    def _tool_get_word_context(self, args: dict) -> Any:
        words = self._get_ct_words()
        if not words:
            return {"error": "No encoded text loaded or no words found"}
        start = args.get("start_word", 0)
        count = min(args.get("count", 20), 50)
        end = min(start + count, len(words))
        if start >= len(words):
            return {"error": f"start_word {start} out of range (0-{len(words)-1})"}

        rows = []
        for i in range(start, end):
            cipher_word = self._cipher_word_display(words[i])
            decoded_word = self._decode_word(words[i])
            rows.append({
                "index": i,
                "cipher": cipher_word,
                "decoded": decoded_word,
            })

        # Also build a readable text snippet
        snippet = " ".join(r["decoded"] for r in rows)
        return {
            "words": rows,
            "readable": snippet,
            "total_words": len(words),
            "showing": f"{start}-{end-1}",
        }

    def _tool_get_unrecognized_words(self, _args: dict) -> Any:
        words = self._get_ct_words()
        if not words:
            return {"error": "No encoded text loaded"}

        bad_words = []
        for i, word_tokens in enumerate(words):
            decoded = self._decode_word(word_tokens)
            if "?" in decoded:
                # Has unmapped symbols — always bad
                context_start = max(0, i - 2)
                context_end = min(len(words), i + 3)
                context = " ".join(
                    self._decode_word(words[j]) for j in range(context_start, context_end)
                )
                bad_words.append({
                    "index": i,
                    "decoded": decoded,
                    "cipher": self._cipher_word_display(word_tokens),
                    "context": context,
                    "reason": "unmapped symbols",
                })
            elif decoded.upper() not in self.word_set and len(decoded) > 1:
                # Not in dictionary
                context_start = max(0, i - 2)
                context_end = min(len(words), i + 3)
                context = " ".join(
                    self._decode_word(words[j]) for j in range(context_start, context_end)
                )
                bad_words.append({
                    "index": i,
                    "decoded": decoded,
                    "cipher": self._cipher_word_display(word_tokens),
                    "context": context,
                    "reason": "not in dictionary",
                })

        return {
            "unrecognized_count": len(bad_words),
            "total_words": len(words),
            "words": bad_words[:30],
        }

    def _tool_identify_cipher_symbol(self, args: dict) -> Any:
        words = self._get_ct_words()
        if not words:
            return {"error": "No encoded text loaded"}
        word_idx = args["word_index"]
        char_idx = args["char_index"]
        if word_idx < 0 or word_idx >= len(words):
            return {"error": f"Word index {word_idx} out of range"}
        word_tokens = words[word_idx]
        if char_idx < 0 or char_idx >= len(word_tokens):
            return {"error": f"Char index {char_idx} out of range for word of length {len(word_tokens)}"}
        ct = self.session.cipher_text
        token_id = word_tokens[char_idx]
        cipher_sym = ct.alphabet.symbol_for(token_id)
        key = self.session.key
        current_mapping = self.session.plaintext_alphabet.symbol_for(key[token_id]) if token_id in key else None
        return {
            "cipher_symbol": cipher_sym,
            "token_id": token_id,
            "current_mapping": current_mapping,
            "word_cipher": self._cipher_word_display(word_tokens),
            "word_decoded": self._decode_word(word_tokens),
        }

    def _tool_fix_word(self, args: dict) -> Any:
        """Fix a word by telling the system what it should decode to.
        Automatically determines and updates the correct mappings."""
        words = self._get_ct_words()
        if not words:
            return {"error": "No encoded text loaded"}
        word_idx = args["word_index"]
        correct = args["correct_word"].upper()
        if word_idx < 0 or word_idx >= len(words):
            return {"error": f"Word index {word_idx} out of range (0-{len(words)-1})"}
        word_tokens = words[word_idx]
        if len(correct) != len(word_tokens):
            return {
                "error": f"Length mismatch: encoded word has {len(word_tokens)} "
                f"characters but '{correct}' has {len(correct)}. "
                f"Cipher word: {self._cipher_word_display(word_tokens)}, "
                f"currently decodes as: {self._decode_word(word_tokens)}"
            }

        ct = self.session.cipher_text
        pt_alpha = self.session.plaintext_alphabet
        changes = []
        conflicts = []

        for i, (token_id, target_char) in enumerate(zip(word_tokens, correct)):
            if not pt_alpha.has_symbol(target_char):
                conflicts.append(f"'{target_char}' is not a valid plaintext letter")
                continue

            pt_id = pt_alpha.id_for(target_char)
            current_key = self.session.key
            current_pt = current_key.get(token_id)
            cipher_sym = ct.alphabet.symbol_for(token_id)

            if current_pt == pt_id:
                continue  # Already correct

            # Check if another encoded symbol is already mapped to this plaintext letter
            reverse_key = {v: k for k, v in current_key.items()}
            if pt_id in reverse_key and reverse_key[pt_id] != token_id:
                # Swap: the other encoded symbol gets our old plaintext mapping
                other_ct_id = reverse_key[pt_id]
                other_cipher_sym = ct.alphabet.symbol_for(other_ct_id)
                old_pt_sym = pt_alpha.symbol_for(current_pt) if current_pt is not None else "?"
                self.session.set_mapping(other_ct_id, current_pt if current_pt is not None else pt_id)
                changes.append(
                    f"  {cipher_sym} -> {target_char} (was {old_pt_sym}), "
                    f"swapped {other_cipher_sym} -> {old_pt_sym}"
                )
            else:
                old_pt_sym = pt_alpha.symbol_for(current_pt) if current_pt is not None else "?"
                changes.append(f"  {cipher_sym} -> {target_char} (was {old_pt_sym})")

            self.session.set_mapping(token_id, pt_id)

        # Show result
        new_decoded = self._decode_word(word_tokens)
        return {
            "status": "ok",
            "word_index": word_idx,
            "cipher_word": self._cipher_word_display(word_tokens),
            "now_decodes_as": new_decoded,
            "changes": changes,
            "conflicts": conflicts if conflicts else None,
        }

    def _tool_swap_decrypted_letters(self, args: dict) -> Any:
        """Swap two letters in the OUTPUT. Finds which encoded symbols currently
        produce these letters and swaps their plaintext assignments."""
        letter1 = args["letter_1"].upper()
        letter2 = args["letter_2"].upper()
        pt_alpha = self.session.plaintext_alphabet

        if not pt_alpha.has_symbol(letter1):
            return {"error": f"'{letter1}' is not a valid plaintext letter"}
        if not pt_alpha.has_symbol(letter2):
            return {"error": f"'{letter2}' is not a valid plaintext letter"}

        pt_id1 = pt_alpha.id_for(letter1)
        pt_id2 = pt_alpha.id_for(letter2)

        # Find which encoded symbols are currently mapped to these plaintext letters
        key = self.session.key
        ct_id_for_letter1 = None
        ct_id_for_letter2 = None
        for ct_id, pt_id in key.items():
            if pt_id == pt_id1:
                ct_id_for_letter1 = ct_id
            if pt_id == pt_id2:
                ct_id_for_letter2 = ct_id

        if ct_id_for_letter1 is None:
            return {"error": f"No encoded symbol is currently mapped to '{letter1}'"}
        if ct_id_for_letter2 is None:
            return {"error": f"No encoded symbol is currently mapped to '{letter2}'"}

        ct = self.session.cipher_text
        sym1 = ct.alphabet.symbol_for(ct_id_for_letter1)
        sym2 = ct.alphabet.symbol_for(ct_id_for_letter2)

        # Do the swap
        self.session.set_mapping(ct_id_for_letter1, pt_id2)
        self.session.set_mapping(ct_id_for_letter2, pt_id1)

        # Show a preview of the effect
        decrypted = self.session.apply_key()
        preview = decrypted[:300]

        return {
            "status": "ok",
            "swapped": f"Every '{letter1}' in output is now '{letter2}' and vice versa",
            "details": f"cipher '{sym1}' now -> {letter2}, cipher '{sym2}' now -> {letter1}",
            "decryption_preview": preview,
        }

    def _tool_rollback(self, args: dict) -> Any:
        """Restore the key to a previous checkpoint."""
        if self.agent_state is None:
            return {"error": "No state tracking available"}
        if not self.agent_state.checkpoints:
            return {"error": "No checkpoints available to roll back to"}

        index = args.get("checkpoint_index", -1)
        checkpoint = self.agent_state.get_checkpoint(index)
        if checkpoint is None:
            return {"error": f"Checkpoint index {index} not found"}

        # Restore the key
        self.session.set_full_key(checkpoint.key)
        decrypted = self.session.apply_key()
        score = dictionary.score_plaintext(decrypted, self.word_set)

        return {
            "status": "ok",
            "restored_to": checkpoint.label,
            "restored_score": round(checkpoint.score, 4),
            "current_score": round(score, 4),
            "checkpoints_available": len(self.agent_state.checkpoints),
            "decryption_preview": decrypted[:300],
        }
