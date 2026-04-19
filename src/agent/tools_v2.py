"""v2 tool definitions and executor.

Tools are organized by namespace:
  workspace_*  — branch lifecycle (fork, list, delete, compare, merge)
  observe_*    — read-only text analysis (frequency, isomorphs)
  decode_*     — read-only views of the transcription (show, unmapped, heatmap)
  score_*      — signal panel and individual signals
  corpus_*     — language data (wordlist, patterns)
  act_*        — branch mutations (set_mapping, bulk_set, anchor_word, clear)
  search_*     — classical algorithms on a branch (hill_climb, anneal)
  meta_*       — declare_solution (terminates the run)

All mutating and reading tools take an explicit branch name.
"""
from __future__ import annotations

import json
import time
from typing import Any

from analysis import dictionary, frequency, ic, ngram, pattern
from analysis import signals as sig
from analysis.frequency import unigram_chi2
from analysis.solver import hill_climb_swaps, simulated_anneal
from artifact.schema import SolutionDeclaration, ToolCall
from models.session import Session  # only used for search tools that take a Session
from workspace import Workspace, WorkspaceError


# ------------------------------------------------------------------
# Tool schemas (Anthropic tool_use)
# ------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ----- workspace_* -----
    {
        "name": "workspace_fork",
        "description": (
            "Create a new branch by copying the key of an existing branch. "
            "Cheap — branches are independent dicts. Use this to test a "
            "hypothesis without committing to main."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "new_name": {"type": "string", "description": "Name for the new branch (alphanumeric, _ or -)."},
                "from_branch": {"type": "string", "description": "Source branch to fork from. Defaults to 'main'.", "default": "main"},
            },
            "required": ["new_name"],
        },
    },
    {
        "name": "workspace_list_branches",
        "description": "List all branches with their mapped-symbol counts and tags.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "workspace_delete",
        "description": "Delete a branch. Cannot delete 'main'.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "workspace_compare",
        "description": (
            "Compare two branches side-by-side: mapping agreements and "
            "disagreements, plus both transcriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch_a": {"type": "string"},
                "branch_b": {"type": "string"},
            },
            "required": ["branch_a", "branch_b"],
        },
    },
    {
        "name": "workspace_merge",
        "description": (
            "Merge mappings from one branch into another. Policies: "
            "'non_conflicting' (default) adds only non-clashing mappings; "
            "'override' — source wins on conflict; 'keep' — destination wins."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_branch": {"type": "string"},
                "into_branch": {"type": "string"},
                "policy": {
                    "type": "string",
                    "enum": ["non_conflicting", "override", "keep"],
                    "default": "non_conflicting",
                },
            },
            "required": ["from_branch", "into_branch"],
        },
    },
    # ----- observe_* -----
    {
        "name": "observe_frequency",
        "description": (
            "Frequency analysis on the encoded text (not on any particular branch)."
            "ngram = 'mono' returns per-symbol counts and percentages; "
            "'bigram' and 'trigram' return the top-N most frequent pairs/triples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ngram": {"type": "string", "enum": ["mono", "bigram", "trigram"], "default": "mono"},
                "top_n": {"type": "integer", "description": "For bigram/trigram, how many to return.", "default": 30},
            },
        },
    },
    {
        "name": "observe_isomorph_clusters",
        "description": (
            "Cipher words grouped by isomorph pattern. Repeated patterns are "
            "highly informative — multiple occurrences of the same pattern "
            "constrain the possible plaintext heavily."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_length": {"type": "integer", "description": "Minimum word length to include.", "default": 2},
                "min_occurrences": {"type": "integer", "description": "Only show patterns occurring at least this many times.", "default": 1},
            },
        },
    },
    {
        "name": "observe_ic",
        "description": "Index of coincidence for the encoded text.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # ----- decode_* -----
    {
        "name": "decode_show",
        "description": (
            "Show the current transcription of a branch as paired rows: encoded"
            "word and decoded word side-by-side. Unmapped symbols render as '?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "start_word": {"type": "integer", "default": 0},
                "count": {"type": "integer", "description": "Max words to show (capped at 50).", "default": 25},
            },
            "required": ["branch"],
        },
    },
    {
        "name": "decode_unmapped_report",
        "description": "Which encoded symbols and plaintext letters are currently unmapped on this branch.",
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    {
        "name": "decode_ngram_heatmap",
        "description": (
            "Per-position n-gram log-probability across the transcription. Lets"
            "you locate regions of low language-model likelihood. Returns the "
            "N worst n-grams by log-prob, each with its character index and span."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "n": {"type": "integer", "enum": [2, 3, 4], "default": 4},
                "worst_k": {"type": "integer", "description": "Return the K worst-scoring n-grams.", "default": 20},
            },
            "required": ["branch"],
        },
    },
    # ----- score_* -----
    {
        "name": "score_panel",
        "description": (
            "Full signal panel for a branch: dictionary_rate, "
            "quadgram_loglik_per_gram, bigram_loglik_per_gram, bigram_chi2, "
            "pattern_consistency, constraint_satisfaction, and mapped counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    {
        "name": "score_quadgram",
        "description": "Mean log10 quadgram probability of the branch's transcription.",
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    {
        "name": "score_dictionary",
        "description": "Dictionary hit-rate of the branch's transcription, plus a sample of unrecognized words.",
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    # ----- corpus_* -----
    {
        "name": "corpus_lookup_word",
        "description": "Is this word in the target-language wordlist? Returns presence and (if found) frequency rank.",
        "input_schema": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
    {
        "name": "corpus_word_candidates",
        "description": (
            "Candidate plaintext words matching an encoded word's isomorph"
            "pattern. If `consistent_with_branch` is set, filters to "
            "candidates whose letter assignments are compatible with existing "
            "mappings on that branch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cipher_word_index": {"type": "integer"},
                "consistent_with_branch": {"type": "string", "description": "Branch to check consistency against. Omit to see all pattern matches."},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["cipher_word_index"],
        },
    },
    # ----- act_* -----
    {
        "name": "act_set_mapping",
        "description": (
            "Set a single encoded-symbol → plaintext-letter mapping on a branch. "
            "cipher_symbol must be the name of a symbol from the cipher alphabet "
            "(e.g. 'S001', 'A', 'X') — NOT a letter you see in the decoded output. "
            "To swap two letters you see in the decoded text, use act_swap_decoded instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_symbol": {"type": "string"},
                "plain_letter": {"type": "string"},
            },
            "required": ["branch", "cipher_symbol", "plain_letter"],
        },
    },
    {
        "name": "act_bulk_set",
        "description": "Set multiple mappings on a branch in one call. Argument `mappings` is an object like {\"S001\": \"E\", \"S002\": \"T\"}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "mappings": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["branch", "mappings"],
        },
    },
    {
        "name": "act_anchor_word",
        "description": (
            "Assert that an encoded word decodes to a specific plaintext word on "
            "a given branch. Directly assigns each cipher symbol to its plaintext "
            "letter. Multiple cipher symbols may map to the same plaintext letter "
            "(homophonic ciphers are fully supported)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_word_index": {"type": "integer"},
                "plaintext": {"type": "string"},
            },
            "required": ["branch", "cipher_word_index", "plaintext"],
        },
    },
    {
        "name": "act_clear_mapping",
        "description": "Remove the mapping for an encoded symbol on a branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_symbol": {"type": "string"},
            },
            "required": ["branch", "cipher_symbol"],
        },
    },
    {
        "name": "act_swap_decoded",
        "description": (
            "Swap two decoded letters throughout the text by exchanging the cipher "
            "symbols that produce them. Use this when you see a letter in the decoded "
            "output that should be a different letter — e.g. if you see T where F "
            "should appear, call act_swap_decoded(branch, 'T', 'F'). This finds "
            "which cipher symbols currently produce each letter and swaps their "
            "plaintext assignments, preserving bijectivity. "
            "If one letter has no mapping yet, it is simply remapped to the target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "letter_a": {"type": "string", "description": "Decoded letter to swap (as it appears in the decoded text)."},
                "letter_b": {"type": "string", "description": "The letter it should become."},
            },
            "required": ["branch", "letter_a", "letter_b"],
        },
    },
    # ----- search_* -----
    {
        "name": "search_hill_climb",
        "description": (
            "Run hill-climbing on a branch's key. Any symbols you have already "
            "anchored are preserved; unmapped symbols are auto-seeded randomly "
            "so the climber works on the full cipher at once. "
            "score_fn: 'dictionary' | 'quadgram' | 'combined'. "
            "Can be called early — even on an empty branch — to get a complete "
            "starting solution that you can then refine. Works for both "
            "bijective and homophonic ciphers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "rounds": {"type": "integer", "default": 30},
                "restarts": {"type": "integer", "default": 5, "description": "Number of random restarts; best result kept."},
                "score_fn": {"type": "string", "enum": ["dictionary", "quadgram", "combined"], "default": "quadgram"},
            },
            "required": ["branch"],
        },
    },
    {
        "name": "search_anneal",
        "description": (
            "Simulated annealing on a branch's key. Stronger than hill_climb "
            "at escaping wrong local optima — accepts temporarily worse moves "
            "to explore a broader solution space. "
            "Mixes single-symbol reassignment (70%) and 2-symbol swaps (30%). "
            "Any symbols you have already anchored are preserved. "
            "score_fn: 'dictionary' | 'quadgram' | 'combined' (default 'combined'). "
            "Use when hill_climb has converged but the decoded text still looks "
            "wrong, or as a more thorough first search. "
            "After annealing, optionally run search_hill_climb to refine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "steps": {
                    "type": "integer",
                    "default": 5000,
                    "description": "Annealing steps per restart.",
                },
                "restarts": {
                    "type": "integer",
                    "default": 3,
                    "description": "Independent restarts; best result kept.",
                },
                "t_start": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Initial temperature. Higher = more exploration early on.",
                },
                "t_end": {
                    "type": "number",
                    "default": 0.005,
                    "description": "Final temperature.",
                },
                "score_fn": {
                    "type": "string",
                    "enum": ["dictionary", "quadgram", "combined"],
                    "default": "combined",
                },
            },
            "required": ["branch"],
        },
    },
    # ----- meta_* -----
    {
        "name": "meta_declare_solution",
        "description": (
            "Terminate the run. Specify the branch whose transcription you want"
            "to submit, a rationale explaining your reasoning and remaining "
            "uncertainty, and your self-confidence (0.0 - 1.0). This ends "
            "the session — call it when you believe you have the best answer "
            "you can produce or when further progress seems impossible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "rationale": {"type": "string"},
                "self_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["branch", "rationale", "self_confidence"],
        },
    },
]


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


class WorkspaceToolExecutor:
    """Dispatches v2 tool calls against a Workspace + language resources."""

    def __init__(
        self,
        workspace: Workspace,
        language: str,
        word_set: set[str],
        word_list: list[str],
        pattern_dict: dict[str, list[str]],
    ) -> None:
        self.workspace = workspace
        self.language = language
        self.word_set = word_set
        self.word_list = word_list
        self.pattern_dict = pattern_dict

        # Frequency rank for lookup (1-based; lower = more common)
        self._freq_rank: dict[str, int] = {
            w.upper(): i + 1 for i, w in enumerate(word_list)
        }

        # Termination state
        self.terminated: bool = False
        self.solution: SolutionDeclaration | None = None

        # Log of all tool calls for the run artifact
        self.call_log: list[ToolCall] = []
        self._current_iteration: int = 0

    def set_iteration(self, n: int) -> None:
        self._current_iteration = n

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_use_id: str = "",
    ) -> str:
        handler = getattr(self, f"_tool_{tool_name}", None)
        started = time.time()
        if handler is None:
            result = _json({"error": f"Unknown tool: {tool_name}"})
        else:
            try:
                result_obj = handler(args)
                result = _json(result_obj)
            except WorkspaceError as e:
                result = _json({"error": f"Workspace error: {e}"})
            except Exception as e:  # noqa: BLE001
                result = _json({"error": f"{type(e).__name__}: {e}"})
        elapsed_ms = int((time.time() - started) * 1000)
        self.call_log.append(
            ToolCall(
                iteration=self._current_iteration,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                arguments=dict(args),
                result=result,
                elapsed_ms=elapsed_ms,
            )
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _alpha(self):
        return self.workspace.cipher_text.alphabet

    def _pt_alpha(self):
        return self.workspace.plaintext_alphabet

    def _cipher_word_str(self, word_tokens: list[int]) -> str:
        alpha = self._alpha()
        sep = " " if alpha._multisym else ""
        return sep.join(alpha.symbol_for(t) for t in word_tokens)

    def _decode_word(self, word_tokens: list[int], branch_name: str) -> str:
        branch = self.workspace.get_branch(branch_name)
        pt_alpha = self._pt_alpha()
        sep = " " if pt_alpha._multisym else ""
        parts = []
        for t in word_tokens:
            if t in branch.key:
                parts.append(pt_alpha.symbol_for(branch.key[t]))
            else:
                parts.append("?")
        return sep.join(parts)

    def _used_ct_ids(self) -> set[int]:
        return set(self.workspace.cipher_text.tokens)

    def _decoded_preview(self, branch_name: str, max_words: int = 30) -> str:
        """Render the first max_words of the branch's current decode.

        Included on mutation and search tool results so the LLM sees the
        textual effect of its action without a follow-up decode_show call.
        """
        ws = self.workspace
        words = ws.cipher_text.words
        branch = ws.get_branch(branch_name)
        pt_alpha = self._pt_alpha()
        sep = " " if pt_alpha._multisym else ""
        out = []
        for w in words[:max_words]:
            parts = [
                pt_alpha.symbol_for(branch.key[t]) if t in branch.key else "?"
                for t in w
            ]
            out.append(sep.join(parts))
        return " | ".join(out)

    # ------------------------------------------------------------------
    # workspace_*
    # ------------------------------------------------------------------
    def _tool_workspace_fork(self, args: dict) -> Any:
        new_name = args["new_name"]
        from_branch = args.get("from_branch", "main")
        branch = self.workspace.fork(new_name, from_branch=from_branch)
        return {
            "status": "ok",
            "created": new_name,
            "parent": from_branch,
            "inherited_mapped_count": len(branch.key),
        }

    def _tool_workspace_list_branches(self, _args: dict) -> Any:
        return {"branches": self.workspace.list_branches()}

    def _tool_workspace_delete(self, args: dict) -> Any:
        name = args["name"]
        self.workspace.delete(name)
        return {"status": "ok", "deleted": name}

    def _tool_workspace_compare(self, args: dict) -> Any:
        return self.workspace.compare(args["branch_a"], args["branch_b"])

    def _tool_workspace_merge(self, args: dict) -> Any:
        return self.workspace.merge(
            args["from_branch"],
            args["into_branch"],
            policy=args.get("policy", "non_conflicting"),
        )

    # ------------------------------------------------------------------
    # observe_*
    # ------------------------------------------------------------------
    def _tool_observe_frequency(self, args: dict) -> Any:
        ngram_type = args.get("ngram", "mono")
        top_n = args.get("top_n", 30)
        ct = self.workspace.cipher_text
        alpha = self._alpha()
        if ngram_type == "mono":
            counts = frequency.sorted_frequency(ct.tokens)
            return [
                {
                    "symbol": alpha.symbol_for(tid),
                    "count": c,
                    "pct": round(c / len(ct.tokens) * 100, 2),
                }
                for tid, c in counts
            ]
        elif ngram_type == "bigram":
            bg = frequency.bigram_frequency(ct.tokens)
            items = sorted(bg.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [
                {"bigram": f"{alpha.symbol_for(a)}{alpha.symbol_for(b)}", "count": c}
                for (a, b), c in items
            ]
        elif ngram_type == "trigram":
            tg = frequency.trigram_frequency(ct.tokens)
            items = sorted(tg.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [
                {
                    "trigram": f"{alpha.symbol_for(a)}{alpha.symbol_for(b)}{alpha.symbol_for(c)}",
                    "count": cnt,
                }
                for (a, b, c), cnt in items
            ]
        return {"error": f"unknown ngram type: {ngram_type}"}

    def _tool_observe_isomorph_clusters(self, args: dict) -> Any:
        min_length = args.get("min_length", 2)
        min_occurrences = args.get("min_occurrences", 1)
        words = self.workspace.cipher_text.words
        clusters: dict[str, list[int]] = {}
        for idx, w in enumerate(words):
            if len(w) < min_length:
                continue
            pat = pattern.word_pattern(w)
            clusters.setdefault(pat, []).append(idx)
        output = []
        for pat, indices in clusters.items():
            if len(indices) < min_occurrences:
                continue
            # show cipher form of the first occurrence
            sample = self._cipher_word_str(words[indices[0]])
            output.append({
                "pattern": pat,
                "length": len(pat.split(".")),
                "occurrences": len(indices),
                "word_indices": indices[:10],
                "sample_cipher_word": sample,
            })
        output.sort(key=lambda d: (-d["occurrences"], d["length"]))
        return {"total_patterns": len(output), "clusters": output[:50]}

    def _tool_observe_ic(self, _args: dict) -> Any:
        ct = self.workspace.cipher_text
        val = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)
        return {
            "ic": round(val, 4),
            "english_reference": ic.ENGLISH_IC,
            "random_reference": round(ic.random_ic(ct.alphabet.size), 4),
            "alphabet_size": ct.alphabet.size,
        }

    # ------------------------------------------------------------------
    # decode_*
    # ------------------------------------------------------------------
    def _tool_decode_show(self, args: dict) -> Any:
        branch = args["branch"]
        start = args.get("start_word", 0)
        count = min(args.get("count", 25), 50)
        words = self.workspace.cipher_text.words
        if not words:
            return {"error": "Cipher has no words"}
        if start >= len(words):
            return {"error": f"start_word {start} exceeds total {len(words)}"}
        end = min(start + count, len(words))
        rows = []
        for i in range(start, end):
            rows.append({
                "index": i,
                "cipher": self._cipher_word_str(words[i]),
                "decoded": self._decode_word(words[i], branch),
            })
        return {
            "branch": branch,
            "range": f"{start}-{end - 1}",
            "total_words": len(words),
            "rows": rows,
        }

    def _tool_decode_unmapped_report(self, args: dict) -> Any:
        branch = args["branch"]
        ws = self.workspace
        unmapped_ct = [ws.cipher_text.alphabet.symbol_for(i) for i in ws.unmapped_cipher_ids(branch)]
        unused_pt = [ws.plaintext_alphabet.symbol_for(i) for i in ws.unused_plain_ids(branch)]
        return {
            "branch": branch,
            "unmapped_cipher_symbols": unmapped_ct,
            "unused_plaintext_letters": unused_pt,
            "mapped_count": len(ws.get_branch(branch).key),
        }

    def _tool_decode_ngram_heatmap(self, args: dict) -> Any:
        branch = args["branch"]
        n = args.get("n", 4)
        worst_k = args.get("worst_k", 20)
        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        lp = ngram.NGRAM_CACHE.get(self.language, n)
        items = ngram.ngram_position_logprobs(normalized, lp, n=n)
        # Worst = lowest log-prob
        items_sorted = sorted(items, key=lambda t: t[2])[:worst_k]
        return {
            "branch": branch,
            "n": n,
            "total_positions": len(items),
            "worst": [
                {"char_index": idx, "ngram": gram, "log_prob": round(lp, 4)}
                for idx, gram, lp in items_sorted
            ],
            "normalized_text": normalized[:400] + ("..." if len(normalized) > 400 else ""),
        }

    # ------------------------------------------------------------------
    # score_*
    # ------------------------------------------------------------------
    def _tool_score_panel(self, args: dict) -> Any:
        branch = args["branch"]
        ws = self.workspace
        decrypted = ws.apply_key(branch)
        panel = sig.compute_panel(
            decrypted=decrypted,
            cipher_words=ws.cipher_text.words,
            key=ws.get_branch(branch).key,
            used_ct_ids=self._used_ct_ids(),
            language=self.language,
            word_set=self.word_set,
            pattern_dict=self.pattern_dict,
        )
        return {"branch": branch, "signals": panel.to_dict()}

    def _tool_score_quadgram(self, args: dict) -> Any:
        branch = args["branch"]
        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        lp = ngram.NGRAM_CACHE.get(self.language, 4)
        score = ngram.normalized_ngram_score(normalized, lp, n=4)
        return {
            "branch": branch,
            "quadgram_loglik_per_gram": round(score, 4) if score != float("-inf") else None,
            "note": "Higher (less negative) is more language-like.",
        }

    def _tool_score_dictionary(self, args: dict) -> Any:
        branch = args["branch"]
        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        words = [w for w in normalized.split() if any(c.isalpha() for c in w)]
        if not words:
            return {"branch": branch, "dictionary_rate": 0.0, "total_words": 0}
        hits = sum(1 for w in words if w in self.word_set)
        unrecognized = [w for w in words if w not in self.word_set]
        return {
            "branch": branch,
            "dictionary_rate": round(hits / len(words), 4),
            "total_words": len(words),
            "recognized": hits,
            "unrecognized_sample": unrecognized[:20],
        }

    # ------------------------------------------------------------------
    # corpus_*
    # ------------------------------------------------------------------
    def _tool_corpus_lookup_word(self, args: dict) -> Any:
        w = args["word"].upper()
        present = w in self.word_set
        return {
            "word": w,
            "in_dictionary": present,
            "frequency_rank": self._freq_rank.get(w),
            "language": self.language,
        }

    def _tool_corpus_word_candidates(self, args: dict) -> Any:
        idx = args["cipher_word_index"]
        consistent_with = args.get("consistent_with_branch")
        limit = args.get("limit", 50)
        words = self.workspace.cipher_text.words
        if idx < 0 or idx >= len(words):
            return {"error": f"cipher_word_index {idx} out of range (0..{len(words) - 1})"}
        word_tokens = words[idx]
        pat = pattern.word_pattern(word_tokens)
        candidates = pattern.match_pattern(pat, self.pattern_dict)

        filtered: list[str] = []
        if consistent_with:
            branch = self.workspace.get_branch(consistent_with)
            pt_alpha = self._pt_alpha()
            for cand in candidates:
                if self._candidate_consistent(word_tokens, cand, branch.key, pt_alpha):
                    filtered.append(cand)
                if len(filtered) >= limit:
                    break
            return {
                "cipher_word_index": idx,
                "cipher_word": self._cipher_word_str(word_tokens),
                "pattern": pat,
                "total_pattern_matches": len(candidates),
                "consistent_with_branch": consistent_with,
                "consistent_candidates": filtered,
            }
        return {
            "cipher_word_index": idx,
            "cipher_word": self._cipher_word_str(word_tokens),
            "pattern": pat,
            "total_pattern_matches": len(candidates),
            "candidates": candidates[:limit],
        }

    def _candidate_consistent(
        self,
        word_tokens: list[int],
        candidate: str,
        key: dict[int, int],
        pt_alpha,
    ) -> bool:
        """Check if assigning candidate to word_tokens is consistent with key."""
        if len(candidate) != len(word_tokens):
            return False
        reverse = {v: k for k, v in key.items()}
        local_fwd: dict[int, int] = {}
        local_rev: dict[int, int] = {}
        for t, ch in zip(word_tokens, candidate):
            if not pt_alpha.has_symbol(ch):
                return False
            pt_id = pt_alpha.id_for(ch)
            # Conflict with existing key?
            if t in key and key[t] != pt_id:
                return False
            if pt_id in reverse and reverse[pt_id] != t:
                return False
            # Conflict within this word?
            if t in local_fwd and local_fwd[t] != pt_id:
                return False
            if pt_id in local_rev and local_rev[pt_id] != t:
                return False
            local_fwd[t] = pt_id
            local_rev[pt_id] = t
        return True

    # ------------------------------------------------------------------
    # act_*
    # ------------------------------------------------------------------
    def _tool_act_set_mapping(self, args: dict) -> Any:
        branch = args["branch"]
        cipher_sym = args["cipher_symbol"]
        plain_letter = args["plain_letter"].upper()
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        if not alpha.has_symbol(cipher_sym):
            return {"error": f"Unknown cipher symbol: {cipher_sym}"}
        if not pt_alpha.has_symbol(plain_letter):
            return {"error": f"Unknown plaintext letter: {plain_letter}"}
        self.workspace.set_mapping(
            branch,
            alpha.id_for(cipher_sym),
            pt_alpha.id_for(plain_letter),
        )
        return {
            "status": "ok",
            "branch": branch,
            "mapping": f"{cipher_sym} -> {plain_letter}",
            "decoded_preview": self._decoded_preview(branch),
        }

    def _tool_act_bulk_set(self, args: dict) -> Any:
        branch = args["branch"]
        mappings = args.get("mappings", {})
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        set_ok: list[str] = []
        errors: list[str] = []
        for cipher_sym, plain_letter in mappings.items():
            pl = plain_letter.upper()
            if not alpha.has_symbol(cipher_sym):
                errors.append(f"unknown cipher symbol: {cipher_sym}")
                continue
            if not pt_alpha.has_symbol(pl):
                errors.append(f"unknown plaintext letter: {pl}")
                continue
            self.workspace.set_mapping(branch, alpha.id_for(cipher_sym), pt_alpha.id_for(pl))
            set_ok.append(f"{cipher_sym}->{pl}")
        return {
            "status": "ok",
            "branch": branch,
            "mappings_set": len(set_ok),
            "details": set_ok,
            "errors": errors if errors else None,
            "decoded_preview": self._decoded_preview(branch),
        }

    def _tool_act_anchor_word(self, args: dict) -> Any:
        branch_name = args["branch"]
        idx = args["cipher_word_index"]
        target = args["plaintext"].upper()
        ws = self.workspace
        words = ws.cipher_text.words
        if idx < 0 or idx >= len(words):
            return {"error": f"cipher_word_index {idx} out of range"}
        word_tokens = words[idx]
        if len(target) != len(word_tokens):
            return {
                "error": "length mismatch",
                "cipher_word_length": len(word_tokens),
                "plaintext_length": len(target),
                "cipher_word": self._cipher_word_str(word_tokens),
            }
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        branch = ws.get_branch(branch_name)
        changes: list[str] = []
        for token_id, target_char in zip(word_tokens, target):
            if not pt_alpha.has_symbol(target_char):
                return {"error": f"Invalid plaintext letter: {target_char}"}
            pt_id = pt_alpha.id_for(target_char)
            if branch.key.get(token_id) == pt_id:
                continue
            branch.key[token_id] = pt_id
            changes.append(f"{alpha.symbol_for(token_id)}->{target_char}")
        decoded = self._decode_word(word_tokens, branch_name)
        return {
            "status": "ok",
            "branch": branch_name,
            "cipher_word": self._cipher_word_str(word_tokens),
            "now_decodes_as": decoded,
            "changes": changes,
            "decoded_preview": self._decoded_preview(branch_name),
        }

    def _tool_act_clear_mapping(self, args: dict) -> Any:
        branch = args["branch"]
        cipher_sym = args["cipher_symbol"]
        alpha = self._alpha()
        if not alpha.has_symbol(cipher_sym):
            return {"error": f"Unknown cipher symbol: {cipher_sym}"}
        self.workspace.clear_mapping(branch, alpha.id_for(cipher_sym))
        return {
            "status": "ok",
            "branch": branch,
            "cleared": cipher_sym,
            "decoded_preview": self._decoded_preview(branch),
        }

    def _tool_act_swap_decoded(self, args: dict) -> Any:
        branch = args["branch"]
        letter_a = args["letter_a"].upper()
        letter_b = args["letter_b"].upper()
        pt_alpha = self._pt_alpha()
        if not pt_alpha.has_symbol(letter_a):
            return {"error": f"Unknown plaintext letter: {letter_a}"}
        if not pt_alpha.has_symbol(letter_b):
            return {"error": f"Unknown plaintext letter: {letter_b}"}
        id_a = pt_alpha.id_for(letter_a)
        id_b = pt_alpha.id_for(letter_b)
        b = self.workspace.get_branch(branch)
        # Find which cipher IDs currently map to each letter
        cipher_for_a = [cid for cid, pid in b.key.items() if pid == id_a]
        cipher_for_b = [cid for cid, pid in b.key.items() if pid == id_b]
        if not cipher_for_a and not cipher_for_b:
            return {"error": f"Neither {letter_a} nor {letter_b} has a current mapping"}
        # Swap: everything that produced A now produces B, and vice versa
        for cid in cipher_for_a:
            self.workspace.set_mapping(branch, cid, id_b)
        for cid in cipher_for_b:
            self.workspace.set_mapping(branch, cid, id_a)
        alpha = self._alpha()
        swapped_a = [alpha.symbol_for(cid) for cid in cipher_for_a]
        swapped_b = [alpha.symbol_for(cid) for cid in cipher_for_b]
        return {
            "status": "ok",
            "branch": branch,
            "swapped": f"{letter_a} ↔ {letter_b}",
            "cipher_symbols_for_a": swapped_a,
            "cipher_symbols_for_b": swapped_b,
            "decoded_preview": self._decoded_preview(branch),
        }

    # ------------------------------------------------------------------
    # search_*
    # ------------------------------------------------------------------

    def _search_declare_note(self) -> str:
        """Return the post-search guidance note, conditioned on whether the
        cipher has word boundaries."""
        no_boundaries = len(self.workspace.cipher_text.words) <= 1
        if no_boundaries:
            return (
                "Read decoded_preview carefully. Segment the stream into words "
                f"and look for non-{self.language} segments — each wrong symbol "
                "mapping corrupts every occurrence of that symbol. Fix the most "
                "obvious errors with act_swap_decoded (to swap two decoded letters) "
                "or act_set_mapping (to assign a specific cipher symbol), then "
                "declare your best branch. Don't exhaust iterations chasing perfection."
            )
        return (
            f"Read decoded_preview carefully. If ANY recognisable {self.language} "
            "words appear, call meta_declare_solution now — a partial solution "
            "scores better than no declaration."
        )

    def _build_score_fns(
        self, temp_session: Any, score_fn_name: str
    ) -> "tuple[Any, Any, Any, Any]":
        """Return (score_dict, score_quad, score_combined, quad_lp) for
        the given session and language.

        score_combined = dict*10 + quadgram - 0.02 * unigram_chi2(language)
        The chi² term penalises decoded text whose letter-frequency
        distribution strays from the target-language reference, making
        the combined score more discriminating between correct solutions
        and plausible-but-wrong local optima.
        """
        quad_lp = ngram.NGRAM_CACHE.get(self.language, 4)
        language = self.language
        word_set = self.word_set

        def score_dict() -> float:
            return dictionary.score_plaintext(temp_session.apply_key(), word_set)

        def score_quad() -> float:
            normalized = sig.normalize_for_scoring(temp_session.apply_key())
            return ngram.normalized_ngram_score(normalized, quad_lp, n=4)

        def score_combined() -> float:
            text = temp_session.apply_key()
            dr = dictionary.score_plaintext(text, word_set)
            normalized = sig.normalize_for_scoring(text)
            quad = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
            chi2 = unigram_chi2(normalized, language)
            return dr * 10 + quad - 0.02 * chi2

        fn = {"dictionary": score_dict, "quadgram": score_quad,
              "combined": score_combined}.get(score_fn_name, score_combined)
        return score_dict, score_quad, score_combined, fn

    def _tool_search_hill_climb(self, args: dict) -> Any:
        branch_name = args["branch"]
        rounds = args.get("rounds", 30)
        restarts = args.get("restarts", 5)
        score_fn_name = args.get("score_fn", "quadgram")
        ws = self.workspace
        branch = ws.get_branch(branch_name)

        import random as _random
        pt_size = ws.plaintext_alphabet.size
        all_ct_ids = sorted(set(ws.cipher_text.tokens))
        anchors = dict(branch.key)  # fixed mappings to preserve across restarts
        seeded_count = len(all_ct_ids) - len(anchors)

        temp_session = _session_for_branch(ws, anchors)
        _, _, _, score_fn = self._build_score_fns(temp_session, score_fn_name)

        # Run multiple random-restart climbs; keep best result.
        # Each restart: start from anchors + fresh random fill for unmapped symbols.
        best_score = float("-inf")
        best_key: dict[int, int] = {}
        before = float("-inf")

        for restart_idx in range(max(1, restarts)):
            seed_key = dict(anchors)
            for ct_id in all_ct_ids:
                if ct_id not in seed_key:
                    seed_key[ct_id] = _random.randrange(pt_size)
            temp_session.set_full_key(seed_key)
            if restart_idx == 0:
                before = score_fn()
            _hill_climb_with_score(temp_session, score_fn, rounds)
            s = score_fn()
            if s > best_score:
                best_score = s
                best_key = dict(temp_session.key)

        temp_session.set_full_key(best_key)

        # Write best key back to the branch
        ws.set_full_key(branch_name, temp_session.key)
        return {
            "branch": branch_name,
            "score_fn": score_fn_name,
            "before": round(before, 4) if before != float("-inf") else None,
            "after": round(best_score, 4) if best_score != float("-inf") else None,
            "improved": best_score > before,
            "rounds_per_restart": rounds,
            "restarts": restarts,
            "auto_seeded_symbols": seeded_count,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "note": (
                self._search_declare_note()
            ),
        }

    def _tool_search_anneal(self, args: dict) -> Any:
        branch_name = args["branch"]
        steps = int(args.get("steps", 5000))
        restarts = int(args.get("restarts", 3))
        t_start = float(args.get("t_start", 1.0))
        t_end = float(args.get("t_end", 0.005))
        score_fn_name = args.get("score_fn", "combined")
        ws = self.workspace
        branch = ws.get_branch(branch_name)

        import random as _random
        pt_size = ws.plaintext_alphabet.size
        all_ct_ids = sorted(set(ws.cipher_text.tokens))
        anchors = dict(branch.key)
        seeded_count = len(all_ct_ids) - len(anchors)

        temp_session = _session_for_branch(ws, anchors)
        _, _, _, score_fn = self._build_score_fns(temp_session, score_fn_name)

        best_score = float("-inf")
        best_key: dict[int, int] = {}
        before = float("-inf")

        for restart_idx in range(max(1, restarts)):
            seed_key = dict(anchors)
            for ct_id in all_ct_ids:
                if ct_id not in seed_key:
                    seed_key[ct_id] = _random.randrange(pt_size)
            temp_session.set_full_key(seed_key)
            if restart_idx == 0:
                before = score_fn()
            s = simulated_anneal(
                temp_session,
                score_fn,
                max_steps=steps,
                t_start=t_start,
                t_end=t_end,
            )
            if s > best_score:
                best_score = s
                best_key = dict(temp_session.key)

        ws.set_full_key(branch_name, best_key)
        return {
            "branch": branch_name,
            "score_fn": score_fn_name,
            "before": round(before, 4) if before != float("-inf") else None,
            "after": round(best_score, 4) if best_score != float("-inf") else None,
            "improved": best_score > before,
            "steps_per_restart": steps,
            "restarts": restarts,
            "auto_seeded_symbols": seeded_count,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "note": (
                self._search_declare_note()
            ),
        }

    # ------------------------------------------------------------------
    # meta_*
    # ------------------------------------------------------------------
    def _tool_meta_declare_solution(self, args: dict) -> Any:
        branch = args["branch"]
        rationale = args["rationale"]
        confidence = float(args["self_confidence"])
        if not self.workspace.has_branch(branch):
            return {"error": f"Branch not found: {branch}"}
        self.solution = SolutionDeclaration(
            branch=branch,
            rationale=rationale,
            self_confidence=confidence,
            declared_at_iteration=self._current_iteration,
        )
        self.terminated = True
        return {
            "status": "ok",
            "accepted": True,
            "branch": branch,
            "declared_at_iteration": self._current_iteration,
            "note": "Run will terminate after this tool result is recorded.",
        }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _session_for_branch(ws: Workspace, key: dict[int, int]) -> Session:
    """Create a headless Session bound to the workspace's cipher, for use
    with solvers that expect a Session interface."""
    s = Session()
    s.plaintext_alphabet = ws.plaintext_alphabet
    s.set_cipher_text(ws.cipher_text)
    s.set_full_key(key)
    return s


def _hill_climb_with_score(session: Session, score_fn, max_rounds: int) -> float:
    """Per-symbol reassignment hill climb using an arbitrary score_fn.

    Delegates to hill_climb_reassign so both bijective and homophonic ciphers
    are handled correctly.
    """
    from analysis.solver import hill_climb_reassign
    return hill_climb_reassign(session, score_fn, max_rounds)
