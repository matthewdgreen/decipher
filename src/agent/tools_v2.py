"""v2 tool definitions and executor.

Tools are organized by namespace:
  workspace_*  — branch lifecycle (fork, list, delete, compare, merge)
  observe_*    — read-only text analysis (frequency, isomorphs)
  decode_*     — read-only views of the transcription (show, unmapped, heatmap, letter_stats, diagnose)
  score_*      — signal panel and individual signals
  corpus_*     — language data (wordlist, patterns)
  act_*        — branch mutations (set_mapping, bulk_set, anchor_word, clear)
  search_*     — classical algorithms on a branch (hill_climb, anneal)
  run_python   — execute arbitrary stdlib Python code (escape hatch)
  meta_*       — declare_solution (terminates the run), request_tool (feedback)

All mutating and reading tools take an explicit branch name.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analysis import dictionary, frequency, homophonic, ic, ngram, pattern
from analysis import signals as sig
from analysis.frequency import unigram_chi2
from analysis.segment import segment_text
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
    {
        "name": "observe_homophone_distribution",
        "description": (
            "Homophonic-cipher diagnostic. Estimates how many cipher symbols "
            "should map to each plaintext letter from reference language "
            "frequencies, and compares that expectation with the current branch "
            "if one is supplied. Use this when the cipher alphabet is larger "
            "than the plaintext alphabet, IC is very low, or many decoded "
            "letters are absent/overrepresented."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Optional branch whose current mapped-symbol counts should be compared.",
                },
            },
        },
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
            "pattern_consistency, constraint_satisfaction, and mapped counts. "
            "dictionary_rate works correctly for BOTH word-boundary and "
            "no-boundary ciphers — for no-boundary text it uses automatic "
            "word segmentation before scoring, so it will return a meaningful "
            "non-zero value even for continuous-letter ciphers."
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
        "description": (
            "Dictionary hit-rate of the branch's transcription, plus a sample "
            "of unrecognized words. For no-boundary ciphers (one continuous run "
            "of letters), automatically segments using a word-boundary finder "
            "before scoring, and returns the segmented text preview and "
            "pseudo-words for diagnosis. Works correctly for all cipher types."
        ),
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
            "For homophonic ciphers this is the safest repair primitive: change "
            "one cipher symbol at a time and inspect score_delta."
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
            "If one letter has no mapping yet, it is simply remapped to the target. "
            "Avoid this for homophonic repairs unless you really want to move every "
            "symbol currently mapped to both decoded letters; use act_set_mapping "
            "for targeted homophonic fixes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "letter_a": {"type": "string", "description": "Decoded letter to swap (as it appears in the decoded text)."},
                "letter_b": {"type": "string", "description": "The letter it should become."},
                "auto_revert_if_worse": {"type": "boolean", "default": True},
            },
            "required": ["branch", "letter_a", "letter_b"],
        },
    },
    # ----- search_* -----
    {
        "name": "search_hill_climb",
        "description": (
            "Greedy per-symbol refinement of a branch's key. ONLY use AFTER "
            "search_anneal has produced a good starting point — hill-climbing "
            "from random/empty starts frequently stalls in wrong local optima "
            "(observed: ~40% accuracy on English). Use when search_anneal's "
            "output is readable but has a few residual errors; hill_climb will "
            "polish the last few symbols. If unsure, use search_anneal. "
            "Works for both bijective and homophonic ciphers."
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
            "Simulated annealing — THE PRIMARY SEARCH TOOL. Use first on any "
            "branch that needs a starting solution (empty, partially-mapped, "
            "or stuck). Mixes single-symbol reassignment (70%) and 2-symbol "
            "swaps (30%). By default, partial manual anchors are preserved, "
            "but a fully mapped inherited branch is treated as a fresh restart "
            "so a fork can escape a bad complete key. Set preserve_existing=true "
            "only when you deliberately want to polish the current full key. "
            "score_fn defaults to 'combined' (dictionary + quadgram + "
            "language chi²). Typically achieves 85%+ on English/Latin in one "
            "call. After annealing, read the decoded text; if a few errors "
            "remain, EITHER declare (if readable) OR call "
            "decode_diagnose_and_fix(branch) to fix all residual errors in one "
            "iteration. Only run search_hill_climb as a final polish."
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
                "preserve_existing": {
                    "type": "boolean",
                    "description": (
                        "Whether to preserve current mappings as fixed anchors. "
                        "If omitted, partial keys are preserved but complete "
                        "inherited keys are restarted from scratch."
                    ),
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "search_homophonic_anneal",
        "description": (
            "Purpose-built automated solver for homophonic no-boundary ciphers. "
            "This is the strongest first move when the cipher has more symbols "
            "than plaintext letters or observe_homophone_distribution says it "
            "is likely homophonic. It independently maps every cipher symbol "
            "to a plaintext letter, uses continuous 5-gram scoring plus a "
            "global letter-distribution objective, and returns a near-complete "
            "branch for reading/refinement. Prefer this over generic "
            "search_anneal for hardest/no-boundary homophonic tests. It can "
            "run directly on `main`; do not spend a turn forking unless you "
            "need to preserve an existing useful key."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "epochs": {
                    "type": "integer",
                    "default": 5,
                    "description": "Independent annealing epochs/restarts.",
                },
                "sampler_iterations": {
                    "type": "integer",
                    "default": 2000,
                    "description": "Sampler iterations per epoch; each visits all cipher symbols.",
                },
                "t_start": {"type": "number", "default": 0.012},
                "t_end": {"type": "number", "default": 0.006},
                "order": {
                    "type": "integer",
                    "enum": [3, 4, 5],
                    "default": 5,
                    "description": "Continuous-letter n-gram order.",
                },
                "preserve_existing": {
                    "type": "boolean",
                    "default": False,
                    "description": "Treat existing mappings as fixed anchors.",
                },
                "model_path": {
                    "type": "string",
                    "description": (
                        "Optional Zenith-style continuous n-gram CSV model. "
                        "If omitted, the tool auto-discovers "
                        "other_tools/zenith-2026.2/zenith-model.csv when "
                        "present; use 'word_list' to force the small fallback."
                    ),
                },
                "max_ngrams": {
                    "type": "integer",
                    "default": 3000000,
                    "description": "Maximum corpus n-grams to load from model_path.",
                },
                "distribution_weight": {
                    "type": "number",
                    "default": 4.0,
                    "description": (
                        "Weight for global plaintext letter-distribution penalty; "
                        "prevents collapsed repeated-letter solutions."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional deterministic random seed.",
                },
                "top_n": {
                    "type": "integer",
                    "default": 1,
                    "description": "Return the top N distinct epoch candidates.",
                },
                "write_candidate_branches": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, write non-best candidates to sibling branches "
                        "named <branch>_cand2, <branch>_cand3, ..."
                    ),
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "decode_letter_stats",
        "description": (
            "Show the letter-frequency distribution of a branch's decoded text "
            "and compare it to the target language's reference distribution. "
            "Highlights absent letters (0 occurrences) and letters with "
            "frequency ratios far from expected — these are the most likely "
            "candidates for wrong mappings. Much faster than re-deriving this "
            "via run_python."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    {
        "name": "decode_ambiguous_letter",
        "description": (
            "When a decoded letter appears to stand for multiple true letters "
            "in different contexts, show which cipher symbols currently produce "
            "that decoded letter and sample contexts for each symbol. Use this "
            "before changing an overused decoded letter such as I, E, or L. "
            "The output lets you make targeted act_set_mapping calls on specific "
            "cipher symbols instead of broad act_swap_decoded calls that can "
            "damage correct occurrences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "decoded_letter": {
                    "type": "string",
                    "description": "The plaintext letter currently shown in the decode, e.g. 'I'.",
                },
                "context": {
                    "type": "integer",
                    "default": 6,
                    "description": "Characters to include on each side.",
                },
                "max_contexts_per_symbol": {
                    "type": "integer",
                    "default": 12,
                    "description": "Maximum context examples for each cipher symbol.",
                },
            },
            "required": ["branch", "decoded_letter"],
        },
    },
    {
        "name": "decode_absent_letter_candidates",
        "description": (
            "When a plaintext letter is absent or badly underrepresented in a "
            "homophonic/no-boundary decode, rank cipher symbols currently mapped "
            "to overrepresented decoded letters as candidates for that missing "
            "letter. For each candidate it returns contexts and the score delta "
            "from temporarily remapping only that cipher symbol. This replaces "
            "ad hoc run_python searches for patterns such as missing U, Y, P, "
            "or V in homophonic ciphers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "missing_letter": {
                    "type": "string",
                    "description": "Plaintext letter to investigate, e.g. U.",
                },
                "source_letters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional decoded letters to inspect as sources. If "
                        "omitted, the tool uses overrepresented letters from "
                        "decode_letter_stats."
                    ),
                },
                "context": {"type": "integer", "default": 6},
                "max_candidates": {"type": "integer", "default": 12},
                "max_contexts_per_symbol": {"type": "integer", "default": 6},
            },
            "required": ["branch", "missing_letter"],
        },
    },
    {
        "name": "decode_diagnose",
        "description": (
            "Analyse a branch's decoded text and rank likely residual single-"
            "letter errors. Runs DP word segmentation, finds pseudo-words, and "
            "for each pseudo-word searches the dictionary for same-length words "
            "at edit distance 1 (single substitution). Groups suggestions by "
            "(wrong, correct) letter pair and ranks by evidence count. "
            "Each candidate includes cipher_symbols_for_wrong (the cipher "
            "symbols currently producing the wrong letter), culprit_symbol "
            "(the specific symbol most likely causing the errors), and "
            "suggested_call (the exact tool call to make the fix). "
            "For homophonic ciphers, suggested_call always uses "
            "act_set_mapping(cipher_symbol=X, plain_letter=Y) on the culprit "
            "symbol only — safer than act_swap_decoded which would move ALL "
            "symbols for those decoded letters and can break correctly-decoded "
            "homophones. "
            "Also returns bulk_fix_call: a single decode_diagnose_and_fix call "
            "that score-checks top candidates in one iteration and skips "
            "worsening repairs by default. "
            "Call this AFTER search_anneal has converged but a few errors remain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "top_k": {"type": "integer", "default": 5,
                           "description": "Return the top K most-supported corrections."},
            },
            "required": ["branch"],
        },
    },
    {
        "name": "decode_diagnose_and_fix",
        "description": (
            "Diagnose residual errors AND apply all high-confidence fixes in a "
            "single call — collapsing many fix iterations into one. "
            "Runs the same analysis as decode_diagnose (DP segmentation + "
            "edit-distance-1 corrections + culprit-symbol identification), then "
            "tests every candidate whose evidence_count >= "
            "min_evidence (default 2) using act_set_mapping on the specific "
            "culprit cipher symbol, reverting candidates that make the branch "
            "worse by default. Returns a combined before/after score_delta, "
            "dict_rate_after, pseudo_words_remaining, and a recommendation "
            "(declare now vs. run again). "
            "Use this immediately after search_anneal when the decoded text is "
            "mostly readable but has a few systematic errors — it replaces "
            "the decode_diagnose → many-act_set_mapping loop with one tool call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max number of fixes to consider.",
                },
                "min_evidence": {
                    "type": "integer",
                    "default": 2,
                    "description": (
                        "Minimum evidence_count to auto-apply a fix. "
                        "Raise to be conservative; lower to 1 to apply all."
                    ),
                },
                "auto_revert_if_worse": {"type": "boolean", "default": True},
            },
            "required": ["branch"],
        },
    },
    # ----- run_python -----
    {
        "name": "run_python",
        "description": (
            "Execute Python 3 code and return stdout/stderr. "
            "This is allowed, but every use is treated as evidence of a possible "
            "tool-design gap. Use it for calculations the built-in tools don't "
            "cover: custom scoring formulas, statistical analysis, cipher-specific "
            "algorithms, or bulk data manipulation. stdlib only — no external "
            "packages. Timeout: 15 s. Print results to stdout. "
            "You MUST provide a justification explaining the question you need "
            "answered, why the first-class tools are insufficient, and what "
            "dedicated tool would have made this Python call unnecessary. "
            "If you find yourself needing this frequently for the same kind of "
            "computation, also call meta_request_tool to document it as a gap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "justification": {
                    "type": "string",
                    "description": (
                        "Why Python is needed here: the question being answered, "
                        "why existing tools are insufficient, and the dedicated "
                        "tool that would avoid this call."
                    ),
                },
                "code": {
                    "type": "string",
                    "description": "Python 3 code to execute. Print outputs to stdout.",
                }
            },
            "required": ["justification", "code"],
        },
    },
    # ----- meta_* -----
    {
        "name": "meta_request_tool",
        "description": (
            "Document a capability gap: call this when you need a calculation or "
            "lookup that none of the available tools cover well. Describe what you "
            "need precisely so it can be added as a permanent tool. "
            "Does NOT execute code — use run_python as a workaround first. "
            "These requests appear prominently in the benchmark report to guide "
            "future tool development."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What you need to compute or look up.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why existing tools don't cover this need.",
                },
                "example_input": {
                    "type": "string",
                    "description": "Optional: example inputs for such a tool.",
                },
                "example_output": {
                    "type": "string",
                    "description": "Optional: example output you would expect.",
                },
            },
            "required": ["description", "rationale"],
        },
    },
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

        # Tool capability requests (meta_request_tool calls)
        self.tool_requests: list[dict] = []

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

    def _is_homophonic_cipher(self) -> bool:
        return self._alpha().size > self._pt_alpha().size

    def _is_no_boundary_cipher(self) -> bool:
        return len(self.workspace.cipher_text.words) <= 1

    def _reference_letter_freq(self) -> dict[str, float]:
        """Return plaintext-letter reference frequencies as fractions."""
        bi_lp = ngram.NGRAM_CACHE.get(self.language, 2)
        ref_freq: dict[str, float] = {}
        if bi_lp:
            raw: dict[str, float] = {}
            for bg, lp in bi_lp.items():
                if bg == "_floor" or len(bg) != 2:
                    continue
                raw[bg[0]] = raw.get(bg[0], 0.0) + 10 ** lp
            s = sum(raw.values())
            if s > 0:
                ref_freq = {k: v / s for k, v in raw.items()}
        if ref_freq:
            return ref_freq

        ref_raw = {
            "E": 12.7, "T": 9.1, "A": 8.2, "O": 7.5, "I": 7.0, "N": 6.7,
            "S": 6.3, "H": 6.1, "R": 6.0, "D": 4.3, "L": 4.0, "C": 2.8,
            "U": 2.8, "M": 2.4, "W": 2.4, "F": 2.2, "G": 2.0, "Y": 2.0,
            "P": 1.9, "B": 1.5, "V": 1.0, "K": 0.8, "J": 0.15, "X": 0.15,
            "Q": 0.10, "Z": 0.07,
        }
        s = sum(ref_raw.values())
        return {k: v / s for k, v in ref_raw.items()}

    def _expected_homophone_counts(self) -> dict[str, int]:
        """Largest-remainder allocation of cipher symbols to plaintext letters."""
        ref = self._reference_letter_freq()
        letters = [self._pt_alpha().symbol_for(i) for i in range(self._pt_alpha().size)]
        raw = {l: ref.get(l, 0.0) * self._alpha().size for l in letters}
        counts = {l: max(1, int(raw[l])) for l in letters}
        diff = self._alpha().size - sum(counts.values())
        if diff > 0:
            order = sorted(letters, key=lambda l: (raw[l] - int(raw[l])), reverse=True)
            for l in order[:diff]:
                counts[l] += 1
        elif diff < 0:
            order = sorted(letters, key=lambda l: (raw[l] - int(raw[l]), raw[l]))
            for l in order:
                if diff == 0:
                    break
                if counts[l] > 1:
                    counts[l] -= 1
                    diff += 1
        return counts

    def _decoded_letter_rows(self, branch: str) -> list[dict[str, Any]]:
        from collections import Counter

        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        letters_only = [c for c in normalized if c.isalpha()]
        total = len(letters_only)
        observed = Counter(letters_only)
        ref_freq = self._reference_letter_freq()
        rows = []
        for i in range(self._pt_alpha().size):
            letter = self._pt_alpha().symbol_for(i)
            obs_count = observed.get(letter, 0)
            obs_pct = (obs_count / total * 100) if total else 0.0
            ref_pct = ref_freq.get(letter, 0.0) * 100
            ratio = (obs_pct / ref_pct) if ref_pct > 0 else float("inf")
            rows.append({
                "letter": letter,
                "count": obs_count,
                "obs_pct": round(obs_pct, 2),
                "ref_pct": round(ref_pct, 2),
                "ratio": round(ratio, 2) if ratio != float("inf") else "∞",
            })
        return rows

    def _homophonic_reliability_warnings(
        self,
        branch: str,
        dict_rate: float | None = None,
        quad: float | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        if not self._is_homophonic_cipher():
            return warnings
        rows = self._decoded_letter_rows(branch)
        absent_common = [
            r["letter"] for r in rows
            if r["count"] == 0 and isinstance(r["ref_pct"], float) and r["ref_pct"] >= 1.0
        ]
        over = [
            r["letter"] for r in rows
            if isinstance(r["ratio"], float) and r["ratio"] >= 1.8 and r["ref_pct"] >= 1.0
        ]
        if absent_common:
            warnings.append(
                "Homophonic warning: common plaintext letters are absent "
                f"({', '.join(absent_common[:8])}). A high segmented "
                "dictionary_rate may be a false positive."
            )
        if over:
            warnings.append(
                "Homophonic warning: decoded letters are overrepresented "
                f"({', '.join(over[:8])}); inspect them with "
                "decode_absent_letter_candidates before declaring."
            )
        if self._is_no_boundary_cipher() and dict_rate is not None and quad is not None:
            if dict_rate >= 0.65 and quad < -4.0:
                warnings.append(
                    "No-boundary homophonic warning: dictionary_rate is high "
                    "but quadgram score is poor. The segmenter can carve wrong "
                    "text into short dictionary words; do not declare from "
                    "dictionary_rate alone."
                )
        return warnings

    def _compute_quick_scores(self, branch_name: str) -> dict[str, float | None]:
        """Return (dict_rate, quad) for a branch, fast. Used for score_delta
        on mutation tools so the agent can immediately see if a change helped.
        """
        from analysis.segment import segment_text
        decrypted = self.workspace.apply_key(branch_name)
        normalized = sig.normalize_for_scoring(decrypted)
        if not normalized.strip():
            return {"dict_rate": None, "quad": None}
        if not any(c.isspace() for c in normalized.strip()):
            seg = segment_text(normalized, self.word_set, self._freq_rank)
            dict_rate = seg.dict_rate
        else:
            words = [w for w in normalized.split() if any(c.isalpha() for c in w)]
            dict_rate = (
                sum(1 for w in words if w in self.word_set) / len(words)
                if words else 0.0
            )
        quad_lp = ngram.NGRAM_CACHE.get(self.language, 4)
        quad = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
        return {
            "dict_rate": round(dict_rate, 4),
            "quad": round(quad, 4) if quad != float("-inf") else None,
        }

    def _score_delta(
        self, before: dict[str, float | None], after: dict[str, float | None]
    ) -> dict[str, Any]:
        """Build a score-delta block from pre/post snapshots."""
        dr_b, dr_a = before["dict_rate"], after["dict_rate"]
        q_b, q_a = before["quad"], after["quad"]
        dict_delta = (
            round(dr_a - dr_b, 4)
            if dr_b is not None and dr_a is not None else None
        )
        quad_delta = (
            round(q_a - q_b, 4)
            if q_b is not None and q_a is not None else None
        )
        dict_improved = dict_delta is not None and dict_delta > 0
        dict_worsened = dict_delta is not None and dict_delta < 0
        quad_improved = quad_delta is not None and quad_delta > 0
        quad_worsened = quad_delta is not None and quad_delta < 0

        if dict_improved and not quad_worsened:
            verdict = "improved"
        elif quad_improved and not dict_worsened:
            verdict = "improved"
        elif dict_worsened and not quad_improved:
            verdict = "worse"
        elif quad_worsened and not dict_improved:
            verdict = "worse"
        elif dict_worsened and quad_improved:
            verdict = "mixed"
        elif dict_improved and quad_worsened:
            verdict = "mixed"
        else:
            verdict = "unchanged"
        return {
            "dict_rate_before": dr_b,
            "dict_rate_after": dr_a,
            "dict_rate_delta": dict_delta,
            "quad_before": q_b,
            "quad_after": q_a,
            "quad_delta": quad_delta,
            "verdict": verdict,
            "improved": verdict == "improved",
        }

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

    def _tool_observe_homophone_distribution(self, args: dict) -> Any:
        expected = self._expected_homophone_counts()
        branch_name = args.get("branch")
        actual_rows: list[dict[str, Any]] = []
        warnings: list[str] = []

        if branch_name:
            branch = self.workspace.get_branch(branch_name)
            pt_alpha = self._pt_alpha()
            alpha = self._alpha()
            by_plain: dict[str, list[str]] = {
                pt_alpha.symbol_for(i): [] for i in range(pt_alpha.size)
            }
            for cid, pid in sorted(branch.key.items()):
                by_plain.setdefault(pt_alpha.symbol_for(pid), []).append(
                    alpha.symbol_for(cid)
                )
            for letter, symbols in by_plain.items():
                exp = expected.get(letter, 0)
                actual_rows.append({
                    "letter": letter,
                    "expected_symbols": exp,
                    "actual_symbols": len(symbols),
                    "delta": len(symbols) - exp,
                    "cipher_symbols": symbols[:20],
                })
            actual_rows.sort(key=lambda r: (-abs(r["delta"]), r["letter"]))
            warnings = self._homophonic_reliability_warnings(branch_name)

        return {
            "cipher_alphabet_size": self._alpha().size,
            "plaintext_alphabet_size": self._pt_alpha().size,
            "is_likely_homophonic": self._is_homophonic_cipher(),
            "expected_symbol_counts": [
                {"letter": k, "expected_symbols": v}
                for k, v in sorted(expected.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "branch": branch_name,
            "actual_vs_expected": actual_rows if branch_name else None,
            "warnings": warnings,
            "note": (
                "For homophonic ciphers, solved keys should usually map several "
                "cipher symbols to common letters and at least one symbol to most "
                "common plaintext letters. Large absences or overloaded letters "
                "are evidence of a structurally wrong branch."
            ),
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

    def _tool_decode_letter_stats(self, args: dict) -> Any:
        branch = args["branch"]
        rows = self._decoded_letter_rows(branch)
        total = sum(int(r["count"]) for r in rows)
        if total == 0:
            return {"branch": branch, "error": "no mapped letters in decoded text"}

        absent = [r["letter"] for r in rows if r["count"] == 0]
        overrepresented = [
            r for r in rows
            if isinstance(r["ratio"], float) and r["ratio"] > 3.0 and r["ref_pct"] > 0.5
        ]
        underrepresented = [
            r for r in rows
            if isinstance(r["ratio"], float) and r["ratio"] < 0.3
            and r["ref_pct"] > 1.0 and r["count"] == 0
        ]
        return {
            "branch": branch,
            "total_letters": total,
            "absent_letters": absent,
            "suspicious_overrepresented": [
                f"{r['letter']}({r['obs_pct']:.1f}% vs ref {r['ref_pct']:.1f}%)"
                for r in overrepresented[:5]
            ],
            "suspicious_underrepresented": [
                f"{r['letter']}({r['obs_pct']:.1f}% vs ref {r['ref_pct']:.1f}%)"
                for r in underrepresented[:5]
            ],
            "full_table": rows,
            "warnings": self._homophonic_reliability_warnings(branch),
            "note": (
                "absent_letters = letters with 0 occurrences in decoded text — "
                "likely wrong mappings. overrepresented = letters appearing much "
                "more than expected (may be absorbing counts for an absent letter). "
                "For homophonic ciphers, prefer decode_absent_letter_candidates "
                "over act_swap_decoded so you can test one cipher symbol at a time."
            ),
        }

    def _tool_decode_ambiguous_letter(self, args: dict) -> Any:
        branch = args["branch"]
        decoded_letter = args["decoded_letter"].upper()
        context = max(1, int(args.get("context", 6)))
        max_contexts = max(1, int(args.get("max_contexts_per_symbol", 12)))
        pt_alpha = self._pt_alpha()
        alpha = self._alpha()
        if not pt_alpha.has_symbol(decoded_letter):
            return {"error": f"Unknown plaintext letter: {decoded_letter}"}

        b = self.workspace.get_branch(branch)
        target_id = pt_alpha.id_for(decoded_letter)
        cipher_ids = [
            cid for cid, pid in sorted(b.key.items())
            if pid == target_id
        ]
        if not cipher_ids:
            return {
                "branch": branch,
                "decoded_letter": decoded_letter,
                "cipher_symbols": [],
                "note": "No cipher symbols currently produce this decoded letter.",
            }

        ct = self.workspace.cipher_text
        flat_decoded = []
        for tok_id in ct.tokens:
            if tok_id in b.key:
                sym = pt_alpha.symbol_for(b.key[tok_id])
                flat_decoded.append(sym if len(sym) == 1 else "?")
            else:
                flat_decoded.append("?")
        decoded = "".join(flat_decoded)
        cipher_symbols = [alpha.symbol_for(cid) for cid in cipher_ids]

        groups = []
        for cid in cipher_ids:
            examples = []
            for pos, tok_id in enumerate(ct.tokens):
                if tok_id != cid:
                    continue
                lo = max(0, pos - context)
                hi = min(len(ct.tokens), pos + context + 1)
                cipher_ctx = "".join(alpha.symbol_for(t) for t in ct.tokens[lo:hi])
                decoded_ctx = decoded[lo:hi]
                examples.append({
                    "char_index": pos,
                    "cipher_context": cipher_ctx,
                    "decoded_context": decoded_ctx,
                })
                if len(examples) >= max_contexts:
                    break
            groups.append({
                "cipher_symbol": alpha.symbol_for(cid),
                "occurrences": sum(1 for t in ct.tokens if t == cid),
                "examples": examples,
                "suggested_next_step": (
                    f"If these contexts point to one true letter, call "
                    f"act_set_mapping(branch='{branch}', "
                    f"cipher_symbol='{alpha.symbol_for(cid)}', "
                    f"plain_letter='...')"
                ),
            })

        return {
            "branch": branch,
            "decoded_letter": decoded_letter,
            "cipher_symbols": cipher_symbols,
            "symbol_count": len(cipher_symbols),
            "groups": groups,
            "note": (
                "Do not use act_swap_decoded when symbol_count > 1 unless all "
                "listed cipher symbols should move together. Usually inspect "
                "the contexts and use targeted act_set_mapping calls."
            ),
        }

    def _tool_decode_absent_letter_candidates(self, args: dict) -> Any:
        branch_name = args["branch"]
        missing_letter = args["missing_letter"].upper()
        context = max(1, int(args.get("context", 6)))
        max_candidates = max(1, int(args.get("max_candidates", 12)))
        max_contexts = max(1, int(args.get("max_contexts_per_symbol", 6)))
        pt_alpha = self._pt_alpha()
        alpha = self._alpha()
        if not pt_alpha.has_symbol(missing_letter):
            return {"error": f"Unknown plaintext letter: {missing_letter}"}

        rows = self._decoded_letter_rows(branch_name)
        source_letters = [s.upper() for s in args.get("source_letters", [])]
        if not source_letters:
            source_letters = [
                r["letter"] for r in rows
                if isinstance(r["ratio"], float) and r["ratio"] >= 1.5 and r["count"] > 0
            ]
        if not source_letters:
            source_letters = [
                r["letter"] for r in rows
                if r["letter"] != missing_letter and r["count"] > 0
            ]

        b = self.workspace.get_branch(branch_name)
        missing_id = pt_alpha.id_for(missing_letter)
        source_ids = {
            pt_alpha.id_for(letter)
            for letter in source_letters
            if pt_alpha.has_symbol(letter)
        }
        candidate_ids = [
            cid for cid, pid in sorted(b.key.items())
            if pid in source_ids and pid != missing_id
        ]

        ct = self.workspace.cipher_text
        decoded_chars = []
        for tok_id in ct.tokens:
            if tok_id in b.key:
                sym = pt_alpha.symbol_for(b.key[tok_id])
                decoded_chars.append(sym if len(sym) == 1 else "?")
            else:
                decoded_chars.append("?")
        decoded = "".join(decoded_chars)

        before = self._compute_quick_scores(branch_name)
        candidates = []
        for cid in candidate_ids:
            old_pid = b.key[cid]
            old_letter = pt_alpha.symbol_for(old_pid)
            examples = []
            for pos, tok_id in enumerate(ct.tokens):
                if tok_id != cid:
                    continue
                lo = max(0, pos - context)
                hi = min(len(ct.tokens), pos + context + 1)
                examples.append({
                    "char_index": pos,
                    "cipher_context": "".join(alpha.symbol_for(t) for t in ct.tokens[lo:hi]),
                    "decoded_context": decoded[lo:hi],
                    "with_candidate": decoded[lo:pos] + missing_letter + decoded[pos + 1:hi],
                })
                if len(examples) >= max_contexts:
                    break

            b.key[cid] = missing_id
            after = self._compute_quick_scores(branch_name)
            b.key[cid] = old_pid
            delta = self._score_delta(before, after)
            candidates.append({
                "cipher_symbol": alpha.symbol_for(cid),
                "current_letter": old_letter,
                "candidate_letter": missing_letter,
                "occurrences": sum(1 for t in ct.tokens if t == cid),
                "score_delta_if_remapped": delta,
                "examples": examples,
                "suggested_call": (
                    f"act_set_mapping(branch='{branch_name}', "
                    f"cipher_symbol='{alpha.symbol_for(cid)}', "
                    f"plain_letter='{missing_letter}')"
                ),
            })

        def rank(c: dict[str, Any]) -> tuple[float, float, int]:
            delta = c["score_delta_if_remapped"]
            verdict_bonus = {"improved": 1.0, "mixed": 0.2, "unchanged": 0.0, "worse": -1.0}
            return (
                verdict_bonus.get(delta["verdict"], 0.0),
                delta["quad_delta"] if delta["quad_delta"] is not None else -999.0,
                c["occurrences"],
            )

        candidates.sort(key=rank, reverse=True)
        return {
            "branch": branch_name,
            "missing_letter": missing_letter,
            "source_letters_considered": source_letters,
            "before_scores": before,
            "candidates": candidates[:max_candidates],
            "warnings": self._homophonic_reliability_warnings(
                branch_name, before.get("dict_rate"), before.get("quad")
            ),
            "note": (
                "Use the score_delta_if_remapped and contexts together. In "
                "homophonic ciphers, remap only one cipher symbol at a time; "
                "do not swap every occurrence of an overrepresented letter."
            ),
        }

    def _diagnose_branch(self, branch: str, top_k: int) -> dict:
        """Shared diagnosis logic: segment decoded text, find edit-distance-1
        corrections, identify culprit cipher symbols. Returns a dict with keys:
        seg, candidates, any_ambiguous, flat_str.
        """
        from collections import Counter
        from analysis.segment import find_one_edit_corrections, segment_text

        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        if not normalized.strip():
            return {"error": "empty decoded text"}

        seg = segment_text(normalized, self.word_set, self._freq_rank)
        pseudo_words = [w for w in seg.pseudo_words if len(w) >= 3]

        evidence: dict[tuple[str, str], list[str]] = {}
        for pw in pseudo_words:
            for cand, wrong, correct in find_one_edit_corrections(pw, self.word_set):
                ev_key = (wrong, correct)
                evidence.setdefault(ev_key, []).append(f"{pw}→{cand}")

        ranked = sorted(evidence.items(), key=lambda kv: -len(kv[1]))[:top_k]

        # Build flat decoded string (one char per token) for culprit detection.
        ct = self.workspace.cipher_text
        b = self.workspace.get_branch(branch)
        pt_alpha = self._pt_alpha()
        alpha = self._alpha()
        flat_decoded = []
        for tok_id in ct.tokens:
            if tok_id in b.key:
                sym = pt_alpha.symbol_for(b.key[tok_id])
                flat_decoded.append(sym if len(sym) == 1 else "?")
            else:
                flat_decoded.append("?")
        flat_str = "".join(flat_decoded)

        candidates = []
        any_ambiguous = False
        homophonic = self._is_homophonic_cipher()
        for (wrong, correct), examples in ranked:
            wrong_id = pt_alpha.id_for(wrong)
            symbols_for_wrong = [
                alpha.symbol_for(cid)
                for cid, pid in b.key.items()
                if pid == wrong_id
            ]

            ambiguous = len(symbols_for_wrong) > 1
            if ambiguous:
                any_ambiguous = True

            sym_counts: Counter = Counter()
            for ex in examples:
                pw = ex.split("→")[0]
                start = 0
                while True:
                    pos = flat_str.find(pw, start)
                    if pos < 0:
                        break
                    for i, ch in enumerate(pw):
                        if ch == wrong:
                            ti = pos + i
                            if ti < len(ct.tokens):
                                sym_counts[alpha.symbol_for(ct.tokens[ti])] += 1
                    start = pos + 1

            culprit = (
                sym_counts.most_common(1)[0][0]
                if sym_counts
                else (symbols_for_wrong[0] if symbols_for_wrong else None)
            )

            if homophonic:
                suggested = (
                    f"act_set_mapping(branch='{branch}', "
                    f"cipher_symbol='{culprit}', plain_letter='{correct}')"
                )
            else:
                suggested = (
                    f"act_swap_decoded(branch='{branch}', "
                    f"letter_a='{wrong}', letter_b='{correct}')"
                )

            candidates.append({
                "wrong": wrong,
                "correct": correct,
                "evidence_count": len(examples),
                "examples": examples[:5],
                "cipher_symbols_for_wrong": symbols_for_wrong,
                "ambiguous": ambiguous,
                "culprit_symbol": culprit,
                "suggested_call": suggested,
            })

        return {
            "seg": seg,
            "candidates": candidates,
            "any_ambiguous": any_ambiguous,
        }

    def _tool_decode_diagnose(self, args: dict) -> Any:
        branch = args["branch"]
        top_k = int(args.get("top_k", 5))
        diag = self._diagnose_branch(branch, top_k)
        if "error" in diag:
            return {"branch": branch, "error": diag["error"]}

        seg = diag["seg"]
        candidates = diag["candidates"]
        any_ambiguous = diag["any_ambiguous"]

        seg_preview = seg.segmented[:600] + ("…" if len(seg.segmented) > 600 else "")

        if candidates:
            note = "Higher evidence_count = more pseudo-words fixed by one change."
            if self._is_homophonic_cipher():
                note += (
                    " Homophonic repair mode: suggested_call uses act_set_mapping "
                    "on the culprit_symbol even when only one cipher symbol is "
                    "currently mapped to the wrong decoded letter. Global decoded "
                    "letter swaps can move correctly decoded homophones in the "
                    "opposite direction."
                )
            note += (
                " decode_diagnose_and_fix score-checks each candidate and skips "
                "changes that make the branch worse by default."
            )
            # Bulk fix suggestion — apply all candidates in one tool call.
            bulk_fix_call = (
                f"decode_diagnose_and_fix(branch='{branch}', "
                f"top_k={len(candidates)})"
            )
        else:
            if self._is_homophonic_cipher() and self._is_no_boundary_cipher():
                note = (
                    "No single-substitution corrections found. In a no-boundary "
                    "homophonic cipher this is NOT evidence that the branch is "
                    "solved; the branch may be structurally wrong and the "
                    "segmenter may be finding short dictionary words by chance. "
                    "Check score_panel warnings, observe_homophone_distribution, "
                    "and decode_absent_letter_candidates before declaring."
                )
            else:
                note = (
                    "No single-substitution corrections found. The branch may be "
                    "nearly solved (consider declaring) or structurally off "
                    "(consider forking + search_anneal)."
                )
            bulk_fix_call = None

        return {
            "branch": branch,
            "total_words_segmented": len(seg.words),
            "total_pseudo_words": len(seg.pseudo_words),
            "dict_rate": round(seg.dict_rate, 4),
            "segmented_text": seg_preview,
            "candidate_corrections": candidates,
            "bulk_fix_call": bulk_fix_call,
            "warnings": self._homophonic_reliability_warnings(
                branch, round(seg.dict_rate, 4), self._compute_quick_scores(branch).get("quad")
            ),
            "note": note,
        }

    def _tool_decode_diagnose_and_fix(self, args: dict) -> Any:
        """Diagnose + apply top fixes atomically, returning a combined score delta."""
        branch = args["branch"]
        top_k = int(args.get("top_k", 5))
        min_evidence = int(args.get("min_evidence", 2))
        auto_revert_if_worse = bool(args.get("auto_revert_if_worse", True))

        diag = self._diagnose_branch(branch, top_k)
        if "error" in diag:
            return {"branch": branch, "error": diag["error"]}

        candidates = diag["candidates"]
        seg = diag["seg"]

        to_apply = [c for c in candidates if c["evidence_count"] >= min_evidence and c["culprit_symbol"]]
        skipped = [c for c in candidates if c["evidence_count"] < min_evidence or not c["culprit_symbol"]]

        if not to_apply:
            return {
                "branch": branch,
                "fixes_applied": [],
                "fixes_skipped": [
                    {"wrong": c["wrong"], "correct": c["correct"],
                     "evidence_count": c["evidence_count"]}
                    for c in skipped
                ],
                "score_delta": None,
                "note": (
                    f"No candidates met min_evidence={min_evidence}. "
                    "Lower min_evidence or inspect with decode_diagnose."
                ),
            }

        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        before = self._compute_quick_scores(branch)

        fixes_applied = []
        fixes_reverted = []
        for c in to_apply:
            sym = c["culprit_symbol"]
            letter = c["correct"]
            if not alpha.has_symbol(sym) or not pt_alpha.has_symbol(letter):
                continue
            cid = alpha.id_for(sym)
            old_pid = self.workspace.get_branch(branch).key.get(cid)
            before_candidate = self._compute_quick_scores(branch)
            self.workspace.set_mapping(
                branch,
                cid,
                pt_alpha.id_for(letter),
            )
            after_candidate = self._compute_quick_scores(branch)
            candidate_delta = self._score_delta(before_candidate, after_candidate)
            if auto_revert_if_worse and candidate_delta["verdict"] == "worse":
                if old_pid is None:
                    self.workspace.clear_mapping(branch, cid)
                else:
                    self.workspace.set_mapping(branch, cid, old_pid)
                fixes_reverted.append({
                    "cipher_symbol": sym,
                    "was": c["wrong"],
                    "attempted": letter,
                    "evidence_count": c["evidence_count"],
                    "examples": c["examples"][:3],
                    "score_delta": candidate_delta,
                    "reason": "auto_revert_if_worse",
                })
                continue
            fixes_applied.append({
                "cipher_symbol": sym,
                "was": c["wrong"],
                "now": letter,
                "evidence_count": c["evidence_count"],
                "examples": c["examples"][:3],
                "score_delta": candidate_delta,
            })

        after = self._compute_quick_scores(branch)

        # Re-diagnose to show what's still left.
        remaining_diag = self._diagnose_branch(branch, top_k)
        remaining_seg = remaining_diag.get("seg")
        remaining_candidates = remaining_diag.get("candidates", [])

        remaining_pseudo = len(remaining_seg.pseudo_words) if remaining_seg else None
        dict_rate_after = round(remaining_seg.dict_rate, 4) if remaining_seg else None

        if dict_rate_after is not None and dict_rate_after >= 0.85:
            if self._is_homophonic_cipher() and self._is_no_boundary_cipher():
                recommendation = (
                    f"dict_rate is now {dict_rate_after:.0%}, but this is a "
                    "no-boundary homophonic cipher. High dictionary_rate alone "
                    "is not enough to declare; check quadgram/bigram signals, "
                    "homophone distribution, absent letters, and read the full "
                    "decode for coherent prose."
                )
            else:
                recommendation = (
                    f"dict_rate is now {dict_rate_after:.0%}. "
                    "The decoded text looks solved — call meta_declare_solution now."
                )
        elif remaining_candidates:
            recommendation = (
                f"{remaining_pseudo} pseudo-words remain. "
                "Call decode_diagnose_and_fix again or declare if the text reads well."
            )
        else:
            if self._is_homophonic_cipher() and self._is_no_boundary_cipher():
                recommendation = (
                    "No further single-substitution fixes found. For a no-boundary "
                    "homophonic cipher this may mean the branch is structurally "
                    "wrong, not solved. Fork/re-anneal or inspect "
                    "observe_homophone_distribution and "
                    "decode_absent_letter_candidates before declaring."
                )
            else:
                recommendation = (
                    "No further single-substitution fixes found. "
                    "Declare your solution."
                )

        return {
            "branch": branch,
            "fixes_applied": fixes_applied,
            "fixes_reverted": fixes_reverted,
            "fixes_skipped": [
                {"wrong": c["wrong"], "correct": c["correct"],
                 "evidence_count": c["evidence_count"]}
                for c in skipped
            ],
            "score_delta": self._score_delta(before, after),
            "dict_rate_after": dict_rate_after,
            "pseudo_words_remaining": remaining_pseudo,
            "remaining_candidates": remaining_candidates[:3],
            "decoded_preview": self._decoded_preview(branch),
            "warnings": self._homophonic_reliability_warnings(
                branch, dict_rate_after, after.get("quad")
            ),
            "recommendation": recommendation,
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
            freq_rank=self._freq_rank,
        )
        signals = panel.to_dict()
        warnings = self._homophonic_reliability_warnings(
            branch,
            signals.get("dictionary_rate"),
            signals.get("quadgram_loglik_per_gram"),
        )
        result: dict[str, Any] = {"branch": branch, "signals": signals}
        if warnings:
            result["warnings"] = warnings
            result["interpretation"] = (
                "Treat dictionary_rate as a weak signal on no-boundary "
                "homophonic ciphers. Do not declare unless the decoded text "
                "reads as coherent prose and n-gram/letter-distribution signals "
                "do not conflict."
            )
        return result

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

        # No-boundary text: segment with DP Viterbi before scoring.
        if normalized.strip() and not any(c.isspace() for c in normalized.strip()):
            from analysis.segment import segment_text
            seg = segment_text(normalized, self.word_set, self._freq_rank)
            preview = seg.segmented[:300] + ("…" if len(seg.segmented) > 300 else "")
            return {
                "branch": branch,
                "dictionary_rate": round(seg.dict_rate, 4),
                "total_words": len(seg.words),
                "recognized": len(seg.words) - len(seg.pseudo_words),
                "segmented_preview": preview,
                "pseudo_words_sample": seg.pseudo_words[:20],
                "warnings": self._homophonic_reliability_warnings(
                    branch,
                    round(seg.dict_rate, 4),
                    self._compute_quick_scores(branch).get("quad"),
                ),
                "note": (
                    "Text was segmented by DP Viterbi — pseudo_words are segments "
                    "that didn't match the dictionary. On homophonic no-boundary "
                    "ciphers, a high dictionary_rate can be a false positive; "
                    "cross-check with score_panel and letter-distribution warnings."
                ),
            }

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
        before = self._compute_quick_scores(branch)
        self.workspace.set_mapping(
            branch,
            alpha.id_for(cipher_sym),
            pt_alpha.id_for(plain_letter),
        )
        after = self._compute_quick_scores(branch)
        return {
            "status": "ok",
            "branch": branch,
            "mapping": f"{cipher_sym} -> {plain_letter}",
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._score_delta(before, after),
        }

    def _tool_act_bulk_set(self, args: dict) -> Any:
        branch = args["branch"]
        mappings = args.get("mappings", {})
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        set_ok: list[str] = []
        errors: list[str] = []
        before = self._compute_quick_scores(branch)
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
        after = self._compute_quick_scores(branch)
        return {
            "status": "ok",
            "branch": branch,
            "mappings_set": len(set_ok),
            "details": set_ok,
            "errors": errors if errors else None,
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._score_delta(before, after),
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
        before = self._compute_quick_scores(branch_name)
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
        after = self._compute_quick_scores(branch_name)
        return {
            "status": "ok",
            "branch": branch_name,
            "cipher_word": self._cipher_word_str(word_tokens),
            "now_decodes_as": decoded,
            "changes": changes,
            "decoded_preview": self._decoded_preview(branch_name),
            "score_delta": self._score_delta(before, after),
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
        auto_revert_if_worse = bool(args.get("auto_revert_if_worse", True))
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
        before = self._compute_quick_scores(branch)
        # Swap: everything that produced A now produces B, and vice versa
        for cid in cipher_for_a:
            self.workspace.set_mapping(branch, cid, id_b)
        for cid in cipher_for_b:
            self.workspace.set_mapping(branch, cid, id_a)
        after = self._compute_quick_scores(branch)
        delta = self._score_delta(before, after)
        alpha = self._alpha()
        swapped_a = [alpha.symbol_for(cid) for cid in cipher_for_a]
        swapped_b = [alpha.symbol_for(cid) for cid in cipher_for_b]
        trial_preview = self._decoded_preview(branch)
        if auto_revert_if_worse and delta["verdict"] == "worse":
            for cid in cipher_for_a:
                self.workspace.set_mapping(branch, cid, id_a)
            for cid in cipher_for_b:
                self.workspace.set_mapping(branch, cid, id_b)
            return {
                "status": "reverted",
                "branch": branch,
                "attempted_swap": f"{letter_a} ↔ {letter_b}",
                "cipher_symbols_for_a": swapped_a,
                "cipher_symbols_for_b": swapped_b,
                "trial_decoded_preview": trial_preview,
                "decoded_preview": self._decoded_preview(branch),
                "score_delta": delta,
                "note": (
                    "Swap made the branch score worse and was automatically "
                    "reverted. For homophonic ciphers, prefer act_set_mapping on "
                    "one culprit cipher symbol."
                ),
            }
        return {
            "status": "ok",
            "branch": branch,
            "swapped": f"{letter_a} ↔ {letter_b}",
            "cipher_symbols_for_a": swapped_a,
            "cipher_symbols_for_b": swapped_b,
            "decoded_preview": trial_preview,
            "score_delta": delta,
        }

    # ------------------------------------------------------------------
    # search_*
    # ------------------------------------------------------------------

    def _search_declare_note(self, search_kind: str = "anneal") -> str:
        """Return the post-search guidance note.

        search_kind ∈ {"anneal", "hill_climb"}. Post-anneal: recommend
        decode_diagnose_and_fix to fix all residual errors in one call.
        Post-hill-climb: the branch has converged — fork + restart with
        anneal if still stuck.
        """
        no_boundaries = len(self.workspace.cipher_text.words) <= 1
        if self._is_homophonic_cipher() and no_boundaries:
            return (
                f"{search_kind} has produced a candidate for a no-boundary "
                "homophonic cipher. Be conservative: high dictionary_rate can "
                "be a false positive because segmentation finds short words in "
                "wrong text. Before declaring, call score_panel and "
                "observe_homophone_distribution; if common letters are absent, "
                "use decode_absent_letter_candidates on those letters. Declare "
                "only when the full decoded stream reads as coherent prose, not "
                "just scattered words."
            )
        if search_kind == "hill_climb":
            if no_boundaries:
                return (
                    "hill_climb has converged. If the decoded_preview is "
                    "readable, declare now. If still garbled, this branch's "
                    "local-search has maxed out — fork a fresh branch and "
                    "restart with search_anneal (stronger explorer)."
                )
            return (
                "hill_climb has converged. If decoded text shows any "
                f"recognisable {self.language} words, declare now. If still "
                "stuck, fork and restart with search_anneal — it escapes "
                "local optima that hill_climb can't."
            )
        # search_kind == "anneal"
        if no_boundaries:
            return (
                "Read decoded_preview carefully. If you can segment most of it "
                f"into {self.language} words, DECLARE NOW — chasing perfection "
                "risks regression and zero score. If a few obvious letter errors "
                "remain, call decode_diagnose_and_fix(branch) — it identifies the "
                "culprit cipher symbol for EACH error and applies ALL fixes in one "
                "call. You can also batch multiple act_set_mapping calls in a "
                "single response. Do NOT fix errors one per iteration; each "
                "iteration costs tokens and grows the context."
            )
        return (
            f"Read decoded_preview carefully. If ANY recognisable {self.language} "
            "words appear, call meta_declare_solution now — a partial solution "
            "scores better than no declaration. If a few residual errors remain, "
            "call decode_diagnose_and_fix(branch) to fix them all in one call, "
            "then declare."
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
        homophonic_no_boundary = (
            self._is_homophonic_cipher() and self._is_no_boundary_cipher()
        )

        def dict_rate_for_text(text: str) -> float:
            normalized = sig.normalize_for_scoring(text)
            if not normalized.strip():
                return 0.0
            if not any(c.isspace() for c in normalized.strip()):
                return segment_text(normalized, word_set, self._freq_rank).dict_rate

            words = [w for w in normalized.split() if any(c.isalpha() for c in w)]
            if not words:
                return 0.0
            return sum(1 for w in words if w in word_set) / len(words)

        def score_dict() -> float:
            return dict_rate_for_text(temp_session.apply_key())

        def score_quad() -> float:
            normalized = sig.normalize_for_scoring(temp_session.apply_key())
            return ngram.normalized_ngram_score(normalized, quad_lp, n=4)

        def score_combined() -> float:
            text = temp_session.apply_key()
            dr = dict_rate_for_text(text)
            normalized = sig.normalize_for_scoring(text)
            quad = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
            chi2 = unigram_chi2(normalized, language)
            if homophonic_no_boundary:
                # Segmented dictionary rate is easy to inflate on continuous
                # homophonic text, so let n-grams and letter distribution
                # dominate the search objective.
                return dr * 2 + quad - 0.05 * chi2
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
            "note": self._search_declare_note("hill_climb"),
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
        existing_key = dict(branch.key)
        if "preserve_existing" in args:
            preserve_existing = bool(args["preserve_existing"])
        else:
            preserve_existing = 0 < len(existing_key) < len(all_ct_ids)
        anchors = existing_key if preserve_existing else {}
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
            "preserve_existing": preserve_existing,
            "preserved_symbols": len(anchors),
            "auto_seeded_symbols": seeded_count,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "note": self._search_declare_note("anneal"),
        }

    def _tool_search_homophonic_anneal(self, args: dict) -> Any:
        branch_name = args["branch"]
        epochs = int(args.get("epochs", 5))
        sampler_iterations = int(args.get("sampler_iterations", 2000))
        t_start = float(args.get("t_start", 0.012))
        t_end = float(args.get("t_end", 0.006))
        order = int(args.get("order", 5))
        preserve_existing = bool(args.get("preserve_existing", False))
        model_path = args.get("model_path")
        max_ngrams = int(args.get("max_ngrams", 3_000_000))
        distribution_weight = float(args.get("distribution_weight", 4.0))
        seed = args.get("seed")
        seed = int(seed) if seed is not None else None
        top_n = max(1, int(args.get("top_n", 1)))
        write_candidate_branches = bool(args.get("write_candidate_branches", False))

        ws = self.workspace
        branch = ws.get_branch(branch_name)
        tokens = list(ws.cipher_text.tokens)
        pt_alpha = ws.plaintext_alphabet
        plaintext_ids = [
            i for i in range(pt_alpha.size)
            if len(pt_alpha.symbol_for(i)) == 1 and pt_alpha.symbol_for(i).isalpha()
        ]
        if not tokens:
            return {"error": "cipher text is empty"}
        if len(plaintext_ids) < 2:
            return {"error": "plaintext alphabet must contain alphabetic letters"}

        existing_key = dict(branch.key)
        fixed_cipher_ids = set(existing_key.keys()) if preserve_existing else set()
        id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
        letter_to_id = {letter: i for i, letter in id_to_letter.items()}
        before = self._compute_quick_scores(branch_name)

        model, model_note = self._homophonic_model(
            model_path=model_path,
            order=order,
            max_ngrams=max_ngrams,
        )
        result = homophonic.homophonic_simulated_anneal(
            tokens=tokens,
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            model=model,
            initial_key=existing_key,
            fixed_cipher_ids=fixed_cipher_ids,
            epochs=max(1, epochs),
            sampler_iterations=max(1, sampler_iterations),
            t_start=t_start,
            t_end=t_end,
            distribution_weight=distribution_weight,
            seed=seed,
            top_n=top_n,
        )

        ws.set_full_key(branch_name, result.key)
        after = self._compute_quick_scores(branch_name)
        distribution = self._tool_observe_homophone_distribution({"branch": branch_name})
        candidates = []
        for rank, candidate in enumerate(result.candidates, start=1):
            candidate_branch = branch_name if rank == 1 else None
            if write_candidate_branches and rank > 1:
                candidate_branch = f"{branch_name}_cand{rank}"
                if not ws.has_branch(candidate_branch):
                    ws.fork(candidate_branch, from_branch=branch_name)
                ws.set_full_key(candidate_branch, candidate.key)
            candidates.append({
                "rank": rank,
                "branch": candidate_branch,
                "epoch": candidate.epoch,
                "anneal_score": round(candidate.normalized_score, 4),
                "raw_anneal_score": round(candidate.score, 2),
                "decoded_preview": candidate.plaintext[:1200],
            })
        return {
            "branch": branch_name,
            "solver": "native_homophonic_anneal",
            "before_scores": before,
            "after_scores": after,
            "score_delta": self._score_delta(before, after),
            "anneal_score": round(result.normalized_score, 4),
            "raw_anneal_score": round(result.score, 2),
            "model_source": model.source,
            "model_ngrams": len(model.log_probs),
            "model_note": model_note,
            "epochs": result.epochs,
            "sampler_iterations": result.sampler_iterations,
            "accepted_moves": result.accepted_moves,
            "improved_moves": result.improved_moves,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "preserve_existing": preserve_existing,
            "fixed_symbols": result.fixed_symbols,
            "cipher_symbols": result.metadata.get("cipher_symbols"),
            "distribution_weight": result.metadata.get("distribution_weight"),
            "top_n": top_n,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "homophone_warnings": distribution.get("warnings", []),
            "note": (
                "This native homophonic annealer is intended to replace manual "
                "guesswork on many-symbol no-boundary ciphers. Read the preview; "
                "if it is mostly correct, use decode_diagnose/decode_ambiguous_letter "
                "for residual rare-letter fixes or declare the solution."
            ),
        }

    def _homophonic_model(
        self,
        model_path: str | None,
        order: int,
        max_ngrams: int,
    ) -> tuple[homophonic.ContinuousNGramModel, str]:
        requested = (model_path or "").strip()
        if requested.lower() in {"word_list", "wordlist", "none", "fallback"}:
            return (
                homophonic.build_continuous_ngram_model(self.word_list, order=order),
                "Using small word-list fallback by request.",
            )

        candidate = Path(requested).expanduser() if requested else _default_homophonic_model_path()
        if candidate and candidate.exists():
            try:
                return (
                    homophonic.load_zenith_csv_model(
                        candidate,
                        order=order,
                        max_ngrams=max(1, max_ngrams),
                    ),
                    "Using continuous corpus n-gram model.",
                )
            except OSError as exc:
                return (
                    homophonic.build_continuous_ngram_model(self.word_list, order=order),
                    f"Could not load corpus model ({exc}); using word-list fallback.",
                )

        return (
            homophonic.build_continuous_ngram_model(self.word_list, order=order),
            "No corpus model found; using small word-list fallback.",
        )

    # ------------------------------------------------------------------
    # run_python
    # ------------------------------------------------------------------
    def _tool_run_python(self, args: dict) -> Any:
        code = args.get("code", "")
        justification = args.get("justification", "")
        output = _execute_python(code)
        return {
            "output": output,
            "design_review": {
                "justification": justification,
                "note": (
                    "run_python is allowed, but this call should be reviewed as "
                    "a possible missing-tool signal."
                ),
            },
        }

    # ------------------------------------------------------------------
    # meta_*
    # ------------------------------------------------------------------
    def _tool_meta_request_tool(self, args: dict) -> Any:
        req: dict = {
            "iteration": self._current_iteration,
            "description": args["description"],
            "rationale": args["rationale"],
        }
        if "example_input" in args:
            req["example_input"] = args["example_input"]
        if "example_output" in args:
            req["example_output"] = args["example_output"]
        self.tool_requests.append(req)
        return {
            "status": "recorded",
            "message": (
                "Tool request noted — it will appear in the benchmark report. "
                "Use run_python as a workaround for this calculation now."
            ),
        }

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


def _default_homophonic_model_path() -> Path | None:
    env_path = os.environ.get("DECIPHER_HOMOPHONIC_MODEL")
    if env_path:
        return Path(env_path).expanduser()

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "other_tools" / "zenith-2026.2" / "zenith-model.csv",
        repo_root / "other_tools" / "zenith" / "zenith-model.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _execute_python(code: str, timeout: int = 15) -> str:
    """Run Python code in a subprocess; return stdout + stderr, truncated."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + "STDERR:\n" + result.stderr
        return output[:3000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
