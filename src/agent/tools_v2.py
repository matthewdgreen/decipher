"""v2 tool definitions and executor.

Tools are organized by namespace:
  workspace_*  — branch lifecycle and branch cards
  observe_*    — read-only text analysis (frequency, isomorphs)
  decode_*     — read-only views, diagnostics, validation, and repair plans
  score_*      — signal panel and individual signals
  corpus_*     — language data (wordlist, patterns)
  act_*        — branch/key/boundary mutations
  search_*     — classical algorithms on a branch (hill_climb, anneal)
  repair_agenda_* — durable reading-repair hypothesis bookkeeping
  run_python   — execute arbitrary stdlib Python code (escape hatch)
  meta_*       — declare_solution (terminates the run), request_tool (feedback)

All mutating and reading tools take an explicit branch name.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analysis import dictionary, frequency, homophonic, ic, ngram, pattern
from analysis import signals as sig
from analysis.frequency import unigram_chi2
from analysis.segment import repair_no_boundary_text, segment_text
from analysis.solver import hill_climb_swaps, simulated_anneal
from analysis.transformers import (
    TransformPipeline,
    apply_transform_pipeline,
)
from analysis.transform_search import (
    inspect_transform_suspicion,
    screen_transform_candidates,
)
from automated import runner as automated_runner
from automated.runner import run_automated
from artifact.schema import SolutionDeclaration, ToolCall
from benchmark.context import ScopedBenchmarkContext, safe_read_benchmark_file
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
            "hypothesis without committing to main. If an automated preflight "
            "or other readable branch exists and you want to repair it, prefer "
            "`workspace_fork_best` so you do not accidentally fork empty `main`."
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
        "name": "workspace_fork_best",
        "description": (
            "Create a new repair branch from the strongest existing branch. "
            "Use this when an automated preflight branch exists or when you "
            "want to repair the current best decode. It avoids accidentally "
            "forking empty `main`; the result tells you exactly which source "
            "branch was copied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "new_name": {
                    "type": "string",
                    "description": "Name for the new branch (alphanumeric, _ or -).",
                },
                "prefer_branch": {
                    "type": "string",
                    "description": (
                        "Optional source branch to copy if it exists. Omit to "
                        "let the tool choose the strongest branch."
                    ),
                },
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
        "name": "workspace_branch_cards",
        "description": (
            "Show compact branch state cards: scores, mapped count, readable "
            "excerpt, applied/held/open repair agenda items, and risk warnings. "
            "Use this before declaration when multiple branches or repair "
            "hypotheses exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Optional branch name; omit to show all branches.",
                },
            },
        },
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
    {
        "name": "observe_transform_pipeline",
        "description": (
            "Inspect ciphertext-transform state for a branch: current grid "
            "metadata, active token-order overlay, and any applied "
            "Zenith-compatible transform pipeline. Use this when a cipher may "
            "be transposition+homophonic and the reading order itself may be wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string", "default": "main"}},
        },
    },
    {
        "name": "observe_transform_suspicion",
        "description": (
            "Cheap diagnostic for deciding whether transform search is worth "
            "trying on a cipher with unknown or incomplete type metadata. It "
            "reports plausible grid dimensions, homophonic/order-scramble "
            "signals, and a conservative recommendation. Use this before "
            "spending solver budget on search_transform_homophonic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "columns": {
                    "type": "integer",
                    "description": "Optional suspected grid width.",
                },
                "baseline_status": {
                    "type": "string",
                    "description": "Optional status from a baseline solve.",
                },
                "baseline_score": {
                    "type": "number",
                    "description": "Optional normalized baseline quality score if available.",
                },
            },
        },
    },
    {
        "name": "search_transform_candidates",
        "description": (
            "Run a structural-only transform candidate search. This can use "
            "fast, broad, or wide breadth without spending homophonic solver "
            "budget. Use it when deciding whether a large transform search is "
            "worth promoting into search_transform_homophonic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "columns": {
                    "type": "integer",
                    "description": "Optional suspected grid width.",
                },
                "breadth": {
                    "type": "string",
                    "enum": ["fast", "broad", "wide"],
                    "default": "broad",
                },
                "top_n": {"type": "integer", "default": 40},
                "max_generated_candidates": {
                    "type": "integer",
                    "default": 25000,
                    "description": "Safety cap for structural candidate generation.",
                },
                "include_program_search": {"type": "boolean", "default": False},
                "program_max_depth": {"type": "integer", "default": 5},
                "program_beam_width": {"type": "integer", "default": 48},
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
            "**This is the default primitive for reading-driven repair.** "
            "cipher_symbol must be the name of a symbol from the cipher alphabet "
            "(e.g. 'S001', 'A', 'X') — NOT a letter you see in the decoded output. "
            "Unidirectional and surgical: only words containing this cipher "
            "symbol change. Result includes a `changed_words` sample (was→now) "
            "so you can decide by reading rather than by score. The score_delta "
            "is advisory — on boundary-preserving ciphers a correct cipher- "
            "symbol fix can drop dictionary_rate while still being correct. "
            "If two or more `changed_words` entries now read as real target- "
            "language words (or fragments of real words), keep the change."
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
            "**Bidirectional letter-population swap** — exchanges the cipher "
            "symbols mapped to two decoded letters across the entire branch. "
            "**Rarely the right primitive for reading-driven repairs**: if you "
            "see decoded T in a word that should be B, this tool will move "
            "every cipher symbol currently producing T to produce B *and* "
            "every symbol producing B to produce T, almost always breaking "
            "correctly-decoded words elsewhere. For 'this letter should be "
            "that letter in this word' fixes, use `act_set_mapping` on the "
            "single cipher symbol producing the wrong letter — that is "
            "unidirectional and surgical. Use act_swap_decoded only when you "
            "deliberately want to swap two whole decoded-letter populations "
            "(e.g. a confirmed full ↔ swap such as A↔E). When auto-reverted, "
            "the result includes `unidirectional_alternatives` listing the "
            "specific act_set_mapping calls that would have made the same "
            "intent without the bidirectional side-effects."
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
    {
        "name": "act_split_cipher_word",
        "description": (
            "Split one ciphertext word into two words on this branch only. "
            "Use this when the transcription's word boundary seems to be "
            "missing, e.g. UELPULLO -> UEL | PULLO."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_word_index": {"type": "integer"},
                "split_at_token_offset": {
                    "type": "integer",
                    "description": "Offset inside the chosen word where the new word should begin.",
                },
            },
            "required": ["branch", "cipher_word_index", "split_at_token_offset"],
        },
    },
    {
        "name": "act_merge_cipher_words",
        "description": (
            "Merge two adjacent ciphertext words on this branch only. "
            "Use this when a transcription likely inserted a spurious "
            "boundary, e.g. CUR | A -> CURA. Numeric word indices shift "
            "after every split/merge; when acting from decoded text, prefer "
            "`act_merge_decoded_words` or re-run decode_show/decode_diagnose "
            "before using another numeric index."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "left_word_index": {
                    "type": "integer",
                    "description": "Index of the left word in the adjacent pair to merge.",
                },
            },
            "required": ["branch", "left_word_index"],
        },
    },
    {
        "name": "act_merge_decoded_words",
        "description": (
            "Merge the currently adjacent cipher words whose decoded forms "
            "match left_decoded and right_decoded. This is the safest boundary "
            "tool when you are reading the plaintext directly, because it "
            "finds the current pair after earlier merges have shifted numeric "
            "word indices. Example: left_decoded='AP', right_decoded='PLY' "
            "merges the current AP | PLY pair into APPLY."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "left_decoded": {"type": "string"},
                "right_decoded": {"type": "string"},
                "occurrence": {
                    "type": "integer",
                    "description": "Which matching adjacent pair to merge; defaults to 0.",
                    "default": 0,
                },
            },
            "required": ["branch", "left_decoded", "right_decoded"],
        },
    },
    {
        "name": "act_apply_boundary_candidate",
        "description": (
            "Apply one of the currently suggested boundary edits for this branch. "
            "Use this when decode_diagnose or decode_diagnose_and_fix returns "
            "boundary_candidates or a recommended_next_tool for split/merge. "
            "It recomputes candidates each call, so repeated calls are safer "
            "than manually reusing old numeric word indices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "candidate_index": {
                    "type": "integer",
                    "description": "Index into the current boundary_candidates list; defaults to 0.",
                    "default": 0,
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "act_apply_word_repair",
        "description": (
            "Apply a same-length reading-driven word repair such as TREUITER "
            "-> BREUITER. This is a low-friction wrapper around the correct "
            "act_set_mapping calls: it locates the word, identifies the "
            "cipher symbols responsible for differing letters, applies those "
            "symbol mappings, and returns changed_words so you can judge the "
            "repair by reading. Provide either cipher_word_index or "
            "decoded_word plus optional occurrence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "target_word": {"type": "string"},
                "cipher_word_index": {"type": "integer"},
                "decoded_word": {"type": "string"},
                "occurrence": {"type": "integer", "default": 0},
            },
            "required": ["branch", "target_word"],
        },
    },
    {
        "name": "act_resegment_by_reading",
        "description": (
            "Replace this branch's word boundaries with a complete reading that "
            "you propose, while preserving the exact decoded character stream. "
            "This is the one-shot boundary-normalization tool: instead of "
            "merging THERE | FORE, AP | PLY, UN | TO, AFTER | WARD one at a "
            "time, provide proposed_text='THEREFORE THE ... APPLY ... UNTO ...' "
            "and the tool applies the new word spans only if the letters match "
            "exactly after removing spaces/punctuation. If you need to change "
            "letters too, first call decode_validate_reading_repair and then "
            "use act_set_mapping/act_bulk_set for the character repairs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": (
                        "Complete best reading with desired word boundaries. "
                        "Must have the same letters as the current branch."
                    ),
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "act_resegment_from_reading_repair",
        "description": (
            "Apply only the word-boundary pattern implied by a proposed best "
            "reading, while preserving the branch's current decoded letters "
            "and key. Use this when the branch is readable-but-damaged and "
            "your best reading changes both spaces and a few letters. Example: "
            "current PHYSICS ER, proposed PHYSICKER has the same character "
            "count, so this tool can safely install one word boundary span and "
            "leave the current letters as PHYSICSER. It returns mismatch spans "
            "so you can then repair the true letter/key errors with "
            "act_set_mapping or act_bulk_set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": (
                        "Best target-language reading. The letters may differ "
                        "from the current branch, but the normalized character "
                        "count must match."
                    ),
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "act_resegment_window_by_reading",
        "description": (
            "Apply word-boundary changes to a local window of decoded words "
            "instead of rewriting the entire plaintext stream. Use this for "
            "repairs like LIBE | BITUR -> LIBEBITUR, A | RI -> ARI, or "
            "POTESTQUIBUS -> POTEST | QUIBUS. The proposed text only needs to "
            "cover the selected window. If proposed letters differ but the "
            "character count matches, the tool applies only the boundary "
            "pattern to the current decoded letters and returns mismatch spans "
            "for later key repair."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "start_word_index": {
                    "type": "integer",
                    "description": "First current decoded word in the local window.",
                },
                "word_count": {
                    "type": "integer",
                    "description": "Number of current decoded words in the window.",
                },
                "proposed_text": {
                    "type": "string",
                    "description": "Desired reading/boundaries for just this window.",
                },
            },
            "required": ["branch", "start_word_index", "word_count", "proposed_text"],
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
        "name": "search_automated_solver",
        "description": (
            "Run Decipher's current best automated local solver stack on the "
            "current ciphertext and install the resulting full key onto the "
            "named branch. This mirrors the no-LLM automated runner used by "
            "frontier/parity evaluation, including the modern `zenith_native` "
            "homophonic path when the routing logic selects it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "homophonic_budget": {
                    "type": "string",
                    "enum": ["full", "screen"],
                    "default": "full",
                },
                "homophonic_refinement": {
                    "type": "string",
                    "enum": ["none", "two_stage", "targeted_repair", "family_repair"],
                    "default": "none",
                },
                "homophonic_solver": {
                    "type": "string",
                    "enum": ["zenith_native", "legacy"],
                    "default": "zenith_native",
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
                "solver_profile": {
                    "type": "string",
                    "enum": ["zenith_native", "legacy"],
                    "default": "zenith_native",
                    "description": "Select the modern or legacy homophonic solver path.",
                },
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
                "diversity_weight": {
                    "type": "number",
                    "default": 1.5,
                    "description": (
                        "Weight for plaintext diversity penalty; useful on short "
                        "homophonic texts where collapsed low-letter solutions can "
                        "look deceptively good to n-gram scoring."
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
        "name": "act_apply_transform_pipeline",
        "description": (
            "Apply a Zenith-compatible ciphertext transform pipeline to this "
            "branch's reading order. This does not change symbol mappings; it "
            "changes the branch's token order so subsequent decode/search tools "
            "operate on the transformed ciphertext."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "pipeline": {
                    "type": "object",
                    "description": (
                        "Pipeline object with optional columns/rows and steps, "
                        "where each step is {name, data}."
                    ),
                },
            },
            "required": ["branch", "pipeline"],
        },
    },
    {
        "name": "search_transform_homophonic",
        "description": (
            "Try a bounded set of ciphertext transform candidates. For each "
            "candidate, apply the transform, run a short homophonic search, "
            "rank the candidates, and optionally write the best transformed "
            "branch. With include_program_search=true, also construct small "
            "transform pipelines from a grammar before probing finalists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "columns": {"type": "integer"},
                "profile": {"type": "string", "enum": ["small", "medium", "wide"], "default": "small"},
                "top_n": {"type": "integer", "default": 3},
                "max_generated_candidates": {
                    "type": "integer",
                    "description": "Safety cap for structural candidate generation before solver promotion.",
                },
                "write_best_branch": {"type": "boolean", "default": True},
                "homophonic_budget": {"type": "string", "enum": ["screen", "full"], "default": "screen"},
                "include_program_search": {"type": "boolean", "default": False},
                "program_max_depth": {"type": "integer", "default": 5},
                "program_beam_width": {"type": "integer", "default": 24},
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
    {
        "name": "decode_repair_no_boundary",
        "description": (
            "Run a conservative text-only repair pass over a branch's decoded "
            "plaintext for no-boundary or drifted output. It segments the "
            "current plaintext, applies confident one-edit word repairs, and "
            "tries local re-segmentation over suspicious windows. This does "
            "not mutate the branch key; it returns a repaired plaintext "
            "candidate and before/after segmentation metrics so you can judge "
            "whether the branch is close-but-drifted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "max_rounds": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum repair rounds.",
                },
                "max_window_words": {
                    "type": "integer",
                    "default": 3,
                    "description": "Largest adjacent word window to re-segment.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "decode_validate_reading_repair",
        "description": (
            "Validate a proposed best reading of a branch. This is read-only. "
            "Use it when your human/agent reading wants to change letters, "
            "archaic spellings, or word boundaries. The tool compares the "
            "proposed text with the branch's current decoded character stream, "
            "reports whether it is character-preserving, shows first mismatch "
            "spans, and scores the proposed words against the target-language "
            "dictionary. If the proposal is character-preserving, apply it "
            "with act_resegment_by_reading. If it changes characters but has "
            "the same character count, apply its boundary pattern safely with "
            "act_resegment_from_reading_repair, then translate the mismatch "
            "spans into act_set_mapping/act_bulk_set repairs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": (
                        "Your best target-language reading. Spaces may differ; "
                        "letters may differ only if this is a repair hypothesis."
                    ),
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "decode_plan_word_repair",
        "description": (
            "Plan a reading-driven word repair such as TREUITER -> BREUITER "
            "without mutating the branch. Use this when you can read a decoded "
            "word or fragment and want the tool to identify the responsible "
            "cipher symbol changes. Provide either cipher_word_index or "
            "decoded_word plus optional occurrence. Same-length repairs return "
            "proposed act_set_mapping-style changes, a changed_words preview, "
            "and an act_apply_word_repair suggested call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "target_word": {
                    "type": "string",
                    "description": "The intended reading for this word, e.g. BREUITER.",
                },
                "cipher_word_index": {
                    "type": "integer",
                    "description": "Optional numeric word index to repair.",
                },
                "decoded_word": {
                    "type": "string",
                    "description": "Optional current decoded word to locate, e.g. TREUITER.",
                },
                "occurrence": {
                    "type": "integer",
                    "description": "Which matching decoded_word occurrence to use; defaults to 0.",
                    "default": 0,
                },
            },
            "required": ["branch", "target_word"],
        },
    },
    {
        "name": "decode_plan_word_repair_menu",
        "description": (
            "Compare several possible same-length readings for the same decoded "
            "word without mutating the branch. Use this before applying an "
            "uncertain word repair: it shows each option's proposed cipher-symbol "
            "mappings, intra-word conflicts, changed-word preview, dictionary-hit "
            "delta, collateral-change count, and suggested act_apply_word_repair "
            "call when safe. This is the repair menu: choose from evidence instead "
            "of applying the first plausible word."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "target_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Candidate intended readings to compare, e.g. "
                        "['PLURES', 'RLURES', 'BLURES']."
                    ),
                },
                "cipher_word_index": {
                    "type": "integer",
                    "description": "Optional numeric word index to repair.",
                },
                "decoded_word": {
                    "type": "string",
                    "description": "Optional current decoded word to locate, e.g. RLURES.",
                },
                "occurrence": {
                    "type": "integer",
                    "description": "Which matching decoded_word occurrence to use; defaults to 0.",
                    "default": 0,
                },
            },
            "required": ["branch", "target_words"],
        },
    },
    {
        "name": "repair_agenda_list",
        "description": (
            "List durable reading-repair agenda items accumulated during this "
            "run. Use this before declaring if you have been making or "
            "considering word-level reading repairs, so open hypotheses are "
            "not lost in the transcript."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Optional branch filter.",
                },
                "status": {
                    "type": "string",
                    "description": "Optional status filter such as open, applied, held, rejected, blocked.",
                },
            },
        },
    },
    {
        "name": "repair_agenda_update",
        "description": (
            "Update the status or notes for a durable reading-repair agenda "
            "item. Use this when you decide a proposed repair is held, "
            "rejected, blocked, or otherwise resolved without applying it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["open", "applied", "held", "rejected", "blocked", "reverted"],
                },
                "notes": {"type": "string"},
            },
            "required": ["item_id", "status"],
        },
    },
    # ----- benchmark_* -----
    {
        "name": "inspect_benchmark_context",
        "description": (
            "Inspect the scoped benchmark context made available for this run: "
            "selected policy, injected layers, target/context records, related "
            "records, and associated documents. This does not read arbitrary "
            "files and only exposes manifest-declared context for the current "
            "benchmark test."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_layer_text": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include the short text of injected context layers.",
                },
            },
        },
    },
    {
        "name": "list_related_records",
        "description": (
            "List benchmark records that this test explicitly allows as "
            "related/context material. Does not expose plaintext; use "
            "`inspect_related_solution` only when policy permits solution access."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_related_transcription",
        "description": (
            "Read the canonical transcription of an allowed context or related "
            "record. The record must be listed by the benchmark split or by the "
            "record's `related_records` metadata; arbitrary filesystem paths "
            "are not accepted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "default": 6000,
                    "description": "Maximum characters to return.",
                },
            },
            "required": ["record_id"],
        },
    },
    {
        "name": "inspect_related_solution",
        "description": (
            "Read plaintext/solution text for an allowed related record, but "
            "only when the benchmark context policy explicitly permits "
            "solution-bearing related context. This is for controlled "
            "context-ablation runs, not blind parity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "default": 6000,
                    "description": "Maximum characters to return.",
                },
            },
            "required": ["record_id"],
        },
    },
    {
        "name": "list_associated_documents",
        "description": (
            "List long-form documents explicitly associated with the current "
            "benchmark record, such as letters, plaintext notes, envelopes, or "
            "source commentary. Use `inspect_associated_document` to read an "
            "allowed document."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_associated_document",
        "description": (
            "Read an explicitly associated benchmark document by document ID. "
            "Only manifest-declared files under the benchmark root are allowed; "
            "arbitrary paths are rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "default": 6000,
                    "description": "Maximum characters to return.",
                },
            },
            "required": ["document_id"],
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
            "uncertainty, a brief human-readable reading summary, whether "
            "more iterations would likely help, and your self-confidence "
            "(0.0 - 1.0). This ends the session — call it when you believe "
            "you have the best answer you can produce or when further progress "
            "seems impossible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "rationale": {"type": "string"},
                "self_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reading_summary": {
                    "type": "string",
                    "description": (
                        "Brief plain-language summary for the final screen: "
                        "what the decipherment appears to say, especially if "
                        "the target language is not the user's native language."
                    ),
                },
                "further_iterations_helpful": {
                    "type": "boolean",
                    "description": (
                        "True if additional iterations would likely improve "
                        "the decipherment; false if remaining gains seem minor."
                    ),
                },
                "further_iterations_note": {
                    "type": "string",
                    "description": (
                        "One or two sentences explaining what further "
                        "iterations should try, or why they are probably not "
                        "needed."
                    ),
                },
                "forced_partial": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Set true only when you are intentionally submitting a "
                        "low-confidence partial hypothesis before the final "
                        "iteration because no useful remaining tool action is "
                        "available. If further_iterations_helpful is true, this "
                        "is usually false."
                    ),
                },
            },
            "required": [
                "branch",
                "rationale",
                "self_confidence",
                "reading_summary",
                "further_iterations_helpful",
                "further_iterations_note",
            ],
        },
    },
]


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


class WorkspaceToolExecutor:
    """Dispatches v2 tool calls against a Workspace + language resources."""

    def __init__(
        self,
        workspace: Workspace,
        language: str,
        word_set: set[str],
        word_list: list[str],
        pattern_dict: dict[str, list[str]],
        benchmark_context: ScopedBenchmarkContext | None = None,
    ) -> None:
        self.workspace = workspace
        self.language = language
        self.word_set = word_set
        self.word_list = word_list
        self.pattern_dict = pattern_dict
        self.benchmark_context = benchmark_context

        # Frequency rank for lookup (1-based; lower = more common)
        self._freq_rank: dict[str, int] = {
            w.upper(): i + 1 for i, w in enumerate(word_list)
        }

        # Termination state
        self.terminated: bool = False
        self.solution: SolutionDeclaration | None = None
        self.max_iterations: int | None = None
        self.allowed_tool_names: set[str] | None = None

        # Log of all tool calls for the run artifact
        self.call_log: list[ToolCall] = []
        self._current_iteration: int = 0

        # Tool capability requests (meta_request_tool calls)
        self.tool_requests: list[dict] = []

        # Durable reading-repair agenda. These are deliberately plain dicts so
        # they serialize directly into run artifacts and tool results.
        self.repair_agenda: list[dict[str, Any]] = []
        self._next_repair_agenda_id: int = 1

    def set_iteration(self, n: int) -> None:
        self._current_iteration = n

    def set_max_iterations(self, n: int | None) -> None:
        self.max_iterations = n

    def set_allowed_tool_names(self, names: set[str] | None) -> None:
        self.allowed_tool_names = set(names) if names is not None else None

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
        if self.allowed_tool_names is not None and tool_name not in self.allowed_tool_names:
            allowed = sorted(self.allowed_tool_names)
            result = _json({
                "error": (
                    f"STOP: `{tool_name}` is no longer allowed on this turn. "
                    "Do not call it again in this gated window."
                ),
                "reason": "tool_gated",
                "attempted_tool": tool_name,
                "allowed_tools": allowed,
                "note": (
                    "This turn is gated. Some tools are intentionally hidden "
                    "and executor-blocked. You must choose only from "
                    "`allowed_tools`. Do not use local split/merge, bulk "
                    "mapping, or search tools now. If the text is readable but "
                    "misaligned, write a complete best reading and use "
                    "`act_resegment_by_reading` or "
                    "`act_resegment_from_reading_repair`; use "
                    "`decode_validate_reading_repair` only if you need to "
                    "decide which resegmentation actuator applies."
                ),
            })
        elif handler is None:
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

    def _is_no_boundary_cipher(self, branch_name: str | None = None) -> bool:
        if branch_name is None:
            return len(self.workspace.cipher_text.words) <= 1
        return len(self.workspace.effective_words(branch_name)) <= 1

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
        if self._is_no_boundary_cipher(branch) and dict_rate is not None and quad is not None:
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
        words = self._decoded_words(branch_name, max_words=max_words)
        return " | ".join(words)

    def _decoded_words(
        self, branch_name: str, max_words: int | None = None
    ) -> list[str]:
        """Return per-word decoded strings for the branch.

        Used by `_changed_words_sample` to diff before/after decodes word by
        word. With `max_words=None` returns the full sequence.
        """
        ws = self.workspace
        words = ws.effective_words(branch_name)
        branch = ws.get_branch(branch_name)
        pt_alpha = self._pt_alpha()
        sep = " " if pt_alpha._multisym else ""
        if max_words is not None:
            words = words[:max_words]
        out: list[str] = []
        for w in words:
            parts = [
                pt_alpha.symbol_for(branch.key[t]) if t in branch.key else "?"
                for t in w
            ]
            out.append(sep.join(parts))
        return out

    def _decoded_words_with_key(
        self,
        branch_name: str,
        key: dict[int, int],
        max_words: int | None = None,
    ) -> list[str]:
        ws = self.workspace
        words = ws.effective_words(branch_name)
        pt_alpha = self._pt_alpha()
        sep = " " if pt_alpha._multisym else ""
        if max_words is not None:
            words = words[:max_words]
        out: list[str] = []
        for w in words:
            parts = [
                pt_alpha.symbol_for(key[t]) if t in key else "?"
                for t in w
            ]
            out.append(sep.join(parts))
        return out

    def _locate_word_for_repair(
        self,
        branch: str,
        args: dict,
    ) -> tuple[int | None, dict[str, Any] | None]:
        words = self.workspace.effective_words(branch)
        if args.get("cipher_word_index") is not None:
            idx = int(args["cipher_word_index"])
            if idx < 0 or idx >= len(words):
                return None, {
                    "error": f"cipher_word_index {idx} out of range (0..{len(words) - 1})"
                }
            return idx, None

        decoded_target = sig.normalize_for_scoring(str(args.get("decoded_word", "")))
        decoded_target = self._reading_char_stream(decoded_target)
        if not decoded_target:
            return None, {
                "error": (
                    "Provide either cipher_word_index or decoded_word to locate "
                    "the repair target."
                )
            }
        occurrence = int(args.get("occurrence", 0))
        decoded_words = [
            self._reading_char_stream(word)
            for word in self._decoded_words(branch, max_words=None)
        ]
        matches = [
            idx for idx, word in enumerate(decoded_words)
            if word == decoded_target
        ]
        if not matches:
            return None, {
                "error": f"decoded_word {decoded_target!r} not found on branch {branch!r}",
                "decoded_word": decoded_target,
                "sample_decoded_words": decoded_words[:40],
            }
        if occurrence < 0 or occurrence >= len(matches):
            return None, {
                "error": (
                    f"occurrence {occurrence} out of range for decoded_word "
                    f"{decoded_target!r}; found {len(matches)} match(es)"
                ),
                "matching_indices": matches,
            }
        return matches[occurrence], None

    def _word_repair_plan(self, args: dict) -> dict[str, Any]:
        branch = args["branch"]
        idx, error = self._locate_word_for_repair(branch, args)
        if error:
            return error
        assert idx is not None

        target = self._reading_char_stream(str(args["target_word"]))
        if not target:
            return {"error": "target_word must contain at least one alphabetic character"}

        words = self.workspace.effective_words(branch)
        word_tokens = words[idx]
        current_word = self._reading_char_stream(self._decode_word(word_tokens, branch))
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        cipher_word = self._cipher_word_str(word_tokens)
        base: dict[str, Any] = {
            "branch": branch,
            "cipher_word_index": idx,
            "cipher_word": cipher_word,
            "current_decoded": current_word,
            "target_word": target,
            "same_length": len(current_word) == len(target),
            "target_in_dictionary": target in self.word_set,
        }
        if len(current_word) != len(target):
            base.update({
                "applicable": False,
                "reason": (
                    "Word repair requires a same-length target. If the target "
                    "adds/removes letters, use boundary/full-reading tools first."
                ),
                "current_length": len(current_word),
                "target_length": len(target),
            })
            return base

        desired_by_token: dict[int, dict[str, Any]] = {}
        conflicts: list[dict[str, Any]] = []
        for pos, (token_id, current_ch, target_ch) in enumerate(
            zip(word_tokens, current_word, target)
        ):
            if not pt_alpha.has_symbol(target_ch):
                conflicts.append({
                    "position": pos,
                    "cipher_symbol": alpha.symbol_for(token_id),
                    "current": current_ch,
                    "target": target_ch,
                    "reason": "target letter is not in plaintext alphabet",
                })
                continue
            target_id = pt_alpha.id_for(target_ch)
            existing = desired_by_token.get(token_id)
            if existing is not None and existing["target_id"] != target_id:
                conflicts.append({
                    "position": pos,
                    "cipher_symbol": alpha.symbol_for(token_id),
                    "current": current_ch,
                    "target": target_ch,
                    "reason": (
                        "same cipher symbol appears multiple times in this word "
                        "and the proposed target would require different letters"
                    ),
                    "previous_position": existing["position"],
                    "previous_target": existing["target"],
                })
                continue
            if existing is None:
                desired_by_token[token_id] = {
                    "target_id": target_id,
                    "target": target_ch,
                    "position": pos,
                    "current": current_ch,
                }

        branch_key = self.workspace.get_branch(branch).key
        changes: list[dict[str, Any]] = []
        token_mappings: dict[int, int] = {}
        for token_id, desired in desired_by_token.items():
            target_id = int(desired["target_id"])
            if branch_key.get(token_id) == target_id:
                continue
            token_mappings[token_id] = target_id
            changes.append({
                "position": desired["position"],
                "cipher_symbol": alpha.symbol_for(token_id),
                "current": desired["current"],
                "target": desired["target"],
                "mapping": f"{alpha.symbol_for(token_id)} -> {desired['target']}",
            })

        preview_key = dict(branch_key)
        preview_key.update(token_mappings)
        before_words = self._decoded_words(branch, max_words=None)
        after_words = self._decoded_words_with_key(branch, preview_key, max_words=None)
        proposed_mappings = {
            alpha.symbol_for(token_id): pt_alpha.symbol_for(pt_id)
            for token_id, pt_id in token_mappings.items()
        }
        base.update({
            "applicable": bool(changes) and not conflicts,
            "changes": changes,
            "conflicts": conflicts,
            "proposed_mappings": proposed_mappings,
            "changed_words_preview": self._changed_words_sample(before_words, after_words),
            "_token_mappings": token_mappings,
        })
        if not changes:
            base["note"] = "Current decoded word already matches target_word."
        elif conflicts:
            base["reason"] = "Repair has conflicting or invalid letter assignments."
        else:
            base["suggested_call"] = (
                "act_apply_word_repair("
                f"branch={branch!r}, cipher_word_index={idx}, "
                f"target_word={target!r})"
            )
        return base

    def _word_repair_effect_summary(
        self,
        branch: str,
        token_mappings: dict[int, int],
    ) -> dict[str, Any]:
        if not token_mappings:
            return {
                "changed_word_count": 0,
                "changed_words_preview": [],
                "dictionary_hit_delta": 0,
            }
        preview_key = dict(self.workspace.get_branch(branch).key)
        preview_key.update(token_mappings)
        before_words = self._decoded_words(branch, max_words=None)
        after_words = self._decoded_words_with_key(branch, preview_key, max_words=None)
        changed_all = [
            {"index": i, "before": b, "after": a}
            for i, (b, a) in enumerate(zip(before_words, after_words))
            if b != a
        ]
        before_hits = sum(
            1 for word in before_words
            if self._reading_char_stream(word) in self.word_set
        )
        after_hits = sum(
            1 for word in after_words
            if self._reading_char_stream(word) in self.word_set
        )
        return {
            "changed_word_count": len(changed_all),
            "changed_words_preview": changed_all[:12],
            "dictionary_hits_before": before_hits,
            "dictionary_hits_after": after_hits,
            "dictionary_hit_delta": after_hits - before_hits,
        }

    def _word_repair_option_recommendation(self, plan: dict[str, Any]) -> str:
        if plan.get("note") and "already matches" in str(plan.get("note")).lower():
            return "already_matches: no mapping change is needed for this option."
        if plan.get("conflicts"):
            return (
                "do_not_apply_directly: this word-level target conflicts with "
                "repeated cipher-symbol constraints. Use boundary repair or a "
                "different reading, or inspect the responsible symbol manually."
            )
        if not plan.get("applicable"):
            return (
                "do_not_apply_directly: this target is not a safe same-length "
                "cipher-symbol repair."
            )
        effect = plan.get("effect_summary") or {}
        changed = int(effect.get("changed_word_count") or 0)
        hit_delta = int(effect.get("dictionary_hit_delta") or 0)
        if changed >= 12 and hit_delta < 0:
            return (
                "review_before_applying: this repair has broad collateral "
                "effects and loses dictionary hits. Read the preview carefully "
                "or fork before applying."
            )
        return "safe_to_try: apply if the changed-word preview reads better."

    def _repair_agenda_status_for_plan(self, plan: dict[str, Any]) -> str:
        if plan.get("error"):
            return "blocked"
        if plan.get("applicable"):
            return "open"
        if plan.get("note"):
            return "applied"
        return "blocked"

    def _find_repair_agenda_item(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        branch = plan.get("branch")
        idx = plan.get("cipher_word_index")
        target = plan.get("target_word")
        for item in self.repair_agenda:
            if (
                item.get("branch") == branch
                and item.get("cipher_word_index") == idx
                and item.get("to") == target
            ):
                return item
        return None

    def _upsert_repair_agenda_item(
        self,
        plan: dict[str, Any],
        *,
        status: str | None = None,
        notes: str | None = None,
        last_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = self._find_repair_agenda_item(plan)
        now_iter = self._current_iteration
        if item is None:
            item = {
                "id": self._next_repair_agenda_id,
                "branch": plan.get("branch"),
                "cipher_word_index": plan.get("cipher_word_index"),
                "cipher_word": plan.get("cipher_word"),
                "from": plan.get("current_decoded"),
                "to": plan.get("target_word"),
                "status": status or self._repair_agenda_status_for_plan(plan),
                "proposed_mappings": dict(plan.get("proposed_mappings") or {}),
                "changed_words_preview": list(plan.get("changed_words_preview") or []),
                "created_iteration": now_iter,
                "updated_iteration": now_iter,
                "notes": notes or "",
            }
            self._next_repair_agenda_id += 1
            self.repair_agenda.append(item)
        else:
            item.update({
                "cipher_word": plan.get("cipher_word", item.get("cipher_word")),
                "from": plan.get("current_decoded", item.get("from")),
                "proposed_mappings": dict(plan.get("proposed_mappings") or item.get("proposed_mappings") or {}),
                "changed_words_preview": list(plan.get("changed_words_preview") or item.get("changed_words_preview") or []),
                "updated_iteration": now_iter,
            })
            if status:
                item["status"] = status
            if notes is not None:
                item["notes"] = notes
        if last_result is not None:
            item["last_result"] = last_result
        return dict(item)

    def _reading_words_from_text(self, text: str) -> list[str]:
        """Return uppercase word-like runs from a proposed reading.

        These are the units the agent wants to install as word boundaries.
        Punctuation is ignored, so a proposal may include light manuscript
        punctuation without changing the character-preservation check.
        """
        normalized = sig.normalize_for_scoring(text)
        return re.findall(r"[A-Z?]+", normalized)

    def _reading_char_stream(self, text: str) -> str:
        return "".join(self._reading_words_from_text(text))

    def _branch_reading_words(self, branch_name: str) -> list[str]:
        return [
            self._reading_char_stream(word)
            for word in self._decoded_words(branch_name, max_words=None)
        ]

    def _dictionary_summary_for_words(self, words: list[str]) -> dict[str, Any]:
        scored = [w for w in words if any(ch.isalpha() for ch in w)]
        hits = [w for w in scored if w in self.word_set]
        unrecognized = [w for w in scored if w not in self.word_set]
        rate = len(hits) / len(scored) if scored else 0.0
        return {
            "dictionary_rate": round(rate, 4),
            "total_words": len(scored),
            "recognized": len(hits),
            "unrecognized_sample": unrecognized[:20],
        }

    def _first_reading_mismatches(
        self,
        current_stream: str,
        proposed_stream: str,
        max_mismatches: int = 8,
    ) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = []
        common_len = min(len(current_stream), len(proposed_stream))
        for idx in range(common_len):
            if current_stream[idx] == proposed_stream[idx]:
                continue
            start = max(0, idx - 8)
            end = min(max(len(current_stream), len(proposed_stream)), idx + 9)
            mismatches.append({
                "char_index": idx,
                "current_char": current_stream[idx],
                "proposed_char": proposed_stream[idx],
                "current_context": current_stream[start:min(end, len(current_stream))],
                "proposed_context": proposed_stream[start:min(end, len(proposed_stream))],
            })
            if len(mismatches) >= max_mismatches:
                break
        if len(mismatches) < max_mismatches and len(current_stream) != len(proposed_stream):
            idx = common_len
            start = max(0, idx - 8)
            mismatches.append({
                "char_index": idx,
                "current_char": current_stream[idx] if idx < len(current_stream) else None,
                "proposed_char": proposed_stream[idx] if idx < len(proposed_stream) else None,
                "current_context": current_stream[start:min(idx + 9, len(current_stream))],
                "proposed_context": proposed_stream[start:min(idx + 9, len(proposed_stream))],
                "note": "Streams differ in length at or after this point.",
            })
        return mismatches

    def _reading_validation(self, branch: str, proposed_text: str) -> dict[str, Any]:
        current_words = self._branch_reading_words(branch)
        proposed_words = self._reading_words_from_text(proposed_text)
        current_stream = "".join(current_words)
        proposed_stream = "".join(proposed_words)
        same = current_stream == proposed_stream
        same_len = len(current_stream) == len(proposed_stream)
        current_summary = self._dictionary_summary_for_words(current_words)
        proposed_summary = self._dictionary_summary_for_words(proposed_words)
        result: dict[str, Any] = {
            "branch": branch,
            "character_preserving": same,
            "same_character_count": same_len,
            "current_char_count": len(current_stream),
            "proposed_char_count": len(proposed_stream),
            "current_word_count": len(current_words),
            "proposed_word_count": len(proposed_words),
            "current_dictionary": current_summary,
            "proposed_dictionary": proposed_summary,
            "proposed_words_preview": proposed_words[:80],
            "mismatches": [] if same else self._first_reading_mismatches(
                current_stream, proposed_stream
            ),
        }
        if same_len and proposed_words:
            projected_words: list[str] = []
            pos = 0
            for word in proposed_words:
                end = pos + len(word)
                projected_words.append(current_stream[pos:end])
                pos = end
            projected_text = " ".join(projected_words)
            word_mismatches = [
                {"index": i, "current_projected": cur, "proposed": prop}
                for i, (cur, prop) in enumerate(zip(projected_words, proposed_words))
                if cur != prop
            ]
            result["boundary_projection"] = {
                "applicable": True,
                "projected_text": projected_text,
                "projected_words_preview": projected_words[:80],
                "projected_dictionary": self._dictionary_summary_for_words(projected_words),
                "word_mismatches_preview": word_mismatches[:20],
                "suggested_call": (
                    f"act_resegment_from_reading_repair(branch='{branch}', "
                    "proposed_text='<same proposed_text>')"
                ),
                "note": (
                    "The proposed reading has the same character count as the "
                    "current branch, so its word boundaries can be applied "
                    "without changing the key or decoded letters. Letter "
                    "differences remain as repair targets."
                ),
            }
        else:
            result["boundary_projection"] = {
                "applicable": False,
                "reason": (
                    "Proposed and current character counts differ; boundary "
                    "projection would drop or duplicate ciphertext tokens."
                ),
            }
        return result

    def _changed_words_sample(
        self,
        before_words: list[str],
        after_words: list[str],
        max_changes: int = 8,
    ) -> list[dict[str, Any]]:
        """Return up to max_changes word-level differences between two decodes.

        The samples are reading-friendly: each entry shows the decoded form
        before and after the change. Lets the agent decide by reading rather
        than by score.
        """
        out: list[dict[str, Any]] = []
        for i, (b, a) in enumerate(zip(before_words, after_words)):
            if b != a:
                out.append({"index": i, "before": b, "after": a})
                if len(out) >= max_changes:
                    break
        return out

    def _reading_score_delta(
        self, before: dict[str, float | None], after: dict[str, float | None]
    ) -> dict[str, Any]:
        """Score delta without verdict/improved fields.

        Used by reading-driven repair primitives (`act_set_mapping`,
        `act_bulk_set`, `act_anchor_word`) so the agent treats the score
        movement as data rather than as an authoritative quality verdict.
        On boundary-preserving ciphers a correct cipher-symbol fix can move
        `dictionary_rate` in the wrong direction; the agent should rely on
        the `changed_words` reading sample, not on a verdict label.
        """
        delta = self._score_delta(before, after)
        return {
            k: v for k, v in delta.items() if k not in ("verdict", "improved")
        }

    def _orthography_risks(
        self,
        before_words: list[str],
        after_words: list[str],
    ) -> list[dict[str, Any]]:
        """Flag broad transcription-style shifts, especially Latin U/V, I/J."""
        if self.language != "la":
            return []
        pairs = {("U", "V"), ("V", "U"), ("I", "J"), ("J", "I")}
        counts: dict[tuple[str, str], int] = {}
        affected: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for idx, (before, after) in enumerate(zip(before_words, after_words)):
            for b_ch, a_ch in zip(before, after):
                pair = (b_ch, a_ch)
                if pair not in pairs:
                    continue
                counts[pair] = counts.get(pair, 0) + 1
                samples = affected.setdefault(pair, [])
                if len(samples) < 6:
                    samples.append({"index": idx, "before": before, "after": after})
        risks: list[dict[str, Any]] = []
        for (src, dst), count in sorted(counts.items(), key=lambda item: -item[1]):
            word_count = len({sample["index"] for sample in affected[(src, dst)]})
            if count < 3 or word_count < 3:
                continue
            risks.append({
                "type": "latin_orthography_shift",
                "from": src,
                "to": dst,
                "changed_char_count": count,
                "affected_word_sample_count": word_count,
                "sample": affected[(src, dst)],
                "note": (
                    f"This repair broadly changes Latin {src}/{dst} "
                    "orthography. Preserve the transcription style unless "
                    "the surrounding decoded text consistently supports the "
                    "shift."
                ),
            })
        return risks

    def _branch_card(self, branch_name: str) -> dict[str, Any]:
        branch = self.workspace.get_branch(branch_name)
        agenda = [
            dict(item) for item in self.repair_agenda
            if item.get("branch") == branch_name
        ]
        risks: list[dict[str, Any]] = []
        for item in agenda:
            last_result = item.get("last_result")
            if isinstance(last_result, dict):
                risks.extend(last_result.get("orthography_risks") or [])
        return {
            "branch": branch_name,
            "mapped_count": len(branch.key),
            "cipher_alphabet_size": self._alpha().size,
            "tags": list(branch.tags),
            "protected_baseline": branch_name == "automated_preflight",
            "scores": self._compute_quick_scores(branch_name),
            "repair_agenda": agenda,
            "repair_counts": {
                status: sum(1 for item in agenda if item.get("status") == status)
                for status in ("open", "applied", "held", "rejected", "blocked", "reverted")
            },
            "orthography_risks": risks,
            "decoded_excerpt": self._decoded_preview(branch_name, max_words=35),
        }

    def _declaration_review(self, branch_name: str) -> dict[str, Any]:
        cards = [self._branch_card(name) for name in self.workspace.branch_names()]
        def score_key(card: dict[str, Any]) -> tuple[float, float, int]:
            scores = card.get("scores") or {}
            dict_rate = scores.get("dict_rate")
            quad = scores.get("quad")
            return (
                dict_rate if isinstance(dict_rate, (int, float)) else float("-inf"),
                quad if isinstance(quad, (int, float)) else float("-inf"),
                int(card.get("mapped_count") or 0),
            )
        best = max(cards, key=score_key) if cards else None
        selected = next((card for card in cards if card.get("branch") == branch_name), None)
        warnings: list[str] = []
        if best and selected and best.get("branch") != branch_name:
            best_scores = best.get("scores") or {}
            selected_scores = selected.get("scores") or {}
            best_dict = best_scores.get("dict_rate")
            selected_dict = selected_scores.get("dict_rate")
            best_quad = best_scores.get("quad")
            selected_quad = selected_scores.get("quad")
            dict_gap = (
                round(float(best_dict) - float(selected_dict), 4)
                if isinstance(best_dict, (int, float)) and isinstance(selected_dict, (int, float))
                else None
            )
            quad_gap = (
                round(float(best_quad) - float(selected_quad), 4)
                if isinstance(best_quad, (int, float)) and isinstance(selected_quad, (int, float))
                else None
            )
            if (dict_gap is not None and dict_gap >= 0.05) or (
                quad_gap is not None and quad_gap >= 0.15
            ):
                warnings.append(
                    f"Branch `{best.get('branch')}` has stronger internal "
                    f"score signals than `{branch_name}` "
                    f"(dict_gap={dict_gap}, quad_gap={quad_gap})."
                )
        if selected and selected.get("orthography_risks"):
            warnings.append(
                "Selected branch has broad orthography-shift warnings; confirm "
                "these are manuscript-faithful rather than modernized spelling."
            )
        open_items = [
            item for item in (selected or {}).get("repair_agenda", [])
            if item.get("status") in {"open", "blocked"}
        ]
        if open_items:
            warnings.append(
                f"Selected branch has {len(open_items)} open/blocked repair "
                "agenda item(s)."
            )
        return {
            "selected_branch": branch_name,
            "best_internal_branch": best.get("branch") if best else None,
            "warnings": warnings,
            "selected_card": selected,
            "best_card": best,
        }

    _READING_DECISION_NOTE = (
        "Score deltas are advisory. Read `changed_words` — if two or more "
        "entries now read as real target-language words (or fragments of "
        "real words), keep the change even when deltas are negative. "
        "Revert only if previously-correct words broke and no real words "
        "were added. Boundary-preserving ciphers in particular can show a "
        "negative `dictionary_rate` delta on a correct cipher-symbol fix."
    )

    _BOUNDARY_DECLARATION_TERMS = (
        "boundary", "word break", "word-break", "word boundary",
        "split", "merge", "merged", "misaligned", "alignment",
    )

    def _branch_used_full_reading_workflow(self, branch: str) -> bool:
        workflow_tools = {
            "act_resegment_by_reading",
            "act_resegment_from_reading_repair",
        }
        for call in self.call_log:
            if call.tool_name not in workflow_tools:
                continue
            args = call.arguments or {}
            if args.get("branch") == branch:
                return True
        return False

    def _should_guard_declaration_for_reading_workflow(
        self,
        branch: str,
        rationale: str,
    ) -> bool:
        if self.max_iterations is not None and self._current_iteration >= self.max_iterations:
            return False
        if self._branch_used_full_reading_workflow(branch):
            return False
        lowered = rationale.lower()
        return any(term in lowered for term in self._BOUNDARY_DECLARATION_TERMS)

    _TRANSFORM_DECLARATION_TERMS = (
        "transform", "transposition", "transpose", "columnar", "vigenere",
        "polyalpha", "polyalphabetic", "period", "key-period", "key period",
        "ic=", "below english", "below english level",
    )

    def _has_seen_any_tool(self, tool_names: set[str]) -> bool:
        return any(call.tool_name in tool_names for call in self.call_log)

    def _declaration_has_untried_transform_work(self, rationale: str, note: str) -> bool:
        text = f"{rationale}\n{note}".lower()
        mentions_transform_family = any(
            term in text for term in self._TRANSFORM_DECLARATION_TERMS
        )
        if not mentions_transform_family:
            return False
        return not self._has_seen_any_tool({
            "observe_transform_pipeline",
            "observe_transform_suspicion",
            "search_transform_candidates",
            "act_apply_transform_pipeline",
            "search_transform_homophonic",
        })

    def _should_guard_low_confidence_declaration(
        self,
        confidence: float,
        further_iterations_helpful: bool,
        forced_partial: bool,
    ) -> bool:
        if forced_partial:
            return False
        if self.max_iterations is not None and self._current_iteration >= self.max_iterations:
            return False
        return confidence < 0.50 and further_iterations_helpful

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

    def _tool_workspace_fork_best(self, args: dict) -> Any:
        new_name = args["new_name"]
        prefer_branch = str(args.get("prefer_branch") or "").strip()
        source = self._select_best_fork_source(prefer_branch or None)
        branch = self.workspace.fork(new_name, from_branch=source)
        card = self._branch_card(source)
        note = (
            "Forked from `automated_preflight`, preserving the no-LLM baseline "
            "as an unchanged comparison branch."
            if source == "automated_preflight"
            else "Forked from the strongest existing branch by quick scores and mapped count."
        )
        return {
            "status": "ok",
            "created": new_name,
            "parent": source,
            "source_branch": source,
            "inherited_mapped_count": len(branch.key),
            "source_scores": card.get("scores"),
            "source_tags": card.get("tags"),
            "note": note,
        }

    def _select_best_fork_source(self, prefer_branch: str | None = None) -> str:
        if prefer_branch and self.workspace.has_branch(prefer_branch):
            return prefer_branch

        def branch_key(name: str) -> tuple[float, float, int, int]:
            branch = self.workspace.get_branch(name)
            scores = self._compute_quick_scores(name)
            dict_rate = scores.get("dict_rate")
            quad = scores.get("quad")
            return (
                dict_rate if isinstance(dict_rate, (int, float)) else float("-inf"),
                quad if isinstance(quad, (int, float)) else float("-inf"),
                len(branch.key),
                1 if name == "automated_preflight" else 0,
            )

        names = self.workspace.branch_names()
        if not names:
            return "main"
        return max(names, key=branch_key)

    def _tool_workspace_list_branches(self, _args: dict) -> Any:
        return {"branches": self.workspace.list_branches()}

    def _tool_workspace_branch_cards(self, args: dict) -> Any:
        branch = args.get("branch")
        if branch:
            if not self.workspace.has_branch(branch):
                return {"error": f"Branch not found: {branch}"}
            cards = [self._branch_card(branch)]
        else:
            cards = [self._branch_card(name) for name in self.workspace.branch_names()]
        return {
            "status": "ok",
            "cards": cards,
            "note": (
                "Use these cards before declaration: compare readability, "
                "internal scores, applied/held repairs, and orthography risks. "
                "Treat `automated_preflight` as a protected no-LLM baseline; "
                "do not discard it in favor of a modernized/classicized edit "
                "unless the edited branch is clearly better in the manuscript "
                "transcription style."
            ),
        }

    def _has_seen_branch_cards(self) -> bool:
        return any(call.tool_name == "workspace_branch_cards" for call in self.call_log)

    def _unresolved_repair_agenda_items(self, branch: str) -> list[dict[str, Any]]:
        return [
            dict(item) for item in self.repair_agenda
            if item.get("branch") == branch and item.get("status") in {"open", "blocked"}
        ]

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

    def _tool_observe_transform_pipeline(self, args: dict) -> Any:
        branch_name = args.get("branch") or "main"
        branch = self.workspace.get_branch(branch_name)
        effective_tokens = self.workspace.effective_tokens(branch_name)
        preview_symbols = self.workspace.cipher_text.alphabet.decode(effective_tokens[:80])
        return {
            "branch": branch_name,
            "token_count": len(self.workspace.cipher_text.tokens),
            "has_transform": branch.token_order is not None,
            "transform_pipeline": branch.transform_pipeline,
            "active_order_preview": branch.token_order[:40] if branch.token_order is not None else None,
            "transformed_cipher_preview": preview_symbols,
            "note": (
                "For transposition+homophonic ciphers, apply or search a "
                "ciphertext transform before judging the homophonic key."
            ),
        }

    def _tool_observe_transform_suspicion(self, args: dict) -> Any:
        branch_name = args.get("branch") or "main"
        tokens = self.workspace.effective_tokens(branch_name)
        columns = args.get("columns")
        columns = int(columns) if columns is not None else None
        report = inspect_transform_suspicion(
            token_count=len(tokens),
            cipher_alphabet_size=self._alpha().size,
            plaintext_alphabet_size=self._pt_alpha().size,
            word_group_count=len(self.workspace.cipher_text.words),
            cipher_system=getattr(self.workspace.cipher_text, "cipher_system", "") or "",
            columns=columns,
            baseline_status=args.get("baseline_status"),
            baseline_score=args.get("baseline_score"),
        )
        screen_profile = "medium" if report["recommendation"] == "run_screen" else "small"
        screen = screen_transform_candidates(
            tokens,
            columns=columns,
            profile=screen_profile,
            top_n=5,
            include_mutations=screen_profile != "small",
            include_program_search=bool(args.get("include_program_search", False)),
        )
        return {
            "branch": branch_name,
            **report,
            "candidate_screen": screen,
            "recommended_next_tool": (
                "search_transform_homophonic"
                if report["recommendation"] in {"run_screen", "consider_screen"}
                else "continue_baseline_or_mapping_diagnostics"
            ),
            "note": (
                "This tool is meant to help the agent decide whether the "
                "ciphertext order itself is suspicious. Escalate only if the "
                "diagnostic reasons and candidate menu look coherent."
            ),
        }

    def _tool_search_transform_candidates(self, args: dict) -> Any:
        branch_name = args.get("branch") or "main"
        tokens = self.workspace.effective_tokens(branch_name)
        columns = args.get("columns")
        columns = int(columns) if columns is not None else None
        breadth = str(args.get("breadth") or "broad").strip().lower()
        if breadth not in {"fast", "broad", "wide"}:
            return {"error": "breadth must be one of: fast, broad, wide"}
        profile = "small" if breadth == "fast" else "medium" if breadth == "broad" else "wide"
        top_n = max(1, min(int(args.get("top_n", 40)), 1000))
        max_generated = args.get("max_generated_candidates")
        max_generated_candidates = int(max_generated) if max_generated is not None else (
            5000 if breadth == "fast" else 10000 if breadth == "broad" else 25000
        )
        include_program_search = bool(args.get("include_program_search", breadth == "wide"))
        screen = screen_transform_candidates(
            tokens,
            columns=columns,
            profile=profile,
            top_n=top_n,
            max_generated_candidates=max_generated_candidates,
            streaming=breadth == "wide",
            include_mutations=False,
            include_program_search=include_program_search,
            program_max_depth=int(args.get("program_max_depth", 5)),
            program_beam_width=int(args.get("program_beam_width", 48 if breadth == "wide" else 24)),
        )
        return {
            "branch": branch_name,
            "breadth": breadth,
            "profile": profile,
            "columns": columns,
            "candidate_count": screen.get("candidate_count", 0),
            "deduped_candidate_count": screen.get("deduped_candidate_count", 0),
            "generation_limit_reached": screen.get("generation_limit_reached", False),
            "family_counts": screen.get("family_counts", {}),
            "top_family_counts": screen.get("top_family_counts", {}),
            "anchor_candidates": screen.get("anchor_candidates", [])[:40],
            "top_candidates": screen.get("top_candidates", [])[:top_n],
            "program_search": screen.get("program_search"),
            "structural_screen": screen,
            "recommended_next_tool": "search_transform_homophonic",
            "note": (
                "This is structural-only. It performs no language-model "
                "annealing and should be used to decide which small set of "
                "candidates deserves solver-backed promotion."
            ),
        }

    # ------------------------------------------------------------------
    # decode_*
    # ------------------------------------------------------------------
    def _tool_decode_show(self, args: dict) -> Any:
        branch = args["branch"]
        start = args.get("start_word", 0)
        count = min(args.get("count", 25), 50)
        words = self.workspace.effective_words(branch)
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

        boundary_candidates = self._boundary_edit_suggestions(branch)

        return {
            "seg": seg,
            "candidates": candidates,
            "boundary_candidates": boundary_candidates,
            "any_ambiguous": any_ambiguous,
        }

    def _boundary_edit_suggestions(self, branch: str, max_candidates: int = 5) -> list[dict[str, Any]]:
        words = self.workspace.effective_words(branch)
        if not words:
            return []

        decoded_words = [
            sig.normalize_for_scoring(self._decode_word(word, branch)).replace(" ", "")
            for word in words
        ]
        candidates: list[dict[str, Any]] = []

        # Missing boundary inside a word: ABCDEF -> ABC | DEF
        for idx, (word_tokens, decoded) in enumerate(zip(words, decoded_words)):
            if len(word_tokens) < 2 or len(decoded) != len(word_tokens):
                continue
            if decoded in self.word_set:
                continue
            for split_at in range(1, len(decoded)):
                left = decoded[:split_at]
                right = decoded[split_at:]
                if left in self.word_set and right in self.word_set:
                    candidates.append({
                        "type": "split",
                        "cipher_word_index": idx,
                        "decoded_before": decoded,
                        "decoded_after": f"{left} | {right}",
                        "split_at_token_offset": split_at,
                        "suggested_call": (
                            f"act_split_cipher_word(branch='{branch}', "
                            f"cipher_word_index={idx}, split_at_token_offset={split_at})"
                        ),
                        "evidence": "both split parts are dictionary words",
                        "score_hint": len(left) + len(right),
                    })

        # Spurious boundary between adjacent words: ABC | DEF -> ABCDEF
        for idx in range(len(decoded_words) - 1):
            left = decoded_words[idx]
            right = decoded_words[idx + 1]
            merged = left + right
            if not left or not right or merged not in self.word_set:
                continue
            both_parts_known = left in self.word_set and right in self.word_set
            candidates.append({
                "type": "merge",
                "left_word_index": idx,
                "decoded_before": f"{left} | {right}",
                "decoded_after": merged,
                "suggested_call": (
                    f"act_merge_cipher_words(branch='{branch}', left_word_index={idx})"
                ),
                "evidence": (
                    "merged form is a dictionary word; both split parts are "
                    "also dictionary words, so apply only if the surrounding "
                    "reading wants the compound"
                    if both_parts_known
                    else "merged form is a dictionary word"
                ),
                "score_hint": len(merged) - (1 if both_parts_known else 0),
            })

        candidates.sort(key=lambda c: (c["score_hint"], c["type"] == "merge"), reverse=True)
        trimmed = candidates[:max_candidates]
        for cand in trimmed:
            cand.pop("score_hint", None)
        return trimmed

    def _recommended_boundary_tool(
        self,
        boundary_candidates: list[dict[str, Any]],
        letter_candidates: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Recommend a boundary edit only when nothing better is available.

        Boundary edits typically yield small `dictionary_rate` gains. When
        letter-level corrections are also visible (`candidate_corrections`
        from the same diagnostic), those are far higher-leverage and should
        be tried first — see "Reading-driven repair" in the system prompt.
        """
        if not boundary_candidates:
            return None
        if letter_candidates:
            # Letter-level fixes dominate; do not promote a boundary edit as
            # the next move when the diagnostic also surfaced letter-level
            # culprits.
            return None
        return "act_apply_boundary_candidate(branch='...', candidate_index=0)"

    def _tool_decode_diagnose(self, args: dict) -> Any:
        branch = args["branch"]
        top_k = int(args.get("top_k", 5))
        diag = self._diagnose_branch(branch, top_k)
        if "error" in diag:
            return {"branch": branch, "error": diag["error"]}

        seg = diag["seg"]
        candidates = diag["candidates"]
        boundary_candidates = diag["boundary_candidates"]
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
            if boundary_candidates:
                note += (
                    " Boundary suggestions are also available below; use "
                    "split/merge edits when the text looks shifted by a bad "
                    "word break rather than by a wrong letter."
                )
            # Bulk fix suggestion — apply all candidates in one tool call.
            bulk_fix_call = (
                f"decode_diagnose_and_fix(branch='{branch}', "
                f"top_k={len(candidates)})"
            )
        else:
            if boundary_candidates:
                note = (
                    "No strong letter-substitution corrections found, but there "
                    "are plausible word-boundary edits below. Try split/merge "
                    "before re-annealing."
                )
            elif self._is_homophonic_cipher() and self._is_no_boundary_cipher(branch):
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
            "boundary_candidates": boundary_candidates,
            "recommended_next_tool": self._recommended_boundary_tool(
                boundary_candidates, letter_candidates=candidates,
            ),
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
        boundary_candidates = diag.get("boundary_candidates", [])
        seg = diag["seg"]

        to_apply = [c for c in candidates if c["evidence_count"] >= min_evidence and c["culprit_symbol"]]
        skipped = [c for c in candidates if c["evidence_count"] < min_evidence or not c["culprit_symbol"]]

        if not to_apply:
            boundary_note = ""
            if boundary_candidates and not skipped:
                boundary_note = (
                    " Boundary edits look more promising than letter swaps here; "
                    "try act_split_cipher_word / act_merge_cipher_words before "
                    "more annealing."
                )
            return {
                "branch": branch,
                "fixes_applied": [],
                "fixes_skipped": [
                    {"wrong": c["wrong"], "correct": c["correct"],
                     "evidence_count": c["evidence_count"]}
                    for c in skipped
                ],
                "boundary_candidates": boundary_candidates,
                "recommended_next_tool": self._recommended_boundary_tool(
                    boundary_candidates, letter_candidates=skipped,
                ),
                "score_delta": None,
                "note": (
                    f"No candidates met min_evidence={min_evidence}. "
                    f"Lower min_evidence or inspect with decode_diagnose.{boundary_note}"
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
        remaining_boundary_candidates = remaining_diag.get("boundary_candidates", [])

        remaining_pseudo = len(remaining_seg.pseudo_words) if remaining_seg else None
        dict_rate_after = round(remaining_seg.dict_rate, 4) if remaining_seg else None

        if dict_rate_after is not None and dict_rate_after >= 0.85:
            if self._is_homophonic_cipher() and self._is_no_boundary_cipher(branch):
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
                "Call decode_diagnose_and_fix again, apply targeted "
                "act_set_mapping fixes from your reading, or declare if "
                "the text reads well."
            )
        elif remaining_boundary_candidates:
            recommendation = (
                "No remaining letter-level corrections, but boundary edits "
                "look plausible. Try act_split_cipher_word or "
                "act_merge_cipher_words before declaration if your reading "
                "suggests the cipher's word breaks are misaligned with "
                "target-language word breaks."
            )
        else:
            if self._is_homophonic_cipher() and self._is_no_boundary_cipher(branch):
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
            "boundary_candidates": remaining_boundary_candidates,
            "recommended_next_tool": self._recommended_boundary_tool(
                remaining_boundary_candidates,
                letter_candidates=remaining_candidates,
            ),
            "decoded_preview": self._decoded_preview(branch),
            "warnings": self._homophonic_reliability_warnings(
                branch, dict_rate_after, after.get("quad")
            ),
            "recommendation": recommendation,
        }

    def _tool_decode_repair_no_boundary(self, args: dict) -> Any:
        branch = args["branch"]
        max_rounds = int(args.get("max_rounds", 3))
        max_window_words = int(args.get("max_window_words", 3))
        decrypted = self.workspace.apply_key(branch)
        normalized = sig.normalize_for_scoring(decrypted)
        if not normalized.strip():
            return {"branch": branch, "error": "empty decoded text"}

        repair = repair_no_boundary_text(
            normalized,
            self.word_set,
            self._freq_rank,
            max_rounds=max_rounds,
            max_window_words=max_window_words,
        )
        return {
            "branch": branch,
            "applied": repair.applied,
            "reason": repair.reason,
            "rounds": repair.rounds,
            "before": {
                "dict_rate": round(repair.before.dict_rate, 4),
                "segmentation_cost": round(repair.before.cost, 3),
                "pseudo_word_count": len(repair.before.pseudo_words),
                "segmented_text": repair.before.segmented[:600],
            },
            "after": {
                "dict_rate": round(repair.after.dict_rate, 4),
                "segmentation_cost": round(repair.after.cost, 3),
                "pseudo_word_count": len(repair.after.pseudo_words),
                "segmented_text": repair.after.segmented[:600],
            },
            "corrections": repair.corrections[:40],
            "repaired_text": repair.repaired_text[:1200],
            "note": (
                "This is a text-only repair preview; it does not mutate the "
                "branch key. Use it to judge whether a no-boundary branch is "
                "close but drifted, especially after search_automated_solver or "
                "search_homophonic_anneal."
            ),
        }

    def _tool_decode_validate_reading_repair(self, args: dict) -> Any:
        branch = args["branch"]
        proposed_text = str(args["proposed_text"])
        validation = self._reading_validation(branch, proposed_text)
        if validation["character_preserving"]:
            validation["recommendation"] = (
                "This proposal only changes word boundaries. Apply it with "
                "act_resegment_by_reading using the same proposed_text."
            )
        elif validation.get("boundary_projection", {}).get("applicable"):
            validation["recommendation"] = (
                "This proposal changes letters but has the same character "
                "count. Apply only its word-boundary pattern with "
                "act_resegment_from_reading_repair using the same proposed_text; "
                "then use the mismatch spans to make targeted "
                "act_set_mapping/act_bulk_set repairs."
            )
        else:
            validation["recommendation"] = (
                "This proposal changes the decoded character stream. Treat it "
                "as a reading hypothesis: inspect the mismatch spans, identify "
                "the responsible cipher symbols with decode_show, then apply "
                "targeted act_set_mapping/act_bulk_set repairs before changing "
                "boundaries."
            )
        return validation

    def _tool_decode_plan_word_repair(self, args: dict) -> Any:
        plan = self._word_repair_plan(args)
        if not plan.get("error"):
            status = self._repair_agenda_status_for_plan(plan)
            plan["agenda_item"] = self._upsert_repair_agenda_item(
                plan,
                status=status,
                notes="Planned by decode_plan_word_repair.",
            )
        plan.pop("_token_mappings", None)
        return plan

    def _tool_decode_plan_word_repair_menu(self, args: dict) -> Any:
        targets = args.get("target_words") or []
        if not isinstance(targets, list) or not targets:
            return {"error": "target_words must be a non-empty list"}

        normalized_targets: list[str] = []
        seen: set[str] = set()
        for raw_target in targets:
            target = self._reading_char_stream(str(raw_target))
            if not target or target in seen:
                continue
            seen.add(target)
            normalized_targets.append(target)
        if not normalized_targets:
            return {"error": "target_words did not contain any alphabetic candidates"}

        options: list[dict[str, Any]] = []
        current_decoded: str | None = None
        cipher_word_index: int | None = None
        for target in normalized_targets[:12]:
            plan_args = {
                key: value for key, value in args.items()
                if key != "target_words"
            }
            plan_args["target_word"] = target
            plan = self._word_repair_plan(plan_args)
            token_mappings = plan.pop("_token_mappings", {})
            if not plan.get("error"):
                current_decoded = plan.get("current_decoded", current_decoded)
                cipher_word_index = plan.get("cipher_word_index", cipher_word_index)
            if token_mappings:
                plan["effect_summary"] = self._word_repair_effect_summary(
                    args["branch"],
                    token_mappings,
                )
            else:
                plan["effect_summary"] = {
                    "changed_word_count": 0,
                    "changed_words_preview": list(plan.get("changed_words_preview") or []),
                    "dictionary_hit_delta": 0,
                }
            plan["recommendation"] = self._word_repair_option_recommendation(plan)
            if plan.get("applicable"):
                plan["suggested_call"] = (
                    "act_apply_word_repair("
                    f"branch={args['branch']!r}, "
                    f"cipher_word_index={plan['cipher_word_index']}, "
                    f"target_word={plan['target_word']!r})"
                )
            options.append(plan)

        return {
            "status": "ok",
            "branch": args["branch"],
            "cipher_word_index": cipher_word_index,
            "current_decoded": current_decoded,
            "option_count": len(options),
            "options": options,
            "note": (
                "This is a read-only menu. Apply only options whose preview "
                "reads better and whose recommendation does not say "
                "do_not_apply_directly. For conflicting repeated-symbol cases, "
                "prefer boundary repair or a different reading rather than "
                "forcing a single mapping."
            ),
        }

    def _tool_repair_agenda_list(self, args: dict) -> Any:
        branch_filter = args.get("branch")
        status_filter = args.get("status")
        items = []
        for item in self.repair_agenda:
            if branch_filter and item.get("branch") != branch_filter:
                continue
            if status_filter and item.get("status") != status_filter:
                continue
            items.append(dict(item))
        unresolved = [
            item for item in items
            if item.get("status") in {"open", "blocked"}
        ]
        return {
            "status": "ok",
            "count": len(items),
            "unresolved_count": len(unresolved),
            "items": items,
            "note": (
                "Resolve open/blocked reading repairs before declaring. Held "
                "items are treated as explicitly resolved but preserved for "
                "the final rationale."
            ),
        }

    def _tool_repair_agenda_update(self, args: dict) -> Any:
        item_id = int(args["item_id"])
        status = str(args["status"])
        notes = str(args.get("notes", ""))
        for item in self.repair_agenda:
            if int(item.get("id", -1)) != item_id:
                continue
            item["status"] = status
            item["updated_iteration"] = self._current_iteration
            if notes:
                item["notes"] = notes
            return {"status": "ok", "agenda_item": dict(item)}
        return {"error": f"repair agenda item {item_id} not found"}

    # ------------------------------------------------------------------
    # score_*
    # ------------------------------------------------------------------
    def _tool_score_panel(self, args: dict) -> Any:
        branch = args["branch"]
        ws = self.workspace
        decrypted = ws.apply_key(branch)
        panel = sig.compute_panel(
            decrypted=decrypted,
            cipher_words=ws.effective_words(branch),
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
        word_branch = consistent_with or "main"
        words = self.workspace.effective_words(word_branch)
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
        before_words = self._decoded_words(branch)
        self.workspace.set_mapping(
            branch,
            alpha.id_for(cipher_sym),
            pt_alpha.id_for(plain_letter),
        )
        after = self._compute_quick_scores(branch)
        after_words = self._decoded_words(branch)
        changed = self._changed_words_sample(before_words, after_words)
        orthography_risks = self._orthography_risks(before_words, after_words)
        return {
            "status": "ok",
            "branch": branch,
            "mapping": f"{cipher_sym} -> {plain_letter}",
            "occurrences_changed": sum(
                1 for w in self.workspace.cipher_text.tokens
                if alpha.symbol_for(w) == cipher_sym
            ),
            "changed_words": changed,
            "orthography_risks": orthography_risks,
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._reading_score_delta(before, after),
            "note": self._READING_DECISION_NOTE,
        }

    def _tool_act_bulk_set(self, args: dict) -> Any:
        branch = args["branch"]
        mappings = args.get("mappings", {})
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        set_ok: list[str] = []
        errors: list[str] = []
        before = self._compute_quick_scores(branch)
        before_words = self._decoded_words(branch)
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
        after_words = self._decoded_words(branch)
        changed = self._changed_words_sample(before_words, after_words)
        orthography_risks = self._orthography_risks(before_words, after_words)
        return {
            "status": "ok",
            "branch": branch,
            "mappings_set": len(set_ok),
            "details": set_ok,
            "errors": errors if errors else None,
            "changed_words": changed,
            "orthography_risks": orthography_risks,
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._reading_score_delta(before, after),
            "note": self._READING_DECISION_NOTE,
        }

    def _tool_act_anchor_word(self, args: dict) -> Any:
        branch_name = args["branch"]
        idx = args["cipher_word_index"]
        target = args["plaintext"].upper()
        ws = self.workspace
        words = ws.effective_words(branch_name)
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
        before_words = self._decoded_words(branch_name)
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
        after_words = self._decoded_words(branch_name)
        changed = self._changed_words_sample(before_words, after_words)
        orthography_risks = self._orthography_risks(before_words, after_words)
        return {
            "status": "ok",
            "branch": branch_name,
            "cipher_word": self._cipher_word_str(word_tokens),
            "now_decodes_as": decoded,
            "changes": changes,
            "changed_words": changed,
            "orthography_risks": orthography_risks,
            "decoded_preview": self._decoded_preview(branch_name),
            "score_delta": self._reading_score_delta(before, after),
            "note": self._READING_DECISION_NOTE,
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

    def _tool_act_split_cipher_word(self, args: dict) -> Any:
        branch = args["branch"]
        idx = int(args["cipher_word_index"])
        split_at = int(args["split_at_token_offset"])
        before = self._compute_quick_scores(branch)
        try:
            info = self.workspace.split_cipher_word(branch, idx, split_at)
        except WorkspaceError as exc:
            return {"error": str(exc)}
        after = self._compute_quick_scores(branch)
        words = self.workspace.effective_words(branch)
        left = self._cipher_word_str(words[idx])
        right = self._cipher_word_str(words[idx + 1])
        return {
            "status": "ok",
            "branch": branch,
            "split_word_index": idx,
            "split_at_token_offset": split_at,
            "left_cipher_word": left,
            "right_cipher_word": right,
            "total_words": len(words),
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._score_delta(before, after),
        }

    def _tool_act_merge_cipher_words(self, args: dict) -> Any:
        branch = args["branch"]
        left_idx = int(args["left_word_index"])
        before = self._compute_quick_scores(branch)
        try:
            info = self.workspace.merge_cipher_words(branch, left_idx)
        except WorkspaceError as exc:
            return {"error": str(exc)}
        after = self._compute_quick_scores(branch)
        words = self.workspace.effective_words(branch)
        merged = self._cipher_word_str(words[left_idx])
        return {
            "status": "ok",
            "branch": branch,
            "merged_left_word_index": left_idx,
            "merged_cipher_word": merged,
            "total_words": len(words),
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": self._score_delta(before, after),
        }

    def _tool_act_merge_decoded_words(self, args: dict) -> Any:
        branch = args["branch"]
        left_target = sig.normalize_for_scoring(str(args["left_decoded"]))
        right_target = sig.normalize_for_scoring(str(args["right_decoded"]))
        occurrence = int(args.get("occurrence", 0))
        if not left_target or not right_target:
            return {"error": "left_decoded and right_decoded must be non-empty"}
        words = self.workspace.effective_words(branch)
        decoded_words = [
            sig.normalize_for_scoring(self._decode_word(word, branch)).replace(" ", "")
            for word in words
        ]
        matches = [
            idx for idx in range(len(decoded_words) - 1)
            if decoded_words[idx] == left_target and decoded_words[idx + 1] == right_target
        ]
        if not matches:
            return {
                "error": (
                    f"No adjacent decoded pair {left_target} | {right_target} "
                    f"found on branch {branch}"
                ),
                "branch": branch,
                "current_decoded_words": decoded_words[:80],
                "note": (
                    "Word indices and decoded pairs change after each boundary "
                    "edit. Re-read the branch with decode_show if the pair is "
                    "not found."
                ),
            }
        if occurrence < 0 or occurrence >= len(matches):
            return {
                "error": f"occurrence {occurrence} out of range (0..{len(matches) - 1})",
                "matches": matches,
            }
        left_idx = matches[occurrence]
        result = self._tool_act_merge_cipher_words({
            "branch": branch,
            "left_word_index": left_idx,
        })
        return result | {
            "matched_decoded_before": f"{left_target} | {right_target}",
            "matched_left_word_index": left_idx,
            "occurrence": occurrence,
        }

    def _tool_act_apply_boundary_candidate(self, args: dict) -> Any:
        branch = args["branch"]
        idx = int(args.get("candidate_index", 0))
        candidates = self._boundary_edit_suggestions(branch)
        if not candidates:
            return {"error": "No boundary_candidates available for this branch"}
        if idx < 0 or idx >= len(candidates):
            return {"error": f"candidate_index {idx} out of range (0..{len(candidates) - 1})"}
        candidate = candidates[idx]
        if candidate["type"] == "split":
            return self._tool_act_split_cipher_word({
                "branch": branch,
                "cipher_word_index": candidate["cipher_word_index"],
                "split_at_token_offset": candidate["split_at_token_offset"],
            }) | {
                "applied_candidate": candidate,
                "candidate_index": idx,
            }
        if candidate["type"] == "merge":
            return self._tool_act_merge_cipher_words({
                "branch": branch,
                "left_word_index": candidate["left_word_index"],
            }) | {
                "applied_candidate": candidate,
                "candidate_index": idx,
            }
        return {"error": f"Unknown boundary candidate type: {candidate['type']}"}

    def _tool_act_apply_word_repair(self, args: dict) -> Any:
        branch = args["branch"]
        plan = self._word_repair_plan(args)
        token_mappings = plan.pop("_token_mappings", {})
        if plan.get("error"):
            return plan
        if not plan.get("applicable"):
            plan["agenda_item"] = self._upsert_repair_agenda_item(
                plan,
                status="blocked",
                notes=str(plan.get("reason") or "Word repair is not directly applicable."),
            )
            return {
                "error": "Word repair is not directly applicable.",
                **plan,
            }

        before = self._compute_quick_scores(branch)
        before_words = self._decoded_words(branch, max_words=None)
        for token_id, pt_id in token_mappings.items():
            self.workspace.set_mapping(branch, token_id, pt_id)
        after = self._compute_quick_scores(branch)
        after_words = self._decoded_words(branch, max_words=None)
        changed_words = self._changed_words_sample(before_words, after_words)
        score_delta = self._reading_score_delta(before, after)
        orthography_risks = self._orthography_risks(before_words, after_words)
        agenda_item = self._upsert_repair_agenda_item(
            plan,
            status="applied",
            notes="Applied by act_apply_word_repair.",
            last_result={
                "mappings_set": len(token_mappings),
                "changed_words": changed_words,
                "score_delta": score_delta,
                "orthography_risks": orthography_risks,
            },
        )
        return {
            "status": "ok",
            "branch": branch,
            "cipher_word_index": plan["cipher_word_index"],
            "cipher_word": plan["cipher_word"],
            "from": plan["current_decoded"],
            "to": plan["target_word"],
            "mappings_set": len(token_mappings),
            "mappings": plan["proposed_mappings"],
            "changed_words": changed_words,
            "orthography_risks": orthography_risks,
            "decoded_preview": self._decoded_preview(branch),
            "score_delta": score_delta,
            "agenda_item": agenda_item,
            "note": self._READING_DECISION_NOTE,
        }

    def _tool_act_resegment_by_reading(self, args: dict) -> Any:
        branch = args["branch"]
        proposed_text = str(args["proposed_text"])
        if self._pt_alpha()._multisym:
            return {
                "error": (
                    "act_resegment_by_reading currently supports single-character "
                    "plaintext alphabets only"
                )
            }

        validation = self._reading_validation(branch, proposed_text)
        if not validation["character_preserving"]:
            projection = validation.get("boundary_projection", {})
            next_step = (
                "If the letter changes are intentional, first use "
                "decode_validate_reading_repair to inspect them, then apply "
                "act_set_mapping/act_bulk_set repairs. If this was meant to "
                "be boundary-only, revise proposed_text so the letters match."
            )
            if projection.get("applicable"):
                next_step = (
                    "This proposal changes letters but has the same character "
                    "count. To apply only its word-boundary pattern while "
                    "preserving the current decoded letters, call "
                    "act_resegment_from_reading_repair with the same "
                    "proposed_text. Then repair the mismatch spans with "
                    "act_set_mapping/act_bulk_set."
                )
            return {
                "error": (
                    "Proposed reading is not character-preserving; no boundaries "
                    "were changed."
                ),
                **validation,
                "next_step": next_step,
            }

        proposed_words = self._reading_words_from_text(proposed_text)
        token_count = len(self.workspace.cipher_text.tokens)
        if sum(len(word) for word in proposed_words) != token_count:
            return {
                "error": (
                    "Proposed words preserve the decoded character stream but "
                    "do not match the ciphertext token count. This can happen "
                    "with multi-character plaintext symbols."
                ),
                **validation,
            }

        before = self._compute_quick_scores(branch)
        before_preview = self._decoded_preview(branch, max_words=80)
        spans: list[tuple[int, int]] = []
        start = 0
        for word in proposed_words:
            end = start + len(word)
            spans.append((start, end))
            start = end
        try:
            self.workspace.set_word_spans(branch, spans)
        except WorkspaceError as exc:
            return {"error": str(exc), **validation}
        after = self._compute_quick_scores(branch)
        after_preview = self._decoded_preview(branch, max_words=80)
        return {
            "status": "ok",
            "branch": branch,
            "applied": True,
            "old_word_count": validation["current_word_count"],
            "new_word_count": validation["proposed_word_count"],
            "dictionary_before": validation["current_dictionary"],
            "dictionary_after": validation["proposed_dictionary"],
            "score_delta": self._reading_score_delta(before, after),
            "decoded_preview_before": before_preview,
            "decoded_preview": after_preview,
            "note": (
                "Applied word-boundary overlay only. The branch key and decoded "
                "characters were unchanged; only the reading's word segmentation "
                "changed."
            ),
        }

    def _tool_act_resegment_from_reading_repair(self, args: dict) -> Any:
        branch = args["branch"]
        proposed_text = str(args["proposed_text"])
        if self._pt_alpha()._multisym:
            return {
                "error": (
                    "act_resegment_from_reading_repair currently supports "
                    "single-character plaintext alphabets only"
                )
            }

        validation = self._reading_validation(branch, proposed_text)
        projection = validation.get("boundary_projection", {})
        if not projection.get("applicable"):
            return {
                "error": (
                    "Cannot project proposed word boundaries because the "
                    "proposed reading does not have the same normalized "
                    "character count as the current branch."
                ),
                **validation,
            }

        proposed_words = self._reading_words_from_text(proposed_text)
        token_count = len(self.workspace.cipher_text.tokens)
        if sum(len(word) for word in proposed_words) != token_count:
            return {
                "error": (
                    "Projected word lengths do not match the ciphertext token "
                    "count. No boundaries were changed."
                ),
                **validation,
            }

        before = self._compute_quick_scores(branch)
        before_preview = self._decoded_preview(branch, max_words=80)
        spans: list[tuple[int, int]] = []
        start = 0
        for word in proposed_words:
            end = start + len(word)
            spans.append((start, end))
            start = end
        try:
            self.workspace.set_word_spans(branch, spans)
        except WorkspaceError as exc:
            return {"error": str(exc), **validation}
        after = self._compute_quick_scores(branch)
        after_preview = self._decoded_preview(branch, max_words=80)
        projected = projection.get("projected_text", "")
        return {
            "status": "ok",
            "branch": branch,
            "applied": True,
            "mode": "boundary_projection_from_repair_reading",
            "character_preserving": validation["character_preserving"],
            "old_word_count": validation["current_word_count"],
            "new_word_count": validation["proposed_word_count"],
            "dictionary_before": validation["current_dictionary"],
            "projected_dictionary_after": projection.get("projected_dictionary"),
            "proposed_dictionary": validation["proposed_dictionary"],
            "mismatches": validation["mismatches"],
            "word_mismatches_preview": projection.get("word_mismatches_preview", []),
            "projected_text": projected[:1200],
            "score_delta": self._reading_score_delta(before, after),
            "decoded_preview_before": before_preview,
            "decoded_preview": after_preview,
            "note": (
                "Applied the proposed reading's word boundaries only. The key "
                "and decoded character stream were preserved; any mismatches "
                "between projected_text and proposed_text are still true "
                "letter/key repair hypotheses."
            ),
        }

    def _nearby_resegment_window_suggestions(
        self,
        branch: str,
        *,
        start_idx: int,
        proposed_words: list[str],
        radius: int = 5,
        max_word_count: int = 6,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        proposed_stream = "".join(proposed_words)
        proposed_len = len(proposed_stream)
        if not proposed_words or proposed_len == 0:
            return []
        current_words = self._branch_reading_words(branch)
        out: list[dict[str, Any]] = []
        lo = max(0, start_idx - radius)
        hi = min(len(current_words), start_idx + radius + 1)
        for cand_start in range(lo, hi):
            stream = ""
            for cand_count in range(1, max_word_count + 1):
                end = cand_start + cand_count
                if end > len(current_words):
                    break
                stream += current_words[end - 1]
                if len(stream) > proposed_len:
                    break
                if len(stream) != proposed_len:
                    continue
                projected_words: list[str] = []
                pos = 0
                for proposed_word in proposed_words:
                    projected_words.append(stream[pos:pos + len(proposed_word)])
                    pos += len(proposed_word)
                mismatches = self._first_reading_mismatches(stream, proposed_stream)
                out.append({
                    "start_word_index": cand_start,
                    "word_count": cand_count,
                    "current_window_words": current_words[cand_start:end],
                    "current_char_count": len(stream),
                    "proposed_char_count": proposed_len,
                    "character_preserving": stream == proposed_stream,
                    "mismatch_count_preview": len(mismatches),
                    "mismatches": mismatches[:6],
                    "projected_words": projected_words,
                    "suggested_call": (
                        "act_resegment_window_by_reading("
                        f"branch={branch!r}, start_word_index={cand_start}, "
                        f"word_count={cand_count}, "
                        f"proposed_text={' '.join(proposed_words)!r})"
                    ),
                })
                break
        out.sort(key=lambda item: (
            not item["character_preserving"],
            item["mismatch_count_preview"],
            abs(int(item["start_word_index"]) - start_idx),
            item["word_count"],
        ))
        return out[:limit]

    def _tool_act_resegment_window_by_reading(self, args: dict) -> Any:
        branch = args["branch"]
        start_idx = int(args["start_word_index"])
        word_count = int(args["word_count"])
        proposed_text = str(args["proposed_text"])
        if self._pt_alpha()._multisym:
            return {
                "error": (
                    "act_resegment_window_by_reading currently supports "
                    "single-character plaintext alphabets only"
                )
            }
        spans = self.workspace.effective_word_spans(branch)
        if start_idx < 0 or start_idx >= len(spans):
            return {"error": f"start_word_index {start_idx} out of range"}
        if word_count <= 0 or start_idx + word_count > len(spans):
            return {
                "error": (
                    f"word_count {word_count} out of range for start_word_index "
                    f"{start_idx}"
                )
            }

        current_words = self._branch_reading_words(branch)
        window_words = current_words[start_idx:start_idx + word_count]
        current_stream = "".join(window_words)
        proposed_words = self._reading_words_from_text(proposed_text)
        proposed_stream = "".join(proposed_words)
        same_len = len(current_stream) == len(proposed_stream)
        same_chars = current_stream == proposed_stream
        base: dict[str, Any] = {
            "branch": branch,
            "start_word_index": start_idx,
            "old_window_word_count": word_count,
            "current_window_words": window_words,
            "proposed_words": proposed_words,
            "character_preserving": same_chars,
            "same_character_count": same_len,
            "current_char_count": len(current_stream),
            "proposed_char_count": len(proposed_stream),
            "mismatches": [] if same_chars else self._first_reading_mismatches(
                current_stream,
                proposed_stream,
            ),
        }
        if not proposed_words:
            return {"error": "proposed_text must contain at least one word", **base}
        if not same_len:
            return {
                "error": (
                    "Local proposed reading does not have the same normalized "
                    "character count as the selected current word window. "
                    "Select a different word_count or revise proposed_text."
                ),
                **base,
                "nearby_compatible_windows": self._nearby_resegment_window_suggestions(
                    branch,
                    start_idx=start_idx,
                    proposed_words=proposed_words,
                ),
                "next_step": (
                    "If nearby_compatible_windows is non-empty, retry with one "
                    "of those suggested calls. This usually means the word index "
                    "shifted after an earlier boundary edit."
                ),
            }

        window_start = spans[start_idx][0]
        window_end = spans[start_idx + word_count - 1][1]
        if sum(len(word) for word in proposed_words) != window_end - window_start:
            return {
                "error": (
                    "Local proposed word lengths do not match the selected "
                    "ciphertext token window."
                ),
                **base,
            }

        projected_words: list[str] = []
        local_spans: list[tuple[int, int]] = []
        pos = window_start
        stream_pos = 0
        for word in proposed_words:
            end = pos + len(word)
            local_spans.append((pos, end))
            projected_words.append(current_stream[stream_pos:stream_pos + len(word)])
            pos = end
            stream_pos += len(word)

        before = self._compute_quick_scores(branch)
        before_preview = self._decoded_preview(branch, max_words=80)
        new_spans = spans[:start_idx] + local_spans + spans[start_idx + word_count:]
        try:
            self.workspace.set_word_spans(branch, new_spans)
        except WorkspaceError as exc:
            return {"error": str(exc), **base}
        after = self._compute_quick_scores(branch)
        after_preview = self._decoded_preview(branch, max_words=80)
        return {
            "status": "ok",
            **base,
            "applied": True,
            "mode": "local_boundary_projection",
            "new_window_word_count": len(proposed_words),
            "old_total_word_count": len(spans),
            "new_total_word_count": len(new_spans),
            "projected_words": projected_words,
            "window_dictionary_before": self._dictionary_summary_for_words(window_words),
            "projected_dictionary_after": self._dictionary_summary_for_words(projected_words),
            "proposed_dictionary": self._dictionary_summary_for_words(proposed_words),
            "score_delta": self._reading_score_delta(before, after),
            "decoded_preview_before": before_preview,
            "decoded_preview": after_preview,
            "note": (
                "Applied a local word-boundary overlay only. The branch key "
                "and decoded character stream were unchanged. If proposed_words "
                "differ from projected_words, treat the mismatches as separate "
                "letter/key repair hypotheses."
            ),
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
            # Suggest the unidirectional cipher-symbol calls that match the
            # *intent* of the swap, so the agent can try a surgical fix
            # instead of a bidirectional one if the reading only justifies
            # changing one direction.
            unidirectional_hints: list[str] = []
            for cs in swapped_a:
                unidirectional_hints.append(
                    f"act_set_mapping(branch='{branch}', "
                    f"cipher_symbol='{cs}', plain_letter='{letter_b}')"
                )
            for cs in swapped_b:
                unidirectional_hints.append(
                    f"act_set_mapping(branch='{branch}', "
                    f"cipher_symbol='{cs}', plain_letter='{letter_a}')"
                )
            return {
                "status": "reverted",
                "branch": branch,
                "attempted_swap": f"{letter_a} ↔ {letter_b}",
                "cipher_symbols_for_a": swapped_a,
                "cipher_symbols_for_b": swapped_b,
                "trial_decoded_preview": trial_preview,
                "decoded_preview": self._decoded_preview(branch),
                "score_delta": delta,
                "unidirectional_alternatives": unidirectional_hints[:6],
                "note": (
                    "Swap made the branch score worse and was automatically "
                    "reverted. act_swap_decoded is bidirectional and changes "
                    "every cipher symbol producing either letter — which "
                    "almost always breaks correctly-decoded words. If your "
                    "reading only justifies changing ONE cipher symbol's "
                    "mapping, use one of `unidirectional_alternatives` "
                    "(act_set_mapping) instead. That is the right primitive "
                    "for reading-driven repairs."
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
                "single response. After targeted fixes, do one anchored polish "
                "search with search_homophonic_anneal(preserve_existing=true, "
                "solver_profile='zenith_native') and then read again. Do NOT "
                "fix errors one per iteration; each iteration costs tokens and "
                "grows the context."
            )
        return (
            f"Read decoded_preview carefully. If ANY recognisable {self.language} "
            "words appear, call meta_declare_solution now — a partial solution "
            "scores better than no declaration. If a few residual errors remain, "
            "call decode_diagnose_and_fix(branch) to fix them all in one call, "
            "then do one anchored polish run with "
            "search_anneal(preserve_existing=true, score_fn='combined') before "
            "declaring."
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
        pt_alpha = ws.plaintext_alphabet
        pt_size = pt_alpha.size
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
        id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in range(pt_alpha.size)}
        letter_to_id = {v: k for k, v in id_to_letter.items()}

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

        key_repair_info = automated_runner._run_key_consistent_repair(
            cipher_text=ws.cipher_text,
            key=best_key,
            language=self.language,
            word_list=self.word_list,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            score_fn=automated_runner._quadgram_key_score_fn(
                list(ws.cipher_text.tokens),
                id_to_letter,
                ngram.NGRAM_CACHE.get(self.language, 4),
            ),
        )
        anchor_refine_info: dict[str, Any] | None = None
        final_key = best_key
        final_score = best_score
        if key_repair_info["applied"]:
            final_key = key_repair_info["key"]
            temp_session.set_full_key(final_key)

            def score_fn_repaired() -> float:
                return ngram.normalized_ngram_score(
                    temp_session.apply_key(),
                    ngram.NGRAM_CACHE.get(self.language, 4),
                    n=4,
                )

            anchor_refine_info = automated_runner._maybe_anchor_refine_substitution(
                cipher_text=ws.cipher_text,
                session=temp_session,
                key=final_key,
                language=self.language,
                id_to_letter=id_to_letter,
                score_fn=score_fn_repaired,
            )
            if anchor_refine_info["applied"]:
                final_key = anchor_refine_info["key"]
                final_score = anchor_refine_info["score"]
            else:
                final_score = score_fn_repaired()

        ws.set_full_key(branch_name, final_key)
        return {
            "branch": branch_name,
            "score_fn": score_fn_name,
            "before": round(before, 4) if before != float("-inf") else None,
            "after": round(final_score, 4) if final_score != float("-inf") else None,
            "improved": final_score > before,
            "steps_per_restart": steps,
            "restarts": restarts,
            "preserve_existing": preserve_existing,
            "preserved_symbols": len(anchors),
            "auto_seeded_symbols": seeded_count,
            "key_repair": key_repair_info,
            "anchor_refine": anchor_refine_info,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "note": self._search_declare_note("anneal"),
        }

    def _tool_search_homophonic_anneal(self, args: dict) -> Any:
        branch_name = args["branch"]
        solver_profile = str(args.get("solver_profile", "zenith_native")).strip().lower()
        epochs = int(args.get("epochs", 5))
        sampler_iterations = int(args.get("sampler_iterations", 2000))
        t_start = float(args.get("t_start", 0.012))
        t_end = float(args.get("t_end", 0.006))
        order = int(args.get("order", 5))
        preserve_existing = bool(args.get("preserve_existing", False))
        model_path = args.get("model_path")
        max_ngrams = int(args.get("max_ngrams", 3_000_000))
        distribution_weight = float(args.get("distribution_weight", 4.0))
        diversity_weight = float(args.get("diversity_weight", 1.5))
        seed = args.get("seed")
        seed = int(seed) if seed is not None else None
        top_n = max(1, int(args.get("top_n", 1)))
        write_candidate_branches = bool(args.get("write_candidate_branches", False))

        ws = self.workspace
        branch = ws.get_branch(branch_name)
        tokens = list(ws.effective_tokens(branch_name))
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
        requested_model = str(model_path or "").strip().lower()
        if solver_profile == "zenith_native" and requested_model in {"word_list", "wordlist", "none", "fallback"}:
            solver_profile = "legacy"

        if solver_profile == "zenith_native":
            from analysis.zenith_solver import load_zenith_binary_model, zenith_solve

            if model_path and str(model_path).strip().lower() != "word_list":
                bin_path = Path(str(model_path)).expanduser()
            else:
                bin_path = automated_runner._zenith_native_model_path(self.language)
            if bin_path is None:
                return {
                    "error": (
                        "zenith_native requires a binary language model; set "
                        f"DECIPHER_NGRAM_MODEL_{self.language.upper()} or pass model_path."
                    )
                }
            model = load_zenith_binary_model(bin_path)
            result = zenith_solve(
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
                seed=seed,
                top_n=top_n,
            )
            model_source = str(bin_path)
            model_ngrams = len(model.log_probs)
            model_note = "zenith_binary"
            solver_name = "zenith_native"
        else:
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
                diversity_weight=diversity_weight,
                seed=seed,
                top_n=top_n,
            )
            model_source = model.source
            model_ngrams = len(model.log_probs)
            solver_name = "native_homophonic_anneal"

        selected_key = result.key
        selected_plaintext = result.plaintext
        key_repair_info: dict[str, Any] | None = None
        anchor_refine_info: dict[str, Any] | None = None

        if solver_profile == "zenith_native":
            key_repair_info = automated_runner._maybe_repair_zenith_native_key(
                cipher_text=ws.effective_cipher_text(branch_name),
                bin_path=bin_path,
                key=selected_key,
                plaintext=selected_plaintext,
                language=self.language,
                word_list=self.word_list,
                id_to_letter=id_to_letter,
                letter_to_id=letter_to_id,
            )
            if key_repair_info["applied"]:
                selected_key = key_repair_info["key"]
                selected_plaintext = key_repair_info["plaintext"]

            budget_params = {
                "seeds": [seed] if seed is not None else [0],
                "epochs": max(1, epochs),
                "sampler_iterations": max(1, sampler_iterations),
            }
            anchor_refine_info = automated_runner._maybe_anchor_refine_zenith_native(
                cipher_text=ws.effective_cipher_text(branch_name),
                bin_path=bin_path,
                key=selected_key,
                plaintext=selected_plaintext,
                anneal_score=result.normalized_score,
                language=self.language,
                word_list=self.word_list,
                plaintext_ids=plaintext_ids,
                id_to_letter=id_to_letter,
                letter_to_id=letter_to_id,
                budget_params=budget_params,
            )
            if anchor_refine_info["applied"]:
                selected_key = anchor_refine_info["key"]
                selected_plaintext = anchor_refine_info["plaintext"]

        ws.set_full_key(branch_name, selected_key)
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
            "solver": solver_name,
            "solver_profile": solver_profile,
            "before_scores": before,
            "after_scores": after,
            "score_delta": self._score_delta(before, after),
            "anneal_score": round(result.normalized_score, 4),
            "raw_anneal_score": round(result.score, 2),
            "model_source": model_source,
            "model_ngrams": model_ngrams,
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
            "diversity_weight": result.metadata.get("diversity_weight"),
            "top_n": top_n,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "homophone_warnings": distribution.get("warnings", []),
            "key_repair": key_repair_info,
            "anchor_refine": anchor_refine_info,
            "note": (
                "This native homophonic annealer is intended to replace manual "
                "guesswork on many-symbol no-boundary ciphers. Read the preview; "
                "if it is mostly correct, use decode_diagnose/decode_ambiguous_letter "
                "for residual rare-letter fixes, then do one preserve_existing=true "
                "polish call or declare the solution."
            ),
        }

    def _tool_search_automated_solver(self, args: dict) -> Any:
        branch_name = args["branch"]
        homophonic_budget = str(args.get("homophonic_budget", "full"))
        homophonic_refinement = str(args.get("homophonic_refinement", "none"))
        homophonic_solver = str(args.get("homophonic_solver", "zenith_native"))

        before = self._compute_quick_scores(branch_name)
        result = run_automated(
            cipher_text=self.workspace.effective_cipher_text(branch_name),
            language=self.language,
            cipher_id="agent_search",
            ground_truth=None,
            cipher_system="",
            homophonic_budget=homophonic_budget,
            homophonic_refinement=homophonic_refinement,
            homophonic_solver=homophonic_solver,
        )
        if result.error_message:
            return {
                "branch": branch_name,
                "status": result.status,
                "solver": result.solver,
                "error": result.error_message,
            }

        artifact = result.artifact or {}
        raw_key = artifact.get("key") or {}
        parsed_key = {int(k): int(v) for k, v in raw_key.items()}
        self.workspace.set_full_key(branch_name, parsed_key)
        after = self._compute_quick_scores(branch_name)
        steps = artifact.get("steps", []) or []
        route_step = steps[0] if steps else None
        primary_step = next(
            (step for step in steps if step.get("name") != "route_automated_solver"),
            route_step,
        )
        return {
            "branch": branch_name,
            "status": result.status,
            "solver": result.solver,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": homophonic_solver,
            "before_scores": before,
            "after_scores": after,
            "score_delta": self._score_delta(before, after),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
            "route_step": route_step,
            "primary_step": primary_step,
            "note": (
                "This runs the same local automated stack used by the "
                "no-LLM frontier/parity harness and installs the resulting key "
                "onto the requested branch."
            ),
        }

    def _tool_act_apply_transform_pipeline(self, args: dict) -> Any:
        branch_name = args["branch"]
        pipeline = TransformPipeline.from_raw(args.get("pipeline"))
        if pipeline is None:
            return {"error": "pipeline is required"}
        before_preview = self._decoded_preview(branch_name, max_words=20)
        result = self.workspace.apply_transform_pipeline(branch_name, pipeline)
        after_preview = self._decoded_preview(branch_name, max_words=20)
        return {
            **result,
            "before_preview": before_preview,
            "after_preview": after_preview,
            "note": (
                "The branch now has a token-order overlay. Subsequent decode "
                "and homophonic search tools operate in this transformed order."
            ),
        }

    def _tool_search_transform_homophonic(self, args: dict) -> Any:
        branch_name = args["branch"]
        columns = args.get("columns")
        columns = int(columns) if columns is not None else None
        profile = str(args.get("profile", "small"))
        top_n = max(1, int(args.get("top_n", 3)))
        write_best_branch = bool(args.get("write_best_branch", True))
        homophonic_budget = str(args.get("homophonic_budget", "screen"))
        base_tokens = self.workspace.effective_tokens(branch_name)
        structural_screen = screen_transform_candidates(
            base_tokens,
            columns=columns,
            profile=profile,
            top_n=max(top_n, 5),
            max_generated_candidates=(
                int(args["max_generated_candidates"])
                if args.get("max_generated_candidates") is not None
                else 25000 if profile == "wide" else 10000 if profile == "medium" else 5000
            ),
            streaming=profile == "wide" and not bool(args.get("include_mutations", profile != "small")),
            include_mutations=bool(args.get("include_mutations", profile != "small")),
            include_program_search=bool(args.get("include_program_search", profile == "medium")),
            program_max_depth=int(args.get("program_max_depth", 5)),
            program_beam_width=int(args.get("program_beam_width", 24)),
        )
        ordered_candidate_dicts = []
        identity = structural_screen.get("identity_candidate")
        if identity:
            ordered_candidate_dicts.append(identity)
        seen_candidate_ids = {item.get("candidate_id") for item in ordered_candidate_dicts}
        for item in structural_screen.get("top_candidates", []):
            if item.get("candidate_id") in seen_candidate_ids:
                continue
            ordered_candidate_dicts.append(item)
            seen_candidate_ids.add(item.get("candidate_id"))
        ranked: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for index, candidate in enumerate(ordered_candidate_dicts):
            try:
                pipeline = TransformPipeline.from_raw(candidate.get("pipeline"))
                if pipeline is None:
                    raise ValueError("missing transform pipeline")
                order = apply_transform_pipeline(list(range(len(base_tokens))), pipeline).tokens
                if sorted(order) != list(range(len(base_tokens))):
                    raise ValueError("transform candidate is not a position permutation")
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "candidate_index": index,
                    "candidate": candidate,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            try:
                transformed_tokens = apply_transform_pipeline(base_tokens, pipeline).tokens
                transformed_cipher = automated_runner._cipher_text_from_tokens(
                    transformed_tokens,
                    self.workspace.cipher_text.alphabet,
                    source=f"agent_transform_candidate:{index}",
                )
                result = run_automated(
                    cipher_text=transformed_cipher,
                    language=self.language,
                    cipher_id=f"agent_transform_candidate_{index}",
                    ground_truth=None,
                    cipher_system="homophonic_substitution",
                    homophonic_budget=homophonic_budget,
                    homophonic_solver="zenith_native",
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "candidate_index": index,
                    "candidate": candidate,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            artifact = result.artifact or {}
            primary_step = next(
                (step for step in artifact.get("steps", []) if step.get("name") != "route_automated_solver"),
                {},
            )
            ranked.append({
                "candidate_index": index,
                "candidate": candidate,
                "pipeline": pipeline.to_raw(),
                "status": result.status,
                "solver": result.solver,
                "anneal_score": primary_step.get("anneal_score"),
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "decoded_preview": result.final_decryption[:800],
                "key": artifact.get("key") or {},
            })
        ranked.sort(
            key=lambda item: (
                item.get("status") == "completed",
                float(item.get("anneal_score") or float("-inf")),
            ),
            reverse=True,
        )
        written_branch = None
        if write_best_branch and ranked:
            best = ranked[0]
            written_branch = f"{branch_name}_transform_best"
            if not self.workspace.has_branch(written_branch):
                self.workspace.fork(written_branch, from_branch=branch_name)
            self.workspace.apply_transform_pipeline(written_branch, best["pipeline"])
            self.workspace.set_full_key(
                written_branch,
                {int(k): int(v) for k, v in (best.get("key") or {}).items()},
            )
        return {
            "branch": branch_name,
            "profile": profile,
            "columns": columns,
            "candidate_count": structural_screen.get("deduped_candidate_count", 0),
            "structural_screen": structural_screen,
            "skipped_candidates": skipped[:20],
            "written_branch": written_branch,
            "top_candidates": ranked[:top_n],
            "note": (
                "This is a bounded screen over simple transform candidates. "
                "Read the previews and compare transformed branches before declaring."
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
    # benchmark_*
    # ------------------------------------------------------------------
    def _tool_inspect_benchmark_context(self, args: dict) -> Any:
        ctx = self.benchmark_context
        if ctx is None:
            return {
                "error": "No scoped benchmark context is available for this run.",
                "available": False,
            }
        include_text = bool(args.get("include_layer_text", True))
        layers = []
        for layer in ctx.injected_layers:
            item = dict(layer)
            if not include_text:
                item.pop("text", None)
            layers.append(item)
        ctx.log_access("inspect_benchmark_context", content_type="metadata")
        return {
            "available": True,
            "policy": ctx.policy,
            "target_record_ids": ctx.target_record_ids,
            "context_record_ids": ctx.context_record_ids,
            "injected_layers": layers,
            "related_records_available": [
                self._benchmark_record_summary(entry)
                for entry in ctx.related_records.values()
            ],
            "associated_documents_available": [
                self._benchmark_document_summary(entry)
                for entry in ctx.associated_documents.values()
            ],
            "related_solution_allowed": ctx.related_solution_allowed,
        }

    def _tool_list_related_records(self, args: dict) -> Any:
        ctx = self.benchmark_context
        if ctx is None:
            return {"error": "No scoped benchmark context is available for this run."}
        ctx.log_access("list_related_records", content_type="metadata")
        return {
            "policy": ctx.policy,
            "target_record_ids": ctx.target_record_ids,
            "context_record_ids": ctx.context_record_ids,
            "context_records": [
                self._benchmark_record_summary(ctx.records[record_id])
                for record_id in ctx.context_record_ids
                if record_id in ctx.records
            ],
            "related_records": [
                self._benchmark_record_summary(entry)
                for entry in ctx.related_records.values()
            ],
            "solution_access": (
                "enabled"
                if ctx.related_solution_allowed
                else "disabled by benchmark context policy"
            ),
        }

    def _tool_inspect_related_transcription(self, args: dict) -> Any:
        ctx = self.benchmark_context
        record_id = str(args.get("record_id") or "")
        max_chars = max(1, int(args.get("max_chars", 6000)))
        if ctx is None:
            return {"error": "No scoped benchmark context is available for this run."}
        entry = ctx.records.get(record_id)
        if entry is None:
            ctx.log_access(
                "inspect_related_transcription",
                record_id=record_id,
                content_type="transcription",
                allowed=False,
                error="record_not_allowed",
            )
            return {
                "error": (
                    f"Record `{record_id}` is not in this run's benchmark "
                    "context allowlist."
                ),
                "allowed_record_ids": sorted(ctx.records),
            }
        rel_path = entry.get("transcription_canonical_file") or ""
        if not rel_path:
            return {"error": f"Record `{record_id}` has no canonical transcription file."}
        try:
            text = safe_read_benchmark_file(entry.get("root") or ctx.benchmark_root, rel_path)
        except Exception as exc:  # noqa: BLE001
            ctx.log_access(
                "inspect_related_transcription",
                record_id=record_id,
                content_type="transcription",
                allowed=False,
                error=str(exc),
            )
            return {"error": f"{type(exc).__name__}: {exc}"}
        ctx.log_access(
            "inspect_related_transcription",
            record_id=record_id,
            content_type="transcription",
        )
        return {
            "record": self._benchmark_record_summary(entry),
            "content_type": "canonical_transcription",
            "truncated": len(text) > max_chars,
            "text": _truncate_text(text, max_chars),
        }

    def _tool_inspect_related_solution(self, args: dict) -> Any:
        ctx = self.benchmark_context
        record_id = str(args.get("record_id") or "")
        max_chars = max(1, int(args.get("max_chars", 6000)))
        if ctx is None:
            return {"error": "No scoped benchmark context is available for this run."}
        if not ctx.related_solution_allowed:
            ctx.log_access(
                "inspect_related_solution",
                record_id=record_id,
                content_type="solution",
                allowed=False,
                error="solution_context_disabled",
            )
            return {
                "error": (
                    "Related solution access is disabled by benchmark context "
                    f"policy `{ctx.policy}`."
                ),
                "policy": ctx.policy,
            }
        if record_id in ctx.target_record_ids:
            ctx.log_access(
                "inspect_related_solution",
                record_id=record_id,
                content_type="solution",
                allowed=False,
                error="target_solution_blocked",
            )
            return {"error": "The target record's solution is never exposed to the agent."}
        entry = ctx.records.get(record_id)
        if entry is None:
            ctx.log_access(
                "inspect_related_solution",
                record_id=record_id,
                content_type="solution",
                allowed=False,
                error="record_not_allowed",
            )
            return {
                "error": (
                    f"Record `{record_id}` is not in this run's benchmark "
                    "context allowlist."
                ),
                "allowed_record_ids": sorted(ctx.records),
            }
        rel_path = entry.get("plaintext_file") or ""
        if not rel_path:
            return {"error": f"Record `{record_id}` has no plaintext/solution file."}
        try:
            text = safe_read_benchmark_file(entry.get("root") or ctx.benchmark_root, rel_path)
        except Exception as exc:  # noqa: BLE001
            ctx.log_access(
                "inspect_related_solution",
                record_id=record_id,
                content_type="solution",
                allowed=False,
                error=str(exc),
            )
            return {"error": f"{type(exc).__name__}: {exc}"}
        ctx.log_access(
            "inspect_related_solution",
            record_id=record_id,
            content_type="solution",
        )
        return {
            "record": self._benchmark_record_summary(entry),
            "content_type": "plaintext_solution",
            "policy": ctx.policy,
            "truncated": len(text) > max_chars,
            "text": _truncate_text(text, max_chars),
        }

    def _tool_list_associated_documents(self, args: dict) -> Any:
        ctx = self.benchmark_context
        if ctx is None:
            return {"error": "No scoped benchmark context is available for this run."}
        ctx.log_access("list_associated_documents", content_type="metadata")
        return {
            "policy": ctx.policy,
            "documents": [
                self._benchmark_document_summary(entry)
                for entry in ctx.associated_documents.values()
            ],
        }

    def _tool_inspect_associated_document(self, args: dict) -> Any:
        ctx = self.benchmark_context
        document_id = str(args.get("document_id") or "")
        max_chars = max(1, int(args.get("max_chars", 6000)))
        if ctx is None:
            return {"error": "No scoped benchmark context is available for this run."}
        entry = ctx.associated_documents.get(document_id)
        if entry is None:
            ctx.log_access(
                "inspect_associated_document",
                document_id=document_id,
                content_type="document",
                allowed=False,
                error="document_not_allowed",
            )
            return {
                "error": (
                    f"Document `{document_id}` is not in this run's benchmark "
                    "context allowlist."
                ),
                "allowed_document_ids": sorted(ctx.associated_documents),
            }
        doc = entry.get("document", {})
        if doc.get("contains_solution") and not ctx.related_solution_allowed:
            ctx.log_access(
                "inspect_associated_document",
                document_id=document_id,
                content_type="document",
                allowed=False,
                error="solution_document_disabled",
            )
            return {
                "error": (
                    "This associated document is marked solution-bearing and "
                    f"cannot be read under policy `{ctx.policy}`."
                ),
                "document": self._benchmark_document_summary(entry),
            }
        safe_layers = set(doc.get("safe_context_layers") or [])
        if safe_layers and ctx.policy not in safe_layers and ctx.policy != "max":
            ctx.log_access(
                "inspect_associated_document",
                document_id=document_id,
                content_type="document",
                allowed=False,
                error="document_policy_not_allowed",
            )
            return {
                "error": (
                    f"Document `{document_id}` is not allowed under policy "
                    f"`{ctx.policy}`."
                ),
                "safe_context_layers": sorted(safe_layers),
            }
        rel_path = doc.get("text_file") or ""
        if not rel_path:
            ctx.log_access(
                "inspect_associated_document",
                document_id=document_id,
                content_type="document_metadata",
            )
            return {
                "document": self._benchmark_document_summary(entry),
                "message": "No text_file is declared for this associated document.",
            }
        root = ctx.benchmark_root
        try:
            text = safe_read_benchmark_file(root, rel_path)
        except Exception as exc:  # noqa: BLE001
            ctx.log_access(
                "inspect_associated_document",
                document_id=document_id,
                content_type="document",
                allowed=False,
                error=str(exc),
            )
            return {"error": f"{type(exc).__name__}: {exc}"}
        ctx.log_access(
            "inspect_associated_document",
            document_id=document_id,
            content_type="document",
        )
        return {
            "document": self._benchmark_document_summary(entry),
            "truncated": len(text) > max_chars,
            "text": _truncate_text(text, max_chars),
        }

    def _benchmark_record_summary(self, entry: dict[str, Any]) -> dict[str, Any]:
        rel = entry.get("relationship", {})
        return {
            "record_id": entry.get("id"),
            "area": entry.get("area"),
            "source": entry.get("source"),
            "status": entry.get("status"),
            "plaintext_language": entry.get("plaintext_language"),
            "cipher_type": entry.get("cipher_type", []),
            "date_or_century": entry.get("date_or_century"),
            "provenance": entry.get("provenance"),
            "notes": entry.get("notes"),
            "relationship": rel.get("relationship"),
            "solution_available": bool(rel.get("solution_available")) or bool(entry.get("plaintext_file")),
        }

    def _benchmark_document_summary(self, entry: dict[str, Any]) -> dict[str, Any]:
        doc = entry.get("document", {})
        return {
            "document_id": doc.get("id"),
            "record_id": entry.get("record_id"),
            "document_type": doc.get("document_type"),
            "title": doc.get("title"),
            "summary": doc.get("summary", ""),
            "language": doc.get("language", ""),
            "contains_solution": bool(doc.get("contains_solution")),
            "contains_plaintext_hint": bool(doc.get("contains_plaintext_hint")),
            "safe_context_layers": doc.get("safe_context_layers", []),
            "has_text_file": bool(doc.get("text_file")),
            "image_files": doc.get("image_files", []),
            "source_url": doc.get("source_url", ""),
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
        further_iterations_helpful = (
            bool(args["further_iterations_helpful"])
            if args.get("further_iterations_helpful") is not None else None
        )
        further_iterations_note = str(args.get("further_iterations_note") or "")
        forced_partial = bool(args.get("forced_partial", False))
        if not self.workspace.has_branch(branch):
            return {"error": f"Branch not found: {branch}"}
        unresolved = self._unresolved_repair_agenda_items(branch)
        if unresolved:
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "repair_agenda_unresolved",
                "unresolved_repair_agenda": unresolved,
                "note": (
                    "This branch has open/blocked repair agenda items. Before "
                    "declaring, apply them or mark them held/rejected with "
                    "repair_agenda_update. If your rationale already explains "
                    "why the repair should not be applied, make that explicit "
                    "in the agenda with status `held` or `rejected`, then call "
                    "meta_declare_solution again."
                ),
                "suggested_next_tools": [
                    "repair_agenda_list",
                    "repair_agenda_update",
                    "meta_declare_solution",
                ],
            }
        if len(self.workspace.branch_names()) > 1 and not self._has_seen_branch_cards():
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "branch_cards_required",
                "note": (
                    "Multiple branches exist. Call workspace_branch_cards before "
                    "declaring so you compare readable excerpts, internal "
                    "scores, repairs, and orthography risks."
                ),
                "suggested_next_tools": [
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ],
            }
        if self._should_guard_declaration_for_reading_workflow(branch, rationale):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "full_reading_workflow_required",
                "note": (
                    "Your rationale still mentions word-boundary/alignment "
                    "issues, but this branch has not gone through the full-"
                    "reading validation workflow. Before declaring, write your "
                    "best complete target-language reading and call "
                    "decode_validate_reading_repair. If it is character-"
                    "preserving, follow with act_resegment_by_reading. If it "
                    "changes letters but has the same character count, follow "
                    "with act_resegment_from_reading_repair, then repair the "
                    "reported mismatch spans with act_set_mapping/act_bulk_set."
                ),
                "suggested_next_tools": [
                    "decode_validate_reading_repair",
                    "act_resegment_by_reading",
                    "act_resegment_from_reading_repair",
                ],
            }
        if (
            further_iterations_helpful is True
            and self._declaration_has_untried_transform_work(rationale, further_iterations_note)
            and not forced_partial
            and not (self.max_iterations is not None and self._current_iteration >= self.max_iterations)
        ):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "transform_work_untried",
                "note": (
                    "Your declaration says further iterations should try a "
                    "transposition/period/poly-alphabetic style hypothesis, "
                    "but this run has not used the available transform tools. "
                    "Before declaring, inspect transform state and run a bounded "
                    "transform+homophonic screen. If that screen is not useful, "
                    "declare again with that negative result in the rationale."
                ),
                "suggested_next_tools": [
                    "observe_transform_pipeline",
                    "observe_transform_suspicion",
                    "search_transform_homophonic",
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ],
            }
        if self._should_guard_low_confidence_declaration(
            confidence,
            further_iterations_helpful is True,
            forced_partial,
        ):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "low_confidence_more_work_required",
                "note": (
                    "This is a low-confidence declaration and you also report "
                    "that further iterations would likely help. Continue using "
                    "the remaining budget instead of terminating early. Try a "
                    "new hypothesis class, compare branches, or mark "
                    "`further_iterations_helpful=false` only if the remaining "
                    "work is unlikely to improve the decipherment. Use "
                    "`forced_partial=true` only for an intentional early "
                    "partial submission when no useful remaining tool action is "
                    "available."
                ),
                "suggested_next_tools": [
                    "observe_transform_pipeline",
                    "observe_transform_suspicion",
                    "search_transform_homophonic",
                    "search_automated_solver",
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ],
            }
        self.solution = SolutionDeclaration(
            branch=branch,
            rationale=rationale,
            self_confidence=confidence,
            declared_at_iteration=self._current_iteration,
            reading_summary=str(args.get("reading_summary") or ""),
            further_iterations_helpful=further_iterations_helpful,
            further_iterations_note=further_iterations_note,
        )
        self.terminated = True
        review = self._declaration_review(branch)
        return {
            "status": "ok",
            "accepted": True,
            "branch": branch,
            "declared_at_iteration": self._current_iteration,
            "declaration_review": review,
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
