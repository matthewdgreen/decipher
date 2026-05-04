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

from analysis import cipher_id, dictionary, frequency, homophonic, ic, ngram, pattern, polyalphabetic
from analysis.finalist_validation import validate_plaintext_finalist
from analysis.pure_transposition import screen_pure_transposition
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
        "description": "Create a new branch from an existing one (from_branch, default empty main). Returns the new branch name.",
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
        "description": "Fork the current best-scoring branch into a new named branch. Equivalent to workspace_fork with from_branch=<best branch>.",
        "input_schema": {
            "type": "object",
            "properties": {
                "new_name": {
                    "type": "string",
                    "description": "Name for the new branch (alphanumeric, _ or -).",
                },
                "prefer_branch": {
                    "type": "string",
                    "description": "Source branch to copy if it exists; omit to choose the strongest branch.",
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
        "description": "Show a summary card for each workspace branch: name, score panel, mapped count, and top decoded words.",
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
        "name": "workspace_create_hypothesis_branch",
        "description": "Create a new hypothesis branch with a cipher_type label and optional evidence_source. Returns the branch name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "new_name": {"type": "string"},
                "cipher_mode": {
                    "type": "string",
                    "description": "Mode hypothesis, e.g. periodic_polyalphabetic.",
                },
                "rationale": {"type": "string"},
                "from_branch": {"type": "string", "default": "main"},
                "evidence_source": {
                    "type": "string",
                    "enum": [
                        "agent_inference",
                        "benchmark_context",
                        "ciphertext_statistics",
                        "solver_result",
                        "related_record",
                        "other",
                    ],
                    "default": "agent_inference",
                    "description": "Origin: `benchmark_context` if from injected layer, `fingerprint` for tool inference.",
                },
                "mode_confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["new_name", "cipher_mode", "rationale"],
        },
    },
    {
        "name": "workspace_reject_hypothesis",
        "description": "Mark a hypothesis branch as rejected and record a reason. Keeps the branch but flags it as eliminated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "reason": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["rejected", "superseded"],
                    "default": "rejected",
                },
                "acknowledge_pending_required_tools": {
                    "type": "boolean",
                    "default": False,
                    "description": "Emergency override: set true only when pending tools are impossible or irrelevant.",
                },
            },
            "required": ["branch", "reason"],
        },
    },
    {
        "name": "workspace_update_hypothesis",
        "description": "Update an existing hypothesis branch: change status (open/testing/eliminated/confirmed), confidence, evidence, or notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_mode": {
                    "type": "string",
                    "description": "Optional replacement mode label.",
                },
                "mode_status": {
                    "type": "string",
                    "enum": ["active", "paused", "rejected", "superseded"],
                    "default": "active",
                },
                "mode_confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "evidence": {"type": "string"},
                "counter_evidence": {"type": "string"},
                "next_recommended_action": {"type": "string"},
                "evidence_source": {
                    "type": "string",
                    "enum": [
                        "agent_inference",
                        "benchmark_context",
                        "ciphertext_statistics",
                        "solver_result",
                        "related_record",
                        "other",
                    ],
                    "description": "Updated origin. Use `benchmark_context` after reading context layers.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "workspace_hypothesis_cards",
        "description": "Show hypothesis cards: cipher_type, confidence, status, and evidence for each active hypothesis branch.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "workspace_hypothesis_next_steps",
        "description": "Return the suggested next tools for a hypothesis branch based on its current status and score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Optional hypothesis branch; omit for all active hypotheses.",
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
        "name": "observe_cipher_id",
        "description": "Compute and return a fresh cipher-type fingerprint for a branch's current token order. Returns suspicion scores, IC, entropy, and periodic IC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "max_period": {"type": "integer", "default": 26},
            },
        },
    },
    {
        "name": "observe_cipher_shape",
        "description": (
            "Compact structural view of a branch before choosing a cipher "
            "mode: token count, symbol inventory, boundary structure, repeated "
            "n-grams, pairability, and coordinate-looking symbol hints."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "top_n": {"type": "integer", "default": 12},
            },
        },
    },
    {
        "name": "observe_periodic_ic",
        "description": (
            "Detailed periodic-IC/Kasiski view for Vigenere-family diagnosis. "
            "Use this when observe_cipher_id suggests polyalphabetic or when "
            "raw IC is depressed but alphabet size is near 26."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "max_period": {"type": "integer", "default": 26},
                "top_n": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "observe_kasiski",
        "description": (
            "Detailed Kasiski repeated-sequence spacing report for periodic "
            "polyalphabetic diagnosis. Shows repeated n-grams, positions, "
            "spacings, and factor/period support."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "min_ngram": {"type": "integer", "default": 3},
                "max_ngram": {"type": "integer", "default": 5},
                "max_period": {"type": "integer", "default": 40},
                "top_n": {"type": "integer", "default": 12},
            },
        },
    },
    {
        "name": "observe_phase_frequency",
        "description": (
            "Show per-phase frequency profiles for a proposed periodic key "
            "length. Use after observe_periodic_ic/Kasiski suggests a period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "period": {
                    "type": "integer",
                    "description": "Period to inspect. Defaults to branch key period or fingerprint best period.",
                },
                "top_n": {"type": "integer", "default": 8},
            },
        },
    },
    {
        "name": "observe_periodic_shift_candidates",
        "description": (
            "For a proposed period and variant, rank likely Caesar shifts for "
            "each key phase using monogram chi-squared. Use before manually "
            "setting or adjusting periodic shifts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "period": {
                    "type": "integer",
                    "description": "Period to inspect. Defaults to branch key period or fingerprint best period.",
                },
                "variant": {
                    "type": "string",
                    "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                    "default": "vigenere",
                },
                "top_n": {"type": "integer", "default": 5},
                "sample": {"type": "integer", "default": 80},
            },
        },
    },
    {
        "name": "observe_homophone_distribution",
        "description": "Show how many cipher symbols map to each decoded plaintext letter. Identifies overloaded letters (possible homophones) and absent letters. Use before homophonic solvers or targeted repairs.",
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
        "description": "Show the current transform pipeline state for a branch: applied transforms, column count, row count, and derived token order.",
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string", "default": "main"}},
        },
    },
    {
        "name": "observe_transform_suspicion",
        "description": "Observe statistical signals (IC, bigram entropy, periodic IC) to estimate how likely a transposition or transform step is present. Returns suspicion scores per family.",
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
        "description": "Generate and score transposition transform candidates for a branch. Returns ranked finalists for review via search_review_transform_finalists.",
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
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
        },
    },
    {
        "name": "search_pure_transposition",
        "description": "Search for a pure columnar transposition key on a branch. Tries candidate column counts and orderings; returns finalist keys ranked by n-gram score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "profile": {
                    "type": "string",
                    "enum": ["small", "medium", "wide"],
                    "default": "wide",
                },
                "top_n": {
                    "type": "integer",
                    "default": 10,
                    "description": "How many ranked candidates to return.",
                },
                "install_top_n": {
                    "type": "integer",
                    "default": 1,
                    "description": "How many top candidates to install as readable workspace branches.",
                },
                "new_branch_prefix": {
                    "type": "string",
                    "default": "trans",
                    "description": "Prefix for installed branch names.",
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "Optional hard cap on generated candidates.",
                },
                "include_transmatrix": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include K3-style TransMatrix candidates.",
                },
                "include_matrix_rotate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include direct MatrixRotate candidates.",
                },
                "transmatrix_min_width": {"type": "integer", "default": 2},
                "transmatrix_max_width": {
                    "type": "integer",
                    "description": "Optional maximum TransMatrix width.",
                },
                "threads": {
                    "type": "integer",
                    "default": 0,
                    "description": "Rust worker count; 0 means auto-size.",
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
        },
    },
    {
        "name": "search_periodic_polyalphabetic",
        "description": "Search for a periodic polyalphabetic (Vigenère-family) key given a known or estimated key length. Returns candidate keys ranked by n-gram score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "max_period": {"type": "integer", "default": 20},
                "periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional exact periods to test.",
                },
                "variants": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                    },
                },
                "top_n": {"type": "integer", "default": 5},
                "install_top_n": {
                    "type": "integer",
                    "default": 1,
                    "description": "How many top candidates to install as branches; 0 means screen only.",
                },
                "new_branch_prefix": {"type": "string", "default": "poly"},
            },
        },
    },
    {
        "name": "search_quagmire3_keyword_alphabet",
        "description": "Search for a Quagmire III cipher key: keyword-generated alphabet with a running key. Returns ranked candidate keyword-alphabet pairs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "default": "main"},
                "keyword_lengths": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Candidate alphabet-keyword lengths, e.g. [7].",
                },
                "cycleword_lengths": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Candidate cycleword/key lengths, e.g. [8].",
                },
                "initial_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Crib keyword seeds. A solution-bearing keyword makes results seeded, not blind.",
                },
                "steps": {"type": "integer", "default": 200},
                "restarts": {"type": "integer", "default": 8},
                "seed": {"type": "integer", "default": 1},
                "top_n": {"type": "integer", "default": 5},
                "install_top_n": {
                    "type": "integer",
                    "default": 1,
                    "description": "Install this many candidates as branches; 0 = screen only.",
                },
                "estimate_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return budget/runtime estimates without running. Recommended before broad searches.",
                },
                "screen_top_n": {"type": "integer", "default": 64},
                "word_weight": {"type": "number", "default": 0.25},
                "dictionary_keyword_limit": {
                    "type": "integer",
                    "default": 0,
                    "description": "Add first N dictionary keyword starts.",
                },
                "engine": {
                    "type": "string",
                    "enum": ["python_screen", "rust_shotgun"],
                    "default": "rust_shotgun",
                    "description": "rust_shotgun=parallel compiled loop (fast); python_screen=quick; python_broad=exhaustive.",
                },
                "hillclimbs": {
                    "type": "integer",
                    "default": 500,
                    "description": "Rust shotgun hillclimb proposals per restart.",
                },
                "threads": {
                    "type": "integer",
                    "default": 0,
                    "description": "Rust worker threads; 0 means all available cores.",
                },
                "slip_probability": {"type": "number", "default": 0.001},
                "backtrack_probability": {"type": "number", "default": 0.15},
                "new_branch_prefix": {"type": "string", "default": "quag3"},
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
        "name": "decode_show_phases",
        "description": "Show the decoded text split into period-length phases (columns). Use with periodic ciphers to inspect per-column letter distributions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "period": {
                    "type": "integer",
                    "description": "Optional period override; defaults to branch periodic metadata.",
                },
                "variant": {
                    "type": "string",
                    "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                },
                "sample": {"type": "integer", "default": 24},
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
        "description": "Signal panel for a branch: dictionary_rate, quadgram + bigram loglik, bigram_chi2, pattern_consistency, constraint_satisfaction, mapped counts. Works for both word-boundary and no-boundary ciphers.",
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
        "description": "Dictionary hit rate for a branch: fraction of decoded word groups matching the target-language dictionary. Works for both word-boundary and no-boundary ciphers.",
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
        "description": "Return dictionary words matching a decoded word pattern (including ? wildcards). Useful for identifying candidate words for act_anchor_word.",
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
        "description": "Map a single cipher symbol to a plaintext letter. cipher_symbol is a symbol name (e.g. 'S001', 'X'), not a decoded letter. Surgical and unidirectional. Returns changed_words sample.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_symbol": {"type": "string"},
                "plain_letter": {"type": "string"},
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Preview changed_words without mutating the branch.",
                },
                "allow_mode_mismatch_repair": {
                    "type": "boolean",
                    "default": False,
                    "description": "Override cipher-mode guard for substitution repair on a non-substitution branch.",
                },
            },
            "required": ["branch", "cipher_symbol", "plain_letter"],
        },
    },
    {
        "name": "act_set_periodic_key",
        "description": "Set a full periodic (Vigenère-style) key on a branch: provide key_length and per-position shift values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "key": {"type": "string"},
                "shifts": {"type": "array", "items": {"type": "integer"}},
                "variant": {
                    "type": "string",
                    "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                    "default": "vigenere",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "act_set_periodic_shift",
        "description": (
            "Set one phase shift in a periodic polyalphabetic branch and "
            "refresh decoded_text. Phase is zero-based. Use decode_show_phases "
            "before and after to inspect the effect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "phase": {"type": "integer"},
                "shift": {"type": "integer"},
                "variant": {
                    "type": "string",
                    "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                },
            },
            "required": ["branch", "phase", "shift"],
        },
    },
    {
        "name": "act_adjust_periodic_shift",
        "description": (
            "Increment/decrement one phase shift in a periodic polyalphabetic "
            "branch and refresh decoded_text. Phase is zero-based; delta "
            "defaults to +1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "phase": {"type": "integer"},
                "delta": {"type": "integer", "default": 1},
                "variant": {
                    "type": "string",
                    "enum": ["vigenere", "beaufort", "variant_beaufort", "gronsfeld"],
                },
            },
            "required": ["branch", "phase"],
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
                "allow_mode_mismatch_repair": {"type": "boolean", "default": False},
            },
            "required": ["branch", "mappings"],
        },
    },
    {
        "name": "act_anchor_word",
        "description": "Anchor a known plaintext word at a cipher position: sets all cipher-symbol mappings implied by word=... at position=... Updates the branch key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "cipher_word_index": {"type": "integer"},
                "plaintext": {"type": "string"},
                "allow_mode_mismatch_repair": {"type": "boolean", "default": False},
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
        "description": "Bidirectional swap: exchanges ALL cipher symbols mapped to two decoded letters. Use only for deliberate full-population swaps. For single-word fixes use act_set_mapping on the specific cipher symbol instead.",
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
        "description": "Merge two adjacent cipher-text word groups into one at position word_index. Affects all branches. Use to fix over-split cipher transcription boundaries.",
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
        "description": "Merge two adjacent decoded words into one at word_index and word_index+1. Safer than act_resegment_by_reading when only one merge is needed; earlier edits cannot stale later indices.",
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
        "description": "Apply a word-boundary candidate returned by decode_diagnose or decode_repair_no_boundary. Provide the boundary_candidate_id from the diagnostic result.",
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
        "description": "Apply a word repair from decode_plan_word_repair_menu or decode_plan_word_repair. Use the suggested_call from those tools rather than constructing this call manually.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "target_word": {"type": "string"},
                "cipher_word_index": {"type": "integer"},
                "decoded_word": {"type": "string"},
                "occurrence": {"type": "integer", "default": 0},
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Preview the repair without mutating branch or agenda.",
                },
                "allow_bad_basin_repair": {
                    "type": "boolean",
                    "default": False,
                    "description": "Override bad-basin guard when the repair is manuscript-faithful despite a score drop.",
                },
                "allow_mode_mismatch_repair": {"type": "boolean", "default": False},
            },
            "required": ["branch", "target_word"],
        },
    },
    {
        "name": "act_resegment_by_reading",
        "description": "Replace a branch's word boundaries with a proposed reading, preserving the exact decoded character stream. Applies new word spans only if letters match exactly. For letter changes use act_set_mapping after.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": "Reading with desired boundaries. Letters must match the current decoded stream exactly.",
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "act_resegment_from_reading_repair",
        "description": "Apply the word-boundary pattern of a proposed reading without changing decoded letters or key. Use when characters are correct but boundaries are wrong. Returns mismatch spans for act_set_mapping follow-up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": "Best reading. Letters may differ from branch; normalised character count must match.",
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "act_resegment_window_by_reading",
        "description": "Replace word boundaries in a window of decoded words (start_word_index..end_word_index) with a proposed reading, preserving the decoded character stream in that window.",
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
        "description": "Greedy per-symbol key refinement. Only use after search_anneal has produced a good starting point. Polishes last few residual symbols.",
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
        "description": "Simulated annealing — primary search tool. Preserves partial anchors by default; treats fully-mapped inherited key as a fresh restart unless preserve_existing=true. score_fn defaults to 'combined'.",
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
                    "description": "Preserve current mappings as anchors. Omitted: partial keys preserved, full key reset.",
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "search_automated_solver",
        "description": "Run the automated no-LLM solver on a branch. Returns solver status, key, and decoded text. Use to get a fresh automated baseline without a full preflight run.",
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
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "search_homophonic_anneal",
        "description": "Primary solver for homophonic ciphers (alphabet_size > 26, no boundaries). Uses 5-gram + letter-distribution scoring. Runs directly on main. Prefer over search_anneal when homophonic evidence is present.",
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
                    "description": "Zenith n-gram CSV path. Auto-discovered if omitted; use 'word_list' for small fallback.",
                },
                "max_ngrams": {
                    "type": "integer",
                    "default": 3000000,
                    "description": "Maximum corpus n-grams to load from model_path.",
                },
                "distribution_weight": {
                    "type": "number",
                    "default": 4.0,
                    "description": "Plaintext letter-distribution penalty weight; prevents collapsed-letter solutions.",
                },
                "diversity_weight": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Plaintext diversity penalty weight; prevents collapsed-letter solutions on short texts.",
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
                    "description": "Write non-best candidates to sibling branches: <branch>_cand2, <branch>_cand3, …",
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "act_apply_transform_pipeline",
        "description": "Apply a transform pipeline (column/row permutation) to a branch's token order. Provide transform_spec as returned by search_transform_candidates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "pipeline": {
                    "type": "object",
                    "description": "Pipeline with optional columns/rows and steps [{name, data}].",
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["branch", "pipeline"],
        },
    },
    {
        "name": "act_install_transform_finalists",
        "description": (
            "Install selected finalists from a previous search_transform_homophonic "
            "session as workspace branches by rank. Use this after reviewing "
            "multiple finalist pages; it avoids rerunning the wide search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "ranks": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-based finalist ranks to install as branches.",
                },
                "branch_prefix": {
                    "type": "string",
                    "description": (
                        "Optional branch prefix. Defaults to "
                        "<source_branch>_transform_rank."
                    ),
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["search_session_id", "ranks"],
        },
    },
    {
        "name": "act_rate_transform_finalist",
        "description": "Rate a transform finalist's quality: accept, refine, or reject it for promotion. Provide finalist_id and rating (accept/refine/reject) plus optional notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "rank": {
                    "type": "integer",
                    "description": "1-based finalist rank to rate.",
                },
                "readability_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 4,
                    "description": "0=garbage 1=word-islands 2=structured 3=partial-clause 4=mostly-readable 5=solved.",
                },
                "label": {
                    "type": "string",
                    "enum": [
                        "coherent_plaintext",
                        "partial_clause",
                        "word_islands_with_some_structure",
                        "word_islands_only",
                        "garbage",
                    ],
                },
                "rationale": {
                    "type": "string",
                    "description": "Reading evidence: quote/paraphrase what it appears to say, or why it is garbage.",
                },
                "coherent_clause": {
                    "type": "string",
                    "description": "Paraphrasable clause if any; empty for word-island/garbage.",
                },
            },
            "required": ["search_session_id", "rank", "readability_score", "label", "rationale"],
        },
    },
    {
        "name": "search_transform_homophonic",
        "description": "Search for transposition + homophonic solve in one pass. Tests candidate column/row permutations then solves the rearranged cipher as homophonic. Use when transform suspicion is present.",
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
                "write_candidate_branches": {
                    "type": "boolean",
                    "default": False,
                    "description": "Write extra branches for top-N finalist candidates, e.g. main_transform_rank1, rank2…",
                },
                "candidate_branch_count": {
                    "type": "integer",
                    "default": 3,
                    "description": "How many top finalists to write when write_candidate_branches=true.",
                },
                "review_chars": {
                    "type": "integer",
                    "default": 600,
                    "description": "Maximum decoded preview characters per finalist in finalist_review.",
                },
                "good_score_gap": {
                    "type": "number",
                    "default": 0.25,
                    "description": "Finalists within this score gap of the best are counted as good-score candidates.",
                },
                "homophonic_budget": {"type": "string", "enum": ["screen", "full"], "default": "screen"},
                "include_program_search": {"type": "boolean", "default": False},
                "program_max_depth": {"type": "integer", "default": 5},
                "program_beam_width": {"type": "integer", "default": 24},
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["branch"],
        },
    },
    {
        "name": "search_review_transform_finalists",
        "description": "Review ranked transform finalists from search_transform_candidates or search_transform_homophonic. Returns finalist IDs, keys, decoded previews, and scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "start_rank": {"type": "integer", "default": 1},
                "count": {"type": "integer", "default": 5},
                "review_chars": {"type": "integer", "default": 600},
                "good_score_gap": {"type": "number", "default": 0.25},
            },
            "required": ["search_session_id"],
        },
    },
    {
        "name": "search_review_pure_transposition_finalists",
        "description": "Review ranked finalists from search_pure_transposition. Returns finalist IDs, column orderings, decoded previews, and scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "start_rank": {"type": "integer", "default": 1},
                "count": {"type": "integer", "default": 5},
                "review_chars": {"type": "integer", "default": 600},
                "good_score_gap": {"type": "number", "default": 0.25},
            },
            "required": ["search_session_id"],
        },
    },
    {
        "name": "act_install_pure_transposition_finalists",
        "description": (
            "Install selected finalists from a previous search_pure_transposition "
            "session as readable transform branches by rank, without rerunning "
            "the Rust screen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "ranks": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-based finalist ranks to install as branches.",
                },
                "branch_prefix": {
                    "type": "string",
                    "description": (
                        "Optional branch prefix. Defaults to "
                        "<source_branch>_pure_rank."
                    ),
                },
                "override_context_cipher_family": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true only when overriding an exposed benchmark-context cipher family.",
                },
                "context_override_rationale": {
                    "type": "string",
                    "description": "Required if override_context_cipher_family=true. Why the benchmark context is wrong.",
                },
            },
            "required": ["search_session_id", "ranks"],
        },
    },
    {
        "name": "decode_letter_stats",
        "description": "Per-letter frequency table for a branch's decoded text: count, frequency, and deviation from language reference. Use to spot overrepresented or missing letters.",
        "input_schema": {
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
    },
    {
        "name": "decode_ambiguous_letter",
        "description": "Show which cipher symbols produce a given decoded letter and sample contexts per symbol. Use before changing an overused decoded letter to make targeted act_set_mapping calls instead of act_swap_decoded.",
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
        "description": "Rank cipher symbols as candidates for a missing plaintext letter. Returns contexts and score delta per candidate. Use when a plaintext letter is absent or underrepresented in homophonic decodes.",
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
                    "description": "Decoded letters to check as sources. Omit to auto-select overrepresented letters.",
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
        "description": "Rank residual single-letter errors in decoded text. Returns suggested_call (act_set_mapping on culprit symbol) and bulk_fix_call (decode_diagnose_and_fix). Call after search_anneal when a few errors remain.",
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
        "description": "Diagnose residual errors and apply high-confidence fixes in one call. Tests each candidate with act_set_mapping, reverts worsening repairs. Returns before/after scores and a recommendation.",
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
                    "description": "Min evidence_count to auto-apply. Raise to be conservative; 1 to apply all candidates.",
                },
                "auto_revert_if_worse": {"type": "boolean", "default": True},
            },
            "required": ["branch"],
        },
    },
    {
        "name": "decode_repair_no_boundary",
        "description": "Apply an automated no-boundary repair pass: runs DP word segmentation then diagnoses and fixes single-symbol errors that break word sequences. Returns score delta and pseudo_words_remaining.",
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
        "description": "Read-only: validate a proposed reading against a branch. Reports character-preserving vs. character-changing, mismatch spans, and word scores. Apply with act_resegment_by_reading or act_resegment_from_reading_repair.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "proposed_text": {
                    "type": "string",
                    "description": "Best reading. Spaces may differ; letters may differ only for character-changing repairs.",
                },
            },
            "required": ["branch", "proposed_text"],
        },
    },
    {
        "name": "decode_plan_word_repair",
        "description": "Plan a single word repair: shows proposed cipher-symbol mappings, conflicts, changed-word preview, and delta scores. Returns suggested act_apply_word_repair call when safe.",
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
        "description": "Compare same-length readings for a decoded word (read-only). Shows proposed cipher-symbol mappings, conflicts, score deltas, and suggested act_apply_word_repair call. Use before committing an uncertain repair.",
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
        "description": "List the current repair agenda: pending and completed repair items with priorities and notes.",
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
        "description": "Read the benchmark context record for the current test. Returns provenance, cipher type, plaintext language, and any injected context layers.",
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
        "description": "Read the canonical transcription of a related benchmark record. Returns the cipher text string.",
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
        "description": "Read the solution for a related benchmark record (if allowed by context policy). Returns plaintext and key metadata.",
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
        "description": "List documents associated with the current benchmark record. Returns document IDs, titles, and types.",
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
        "description": "Execute Python 3 (stdlib only, 15 s timeout). Requires justification: why built-in tools are insufficient and what dedicated tool would replace this. Print results to stdout. Call meta_request_tool to flag tool gaps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "justification": {
                    "type": "string",
                    "description": "Why built-in tools are insufficient and what dedicated tool would replace this call.",
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
        "description": "Request a tool that is not currently in your tool list. Provide tool_name and reason. Use when you need a filtered-out tool or to flag a tool-design gap.",
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
        "description": "Declare the run solved: provide final_branch, confidence (0–1), and rationale. Required fields: final_branch, confidence, rationale. Optional: partial_flag, notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "rationale": {"type": "string"},
                "self_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reading_summary": {
                    "type": "string",
                    "description": "What the decipherment appears to say (1–2 sentences for the final screen).",
                },
                "further_iterations_helpful": {
                    "type": "boolean",
                    "description": "Whether more iterations would likely improve results.",
                },
                "further_iterations_note": {
                    "type": "string",
                    "description": "What further iterations should try, or why they won't help (1–2 sentences).",
                },
                "forced_partial": {
                    "type": "boolean",
                    "default": False,
                    "description": "Non-final partial submission. Does not end the run; a full declaration is still required.",
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
    {
        "name": "meta_declare_unsolved",
        "description": "Declare the run exhausted without a solution. Provide rationale. Required when no coherent decryption was found after all planned approaches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "branches_considered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Branches or hypotheses reviewed before stopping.",
                },
                "best_branch": {
                    "type": "string",
                    "description": "Best available branch for inspection, if any.",
                },
                "reading_summary": {
                    "type": "string",
                    "description": "Human-readable summary of what, if anything, was learned.",
                },
                "further_iterations_helpful": {"type": "boolean"},
                "further_iterations_note": {"type": "string"},
            },
            "required": [
                "rationale",
                "branches_considered",
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
        self.unsolved_declaration: dict[str, Any] | None = None
        self.max_iterations: int | None = None
        self.allowed_tool_names: set[str] | None = None

        # Log of all tool calls for the run artifact
        self.call_log: list[ToolCall] = []
        self._current_iteration: int = 0

        # Tool capability requests (meta_request_tool calls)
        self.tool_requests: list[dict] = []
        self.context_family_overrides: list[dict[str, Any]] = []

        # Durable reading-repair agenda. These are deliberately plain dicts so
        # they serialize directly into run artifacts and tool results.
        self.repair_agenda: list[dict[str, Any]] = []
        self._next_repair_agenda_id: int = 1

        # In-memory wide-search sessions. These let the agent page through
        # finalist reviews and install selected branches without rerunning an
        # expensive transform+homophonic screen in the same run.
        self._transform_search_sessions: dict[str, dict[str, Any]] = {}
        self._next_transform_search_session_id: int = 1
        self._pure_transposition_sessions: dict[str, dict[str, Any]] = {}
        self._next_pure_transposition_session_id: int = 1

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
        elif (context_block := self._context_cipher_family_tool_block(tool_name, args)) is not None:
            result = _json(context_block)
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

    def _resolve_observation_branch(self, requested: str | None) -> tuple[str, dict[str, Any]]:
        branch_name = requested or "main"
        if self.workspace.has_branch(branch_name):
            return branch_name, {}
        if branch_name == "automated_preflight" and self.workspace.has_branch("main"):
            return "main", {
                "requested_branch": branch_name,
                "branch_fallback": "main",
                "warning": (
                    "`automated_preflight` is not installed. This usually means "
                    "the no-LLM preflight ran but did not produce a usable key. "
                    "I used `main` for this observation instead."
                ),
            }
        return branch_name, {}

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

    def _recommended_tool_for_cipher_mode(self, mode: str) -> str:
        return {
            "monoalphabetic_substitution": "search_anneal",
            "simple_substitution": "search_anneal",
            "homophonic_substitution": "search_homophonic_anneal",
            "transposition": "search_pure_transposition",
            "transposition_homophonic": "search_transform_candidates",
            "polyalphabetic_vigenere": "observe_periodic_ic",
            "periodic_polyalphabetic": "search_periodic_polyalphabetic",
            "quagmire3": "search_quagmire3_keyword_alphabet",
            "keyed_tableau_polyalphabetic": "search_quagmire3_keyword_alphabet",
            "playfair": "observe_cipher_shape",
        }.get(mode, "workspace_create_hypothesis_branch")

    _POLY_MODE_NAMES = {
        "periodic_polyalphabetic",
        "polyalphabetic_vigenere",
        "keyed_tableau_polyalphabetic",
        "quagmire3",
    }

    _KEYED_TABLEAU_OFF_FAMILY_TOOLS = {
        "search_anneal": "monoalphabetic/simple substitution",
        "search_automated_solver": "generic automated routing",
        "search_homophonic_anneal": "homophonic substitution",
        "search_transform_candidates": "ciphertext-order transform screening",
        "search_pure_transposition": "pure-transposition search",
        "search_review_pure_transposition_finalists": "pure-transposition finalist review",
        "search_transform_homophonic": "transposition+homophonic search",
        "act_apply_transform_pipeline": "manual ciphertext-order transform",
        "act_rate_transform_finalist": "transform finalist rating",
        "act_install_transform_finalists": "transform finalist promotion",
        "act_install_pure_transposition_finalists": "pure-transposition finalist promotion",
    }

    def _context_cipher_family_assumptions(self) -> list[dict[str, Any]]:
        prior = self._context_keyed_tableau_prior()
        if not prior:
            return []
        return [{
            "cipher_mode": prior["prior"],
            "confidence": prior["confidence"],
            "source": "agent_declared_benchmark_context_assumption",
            "evidence": prior["evidence"],
            "assumption": (
                "The agent read exposed benchmark context and explicitly "
                "recorded it as a controlling cipher-family assumption."
            ),
            "risk": (
                "If the benchmark context is wrong or too specific, this "
                "assumption could suppress useful off-family searches."
            ),
        }]

    def _context_cipher_family_tool_block(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Prevent accidental drift away from an exposed context cipher family."""
        prior = self._context_keyed_tableau_prior()
        if not prior:
            return None
        attempted_family = self._KEYED_TABLEAU_OFF_FAMILY_TOOLS.get(tool_name)
        if attempted_family is None:
            return None
        if bool(args.get("override_context_cipher_family", False)):
            rationale = str(args.get("context_override_rationale") or "").strip()
            if len(rationale) < 40:
                return {
                    "status": "blocked",
                    "accepted": False,
                    "reason": "context_override_rationale_required",
                    "attempted_tool": tool_name,
                    "attempted_family": attempted_family,
                    "context_cipher_family": prior["prior"],
                    "context_assumption": prior,
                    "note": (
                        "Benchmark context explicitly supports a keyed-"
                        "tableau/Kryptos-style polyalphabetic cipher. To "
                        "leave that family, call this tool again with "
                        "`override_context_cipher_family=true` and a concrete "
                        "`context_override_rationale` explaining the evidence "
                        "that context may be wrong or exhausted."
                    ),
                    "suggested_next_tools": [
                        "workspace_hypothesis_next_steps",
                        "search_periodic_polyalphabetic",
                        "search_quagmire3_keyword_alphabet",
                        "workspace_branch_cards",
                    ],
                }
            self.context_family_overrides.append({
                "iteration": self._current_iteration,
                "tool": tool_name,
                "attempted_family": attempted_family,
                "context_cipher_family": prior["prior"],
                "rationale": rationale,
                "warning": (
                    "Agent explicitly overrode exposed benchmark cipher-family "
                    "context; compare this run separately from context-trusting "
                    "runs."
                ),
            })
            return None
        return {
            "status": "blocked",
            "accepted": False,
            "reason": "context_cipher_family_mismatch",
            "attempted_tool": tool_name,
            "attempted_family": attempted_family,
            "context_cipher_family": prior["prior"],
            "context_assumption": prior,
            "note": (
                "Exposed benchmark context says this cipher is keyed-"
                "Vigenere/Kryptos-style polyalphabetic. Trust that context "
                "as the working cipher family. Do not spend budget on "
                f"{attempted_family} tools unless you explicitly override the "
                "context and record why the context may be wrong."
            ),
            "suggested_next_tools": [
                "workspace_hypothesis_next_steps",
                "search_periodic_polyalphabetic",
                "search_quagmire3_keyword_alphabet",
                "workspace_branch_cards",
            ],
            "override_fields": {
                "override_context_cipher_family": True,
                "context_override_rationale": (
                    "Required if deliberately leaving the context-supported "
                    "cipher family."
                ),
            },
        }

    def _context_keyed_tableau_prior(self) -> dict[str, Any] | None:
        """Return a keyed-tableau prior explicitly declared by the agent.

        The executor does not parse benchmark prose to infer cipher type. If
        context says "this is keyed Vigenere" or similar, the agent must record
        that reading via workspace_create_hypothesis_branch or
        workspace_update_hypothesis with evidence_source="benchmark_context".
        """
        rows = []
        for name in self.workspace.branch_names():
            branch = self.workspace.get_branch(name)
            metadata = branch.metadata
            if not metadata.get("context_supported_mode"):
                continue
            if metadata.get("mode_status", "active") in {"rejected", "superseded"}:
                continue
            mode = str(metadata.get("cipher_mode") or "").strip()
            if mode not in {"keyed_tableau_polyalphabetic", "quagmire3"}:
                continue
            rows.append({
                "branch": name,
                "cipher_mode": mode,
                "evidence_source": metadata.get("evidence_source"),
                "mode_evidence": metadata.get("mode_evidence") or metadata.get("hypothesis_notes"),
                "context_assumption_note": metadata.get("context_assumption_note"),
            })
        if not rows:
            return None
        return {
            "prior": "keyed_tableau_polyalphabetic",
            "confidence": "agent_declared_context_supported",
            "evidence": rows,
            "required_tools_before_rejection": [
                "search_quagmire3_keyword_alphabet",
            ],
            "note": (
                "The agent has recorded exposed benchmark context as supporting "
                "a keyed-tableau/Kryptos-style hypothesis. Plain A-Z Vigenere "
                "failure does not reject this family; escalate to "
                "keyed-tableau/Quagmire search before rejecting it."
            ),
        }

    def _polyalphabetic_signal_prior(self) -> dict[str, Any] | None:
        fp = cipher_id.compute_cipher_fingerprint(
            self.workspace.cipher_text.tokens,
            self.workspace.cipher_text.alphabet.size,
            language=self.language,
            word_group_count=len(self.workspace.cipher_text.words),
        )
        scores = fp.suspicion_scores
        poly_score = float(scores.get("polyalphabetic_vigenere", 0.0))
        hom_score = float(scores.get("homophonic_substitution", 0.0))
        period_lift = (
            float(fp.best_period_ic or 0.0) - float(fp.ic or 0.0)
            if fp.best_period_ic is not None else 0.0
        )
        if poly_score < 0.55 and period_lift < 0.006:
            return None
        return {
            "prior": "polyalphabetic_family",
            "confidence": "statistical",
            "poly_score": round(poly_score, 4),
            "homophonic_score": round(hom_score, 4),
            "best_period": fp.best_period,
            "best_period_ic": round(float(fp.best_period_ic), 6) if fp.best_period_ic is not None else None,
            "raw_ic": round(float(fp.ic), 6),
            "period_lift": round(period_lift, 6),
            "note": (
                "Statistics make a periodic polyalphabetic family plausible. "
                "If ordinary Vigenere-family shifts fail, that only rules out "
                "one child model; keyed-tableau/Quagmire remains pending."
            ),
        }

    def _polyalphabetic_required_tools_for_mode(self, mode: str) -> list[str]:
        mode = (mode or "").strip()
        context_prior = self._context_keyed_tableau_prior()
        if context_prior and mode in self._POLY_MODE_NAMES:
            return list(context_prior["required_tools_before_rejection"])
        if mode in {"quagmire3", "keyed_tableau_polyalphabetic"}:
            return ["search_quagmire3_keyword_alphabet"]
        return []

    def _mode_playbook(self, mode: str) -> list[dict[str, str]]:
        mode = (mode or "unknown").strip()
        playbooks: dict[str, list[dict[str, str]]] = {
            "monoalphabetic_substitution": [
                {"tool": "search_anneal", "purpose": "Run the main substitution solver."},
                {"tool": "workspace_branch_cards", "purpose": "Read the candidate and compare branches."},
                {"tool": "decode_diagnose", "purpose": "Plan local repairs only if the text is near-readable."},
                {"tool": "score_panel", "purpose": "Use scores as supporting evidence."},
            ],
            "simple_substitution": [
                {"tool": "search_anneal", "purpose": "Run the main substitution solver."},
                {"tool": "workspace_branch_cards", "purpose": "Read the candidate and compare branches."},
                {"tool": "decode_diagnose", "purpose": "Plan local repairs only if the text is near-readable."},
                {"tool": "score_panel", "purpose": "Use scores as supporting evidence."},
            ],
            "homophonic_substitution": [
                {"tool": "search_automated_solver", "purpose": "Run the modern no-LLM solver stack first."},
                {"tool": "search_homophonic_anneal", "purpose": "Run focused homophonic annealing if needed."},
                {"tool": "workspace_branch_cards", "purpose": "Check for coherent clauses, not just word islands."},
                {"tool": "decode_absent_letter_candidates", "purpose": "Diagnose missing/overused letters before local repair."},
            ],
            "periodic_polyalphabetic": [
                {"tool": "observe_periodic_ic", "purpose": "Confirm candidate periods and IC recovery."},
                {"tool": "observe_kasiski", "purpose": "Corroborate periods with repeated-sequence spacings."},
                {"tool": "observe_phase_frequency", "purpose": "Inspect phase frequency once a period is plausible."},
                {"tool": "search_periodic_polyalphabetic", "purpose": "Try Vigenere/Beaufort/Gronsfeld candidates."},
                {"tool": "search_quagmire3_keyword_alphabet", "purpose": "Escalate to keyed-tableau/Quagmire III if standard Vigenere-family search fails."},
                {"tool": "decode_show_phases", "purpose": "Read installed periodic branches by phase."},
                {"tool": "act_set_periodic_shift", "purpose": "Adjust individual phase shifts only after reading evidence."},
            ],
            "quagmire3": [
                {"tool": "observe_periodic_ic", "purpose": "Confirm periodic evidence and plausible cycleword lengths."},
                {"tool": "search_periodic_polyalphabetic", "purpose": "Clear ordinary Vigenere-family hypotheses first."},
                {"tool": "search_quagmire3_keyword_alphabet", "purpose": "Search keyword-shaped shared alphabets plus derived cyclewords."},
                {"tool": "workspace_branch_cards", "purpose": "Read installed candidates and compare coherence."},
                {"tool": "workspace_update_hypothesis", "purpose": "Record whether Quagmire III remains plausible."},
            ],
            "keyed_tableau_polyalphabetic": [
                {"tool": "observe_periodic_ic", "purpose": "Confirm periodic evidence and plausible cycleword lengths."},
                {"tool": "search_periodic_polyalphabetic", "purpose": "Clear ordinary Vigenere-family hypotheses first."},
                {"tool": "search_quagmire3_keyword_alphabet", "purpose": "Try the first structured keyed-tableau hypothesis, Quagmire III."},
                {"tool": "workspace_branch_cards", "purpose": "Read installed candidates and compare coherence."},
                {"tool": "workspace_update_hypothesis", "purpose": "Record whether broader keyed-tableau search is needed."},
            ],
            "polyalphabetic_vigenere": [
                {"tool": "observe_periodic_ic", "purpose": "Confirm candidate periods and IC recovery."},
                {"tool": "observe_kasiski", "purpose": "Corroborate periods with repeated-sequence spacings."},
                {"tool": "search_periodic_polyalphabetic", "purpose": "Try Vigenere-family candidates."},
                {"tool": "search_quagmire3_keyword_alphabet", "purpose": "Escalate to keyed-tableau/Quagmire III if plain Vigenere-family candidates fail."},
                {"tool": "decode_show_phases", "purpose": "Read installed periodic branches by phase."},
            ],
            "transposition_homophonic": [
                {"tool": "observe_transform_suspicion", "purpose": "Check whether token order is suspicious."},
                {"tool": "workspace_update_hypothesis", "purpose": "Record why this is order plus homophonic/keying, not pure transposition."},
                {"tool": "search_transform_candidates", "purpose": "Run structural transform screening."},
                {"tool": "search_transform_homophonic", "purpose": "Promote candidates with homophonic solving."},
                {"tool": "search_review_transform_finalists", "purpose": "Page through multiple finalist readings."},
                {"tool": "act_rate_transform_finalist", "purpose": "Record contextual readability before choosing branches."},
                {"tool": "act_install_transform_finalists", "purpose": "Install selected finalists after review without rerunning search."},
            ],
            "transposition": [
                {"tool": "observe_transform_suspicion", "purpose": "Check whether token order is suspicious."},
                {"tool": "workspace_update_hypothesis", "purpose": "Record why this is wrong order only, with no substitution/homophonic key layer."},
                {"tool": "search_pure_transposition", "purpose": "Run the direct language-scored pure-transposition screen."},
                {"tool": "search_review_pure_transposition_finalists", "purpose": "Page through multiple pure-order finalist readings."},
                {"tool": "act_rate_transform_finalist", "purpose": "Record contextual readability before choosing branches."},
                {"tool": "act_install_pure_transposition_finalists", "purpose": "Install selected pure-order finalists without rerunning search."},
                {"tool": "search_transform_candidates", "purpose": "Use structural-only screening only if language-scored search is inconclusive."},
                {"tool": "act_apply_transform_pipeline", "purpose": "Apply a known or selected transform pipeline."},
                {"tool": "workspace_update_hypothesis", "purpose": "Record whether order correction helped."},
            ],
            "fractionation_transposition": [
                {"tool": "observe_cipher_shape", "purpose": "Inspect pairability, coordinate-like symbols, and token structure."},
                {"tool": "observe_cipher_id", "purpose": "Re-evaluate after any normalization/filtering."},
                {"tool": "meta_request_tool", "purpose": "Request missing fractionation/Polybius tooling if evidence supports it."},
                {"tool": "workspace_update_hypothesis", "purpose": "Mark as capability gap or paused if no solver exists yet."},
            ],
            "playfair": [
                {"tool": "observe_cipher_shape", "purpose": "Inspect digraph constraints and doubled-letter behavior."},
                {"tool": "meta_request_tool", "purpose": "Request Playfair/polygraphic tooling if evidence supports it."},
                {"tool": "workspace_update_hypothesis", "purpose": "Mark as capability gap or paused if no solver exists yet."},
            ],
        }
        return playbooks.get(mode, [
            {"tool": "observe_cipher_id", "purpose": "Rank plausible cipher families."},
            {"tool": "observe_cipher_shape", "purpose": "Inspect structural clues before choosing a mode."},
            {"tool": "workspace_create_hypothesis_branch", "purpose": "Create an explicit mode hypothesis."},
            {"tool": "workspace_update_hypothesis", "purpose": "Record evidence and next action."},
        ])

    def _mode_tool_menu(self, mode: str) -> dict[str, Any]:
        mode = (mode or "unknown").strip()
        always_available = [
            "inspect_benchmark_context",
            "observe_cipher_id",
            "observe_cipher_shape",
            "workspace_branch_cards",
            "workspace_hypothesis_cards",
            "workspace_hypothesis_next_steps",
            "workspace_update_hypothesis",
            "workspace_reject_hypothesis",
            "meta_declare_solution",
        ]
        escape_tools = [
            "workspace_update_hypothesis",
            "workspace_reject_hypothesis",
            "workspace_create_hypothesis_branch",
            "meta_request_tool",
        ]
        menus: dict[str, dict[str, Any]] = {
            "monoalphabetic_substitution": {
                "foreground_tools": [
                    "observe_frequency",
                    "observe_patterns",
                    "corpus_word_candidates",
                    "search_anneal",
                    "decode_diagnose",
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_anchor_word",
                    "act_apply_word_repair",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "search_transform_candidates",
                    "search_transform_homophonic",
                    "act_set_periodic_key",
                    "act_set_periodic_shift",
                ],
                "mode_warning": (
                    "Use local mapping repairs only after search has produced "
                    "a near-readable substitution basin."
                ),
            },
            "simple_substitution": {
                "foreground_tools": [
                    "observe_frequency",
                    "observe_patterns",
                    "corpus_word_candidates",
                    "search_anneal",
                    "decode_diagnose",
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_anchor_word",
                    "act_apply_word_repair",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "search_transform_candidates",
                    "search_transform_homophonic",
                    "act_set_periodic_key",
                    "act_set_periodic_shift",
                ],
                "mode_warning": (
                    "If this produces only word islands, reject the basin "
                    "instead of polishing local spellings."
                ),
            },
            "homophonic_substitution": {
                "foreground_tools": [
                    "search_automated_solver",
                    "search_homophonic_anneal",
                    "decode_letter_stats",
                    "decode_absent_letter_candidates",
                    "decode_ambiguous_letter",
                    "decode_repair_no_boundary",
                    "workspace_branch_cards",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "search_anneal",
                    "act_swap_decoded",
                    "act_set_periodic_key",
                    "search_periodic_polyalphabetic",
                ],
                "mode_warning": (
                    "Segmented dictionary hits can be word islands; require "
                    "coherent clauses before local repair."
                ),
            },
            "periodic_polyalphabetic": {
                "foreground_tools": [
                    "observe_periodic_ic",
                    "observe_kasiski",
                    "observe_phase_frequency",
                    "observe_periodic_shift_candidates",
                    "search_periodic_polyalphabetic",
                    "search_quagmire3_keyword_alphabet",
                    "decode_show_phases",
                    "act_set_periodic_key",
                    "act_set_periodic_shift",
                    "act_adjust_periodic_shift",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_anchor_word",
                    "act_apply_word_repair",
                    "search_homophonic_anneal",
                    "search_transform_homophonic",
                ],
                "mode_warning": (
                    "Do not use single-symbol substitution repairs on this "
                    "mode unless explicitly abandoning the periodic model."
                ),
            },
            "polyalphabetic_vigenere": {
                "foreground_tools": [
                    "observe_periodic_ic",
                    "observe_kasiski",
                    "observe_phase_frequency",
                    "observe_periodic_shift_candidates",
                    "search_periodic_polyalphabetic",
                    "search_quagmire3_keyword_alphabet",
                    "decode_show_phases",
                    "act_set_periodic_key",
                    "act_set_periodic_shift",
                    "act_adjust_periodic_shift",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_homophonic_anneal",
                ],
                "mode_warning": (
                    "Treat this as periodic key recovery, not a static "
                    "substitution mapping."
                ),
            },
            "transposition_homophonic": {
                "foreground_tools": [
                    "observe_transform_suspicion",
                    "search_transform_candidates",
                    "search_transform_homophonic",
                    "search_review_transform_finalists",
                    "act_install_transform_finalists",
                    "act_rate_transform_finalist",
                    "act_apply_transform_pipeline",
                    "search_homophonic_anneal",
                    "workspace_branch_cards",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_periodic_polyalphabetic",
                ],
                "mode_warning": (
                    "Do not repair local words until a transformed branch "
                    "contains coherent text, not just isolated words."
                ),
            },
            "quagmire3": {
                "foreground_tools": [
                    "observe_periodic_ic",
                    "observe_kasiski",
                    "observe_phase_frequency",
                    "search_periodic_polyalphabetic",
                    "search_quagmire3_keyword_alphabet",
                    "workspace_branch_cards",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_homophonic_anneal",
                    "search_transform_homophonic",
                ],
                "mode_warning": (
                    "This is a keyed-tableau periodic hypothesis. Clear "
                    "ordinary Vigenere first, then test Quagmire III. "
                    "Supplying context keywords should be labeled as seeded, "
                    "not blind."
                ),
            },
            "keyed_tableau_polyalphabetic": {
                "foreground_tools": [
                    "observe_periodic_ic",
                    "observe_kasiski",
                    "observe_phase_frequency",
                    "search_periodic_polyalphabetic",
                    "search_quagmire3_keyword_alphabet",
                    "workspace_branch_cards",
                    "score_panel",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_homophonic_anneal",
                    "search_transform_homophonic",
                ],
                "mode_warning": (
                    "This mode covers structured keyed tableaux. Quagmire III "
                    "is the first implemented child hypothesis; broader "
                    "Quagmire/tableau search remains experimental."
                ),
            },
            "transposition": {
                "foreground_tools": [
                    "observe_transform_suspicion",
                    "search_pure_transposition",
                    "search_review_pure_transposition_finalists",
                    "act_rate_transform_finalist",
                    "act_install_pure_transposition_finalists",
                    "search_transform_candidates",
                    "act_apply_transform_pipeline",
                    "workspace_branch_cards",
                    "workspace_update_hypothesis",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "act_set_periodic_key",
                ],
                "mode_warning": "Focus on order correction before key repair.",
            },
            "fractionation_transposition": {
                "foreground_tools": [
                    "observe_cipher_shape",
                    "observe_cipher_id",
                    "meta_request_tool",
                    "workspace_update_hypothesis",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_anneal",
                ],
                "mode_warning": (
                    "Current support is diagnostic/capability-gap oriented; "
                    "request missing Polybius/fractionation tools if evidence "
                    "supports the mode."
                ),
            },
            "playfair": {
                "foreground_tools": [
                    "observe_cipher_shape",
                    "observe_cipher_id",
                    "meta_request_tool",
                    "workspace_update_hypothesis",
                ],
                "discouraged_tools": [
                    "act_set_mapping",
                    "act_bulk_set",
                    "act_apply_word_repair",
                    "search_anneal",
                ],
                "mode_warning": (
                    "Current support is diagnostic/capability-gap oriented; "
                    "single-symbol substitution repairs are the wrong key model."
                ),
            },
        }
        entry = menus.get(mode, {
            "foreground_tools": [
                "observe_cipher_id",
                "observe_cipher_shape",
                "workspace_create_hypothesis_branch",
                "workspace_hypothesis_next_steps",
            ],
            "discouraged_tools": [],
            "mode_warning": (
                "No focused menu exists for this mode yet. Use diagnostics, "
                "record evidence, and switch hypotheses if needed."
            ),
        })
        return {
            "always_available": always_available,
            "foreground_tools": entry["foreground_tools"],
            "escape_tools": escape_tools,
            "discouraged_tools": entry["discouraged_tools"],
            "mode_warning": entry["mode_warning"],
            "note": (
                "This is a soft foreground menu, not hard gating. Tools may "
                "still be callable, but discouraged tools usually indicate a "
                "mode mismatch or premature local repair."
            ),
        }

    def _mode_scoped_suggestions(
        self,
        branch_name: str,
        fallback: list[str],
    ) -> list[str]:
        """Keep guardrail suggestions inside the branch's active cipher mode."""
        if not self.workspace.has_branch(branch_name):
            return fallback
        branch = self.workspace.get_branch(branch_name)
        mode = str(branch.metadata.get("cipher_mode") or "").strip()
        if mode in {
            "periodic_polyalphabetic",
            "polyalphabetic_vigenere",
            "keyed_tableau_polyalphabetic",
            "quagmire3",
        }:
            tools = [
                "workspace_hypothesis_next_steps",
                "observe_periodic_ic",
                "observe_kasiski",
                "observe_phase_frequency",
                "search_periodic_polyalphabetic",
                "search_quagmire3_keyword_alphabet",
                "workspace_branch_cards",
            ]
            return [tool for tool in tools if tool]
        if mode in {"transposition", "transposition_homophonic"}:
            return [
                "workspace_hypothesis_next_steps",
                "observe_transform_suspicion",
                "search_pure_transposition",
                "search_review_pure_transposition_finalists",
                "act_rate_transform_finalist",
                "act_install_pure_transposition_finalists",
                "search_transform_candidates",
                "search_transform_homophonic",
                "search_review_transform_finalists",
                "act_install_transform_finalists",
                "workspace_branch_cards",
            ]
        if mode in {"monoalphabetic_substitution", "simple_substitution", "homophonic_substitution"}:
            return fallback
        return fallback

    def _tool_tried_for_branch(self, tool_name: str, branch_name: str) -> bool:
        for call in self.call_log:
            if call.tool_name != tool_name:
                continue
            args = call.arguments or {}
            if tool_name == "search_quagmire3_keyword_alphabet" and bool(args.get("estimate_only", False)):
                continue
            called_branch = args.get("branch")
            if called_branch is None or called_branch == branch_name:
                return True
        return False

    def _quagmire_budget_class(self, nominal_proposals: int | float | None) -> str:
        if nominal_proposals is None:
            return "unknown"
        try:
            proposals = int(nominal_proposals)
        except (TypeError, ValueError):
            return "unknown"
        if proposals >= 50_000_000:
            return "broad"
        if proposals >= 1_000_000:
            return "moderate"
        return "diagnostic"

    def _quagmire_budget_rank(self, budget_class: str) -> int:
        return {
            "unknown": 0,
            "diagnostic": 1,
            "moderate": 2,
            "broad": 3,
        }.get(budget_class, 0)

    def _quagmire_search_status(self, branch_name: str | None = None) -> dict[str, Any]:
        best: dict[str, Any] | None = None
        if branch_name is not None and self.workspace.has_branch(branch_name):
            metadata = self.workspace.get_branch(branch_name).metadata
            search_metadata = metadata.get("search_metadata")
            if isinstance(search_metadata, dict):
                nominal = search_metadata.get("nominal_proposals")
                budget_class = str(
                    search_metadata.get("budget_class")
                    or self._quagmire_budget_class(nominal)
                )
                best = {
                    "iteration": None,
                    "branch": branch_name,
                    "status": "completed",
                    "engine": search_metadata.get("engine"),
                    "budget_class": budget_class,
                    "nominal_proposals": nominal,
                    "source": "branch_search_metadata",
                    "best_candidate_preview": metadata.get("decoded_text", "")[:240],
                }
        for name in self.workspace.branch_names():
            metadata = self.workspace.get_branch(name).metadata
            search_metadata = metadata.get("search_metadata")
            if not isinstance(search_metadata, dict):
                continue
            nominal = search_metadata.get("nominal_proposals")
            budget_class = str(
                search_metadata.get("budget_class")
                or self._quagmire_budget_class(nominal)
            )
            entry = {
                "iteration": None,
                "branch": name,
                "status": "completed",
                "engine": search_metadata.get("engine"),
                "budget_class": budget_class,
                "nominal_proposals": nominal,
                "source": "branch_search_metadata",
                "best_candidate_preview": metadata.get("decoded_text", "")[:240],
            }
            if (
                best is None
                or self._quagmire_budget_rank(budget_class)
                > self._quagmire_budget_rank(str(best.get("budget_class") or "unknown"))
            ):
                best = entry
        for call in self.call_log:
            if call.tool_name != "search_quagmire3_keyword_alphabet":
                continue
            args = call.arguments or {}
            if bool(args.get("estimate_only", False)):
                continue
            try:
                payload = json.loads(call.result or "{}")
            except json.JSONDecodeError:
                payload = {}
            nominal = payload.get("nominal_proposals")
            if nominal is None:
                nominal = payload.get("budget_estimate", {}).get("nominal_proposals")
            budget_class = str(payload.get("budget_class") or self._quagmire_budget_class(nominal))
            entry = {
                "iteration": call.iteration,
                "branch": args.get("branch"),
                "status": payload.get("status"),
                "engine": payload.get("engine"),
                "budget_class": budget_class,
                "nominal_proposals": nominal,
                "source": "tool_call",
                "best_candidate_preview": (
                    (payload.get("top_candidates") or [{}])[0].get("preview")
                    if isinstance(payload.get("top_candidates"), list) and payload.get("top_candidates")
                    else None
                ),
            }
            if (
                best is None
                or self._quagmire_budget_rank(budget_class)
                > self._quagmire_budget_rank(str(best.get("budget_class") or "unknown"))
            ):
                best = entry
        if best is None:
            return {
                "seen": False,
                "sufficient_to_reject": False,
                "required_minimum_budget_class": "moderate",
            }
        budget_class = str(best.get("budget_class") or "unknown")
        return {
            "seen": True,
            "sufficient_to_reject": (
                self._quagmire_budget_rank(budget_class)
                >= self._quagmire_budget_rank("moderate")
            ),
            "required_minimum_budget_class": "moderate",
            **best,
        }

    def _quagmire_budget_guidance(self, mode: str) -> dict[str, Any] | None:
        if mode not in {"periodic_polyalphabetic", "polyalphabetic_vigenere", "quagmire3", "keyed_tableau_polyalphabetic"}:
            return None
        return {
            "tool": "search_quagmire3_keyword_alphabet",
            "principle": (
                "Size Quagmire III search explicitly. First call with "
                "estimate_only=true, using plausible cycleword lengths from "
                "periodic IC/Kasiski/phase evidence. Then run the smallest "
                "budget that can plausibly test the hypothesis."
            ),
            "profiles": [
                {
                    "name": "diagnostic",
                    "purpose": "Quick smoke to verify tool path and candidate shape.",
                    "suggested_args": {
                        "engine": "rust_shotgun",
                        "estimate_only": True,
                        "keyword_lengths": [7],
                        "cycleword_lengths": [8],
                        "hillclimbs": 1000,
                        "restarts": 100,
                        "threads": 0,
                    },
                },
                {
                    "name": "moderate",
                    "purpose": "First serious K2-sized run once period/length evidence is plausible.",
                    "suggested_args": {
                        "engine": "rust_shotgun",
                        "estimate_only": True,
                        "keyword_lengths": [7],
                        "cycleword_lengths": [8],
                        "hillclimbs": 10000,
                        "restarts": 500,
                        "threads": 0,
                        "top_n": 10,
                        "install_top_n": 3,
                    },
                },
                {
                    "name": "broader_length_sweep",
                    "purpose": "Use when keyword/cycleword lengths are uncertain and the user accepts a larger run.",
                    "suggested_args": {
                        "engine": "rust_shotgun",
                        "estimate_only": True,
                        "keyword_lengths": [5, 6, 7, 8, 9],
                        "cycleword_lengths": [6, 7, 8, 9, 10],
                        "hillclimbs": 5000,
                        "restarts": 250,
                        "threads": 0,
                    },
                },
            ],
            "seeded_keyword_warning": (
                "Do not pass likely title/source words as initial_keywords "
                "unless intentionally running a context-seeded experiment. "
                "If initial_keywords are supplied, label the result as seeded "
                "rather than blind key recovery."
            ),
        }

    def _pending_required_tools_for_branch(self, branch_name: str) -> dict[str, Any]:
        branch = self.workspace.get_branch(branch_name)
        metadata = branch.metadata
        mode = str(metadata.get("cipher_mode") or "unknown").strip()
        required = list(metadata.get("required_tools_before_rejection") or [])
        required.extend(
            tool for tool in self._polyalphabetic_required_tools_for_mode(mode)
            if tool not in required
        )

        context_prior = metadata.get("context_mode_prior")
        statistical_prior = None
        family_coverage_debt = None
        if mode in {"periodic_polyalphabetic", "polyalphabetic_vigenere"}:
            search_tried = self._tool_tried_for_branch(
                "search_periodic_polyalphabetic",
                branch_name,
            )
            quagmire_tried = self._tool_tried_for_branch(
                "search_quagmire3_keyword_alphabet",
                branch_name,
            )
            if search_tried and not quagmire_tried:
                statistical_prior = self._polyalphabetic_signal_prior()
                if context_prior or statistical_prior:
                    if "search_quagmire3_keyword_alphabet" not in required:
                        required.append("search_quagmire3_keyword_alphabet")
                    family_coverage_debt = (
                        "Ordinary A-Z Vigenere/Beaufort/Gronsfeld has been "
                        "tested, but keyed-tableau/Quagmire has not. Do not "
                        "treat the whole polyalphabetic family as rejected yet."
                    )

        quagmire_status = self._quagmire_search_status(branch_name)
        pending = []
        for tool in required:
            if tool == "search_quagmire3_keyword_alphabet":
                if not quagmire_status["sufficient_to_reject"]:
                    pending.append(tool)
                continue
            if not self._tool_tried_for_branch(tool, branch_name):
                pending.append(tool)
        return {
            "required_tools_before_rejection": required,
            "pending_required_tools": pending,
            "context_mode_prior": context_prior,
            "statistical_family_prior": statistical_prior,
            "family_coverage_debt": family_coverage_debt,
            "quagmire_search_status": (
                quagmire_status
                if "search_quagmire3_keyword_alphabet" in required
                else None
            ),
        }

    def _hypothesis_next_steps(self, branch_name: str) -> dict[str, Any]:
        branch = self.workspace.get_branch(branch_name)
        mode = str(branch.metadata.get("cipher_mode") or "unknown")
        playbook = self._mode_playbook(mode)
        steps = []
        for index, step in enumerate(playbook, 1):
            tried = self._tool_tried_for_branch(step["tool"], branch_name)
            steps.append({
                "index": index,
                "tool": step["tool"],
                "purpose": step["purpose"],
                "status": "tried" if tried else "pending",
            })
        next_step = next((step for step in steps if step["status"] == "pending"), None)
        required = self._pending_required_tools_for_branch(branch_name)
        if required["pending_required_tools"]:
            required_tool = required["pending_required_tools"][0]
            next_step = next(
                (step for step in steps if step["tool"] == required_tool),
                {
                    "index": None,
                    "tool": required_tool,
                    "purpose": "Required before rejecting this cipher-family hypothesis.",
                    "status": "pending_required",
                },
            )
        return {
            "branch": branch_name,
            "cipher_mode": mode,
            "mode_status": branch.metadata.get("mode_status", "active"),
            "mode_confidence": branch.metadata.get("mode_confidence"),
            "mode_evidence": branch.metadata.get("mode_evidence") or branch.metadata.get("hypothesis_notes"),
            "mode_counter_evidence": branch.metadata.get("mode_counter_evidence"),
            "tool_menu": self._mode_tool_menu(mode),
            "playbook": steps,
            "next_step": next_step,
            "already_tried_tools": [step["tool"] for step in steps if step["status"] == "tried"],
            "pending_tools": [step["tool"] for step in steps if step["status"] == "pending"],
            "quagmire_budget_guidance": self._quagmire_budget_guidance(mode),
            **required,
        }

    _SUBSTITUTION_REPAIR_COMPATIBLE_MODES = {
        "",
        "unknown",
        "monoalphabetic_substitution",
        "simple_substitution",
        "homophonic_substitution",
    }

    def _mode_mismatch_repair_block(
        self,
        branch_name: str,
        tool_name: str,
        *,
        allow_override: bool = False,
    ) -> dict[str, Any] | None:
        """Block accidental substitution-style edits on non-substitution modes."""
        branch = self.workspace.get_branch(branch_name)
        mode = str(branch.metadata.get("cipher_mode") or "").strip()
        key_type = str(branch.metadata.get("key_type") or "").strip()
        if allow_override:
            return None
        if mode in self._SUBSTITUTION_REPAIR_COMPATIBLE_MODES and key_type != "PeriodicShiftKey":
            return None
        if mode == "periodic_polyalphabetic" or key_type == "PeriodicShiftKey":
            next_tools = [
                "decode_show_phases",
                "observe_phase_frequency",
                "act_set_periodic_shift",
                "act_adjust_periodic_shift",
                "search_periodic_polyalphabetic",
                "workspace_update_hypothesis",
            ]
            note = (
                "This branch is a periodic polyalphabetic hypothesis. A "
                "cipher-symbol -> plaintext-letter mapping edits the wrong key "
                "model. Use phase/key tools instead, or override only if you "
                "are intentionally abandoning the periodic hypothesis."
            )
        elif mode in {"transposition", "transposition_homophonic"}:
            next_tools = [
                "observe_transform_suspicion",
                "search_pure_transposition",
                "search_transform_candidates",
                "search_transform_homophonic",
                "act_apply_transform_pipeline",
                "workspace_update_hypothesis",
            ]
            note = (
                "This branch is an order/transform hypothesis. Local letter "
                "repair is premature unless the transformed reading is already "
                "coherent."
            )
        elif mode in {"fractionation_transposition", "playfair", "polygraphic_substitution"}:
            next_tools = [
                "observe_cipher_shape",
                "workspace_update_hypothesis",
            ]
            note = (
                "This branch is not modeled as single-symbol substitution. "
                "Use mode-specific diagnostics before changing individual "
                "symbol mappings."
            )
        else:
            next_tools = [
                self._recommended_tool_for_cipher_mode(mode),
                "workspace_update_hypothesis",
            ]
            note = (
                "This branch carries a non-substitution cipher-mode hypothesis. "
                "Use mode-appropriate tools or explicitly override if you are "
                "changing hypotheses."
            )
        return {
            "status": "blocked",
            "accepted": False,
            "branch": branch_name,
            "attempted_tool": tool_name,
            "reason": "mode_mismatch_substitution_repair_blocked",
            "cipher_mode": mode or "unknown",
            "key_type": key_type or None,
            "suggested_next_tools": next_tools,
            "override_argument": "allow_mode_mismatch_repair=true",
            "note": note,
        }

    def _branch_decoded_text(self, branch_name: str) -> str:
        branch = self.workspace.get_branch(branch_name)
        decoded = branch.metadata.get("decoded_text")
        if isinstance(decoded, str) and decoded.strip():
            return decoded
        return self.workspace.apply_key(branch_name)

    def _compute_quick_scores(self, branch_name: str) -> dict[str, float | None]:
        """Return (dict_rate, quad) for a branch, fast. Used for score_delta
        on mutation tools so the agent can immediately see if a change helped.
        """
        from analysis.segment import segment_text
        decrypted = self._branch_decoded_text(branch_name)
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
        branch_state = ws.get_branch(branch_name)
        metadata_text = branch_state.metadata.get("decoded_text")
        if isinstance(metadata_text, str) and metadata_text.strip():
            if any(ch.isspace() for ch in metadata_text.strip()):
                words = metadata_text.split()
            elif branch_state.word_spans is not None:
                words = [
                    metadata_text[start:end]
                    for start, end in ws.effective_word_spans(branch_name)
                ]
            else:
                words = [
                    metadata_text[i : i + 80]
                    for i in range(0, len(metadata_text), 80)
                ]
            return words[:max_words] if max_words is not None else words
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

    def _has_escalated_transform_search(self) -> bool:
        """Return true once the run has tried more than a tiny transform probe."""
        for call in self.call_log:
            args = call.arguments or {}
            if call.tool_name == "search_transform_homophonic":
                profile = str(args.get("profile") or "small").lower()
                if profile in {"medium", "wide"}:
                    return True
                if bool(args.get("include_program_search")):
                    return True
                generated = args.get("max_generated_candidates")
                if generated is not None:
                    try:
                        if int(generated) >= 10000:
                            return True
                    except (TypeError, ValueError):
                        pass
            if call.tool_name == "search_transform_candidates":
                breadth = str(args.get("breadth") or "").lower()
                if breadth in {"broad", "wide"}:
                    return True
                if bool(args.get("include_program_search")):
                    return True
            if call.tool_name == "search_pure_transposition":
                profile = str(args.get("profile") or "wide").lower()
                if profile in {"medium", "wide"}:
                    return True
                max_candidates = args.get("max_candidates")
                if max_candidates is not None:
                    try:
                        if int(max_candidates) >= 10000:
                            return True
                    except (TypeError, ValueError):
                        pass
        return False

    def _repair_policy_blocks_word_repair(self, basin: dict[str, Any]) -> bool:
        return basin.get("repair_policy") in {
            "search_before_local_repair",
            "search_or_map_before_repair",
        }

    def _longest_dictionary_run(self, words: list[str]) -> dict[str, Any]:
        best: list[str] = []
        current: list[str] = []
        for word in words:
            if word in self.word_set:
                current.append(word)
                if len("".join(current)) > len("".join(best)):
                    best = list(current)
            else:
                current = []
        return {
            "word_count": len(best),
            "char_count": len("".join(best)),
            "sample": best[:12],
        }

    def _branch_basin_status(self, branch_name: str) -> dict[str, Any]:
        """Classify whether a branch is suitable for local reading repair.

        This is deliberately conservative. It is not trying to prove semantic
        coherence; it flags the common failure mode where a no-boundary
        homophonic/transposition candidate has been segmented into many short
        English-looking islands while the language-model score still says the
        continuous text is poor.
        """
        scores = self._compute_quick_scores(branch_name)
        dict_rate = scores.get("dict_rate")
        quad = scores.get("quad")
        branch = self.workspace.get_branch(branch_name)
        words = [
            self._reading_char_stream(word)
            for word in self._decoded_words(branch_name, max_words=None)
            if self._reading_char_stream(word)
        ]
        scored = [word for word in words if any(ch.isalpha() for ch in word)]
        hits = [word for word in scored if word in self.word_set]
        long_hits = [word for word in hits if len(word) >= 5]
        short_hit_fraction = (
            sum(1 for word in hits if len(word) <= 3) / len(hits)
            if hits else 0.0
        )
        avg_word_len = (
            sum(len(word) for word in scored) / len(scored)
            if scored else 0.0
        )
        longest_run = self._longest_dictionary_run(scored)
        original_no_boundary = len(self.workspace.cipher_text.words) <= 1
        no_boundary_family = original_no_boundary and self._is_homophonic_cipher()
        quad_poor = not isinstance(quad, (int, float)) or quad < -4.05
        dict_numeric = float(dict_rate) if isinstance(dict_rate, (int, float)) else 0.0
        coverage = (
            len(branch.key) / self._alpha().size
            if self._alpha().size else 0.0
        )

        status = "unknown"
        repair_policy = "normal"
        reason = (
            "Not enough evidence to classify semantic basin quality. Use "
            "decoded text and branch comparison."
        )
        suggested_next_tools: list[str] = ["decode_show", "score_panel"]

        if coverage < 0.25:
            status = "garbage"
            repair_policy = "search_or_map_before_repair"
            reason = "Very little of the cipher alphabet is mapped."
            suggested_next_tools = [
                "search_automated_solver",
                "search_homophonic_anneal",
                "observe_transform_suspicion",
            ]
        elif no_boundary_family and dict_numeric >= 0.30 and quad_poor:
            status = "word_islands_only"
            repair_policy = "search_before_local_repair"
            reason = (
                "This no-boundary homophonic branch has dictionary-looking "
                "segments but poor continuous-language score. That pattern is "
                "typical of word islands in a bad basin, not a near-solved "
                "plaintext. Only do local repairs if you can paraphrase a "
                "coherent clause."
            )
            suggested_next_tools = [
                "observe_transform_suspicion",
                (
                    "search_transform_homophonic(profile='medium', "
                    "include_program_search=true)"
                    if not self._has_escalated_transform_search()
                    else "search_transform_homophonic(profile='wide', include_program_search=true)"
                ),
                "search_automated_solver",
                "workspace_branch_cards",
            ]
        elif no_boundary_family and dict_numeric < 0.18 and quad_poor:
            status = "garbage"
            repair_policy = "search_before_local_repair"
            reason = (
                "This no-boundary homophonic branch has weak dictionary and "
                "quadgram signals. It is not ready for word-level repair."
            )
            suggested_next_tools = [
                "observe_transform_suspicion",
                "search_transform_homophonic(profile='medium', include_program_search=true)",
                "search_automated_solver",
            ]
        elif (
            isinstance(quad, (int, float)) and quad >= -4.05
            and (longest_run["char_count"] >= 14 or len(long_hits) >= 3)
        ):
            status = "coherent_basin"
            repair_policy = "local_repair_ok"
            reason = (
                "Internal language score and dictionary runs are consistent "
                "with a branch that may be close enough for reading-driven "
                "repair."
            )
            suggested_next_tools = [
                "decode_plan_word_repair_menu",
                "act_apply_word_repair",
                "decode_validate_reading_repair",
            ]

        return {
            "status": status,
            "repair_policy": repair_policy,
            "reason": reason,
            "original_no_boundary": original_no_boundary,
            "homophonic_like": self._is_homophonic_cipher(),
            "mapping_coverage": round(coverage, 4),
            "dictionary_rate": dict_rate,
            "quad": quad,
            "word_count": len(scored),
            "recognized_word_count": len(hits),
            "long_recognized_word_count": len(long_hits),
            "average_word_length": round(avg_word_len, 2),
            "short_hit_fraction": round(short_hit_fraction, 4),
            "longest_dictionary_run": longest_run,
            "escalated_transform_search_seen": self._has_escalated_transform_search(),
            "suggested_next_tools": self._mode_scoped_suggestions(
                branch_name,
                suggested_next_tools,
            ),
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
            "metadata": dict(branch.metadata),
            "transform_finalist": branch.metadata.get("transform_finalist"),
            "protected_baseline": branch_name == "automated_preflight",
            "scores": self._compute_quick_scores(branch_name),
            "basin": self._branch_basin_status(branch_name),
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

    def _word_boundary_declaration_block(
        self,
        branch_name: str,
        *,
        confidence: float,
        forced_partial: bool,
    ) -> dict[str, Any] | None:
        """Require one segmentation pass before declaring a solved stream.

        Solver-backed polyalphabetic branches can carry a plaintext stream in
        metadata instead of in the substitution key. When that stream is
        continuous text, a declaration is usually premature until the agent has
        installed an explicit word-boundary overlay for the artifact and final
        scoring/reporting.
        """
        if forced_partial or confidence < 0.75:
            return None
        branch = self.workspace.get_branch(branch_name)
        metadata_text = branch.metadata.get("decoded_text")
        if not isinstance(metadata_text, str) or not metadata_text.strip():
            return None
        decoded = metadata_text.strip()
        if any(ch.isspace() for ch in decoded):
            return None
        if branch.word_spans is not None:
            return None
        if len(sig.normalize_for_scoring(decoded)) < 80:
            return None

        mode = str(branch.metadata.get("cipher_mode") or "").strip()
        key_type = str(branch.metadata.get("key_type") or "").strip()
        if mode not in self._POLY_MODE_NAMES and key_type not in {
            "PeriodicShiftKey",
            "QuagmireKey",
        }:
            return None

        return {
            "status": "blocked",
            "accepted": False,
            "branch": branch_name,
            "reason": "word_boundary_pass_required",
            "cipher_mode": mode or None,
            "key_type": key_type or None,
            "decoded_stream_preview": decoded[:220],
            "note": (
                "This branch has a high-confidence decoded stream but no "
                "word-boundary overlay. Before declaring, do a boundary pass "
                "so the final artifact and word scoring reflect the readable "
                "plaintext. If the letters are already correct, propose the "
                "same letters with spaces and call act_resegment_by_reading."
            ),
            "suggested_next_tools": [
                "decode_repair_no_boundary",
                "decode_validate_reading_repair",
                "act_resegment_by_reading",
                "workspace_branch_cards",
                "meta_declare_solution",
            ],
        }

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
            "search_pure_transposition",
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

    def _should_guard_more_work_declaration(
        self,
        further_iterations_helpful: bool,
        forced_partial: bool,
    ) -> bool:
        if not further_iterations_helpful:
            return False
        if self.max_iterations is None:
            return False
        if self.max_iterations is not None and self._current_iteration >= self.max_iterations:
            return False
        return True

    def _should_guard_premature_partial_declaration(
        self,
        confidence: float,
        forced_partial: bool,
    ) -> bool:
        if not forced_partial:
            return False
        if self.max_iterations is None:
            return False
        if self._current_iteration >= self.max_iterations:
            return False
        final_stretch = max(1, int(self.max_iterations * 0.8))
        if self._current_iteration >= final_stretch:
            return False
        return confidence < 0.50

    def _unrated_transform_finalist_metadata(self, branch_name: str) -> dict[str, Any] | None:
        if self.max_iterations is not None and self._current_iteration >= self.max_iterations:
            return None
        branch = self.workspace.get_branch(branch_name)
        metadata = branch.metadata.get("transform_finalist")
        if not isinstance(metadata, dict):
            return None
        if metadata.get("agent_readability_score") is not None:
            return None
        return metadata

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

    def _tool_workspace_create_hypothesis_branch(self, args: dict) -> Any:
        new_name = args["new_name"]
        from_branch = args.get("from_branch", "main")
        mode = str(args["cipher_mode"]).strip()
        rationale = str(args["rationale"]).strip()
        confidence = str(args.get("mode_confidence") or "medium").strip().lower()
        evidence_source = str(args.get("evidence_source") or "agent_inference").strip()
        branch = self.workspace.fork(new_name, from_branch=from_branch)
        branch.metadata.update({
            "cipher_mode": mode,
            "mode_confidence": confidence,
            "mode_status": "active",
            "mode_evidence": rationale,
            "hypothesis_notes": rationale,
            "evidence_source": evidence_source,
        })
        context_prior = None
        if evidence_source == "benchmark_context" and mode in self._POLY_MODE_NAMES:
            required = self._polyalphabetic_required_tools_for_mode(mode)
            if mode in {"keyed_tableau_polyalphabetic", "quagmire3"}:
                required = ["search_quagmire3_keyword_alphabet"]
            context_prior = {
                "prior": (
                    "keyed_tableau_polyalphabetic"
                    if mode in {"keyed_tableau_polyalphabetic", "quagmire3"}
                    else mode
                ),
                "confidence": "agent_declared_context_supported",
                "evidence": [{
                    "branch": new_name,
                    "cipher_mode": mode,
                    "evidence_source": evidence_source,
                    "mode_evidence": rationale,
                }],
                "required_tools_before_rejection": required,
                "note": (
                    "The agent read exposed benchmark context and recorded this "
                    "as a controlling cipher-family assumption."
                ),
            }
            branch.metadata.update({
                "context_supported_mode": True,
                "context_mode_prior": context_prior,
                "context_assumption_note": (
                    "Agent-declared from exposed benchmark context. If this "
                    "assumption is wrong, off-family searches may be suppressed."
                ),
                "required_tools_before_rejection": required,
                "next_recommended_action": (
                    "Context-supported cipher family recorded. Use the "
                    "mode-specific playbook and run required tools before "
                    "rejecting or leaving this family."
                ),
            })
        self.workspace.tag(new_name, "hypothesis")
        self.workspace.tag(new_name, f"mode:{mode}")
        return {
            "status": "ok",
            "created": new_name,
            "parent": from_branch,
            "cipher_mode": mode,
            "mode_confidence": confidence,
            "mode_status": "active",
            "rationale": rationale,
            "evidence_source": evidence_source,
            "context_mode_prior": context_prior,
            "required_tools_before_rejection": (
                context_prior["required_tools_before_rejection"]
                if context_prior else []
            ),
            "note": (
                "This branch records a cipher-type hypothesis. Use mode-"
                "appropriate tools on it, and reject/supersede it if the "
                "decoded text or diagnostics do not support the hypothesis. "
                "If you set evidence_source='benchmark_context', the executor "
                "treats that as your explicit context assumption and requires "
                "mode-appropriate tools before rejection."
            ),
        }

    def _tool_workspace_reject_hypothesis(self, args: dict) -> Any:
        branch_name = args["branch"]
        branch = self.workspace.get_branch(branch_name)
        status = str(args.get("status") or "rejected").strip().lower()
        if status not in {"rejected", "superseded"}:
            return {"error": "status must be rejected or superseded"}
        reason = str(args["reason"]).strip()
        pending = self._pending_required_tools_for_branch(branch_name)
        if (
            pending["pending_required_tools"]
            and not bool(args.get("acknowledge_pending_required_tools", False))
        ):
            return {
                "status": "blocked",
                "branch": branch_name,
                "reason": "pending_required_tools_before_rejection",
                "cipher_mode": branch.metadata.get("cipher_mode"),
                "attempted_status": status,
                **pending,
                "note": (
                    "Rejection blocked. This hypothesis still has required "
                    "family-level work pending. Plain Vigenere failure does "
                    "not reject keyed-tableau/Kryptos-style polyalphabetic "
                    "ciphers. Run the pending tools, or if this is truly "
                    "impossible call workspace_reject_hypothesis again with "
                    "acknowledge_pending_required_tools=true and explain why."
                ),
            }
        branch.metadata["mode_status"] = status
        branch.metadata["rejection_reason"] = reason
        self.workspace.tag(branch_name, status)
        return {
            "status": "ok",
            "branch": branch_name,
            "mode_status": status,
            "cipher_mode": branch.metadata.get("cipher_mode"),
            "reason": reason,
        }

    def _tool_workspace_update_hypothesis(self, args: dict) -> Any:
        branch_name = args["branch"]
        branch = self.workspace.get_branch(branch_name)
        status = str(args.get("mode_status") or "active").strip().lower()
        if status not in {"active", "paused", "rejected", "superseded"}:
            return {"error": "mode_status must be active, paused, rejected, or superseded"}
        if args.get("cipher_mode"):
            mode = str(args["cipher_mode"]).strip()
            branch.metadata["cipher_mode"] = mode
            self.workspace.tag(branch_name, f"mode:{mode}")
        if args.get("evidence_source"):
            evidence_source = str(args["evidence_source"]).strip()
            branch.metadata["evidence_source"] = evidence_source
            mode = str(branch.metadata.get("cipher_mode") or "").strip()
            if evidence_source == "benchmark_context" and mode in self._POLY_MODE_NAMES:
                required = self._polyalphabetic_required_tools_for_mode(mode)
                if mode in {"keyed_tableau_polyalphabetic", "quagmire3"}:
                    required = ["search_quagmire3_keyword_alphabet"]
                context_prior = {
                    "prior": (
                        "keyed_tableau_polyalphabetic"
                        if mode in {"keyed_tableau_polyalphabetic", "quagmire3"}
                        else mode
                    ),
                    "confidence": "agent_declared_context_supported",
                    "evidence": [{
                        "branch": branch_name,
                        "cipher_mode": mode,
                        "evidence_source": evidence_source,
                        "mode_evidence": branch.metadata.get("mode_evidence"),
                    }],
                    "required_tools_before_rejection": required,
                    "note": (
                        "The agent read exposed benchmark context and recorded "
                        "this as a controlling cipher-family assumption."
                    ),
                }
                branch.metadata.update({
                    "context_supported_mode": True,
                    "context_mode_prior": context_prior,
                    "context_assumption_note": (
                        "Agent-declared from exposed benchmark context."
                    ),
                    "required_tools_before_rejection": required,
                })
        if args.get("mode_confidence"):
            branch.metadata["mode_confidence"] = str(args["mode_confidence"]).strip().lower()
        if args.get("evidence"):
            branch.metadata["mode_evidence"] = str(args["evidence"]).strip()
        if args.get("counter_evidence"):
            branch.metadata["mode_counter_evidence"] = str(args["counter_evidence"]).strip()
        if args.get("next_recommended_action"):
            branch.metadata["next_recommended_action"] = str(args["next_recommended_action"]).strip()
        branch.metadata["mode_status"] = status
        self.workspace.tag(branch_name, "hypothesis")
        if status in {"rejected", "superseded"}:
            self.workspace.tag(branch_name, status)
        return {
            "status": "ok",
            "branch": branch_name,
            "cipher_mode": branch.metadata.get("cipher_mode", "unknown"),
            "mode_status": branch.metadata.get("mode_status"),
            "mode_confidence": branch.metadata.get("mode_confidence"),
            "mode_evidence": branch.metadata.get("mode_evidence"),
            "mode_counter_evidence": branch.metadata.get("mode_counter_evidence"),
            "next_recommended_action": branch.metadata.get("next_recommended_action"),
            "evidence_source": branch.metadata.get("evidence_source"),
            "context_supported_mode": branch.metadata.get("context_supported_mode"),
            "required_tools_before_rejection": branch.metadata.get("required_tools_before_rejection", []),
            "note": (
                "Hypothesis metadata updated only; no key, token order, or "
                "decoded text changed. Use workspace_hypothesis_cards to "
                "compare active and rejected modes."
            ),
        }

    def _tool_workspace_hypothesis_cards(self, _args: dict) -> Any:
        cards = []
        for name in self.workspace.branch_names():
            branch = self.workspace.get_branch(name)
            metadata = branch.metadata
            mode = metadata.get("cipher_mode")
            if not mode and "hypothesis" not in branch.tags:
                continue
            card = self._branch_card(name)
            cards.append({
                "branch": name,
                "cipher_mode": mode or "unknown",
                "mode_status": metadata.get("mode_status", "active"),
                "mode_confidence": metadata.get("mode_confidence"),
                "mode_evidence": metadata.get("mode_evidence") or metadata.get("hypothesis_notes"),
                "mode_counter_evidence": metadata.get("mode_counter_evidence"),
                "next_recommended_action": metadata.get("next_recommended_action"),
                "rejection_reason": metadata.get("rejection_reason"),
                "context_supported_mode": metadata.get("context_supported_mode"),
                "context_mode_prior": metadata.get("context_mode_prior"),
                "required_tools_before_rejection": metadata.get("required_tools_before_rejection", []),
                "periodic_key": metadata.get("periodic_key"),
                "periodic_variant": metadata.get("periodic_variant"),
                "scores": card.get("scores"),
                "decoded_excerpt": card.get("decoded_excerpt"),
                "tags": card.get("tags"),
            })
        return {
            "status": "ok",
            "hypotheses": cards,
            "note": (
                "Compare hypotheses by mode evidence and coherent text, not by "
                "raw scores alone. If all active cards are word islands, reject "
                "or supersede the bad basin and try another mode."
            ),
        }

    def _tool_workspace_hypothesis_next_steps(self, args: dict) -> Any:
        requested = args.get("branch")
        if requested:
            branches = [str(requested)]
        else:
            branches = []
            for name in self.workspace.branch_names():
                branch = self.workspace.get_branch(name)
                metadata = branch.metadata
                if not metadata.get("cipher_mode") and "hypothesis" not in branch.tags:
                    continue
                if metadata.get("mode_status", "active") in {"rejected", "superseded"}:
                    continue
                branches.append(name)
        reports = [self._hypothesis_next_steps(name) for name in branches]
        return {
            "status": "ok",
            "branches": branches,
            "reports": reports,
            "note": (
                "Use `next_step` as the default move unless your reading of "
                "the decoded text gives a stronger reason. If the playbook is "
                "exhausted and the text is incoherent, reject or pause the "
                "hypothesis instead of doing local repairs."
            ),
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
                "internal scores, basin status, applied/held repairs, and "
                "orthography risks. If a branch card says `word_islands_only`, "
                "do not spend iterations on local word repairs unless you can "
                "paraphrase a coherent clause; escalate search instead. "
                "Treat `automated_preflight` as a protected no-LLM baseline; "
                "do not discard it in favor of a modernized/classicized edit "
                "unless the edited branch is clearly better in the manuscript "
                "transcription style."
            ),
        }

    def _has_seen_branch_cards(self, branch_name: str | None = None) -> bool:
        min_iteration = 0
        if branch_name and self.workspace.has_branch(branch_name):
            min_iteration = int(self.workspace.get_branch(branch_name).created_iteration or 0)
        for call in self.call_log:
            if call.tool_name != "workspace_branch_cards":
                continue
            if int(getattr(call, "iteration", 0) or 0) < min_iteration:
                continue
            args = call.arguments or {}
            requested_branch = args.get("branch")
            if branch_name and requested_branch not in {None, "", branch_name}:
                continue
            return True
        return False

    def _has_seen_hypothesis_next_steps(self, branch_name: str) -> bool:
        min_iteration = 0
        if self.workspace.has_branch(branch_name):
            min_iteration = int(self.workspace.get_branch(branch_name).created_iteration or 0)
        for call in self.call_log:
            if call.tool_name != "workspace_hypothesis_next_steps":
                continue
            if int(getattr(call, "iteration", 0) or 0) < min_iteration:
                continue
            args = call.arguments or {}
            requested_branch = args.get("branch")
            if requested_branch not in {None, "", branch_name}:
                continue
            return True
        return False

    def _hypothesis_declaration_block(self, branch_name: str) -> dict[str, Any] | None:
        branch = self.workspace.get_branch(branch_name)
        metadata = branch.metadata
        if not metadata.get("cipher_mode") and "hypothesis" not in branch.tags:
            return None
        if self._has_seen_hypothesis_next_steps(branch_name):
            return None
        return {
            "status": "blocked",
            "accepted": False,
            "branch": branch_name,
            "reason": "hypothesis_next_steps_required",
            "cipher_mode": metadata.get("cipher_mode", "unknown"),
            "mode_status": metadata.get("mode_status", "active"),
            "note": (
                "This branch is tagged as a cipher-mode hypothesis. Before "
                "declaring, call workspace_hypothesis_next_steps so the "
                "artifact records the mode-specific playbook, the tools "
                "already tried, and any pending bigger-swing actions."
            ),
            "suggested_next_tools": [
                "workspace_hypothesis_next_steps",
                "workspace_branch_cards",
                "meta_declare_solution",
            ],
        }

    def _family_coverage_declaration_block(
        self,
        branch_name: str,
        *,
        forced_partial: bool,
    ) -> dict[str, Any] | None:
        if forced_partial:
            return None
        if self.max_iterations is not None and self._current_iteration >= self.max_iterations:
            return None

        branch_reports = []
        for name in self.workspace.branch_names():
            branch = self.workspace.get_branch(name)
            metadata = branch.metadata
            if not metadata.get("cipher_mode") and "hypothesis" not in branch.tags:
                continue
            if metadata.get("mode_status", "active") in {"rejected", "superseded"}:
                continue
            pending = self._pending_required_tools_for_branch(name)
            if pending["pending_required_tools"]:
                branch_reports.append({
                    "branch": name,
                    "cipher_mode": metadata.get("cipher_mode"),
                    "mode_evidence": metadata.get("mode_evidence") or metadata.get("hypothesis_notes"),
                    **pending,
                })

        context_prior = self._context_keyed_tableau_prior()
        quagmire_status = self._quagmire_search_status()
        if context_prior and not quagmire_status["sufficient_to_reject"]:
            if not branch_reports:
                branch_reports.append({
                    "branch": None,
                    "cipher_mode": context_prior["prior"],
                    "context_mode_prior": context_prior,
                    "pending_required_tools": ["search_quagmire3_keyword_alphabet"],
                    "required_tools_before_rejection": context_prior["required_tools_before_rejection"],
                    "quagmire_search_status": quagmire_status,
                    "family_coverage_debt": context_prior["note"],
                })

        statistical_prior = self._polyalphabetic_signal_prior()
        if (
            statistical_prior
            and self._has_seen_any_tool({"search_periodic_polyalphabetic"})
            and not quagmire_status["sufficient_to_reject"]
        ):
            if not branch_reports:
                branch_reports.append({
                    "branch": None,
                    "cipher_mode": "polyalphabetic_family",
                    "statistical_family_prior": statistical_prior,
                    "pending_required_tools": ["search_quagmire3_keyword_alphabet"],
                    "required_tools_before_rejection": ["search_quagmire3_keyword_alphabet"],
                    "quagmire_search_status": quagmire_status,
                    "family_coverage_debt": statistical_prior["note"],
                })

        if not branch_reports:
            return None

        pending_tools = sorted({
            tool
            for report in branch_reports
            for tool in report.get("pending_required_tools", [])
        })
        return {
            "status": "blocked",
            "accepted": False,
            "branch": branch_name,
            "reason": "family_coverage_pending",
            "pending_family_coverage": branch_reports,
            "note": (
                "Declaration blocked because a supported cipher-family "
                "hypothesis still has required higher-level work pending. "
                "A bad ordinary Vigenere or word-island branch is not enough "
                "to stop while keyed-tableau/Quagmire remains untested and "
                "there is still iteration budget. Take the pending bigger "
                "swing, then compare branches again."
            ),
            "suggested_next_tools": [
                *pending_tools,
                "workspace_hypothesis_next_steps",
                "workspace_branch_cards",
                "meta_declare_solution",
            ],
        }

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

    def _tool_observe_cipher_id(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        max_period = int(args.get("max_period", 26))
        fp = cipher_id.compute_cipher_fingerprint(
            effective.tokens,
            effective.alphabet.size,
            max_period=max_period,
            language=self.language,
            word_group_count=len(self.workspace.effective_word_spans(branch_name)),
        )
        ranked = sorted(fp.suspicion_scores.items(), key=lambda item: item[1], reverse=True)
        return {
            "branch": branch_name,
            **branch_note,
            "fingerprint": fp.to_dict(),
            "ranked_hypotheses": [
                {
                    "cipher_mode": mode,
                    "score": round(score, 4),
                    "confidence": "high" if score >= 0.65 else "medium" if score >= 0.35 else "low",
                    "recommended_next_tool": self._recommended_tool_for_cipher_mode(mode),
                }
                for mode, score in ranked
            ],
            "context_block": cipher_id.format_fingerprint_for_context(fp),
            "note": (
                "These scores are evidence weights, not probabilities. Use "
                "them to create mode-specific hypothesis branches and choose "
                "diagnostic/search tools."
            ),
        }

    def _tool_observe_cipher_shape(self, args: dict) -> Any:
        from collections import Counter

        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        tokens = self.workspace.effective_tokens(branch_name)
        alpha = self._alpha()
        top_n = max(1, min(int(args.get("top_n", 12)), 50))
        counts = Counter(tokens)
        bigrams = Counter(zip(tokens, tokens[1:]))
        trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
        word_spans = self.workspace.effective_word_spans(branch_name)
        symbols = [alpha.symbol_for(t).upper() for t in tokens]
        coord_like_symbols = [
            sym for sym in sorted(set(symbols))
            if sym in {"1", "2", "3", "4", "5", "6", "A", "B", "C", "D", "E", "F"}
        ]
        repeated_trigrams = [
            {
                "trigram": " ".join(alpha.symbol_for(t) for t in gram),
                "count": count,
            }
            for gram, count in trigrams.most_common(top_n)
            if count > 1
        ]
        return {
            "branch": branch_name,
            **branch_note,
            "token_count": len(tokens),
            "unique_symbols": len(counts),
            "alphabet_size": alpha.size,
            "word_group_count": len(word_spans),
            "single_word_or_no_boundary": len(word_spans) <= 1,
            "even_token_count": len(tokens) % 2 == 0,
            "divisible_by_3": len(tokens) % 3 == 0,
            "top_symbols": [
                {
                    "symbol": alpha.symbol_for(token_id),
                    "count": count,
                    "pct": round(count / len(tokens) * 100, 2) if tokens else 0.0,
                }
                for token_id, count in counts.most_common(top_n)
            ],
            "top_bigrams": [
                {
                    "bigram": " ".join(alpha.symbol_for(t) for t in gram),
                    "count": count,
                }
                for gram, count in bigrams.most_common(top_n)
            ],
            "repeated_trigrams": repeated_trigrams,
            "coordinate_like_symbols": coord_like_symbols,
            "coordinate_like_fraction": round(
                sum(1 for sym in symbols if sym in coord_like_symbols) / len(symbols),
                4,
            ) if symbols else 0.0,
            "recommended_next_tools": [
                "observe_cipher_id",
                "observe_periodic_ic",
                "observe_transform_suspicion",
            ],
            "note": (
                "Use this as a shape card for unknown ciphers. Even length and "
                "coordinate-like symbols support Polybius/fractionation probes; "
                "single no-boundary dense alphabets support homophonic or "
                "transposition+homophonic probes; depressed IC with period "
                "recovery supports periodic polyalphabetic probes."
            ),
        }

    def _tool_observe_periodic_ic(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        max_period = int(args.get("max_period", 26))
        top_n = max(1, min(int(args.get("top_n", 10)), 40))
        fp = cipher_id.compute_cipher_fingerprint(
            effective.tokens,
            effective.alphabet.size,
            max_period=max_period,
            language=self.language,
            word_group_count=len(self.workspace.effective_word_spans(branch_name)),
        )
        periodic_rows = [
            {
                "period": period,
                "mean_ic": round(value, 6),
                "recovery_vs_raw_ic": round(value - fp.ic, 6) if fp.ic == fp.ic else None,
                "near_language_reference": (
                    value >= (fp.language_ic_reference or 0.0) - 0.015
                    if fp.language_ic_reference is not None else False
                ),
                "kasiski_support": fp.kasiski_spacing_gcds.get(period, 0),
            }
            for period, value in sorted(fp.periodic_ic.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "branch": branch_name,
            **branch_note,
            "raw_ic": round(fp.ic, 6) if fp.ic == fp.ic else None,
            "language_reference_ic": fp.language_ic_reference,
            "best_period": fp.best_period,
            "best_period_ic": round(fp.best_period_ic, 6) if fp.best_period_ic is not None else None,
            "kasiski_best_period": fp.kasiski_best_period,
            "periodic_ic_top": periodic_rows[:top_n],
            "all_periodic_ic": {str(k): round(v, 6) for k, v in fp.periodic_ic.items()},
            "kasiski_spacing_gcds": {str(k): v for k, v in fp.kasiski_spacing_gcds.items()},
            "recommended_next_tool": "search_periodic_polyalphabetic",
            "note": (
                "For Vigenere-family ciphers, promising periods usually show "
                "periodic IC recovery above raw IC, ideally with Kasiski support. "
                "Use search_periodic_polyalphabetic to test candidate periods."
            ),
        }

    def _tool_observe_kasiski(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        min_ngram = max(2, min(int(args.get("min_ngram", 3)), 8))
        max_ngram = max(min_ngram, min(int(args.get("max_ngram", 5)), 10))
        max_period = max(2, min(int(args.get("max_period", 40)), 120))
        top_n = max(1, min(int(args.get("top_n", 12)), 50))
        report = cipher_id.kasiski_report(
            effective.tokens,
            min_n=min_ngram,
            max_n=max_ngram,
            max_period=max_period,
            top_n=top_n,
        )
        alpha = effective.alphabet
        repeated = []
        for row in report.get("repeated_sequences", []):
            ngram_ids = row.get("ngram", [])
            repeated.append({
                **row,
                "ngram_symbols": " ".join(alpha.symbol_for(int(t)) for t in ngram_ids),
            })
        report = dict(report)
        report["repeated_sequences"] = repeated
        return {
            "branch": branch_name,
            **branch_note,
            **report,
            "recommended_next_tools": [
                "observe_phase_frequency",
                "observe_periodic_shift_candidates",
                "search_periodic_polyalphabetic",
            ],
            "note": (
                "Kasiski support is corroborating evidence, not proof. Periods "
                "that agree with periodic IC peaks deserve a solver screen."
            ),
        }

    def _tool_observe_phase_frequency(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        period = self._period_from_args_or_branch(branch_name, args)
        top_n = max(1, min(int(args.get("top_n", 8)), 26))
        report = polyalphabetic.phase_frequency_report(
            self.workspace.effective_cipher_text(branch_name),
            period=period,
            top_n=top_n,
        )
        return {
            "branch": branch_name,
            **branch_note,
            **report,
            "recommended_next_tools": [
                "observe_periodic_shift_candidates",
                "search_periodic_polyalphabetic",
            ],
            "note": (
                "Each phase should look roughly like a Caesar-shifted language "
                "frequency distribution when the period is correct."
            ),
        }

    def _tool_observe_periodic_shift_candidates(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        period = self._period_from_args_or_branch(branch_name, args)
        variant = str(args.get("variant") or self._periodic_variant_for_branch(branch_name, None))
        top_n = max(1, min(int(args.get("top_n", 5)), 26))
        sample = max(1, min(int(args.get("sample", 80)), 200))
        report = polyalphabetic.periodic_shift_candidates(
            self.workspace.effective_cipher_text(branch_name),
            period=period,
            variant=variant,
            top_n=top_n,
            sample=sample,
        )
        return {
            "branch": branch_name,
            **branch_note,
            **report,
            "recommended_next_tools": [
                "act_set_periodic_key",
                "act_set_periodic_shift",
                "search_periodic_polyalphabetic",
            ],
            "note": (
                "These are phase-local hints. The best global key is the one "
                "that makes the full decoded stream read coherently."
            ),
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
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        branch = self.workspace.get_branch(branch_name)
        effective_tokens = self.workspace.effective_tokens(branch_name)
        preview_symbols = self.workspace.cipher_text.alphabet.decode(effective_tokens[:80])
        return {
            "branch": branch_name,
            **branch_note,
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
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
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
            **branch_note,
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
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
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
            **branch_note,
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

    def _tool_search_pure_transposition(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        profile = str(args.get("profile") or "wide").strip().lower()
        if profile not in {"small", "medium", "wide"}:
            return {"error": "profile must be one of: small, medium, wide"}
        top_n = max(1, min(int(args.get("top_n", 10)), 100))
        install_top_n = max(0, min(int(args.get("install_top_n", 1)), min(top_n, 20)))
        max_candidates_arg = args.get("max_candidates")
        max_candidates = (
            int(max_candidates_arg)
            if max_candidates_arg is not None
            else 5000 if profile == "small" else 25000 if profile == "medium" else None
        )
        transmatrix_max_arg = args.get("transmatrix_max_width")
        result = screen_pure_transposition(
            effective,
            language=self.language,
            profile=profile,
            top_n=top_n,
            max_candidates=max_candidates,
            include_matrix_rotate=bool(args.get("include_matrix_rotate", True)),
            include_transmatrix=bool(args.get("include_transmatrix", True)),
            transmatrix_min_width=max(2, int(args.get("transmatrix_min_width", 2))),
            transmatrix_max_width=(
                int(transmatrix_max_arg)
                if transmatrix_max_arg is not None
                else None
            ),
            threads=max(0, min(int(args.get("threads", 0)), 256)),
        )
        search_session_id = self._new_pure_transposition_session(
            source_branch=branch_name,
            profile=profile,
            result=result,
        )
        session = self._pure_transposition_session(search_session_id)
        assert session is not None

        installed: list[dict[str, Any]] = []
        prefix = str(args.get("new_branch_prefix") or "trans").strip() or "trans"
        if result.get("status") == "completed" and install_top_n > 0:
            for rank, candidate in enumerate(result.get("top_candidates", [])[:install_top_n], 1):
                family = str(candidate.get("family") or "candidate")
                candidate_id = str(candidate.get("candidate_id") or f"rank{rank}")
                branch_out = self._unique_branch_name(f"{prefix}_{family}_{candidate_id}_{rank}")
                installed.append(self._install_pure_transposition_finalist_branch(
                    session=session,
                    rank=rank,
                    branch_name=branch_out,
                ))

        top_candidates = []
        for candidate in result.get("top_candidates", [])[:top_n]:
            row = dict(candidate)
            plaintext = str(row.pop("plaintext", "") or "")
            row["preview"] = _truncate_text(plaintext or str(row.get("preview") or ""), 600)
            top_candidates.append(row)
        review = self._pure_transposition_finalist_review(
            session_id=search_session_id,
            start_rank=1,
            count=top_n,
            review_chars=int(args.get("review_chars", 600)),
        )

        return {
            "branch": branch_name,
            **branch_note,
            "status": result.get("status"),
            "search_session_id": search_session_id,
            "solver": result.get("solver"),
            "engine": result.get("engine"),
            "language": result.get("language"),
            "profile": profile,
            "threads": result.get("threads"),
            "candidate_count": result.get("candidate_count"),
            "valid_candidate_count": result.get("valid_candidate_count"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "cache": result.get("cache"),
            "candidate_plan": result.get("candidate_plan"),
            "family_counts": result.get("family_counts", {}),
            "top_family_counts": result.get("top_family_counts", {}),
            "best_candidate": top_candidates[0] if top_candidates else None,
            "top_candidates": top_candidates,
            "installed_branches": installed,
            **review,
            "scope_note": (
                "This is a pure-transposition screen. It assumes the symbols "
                "already are plaintext letters in the wrong order. For "
                "Zodiac/Z340-style transformed homophonic ciphers, use "
                "search_transform_homophonic instead."
            ),
            "recommended_next_tools": [
                "search_review_pure_transposition_finalists",
                "act_rate_transform_finalist",
                "act_install_pure_transposition_finalists",
                "workspace_branch_cards",
                "decode_show",
                "workspace_update_hypothesis",
                "search_transform_homophonic",
            ],
            "note": (
                "Installed branches carry both a transform pipeline and "
                "mode-specific decoded_text metadata, so read them directly "
                "with workspace_branch_cards or decode_show before declaring."
            ),
        }

    def _tool_search_periodic_polyalphabetic(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        top_n = max(1, min(int(args.get("top_n", 5)), 20))
        install_top_n = max(0, min(int(args.get("install_top_n", 1)), top_n))
        periods = args.get("periods")
        if periods is not None:
            periods = [int(p) for p in periods]
        variants = args.get("variants")
        result = polyalphabetic.search_periodic_polyalphabetic(
            effective,
            language=self.language,
            periods=periods,
            max_period=int(args.get("max_period", 20)),
            variants=variants,
            top_n=top_n,
            refine=True,
        )
        installed: list[dict[str, Any]] = []
        prefix = str(args.get("new_branch_prefix") or "poly").strip() or "poly"
        if result.get("status") == "completed" and install_top_n > 0:
            for idx, candidate in enumerate(result.get("top_candidates", [])[:install_top_n], 1):
                base_name = f"{prefix}_{candidate['variant']}_p{candidate['period']}_{idx}"
                new_name = self._unique_branch_name(base_name)
                new_branch = self.workspace.fork(new_name, from_branch=branch_name)
                new_branch.metadata.update({
                    "cipher_mode": "periodic_polyalphabetic",
                    "mode_status": "active",
                    "mode_confidence": "medium",
                    "mode_evidence": (
                        "Installed by search_periodic_polyalphabetic from "
                        f"{candidate['variant']} period {candidate['period']}."
                    ),
                    "key_type": "PeriodicShiftKey",
                    "periodic_variant": candidate["variant"],
                    "periodic_period": candidate["period"],
                    "periodic_key": candidate["key"],
                    "periodic_shifts": candidate["shifts"],
                    "periodic_score": candidate["score"],
                    "decoded_text": candidate["plaintext"],
                    "decoded_text_source": "search_periodic_polyalphabetic",
                })
                self.workspace.tag(new_name, "hypothesis")
                self.workspace.tag(new_name, "mode:periodic_polyalphabetic")
                installed.append({
                    "branch": new_name,
                    "variant": candidate["variant"],
                    "period": candidate["period"],
                    "key": candidate["key"],
                    "score": candidate["score"],
                    "preview": candidate["preview"],
                })
        return {
            "branch": branch_name,
            **branch_note,
            **result,
            "installed_branches": installed,
            "scope_note": (
                "This search tested ordinary A-Z periodic-shift families "
                "(Vigenere, Beaufort, variant Beaufort, Gronsfeld). It does "
                "not rule out keyed-tableau/Kryptos/Quagmire-style "
                "polyalphabetic ciphers."
            ),
            "recommended_next_tools": [
                "search_quagmire3_keyword_alphabet",
                "workspace_hypothesis_next_steps",
                "workspace_branch_cards",
            ],
            "note": (
                "Periodic polyalphabetic candidates are stored as mode-specific "
                "branch metadata, not substitution keys. Use decode_show, "
                "decode_show_phases, act_set_periodic_key, "
                "act_set_periodic_shift, workspace_hypothesis_cards, and "
                "contextual reading to judge whether a candidate is coherent "
                "before repairing locally. If these ordinary shift candidates "
                "are incoherent while context or statistics still supports a "
                "polyalphabetic family, escalate to "
                "search_quagmire3_keyword_alphabet rather than rejecting the "
                "family."
            ),
        }

    def _tool_search_quagmire3_keyword_alphabet(self, args: dict) -> Any:
        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        effective = self.workspace.effective_cipher_text(branch_name)
        top_n = max(1, min(int(args.get("top_n", 5)), 20))
        install_top_n = max(0, min(int(args.get("install_top_n", 1)), top_n))

        def int_list(name: str) -> list[int] | None:
            raw = args.get(name)
            if raw is None:
                return None
            if isinstance(raw, str):
                raw = [item.strip() for item in raw.split(",") if item.strip()]
            return [int(item) for item in raw]

        def str_list(name: str) -> list[str]:
            raw = args.get(name) or []
            if isinstance(raw, str):
                raw = [item.strip() for item in raw.split(",") if item.strip()]
            return [str(item).strip() for item in raw if str(item).strip()]

        initial_keywords = str_list("initial_keywords")
        engine = str(args.get("engine") or "rust_shotgun").strip().lower()
        keyword_lengths = int_list("keyword_lengths")
        cycleword_lengths = int_list("cycleword_lengths")
        hillclimbs = max(0, min(int(args.get("hillclimbs", 500)), 1_000_000))
        restarts = max(1, min(int(args.get("restarts", 8)), 1_000_000))
        threads = max(0, min(int(args.get("threads", 0)), 256))
        budget_estimate: dict[str, Any] | None = None
        if engine == "rust_shotgun":
            try:
                from analysis.polyalphabetic_fast import estimate_quagmire3_shotgun_budget

                budget_estimate = estimate_quagmire3_shotgun_budget(
                    keyword_lengths=keyword_lengths,
                    cycleword_lengths=cycleword_lengths,
                    hillclimbs=hillclimbs,
                    restarts=restarts,
                    threads=threads,
                )
            except Exception as exc:  # noqa: BLE001
                budget_estimate = {
                    "engine": "rust_shotgun",
                    "status": "estimate_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            steps = max(0, min(int(args.get("steps", 200)), 20_000))
            keyword_count = len(keyword_lengths or [7])
            cycleword_count = len(cycleword_lengths or [8])
            budget_estimate = {
                "engine": "python_screen",
                "keyword_lengths": keyword_lengths or [7],
                "cycleword_lengths": cycleword_lengths or [8],
                "restarts": restarts,
                "steps_per_start": steps,
                "approx_screen_mutations": keyword_count * cycleword_count * restarts * max(1, steps),
                "note": (
                    "Python screen is reference/diagnostic scaffolding. Use "
                    "small budgets only; broad Quagmire searches should use "
                    "rust_shotgun."
                ),
            }
        rust_kernel_status: dict[str, Any] | None = None
        try:
            from analysis.polyalphabetic_fast import fast_kernel_status

            rust_kernel_status = fast_kernel_status()
        except Exception as exc:  # noqa: BLE001
            rust_kernel_status = {
                "available": False,
                "import_error": f"{type(exc).__name__}: {exc}",
            }
        if bool(args.get("estimate_only", False)):
            estimate_nominal = (
                budget_estimate.get("nominal_proposals")
                if isinstance(budget_estimate, dict)
                else None
            )
            estimate_budget_class = self._quagmire_budget_class(estimate_nominal)
            return {
                "branch": branch_name,
                **branch_note,
                "status": "estimated",
                "solver": "quagmire3_budget_estimate",
                "engine": engine,
                "budget_class": estimate_budget_class,
                "budget_sufficiency": (
                    "diagnostic_only"
                    if self._quagmire_budget_rank(estimate_budget_class)
                    < self._quagmire_budget_rank("moderate")
                    else "sufficient_to_test_family_rejection"
                ),
                "budget_estimate": budget_estimate,
                "rust_fast_kernel": rust_kernel_status,
                "seeded_initial_keywords": bool(initial_keywords),
                "installed_branches": [],
                "recommended_next_action": (
                    "If the estimated runtime is acceptable and the search is "
                    "the right cipher-family hypothesis, rerun this tool with "
                    "estimate_only=false. Start diagnostic, then moderate, "
                    "then broad unless context strongly justifies a broad run."
                ),
                "recommended_next_tools": [
                    "search_quagmire3_keyword_alphabet",
                    "workspace_hypothesis_next_steps",
                    "workspace_branch_cards",
                ],
            }
        if engine == "rust_shotgun":
            try:
                from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast

                result = search_quagmire3_shotgun_fast(
                    effective,
                    language=self.language,
                    keyword_lengths=keyword_lengths,
                    cycleword_lengths=cycleword_lengths,
                    hillclimbs=hillclimbs,
                    restarts=restarts,
                    seed=int(args.get("seed", 1)),
                    top_n=top_n,
                    slip_probability=max(0.0, float(args.get("slip_probability", 0.001))),
                    backtrack_probability=max(0.0, float(args.get("backtrack_probability", 0.15))),
                    threads=threads,
                    initial_keywords=initial_keywords,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "error",
                    "solver": "quagmire3_shotgun_rust",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rust_fast_kernel": rust_kernel_status,
                    "diagnostic_engine_available": "python_screen",
                    "engine_equivalence": (
                        "`python_screen` is available as a smaller diagnostic "
                        "search, but it is not equivalent to the compiled "
                        "parallel `rust_shotgun` loop and should not be used "
                        "for large-scale/blind Quagmire claims."
                    ),
                    "recommended_next_action": (
                        "Ask the user to run `scripts/build_rust_fast.sh`, "
                        "verify with `decipher doctor`, and rerun with "
                        "engine='rust_shotgun'. Only use engine='python_screen' "
                        "for a deliberately small diagnostic probe."
                    ),
                    "note": (
                        "Rust shotgun engine failed. Do not silently treat a "
                        "Python diagnostic path as the same experiment. Run "
                        "`scripts/build_rust_fast.sh` and then "
                        "`PYTHONPATH=src .venv/bin/decipher doctor`."
                    ),
                }
        else:
            result = polyalphabetic.search_quagmire3_keyword_alphabet(
                effective,
                language=self.language,
                keyword_lengths=keyword_lengths,
                cycleword_lengths=cycleword_lengths,
                initial_keywords=initial_keywords,
                steps=max(0, min(int(args.get("steps", 200)), 20_000)),
                restarts=restarts,
                seed=int(args.get("seed", 1)),
                top_n=top_n,
                refine=True,
                screen_top_n=max(1, min(int(args.get("screen_top_n", 64)), 5_000)),
                word_weight=max(0.0, float(args.get("word_weight", 0.25))),
                dictionary_keyword_limit=max(
                    0,
                    min(int(args.get("dictionary_keyword_limit", 0)), 50_000),
                ),
            )

        nominal_proposals = result.get("nominal_proposals")
        if nominal_proposals is None and isinstance(budget_estimate, dict):
            nominal_proposals = budget_estimate.get("nominal_proposals")
        budget_class = self._quagmire_budget_class(nominal_proposals)
        budget_sufficiency = (
            "sufficient_to_test_family_rejection"
            if self._quagmire_budget_rank(budget_class) >= self._quagmire_budget_rank("moderate")
            else "diagnostic_only"
        )

        installed: list[dict[str, Any]] = []
        prefix = str(args.get("new_branch_prefix") or "quag3").strip() or "quag3"
        if result.get("status") == "completed" and install_top_n > 0:
            for idx, candidate in enumerate(result.get("top_candidates", [])[:install_top_n], 1):
                metadata = dict(candidate.get("metadata") or {})
                alphabet_keyword = str(metadata.get("alphabet_keyword") or "unknown")
                cycleword = str(metadata.get("cycleword") or candidate.get("key") or "unknown")
                base_name = f"{prefix}_{alphabet_keyword}_{cycleword}_{idx}"
                new_name = self._unique_branch_name(base_name)
                new_branch = self.workspace.fork(new_name, from_branch=branch_name)
                new_branch.metadata.update({
                    "cipher_mode": "quagmire3",
                    "mode_status": "active",
                    "mode_confidence": "medium",
                    "mode_evidence": (
                        "Installed by search_quagmire3_keyword_alphabet from "
                        f"alphabet keyword {alphabet_keyword!r} and cycleword "
                        f"{cycleword!r}."
                    ),
                    "mode_counter_evidence": (
                        "This bounded search can overfit word islands. Treat "
                        "explicit initial_keywords as seeded context, not a "
                        "blind recovery."
                    ),
                    "key_type": "QuagmireKey",
                    "quagmire_type": metadata.get("quagmire_type", "quag3"),
                    "alphabet_keyword": alphabet_keyword,
                    "cycleword": cycleword,
                    "cycleword_shifts": metadata.get("cycleword_shifts", candidate.get("shifts")),
                    "plaintext_alphabet": metadata.get("plaintext_alphabet"),
                    "ciphertext_alphabet": metadata.get("ciphertext_alphabet"),
                    "quagmire_score": candidate.get("score"),
                    "quagmire_selection_score": candidate.get("selection_score"),
                    "decoded_text": candidate.get("plaintext", ""),
                    "decoded_text_source": "search_quagmire3_keyword_alphabet",
                    "search_metadata": {
                        "solver": result.get("solver"),
                        "seed": result.get("seed"),
                        "engine": engine,
                        "threads": result.get("threads"),
                        "hillclimbs_per_restart": result.get("hillclimbs_per_restart"),
                        "nominal_proposals": nominal_proposals,
                        "budget_class": budget_class,
                        "budget_sufficiency": budget_sufficiency,
                        "restart_jobs": result.get("restart_jobs"),
                        "steps_per_start": result.get("steps_per_start"),
                        "restarts_per_length": result.get("restarts_per_length"),
                        "keyword_states_screened": result.get("keyword_states_screened"),
                        "screen_top_n": result.get("screen_top_n"),
                        "word_weight": result.get("word_weight"),
                        "seeded_initial_keywords": bool(initial_keywords),
                        "start_type": metadata.get("start_type"),
                    },
                })
                self.workspace.tag(new_name, "hypothesis")
                self.workspace.tag(new_name, "mode:quagmire3")
                self.workspace.tag(new_name, "mode:keyed_tableau_polyalphabetic")
                installed.append({
                    "branch": new_name,
                    "alphabet_keyword": alphabet_keyword,
                    "cycleword": cycleword,
                    "period": candidate.get("period"),
                    "score": candidate.get("score"),
                    "selection_score": candidate.get("selection_score"),
                    "start_type": metadata.get("start_type"),
                    "seeded_initial_keywords": bool(initial_keywords),
                    "preview": candidate.get("preview"),
                })

        return {
            "branch": branch_name,
            **branch_note,
            **result,
            "installed_branches": installed,
            "seeded_initial_keywords": bool(initial_keywords),
            "rust_fast_kernel": rust_kernel_status,
            "budget_estimate": budget_estimate,
            "nominal_proposals": nominal_proposals,
            "budget_class": budget_class,
            "budget_sufficiency": budget_sufficiency,
            "budget_note": (
                "For rust_shotgun, nominal_proposals = "
                "len(keyword_lengths) * len(cycleword_lengths) * restarts * "
                "hillclimbs. Use estimate_only=true before broad searches. "
                "Diagnostic searches prove the tool path works, but do not "
                "justify rejecting the Quagmire/keyed-tableau family."
            ),
            "engine_equivalence_note": (
                "`rust_shotgun` is the compiled parallel broad-search engine. "
                "`python_screen` is the older bounded diagnostic engine; it is "
                "reference/test scaffolding and useful for quick probes, but "
                "not an equivalent runtime path for large-scale search budgets."
            ),
            "recommended_next_tools": [
                "workspace_branch_cards",
                "workspace_hypothesis_next_steps",
                "workspace_update_hypothesis",
            ],
            "note": (
                "Quagmire III candidates are stored as mode-specific decoded "
                "branches, not substitution keys. Read multiple candidates for "
                "coherent text. If initial_keywords were supplied, label the "
                "run as context-seeded rather than blind."
            ),
        }

    def _apply_periodic_key_to_branch(
        self,
        branch_name: str,
        shifts: list[int],
        *,
        variant: str,
        source: str,
    ) -> dict[str, Any]:
        branch = self.workspace.get_branch(branch_name)
        decoded = polyalphabetic.decode_cipher_text(
            self.workspace.effective_cipher_text(branch_name),
            shifts,
            variant=variant,
        )
        if decoded.get("status") != "completed":
            return decoded
        branch.metadata.update({
            "cipher_mode": "periodic_polyalphabetic",
            "mode_status": branch.metadata.get("mode_status", "active"),
            "mode_confidence": branch.metadata.get("mode_confidence", "medium"),
            "key_type": "PeriodicShiftKey",
            "periodic_variant": variant,
            "periodic_period": decoded["period"],
            "periodic_key": decoded["key"],
            "periodic_shifts": decoded["shifts"],
            "decoded_text": decoded["plaintext"],
            "decoded_text_source": source,
        })
        self.workspace.tag(branch_name, "hypothesis")
        self.workspace.tag(branch_name, "mode:periodic_polyalphabetic")
        return {
            "status": "ok",
            "branch": branch_name,
            "cipher_mode": "periodic_polyalphabetic",
            "variant": decoded["variant"],
            "period": decoded["period"],
            "key": decoded["key"],
            "shifts": decoded["shifts"],
            "decoded_preview": decoded["plaintext"][:240],
            "scores": self._compute_quick_scores(branch_name),
        }

    def _periodic_variant_for_branch(self, branch_name: str, override: str | None = None) -> str:
        if override:
            return override.strip().lower()
        branch = self.workspace.get_branch(branch_name)
        return str(branch.metadata.get("periodic_variant") or "vigenere").strip().lower()

    def _periodic_shifts_for_branch(self, branch_name: str) -> list[int]:
        branch = self.workspace.get_branch(branch_name)
        raw = branch.metadata.get("periodic_shifts")
        if isinstance(raw, list) and raw:
            return [int(v) for v in raw]
        key = branch.metadata.get("periodic_key")
        variant = self._periodic_variant_for_branch(branch_name)
        if isinstance(key, str) and key.strip():
            return polyalphabetic.parse_periodic_key(key=key, variant=variant)
        raise WorkspaceError(
            "Branch has no periodic key. Use search_periodic_polyalphabetic "
            "or act_set_periodic_key first."
        )

    def _period_from_args_or_branch(self, branch_name: str, args: dict) -> int:
        explicit = args.get("period")
        if explicit is not None:
            period = int(explicit)
            if period <= 0:
                raise ValueError("period must be >= 1")
            return period
        branch = self.workspace.get_branch(branch_name)
        metadata_period = branch.metadata.get("periodic_period")
        if metadata_period is not None:
            period = int(metadata_period)
            if period > 0:
                return period
        try:
            return len(self._periodic_shifts_for_branch(branch_name))
        except WorkspaceError:
            pass
        effective = self.workspace.effective_cipher_text(branch_name)
        fp = cipher_id.compute_cipher_fingerprint(
            effective.tokens,
            effective.alphabet.size,
            max_period=26,
            language=self.language,
            word_group_count=len(self.workspace.effective_word_spans(branch_name)),
        )
        if fp.best_period is not None:
            return int(fp.best_period)
        raise WorkspaceError(
            "No period available. Provide period=..., run observe_periodic_ic, "
            "or create a periodic branch with search_periodic_polyalphabetic."
        )

    def _periodic_key_symbol(self, shift: int, variant: str) -> str:
        if variant == "gronsfeld":
            return str(int(shift) % 10)
        return chr(ord("A") + (int(shift) % 26))

    def _unique_branch_name(self, base_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_") or "branch"
        name = sanitized
        i = 2
        while self.workspace.has_branch(name):
            name = f"{sanitized}_{i}"
            i += 1
        return name

    # ------------------------------------------------------------------
    # decode_*
    # ------------------------------------------------------------------
    def _tool_decode_show_phases(self, args: dict) -> Any:
        from collections import Counter

        branch_name, branch_note = self._resolve_observation_branch(args.get("branch"))
        variant = self._periodic_variant_for_branch(branch_name, args.get("variant"))
        sample = max(1, min(int(args.get("sample", 24)), 120))
        explicit_period = args.get("period")
        shifts: list[int] | None
        try:
            shifts = self._periodic_shifts_for_branch(branch_name)
        except WorkspaceError:
            shifts = None
        period = int(explicit_period) if explicit_period is not None else (len(shifts) if shifts else 0)
        if period <= 0:
            return {
                "error": (
                    "No period available. Provide period=... or set a periodic "
                    "key/search result on this branch first."
                )
            }
        values, skipped = polyalphabetic.cipher_values_from_text(
            self.workspace.effective_cipher_text(branch_name)
        )
        if skipped:
            return {
                "error": "decode_show_phases currently requires clean A-Z ciphertext",
                "skipped_symbols": skipped[:20],
            }
        rows = []
        for phase in range(period):
            stream = values[phase::period]
            shift = shifts[phase] if shifts and phase < len(shifts) else None
            decoded_sample = (
                polyalphabetic.decode_values(stream[:sample], [shift], variant=variant)
                if shift is not None else None
            )
            counts = Counter(stream)
            rows.append({
                "phase": phase,
                "length": len(stream),
                "shift": shift,
                "key_symbol": self._periodic_key_symbol(shift, variant) if shift is not None else None,
                "cipher_sample": "".join(chr(ord("A") + v) for v in stream[:sample]),
                "decoded_sample": decoded_sample,
                "top_cipher_letters": [
                    {
                        "letter": chr(ord("A") + value),
                        "count": count,
                        "pct": round(count / len(stream) * 100, 2) if stream else 0.0,
                    }
                    for value, count in counts.most_common(8)
                ],
            })
        return {
            "branch": branch_name,
            **branch_note,
            "cipher_mode": self.workspace.get_branch(branch_name).metadata.get("cipher_mode"),
            "variant": variant,
            "period": period,
            "key": self.workspace.get_branch(branch_name).metadata.get("periodic_key"),
            "phase_rows": rows,
            "note": (
                "Each row is one key phase. If a phase's decoded_sample looks "
                "systematically shifted, use act_set_periodic_shift or "
                "act_adjust_periodic_shift on that phase."
            ),
        }

    def _tool_decode_show(self, args: dict) -> Any:
        branch = args["branch"]
        start = args.get("start_word", 0)
        count = min(args.get("count", 25), 50)
        metadata_text = self.workspace.get_branch(branch).metadata.get("decoded_text")
        if isinstance(metadata_text, str) and metadata_text.strip():
            chunks = self._decoded_words(branch)
            if start >= len(chunks):
                return {"error": f"start_word {start} exceeds total {len(chunks)}"}
            end = min(start + count, len(chunks))
            return {
                "branch": branch,
                "range": f"{start}-{end - 1}",
                "total_words": len(chunks),
                "cipher_mode": self.workspace.get_branch(branch).metadata.get("cipher_mode"),
                "rows": [
                    {
                        "index": i,
                        "cipher": "(periodic-key branch)",
                        "decoded": chunks[i],
                    }
                    for i in range(start, end)
                ],
                "note": (
                    "This branch stores mode-specific decoded text in metadata "
                    "rather than a substitution mapping."
                ),
            }
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
        branch = args["branch"]
        basin = self._branch_basin_status(branch)
        basin_warning: dict[str, Any] | None = None
        if self._repair_policy_blocks_word_repair(basin):
            basin_warning = {
                "status": basin.get("status"),
                "repair_policy": basin.get("repair_policy"),
                "reason": basin.get("reason"),
                "suggested_next_tools": basin.get("suggested_next_tools", []),
                "note": (
                    "This menu is still safe because it is read-only, but do "
                    "not apply a local word repair from this branch unless you "
                    "can paraphrase a coherent clause in the decoded text. "
                    "Prefer the suggested search tools first."
                ),
            }

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
                    branch,
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
                    f"branch={branch!r}, "
                    f"cipher_word_index={plan['cipher_word_index']}, "
                    f"target_word={plan['target_word']!r})"
                )
                if basin_warning:
                    plan["recommendation"] = (
                        "search_before_apply: branch looks like word islands "
                        "rather than a coherent basin; apply only with "
                        "allow_bad_basin_repair=true after you can paraphrase "
                        "a coherent clause."
                    )
            options.append(plan)

        return {
            "status": "ok",
            "branch": branch,
            "cipher_word_index": cipher_word_index,
            "current_decoded": current_decoded,
            "option_count": len(options),
            "options": options,
            "basin": basin,
            "basin_warning": basin_warning,
            "note": (
                "This is a read-only menu. Apply only options whose preview "
                "reads better and whose recommendation does not say "
                "do_not_apply_directly. If basin_warning is present, local "
                "word repairs are probably premature: first take a bigger "
                "search/transform swing unless you can paraphrase a coherent "
                "clause. For conflicting repeated-symbol cases, "
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
        blocked = self._mode_mismatch_repair_block(
            branch,
            "act_set_mapping",
            allow_override=bool(args.get("allow_mode_mismatch_repair", False)),
        )
        if blocked is not None:
            return blocked
        cipher_sym = args["cipher_symbol"]
        plain_letter = args["plain_letter"].upper()
        dry_run = bool(args.get("dry_run", False))
        alpha = self._alpha()
        pt_alpha = self._pt_alpha()
        if not alpha.has_symbol(cipher_sym):
            return {"error": f"Unknown cipher symbol: {cipher_sym}"}
        if not pt_alpha.has_symbol(plain_letter):
            return {"error": f"Unknown plaintext letter: {plain_letter}"}
        ct_id = alpha.id_for(cipher_sym)
        pt_id = pt_alpha.id_for(plain_letter)
        branch_state = self.workspace.get_branch(branch)
        had_previous = ct_id in branch_state.key
        previous_pt_id = branch_state.key.get(ct_id)
        previous_plain = (
            pt_alpha.symbol_for(previous_pt_id)
            if previous_pt_id is not None else None
        )
        before = self._compute_quick_scores(branch)
        before_words = self._decoded_words(branch)
        self.workspace.set_mapping(branch, ct_id, pt_id)
        after = self._compute_quick_scores(branch)
        after_words = self._decoded_words(branch)
        changed = self._changed_words_sample(before_words, after_words)
        orthography_risks = self._orthography_risks(before_words, after_words)
        decoded_preview = self._decoded_preview(branch)
        if dry_run:
            if had_previous and previous_pt_id is not None:
                self.workspace.set_mapping(branch, ct_id, previous_pt_id)
            else:
                self.workspace.clear_mapping(branch, ct_id)
        return {
            "status": "preview" if dry_run else "ok",
            "branch": branch,
            "mapping": f"{cipher_sym} -> {plain_letter}",
            "dry_run": dry_run,
            "previous_mapping": (
                f"{cipher_sym} -> {previous_plain}"
                if previous_plain is not None else None
            ),
            "occurrences_changed": sum(
                1 for w in self.workspace.cipher_text.tokens
                if alpha.symbol_for(w) == cipher_sym
            ),
            "changed_words": changed,
            "orthography_risks": orthography_risks,
            "decoded_preview": decoded_preview,
            "score_delta": self._reading_score_delta(before, after),
            "undo_call": (
                f"act_set_mapping(branch={branch!r}, cipher_symbol={cipher_sym!r}, "
                f"plain_letter={previous_plain!r})"
                if previous_plain is not None
                else f"act_clear_mapping(branch={branch!r}, cipher_symbol={cipher_sym!r})"
            ),
            "note": (
                "Dry run only: branch state was restored after preview. "
                "Call again with dry_run=false to keep this mapping. "
                + self._READING_DECISION_NOTE
                if dry_run else
                "If this speculative repair broke already-correct words, use "
                "`undo_call` immediately or make such trials on a forked "
                "repair branch. "
                + self._READING_DECISION_NOTE
            ),
        }

    def _tool_act_set_periodic_key(self, args: dict) -> Any:
        branch = args["branch"]
        variant = self._periodic_variant_for_branch(branch, args.get("variant"))
        shifts_arg = args.get("shifts")
        shifts = polyalphabetic.parse_periodic_key(
            key=args.get("key"),
            shifts=shifts_arg if isinstance(shifts_arg, list) else None,
            variant=variant,
        )
        before = {
            "preview": self._decoded_preview(branch, max_words=4),
            "scores": self._compute_quick_scores(branch),
            "key": self.workspace.get_branch(branch).metadata.get("periodic_key"),
        }
        out = self._apply_periodic_key_to_branch(
            branch,
            shifts,
            variant=variant,
            source="act_set_periodic_key",
        )
        out["before"] = before
        out["after_preview"] = self._decoded_preview(branch, max_words=4)
        return out

    def _tool_act_set_periodic_shift(self, args: dict) -> Any:
        branch = args["branch"]
        variant = self._periodic_variant_for_branch(branch, args.get("variant"))
        shifts = self._periodic_shifts_for_branch(branch)
        phase = int(args["phase"])
        if phase < 0 or phase >= len(shifts):
            return {"error": f"phase {phase} out of range for period {len(shifts)}"}
        modulus = 10 if variant == "gronsfeld" else 26
        before = {
            "preview": self._decoded_preview(branch, max_words=4),
            "scores": self._compute_quick_scores(branch),
            "key": self.workspace.get_branch(branch).metadata.get("periodic_key"),
            "shift": shifts[phase],
        }
        shifts[phase] = int(args["shift"]) % modulus
        out = self._apply_periodic_key_to_branch(
            branch,
            shifts,
            variant=variant,
            source="act_set_periodic_shift",
        )
        out["changed_phase"] = phase
        out["before"] = before
        out["after_preview"] = self._decoded_preview(branch, max_words=4)
        return out

    def _tool_act_adjust_periodic_shift(self, args: dict) -> Any:
        branch = args["branch"]
        variant = self._periodic_variant_for_branch(branch, args.get("variant"))
        shifts = self._periodic_shifts_for_branch(branch)
        phase = int(args["phase"])
        if phase < 0 or phase >= len(shifts):
            return {"error": f"phase {phase} out of range for period {len(shifts)}"}
        modulus = 10 if variant == "gronsfeld" else 26
        delta = int(args.get("delta", 1))
        return self._tool_act_set_periodic_shift({
            "branch": branch,
            "phase": phase,
            "shift": (shifts[phase] + delta) % modulus,
            "variant": variant,
        })

    def _tool_act_bulk_set(self, args: dict) -> Any:
        branch = args["branch"]
        blocked = self._mode_mismatch_repair_block(
            branch,
            "act_bulk_set",
            allow_override=bool(args.get("allow_mode_mismatch_repair", False)),
        )
        if blocked is not None:
            return blocked
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
        blocked = self._mode_mismatch_repair_block(
            branch_name,
            "act_anchor_word",
            allow_override=bool(args.get("allow_mode_mismatch_repair", False)),
        )
        if blocked is not None:
            return blocked
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
        blocked = self._mode_mismatch_repair_block(
            branch,
            "act_apply_word_repair",
            allow_override=bool(args.get("allow_mode_mismatch_repair", False)),
        )
        if blocked is not None:
            return blocked
        dry_run = bool(args.get("dry_run", False))
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
        basin = self._branch_basin_status(branch)
        allow_bad_basin_repair = bool(args.get("allow_bad_basin_repair", False))
        if (
            self._repair_policy_blocks_word_repair(basin)
            and not allow_bad_basin_repair
        ):
            plan["agenda_item"] = self._upsert_repair_agenda_item(
                plan,
                status="blocked",
                notes=(
                    "Blocked by bad-basin guard: branch looks like word "
                    "islands, not a coherent reading basin."
                ),
            )
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "bad_basin_word_repair_blocked",
                "basin": basin,
                **plan,
                "note": (
                    "The branch currently looks like isolated dictionary word "
                    "islands rather than coherent prose. Do not polish local "
                    "words yet. First run the suggested broader search tools, "
                    "or call this tool again with allow_bad_basin_repair=true "
                    "only if you can paraphrase a coherent clause and are "
                    "intentionally overriding the guard."
                ),
                "suggested_next_tools": basin.get("suggested_next_tools", []),
            }

        before = self._compute_quick_scores(branch)
        before_words = self._decoded_words(branch, max_words=None)
        branch_state = self.workspace.get_branch(branch)
        previous_mappings: dict[int, int | None] = {
            token_id: branch_state.key.get(token_id)
            for token_id in token_mappings
        }
        for token_id, pt_id in token_mappings.items():
            self.workspace.set_mapping(branch, token_id, pt_id)
        after = self._compute_quick_scores(branch)
        after_words = self._decoded_words(branch, max_words=None)
        changed_words = self._changed_words_sample(before_words, after_words)
        score_delta = self._reading_score_delta(before, after)
        orthography_risks = self._orthography_risks(before_words, after_words)
        decoded_preview = self._decoded_preview(branch)
        if dry_run:
            for token_id, previous_pt_id in previous_mappings.items():
                if previous_pt_id is None:
                    self.workspace.clear_mapping(branch, token_id)
                else:
                    self.workspace.set_mapping(branch, token_id, previous_pt_id)
            agenda_item = None
        else:
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
        pt_alpha = self._pt_alpha()
        alpha = self._alpha()
        undo_mappings = {
            alpha.symbol_for(token_id): (
                pt_alpha.symbol_for(previous_pt_id)
                if previous_pt_id is not None else None
            )
            for token_id, previous_pt_id in previous_mappings.items()
        }
        return {
            "status": "preview" if dry_run else "ok",
            "branch": branch,
            "dry_run": dry_run,
            "cipher_word_index": plan["cipher_word_index"],
            "cipher_word": plan["cipher_word"],
            "from": plan["current_decoded"],
            "to": plan["target_word"],
            "mappings_set": len(token_mappings),
            "mappings": plan["proposed_mappings"],
            "undo_mappings": undo_mappings,
            "changed_words": changed_words,
            "orthography_risks": orthography_risks,
            "decoded_preview": decoded_preview,
            "score_delta": score_delta,
            "basin_before": basin,
            "basin_after": self._branch_basin_status(branch),
            "agenda_item": agenda_item,
            "note": (
                "Dry run only: branch state and repair agenda were unchanged. "
                "Call again with dry_run=false to keep this repair. "
                + self._READING_DECISION_NOTE
                if dry_run else self._READING_DECISION_NOTE
            ),
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
        token_count = len(self.workspace.effective_tokens(branch))
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
        token_count = len(self.workspace.effective_tokens(branch))
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
                f"coherent {self.language} phrases, consider declaring; a "
                "few recognisable words alone are not enough if useful search "
                "or repair tools remain. If still stuck, fork and restart "
                "with search_anneal — it escapes local optima that hill_climb "
                "can't."
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
            f"Read decoded_preview carefully. If coherent {self.language} "
            "phrases appear, repair obvious residual errors and then consider "
            "declaring. Do not declare early merely because a few recognizable "
            "words appear; if useful tools remain, continued exploration beats "
            "a low-confidence partial. If a few residual errors remain, "
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

    def _new_transform_search_session(
        self,
        *,
        source_branch: str,
        profile: str,
        columns: int | None,
        ranked: list[dict[str, Any]],
        structural_screen: dict[str, Any],
    ) -> str:
        session_id = f"transform_search_{self._next_transform_search_session_id}"
        self._next_transform_search_session_id += 1
        self._transform_search_sessions[session_id] = {
            "source_branch": source_branch,
            "profile": profile,
            "columns": columns,
            "ranked": ranked,
            "structural_screen_summary": {
                "candidate_count": structural_screen.get("candidate_count"),
                "deduped_candidate_count": structural_screen.get("deduped_candidate_count"),
                "generation_limit_reached": structural_screen.get("generation_limit_reached"),
                "top_family_counts": structural_screen.get("top_family_counts", {}),
            },
        }
        return session_id

    def _transform_session(self, session_id: str) -> dict[str, Any] | None:
        return self._transform_search_sessions.get(str(session_id))

    def _new_pure_transposition_session(
        self,
        *,
        source_branch: str,
        profile: str,
        result: dict[str, Any],
    ) -> str:
        session_id = f"pure_transposition_{self._next_pure_transposition_session_id}"
        self._next_pure_transposition_session_id += 1
        self._pure_transposition_sessions[session_id] = {
            "source_branch": source_branch,
            "profile": profile,
            "ranked": list(result.get("top_candidates") or []),
            "candidate_count": result.get("candidate_count"),
            "valid_candidate_count": result.get("valid_candidate_count"),
            "family_counts": result.get("family_counts", {}),
            "top_family_counts": result.get("top_family_counts", {}),
            "candidate_plan": result.get("candidate_plan"),
            "cache": result.get("cache"),
        }
        return session_id

    def _pure_transposition_session(self, session_id: str) -> dict[str, Any] | None:
        return self._pure_transposition_sessions.get(str(session_id))

    def _pure_transposition_ranking_score(
        self,
        *,
        candidate: dict[str, Any],
        score_gap_from_best: float | None,
    ) -> dict[str, Any]:
        rating = self._transform_candidate_rating(candidate)
        return {
            "primary": {
                "name": "agent_contextual_readability",
                "value": rating.get("readability_score") if rating else None,
                "label": rating.get("label") if rating else None,
                "rationale": rating.get("rationale") if rating else None,
                "coherent_clause": rating.get("coherent_clause") if rating else None,
                "must_be_supplied_by_agent": rating is None,
                "scale": "0..4",
                "note": (
                    "This is intentionally not computed by the tool. The "
                    "agent must judge whether the plaintext preview reads "
                    "coherently in context."
                ),
            },
            "supporting": {
                "score": candidate.get("score"),
                "selection_score": candidate.get("selection_score"),
                "validated_selection_score": candidate.get("validated_selection_score"),
                "validation": candidate.get("validation"),
                "score_gap_from_best": score_gap_from_best,
            },
            "ranking_rule": (
                "For pure transposition, rank finalists by contextual "
                "readability first. Use finalist validation and direct "
                "language score as supporting evidence or tie-breakers."
            ),
        }

    def _mirror_pure_rating_to_branches(
        self,
        *,
        session_id: str,
        rank: int,
        rating: dict[str, Any],
    ) -> list[str]:
        updated: list[str] = []
        for branch_name in self.workspace.branch_names():
            branch = self.workspace.get_branch(branch_name)
            metadata = branch.metadata.get("pure_transposition_finalist")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("search_session_id") != session_id:
                continue
            if int(metadata.get("rank") or -1) != rank:
                continue
            metadata["agent_readability_score"] = rating["readability_score"]
            metadata["agent_readability_label"] = rating["label"]
            metadata["agent_readability_rationale"] = rating["rationale"]
            metadata["agent_coherent_clause"] = rating.get("coherent_clause", "")
            metadata["agent_readability_iteration"] = rating.get("iteration")
            branch.metadata["agent_readability_score"] = rating["readability_score"]
            branch.metadata["agent_readability_label"] = rating["label"]
            branch.metadata["agent_readability_rationale"] = rating["rationale"]
            updated.append(branch_name)
        return updated

    def _transform_score_value(self, candidate: dict[str, Any]) -> float | None:
        raw = candidate.get("anneal_score")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _transform_readability_judgment() -> dict[str, Any]:
        return {
            "primary_signal": "agent_semantic_reading",
            "agent_task": (
                "Read decoded_preview using all permitted benchmark/context "
                "knowledge. Judge semantic coherence before trusting numeric "
                "scores: does it form a paraphrasable clause or only word "
                "islands?"
            ),
            "scale": [
                {"score": 4, "label": "coherent_plaintext"},
                {"score": 3, "label": "partial_clause"},
                {"score": 2, "label": "word_islands_with_some_structure"},
                {"score": 1, "label": "word_islands_only"},
                {"score": 0, "label": "garbage"},
            ],
            "record_in_reasoning": (
                "State a 0-4 contextual readability score, label, and short "
                "paraphrase/evidence before repairing, installing more "
                "branches, or declaring."
            ),
        }

    @staticmethod
    def _default_transform_label_for_score(score: float) -> str:
        if score >= 3.5:
            return "coherent_plaintext"
        if score >= 2.5:
            return "partial_clause"
        if score >= 1.5:
            return "word_islands_with_some_structure"
        if score >= 0.5:
            return "word_islands_only"
        return "garbage"

    def _transform_candidate_rating(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        rating = candidate.get("agent_readability_judgment")
        return rating if isinstance(rating, dict) else None

    def _transform_ranking_score(
        self,
        *,
        candidate: dict[str, Any],
        quick_scores: dict[str, Any] | None,
        score_gap_from_best: float | None,
    ) -> dict[str, Any]:
        rating = self._transform_candidate_rating(candidate)
        return {
            "primary": {
                "name": "agent_contextual_readability",
                "value": rating.get("readability_score") if rating else None,
                "label": rating.get("label") if rating else None,
                "rationale": rating.get("rationale") if rating else None,
                "coherent_clause": rating.get("coherent_clause") if rating else None,
                "must_be_supplied_by_agent": rating is None,
                "scale": "0..4",
                "note": (
                    "This is intentionally not computed by the tool. The "
                    "agent must judge whether the preview reads coherently "
                    "given the available context."
                ),
            },
            "supporting": {
                "anneal_score": candidate.get("anneal_score"),
                "score_gap_from_best": score_gap_from_best,
                "validation": candidate.get("validation"),
                "quick_scores": quick_scores,
            },
            "ranking_rule": (
                "Rank finalists by agent_contextual_readability first. Use "
                "finalist validation, anneal_score, quick_scores, and basin "
                "status only as supporting evidence or tie-breakers."
            ),
        }

    def _mirror_transform_rating_to_branches(
        self,
        *,
        session_id: str,
        rank: int,
        rating: dict[str, Any],
    ) -> list[str]:
        updated: list[str] = []
        for branch_name in self.workspace.branch_names():
            branch = self.workspace.get_branch(branch_name)
            metadata = branch.metadata.get("transform_finalist")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("search_session_id") != session_id:
                continue
            if int(metadata.get("rank") or -1) != rank:
                continue
            metadata["agent_readability_score"] = rating["readability_score"]
            metadata["agent_readability_label"] = rating["label"]
            metadata["agent_readability_rationale"] = rating["rationale"]
            metadata["agent_coherent_clause"] = rating.get("coherent_clause", "")
            metadata["agent_readability_iteration"] = rating.get("iteration")
            updated.append(branch_name)
        return updated

    def _install_transform_finalist_branch(
        self,
        *,
        session: dict[str, Any],
        rank: int,
        branch_name: str,
    ) -> dict[str, Any]:
        ranked = session.get("ranked") or []
        if rank < 1 or rank > len(ranked):
            raise WorkspaceError(f"rank {rank} out of range (1..{len(ranked)})")
        candidate = ranked[rank - 1]
        source_branch = str(session["source_branch"])
        if not self.workspace.has_branch(branch_name):
            self.workspace.fork(branch_name, from_branch=source_branch)
        self.workspace.apply_transform_pipeline(branch_name, candidate["pipeline"])
        self.workspace.set_full_key(
            branch_name,
            {int(k): int(v) for k, v in (candidate.get("key") or {}).items()},
        )
        candidate["branch"] = branch_name
        branch = self.workspace.get_branch(branch_name)
        candidate_info = candidate.get("candidate") or {}
        params = candidate_info.get("params") or {}
        source_branch = str(session["source_branch"])
        for tag in (
            "transform_finalist",
            f"transform_rank_{rank}",
            f"transform_profile_{session.get('profile')}",
        ):
            if tag not in branch.tags:
                branch.tags.append(tag)
        branch.metadata["transform_finalist"] = {
            "search_session_id": next(
                (
                    sid for sid, stored in self._transform_search_sessions.items()
                    if stored is session
                ),
                None,
            ),
            "source_branch": source_branch,
            "rank": rank,
            "candidate_id": candidate_info.get("candidate_id"),
            "family": candidate_info.get("family"),
            "template": params.get("template"),
            "operation_labels": params.get("operation_labels", [])[:10],
            "anneal_score": candidate.get("anneal_score"),
            "status": candidate.get("status"),
            "elapsed_seconds": candidate.get("elapsed_seconds"),
            "validation": candidate.get("validation"),
            "score_gap_from_best": (
                round(
                    (self._transform_score_value((session.get("ranked") or [candidate])[0]) or 0.0)
                    - (self._transform_score_value(candidate) or 0.0),
                    4,
                )
                if session.get("ranked") else None
            ),
            "primary_ranking_signal": "agent_contextual_readability",
            "agent_readability_score": (
                (self._transform_candidate_rating(candidate) or {}).get("readability_score")
            ),
            "agent_readability_label": (
                (self._transform_candidate_rating(candidate) or {}).get("label")
            ),
            "agent_readability_rationale": (
                (self._transform_candidate_rating(candidate) or {}).get("rationale")
            ),
            "agent_coherent_clause": (
                (self._transform_candidate_rating(candidate) or {}).get("coherent_clause")
            ),
            "numeric_scores_role": "supporting_evidence",
        }
        score = self._transform_score_value(candidate)
        best_score = self._transform_score_value((session.get("ranked") or [candidate])[0])
        score_gap = (
            round(best_score - score, 4)
            if best_score is not None and score is not None else None
        )
        quick_scores = self._compute_quick_scores(branch_name)
        return {
            "rank": rank,
            "branch": branch_name,
            "candidate_id": (candidate.get("candidate") or {}).get("candidate_id"),
            "anneal_score": candidate.get("anneal_score"),
            "quick_scores": quick_scores,
            "ranking_score": self._transform_ranking_score(
                candidate=candidate,
                quick_scores=quick_scores,
                score_gap_from_best=score_gap,
            ),
            "basin": self._branch_basin_status(branch_name),
            "decoded_preview": self._decoded_preview(branch_name, max_words=40),
        }

    def _install_pure_transposition_finalist_branch(
        self,
        *,
        session: dict[str, Any],
        rank: int,
        branch_name: str,
    ) -> dict[str, Any]:
        ranked = session.get("ranked") or []
        if rank < 1 or rank > len(ranked):
            raise WorkspaceError(f"rank {rank} out of range (1..{len(ranked)})")
        candidate = ranked[rank - 1]
        pipeline = candidate.get("pipeline")
        if not pipeline:
            raise WorkspaceError(f"rank {rank} has no transform pipeline")
        source_branch = str(session["source_branch"])
        if not self.workspace.has_branch(branch_name):
            self.workspace.fork(branch_name, from_branch=source_branch)
        branch = self.workspace.get_branch(branch_name)
        branch.key.clear()
        transform_result = self.workspace.apply_transform_pipeline(branch_name, pipeline)
        plaintext = str(candidate.get("plaintext") or "")
        candidate["branch"] = branch_name
        family = str(candidate.get("family") or "candidate")
        session_id = next(
            (
                sid for sid, stored in self._pure_transposition_sessions.items()
                if stored is session
            ),
            None,
        )
        for tag in (
            "pure_transposition_finalist",
            f"pure_rank_{rank}",
            f"pure_profile_{session.get('profile')}",
            "mode:transposition",
            "transform",
        ):
            if tag not in branch.tags:
                branch.tags.append(tag)
        rating = self._transform_candidate_rating(candidate) or {}
        branch.metadata.update({
            "cipher_mode": "transposition",
            "mode_status": "active",
            "mode_confidence": "medium",
            "mode_evidence": (
                "Installed by search_pure_transposition from direct "
                f"language-scored candidate rank {rank}."
            ),
            "mode_counter_evidence": (
                "This branch assumes order-only transposition. If the text "
                "is still word islands or letter-substitution-like, switch "
                "to a mixed transposition+homophonic hypothesis instead of "
                "doing local word repair."
            ),
            "key_type": "TransformPipelineKey",
            "transform_family": family,
            "transform_candidate_id": candidate.get("candidate_id"),
            "transform_rank": rank,
            "transform_score": candidate.get("score"),
            "transform_selection_score": candidate.get("selection_score"),
            "transform_pipeline": pipeline,
            "decoded_text": plaintext,
            "decoded_text_source": "search_pure_transposition",
            "pure_transposition_finalist": {
                "search_session_id": session_id,
                "source_branch": source_branch,
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "family": family,
                "score": candidate.get("score"),
                "selection_score": candidate.get("selection_score"),
                "validated_selection_score": candidate.get("validated_selection_score"),
                "validation": candidate.get("validation"),
                "primary_ranking_signal": "agent_contextual_readability",
                "agent_readability_score": rating.get("readability_score"),
                "agent_readability_label": rating.get("label"),
                "agent_readability_rationale": rating.get("rationale"),
                "agent_coherent_clause": rating.get("coherent_clause"),
                "numeric_scores_role": "supporting_evidence",
            },
            "search_metadata": {
                "solver": "pure_transposition_screen_rust",
                "profile": session.get("profile"),
                "candidate_count": session.get("candidate_count"),
                "valid_candidate_count": session.get("valid_candidate_count"),
                "candidate_plan": session.get("candidate_plan"),
                "cache": session.get("cache"),
                "source_branch": source_branch,
            },
        })
        score = candidate.get("score")
        best = (session.get("ranked") or [candidate])[0]
        try:
            score_gap = round(float(best.get("score")) - float(score), 4)
        except (TypeError, ValueError):
            score_gap = None
        return {
            "rank": rank,
            "branch": branch_name,
            "candidate_id": candidate.get("candidate_id"),
            "family": family,
            "score": candidate.get("score"),
            "selection_score": candidate.get("selection_score"),
            "validated_selection_score": candidate.get("validated_selection_score"),
            "validation": candidate.get("validation"),
            "score_gap_from_best": score_gap,
            "quick_scores": self._compute_quick_scores(branch_name),
            "ranking_score": self._pure_transposition_ranking_score(
                candidate=candidate,
                score_gap_from_best=score_gap,
            ),
            "basin": self._branch_basin_status(branch_name),
            "decoded_preview": _truncate_text(plaintext, 600),
            "transform_result": transform_result,
        }

    def _pure_transposition_finalist_review(
        self,
        *,
        session_id: str,
        start_rank: int = 1,
        count: int = 5,
        review_chars: int = 600,
        good_score_gap: float = 0.25,
    ) -> dict[str, Any]:
        session = self._pure_transposition_session(session_id)
        if session is None:
            return {"error": f"Unknown pure transposition search session: {session_id}"}
        ranked = list(session.get("ranked") or [])
        if not ranked:
            return {
                "search_session_id": session_id,
                "finalist_review": [],
                "finalist_review_count": 0,
                "total_finalist_count": 0,
            }
        start_rank = max(1, int(start_rank))
        count = max(1, min(int(count), 50))
        review_chars = max(120, min(int(review_chars), 1600))
        good_score_gap = max(0.0, float(good_score_gap))
        try:
            best_score = float(ranked[0].get("score"))
        except (TypeError, ValueError):
            best_score = None
        good_threshold = best_score - good_score_gap if best_score is not None else None
        good_ranked_finalists = [
            (rank, candidate)
            for rank, candidate in enumerate(ranked, start=1)
            if good_threshold is not None
            and candidate.get("score") is not None
            and float(candidate.get("score")) >= good_threshold
        ]
        page_start = start_rank - 1
        page = ranked[page_start:page_start + count]
        finalist_review: list[dict[str, Any]] = []
        for offset, candidate in enumerate(page):
            rank = start_rank + offset
            branch_for_candidate = candidate.get("branch")
            quick_scores = None
            basin = None
            if branch_for_candidate and self.workspace.has_branch(branch_for_candidate):
                quick_scores = self._compute_quick_scores(branch_for_candidate)
                basin = self._branch_basin_status(branch_for_candidate)
            try:
                score_gap = round(best_score - float(candidate.get("score")), 4) if best_score is not None else None
            except (TypeError, ValueError):
                score_gap = None
            if branch_for_candidate:
                recommended_next = "inspect_branch_or_compare_branch_cards"
            else:
                recommended_next = "install_if_preview_promising_or_review_next_page"
            finalist_review.append({
                "rank": rank,
                "branch": branch_for_candidate,
                "candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "params": candidate.get("params"),
                "score": candidate.get("score"),
                "selection_score": candidate.get("selection_score"),
                "validated_selection_score": candidate.get("validated_selection_score"),
                "score_gap_from_best": score_gap,
                "quick_scores": quick_scores,
                "basin": basin,
                "ranking_score": self._pure_transposition_ranking_score(
                    candidate=candidate,
                    score_gap_from_best=score_gap,
                ),
                "pipeline_step_count": len((candidate.get("pipeline") or {}).get("steps", [])),
                "pipeline_columns": (candidate.get("pipeline") or {}).get("columns"),
                "pipeline_rows": (candidate.get("pipeline") or {}).get("rows"),
                "decoded_preview": str(candidate.get("plaintext") or candidate.get("preview") or "")[:review_chars],
                "validation": candidate.get("validation"),
                "readability_judgment": self._transform_readability_judgment(),
                "agent_readability_judgment": self._transform_candidate_rating(candidate),
                "recommended_next": recommended_next,
            })
        reviewed_ranks = set(range(start_rank, start_rank + len(page)))
        more_good_after_page = [
            rank for rank, _candidate in good_ranked_finalists
            if rank not in reviewed_ranks and rank > start_rank + len(page) - 1
        ]
        return {
            "search_session_id": session_id,
            "source_branch": session.get("source_branch"),
            "profile": session.get("profile"),
            "candidate_count": session.get("candidate_count"),
            "valid_candidate_count": session.get("valid_candidate_count"),
            "family_counts": session.get("family_counts", {}),
            "top_family_counts": session.get("top_family_counts", {}),
            "candidate_plan": session.get("candidate_plan"),
            "start_rank": start_rank,
            "count": count,
            "finalist_review": finalist_review,
            "finalist_review_count": len(finalist_review),
            "total_finalist_count": len(ranked),
            "has_more_finalists": start_rank + len(page) <= len(ranked),
            "next_start_rank": (
                start_rank + len(page)
                if start_rank + len(page) <= len(ranked) else None
            ),
            "good_score_gap": good_score_gap,
            "good_score_threshold": round(good_threshold, 4) if good_threshold is not None else None,
            "good_score_finalist_count": len(good_ranked_finalists),
            "more_good_score_finalists": bool(more_good_after_page),
            "more_good_score_finalist_count": len(more_good_after_page),
            "next_good_score_ranks": more_good_after_page[:12],
            "primary_ranking_signal": "agent_contextual_readability",
            "numeric_scores_role": "supporting_evidence",
            "rated_finalist_count": sum(
                1 for candidate in ranked
                if self._transform_candidate_rating(candidate) is not None
            ),
            "review_instruction": (
                "Review these order-only plaintext previews first. If a "
                "preview is coherent, rate it with act_rate_transform_finalist "
                "and install selected ranks with "
                "act_install_pure_transposition_finalists. If all previews are "
                "word islands and the source alphabet suggests homophonic "
                "keying, switch to search_transform_homophonic instead of "
                "local word repair."
            ),
        }

    def _transform_finalist_review(
        self,
        *,
        session_id: str,
        start_rank: int = 1,
        count: int = 5,
        review_chars: int = 600,
        good_score_gap: float = 0.25,
    ) -> dict[str, Any]:
        session = self._transform_session(session_id)
        if session is None:
            return {"error": f"Unknown transform search session: {session_id}"}
        ranked = list(session.get("ranked") or [])
        if not ranked:
            return {
                "search_session_id": session_id,
                "finalist_review": [],
                "finalist_review_count": 0,
                "total_finalist_count": 0,
            }
        start_rank = max(1, int(start_rank))
        count = max(1, min(int(count), 50))
        review_chars = max(120, min(int(review_chars), 1600))
        good_score_gap = max(0.0, float(good_score_gap))
        best_score = self._transform_score_value(ranked[0])
        good_threshold = best_score - good_score_gap if best_score is not None else None
        good_ranked_finalists = [
            (rank, candidate)
            for rank, candidate in enumerate(ranked, start=1)
            if candidate.get("status") == "completed"
            and self._transform_score_value(candidate) is not None
            and good_threshold is not None
            and self._transform_score_value(candidate) >= good_threshold
        ]
        page_start = start_rank - 1
        page = ranked[page_start:page_start + count]
        finalist_review: list[dict[str, Any]] = []
        for offset, candidate in enumerate(page):
            rank = start_rank + offset
            branch_for_candidate = candidate.get("branch")
            quick_scores: dict[str, Any] | None = None
            basin: dict[str, Any] | None = None
            if branch_for_candidate and self.workspace.has_branch(branch_for_candidate):
                quick_scores = self._compute_quick_scores(branch_for_candidate)
                basin = self._branch_basin_status(branch_for_candidate)
            candidate_info = candidate.get("candidate") or {}
            params = candidate_info.get("params") or {}
            pipeline = candidate.get("pipeline") or {}
            score = self._transform_score_value(candidate)
            if basin and basin.get("status") == "coherent_basin":
                recommended_next = "inspect_or_declare_if_preview_reads_coherently"
            elif basin and self._repair_policy_blocks_word_repair(basin):
                recommended_next = "treat_as_word_islands_or_request_more_finalists"
            elif branch_for_candidate:
                recommended_next = "inspect_branch_or_compare_branch_cards"
            else:
                recommended_next = "install_if_preview_promising_or_review_next_page"
            score_gap = (
                round(best_score - score, 4)
                if best_score is not None and score is not None else None
            )
            readability_judgment = self._transform_readability_judgment()
            finalist_review.append({
                "rank": rank,
                "branch": branch_for_candidate,
                "candidate_id": candidate_info.get("candidate_id"),
                "family": candidate_info.get("family"),
                "template": params.get("template"),
                "operation_labels": params.get("operation_labels", [])[:10],
                "status": candidate.get("status"),
                "anneal_score": candidate.get("anneal_score"),
                "score_gap_from_best": score_gap,
                "elapsed_seconds": candidate.get("elapsed_seconds"),
                "quick_scores": quick_scores,
                "basin": basin,
                "ranking_score": self._transform_ranking_score(
                    candidate=candidate,
                    quick_scores=quick_scores,
                    score_gap_from_best=score_gap,
                ),
                "pipeline_step_count": len(pipeline.get("steps", [])),
                "pipeline_columns": pipeline.get("columns"),
                "pipeline_rows": pipeline.get("rows"),
                "decoded_preview": str(candidate.get("decoded_preview") or "")[:review_chars],
                "validation": candidate.get("validation"),
                "readability_judgment": readability_judgment,
                "agent_readability_judgment": self._transform_candidate_rating(candidate),
                "recommended_next": recommended_next,
            })
        reviewed_ranks = set(range(start_rank, start_rank + len(page)))
        more_good_after_page = [
            rank for rank, _candidate in good_ranked_finalists
            if rank not in reviewed_ranks and rank > start_rank + len(page) - 1
        ]
        return {
            "search_session_id": session_id,
            "source_branch": session.get("source_branch"),
            "profile": session.get("profile"),
            "columns": session.get("columns"),
            "start_rank": start_rank,
            "count": count,
            "finalist_review": finalist_review,
            "finalist_review_count": len(finalist_review),
            "total_finalist_count": len(ranked),
            "has_more_finalists": start_rank + len(page) <= len(ranked),
            "next_start_rank": (
                start_rank + len(page)
                if start_rank + len(page) <= len(ranked) else None
            ),
            "good_score_gap": good_score_gap,
            "good_score_threshold": (
                round(good_threshold, 4)
                if good_threshold is not None else None
            ),
            "good_score_finalist_count": len(good_ranked_finalists),
            "more_good_score_finalists": bool(more_good_after_page),
            "more_good_score_finalist_count": len(more_good_after_page),
            "next_good_score_ranks": more_good_after_page[:12],
            "primary_ranking_signal": "agent_contextual_readability",
            "numeric_scores_role": "supporting_evidence",
            "rated_finalist_count": sum(
                1 for candidate in ranked
                if self._transform_candidate_rating(candidate) is not None
            ),
            "review_instruction": (
                "Review these previews first. To inspect more without "
                "rerunning search, call search_review_transform_finalists "
                "with next_start_rank. To install promising candidates, call "
                "act_install_transform_finalists with the desired ranks. "
                "Use act_rate_transform_finalist to record your contextual "
                "readability score for promising or rejected finalists; that "
                "agent judgment is the primary ranking signal and numeric "
                "scores are supporting evidence."
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

    def _tool_act_rate_transform_finalist(self, args: dict) -> Any:
        session_id = str(args["search_session_id"])
        session = self._transform_session(session_id)
        if session is None:
            pure_session = self._pure_transposition_session(session_id)
            if pure_session is None:
                return {"error": f"Unknown transform search session: {session_id}"}
            ranked = pure_session.get("ranked") or []
            rank = int(args["rank"])
            if rank < 1 or rank > len(ranked):
                return {
                    "error": f"rank {rank} out of range (1..{len(ranked)})",
                    "search_session_id": session_id,
                }
            score = float(args["readability_score"])
            if score < 0 or score > 4:
                return {"error": "readability_score must be between 0 and 4"}
            allowed_labels = {
                "coherent_plaintext",
                "partial_clause",
                "word_islands_with_some_structure",
                "word_islands_only",
                "garbage",
            }
            label = str(args.get("label") or self._default_transform_label_for_score(score))
            if label not in allowed_labels:
                return {
                    "error": f"label must be one of {sorted(allowed_labels)}",
                    "search_session_id": session_id,
                }
            rationale = str(args.get("rationale") or "").strip()
            if not rationale:
                return {"error": "rationale is required"}
            coherent_clause = str(args.get("coherent_clause") or "").strip()
            candidate = ranked[rank - 1]
            rating = {
                "readability_score": round(score, 2),
                "label": label,
                "rationale": rationale,
                "coherent_clause": coherent_clause,
                "iteration": self._current_iteration,
                "primary_ranking_signal": "agent_contextual_readability",
            }
            candidate["agent_readability_judgment"] = rating
            updated_branches = self._mirror_pure_rating_to_branches(
                session_id=session_id,
                rank=rank,
                rating=rating,
            )
            review = self._pure_transposition_finalist_review(
                session_id=session_id,
                start_rank=rank,
                count=1,
            )
            return {
                "status": "ok",
                "search_session_id": session_id,
                "session_type": "pure_transposition",
                "rank": rank,
                "rating": rating,
                "updated_branches": updated_branches,
                "finalist": (review.get("finalist_review") or [None])[0],
                "note": (
                    "Recorded the agent's contextual readability judgment. "
                    "For pure transposition this judgment is the primary "
                    "ranking signal; numeric direct-language scores remain "
                    "supporting evidence."
                ),
            }
        ranked = session.get("ranked") or []
        rank = int(args["rank"])
        if rank < 1 or rank > len(ranked):
            return {
                "error": f"rank {rank} out of range (1..{len(ranked)})",
                "search_session_id": session_id,
            }
        score = float(args["readability_score"])
        if score < 0 or score > 4:
            return {"error": "readability_score must be between 0 and 4"}
        allowed_labels = {
            "coherent_plaintext",
            "partial_clause",
            "word_islands_with_some_structure",
            "word_islands_only",
            "garbage",
        }
        label = str(args.get("label") or self._default_transform_label_for_score(score))
        if label not in allowed_labels:
            return {
                "error": f"label must be one of {sorted(allowed_labels)}",
                "search_session_id": session_id,
            }
        rationale = str(args.get("rationale") or "").strip()
        if not rationale:
            return {"error": "rationale is required"}
        coherent_clause = str(args.get("coherent_clause") or "").strip()
        candidate = ranked[rank - 1]
        rating = {
            "readability_score": round(score, 2),
            "label": label,
            "rationale": rationale,
            "coherent_clause": coherent_clause,
            "iteration": self._current_iteration,
            "primary_ranking_signal": "agent_contextual_readability",
        }
        candidate["agent_readability_judgment"] = rating
        updated_branches = self._mirror_transform_rating_to_branches(
            session_id=session_id,
            rank=rank,
            rating=rating,
        )
        review = self._transform_finalist_review(
            session_id=session_id,
            start_rank=rank,
            count=1,
        )
        return {
            "status": "ok",
            "search_session_id": session_id,
            "rank": rank,
            "rating": rating,
            "updated_branches": updated_branches,
            "finalist": (review.get("finalist_review") or [None])[0],
            "note": (
                "Recorded the agent's contextual readability judgment. This "
                "rating is now the finalist's primary ranking signal; numeric "
                "anneal/dictionary scores remain supporting evidence."
            ),
        }

    def _tool_act_install_transform_finalists(self, args: dict) -> Any:
        session_id = str(args["search_session_id"])
        session = self._transform_session(session_id)
        if session is None:
            return {"error": f"Unknown transform search session: {session_id}"}
        ranks = args.get("ranks") or []
        if not isinstance(ranks, list) or not ranks:
            return {"error": "ranks must be a non-empty list of 1-based finalist ranks"}
        source_branch = str(session["source_branch"])
        prefix = str(args.get("branch_prefix") or f"{source_branch}_transform_rank")
        installed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for raw_rank in ranks:
            try:
                rank = int(raw_rank)
                branch_name = f"{prefix}{rank}"
                installed.append(self._install_transform_finalist_branch(
                    session=session,
                    rank=rank,
                    branch_name=branch_name,
                ))
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "rank": raw_rank,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        return {
            "status": "ok" if installed else "error",
            "search_session_id": session_id,
            "installed_count": len(installed),
            "installed": installed,
            "errors": errors,
            "note": (
                "Installed selected transform finalists as branches without "
                "rerunning the search. Use workspace_branch_cards to compare "
                "installed branches if the finalist_review was ambiguous."
            ),
        }

    def _tool_search_review_transform_finalists(self, args: dict) -> Any:
        return self._transform_finalist_review(
            session_id=str(args["search_session_id"]),
            start_rank=int(args.get("start_rank", 1)),
            count=int(args.get("count", 5)),
            review_chars=int(args.get("review_chars", 600)),
            good_score_gap=float(args.get("good_score_gap", 0.25)),
        )

    def _tool_search_review_pure_transposition_finalists(self, args: dict) -> Any:
        return self._pure_transposition_finalist_review(
            session_id=str(args["search_session_id"]),
            start_rank=int(args.get("start_rank", 1)),
            count=int(args.get("count", 5)),
            review_chars=int(args.get("review_chars", 600)),
            good_score_gap=float(args.get("good_score_gap", 0.25)),
        )

    def _tool_act_install_pure_transposition_finalists(self, args: dict) -> Any:
        session_id = str(args["search_session_id"])
        session = self._pure_transposition_session(session_id)
        if session is None:
            return {"error": f"Unknown pure transposition search session: {session_id}"}
        ranks = args.get("ranks") or []
        if not isinstance(ranks, list) or not ranks:
            return {"error": "ranks must be a non-empty list of 1-based finalist ranks"}
        source_branch = str(session["source_branch"])
        prefix = str(args.get("branch_prefix") or f"{source_branch}_pure_rank")
        installed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for raw_rank in ranks:
            try:
                rank = int(raw_rank)
                branch_name = f"{prefix}{rank}"
                if self.workspace.has_branch(branch_name):
                    branch_name = self._unique_branch_name(branch_name)
                installed.append(self._install_pure_transposition_finalist_branch(
                    session=session,
                    rank=rank,
                    branch_name=branch_name,
                ))
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "rank": raw_rank,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        return {
            "status": "ok" if installed else "error",
            "search_session_id": session_id,
            "installed_count": len(installed),
            "installed": installed,
            "errors": errors,
            "note": (
                "Installed selected pure-transposition finalists as readable "
                "transform branches without rerunning the Rust screen. Rate "
                "the previews with act_rate_transform_finalist first when the "
                "menu is ambiguous."
            ),
        }

    def _tool_search_transform_homophonic(self, args: dict) -> Any:
        branch_name = args["branch"]
        if not self.workspace.has_branch(branch_name):
            return {"error": f"Branch not found: {branch_name}"}
        branch = self.workspace.get_branch(branch_name)
        if branch.token_order is not None:
            return {
                "status": "blocked",
                "reason": "transform_branch_not_supported",
                "branch": branch_name,
                "active_transform_pipeline": branch.transform_pipeline,
                "note": (
                    "`search_transform_homophonic` must start from an "
                    "untransformed branch. Searching again on a branch that "
                    "already has a token-order transform would compose orders, "
                    "but this tool cannot safely write that composite branch "
                    "yet. Run the broader transform search from `main`, or "
                    "use `search_homophonic_anneal` to polish this transformed "
                    "branch's homophonic key."
                ),
                "suggested_next_tools": [
                    "search_transform_homophonic(branch='main', profile='medium' or 'wide')",
                    "search_homophonic_anneal",
                    "workspace_branch_cards",
                ],
            }
        columns = args.get("columns")
        columns = int(columns) if columns is not None else None
        profile = str(args.get("profile", "small"))
        top_n = max(1, int(args.get("top_n", 3)))
        write_best_branch = bool(args.get("write_best_branch", True))
        write_candidate_branches = bool(args.get("write_candidate_branches", False))
        candidate_branch_count = max(0, int(args.get("candidate_branch_count", 3)))
        review_chars = max(120, min(int(args.get("review_chars", 600)), 1600))
        good_score_gap = max(0.0, float(args.get("good_score_gap", 0.25)))
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
            decoded_preview = result.final_decryption[:800]
            ranked.append({
                "candidate_index": index,
                "candidate": candidate,
                "pipeline": pipeline.to_raw(),
                "status": result.status,
                "solver": result.solver,
                "anneal_score": primary_step.get("anneal_score"),
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "decoded_preview": decoded_preview,
                "validation": validate_plaintext_finalist(
                    decoded_preview,
                    language=self.language,
                    word_set=self.word_set,
                    word_list=self.word_list,
                ),
                "key": artifact.get("key") or {},
            })
        ranked.sort(
            key=lambda item: (
                item.get("status") == "completed",
                float(item.get("anneal_score") or float("-inf")),
            ),
            reverse=True,
        )
        search_session_id = self._new_transform_search_session(
            source_branch=branch_name,
            profile=profile,
            columns=columns,
            ranked=ranked,
            structural_screen=structural_screen,
        )
        session = self._transform_session(search_session_id)
        assert session is not None
        written_branch = None
        written_candidate_branches: list[str] = []

        if write_best_branch and ranked:
            written_branch = f"{branch_name}_transform_best"
            self._install_transform_finalist_branch(
                session=session,
                rank=1,
                branch_name=written_branch,
            )

        if write_candidate_branches and ranked and candidate_branch_count:
            for rank in range(1, min(candidate_branch_count, len(ranked)) + 1):
                branch_out = f"{branch_name}_transform_rank{rank}"
                self._install_transform_finalist_branch(
                    session=session,
                    rank=rank,
                    branch_name=branch_out,
                )
                written_candidate_branches.append(branch_out)
        review_limit = max(
            top_n,
            candidate_branch_count if write_candidate_branches else 0,
        )
        review = self._transform_finalist_review(
            session_id=search_session_id,
            start_rank=1,
            count=review_limit,
            review_chars=review_chars,
            good_score_gap=good_score_gap,
        )
        return {
            "branch": branch_name,
            "profile": profile,
            "columns": columns,
            "search_session_id": search_session_id,
            "candidate_count": structural_screen.get("deduped_candidate_count", 0),
            "structural_screen": structural_screen,
            "skipped_candidates": skipped[:20],
            "written_branch": written_branch,
            "written_candidate_branches": written_candidate_branches,
            "top_candidates": ranked[:top_n],
            **review,
            "note": (
                "This is a bounded screen over simple transform candidates. "
                "Read finalist_review and compare transformed branches before declaring. "
                "Use search_review_transform_finalists to page through more "
                "finalists, and act_install_transform_finalists to install "
                "specific ranks without rerunning search."
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
        unrated_transform = self._unrated_transform_finalist_metadata(branch)
        if unrated_transform is not None:
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "transform_finalist_readability_required",
                "transform_finalist": unrated_transform,
                "note": (
                    "This branch came from transform finalist search, but the "
                    "agent has not recorded its contextual readability score. "
                    "Before declaring, call act_rate_transform_finalist for "
                    "this search_session_id/rank. The rating should say "
                    "whether the text is coherent prose, a partial clause, "
                    "word islands, or garbage; numeric scores are not enough."
                ),
                "suggested_next_tools": [
                    "act_rate_transform_finalist",
                    "search_review_transform_finalists",
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ],
            }
        if len(self.workspace.branch_names()) > 1 and not self._has_seen_branch_cards(branch):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "branch_cards_required",
                "note": (
                    "Multiple branches exist, or this branch was created after "
                    "the last branch-card review. Call workspace_branch_cards "
                    "before declaring so you compare readable excerpts, "
                    "internal scores, repairs, and orthography risks."
                ),
                "suggested_next_tools": [
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ],
            }
        hypothesis_block = self._hypothesis_declaration_block(branch)
        if hypothesis_block is not None:
            return hypothesis_block
        family_coverage_block = self._family_coverage_declaration_block(
            branch,
            forced_partial=forced_partial,
        )
        if family_coverage_block is not None:
            return family_coverage_block
        word_boundary_block = self._word_boundary_declaration_block(
            branch,
            confidence=confidence,
            forced_partial=forced_partial,
        )
        if word_boundary_block is not None:
            return word_boundary_block
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
                    "Before declaring, inspect transform state and run the "
                    "mode-appropriate transform screen: "
                    "search_pure_transposition for transposition-only, or "
                    "search_transform_homophonic for Zodiac/Z340-style "
                    "transposition+homophonic. If that screen is not useful, "
                    "declare again with that negative result in the rationale."
                ),
                "suggested_next_tools": [
                    "observe_transform_pipeline",
                    "observe_transform_suspicion",
                    "search_pure_transposition",
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
                "suggested_next_tools": self._mode_scoped_suggestions(branch, [
                    "observe_transform_pipeline",
                    "observe_transform_suspicion",
                    "search_transform_homophonic",
                    "search_automated_solver",
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ]),
            }
        if self._should_guard_more_work_declaration(
            further_iterations_helpful is True,
            forced_partial,
        ):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "further_iterations_requested",
                "note": (
                    "Your declaration says further iterations would be helpful, "
                    "and this run still has iteration budget. Continue working "
                    "instead of terminating early. `forced_partial=true` does "
                    "not override this: if you can name useful next work, take "
                    "that bigger swing before submitting a partial/hypothesis "
                    "result."
                ),
                "suggested_next_tools": self._mode_scoped_suggestions(branch, [
                    "workspace_branch_cards",
                    "decode_diagnose",
                    "search_homophonic_anneal",
                    "observe_transform_suspicion",
                    "meta_declare_solution",
                ]),
            }
        if self._should_guard_premature_partial_declaration(
            confidence,
            forced_partial,
        ):
            return {
                "status": "blocked",
                "accepted": False,
                "branch": branch,
                "reason": "partial_too_early",
                "note": (
                    "This is an early low-confidence partial declaration. "
                    "Keep working and take a bigger swing before stopping: "
                    "try a broader transform screen, promote/polish the best "
                    "transformed branch, run a fresh automated route, or make "
                    "a concrete reading-driven repair. Save forced partial "
                    "declarations for the final stretch of the run unless the "
                    "text is already plausibly readable."
                ),
                "suggested_next_tools": self._mode_scoped_suggestions(branch, [
                    "observe_transform_suspicion",
                    "search_transform_homophonic",
                    "search_homophonic_anneal",
                    "search_automated_solver",
                    "workspace_branch_cards",
                    "meta_declare_solution",
                ]),
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
            "context_cipher_family_assumptions": self._context_cipher_family_assumptions(),
            "context_cipher_family_overrides": self.context_family_overrides,
            "note": "Run will terminate after this tool result is recorded.",
        }

    def _tool_meta_declare_unsolved(self, args: dict) -> Any:
        further_iterations_helpful = bool(args.get("further_iterations_helpful"))
        if (
            further_iterations_helpful
            and self.max_iterations is not None
            and self._current_iteration < self.max_iterations
        ):
            return {
                "status": "blocked",
                "accepted": False,
                "reason": "unsolved_but_more_work_requested",
                "note": (
                    "You are trying to stop as unsolved while also saying more "
                    "iterations would help and there is still iteration budget. "
                    "Take the next high-leverage action instead, or call this "
                    "again only when the remaining work is genuinely not useful."
                ),
                "suggested_next_tools": [
                    "workspace_hypothesis_next_steps",
                    "workspace_branch_cards",
                    "search_quagmire3_keyword_alphabet",
                    "search_transform_homophonic",
                ],
            }
        reports = []
        for name in self.workspace.branch_names():
            if not self.workspace.has_branch(name):
                continue
            branch = self.workspace.get_branch(name)
            if not branch.metadata.get("cipher_mode") and "hypothesis" not in branch.tags:
                continue
            if branch.metadata.get("mode_status", "active") in {"rejected", "superseded"}:
                continue
            pending = self._pending_required_tools_for_branch(name)
            if pending["pending_required_tools"]:
                reports.append({
                    "branch": name,
                    "cipher_mode": branch.metadata.get("cipher_mode"),
                    **pending,
                })
        if reports and not (
            self.max_iterations is not None and self._current_iteration >= self.max_iterations
        ):
            pending_tools = sorted({
                tool
                for report in reports
                for tool in report.get("pending_required_tools", [])
            })
            return {
                "status": "blocked",
                "accepted": False,
                "reason": "family_coverage_pending_before_unsolved",
                "pending_family_coverage": reports,
                "note": (
                    "Do not declare the run unsolved while an active "
                    "cipher-family hypothesis still has required higher-level "
                    "work pending. Run the pending tools, then stop as unsolved "
                    "if no coherent branch emerges."
                ),
                "suggested_next_tools": [
                    *pending_tools,
                    "workspace_hypothesis_next_steps",
                    "workspace_branch_cards",
                    "meta_declare_unsolved",
                ],
            }
        best_branch = str(args.get("best_branch") or "").strip() or None
        if best_branch and not self.workspace.has_branch(best_branch):
            return {"error": f"Branch not found: {best_branch}"}
        self.unsolved_declaration = {
            "rationale": str(args["rationale"]),
            "branches_considered": list(args.get("branches_considered") or []),
            "best_branch": best_branch,
            "reading_summary": str(args.get("reading_summary") or ""),
            "further_iterations_helpful": further_iterations_helpful,
            "further_iterations_note": str(args.get("further_iterations_note") or ""),
            "declared_at_iteration": self._current_iteration,
        }
        self.terminated = True
        return {
            "status": "ok",
            "accepted": True,
            "outcome": "unsolved",
            "declared_at_iteration": self._current_iteration,
            "unsolved_declaration": self.unsolved_declaration,
            "note": "Run will terminate after this tool result is recorded as unsolved.",
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
