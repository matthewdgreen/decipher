"""Composite hypothesis-action tests (M3 Parts 2-5, Part 8).

Covers the Phase 6 apply_reading matrix, the hypothesis_test_word parity /
injected / error / install paths, branch_adjudicate, the dispatcher contract,
and the v2-untouched pins.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import TOOL_DEFINITIONS, VALID_TOOL_NAMES, NoGatesPolicy, WorkspaceToolExecutor
from analysis import dictionary as dictionary_module
from automated import runner as automated_runner
from investigation.actions import (
    COMPOSITE_TOOL_DEFINITIONS,
    COMPOSITE_TOOL_NAMES,
    execute_composite,
)
from investigation.episodes import EPISODE_KINDS
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# The seven v2 boundary actuators that must appear in NO episode toolset (Scope).
BOUNDARY_ACTUATORS = frozenset({
    "act_split_cipher_word", "act_merge_cipher_words", "act_merge_decoded_words",
    "act_apply_boundary_candidate", "act_resegment_by_reading",
    "act_resegment_from_reading_repair", "act_resegment_window_by_reading",
})


def _synthetic_executor(raw="abc de", sep=" ", key=None, unmapped=(), words=None):
    """A tiny known-key monoalphabetic executor (default decode: CAT ON)."""
    alpha = Alphabet.from_text(raw, ignore_chars={sep} if sep else set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=sep)
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    default_key = {"a": "C", "b": "A", "c": "T", "d": "O", "e": "N"}
    key = key or default_key
    for sym, letter in key.items():
        if sym in unmapped:
            continue
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    ex = WorkspaceToolExecutor(
        ws, "en", set(words or ["CAT", "CAR", "ON", "CATON"]), ["CAT"], {},
        declaration_policy=NoGatesPolicy(),
    )
    ex.set_iteration(1)
    return ex


_ENGLISH_PLAIN = (
    "THE QUICK BROWN FOXES JUMPED OVER THE LAZY SLEEPING HOUNDS WHILE "
    "SEVERAL PEOPLE WATCHED FROM THEIR HOUSES NEAR THE RIVER"
)


def _english_basin(damage_symbol="r", damage_to="X"):
    """A fully-mapped English substitution basin with one damaged mapping.

    Cipher symbol = shifted-lowercase of the plaintext letter; the correct key
    maps back. One symbol is damaged so the decode has a repairable word.
    """
    def enc(ch):
        return chr(ord("a") + (ord(ch) - 65))
    words = _ENGLISH_PLAIN.split()
    raw = " ".join("".join(enc(c) for c in w) for w in words)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    for sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(sym),
                       pt.id_for(chr(ord("A") + (ord(sym) - ord("a")))))
    ws.set_mapping("main", alpha.id_for(damage_symbol), pt.id_for(damage_to))
    ex = WorkspaceToolExecutor(
        ws, "en",
        dictionary_module.load_word_set(dictionary_module.get_dictionary_path("en")),
        [], {}, declaration_policy=NoGatesPolicy(),
    )
    ex.set_iteration(1)
    return ex, alpha


def _apply(ex, args, state_readings=None, turn=1):
    return execute_composite("hypothesis_apply_reading", args, executor=ex,
                             state_readings=state_readings or {}, turn=turn)


def _test_word(ex, args, state_readings=None, turn=1):
    return execute_composite("hypothesis_test_word", args, executor=ex,
                             state_readings=state_readings or {}, turn=turn)


# ---------------------------------------------------------------------------
# A5a pins: v2 untouched
# ---------------------------------------------------------------------------
def test_composite_names_disjoint_from_v2_and_tool_count_pinned():
    assert COMPOSITE_TOOL_NAMES == {
        "hypothesis_apply_reading", "hypothesis_test_word", "branch_adjudicate"
    }
    assert COMPOSITE_TOOL_NAMES.isdisjoint(VALID_TOOL_NAMES)
    # v2 tool surface is unchanged (composites live only on v3 surfaces).
    # Pinned to the post-variant-registry count (main added the model-variant
    # tools); composites still add zero to TOOL_DEFINITIONS.
    assert len(TOOL_DEFINITIONS) == 94


def test_v2_executor_never_sees_composites():
    ex = _synthetic_executor()
    out = json.loads(ex.execute("hypothesis_test_word", {"branch": "main", "word": "CAT"}))
    assert "Unknown tool" in out["error"]


def test_boundary_actuators_absent_from_every_episode_toolset():
    for kind, entry in EPISODE_KINDS.items():
        assert BOUNDARY_ACTUATORS.isdisjoint(entry["toolset"]), kind


# ---------------------------------------------------------------------------
# Phase 6 matrix — hypothesis_apply_reading
# ---------------------------------------------------------------------------
def test_phase6_a_char_preserving_boundaries_only():
    ex = _synthetic_executor()
    res = _apply(ex, {"branch": "main", "reading_text": "CAT ON"})
    assert res["status"] == "ok"
    assert res["character_preserving"] is True
    assert res["edits"] == []
    assert res["alignment"]["mismatches"] == 0
    # A fork was created, decode unchanged.
    assert res["fork"] and ex.workspace.has_branch(res["fork"])
    # _reading_validation agrees on character-preservation.
    assert ex._reading_validation("main", "CAT ON")["character_preserving"] is True


def test_phase6_b_char_changing_same_count_edit():
    ex = _synthetic_executor()
    res = _apply(ex, {"branch": "main", "reading_text": "CAR ON"})
    assert res["status"] == "ok"
    assert res["character_preserving"] is False
    assert res["edits"] == ["c=R"]        # c: T -> R
    assert res["conflicts"] == []
    assert "CAR" in ex._decoded_preview(res["fork"])


def test_phase6_b_conflict_majority_and_tie():
    # Cipher 'a a a a' -> decode 'X X X X' (symbol a). Propose 'PPQR': symbol a
    # votes P (x2), Q (x1), R (x1) -> majority P; a second symbol with a tie.
    ex = _synthetic_executor(
        raw="aaaa bb", sep=" ",
        key={"a": "X", "b": "Y"}, words=["PPQR"],
    )
    res = _apply(ex, {"branch": "main", "reading_text": "PPQR YZ"})
    assert res["status"] == "ok"
    # symbol a: majority winner P (reported as majority conflict).
    a_edit = [e for e in res["edits"] if e.startswith("a=")]
    assert a_edit == ["a=P"]
    majority = [c for c in res["conflicts"] if c["symbol"] == "a"]
    assert majority and majority[0]["reason"] == "majority" and majority[0]["chosen"] == "P"
    # symbol b: 'bb' decodes 'YY', proposed 'YZ' -> b votes Z once only (Y==Y no
    # change) => single edit, no tie. Use a dedicated tie construction instead:
    tie = _synthetic_executor(raw="bb", sep=None, key={"b": "Y"}, words=["PQ"])
    tie_res = _apply(tie, {"branch": "main", "reading_text": "PQ"})
    # symbol b votes P and Q equally -> tie -> dropped, reported.
    assert all(not e.startswith("b=") for e in tie_res["edits"])
    tie_conflict = [c for c in tie_res["conflicts"] if c["symbol"] == "b"]
    assert tie_conflict and tie_conflict[0]["reason"] == "tie"


def test_phase6_narrow_fragment_keeps_untouched_boundaries():
    """F1 regression: a fragment narrower than the (default whole-text) window
    must NOT delete boundaries between words it does not cover."""
    ex = _synthetic_executor(
        raw="ab cd ef", sep=" ",
        key={"a": "C", "b": "A", "c": "T", "d": "O", "e": "N", "f": "C"},
        words=["CA", "TO", "NC"],
    )
    assert ex._decoded_preview("main") == "CA | TO | NC"
    before = ex.workspace.effective_word_spans("main")  # [(0,2),(2,4),(4,6)]
    # Char-preserving fragment covering ONLY the first word, no call-level window.
    res = _apply(ex, {"branch": "main",
                      "fragments": [{"start": 0, "end": 2, "text": "CA"}]})
    assert res["status"] == "ok"
    assert res["character_preserving"] is True
    assert res["edits"] == []
    assert res["holes"] == []
    fork_spans = ex.workspace.effective_word_spans(res["fork"])
    assert fork_spans == before                 # TO|NC boundary preserved
    assert res["boundary_change_count"] == 0


def test_f6_composites_reject_transform_and_decoded_text_branches():
    # token_order set -> unsupported (both composites).
    ex = _synthetic_executor(raw="abcde", sep=None)
    ex.workspace.get_branch("main").token_order = [4, 3, 2, 1, 0]
    r1 = _apply(ex, {"branch": "main", "reading_text": "NOTAC"})
    assert r1["status"] == "unsupported" and "token_order" in r1["error"]
    r2 = _test_word(ex, {"branch": "main", "char_start": 0, "word": "NOTAC"})
    assert r2["status"] == "unsupported" and "token_order" in r2["error"]
    # decoded_text metadata present -> unsupported.
    ex2 = _synthetic_executor(raw="abcde", sep=None)
    ex2.workspace.get_branch("main").metadata["decoded_text"] = "CATON"
    r3 = _apply(ex2, {"branch": "main", "reading_text": "CATON"})
    assert r3["status"] == "unsupported" and "decoded_text" in r3["error"]
    r4 = _test_word(ex2, {"branch": "main", "word_index": 0, "word": "CATON"})
    assert r4["status"] == "unsupported" and "decoded_text" in r4["error"]


def test_phase6_c_window_scoped_outside_byte_identical():
    # decode: CAT ON  (tokens a b c | d e). Scope the window to tokens [3,5) and
    # change only symbol e; tokens/spans/key outside must stay byte-identical.
    ex = _synthetic_executor()
    before_span = ex.workspace.effective_word_spans("main")
    res = _apply(ex, {"branch": "main", "reading_text": "OK",
                      "window": {"start": 3, "end": 5}})
    assert res["status"] == "ok"
    assert res["edits"] == ["e=K"]        # only symbol e touched (d->O stays)
    fork = ex.workspace.get_branch(res["fork"])
    # symbol 'a' (outside window) unchanged in the fork key.
    alpha = ex.workspace.cipher_text.alphabet
    assert fork.key[alpha.id_for("a")] == ex.workspace.get_branch("main").key[alpha.id_for("a")]
    # spans outside the window [0,3) unchanged.
    fork_spans = ex.workspace.effective_word_spans(res["fork"])
    assert fork_spans[0] == before_span[0] == (0, 3)


def test_phase6_d_banded_miscount_reports_hole_no_cascade():
    ex = _synthetic_executor(raw="abcde", sep=None)  # decode CATON (5 tokens)
    res = _apply(ex, {"branch": "main", "reading_text": "CATTON"})  # +1 doubled T
    assert res["status"] == "ok"
    assert res["fragments"][0]["mode"] == "banded"
    assert res["alignment"]["gaps"] == 1
    assert res["holes"] and "inserted" in res["holes"][0]
    # No spurious edits (identical-run gap does not cascade into mismatches).
    assert res["edits"] == []


def test_phase6_e_count_mismatch_too_large():
    ex = _synthetic_executor(raw="abcde", sep=None)  # span_len 5, tolerance 2
    res = _apply(ex, {"branch": "main", "reading_text": "CATONABCDEFGHIJK"})
    assert res["status"] == "error"
    assert res["error"] == "count_mismatch_too_large"
    assert res["span_len"] == 5 and res["proposed_len"] == 16
    assert "first_divergence" in res


def test_phase6_f_multisym_guard():
    alpha = Alphabet(["S001", "S002", "S003"])
    ct = CipherText(raw="S001 S002 S003", alphabet=alpha, separator=" ")
    ws = Workspace(ct, plaintext_alphabet=Alphabet(["AA", "BB", "CC"]))
    ex = WorkspaceToolExecutor(ws, "en", set(), [], {}, declaration_policy=NoGatesPolicy())
    ex.set_iteration(1)
    res = _apply(ex, {"branch": "main", "reading_text": "AA"})
    assert res["status"] == "unsupported"
    assert "single-character plaintext alphabets only" in res["error"]


def test_phase6_g_question_heavy_banded_stream():
    # Several unmapped symbols (?), plus a +1 count mismatch -> banded path.
    # decode: 'C?T?N' (b,d unmapped); propose 'CATTON' (extra T).
    ex = _synthetic_executor(raw="abcde", sep=None, unmapped=("b", "d"))
    assert "".join(ex._decoded_words("main")) == "C?T?N"
    res = _apply(ex, {"branch": "main", "reading_text": "CATTON"})
    assert res["status"] == "ok"
    assert res["fragments"][0]["mode"] == "banded"
    # Deterministic gaps-late alignment: the two unmapped positions that align
    # get filled (b->A, d->T); the extra proposed char lands last as a hole.
    assert set(res["edits"]) == {"b=A", "d=T"}
    assert res["alignment"]["gaps"] == 1
    assert res["holes"] and "inserted 'O'" in res["holes"][0]


def test_f9_banded_boundary_reprojection_through_gap():
    """F9: a word boundary in a banded (count-mismatched) fragment re-projects
    to a token index through the alignment (boundary-at-gap rule)."""
    ex = _synthetic_executor(raw="abcde", sep=None)  # single word, decode CATON
    res = _apply(ex, {"branch": "main", "reading_text": "CAT TON"})  # +1, split@3
    assert res["status"] == "ok"
    assert res["fragments"][0]["mode"] == "banded"
    assert res["edits"] == []              # identical-run gap -> no edits
    assert res["holes"]                    # the extra char is a hole
    # The proposed CAT|TON boundary re-projects to token 3 -> spans (0,3),(3,5).
    assert ex.workspace.effective_word_spans(res["fork"]) == [(0, 3), (3, 5)]


def test_apply_reading_dry_run_creates_nothing():
    ex = _synthetic_executor()
    before = set(ex.workspace.branch_names())
    res = _apply(ex, {"branch": "main", "reading_text": "CAR ON", "dry_run": True})
    assert res["status"] == "ok"
    assert res["dry_run"] is True
    assert res["fork"] is None
    assert res["edits"] == ["c=R"]         # still computed
    assert set(ex.workspace.branch_names()) == before   # nothing created


def test_apply_reading_from_stored_reading_id():
    from investigation.reading import Reading, ReadingFragment
    ex = _synthetic_executor()
    reading = Reading(branch="main", source="episode:x",
                      fragments=[ReadingFragment(text="CAR ON")])
    readings = {reading.reading_id: reading.to_dict()}
    res = _apply(ex, {"branch": "main", "reading_id": reading.reading_id},
                 state_readings=readings)
    assert res["status"] == "ok"
    assert res["edits"] == ["c=R"]
    assert res["fork"].startswith(f"reading_{reading.reading_id[:6]}_main")


def test_apply_reading_fragment_outside_window_is_error():
    ex = _synthetic_executor(raw="abcde", sep=None)
    res = _apply(ex, {
        "branch": "main",
        "window": {"start": 0, "end": 3},
        "fragments": [{"start": 2, "end": 5, "text": "TON"}],  # end 5 > window 3
    })
    assert res["status"] == "error"
    assert "outside the window" in res["error"]
    assert res["fragment_index"] == 0


def test_apply_reading_unknown_reading_id():
    ex = _synthetic_executor()
    res = _apply(ex, {"branch": "main", "reading_id": "nope"})
    assert res["status"] == "error"
    assert "unknown reading_id" in res["error"]


def test_apply_reading_requires_exactly_one_source():
    ex = _synthetic_executor()
    both = _apply(ex, {"branch": "main", "reading_id": "x", "reading_text": "CAT ON"})
    assert both["status"] == "error"
    neither = _apply(ex, {"branch": "main"})
    assert neither["status"] == "error"


# ---------------------------------------------------------------------------
# hypothesis_test_word — parity, injected, install, errors
# ---------------------------------------------------------------------------
def test_test_word_menu_backed_parity():
    ex, alpha = _english_basin()
    # Build the menu directly the way the tool does, to find a representable word.
    menu = automated_runner.build_word_repair_menu(
        cipher_text=ex.workspace.cipher_text,
        base_key=dict(ex.workspace.get_branch("main").key),
        mask=(), language="en", config=ex._word_repair_menu_config({}),
        dictionary_path=dictionary_module.get_dictionary_path("en"),
        model_path=automated_runner.zenith_native_model_path("en"),
        source_branch="main",
    )
    packet = hyp = None
    for p in menu.packets:
        hyps = (p.provenance or {}).get("word_hypotheses") or []
        if hyps:
            packet, hyp = p, hyps[0]
            break
    assert packet is not None, "menu proposed no word hypotheses"

    res = _test_word(ex, {"branch": "main", "char_start": hyp["start"],
                          "word": hyp["target"]}, turn=3)
    assert res["menu_backed"] is True
    # Parity MODULO hint fields: edits, acceptance verdict, adjudication number.
    assert res["edits"] == packet.provenance["edits"]
    assert res["acceptance"]["accepted"] == bool((packet.validation or {}).get("accepted"))
    assert (res["solver_scores"]["adjudication_score"]
            == (packet.solver_scores or {}).get("adjudication_score"))


def test_test_word_install_forks_and_applies():
    ex, alpha = _english_basin()
    menu = automated_runner.build_word_repair_menu(
        cipher_text=ex.workspace.cipher_text,
        base_key=dict(ex.workspace.get_branch("main").key),
        mask=(), language="en", config=ex._word_repair_menu_config({}),
        dictionary_path=dictionary_module.get_dictionary_path("en"),
        model_path=automated_runner.zenith_native_model_path("en"),
        source_branch="main",
    )
    hyp = next(h for p in menu.packets for h in ((p.provenance or {}).get("word_hypotheses") or []))
    res = _test_word(ex, {"branch": "main", "char_start": hyp["start"],
                          "word": hyp["target"], "install": True}, turn=4)
    assert res["installed_fork"] == "wordtest_4_main"
    # The damaged X-word now reads correctly on the installed fork.
    assert hyp["target"] in ex._decoded_preview(res["installed_fork"], max_words=40)


def test_test_word_install_rejection_surfaced(monkeypatch):
    """Whole-candidate rejection from apply_word_repair_edits is reported."""
    ex, alpha = _english_basin()
    menu = automated_runner.build_word_repair_menu(
        cipher_text=ex.workspace.cipher_text,
        base_key=dict(ex.workspace.get_branch("main").key),
        mask=(), language="en", config=ex._word_repair_menu_config({}),
        dictionary_path=dictionary_module.get_dictionary_path("en"),
        model_path=automated_runner.zenith_native_model_path("en"),
        source_branch="main",
    )
    hyp = next(h for p in menu.packets for h in ((p.provenance or {}).get("word_hypotheses") or []))
    import investigation.actions as actions_mod
    monkeypatch.setattr(
        actions_mod, "_word_hypothesis_result",
        actions_mod._word_hypothesis_result,
    )
    monkeypatch.setattr(
        automated_runner, "apply_word_repair_edits",
        lambda **_kw: (None, [], "no_applicable_edits"),
    )
    res = _test_word(ex, {"branch": "main", "char_start": hyp["start"],
                          "word": hyp["target"], "install": True}, turn=9)
    assert res["installed_fork"] is None
    assert "rejected" in (res["install_note"] or "")


def test_test_word_injected_non_dictionary():
    ex, alpha = _english_basin()
    # Probe word 0 ("THE" span, len 3) with a non-dictionary triple.
    res = _test_word(ex, {"branch": "main", "word_index": 0, "word": "ZQJ"})
    assert res["menu_backed"] is False
    assert res["in_dictionary"] is False
    assert res["verdict"] in {"accept", "hold_for_review"}
    assert res["edits"]           # non-empty edit set was derived


def test_test_word_injected_numeric_parity_with_library():
    """F8: the injected path reports EXACTLY the library's numbers for the same
    (span, target) — parity by construction, verified numerically."""
    from analysis.word_hypothesis_repair import score_injected_word_hypothesis
    ex, alpha = _english_basin()
    base_key = dict(ex.workspace.get_branch("main").key)
    # Word 0 span [0,3) is fully mapped, so decode coords == projection coords.
    res = _test_word(ex, {"branch": "main", "word_index": 0, "word": "ZQJ"})
    assert res["menu_backed"] is False
    pages, alpha2 = automated_runner._single_page_group(ex.workspace.cipher_text)
    lib = score_injected_word_hypothesis(
        pages=pages, shared_key=base_key,
        dictionary_path=dictionary_module.get_dictionary_path("en"),
        start=0, end=3, target="ZQJ", language="en",
        config=ex._word_repair_menu_config({}), mask=(), alphabet=alpha2,
        source_branch="main",
        model_path=automated_runner.zenith_native_model_path("en"),
    )
    assert res["edits"] == lib["edits"]
    assert res["in_dictionary"] == lib["in_dictionary"]
    assert (res["solver_scores"]["adjudication_score"]
            == (lib["packet"].solver_scores or {}).get("adjudication_score"))
    assert res["acceptance"]["accepted"] == bool((lib["packet"].validation or {}).get("accepted"))


def test_test_word_no_valid_edits_on_conflict():
    # Cipher 'aa' -> decode 'XX' (symbol a twice). Propose 'PQ': symbol a wants
    # both P and Q -> conflicting assignment -> empty edits -> no_valid_edits.
    ex = _synthetic_executor(raw="aa", sep=None, key={"a": "X"}, words=["PQ"])
    res = _test_word(ex, {"branch": "main", "word_index": 0, "word": "PQ"})
    assert res["verdict"] == "no_valid_edits"
    assert res["edits"] == []
    assert res["installed_fork"] is None


def test_test_word_unmapped_span_is_structured_error():
    ex, alpha = _english_basin()
    ex.workspace.clear_mapping("main", alpha.id_for("q"))  # unmap 'q' (in QUICK)
    res = _test_word(ex, {"branch": "main", "word_index": 1, "word": "QUICK"})
    assert res["status"] == "error"
    assert "unmapped or masked" in res["error"]
    assert res["offending_token_positions"]


def test_test_word_masked_span_is_structured_error():
    ex, alpha = _english_basin()
    # Mask 'r' (appears in SEVERAL etc.). A span containing it -> A2 span error.
    ex.workspace.get_branch("main").metadata["null_mask_selected"] = {"mask": ["r"]}
    # word index of BROWN (contains 'r').
    words = _ENGLISH_PLAIN.split()
    idx = words.index("BROWN")
    res = _test_word(ex, {"branch": "main", "word_index": idx, "word": "BROWN"})
    assert res["status"] == "error"
    assert "unmapped or masked" in res["error"]


def test_test_word_span_length_mismatch():
    ex = _synthetic_executor()
    res = _test_word(ex, {"branch": "main", "word_index": 0, "word": "TOOLONG"})
    assert res["status"] == "error"
    assert res["span_length"] == 3 and res["word_length"] == 7


# ---------------------------------------------------------------------------
# branch_adjudicate
# ---------------------------------------------------------------------------
def test_branch_adjudicate_table_and_ranking():
    ex = _synthetic_executor()
    ex.workspace.fork("weak", "main")
    # Break the weak branch so it scores worse.
    alpha = ex.workspace.cipher_text.alphabet
    ex.workspace.set_mapping("weak", alpha.id_for("a"), ex.workspace.plaintext_alphabet.id_for("Z"))
    res = execute_composite("branch_adjudicate",
                            {"branches": ["main", "weak"], "include_window": True},
                            executor=ex, state_readings={}, turn=1)
    assert res["status"] == "ok"
    assert res["baseline_branch"] == "main"
    assert [r["branch"] for r in res["rows"]] == ["main", "weak"]
    assert "weak" in res["deltas"]
    assert res["ranking"][0] == "main"      # main reads better than weak
    assert all("window_text" in r for r in res["rows"])


def test_branch_adjudicate_reading_and_range_errors():
    ex = _synthetic_executor()
    ex.workspace.fork("b2", "main")
    from investigation.reading import Reading, ReadingFragment
    reading = Reading(branch="main", source="lead", created_turn=2,
                      fragments=[ReadingFragment(text="CAT ON")],
                      overall_confidence=0.7)
    readings = {reading.reading_id: reading.to_dict()}
    res = execute_composite("branch_adjudicate", {"branches": ["main", "b2"]},
                            executor=ex, state_readings=readings, turn=3)
    main_row = next(r for r in res["rows"] if r["branch"] == "main")
    assert main_row["reading"]["reading_id"] == reading.reading_id
    assert main_row["reading"]["overall_confidence"] == 0.7
    # Out-of-range branch counts.
    one = execute_composite("branch_adjudicate", {"branches": ["main"]},
                            executor=ex, state_readings={}, turn=1)
    assert one["status"] == "error"
    unknown = execute_composite("branch_adjudicate", {"branches": ["main", "ghost"]},
                                executor=ex, state_readings={}, turn=1)
    assert unknown["status"] == "error" and "ghost" in unknown["unknown"]


# ---------------------------------------------------------------------------
# Dispatcher contract (A4)
# ---------------------------------------------------------------------------
def test_dispatcher_logs_toolcall_with_id_and_serialized_result():
    ex = _synthetic_executor()
    before = len(ex.call_log)
    res = execute_composite("branch_adjudicate", {"branches": ["main", "main"]},
                            executor=ex, state_readings={}, turn=7,
                            tool_use_id="tu_42")
    assert isinstance(res, dict)
    call = ex.call_log[-1]
    assert len(ex.call_log) == before + 1
    assert call.tool_name == "branch_adjudicate"
    assert call.tool_use_id == "tu_42"
    assert call.iteration == 7
    assert call.episode_id is None            # lead-side stamp
    # result is the SERIALIZED string.
    assert json.loads(call.result)["status"] == res["status"]


def test_dispatcher_filters_hints_when_episode_toolset_active(monkeypatch):
    import investigation.actions as actions_mod
    ex = _synthetic_executor()
    ex.episode_toolset = {"branch_adjudicate", "decode_show"}
    ex.episode_id = "ep_1"
    monkeypatch.setattr(
        actions_mod, "_branch_adjudicate",
        lambda executor, args, state_readings: {
            "status": "ok",
            "suggested_next_tools": ["decode_show", "search_anneal", "meta_declare_solution"],
        },
    )
    res = execute_composite("branch_adjudicate", {"branches": ["main", "main"]},
                            executor=ex, state_readings={}, turn=2, tool_use_id="t")
    # Off-toolset hints dropped; the ToolCall carries the episode id (A11).
    assert res["suggested_next_tools"] == ["decode_show"]
    assert ex.call_log[-1].episode_id == "ep_1"


def test_dispatcher_unknown_composite_is_structured():
    ex = _synthetic_executor()
    res = execute_composite("hypothesis_bogus", {}, executor=ex,
                            state_readings={}, turn=1)
    assert "Unknown composite tool" in res["error"]


def test_composite_defs_have_schemas():
    for d in COMPOSITE_TOOL_DEFINITIONS:
        assert "name" in d and "description" in d and "input_schema" in d
        assert d["input_schema"]["type"] == "object"
