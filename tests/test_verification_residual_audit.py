"""R3 grading-side diagnostics; no solving, provider calls, or historical claims."""
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_verification_residuals.py"
spec = importlib.util.spec_from_file_location("residual_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def brute_partners(a, b):
    paths = []
    def visit(i, j, score, pairs):
        if i == len(a) and j == len(b):
            paths.append((score, pairs))
            return
        if i < len(a) and j < len(b):
            visit(i+1, j+1, score+(2 if a[i] == b[j] else -1), pairs+[(i,j)])
        if i < len(a):
            visit(i+1, j, score-1, pairs+[(i,None)])
        if j < len(b):
            visit(i, j+1, score-1, pairs)
    visit(0, 0, 0, [])
    best = max(s for s, _ in paths)
    result = [set() for _ in a]
    for score, pairs in paths:
        if score == best:
            for i, j in pairs:
                result[i].add(j)
    return result


def test_optimal_alignment_partners_exhaustive_short_strings():
    words = ["".join(chars) for n in range(4) for chars in itertools.product("AB", repeat=n)]
    for a in words:
        for b in words:
            assert audit.optimal_partners(a, b) == brute_partners(a, b)


def test_mapping_pattern_and_occurrence_conflict_are_not_source_claims():
    r = audit.residuals("ZZZ", "AAA", ["s"]*3)
    assert r["character_class_counts"] == {"E1_consistent_mapping_candidate": 3}
    r = audit.residuals("ABA", "ABB", list("ABA"))
    assert r["character_class_counts"] == {"E3_occurrence_conflict_candidate": 1}
    assert r["character_residuals"][0]["source_support"] == "unreviewed"
    assert r["reviewed_fractions"] is None
    assert "NOT evidence" in " ".join(r["character_residuals"][0]["evidence"])


def test_alignment_tie_and_missing_token_binding_remain_unknown():
    r = audit.residuals("AAA", "AA", ["s"]*3)
    assert r["character_class_counts"] == {"ambiguous_unclassified": 1}
    assert "Multiple optimal" in " ".join(r["character_residuals"][0]["evidence"])
    r = audit.residuals("ZZZ", "AAA", None)
    assert r["character_class_counts"] == {"ambiguous_unclassified": 3}


def test_boundaries_and_editorial_notation_use_separate_denominators():
    r = audit.residuals("THECAT SLE EPS", "THE CAT SLEEPS")
    assert r["character_residual_count"] == 0
    assert r["word_residual_count"] > 0
    assert {row["class"] for row in r["word_residuals"]} == {"E2_boundary"}
    r = audit.residuals("WE SENT MEN", "WE SENT [SIX] MEN *nee*")
    assert r["reference_letter_exact"] is True
    assert len(r["notation_units"]) == 2
    assert r["current_scorer"]["char_accuracy"] < 1  # star marker not removed by existing scorer
    assert r["reviewed_character_count"] == 0


def test_packet_allowlist_and_traceable_labels():
    runtime, labels = audit.synthetic_packet()
    assert len(runtime) == len(labels) == 13
    for r in runtime:
        assert set(r) == audit.RUNTIME_FIELDS
        assert r["content_hash"] == audit._candidate_content_hash(r["candidate_text"])
        assert not any(key in r for key in ("reference", "labels", "kind", "known_key"))
    assert len({r["case_id"] for r in runtime}) == len(runtime)
    by_kind = {r["kind"]: r for r in labels}
    assert by_kind["fluent_wrong"]["labels"]["cipher_reconstruction_exact"] is False
    assert by_kind["fluent_wrong"]["labels"]["intelligible_reading"] is True
    assert by_kind["abbreviation"]["labels"]["intelligible_reading"] is None
    assert by_kind["segmentation_ambiguity"]["labels"]["intelligible_reading"] is None
    assert all(r["human_review"] is None for r in labels)
    by_id = {r["case_id"]: r for r in runtime}
    good = by_kind["fluent_right_same_text"]
    bad = by_kind["fluent_wrong"]
    assert by_id[good["case_id"]]["candidate_text"] == by_id[bad["case_id"]]["candidate_text"]
    assert good["labels"]["cipher_reconstruction_exact"] != bad["labels"]["cipher_reconstruction_exact"]
    assert (runtime, labels) == audit.synthetic_packet()


def test_review_queue_covers_every_e3_low_and_tenth_of_remainder():
    rows = [{"class": "E1_consistent_mapping_candidate", "confidence": "conditional"} for _ in range(21)]
    rows += [{"class": "E3_occurrence_conflict_candidate", "confidence": "conditional"},
             {"class": "ambiguous_unclassified", "confidence": "low"}]
    q = audit.review_queue([{"page": "sample", "sources": {}, "audit": {
        "character_residuals": rows, "word_residuals": [], "notation_units": []}}])
    assert {r["index"] for r in q} == {0, 10, 20, 21, 22}
    assert all(r["review"] is None for r in q)


def test_write_refuses_changed_evidence(tmp_path):
    target = tmp_path / "report.json"
    audit.write_new(target, "original")
    audit.write_new(target, "original")
    with pytest.raises(ValueError, match="Refusing"):
        audit.write_new(target, "changed")
    assert target.read_text() == "original"


def test_runtime_does_not_import_grading_audit():
    root = SCRIPT.parents[1] / "src"
    assert not any("audit_verification_residuals" in p.read_text() for p in root.rglob("*.py"))


def test_verdict_denominators_require_original_hash_and_recorded_acceptance(tmp_path, monkeypatch):
    artifact = {"run_id": "test", "cipher_id": "synth_en_250nb_s4", "ground_truth": "CAT",
                "branches": [{"name": "old", "decryption": "CAT"}, {"name": "current", "decryption": "DOG"}],
                "attestations": [{"content_hash": audit._candidate_content_hash("CAT"), "reader_accepts_as_solution": False},
                                 {"content_hash": audit._candidate_content_hash("DOG"), "reader_accepts_as_solution": True},
                                 {"content_hash": "missing", "reader_accepts_as_solution": True},
                                 {"content_hash": audit._candidate_content_hash("CAT"), "reader_accepts": True, "coherence": 10}]}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    report = audit.verifier_audit({"saved_artifacts": [{"artifact": "artifact.json", "artifact_sha256": audit.source_record(path)["sha256"]}]})
    counts = report["counts"]
    assert counts["original_content_recovered"] == 3
    assert counts["strict_reference_false_rejects"] == counts["strict_reference_positive_denominator"] == 1
    assert counts["strict_reference_false_accepts"] == counts["strict_reference_negative_denominator"] == 1
    assert report["verdicts"][2]["reference_reconstruction_exact"] is None
    assert report["verdicts"][3]["accepted_under_recorded_contract"] is None
    assert counts["policy_labeled_denominator"] == 0
    assert json.loads(path.read_text()) == artifact
