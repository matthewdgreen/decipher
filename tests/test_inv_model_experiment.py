"""Tests for the INV model-diagnosis experiment harness.

Covers (spec Part 4):
  * suite manifest schema + firewall (assert_no_ground_truth_leak),
  * served-model gate-detection (fake response, both directions) + additive
    RunArtifact fields,
  * runner with a FAKE provider end-to-end -> artifacts -> scorer -> report,
    with the gate-fired run excluded,
  * scoring unit tests (correct / wrong / never-diagnosed + composite credit +
    automated baseline over a warm cache).

No network. Uses a fake plaintext generator (warm temp cache) and a fake model
provider with scripted diagnosis responses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent.model_provider import (  # noqa: E402
    ModelResponse,
    ModelUsage,
    ToolUseBlock,
    served_model_from_response,
    served_model_matches,
)
from artifact.schema import RunArtifact  # noqa: E402
from experiments.inv_diagnosis import (  # noqa: E402
    MANIFEST_REQUIRED_KEYS,
    SUITE_ROWS,
    DiagnosisFamily,
    GroundTruthLeak,
    assert_no_ground_truth_leak,
    build_case,
    build_spec,
    coerce_family,
    family_from_truth_label,
    is_diagnosis_correct,
    is_pareidolia,
    manifest_row,
    map_argmax_suspicion,
)
from testgen.builder import build_test_case  # noqa: E402
from testgen.cache import PlaintextCache  # noqa: E402

import run_inv_model_experiment as runner  # noqa: E402
import report_inv_model_experiment as report  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

_FAKE_PLAINTEXTS = {
    "medieval_scriptoria": (
        "IN THE QUIET HALLS OF MEDIEVAL MONASTERIES SKILLED SCRIBES LABORED FOR "
        "LONG HOURS COPYING SACRED TEXTS AND WORLDLY RECORDS ONTO PREPARED SHEETS "
        "OF PARCHMENT USING CAREFULLY MIXED INKS AND SHARPENED QUILLS THEY "
        "PRESERVED KNOWLEDGE THAT MIGHT OTHERWISE HAVE VANISHED FROM HUMAN MEMORY "
        "ACROSS THE LONG TURNING CENTURIES OF QUIET STUDY AND DEVOTION EACH DAY"
    ),
    "alpine_botany": (
        "HIGH UPON THE COLD MOUNTAIN SLOPES A REMARKABLE VARIETY OF HARDY PLANTS "
        "ENDURES THE FIERCE WINDS AND THIN SOILS THAT WOULD DEFEAT LESS RESILIENT "
        "SPECIES BOTANISTS WHO CLIMB THESE RUGGED HEIGHTS RECORD EACH DELICATE "
        "FLOWER AND CUSHION OF MOSS NOTING HOW THEY SURVIVE THE BRIEF GROWING "
        "SEASON BETWEEN THE MELTING AND RETURN OF THE DEEP WINTER SNOW EACH YEAR"
    ),
    "harbor_trade_ledgers": (
        "THE BUSY HARBOR CLERKS KEPT DETAILED LEDGERS OF EVERY CARGO THAT PASSED "
        "ACROSS THE CROWDED WOODEN WHARVES RECORDING BARRELS OF WINE BALES OF "
        "WOOL AND CHESTS OF SPICE ALONGSIDE THE NAMES OF CAPTAINS AND THE DISTANT "
        "PORTS FROM WHICH THEIR LADEN VESSELS HAD SAILED THROUGH STORM AND CALM "
        "TO REACH THIS THRIVING CENTER OF TRADE AND HONEST COMMERCE EACH SEASON"
    ),
    "clockmaking_guilds": (
        "WITHIN THE NARROW WORKSHOPS OF THE OLD CLOCKMAKING GUILDS PATIENT MASTERS "
        "TAUGHT THEIR APPRENTICES TO CUT TINY GEARS AND POLISH SLENDER SPRINGS "
        "UNTIL EACH MECHANISM KEPT FAITHFUL TIME THEIR REPUTATION RESTED ON "
        "PRECISION AND EVERY FINISHED PIECE CARRIED THE QUIET PRIDE OF PATIENT "
        "HANDS THAT HAD SPENT LONG YEARS PERFECTING A DIFFICULT DELICATE CRAFT"
    ),
    "river_navigation": (
        "EXPERIENCED PILOTS WHO GUIDED HEAVY BARGES ALONG THE WINDING RIVER "
        "LEARNED TO READ THE SUBTLE SIGNS OF SHIFTING SANDBARS AND HIDDEN CURRENTS "
        "THAT COULD GROUND AN UNWARY CREW THEY MEMORIZED EACH BEND AND LANDMARK "
        "PASSING SAFELY BENEATH LOW BRIDGES AND THROUGH NARROW CHANNELS WHERE THE "
        "WATER RAN SWIFT AND THE MARGIN FOR HONEST ERROR WAS ALWAYS VERY SLIM"
    ),
    "observatory_records": (
        "NIGHT AFTER NIGHT THE DEDICATED ASTRONOMERS OF THE HILLTOP OBSERVATORY "
        "RECORDED THE SLOW WHEELING OF THE STARS AND THE WANDERING PATHS OF THE "
        "PLANETS ACROSS THEIR CHARTS THEIR PATIENT MEASUREMENTS GATHERED OVER "
        "MANY SEASONS ALLOWED LATER SCHOLARS TO PREDICT ECLIPSES AND TO REFINE "
        "THE CALENDAR THAT ORDERED THE AFFAIRS OF THE WHOLE QUIET COUNTRY SIDE"
    ),
}


class _FakeGen:
    """Fake plaintext generator: returns a fixed plaintext per topic."""

    def generate(self, spec):
        return _FAKE_PLAINTEXTS[spec.topic]


class _FakeProvider:
    """Scripted provider: observe_cipher_id on call 1, declare on call 2."""

    def __init__(self, *, model, provider_name, served_model, family, confidence):
        self.model = model
        self.provider_name = provider_name
        self._served = served_model
        self._family = family
        self._conf = confidence
        self.calls = 0

    def send(self, *, messages, tools, system, max_tokens):
        self.calls += 1
        if self.calls == 1:
            content = [ToolUseBlock(id="c1", name="observe_cipher_id", input={})]
        else:
            content = [
                ToolUseBlock(
                    id="c2",
                    name="meta_declare_diagnosis",
                    input={
                        "family": self._family,
                        "recommended_attack": "run the appropriate solver",
                        "confidence": self._conf,
                    },
                )
            ]
        raw = SimpleNamespace(model=self._served) if self._served else SimpleNamespace()
        return ModelResponse(
            content=content,
            usage=ModelUsage(input_tokens=120, output_tokens=30),
            raw=raw,
        )


class _TruncatingProvider:
    """Turn 1: max_tokens truncation (no tool_use). Turn 2+: declare.

    Models a verbose/thinking-heavy model that gets cut off before it can call a
    tool, then declares once nudged (F-1).
    """

    def __init__(self, *, model, provider_name, family):
        self.model = model
        self.provider_name = provider_name
        self._family = family
        self.calls = 0

    def send(self, *, messages, tools, system, max_tokens):
        self.calls += 1
        if self.calls == 1:
            content = []  # truncated before any complete tool_use
            stop_reason = "max_tokens"
        else:
            content = [
                ToolUseBlock(
                    id="d1",
                    name="meta_declare_diagnosis",
                    input={
                        "family": self._family,
                        "recommended_attack": "run the solver",
                        "confidence": 0.7,
                    },
                )
            ]
            stop_reason = "tool_use"
        raw = SimpleNamespace(model=self.model, stop_reason=stop_reason)
        return ModelResponse(
            content=content,
            usage=ModelUsage(input_tokens=200, output_tokens=8192),
            raw=raw,
        )


@pytest.fixture()
def warm_cache(tmp_path):
    """A PlaintextCache pre-warmed with fake plaintexts for every suite row."""
    cache = PlaintextCache(tmp_path / "cache")
    gen = _FakeGen()
    for row in SUITE_ROWS:
        build_case(row, cache, generator=gen)  # populates cache
    return cache


def _build_all_cases(cache):
    return {row.case_id: build_case(row, cache, offline=True) for row in SUITE_ROWS}


# ---------------------------------------------------------------------------
# Suite manifest schema + firewall
# ---------------------------------------------------------------------------


def test_suite_manifest_schema(warm_cache):
    rows = []
    for row in SUITE_ROWS:
        td = build_case(row, warm_cache, offline=True)
        rec = manifest_row(row, td)
        for key in MANIFEST_REQUIRED_KEYS:
            assert key in rec, f"{row.case_id} missing {key}"
        # true_family is a valid canonical enum value
        assert rec["true_family"] in {f.value for f in DiagnosisFamily}
        # cipher_system maps to the declared true family
        assert family_from_truth_label(rec["cipher_system"]).value == rec["true_family"]
        assert rec["case_id"] == row.case_id  # neutral id
        rows.append(rec)
    # distinct plaintext hashes (distinct topics => distinct plaintexts, F5)
    hashes = [r["plaintext_sha256"] for r in rows]
    assert len(set(hashes)) == len(hashes)
    # every declared family in the spec's family list is represented
    families = {r["true_family"] for r in rows}
    assert DiagnosisFamily.TRANSPOSITION_SUBSTITUTION.value in families  # composite
    assert DiagnosisFamily.COLUMNAR_TRANSPOSITION.value in families


def test_neutral_case_id_has_no_family_flag():
    # Neutral case_ids must not encode the family (no honb/vig/tnb/... flags).
    for row in SUITE_ROWS:
        low = row.case_id.lower()
        for flag in ("honb", "thonb", "ptnb", "tnb", "q3nb", "vignb", "vigwb"):
            assert flag not in low


def test_firewall_over_rendered_prompt(warm_cache):
    from benchmark.loader import parse_canonical_transcription

    for row in SUITE_ROWS:
        td = build_case(row, warm_cache, offline=True)
        # The ORIGINAL builder test_id (family-encoding) — rebuild without override.
        builder_td = build_test_case(build_spec(row), warm_cache, api_key="")
        builder_test_id = builder_td.test.test_id
        assert builder_test_id != row.case_id  # sanity: it does encode family

        ct = parse_canonical_transcription(td.canonical_transcription)
        prompt = runner.render_diagnosis_prompt(ct, td.canonical_transcription)
        # Must not leak true_family, plaintext, builder test_id, or builder desc.
        assert_no_ground_truth_leak(
            prompt,
            true_family=row.true_family,
            plaintext=td.plaintext,
            builder_test_id=builder_test_id,
            builder_description=td.test.description,
        )


def test_firewall_catches_leaks():
    # true family present
    with pytest.raises(GroundTruthLeak):
        assert_no_ground_truth_leak(
            "this looks like a vigenere cipher to me",
            true_family="vigenere",
            plaintext="ZZZZ",
        )
    # plaintext present
    with pytest.raises(GroundTruthLeak):
        assert_no_ground_truth_leak(
            "the answer is ATTACKATDAWNORDIE and more",
            true_family="unknown_x",
            plaintext="ATTACKATDAWNORDIE",
        )
    # builder flag token present
    with pytest.raises(GroundTruthLeak):
        assert_no_ground_truth_leak(
            "case synth_en_145ptnb_s105 details",
            true_family="none_x",
            plaintext="ZZZZ",
            builder_test_id="synth_en_145ptnb_s105",
        )


# ---------------------------------------------------------------------------
# Served-model gate detection (F7) + additive RunArtifact fields
# ---------------------------------------------------------------------------


def test_served_model_from_fake_response_differs_sets_flag():
    # Fake Anthropic response whose .model differs from the request -> gate.
    resp = ModelResponse(content=[], raw=SimpleNamespace(model="claude-opus-4-8"))
    served = served_model_from_response(resp)
    assert served == "claude-opus-4-8"
    assert served_model_matches("claude-fable-5", served) is False


def test_served_model_matching_does_not_fire():
    # A matching (prefix) served model does not fire the gate.
    resp = ModelResponse(content=[], raw=SimpleNamespace(model="claude-fable-5-20260101"))
    served = served_model_from_response(resp)
    assert served_model_matches("claude-fable-5", served) is True
    # OpenAI-style date suffix also matches.
    assert served_model_matches("gpt-5.5", "gpt-5.5-2026-01-01") is True


def test_served_model_unknown_is_not_a_gate():
    resp = ModelResponse(content=[], raw=SimpleNamespace())  # no .model
    assert served_model_from_response(resp) is None
    assert served_model_matches("gpt-5.5", None) is True


def test_run_artifact_served_model_fields_are_additive():
    art = RunArtifact(run_id="r", cipher_id="c", model="claude-fable-5", language="en")
    assert art.served_models == []
    assert art.safety_gate_fired is False
    # to_dict round-trips the new fields
    d = art.to_dict()
    assert d["served_models"] == []
    assert d["safety_gate_fired"] is False


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------


def test_is_diagnosis_correct_basic():
    assert is_diagnosis_correct(DiagnosisFamily.VIGENERE, "vigenere")
    assert not is_diagnosis_correct(DiagnosisFamily.VIGENERE, "monoalphabetic_substitution")
    assert not is_diagnosis_correct(DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION, None)


def test_composite_credit_rule():
    comp = DiagnosisFamily.TRANSPOSITION_SUBSTITUTION
    assert is_diagnosis_correct(comp, "transposition_substitution")
    assert is_diagnosis_correct(comp, "unknown")  # honest non-commit credited
    assert not is_diagnosis_correct(comp, "monoalphabetic_substitution")


def test_pareidolia_flag():
    col = DiagnosisFamily.COLUMNAR_TRANSPOSITION
    assert is_pareidolia(col, "monoalphabetic_substitution", 0.9)  # confident wrong
    assert not is_pareidolia(col, "monoalphabetic_substitution", 0.3)  # unsure wrong
    assert not is_pareidolia(col, "unknown", 0.95)  # honest non-commit
    assert not is_pareidolia(col, "columnar_transposition", 0.95)  # correct
    assert not is_pareidolia(col, None, None)  # never declared


def test_map_argmax_suspicion():
    assert map_argmax_suspicion(
        {"monoalphabetic_substitution": 1.0, "transposition": 0.25}
    ) == DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION
    assert map_argmax_suspicion(
        {"polyalphabetic_vigenere": 0.85, "monoalphabetic_substitution": 0.5}
    ) == DiagnosisFamily.VIGENERE
    assert map_argmax_suspicion({"playfair": 0.5}) == DiagnosisFamily.UNKNOWN
    assert map_argmax_suspicion({}) == DiagnosisFamily.UNKNOWN


def test_coerce_family_aliases():
    assert coerce_family("Vigenere") == DiagnosisFamily.VIGENERE
    assert coerce_family("polyalphabetic_vigenere") == DiagnosisFamily.VIGENERE
    assert coerce_family("composite") == DiagnosisFamily.TRANSPOSITION_SUBSTITUTION
    assert coerce_family("garbage-nonsense") is None


def test_score_artifact_correct_wrong_never():
    correct = report.score_artifact(
        {
            "case_id": "x",
            "model": "gpt-5.5",
            "diagnosis": {"family": "vigenere", "confidence": 0.8},
            "declared_iteration": 3,
            "tool_call_count": 4,
            "safety_gate_fired": False,
        },
        "vigenere",
    )
    assert correct["diagnosis_correct"] and not correct["pareidolia_flag"]
    assert correct["turns_to_first_correct"] == 3

    wrong = report.score_artifact(
        {
            "case_id": "x",
            "model": "gpt-5.5",
            "diagnosis": {"family": "monoalphabetic_substitution", "confidence": 0.9},
            "declared_iteration": 2,
            "tool_call_count": 3,
            "safety_gate_fired": False,
        },
        "columnar_transposition",
    )
    assert not wrong["diagnosis_correct"] and wrong["pareidolia_flag"]
    assert wrong["turns_to_first_correct"] == float("inf")

    never = report.score_artifact(
        {
            "case_id": "x",
            "model": "gpt-5.5",
            "diagnosis": None,
            "declared_iteration": None,
            "tool_call_count": 6,
            "safety_gate_fired": False,
        },
        "homophonic_substitution",
    )
    assert not never["diagnosis_correct"] and not never["pareidolia_flag"]
    assert never["turns_to_first_correct"] == float("inf")


# ---------------------------------------------------------------------------
# Free automated (non-LLM) baseline
# ---------------------------------------------------------------------------


def test_automated_baseline_over_warm_cache(warm_cache):
    true_families = {r.case_id: r.true_family for r in SUITE_ROWS}
    baseline = report.automated_baseline(true_families, warm_cache.cache_dir)
    assert baseline["n"] == len(SUITE_ROWS)
    assert not baseline["skipped"]
    per_case = {c["case_id"]: c for c in baseline["per_case"]}
    # The fingerprint reliably names mono, homophonic, vigenere...
    assert per_case["inv_diag_case_01"]["diagnosis_correct"]  # mono wb
    assert per_case["inv_diag_case_02"]["diagnosis_correct"]  # mono nb
    assert per_case["inv_diag_case_03"]["diagnosis_correct"]  # homophonic
    assert per_case["inv_diag_case_04"]["diagnosis_correct"]  # vigenere
    # ...but misses columnar transposition and the composite (expected floor).
    assert not per_case["inv_diag_case_05"]["diagnosis_correct"]  # columnar
    assert not per_case["inv_diag_case_06"]["diagnosis_correct"]  # composite


# ---------------------------------------------------------------------------
# Runner end-to-end with a fake provider -> scorer -> report
# ---------------------------------------------------------------------------


def test_runner_end_to_end_fake_provider(warm_cache, tmp_path):
    cases = _build_all_cases(warm_cache)
    artifact_root = tmp_path / "artifacts"

    # Model A (OpenAI): declares the TRUE family for every case -> all correct,
    # served model matches -> included.
    model_a_dir = artifact_root / "gpt-5.5"
    model_a_dir.mkdir(parents=True)
    for case_id, td in cases.items():
        true_family = next(r.true_family for r in SUITE_ROWS if r.case_id == case_id)
        prov = _FakeProvider(
            model="gpt-5.5",
            provider_name="openai",
            served_model="gpt-5.5-2026",
            family=true_family,
            confidence=0.85,
        )
        art = runner.run_diagnosis_task(
            test_data=td,
            model_provider=prov,
            requested_model="gpt-5.5",
            provider_name="openai",
            max_iterations=6,
        )
        assert art["automated_preflight"] is None  # F4: preflight OFF
        assert art["final_char_accuracy"] is None   # diagnosis-only
        (model_a_dir / f"{case_id}.json").write_text(json.dumps(art), encoding="utf-8")

    # Model B (Anthropic): SERVED by opus for every case -> gate fires -> excluded.
    model_b_dir = artifact_root / "claude-fable-5"
    model_b_dir.mkdir(parents=True)
    for case_id, td in cases.items():
        prov = _FakeProvider(
            model="claude-fable-5",
            provider_name="anthropic",
            served_model="claude-opus-4-8",
            family="monoalphabetic_substitution",
            confidence=0.9,
        )
        art = runner.run_diagnosis_task(
            test_data=td,
            model_provider=prov,
            requested_model="claude-fable-5",
            provider_name="anthropic",
            max_iterations=6,
        )
        assert art["safety_gate_fired"] is True
        (model_b_dir / f"{case_id}.json").write_text(json.dumps(art), encoding="utf-8")

    # Scorer reads the artifacts; report aggregates.
    artifacts = report.load_artifacts(artifact_root)
    assert len(artifacts) == 2 * len(SUITE_ROWS)
    true_families = {r.case_id: r.true_family for r in SUITE_ROWS}
    agg = report.aggregate(artifacts, true_families)

    # OpenAI arm: included, perfect accuracy, no pareidolia.
    assert "gpt-5.5" in agg["per_model"]
    a = agg["per_model"]["gpt-5.5"]
    assert a["n"] == len(SUITE_ROWS)
    assert a["diagnosis_accuracy"] == 1.0
    assert a["pareidolia_rate"] == 0.0
    assert a["total_cost_usd"] > 0.0  # openai is priced

    # Anthropic arm: entirely in the excluded gate-fired bucket.
    assert "claude-fable-5" not in agg["per_model"]
    assert "claude-fable-5" in agg["excluded"]
    assert agg["excluded"]["claude-fable-5"]["n"] == len(SUITE_ROWS)
    assert "claude-opus-4-8" in agg["excluded"]["claude-fable-5"]["served_models"]

    # Report renders with baseline + excluded section.
    baseline = report.automated_baseline(true_families, warm_cache.cache_dir)
    text = report.format_report(agg, baseline)
    assert "FREE automated (non-LLM) diagnosis baseline" in text
    assert "EXCLUDED" in text
    assert "gpt-5.5" in text


# ---------------------------------------------------------------------------
# F-1: max_tokens truncation nudge
# ---------------------------------------------------------------------------


def test_truncation_nudge_converts_cutoff_into_declaration(warm_cache):
    # A max_tokens truncation with no tool_use must NOT be scored as a miss: the
    # runner nudges once and the model declares on the next turn (F-1).
    td = build_case(SUITE_ROWS[0], warm_cache, offline=True)
    prov = _TruncatingProvider(
        model="claude-fable-5",
        provider_name="anthropic",
        family="monoalphabetic_substitution",
    )
    art = runner.run_diagnosis_task(
        test_data=td,
        model_provider=prov,
        requested_model="claude-fable-5",
        provider_name="anthropic",
        max_iterations=6,
        max_tokens=8192,
    )
    assert art["status"] == "declared"
    assert art["truncation_nudged"] is True
    assert art["diagnosis"]["family"] == "monoalphabetic_substitution"
    assert "max_tokens" in art["stop_reasons"]
    assert prov.calls == 2  # exactly one nudge -> one extra turn


def test_max_tokens_flag_threads_into_send(warm_cache):
    # The --max-tokens value reaches the provider .send() call.
    td = build_case(SUITE_ROWS[0], warm_cache, offline=True)
    seen = {}

    class _CaptureProvider:
        model = "gpt-5.5"
        provider_name = "openai"

        def send(self, *, messages, tools, system, max_tokens):
            seen["max_tokens"] = max_tokens
            return ModelResponse(
                content=[
                    ToolUseBlock(
                        id="d",
                        name="meta_declare_diagnosis",
                        input={
                            "family": "monoalphabetic_substitution",
                            "recommended_attack": "x",
                            "confidence": 0.5,
                        },
                    )
                ],
                usage=ModelUsage(input_tokens=10, output_tokens=5),
                raw=SimpleNamespace(model="gpt-5.5", stop_reason="tool_use"),
            )

    runner.run_diagnosis_task(
        test_data=td,
        model_provider=_CaptureProvider(),
        requested_model="gpt-5.5",
        provider_name="openai",
        max_iterations=3,
        max_tokens=12345,
    )
    assert seen["max_tokens"] == 12345


# ---------------------------------------------------------------------------
# F-2: observe-only lock is enforced by the executor
# ---------------------------------------------------------------------------


def test_diagnosis_executor_rejects_non_observe_tool(warm_cache):
    # Pins the observe-only lock: a solver/search tool routed through the
    # diagnosis executor is rejected. If the set_allowed_tool_names line is
    # removed, this test fails (search_* would otherwise execute).
    from benchmark.loader import parse_canonical_transcription

    td = build_case(SUITE_ROWS[0], warm_cache, offline=True)
    ct = parse_canonical_transcription(td.canonical_transcription)
    executor = runner._build_executor(ct, "en")
    result = executor.execute("search_hill_climb", {"branch": "main"}, "tuid")
    assert "search_hill_climb" in result
    # The gated-tool rejection identifies why and never runs the solver.
    assert ("tool_gated" in result) or ("not available" in result)
    # A sanity check: an observe tool IS allowed through the same executor.
    ok = executor.execute("observe_ic", {}, "tuid2")
    assert "tool_gated" not in ok
