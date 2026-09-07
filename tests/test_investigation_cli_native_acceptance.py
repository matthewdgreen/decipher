"""Opt-in I-6 end-to-end: real round-6 search, local fake verifier, CLI only.

Run with DECIPHER_RUN_NATIVE_CLI_ACCEPTANCE=1. This checks transport/gate
composition; the fake reader is NOT evidence of independent readability.
"""
import hashlib
import os

import pytest

import investigation_cli as cli
from tests.test_investigation_cli import _PositiveVerifyProvider, _run


# Ciphertext only from the historical round-6 investigation a0019702343e.
# No known key, plaintext, or solution-bearing metadata enters the workflow.
CIPHERTEXT = (
    "CYOUPUNPNMCSPUGAQOJCPICASTPJMXNHXMWYDXHVEESZOKETXOVSSJOYJVVIDCDJXKVIPGDYCORZVXNUIPRQVSBGIZNQDTJFFBKZUQXCJPRIKSCZFBOQMMWCFKSEMJNUJQOZJJNPZRJIIMBICYOUXUOUJPOXUXOAEORZSENZPERZIXSGEPZAWFWCFXTQVNWAXOJACEUJARLVPYEGPORFHWBYOQTZKIZGPJNUCXICJFZZOLBMCYOEKRBNPSOPPUQXFFCZUGDYCYJYNUKNXRNJKKBGZORQMWBYXTWSSJYDMVJWPUONBZNSSJVRXOVDIEOJMQOIMMZICYOCXLDUITGJPNWNFVPQVIWCBOLVPFBGMMWJIJNAEYSDQUKHWBNZCGDAVMLZCXJWBVUEXLDUXMEFHSCWMRTTASVABFYZOMEHVYEVPGNRXOOYKSKGENVFZRKMFORQVGDAORHXSENUXMWYSJNAQVSZOGDYCCJGPUNANRURPUOSIMLSSJOABCTZQJNNCTPFCLBUCYOESJHJDWOYMMBGFQQZCNJDCFKIKWOWBFONVGEGXCTAVL"
)


@pytest.mark.skipif(os.environ.get("DECIPHER_RUN_NATIVE_CLI_ACCEPTANCE") != "1",
                    reason="opt-in native CLI acceptance")
def test_round6_cli_lifecycle(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DECIPHER_PARALLEL_WORKERS", "4")
    monkeypatch.setenv("DECIPHER_QUAGMIRE_THREADS", "4")
    provider = _PositiveVerifyProvider()
    monkeypatch.setattr(cli, "_make_cli_verify_provider", lambda *_: provider)

    code, started, _ = _run(tmp_path, ["start", "--ciphertext", CIPHERTEXT,
                                     "--format", "letters", "--language", "en"], capsys)
    assert code == 0
    iid = started["investigation_id"]

    def revision():
        code, status, _ = _run(tmp_path, ["status", iid], capsys)
        assert code == 0
        return str(status["revision"])

    code, _, _ = _run(tmp_path, ["diagnose", iid], capsys)
    assert code == 0
    code, submitted, _ = _run(tmp_path, [
        "experiment-submit", iid, "--revision", revision(),
        "--type", "quagmire3_shotgun", "--branch", "main", "--config-json", "{}", "--wait",
    ], capsys)
    assert code == 0, submitted
    code, collected, _ = _run(tmp_path, [
        "experiment-collect", iid, "--revision", revision(),
        "--experiment-id", submitted["experiment_id"], "--install", "--candidate-rank", "1",
    ], capsys)
    assert code == 0 and collected.get("installed_as"), collected
    branch = collected["installed_as"]
    code, decoded, _ = _run(tmp_path, ["decode", iid, "--branch", branch], capsys)
    assert code == 0
    text = "".join(row["decoded"] for row in decoded["rows"])
    assert len(text) == len(CIPHERTEXT) and "?" not in text

    declaration = ["--branch", branch, "--rationale", "local gate-composition test",
                   "--self-confidence", "0.9", "--reading-summary", "test-only fake reading",
                   "--no-further-iterations-helpful", "--further-iterations-note", "test complete"]
    code, keyless, _ = _run(tmp_path, ["verify", iid, "--revision", revision(), "--branch", branch], capsys)
    assert code == 1 and keyless["reason"] == "no_verification_provider"
    code, refused, _ = _run(tmp_path, ["declare-solution", iid, "--revision", revision(), *declaration], capsys)
    assert code == 3 and refused["reason"] == "attestation_required"
    assert provider.calls == 0

    code, verified, _ = _run(tmp_path, [
        "--verify-provider", "openai", "--allow-external",
        "verify", iid, "--revision", revision(), "--branch", branch,
    ], capsys)
    assert code == 0 and verified.get("attestation"), verified
    assert provider.calls == 1
    code, declared, _ = _run(tmp_path, ["declare-solution", iid, "--revision", revision(), *declaration], capsys)
    assert code == 0 and declared["terminal_status"] == "solved", declared
    # Every call loads a fresh service from disk; the final reload must agree.
    code, after, _ = _run(tmp_path, ["decode", iid, "--branch", branch], capsys)
    assert code == 0 and after == decoded
    # Post-hoc assertion ONLY, after selection and declaration. The expected
    # hash is from the sealed round-6 answer, never passed to any runtime call.
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "782fc5bc7707001ca0b2b4761c284690b35ed90d116e59607e5d66ecfc6144d4"
    )
