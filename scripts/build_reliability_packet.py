#!/usr/bin/env python3
"""R0 grading-side packet builder and saved-artifact audit; no solver/LLM calls.

Runtime cases are an explicit allowlist. Source text, family labels, keys,
expectations, and development/holdout membership live in the separate grading
store. This script must never be imported by a solver or agent runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from analysis.model_registry import resolve_language_model
from analysis.polyalphabetic import encode_quagmire_plaintext
from benchmark.loader import BenchmarkLoader
from ciphers.fractionation import BifidCipher
from ciphers.polyalphabetic import VigenereCipher
from ciphers.transposition import columnar_decrypt, columnar_encrypt
from investigation.state import InvestigationState
from mcp_server.intake import build_cipher_text
from report_v3_candidate_selection import analyze_artifact_data
from testgen.builder import _make_homophonic_key
from testgen.corpus_library import load_library

PACKET_VERSION = "reliability-r0-v1"
FAMILIES = (
    "simple_substitution", "homophonic_substitution", "vigenere",
    "quagmire3", "columnar_transposition", "substitution_transposition",
)
HISTORICAL = (
    ("borg_tests.jsonl", "borg_single_B_borg_0109v", "simple_substitution"),
    ("borg_tests.jsonl", "borg_single_B_borg_0045v", "simple_substitution"),
    ("copiale_tests.jsonl", "copiale_single_B_copiale_p017", "homophonic_substitution"),
    ("copiale_tests.jsonl", "copiale_single_B_copiale_p068", "homophonic_substitution"),
)
AUDIT_RUNS = (
    "8d5bce9769b1", "ac129831aebc", "29b8f89ad6ee", "d65f5f4876a7",
    "9f547bfcf55a", "640e959623f4", "407e29ec7c70", "1d6d78083226",
)
RUNTIME_FIELDS = {"case_id", "ciphertext", "format", "language", "ciphertext_sha256"}


def digest(value: Any) -> str:
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def opaque_id(index: int) -> str:
    return "r" + digest(f"{PACKET_VERSION}:case:{index}")[:12]


def letters(text: str) -> str:
    return re.sub("[^A-Z]", "", text.upper())


def runtime_case(case_id: str, ciphertext: str, fmt: str, language: str) -> dict:
    case = dict(case_id=case_id, ciphertext=ciphertext, format=fmt, language=language,
                ciphertext_sha256=digest(ciphertext))
    parsed = build_cipher_text(ciphertext, fmt)
    if not parsed.tokens:
        raise ValueError("empty parsed ciphertext")
    return case


def encrypt_case(family: str, text: str, rng: random.Random, *, variant: int) -> tuple[str, str, dict]:
    """Generate once; verify the mechanism with known keys, never a search."""
    plain = letters(text)
    alpha = string.ascii_uppercase
    key: dict[str, Any]
    fmt = "letters"
    if family in {"simple_substitution", "substitution_transposition"}:
        perm = "".join(rng.sample(alpha, 26))
        substitution = dict(zip(alpha, perm))
        inverse = str.maketrans(perm, alpha)
        substituted = plain.translate(str.maketrans(substitution))
        key = {"substitution": substitution}
        if family == "simple_substitution":
            ciphertext = " ".join(word.translate(str.maketrans(substitution)) for word in text.split())
            decoded = letters(ciphertext.translate(inverse))
        else:
            keyword = "".join(rng.sample(alpha, 6 + variant))
            ciphertext = columnar_encrypt(substituted, keyword)
            decoded = columnar_decrypt(ciphertext, keyword).translate(inverse)
            key["columnar_keyword"] = keyword
    elif family == "homophonic_substitution":
        homophones = _make_homophonic_key(rng)
        mapping = {"S" + symbol.zfill(3): ch for ch, symbols in homophones.items() for symbol in symbols}
        tokens = ["S" + rng.choice(homophones[ch]).zfill(3) for ch in plain]
        ciphertext = " ".join(tokens)
        decoded = "".join(mapping[token] for token in tokens)
        key, fmt = {"decode_map": mapping}, "canonical"
    elif family == "vigenere":
        keyword = "".join(rng.choice(alpha) for _ in range(5 + variant))
        ciphertext = VigenereCipher().encrypt(plain, keyword)
        # Independent arithmetic inverse, not the encryptor's inverse implementation.
        decoded = "".join(alpha[(ord(ch) - ord(keyword[i % len(keyword)])) % 26]
                          for i, ch in enumerate(ciphertext))
        key = {"keyword": keyword}
    elif family == "quagmire3":
        keyword = "".join(rng.sample(alpha, 7))
        cycleword = "".join(rng.choice(alpha) for _ in range(8))
        ciphertext = encode_quagmire_plaintext(plain, cycleword=cycleword, alphabet_keyword=keyword)
        tableau = "".join(dict.fromkeys(keyword + alpha))
        decoded = "".join(tableau[(tableau.index(ch) - tableau.index(cycleword[i % 8])) % 26]
                          for i, ch in enumerate(ciphertext))
        key = {"alphabet_keyword": keyword, "cycleword": cycleword}
    elif family == "columnar_transposition":
        keyword = "".join(rng.sample(alpha, 7 if variant == 0 else 11))
        ciphertext = columnar_encrypt(plain, keyword)
        decoded = columnar_decrypt(ciphertext, keyword)
        key = {"columnar_keyword": keyword}
    elif family == "bifid":
        square = "".join(rng.sample(alpha.replace("J", ""), 25))
        plain = plain.replace("J", "I")
        cipher = BifidCipher(period=7)
        ciphertext = cipher.encrypt(plain, square)
        decoded = cipher.decrypt(ciphertext, square)
        key = {"square": square, "period": 7}
    else:
        raise ValueError(f"unsupported generation family: {family}")
    if decoded != plain:
        raise ValueError(f"{family}: known-key inversion failed")
    if len(build_cipher_text(ciphertext, fmt).tokens) != len(plain):
        raise ValueError(f"{family}: intake changed token count")
    return ciphertext, fmt, {"key": key, "plaintext": plain,
                             "plaintext_spaced": text.replace("J", "I") if family == "bifid" else text,
                             "roundtrip_verified": True}


def build_packet(benchmark_root: Path, *, library_dir: Path | None = None) -> tuple[list, dict]:
    library = sorted(load_library("en", library_dir=library_dir), key=lambda row: row.id)
    # Source-document separation, not just distinct offsets from the same book.
    sources: set[str] = set()
    passages = []
    for record in library:
        if record.length_words >= 120 and record.source_file and record.source_file not in sources:
            passages.append(record)
            sources.add(record.source_file)
    if len(passages) < 13:
        raise ValueError("need 13 independent source documents in the English library")
    random.Random(20260906).shuffle(passages)
    runtime, grading = [], []
    for index, (family, variant) in enumerate((fam, v) for fam in FAMILIES for v in (0, 1)):
        record = passages[index]
        # Whole-word prefix near 500 letters; no solver-based reroll or filtering.
        words, count = [], 0
        for word in record.text.split():
            words.append(word)
            count += len(word)
            if count >= (500 if family == "quagmire3" else 320):
                break
        text = " ".join(words)
        seed = 60000 + index
        ciphertext, fmt, validation = encrypt_case(family, text, random.Random(seed), variant=variant)
        case_id = opaque_id(index)
        runtime.append(runtime_case(case_id, ciphertext, fmt, "en"))
        grading.append(dict(case_id=case_id, family=family,
                            role="development" if variant == 0 else "held_out",
                            generation_seed=seed, source_file=record.source_file,
                            source_record=record.id, source_hash=record.content_hash,
                            source_provenance=record.provenance,
                            expectation="measure_recovery_no_success_filter",
                            **validation))
    unavailable = []
    try:
        loader = BenchmarkLoader(benchmark_root)
    except (OSError, ValueError) as exc:
        loader = None
        unavailable.append({"component": "benchmark", "reason": type(exc).__name__})
    for index, (split, test_id, family) in enumerate(HISTORICAL, start=12):
        case_id = opaque_id(index)
        try:
            if loader is None:
                raise FileNotFoundError("benchmark unavailable")
            test = next(t for t in loader.load_tests(split, track="transcription2plaintext") if t.test_id == test_id)
            data = loader.load_test_data(test)
            if not data.canonical_transcription or not data.plaintext:
                raise ValueError("missing transcription or plaintext")
            runtime.append(runtime_case(case_id, data.canonical_transcription, "canonical", data.plaintext_language))
            grading.append(dict(case_id=case_id, family=family, role="historical_diagnostic",
                                benchmark_test_id=test_id, plaintext_spaced=data.plaintext,
                                plaintext=letters(data.plaintext), source_hash=digest(data.canonical_transcription),
                                expectation="measure_partial_recovery_not_fresh_holdout",
                                roundtrip_verified=None, limitation="historical transcription/key fidelity requires source review"))
        except (OSError, ValueError, StopIteration) as exc:
            unavailable.append({"case_id": case_id, "reason": type(exc).__name__})
    record = passages[12]
    text = " ".join(record.text.split()[:75])
    ciphertext, fmt, validation = encrypt_case("bifid", text, random.Random(60100), variant=0)
    runtime.append(runtime_case(opaque_id(16), ciphertext, fmt, "en"))
    grading.append(dict(case_id=opaque_id(16), family="bifid", role="unsupported_control",
                        source_file=record.source_file, source_hash=record.content_hash,
                        expectation="no_unsupported_solve_claim", **validation))
    rng = random.Random(60101)
    runtime.append(runtime_case(opaque_id(17), "".join(rng.choice(string.ascii_uppercase) for _ in range(320)), "letters", "en"))
    grading.append(dict(case_id=opaque_id(17), family=None, role="random_control", plaintext=None,
                        expectation="no_plaintext_claim", roundtrip_verified=None))
    # Runtime order must not advertise family or development/holdout pairs.
    runtime.sort(key=lambda row: row["case_id"])
    protocol = {
        "arms": ["blind_family", "family_supplied"], "language": "same supplied language in both arms",
        "wall_seconds_per_run": 180, "cpu_seconds_per_process": 720,
        "workers": 4, "concurrent_runs": 1, "solver_profile": "shipped_defaults",
        "homophonic_budget": "screen", "transform_search": "off",
        "primary_seed": 61001, "development_seeds": [61001, 61002, 61003],
        "replication": "all six development cases; both arms; no adaptive repeats",
        "held_out_seeds": [61001], "additional_repeats": "none",
        "max_runs": 60, "max_wall_seconds": 10800,
        "timeouts": "report timeout without treating partial output as completion",
        "thresholds": {"near_exact_char": 0.99, "material_char_gap": 0.02},
        "interpretation": "small descriptive pilot; no family-wide or live-agent success claims",
        "seed_limitation": "R4 must record which engines honor injected seeds; fixed internal seeds are not independent replicates",
    }
    return runtime, {"packet_version": PACKET_VERSION, "cases": grading, "unavailable": unavailable,
                     "protocol": protocol,
                     "contamination": "fresh keys/ciphertexts from published corpus; source-held-out within packet, not guaranteed absent from model training",
                     "generation_policy": "one generation per case; no solve, grade, reroll, or success filtering"}


def _model_evidence(value: Any, path: str = "") -> list[dict]:
    """Extract explicit model/config provenance without candidate prose or keys."""
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}" if path else key
            if key in {"model_path", "model_variant", "model_sha256", "model_checksum", "git_commit", "git_head", "solver_profile"} and isinstance(child, (str, int, float)):
                rows.append({"field": here, "value": child})
            elif key not in {"messages", "session_transcript", "raw", "decryption", "ground_truth", "key"}:
                rows.extend(_model_evidence(child, here))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            rows.extend(_model_evidence(child, f"{path}[{i}]"))
    return rows


def audit_artifact(artifact: dict, path: Path | None = None) -> dict:
    base = {"run_id": artifact.get("run_id"), "artifact": str(path.relative_to(ROOT)) if path and path.is_relative_to(ROOT) else str(path) if path else None,
            "artifact_sha256": file_digest(path) if path else digest(artifact),
            "test_id": artifact.get("cipher_id"), "historical_status": artifact.get("status"),
            "historical_delivered_char": artifact.get("char_accuracy"),
            "historical_delivered_word": artifact.get("word_accuracy"),
            "historical_cost_usd": artifact.get("estimated_cost_usd"),
            "requested_model": artifact.get("model"), "served_models": artifact.get("served_models") or [],
            "explicit_provenance": _model_evidence(artifact),
            "historical_code_revision": artifact.get("git_commit") or artifact.get("git_head"),
            "generated_menu_completeness": "unknown: final snapshots do not prove all generated candidates were saved"}
    if not isinstance(artifact.get("investigation_state"), dict):
        return dict(base, replay_status="unavailable", reason="missing_investigation_state")
    state = InvestigationState.from_artifact_dict(artifact["investigation_state"])
    restored = InvestigationState.from_artifact_dict(state.to_artifact_dict())
    report = analyze_artifact_data(artifact)
    base["ciphertext_sha256"] = digest(state.workspace.cipher_text.raw)
    snapshots = {b["name"]: b for b in artifact.get("branches", [])}
    selected = (report.get("selected_branch") or (artifact.get("solution") or {}).get("branch")
                or (artifact.get("fallback_selection") or {}).get("branch"))
    branches = []
    for row in report["branches"]:
        name = row["branch"]
        text = _decoded_text_for_panel(state.workspace, name)
        after = _decoded_text_for_panel(restored.workspace, name)
        snapshot = snapshots.get(name)
        old = row.get("unsegmented_variant") or {}
        branches.append({key: row.get(key) for key in (
            "branch", "roles", "renderer", "boundary_mode", "content_hash", "mapped_count",
            "post_hoc", "solver_rank", "post_hoc_rank", "historical_boundary_loss")} | {
                "roundtrip_content_equal": text == after,
                "roundtrip_structure_equal": state.workspace.get_branch(name).word_spans == restored.workspace.get_branch(name).word_spans
                    and state.workspace.get_branch(name).token_order == restored.workspace.get_branch(name).token_order
                    and state.workspace.get_branch(name).transform_pipeline == restored.workspace.get_branch(name).transform_pipeline,
                "saved_render_hash": _candidate_content_hash(snapshot["decryption"]) if snapshot and isinstance(snapshot.get("decryption"), str) else None,
                "matching_verdicts": [{key: att.get(key) for key in ("content_hash", "reader_accepts_as_solution", "damage_scope", "episode_id")}
                                      for att in row["attestations"]],
                "historical_unsegmented_score": old.get("post_hoc"),
                "historical_unsegmented_verdict_count": len(old.get("attestations") or []),
            })
    best = next((b for b in branches if b["branch"] == report["post_hoc_best_branch"]), None)
    chosen = next((b for b in branches if b["branch"] == selected), None)
    gap = None
    if best and chosen and best.get("post_hoc") and chosen.get("post_hoc"):
        gap = best["post_hoc"]["char_accuracy"] - chosen["post_hoc"]["char_accuracy"]
    return base | {"replay_status": "completed", "selected_branch": selected,
                   "current_replay_best_saved_char": best["post_hoc"]["char_accuracy"] if best and best.get("post_hoc") else None,
                   "current_replay_selection_char_gap": gap, "branches": branches,
                   "findings": report["findings"],
                   "roundtrip_failures": [b["branch"] for b in branches if not b["roundtrip_content_equal"] or not b["roundtrip_structure_equal"]]}


def audit_saved(artifact_root: Path) -> list[dict]:
    paths: dict[str, list[Path]] = {run: [] for run in AUDIT_RUNS}
    for path in artifact_root.rglob("*.json"):
        if path.stem in paths:
            paths[path.stem].append(path)
    rows = []
    for run, matches in paths.items():
        if len(matches) != 1:
            rows.append({"run_id": run, "replay_status": "unavailable", "reason": "missing_artifact" if not matches else "ambiguous_artifact", "match_count": len(matches)})
            continue
        path = matches[0]
        try:
            rows.append(audit_artifact(json.loads(path.read_text()), path))
        except (ValueError, KeyError, TypeError) as exc:
            rows.append({"run_id": run, "artifact_sha256": file_digest(path), "replay_status": "unavailable", "reason": type(exc).__name__})
    return rows


def markdown_report(audit: dict) -> str:
    lines = ["# R0 reliability baseline", "", "Grading-side diagnostic. No solver or provider calls were made.", "",
             f"Audit code revision: `{audit['audit_revision']}`; packet `{PACKET_VERSION}`.", "",
             "## Saved artifacts", "", "Historical delivery and current-code replay are separate measurements.", "",
             "| Run | Replay | Historical delivered char | Best saved char, current render | Current selection gap | Roundtrip failures |",
             "|---|---|---:|---:|---:|---:|"]
    pct = lambda x: "unknown" if x is None else f"{100*x:.2f}%"
    for row in audit["saved_artifacts"]:
        lines.append(f"| `{row['run_id']}` | {row['replay_status']} | {pct(row.get('historical_delivered_char'))} | {pct(row.get('current_replay_best_saved_char'))} | {pct(row.get('current_replay_selection_char_gap'))} | {len(row['roundtrip_failures']) if 'roundtrip_failures' in row else 'unknown'} |")
    lines += ["", "## Pilot", "", f"{audit['available_cases']}/18 runtime cases available. Generation used fixed keys/seeds without solver-based rerolls.",
              "Six supported families have distinct-source development and held-out cases; four historical anchors and two negative controls complete the pilot.",
              "Plaintext, keys, family labels, source references and role membership are in the gitignored grading store. Runtime JSON contains only opaque id, ciphertext, format, language and ciphertext hash.",
              "Published corpus passages may be known to LLMs; freshness refers to keys/ciphertexts and the within-packet source split.", "",
              "## Fixed R4 protocol", "", "Two family-blind/family-supplied arms; identical language, models and limits. One primary run per case/arm, plus two additional predeclared seeds on each of six development cases: at most 60 runs.",
              "180 seconds wall per run, 720 CPU seconds per process, four workers, one concurrent run, three hours total wall ceiling. Timeouts and engines ignoring supplied seeds are reported, not rerolled.",
              "Near-exact character recovery: 99%; material paired character gap: 2 percentage points. Both are descriptive criteria, not declaration gates.", "",
              "## Evidence limitations and next work", "",
              "- No historical artifact is assumed to include every generated candidate. Best-saved is a lower bound on best-generated quality.",
              "- Missing historical revision/model checksum remains unknown; the current model inventory cannot retroactively establish old provenance.",
              "- Saved verifier verdicts apply only to their original content hashes. Corrected rendering requires a new verdict.",
              "- R1 exercises persistence and interface parity; R2 tests comparison/retention, guided by the per-run findings below.",
              "- R3 audits residual/verification labels; R4 alone measures new routing outcomes. Local replay makes no claim about Astra or live-agent improvement."]
    for row in audit["saved_artifacts"]:
        lines += ["", f"### `{row['run_id']}`"]
        lines += [f"- {finding}" for finding in row.get("findings", [])] or [f"- {row.get('reason', 'No candidate-selection finding in this saved state.')}"]
    return "\n".join(lines) + "\n"


def write_new_or_identical(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"refusing to overwrite frozen output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-root", type=Path, default=ROOT.parent / "cipher_benchmark" / "benchmark")
    ap.add_argument("--out", type=Path, required=True, help="New packet directory, containing runtime and grading subdirectories")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/reports/reliability_r0_baseline.md")
    args = ap.parse_args()
    runtime, grading = build_packet(args.benchmark_root)
    models = {}
    for language in ("en", "la", "de"):
        path = resolve_language_model(language)
        models[language] = {"path": str(path), "sha256": file_digest(path)} if path else None
    grading["model_files"] = models
    grading["runtime_manifest_sha256"] = digest(runtime)
    audit = {"audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "audit_script_sha256": file_digest(Path(__file__)), "available_cases": len(runtime),
             "runtime_manifest_sha256": digest(runtime), "model_files": models,
             "unavailable_cases": grading["unavailable"], "saved_artifacts": audit_saved(ROOT / "artifacts")}
    encode = lambda x: json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    outputs = {
        args.out / "runtime/cases.jsonl": "".join(json.dumps(case, sort_keys=True) + "\n" for case in runtime),
        args.out / "grading/packet.json": encode(grading),
        args.report: markdown_report(audit), args.report.with_suffix(".json"): encode(audit),
    }
    # Check all conflicts before writing anything; reruns are idempotent.
    for path, content in outputs.items():
        if path.exists() and path.read_text() != content:
            raise ValueError(f"refusing to overwrite frozen output: {path}")
    for path, content in outputs.items():
        write_new_or_identical(path, content)
    print(json.dumps({"runtime_cases": len(runtime), "audit_rows": len(audit["saved_artifacts"]),
                      "report": str(args.report), "packet": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
