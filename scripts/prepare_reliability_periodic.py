#!/usr/bin/env python3
"""R5a grading-side construction and freeze. Never runs a solver or provider."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import string
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_reliability_packet import (
    digest, encrypt_case, file_digest, letters, runtime_case,
)
from analysis.polyalphabetic import encode_quagmire_plaintext
from ciphers.polyalphabetic import VigenereCipher
from ciphers.transposition import columnar_encrypt, columnar_decrypt
from testgen.corpus_library import load_library
from run_reliability_routing import write_once

VERSION = "reliability-r5-v1"
BASELINE_REVISION = "e1536c1"
R0_GRADING_SHA256 = "9941e6714cc3e42017ee1c527b8b7d59a4350c8cdae2bb619d349eccdbf92d5b"
SPEC_PATH = ROOT / "docs/specs/reliability_r5_periodic_routing_spec.md"
RUNTIME_FIELDS = {"case_id", "ciphertext", "format", "language", "ciphertext_sha256"}
REQUEST_FIELDS = RUNTIME_FIELDS | {"arm", "seed"}
ARMS = ("baseline", "prototype")
DESIGN = (
    ("vigenere", 240, 4, None, "simple_substitution"),
    ("vigenere", 360, 7, None, "simple_substitution"),
    ("vigenere", 600, 11, None, "columnar_transposition"),
    ("quagmire3", 360, 6, 6, "homophonic_substitution"),
    ("quagmire3", 500, 8, 7, "bifid"),
    ("quagmire3", 700, 10, 8, "random"),
)
PROTOCOL = {
    "max_runs": 32, "max_wall_seconds": 5760, "wall_seconds_per_run": 180,
    "cpu_seconds_per_process": 720, "workers": 4, "concurrent_runs": 1,
    "seed": 61001, "homophonic_budget": "screen", "transform_search": "off",
    "family_context": "none_in_both_arms", "additional_repeats": "none",
    "near_exact_char": 0.99, "material_char_gap": 0.02,
    "min_positive_gains": 3, "required_anchor_passes": 4,
    "missing_pair_policy": "cannot_pass_acceptance",
    "random_accuracy": "unknown_no_reference",
    "prototype_switch": "DECIPHER_PERIODIC_ROUTING=probe_v1",
    "baseline_switch": "DECIPHER_PERIODIC_ROUTING=off",
    "probe_wall_seconds": 45,
}


def prefix(record, target):
    words, count = [], 0
    for word in record.text.split():
        clean = letters(word)
        if clean:
            words.append(clean)
            count += len(clean)
        if count >= target:
            return " ".join(words)
    raise ValueError("source shorter than prescribed target")


def choose_sources(records, old_labels):
    excluded = {r.get("source_file") for r in old_labels}
    old_content = {digest(letters(r.get("plaintext") or "")) for r in old_labels}
    by_source = {}
    for row in sorted(records, key=lambda r: r.id):
        if (not row.source_file or row.source_file in excluded
                or row.source_file in by_source or len(letters(row.text)) < 700):
            continue
        if digest(letters(row.text)) in old_content:
            continue
        by_source[row.source_file] = row
    candidates = list(by_source.values())
    random.Random(20260908).shuffle(candidates)
    chosen, contents = [], set(old_content)
    for row in candidates:
        content = digest(letters(row.text))
        if content not in contents:
            chosen.append(row)
            contents.add(content)
        if len(chosen) == 11:
            return chosen
    raise ValueError("need eleven disjoint unused source documents/passages")


def encrypt_positive(text, family, period, keyword_length, seed):
    rng, alpha, plain = random.Random(seed), string.ascii_uppercase, letters(text)
    if family == "vigenere":
        key = "".join(rng.choice(alpha) for _ in range(period))
        cipher = VigenereCipher().encrypt(plain, key)
        inverse = "".join(alpha[(ord(ch) - ord(key[i % period])) % 26]
                          for i, ch in enumerate(cipher))
        private_key = {"keyword": key}
    else:
        keyword = "".join(rng.sample(alpha, keyword_length))
        key = "".join(rng.choice(alpha) for _ in range(period))
        tableau = "".join(dict.fromkeys(keyword + alpha))
        cipher = encode_quagmire_plaintext(plain, cycleword=key, alphabet_keyword=keyword)
        inverse = "".join(tableau[(tableau.index(ch) - tableau.index(key[i % period])) % 26]
                          for i, ch in enumerate(cipher))
        private_key = {"alphabet_keyword": keyword, "cycleword": key}
    if inverse != plain:
        raise ValueError("independent periodic arithmetic inverse failed")
    return cipher, "letters", {
        "plaintext": plain, "plaintext_spaced": text, "key": private_key,
        "roundtrip_verified": True, "period": period, "keyword_length": keyword_length,
    }


def build_packet(records, r0_runtime, r0_grading):
    if digest(r0_runtime) != r0_grading["runtime_manifest_sha256"]:
        raise ValueError("R0 runtime identity mismatch")
    old = r0_grading["cases"]
    sources = iter(choose_sources(records, old))
    runtime, grading = [], []
    for pair_index, (family, target, period, keyword_length, control) in enumerate(DESIGN):
        positive_length = None
        for side, current_family in enumerate((family, control)):
            index = 2 * pair_index + side
            case_id, seed = "r" + digest(f"{VERSION}:{index}")[:12], 62000 + index
            source = next(sources) if current_family != "random" else None
            text = prefix(source, target) if source else None
            if side == 0:
                cipher, fmt, private = encrypt_positive(text, family, period, keyword_length, seed)
                positive_length = len(cipher)
            elif current_family == "random":
                rng = random.Random(seed)
                cipher = "".join(rng.choice(string.ascii_uppercase) for _ in range(positive_length))
                fmt, private = "letters", {"plaintext": None, "key": None, "roundtrip_verified": None}
            elif current_family == "columnar_transposition":
                keyword = "".join(random.Random(seed).sample(string.ascii_uppercase, 9))
                plain = letters(text)
                cipher = columnar_encrypt(plain, keyword)
                if columnar_decrypt(cipher, keyword) != plain:
                    raise ValueError("columnar inverse failed")
                fmt, private = "letters", {"plaintext": plain, "plaintext_spaced": text,
                    "key": {"columnar_keyword": keyword}, "roundtrip_verified": True}
            else:
                cipher, fmt, private = encrypt_case(current_family, text, random.Random(seed), variant=0)
                if current_family == "simple_substitution":
                    cipher = letters(cipher)  # boundary-free controls, like positives
            case = runtime_case(case_id, cipher, fmt, "en")
            if private["plaintext"] and len(private["plaintext"]) != (
                    len(cipher.split()) if fmt == "canonical" else len(letters(cipher))):
                raise ValueError("ciphertext/reference token count mismatch")
            runtime.append(case)
            grading.append(dict(
                case_id=case_id, role="fresh_positive" if side == 0 else "fresh_control",
                family=current_family, matched_pair=pair_index + 1, generation_seed=seed,
                target_letters=target, token_count=len(cipher.split()) if fmt == "canonical" else len(cipher),
                source_file=source.source_file if source else None,
                source_record=source.id if source else None,
                source_hash=source.content_hash if source else None,
                source_text_sha256=digest(letters(source.text)) if source else None,
                source_provenance=source.provenance if source else None, **private,
            ))
    anchors = [row for row in old if row["family"] in {"vigenere", "quagmire3"}
               and row["role"] in {"development", "held_out"}]
    if len(anchors) != 4 or len({row["case_id"] for row in anchors}) != 4:
        raise ValueError("need all four R4 periodic regression anchors")
    by_id = {row["case_id"]: row for row in r0_runtime}
    for row in sorted(anchors, key=lambda r: r["case_id"]):
        runtime.append(dict(by_id[row["case_id"]]))
        grading.append(dict(row, role="regression_anchor", previous_role=row["role"],
                            anchor_source="R0/R4; already inspected, not held-out"))
    # Reject coincident source prefixes too: no outcome-dependent replacement.
    truths = [digest(letters(row["plaintext"])) for row in grading if row.get("plaintext")]
    if len(set(truths)) != len(truths):
        raise ValueError("duplicate construction plaintext; amend packet before proceeding")
    plan = make_plan(runtime)
    private = {"schema": VERSION, "runtime_manifest_sha256": digest(runtime),
               "protocol": PROTOCOL, "cases": grading,
               "excluded_r0_sources": sorted({r["source_file"] for r in old if r.get("source_file")}),
               "generation": "fresh keys; no search or solver-driven selection",
               "training_contamination": "published source text may overlap model training"}
    return runtime, private, plan


def make_plan(runtime):
    if len(runtime) != 16 or len({r["case_id"] for r in runtime}) != 16:
        raise ValueError("expected sixteen distinct cases")
    jobs = []
    for row in sorted(runtime, key=lambda r: r["case_id"]):
        if set(row) != RUNTIME_FIELDS or row["ciphertext_sha256"] != digest(row["ciphertext"]):
            raise ValueError("runtime allowlist/hash violation")
        arms = ARMS if int(digest(row["case_id"])[:8], 16) % 2 == 0 else ARMS[::-1]
        for arm in arms:
            request = dict(row, arm=arm, seed=61001)
            jobs.append({"job_id": "j" + digest(request)[:16], "request": request,
                         "request_sha256": digest(request)})
    return {"schema": VERSION, "status": "prepared_not_run", "protocol": PROTOCOL,
            "runtime_manifest_sha256": digest(runtime), "jobs": jobs,
            "pending": "R5b implementation, worker/ledger tests and pinned campaign adapter"}


def capture_baseline():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    scopes = ("src", "rust", "resources", "models", "pyproject.toml")
    baseline = git("rev-parse", BASELINE_REVISION)
    if git("diff", baseline, "--", *scopes) or git("ls-files", "--others", "--exclude-standard", "--", *scopes):
        raise ValueError("freeze packet before solver/resource implementation changes")
    files = git("ls-files", "--", *scopes).splitlines()
    native = importlib.util.find_spec("decipher_fast")
    if native is None or not native.origin:
        raise ValueError("native module unavailable")
    path = Path(native.origin)
    natives = [path] if path.suffix != ".py" else [path, *sorted(path.parent.glob("*.so"))]
    if not any(p.suffix in {".so", ".pyd"} for p in natives):
        raise ValueError("native extension binary absent")
    return {"baseline_revision": baseline,
            "tracked_input_hashes": {p: file_digest(ROOT / p) for p in files},
            "auxiliary_resource_hashes": {str(p.relative_to(ROOT)): file_digest(p)
                for p in sorted((ROOT / "resources").rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts},
            "native_files": {str(p): file_digest(p) for p in natives},
            "native_build_correspondence": "installed binary pinned; not independently attested",
            "python": sys.version, "executable": sys.executable,
            "generator_sha256": file_digest(Path(__file__)), "spec_sha256": file_digest(SPEC_PATH)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=Path, default=ROOT / "artifacts/reliability_r0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grading_path = args.r0 / "grading/packet.json"
    if file_digest(grading_path) != R0_GRADING_SHA256:
        raise ValueError("R0 grading packet differs from the frozen R4 input")
    old = json.loads(grading_path.read_text())
    prior = [json.loads(line) for line in (args.r0 / "runtime/cases.jsonl").read_text().splitlines() if line.strip()]
    runtime, private, plan = build_packet(load_library("en"), prior, old)
    plan["provenance"] = capture_baseline()
    plan["model_files"] = old["model_files"]
    for model in plan["model_files"].values():
        if file_digest(Path(model["path"])) != model["sha256"]:
            raise ValueError("R0 model changed")
    plan["grading_packet_sha256"] = digest(private)
    outputs = {args.out / "runtime/cases.json": runtime,
               args.out / "grading/packet.json": private, args.out / "control/plan.json": plan}
    outputs.update({args.out / "runtime" / (job["job_id"] + ".json"): job["request"] for job in plan["jobs"]})
    # Validate all destinations before writing any; write_once is atomic and immutable.
    for path, value in outputs.items():
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"refusing to overwrite prepared output: {path}")
    for path, value in outputs.items():
        write_once(path, value)
    print(json.dumps({"status": plan["status"], "cases": len(runtime), "jobs": len(plan["jobs"]),
                      "runtime_sha256": digest(runtime), "grading_sha256": digest(private),
                      "plan_sha256": digest(plan), "baseline_revision": plan["provenance"]["baseline_revision"]}))


if __name__ == "__main__":
    raise SystemExit(main())
