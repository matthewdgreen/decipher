#!/usr/bin/env python3
"""R3 grading-side audit. Never import from solving/verification/runtime code.

Reads fixed saved candidates, never searches, verifies, edits an investigation,
or changes a runtime policy. Human source review is an explicit unfinished gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from benchmark.loader import parse_canonical_transcription
from benchmark.scorer import align_char_sequences, align_word_sequences, normalize_text, score_decryption
from investigation.candidates import candidate_packet_for_branch
from investigation.state import InvestigationState
from report_v3_candidate_selection import _unsegmented_null_mask_text

VERSION = "reliability-r3-v1"
ANCHORS = [
    ("borg", "borg_0109v", "artifacts/m5_3_targeted_smokes/repair_path/v3/borg_single_B_borg_0109v/1/borg_single_B_borg_0109v/8d5bce9769b1.json"),
    ("borg", "borg_0045v", "artifacts/m5_1_stage1_20260715_retry1/v3/borg_single_B_borg_0045v/1/borg_single_B_borg_0045v/683a61a9b8cb.json"),
    ("copiale", "copiale_p017", "artifacts/baseline_20260713/copiale_null_masks/decipher/automated_only/copiale_single_B_copiale_p017/4220d4ce8509.json"),
    ("copiale", "copiale_p068", "artifacts/baseline_20260713/copiale_null_masks/decipher/automated_only/copiale_single_B_copiale_p068/f4d013acfd27.json"),
]
RUNTIME_FIELDS = {"case_id", "language", "candidate_text", "content_hash"}


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else {"path": str(path), "missing": True}


def stream(text, *, unknowns=False):
    return "".join(c for c in text.upper() if c.isalpha() or (unknowns and c == "?"))


def reference_view(text):
    """Diagnostic projection, not a new scoring policy or reconstructed source.

    Brackets/correction marks follow the existing scorer. Starred Copiale units
    are retained in a separate ledger, NOT interpreted as spelled-out letters.
    """
    annotations = [{"start": m.start(), "end": m.end(), "text": m.group(),
                    "class": "editorial_notation", "confidence": "notation_only",
                    "source_support": "unreviewed"}
                   for m in re.finditer(r"\[.*?\]|<[=!].*?>|\(\?\)", text)]
    units = [{"start": m.start(), "end": m.end(), "text": m.group(),
              "class": "logogram_marker", "confidence": "notation_only",
              "source_support": "unreviewed"}
             for m in re.finditer(r"\*[^*]+\*", text)]
    normalized = normalize_text(text)
    projected = re.sub(r"\*[^*]+\*", "", normalized)
    return projected, annotations + units


def optimal_partners(a, b):
    """All partners/gaps on optimal paths under the *existing* scorer weights.

    A deterministic tie-break is not evidence for a unique token alignment.
    Return every possible reference partner of each decoded character.
    """
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    r = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        f[i][0], r[i][n] = -i, -(m-i)
    for j in range(n + 1):
        f[0][j], r[m][j] = -j, -(n-j)
    for i in range(1, m+1):
        for j in range(1, n+1):
            f[i][j] = max(f[i-1][j-1] + (2 if a[i-1] == b[j-1] else -1), f[i-1][j]-1, f[i][j-1]-1)
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            r[i][j] = max(r[i+1][j+1] + (2 if a[i] == b[j] else -1), r[i+1][j]-1, r[i][j+1]-1)
    partners = [set() for _ in a]
    for i in range(m):
        for j in range(n+1):
            if f[i][j] - 1 + r[i+1][j] == f[m][n]:
                partners[i].add(None)
            if j < n and f[i][j] + (2 if a[i] == b[j] else -1) + r[i+1][j+1] == f[m][n]:
                partners[i].add(j)
    return partners


def residuals(candidate, reference, symbols=None):
    """Exhaustive character edit ledger; causal labels are provisional patterns."""
    projected, notation = reference_view(reference)
    a, b = stream(candidate, unknowns=True), stream(projected)
    aligned = align_char_sequences(a, b)
    partners = optimal_partners(a, b)
    indexed = {row.decoded_index: row for row in aligned if row.decoded_index is not None}
    usable = symbols is not None and len(symbols) == len(a)
    occurrences = defaultdict(list)
    if usable:
        for i, symbol in enumerate(symbols):
            occurrences[symbol].append(i)
    rows = []
    next_decoded, next_reference = 0, 0
    for row in aligned:
        ci = row.decoded_index if row.decoded_index is not None else next_decoded
        cj = row.ground_truth_index if row.ground_truth_index is not None else next_reference
        if row.decoded_index is not None:
            next_decoded = ci + 1
        if row.ground_truth_index is not None:
            next_reference = cj + 1
        if row.op == "match":
            continue
        i, j = row.decoded_index, row.ground_truth_index
        label, confidence = "ambiguous_unclassified", "low"
        evidence = ["Edit alignment alone does not establish the cause."]
        symbol = symbols[i] if usable and i is not None else None
        unique = i is not None and partners[i] == {j} and j is not None
        if row.op == "substitute" and unique and symbol is not None:
            positions = occurrences[symbol]
            group = [indexed[p] for p in positions]
            stable = all(partners[p] == {indexed[p].ground_truth_index}
                         and indexed[p].ground_truth_index is not None for p in positions)
            mismatches = [x for x in group if x.op != "match"]
            if stable and len(mismatches) == len(group) and len({x.ground_truth for x in group}) == 1:
                label = "E1_consistent_mapping_candidate"
                confidence = "conditional" if len(group) > 1 else "low"
                evidence = [f"All {len(group)} rendered occurrences align to the same different reference letter.",
                            "Conditional on reference fidelity and the one-letter-per-rendered-token model."]
            elif stable and len(mismatches) == 1 and len(group) > 1:
                label = "E3_occurrence_conflict_candidate"
                evidence = [f"One of {len(group)} aligned occurrences differs; the others agree.",
                            "This is NOT evidence of a transcription scar or the proposed replacement's truth."]
        if i is not None and len(partners[i]) != 1:
            evidence.append("Multiple optimal alignments give this character different partners/gaps.")
        rows.append({**asdict(row), "class": label, "confidence": confidence, "symbol": symbol,
                     "unique_optimal_partner": unique, "evidence": evidence,
                     "candidate_context": a[max(0,ci-12):ci+13],
                     "reference_context": b[max(0,cj-12):cj+13],
                     "source_support": "unreviewed", "review": None})
    # Boundary attribution is safe only when the underlying letter stream is
    # identical. Otherwise word edits are recorded, not relabeled as boundaries.
    def words(text):
        return [v for w in text.replace("|", " ").split() if (v := stream(w, unknowns=True))]
    aw, bw = words(candidate), words(projected)
    word_rows = [{**asdict(row), "class": "E2_boundary" if a == b else "ambiguous_unclassified",
                  "confidence": "conditional" if a == b else "low", "source_support": "unreviewed", "review": None}
                 for row in align_word_sequences(aw, bw) if row.op != "match"]
    counts = dict(Counter(row["class"] for row in rows))
    confidence_counts = dict(Counter(f"{row['class']}:{row['confidence']}" for row in rows))
    score = asdict(score_decryption("r3-post-hoc", candidate, reference, 0, "diagnostic"))
    return {"candidate_letters": len(a), "reference_letters": len(b),
            "candidate_letter_sha256": digest(a), "reference_letter_sha256": digest(b),
            "reference_letter_exact": a == b, "token_alignment_available": usable,
            "character_residual_count": len(rows), "character_class_counts": counts,
            "character_fractions": {k: v/len(rows) for k, v in counts.items()} if rows else {},
            "class_confidence_counts": confidence_counts,
            "class_confidence_fractions": {k: v/len(rows) for k, v in confidence_counts.items()} if rows else {},
            "character_residuals": rows, "word_residual_count": len(word_rows), "word_residuals": word_rows,
            "notation_units": notation, "current_scorer": score,
            "normalization_effect": {"literal_reference_letters": len(stream(reference)),
                                     "projected_reference_letters": len(b),
                                     "excluded_notation_units": len(notation),
                                     "note": "Diagnostic letters differ from punctuation/logogram-sensitive score; not extra accuracy credit."},
            "reviewed_character_count": 0, "reviewed_fractions": None}


def saved_candidate(a, canonical):
    """Select only the recorded delivered branch, never a grading-selected best."""
    if a.get("investigation_state"):
        state = InvestigationState.from_artifact_dict(a["investigation_state"])
        name = ((a.get("branch_roles") or {}).get("declared_or_selected_branch")
                or (a.get("solution") or {}).get("branch")
                or (a.get("fallback_selection") or {}).get("branch"))
        if not name:
            recorded = re.search(r"Best inspected branch: ([\w.-]+)\.", a.get("final_summary") or "")
            name = recorded.group(1) if recorded else None
        if not name or not state.workspace.has_branch(name):
            raise ValueError("no retrievable recorded delivered branch")
        packet = candidate_packet_for_branch(state.workspace, name)
        ids = state.workspace.effective_tokens(name)
        symbols = [state.cipher.alphabet.symbol_for(ids[i]) for i in packet.rendered_token_indices]
        kept = set(packet.rendered_token_indices)
        replay = "".join(state.workspace.plaintext_alphabet.symbol_for(packet.key[t]) if t in packet.key else "?"
                         for i, t in enumerate(ids) if i in kept)
        consistent = replay == stream(packet.text, unknowns=True)
        canonical_ct = parse_canonical_transcription(canonical)
        native_tokens = state.cipher.tokens
        forward, backward = defaultdict(set), defaultdict(set)
        for native, source in zip(native_tokens, canonical_ct.tokens):
            forward[native].add(source)
            backward[source].add(native)
        corresponds = (len(native_tokens) == len(canonical_ct.tokens)
                       and all(len(v) == 1 for v in forward.values())
                       and all(len(v) == 1 for v in backward.values()))
        positions = [(packet.token_order[i] if packet.token_order is not None else i)
                     for i in packet.rendered_token_indices]
        structure = {"null_mask": list(packet.null_mask), "word_spans": packet.word_spans,
                     "token_order": packet.token_order, "source_cipher_correspondence": corresponds,
                     "rendered_index_to_source_token": positions if corresponds else None,
                     "source_symbols": [canonical_ct.alphabet.symbol_for(canonical_ct.tokens[i]) for i in positions] if corresponds else None}
        return name, packet.text, symbols if consistent else None, consistent, structure
    ct = parse_canonical_transcription(canonical)
    key = {int(k): int(v) for k, v in a.get("key", {}).items()}
    mask = next((set(step.get("selected_mask") or []) for step in reversed(a.get("steps") or [])
                 if step.get("name") == "search_null_masks"), set())
    tokens = [t for t in ct.tokens if ct.alphabet.symbol_for(t) not in mask]
    replay = "".join(string.ascii_uppercase[key[t]] if t in key else "?" for t in tokens)
    text = a.get("decryption")
    if not isinstance(text, str):
        raise ValueError("saved decryption missing")
    consistent = replay == stream(text, unknowns=True)
    positions = [i for i, t in enumerate(ct.tokens) if ct.alphabet.symbol_for(t) not in mask]
    structure = {"null_mask": sorted(mask), "source_cipher_correspondence": "key_replay_only",
                 "rendered_index_to_source_token": positions if consistent else None,
                 "source_symbols": [ct.alphabet.symbol_for(t) for t in tokens] if consistent else None}
    return "recorded_automated_result", text, [ct.alphabet.symbol_for(t) for t in tokens] if consistent else None, consistent, structure


def historical_audit(benchmark_root):
    result = []
    for source, page, artifact in ANCHORS:
        base = benchmark_root / "sources" / source
        sources = {"canonical": source_record(base / "transcriptions" / f"{page}.canonical.txt"),
                   "diplomatic": source_record(base / "transcriptions" / f"{page}.diplomatic.txt"),
                   "reference": source_record(base / "plaintext" / f"{page}.txt"),
                   "image": source_record(base / "images" / f"{page}.{'jpg' if source == 'borg' else 'png'}"),
                   "symbol_map": source_record(base / "metadata" / f"{source}_symbol_map.json")}
        row = {"page": page, "source": source, "sources": sources, "artifact": source_record(ROOT / artifact),
               "selection_rule": "recorded delivered candidate; Borg 0045v Stage-1 retry cohort replicate 1 (endgame partial, not failed initial preflight), fixed July-13 Copiale baseline; not a fresh solve or runtime selection",
               "human_source_review": "pending"}
        try:
            a = load(ROOT / artifact)
            canonical = Path(sources["canonical"]["path"]).read_text()
            ref = Path(sources["reference"]["path"]).read_text()
            branch, text, symbols, replay, structure = saved_candidate(a, canonical)
            row.update(branch=branch, candidate_text=text, content_hash=_candidate_content_hash(text),
                       key_replay_consistent=replay, language=a.get("language"),
                       source_reference=ref, candidate_structure=structure, audit=residuals(text, ref, symbols))
            positions = structure.get("rendered_index_to_source_token")
            source_symbols = structure.get("source_symbols")
            for residual in row["audit"]["character_residuals"]:
                i = residual["decoded_index"]
                residual["source_token_index"] = positions[i] if positions is not None and i is not None else None
                residual["canonical_symbol"] = source_symbols[i] if source_symbols is not None and i is not None else None
        except (OSError, KeyError, ValueError) as exc:
            row["unavailable"] = f"{type(exc).__name__}: {exc}"
        result.append(row)
    return result


def synthetic_packet():
    """Known construction labels; no assertion of historical source review."""
    text = "THE PHYSICIAN SENT WATER TO THE HOUSE AND WAITED UNTIL MORNING"
    cases = [
        ("exact_spaced", text, text, True, True, "none", "Identity construction."),
        ("exact_unspaced", text, stream(text), True, True, "none", "Remove spaces only."),
        ("historical_orthography", "THE PHYSICK HATH DONE HIM GOOD", "THE PHYSICK HATH DONE HIM GOOD", True, True, "none", "Constructed archaic spelling, not an authenticated historical quotation."),
        ("abbreviation", "RX AQ ROS", "RX AQ ROS", True, None, "unresolved", "Literal abbreviation is known; its expansion/meaning is deliberately unspecified."),
        ("fluent_wrong", "CAT", "DOG", False, True, "none", "CAT encrypted by Caesar+3 is FDW; both FDW->CAT and FDW->DOG are consistent partial bijections. Replay cannot prove the intended key."),
        ("localized_key", text, text.replace("P", "Z"), False, None, "none", "One globally changed symbol mapping (P->Z)."),
        ("distributed_key", text, text.replace("T", "Z"), False, None, "none", "All occurrences of one cipher symbol changed; distributed damage can be one key error."),
        ("boundaries", "THE CAT SLEEPS", "THECAT SLE EPS", True, None, "none", "Boundaries changed; letters untouched."),
        ("segmentation_ambiguity", "THERAPIST", "THE RAPIST", True, None, "unresolved", "Same letters support different word readings; no intended meaning supplied."),
        ("editorial_addition", "WE SENT MEN", "WE SENT SIX MEN", False, True, "proposed_only", "SIX is an editorial proposal absent from the encoded stream; never merge into canonical decode."),
        ("occurrence_damage", text, text.replace("T", "Z", 1), False, None, "oracle_only", "Exactly one decoded occurrence was changed by the fixture; not a historical scar claim."),
        ("editorial_reference", "WE SENT [SIX] MEN", "WE SENT MEN", True, True, "reference_notation", "Existing scorer excludes bracketed material; SIX is recorded separately, not cipher-derived."),
        ("fluent_right_same_text", "DOG", "DOG", True, True, "none", "Same candidate text as fluent_wrong, but DOG really was encoded. A candidate-only reader cannot distinguish their reconstruction truth."),
    ]
    runtime, grading = [], []
    for i, (kind, ref, candidate, exact, readable, editorial, construction) in enumerate(cases):
        cid = "v" + digest(f"{VERSION}:{i}")[:12]
        runtime.append({"case_id": cid, "language": "en", "candidate_text": candidate,
                        "content_hash": _candidate_content_hash(candidate)})
        cipher_plain = stream(reference_view(ref)[0])
        ciphertext = cipher_plain.translate(str.maketrans(string.ascii_uppercase, string.ascii_uppercase[3:]+string.ascii_uppercase[:3]))
        grading.append({"case_id": cid, "kind": kind, "reference": ref, "ciphertext": ciphertext,
                        "known_key": "Caesar+3", "construction": construction,
                        "labels": {"cipher_reconstruction_exact": exact, "intelligible_reading": readable,
                                   "editorial_restoration": editorial, "historical_policy_acceptance": None},
                        "label_basis": "synthetic construction; readability is author-provisional, not independent validation",
                        "human_review": None})
    return runtime, grading


def text_pool(a):
    """Locate exact historical attested content, never bind to changed text."""
    pool = {}
    def add(text, origin):
        if isinstance(text, str) and text:
            pool.setdefault(_candidate_content_hash(text), (text, origin))
    for branch in a.get("branches") or []:
        add(branch.get("decryption"), "artifact_branch_snapshot")
    for event in a.get("loop_events") or []:
        add((event.get("payload") or {}).get("decryption"), "captured_loop_event")
    add((a.get("automated_preflight") or {}).get("decryption"), "saved_preflight")
    if a.get("investigation_state"):
        state = InvestigationState.from_artifact_dict(a["investigation_state"])
        for name in state.workspace.branch_names():
            add(_decoded_text_for_panel(state.workspace, name), "hash_matching_current_replay")
            add(_unsegmented_null_mask_text(state.workspace, name), "hash_matching_legacy_null_render")
    return pool


def verifier_audit(r0):
    rows = []
    for saved in r0["saved_artifacts"]:
        path = ROOT / saved["artifact"]
        a = load(path)
        if source_record(path)["sha256"] != saved["artifact_sha256"]:
            raise ValueError(f"Frozen artifact changed: {path}")
        state = a.get("investigation_state") or {}
        pool = text_pool(a)
        episodes = {e.get("episode_id"): e for e in state.get("episode_ledger") or []}
        for att in a.get("attestations") or state.get("verify_attestations") or []:
            h = att.get("content_hash")
            found = pool.get(h)
            episode = episodes.get(att.get("episode_id"), {})
            budgets = [b for b in episode.get("budget_entries") or [] if b.get("category") == "episode:verify"]
            modern = "reader_accepts_as_solution" in att
            # Do not apply today's legacy threshold retroactively: old prompt/
            # policy versions are unavailable. Preserve raw legacy values only.
            acceptance = att.get("reader_accepts_as_solution") if modern else None
            truth = a.get("ground_truth")
            synthetic = a.get("cipher_id") in {"m5_3_positive_word_boundary_control", "synth_en_250nb_s4"}
            exact = stream(found[0]) == stream(reference_view(truth)[0]) if found and truth and synthetic else None
            rows.append({"run_id": a.get("run_id"), "artifact": source_record(path),
                         "episode_id": att.get("episode_id"), "content_hash": h,
                         "original_content_available": found is not None,
                         "content_recovery": found[1] if found else None,
                         "candidate_text": found[0] if found else None,
                         "original_attestation": att,
                         "recorded_model_usage": [{k: b.get(k) for k in ("provider", "model")} for b in budgets],
                         "served_identity": None, "policy_version": att.get("policy_version"),
                         "contract_observed": "explicit_solution_acceptance" if modern else "legacy_acceptance_unresolved",
                         "original_prompt_available": False, "accepted_under_recorded_contract": acceptance,
                         "reference_reconstruction_exact": exact,
                         "reference_label_basis": "saved synthetic reference, letter identity" if exact is not None else "unreviewed_or_reference_unavailable",
                         "historical_policy_acceptance_label": None,
                         "limitation": "Unknown prompt/policy version; current-provider quality rates cannot be inferred."})
    known = [r for r in rows if r["reference_reconstruction_exact"] is not None
             and isinstance(r["accepted_under_recorded_contract"], bool)]
    negatives = [r for r in known if not r["reference_reconstruction_exact"]]
    positives = [r for r in known if r["reference_reconstruction_exact"]]
    fp = sum(r["accepted_under_recorded_contract"] is True for r in negatives)
    fn = sum(r["accepted_under_recorded_contract"] is False for r in positives)
    return {"verdicts": rows, "counts": {"total": len(rows),
            "original_content_recovered": sum(r["original_content_available"] for r in rows),
            "accepted": sum(r["accepted_under_recorded_contract"] is True for r in rows),
            "unknown_reconstruction_labels": len(rows)-len(known),
            "strict_reference_false_accepts": fp, "strict_reference_negative_denominator": len(negatives),
            "strict_reference_false_rejects": fn, "strict_reference_positive_denominator": len(positives),
            "strict_reference_false_accept_rate": fp/len(negatives) if negatives else None,
            "strict_reference_false_reject_rate": fn/len(positives) if positives else None,
            "policy_false_accepts": None, "policy_false_rejects": None, "policy_labeled_denominator": 0},
            "note": "Reference exactness and historical reader acceptance are different targets. Zero denominator is unknown, not zero error rate."}


def review_queue(anchors):
    rows = []
    for anchor in anchors:
        if "audit" not in anchor:
            continue
        a = anchor["audit"]
        for unit, entries in (("character", a["character_residuals"]), ("word", a["word_residuals"]), ("notation", a["notation_units"])):
            mandatory = {i for i, row in enumerate(entries) if row.get("confidence") == "low"
                         or row["class"].startswith("E3") or unit == "notation"}
            remainder = [i for i in range(len(entries)) if i not in mandatory]
            sampled = set(remainder[::10])
            for i, row in enumerate(entries):
                # Review ALL low-confidence/E3 cases and a deterministic >=10%
                # sample of other units. No sample selected using a verdict.
                required = i in mandatory or i in sampled
                if required:
                    rows.append({"review_id": f"{anchor['page']}:{unit}:{i}", "page": anchor["page"],
                                 "unit": unit, "index": i, "provisional": row,
                                 "candidate_content_hash": anchor.get("content_hash"),
                                 "review_binding": digest([anchor.get("content_hash"), anchor["sources"], row]),
                                 "sources": anchor["sources"], "review": None})
    return rows


def build(benchmark_root):
    r0 = load(ROOT / "docs/reports/reliability_r0_baseline.json")
    anchors = historical_audit(benchmark_root)
    runtime, grading = synthetic_packet()
    for anchor in anchors:
        if "candidate_text" not in anchor:
            continue
        cid = "v" + digest(anchor["page"])[:12]
        runtime.append({"case_id": cid, "language": anchor["language"], "candidate_text": anchor["candidate_text"], "content_hash": anchor["content_hash"]})
        grading.append({"case_id": cid, "kind": "historical_unreviewed", "page": anchor["page"],
                        "labels": {"cipher_reconstruction_exact": None, "intelligible_reading": None,
                                   "editorial_restoration": "unreviewed", "historical_policy_acceptance": None},
                        "label_basis": "Awaiting independent source review; diagnostic mismatch is not a causal label.",
                        "sources": anchor["sources"], "human_review": None})
    audit = {"version": VERSION, "script_sha256": source_record(__file__)["sha256"],
             "scope": "Grading side only; saved evidence; no solver or provider calls.",
             "anchors": anchors, "verifier_audit": verifier_audit(r0), "review_queue": review_queue(anchors),
             "packet_count": len(runtime), "runtime_packet_sha256": digest(runtime),
             "source_review_gate": "pending_human_review", "runtime_gate_changes": False}
    return runtime, grading, audit


def markdown(audit):
    c = audit["verifier_audit"]["counts"]
    lines = ["# R3 — Verification and historical residual audit", "", f"Packet: {audit['version']}. No solver/provider calls or runtime changes.", "",
             "**Status: local audit complete; required human source review is pending.**",
             "All fractions below are provisional reference-alignment patterns, not reviewed historical causes.", "",
             "## Four fixed historical anchors", "",
             "| Page | Character edit units | E1 candidate | E3 candidate | Unclassified | Key replay matches | Human reviewed |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for row in audit["anchors"]:
        a = row.get("audit")
        if a is None:
            lines.append(f"| {row['page']} | unavailable | — | — | — | — | 0 |")
            continue
        counts = a["character_class_counts"]
        lines.append(f"| {row['page']} | {a['character_residual_count']} | {counts.get('E1_consistent_mapping_candidate',0)} | {counts.get('E3_occurrence_conflict_candidate',0)} | {counts.get('ambiguous_unclassified',0)} | {row['key_replay_consistent']} | 0 |")
    lines += ["", "Each row uses its recorded delivered candidate, not a post-hoc best branch. Full",
              "character and word ledgers, notation units, confidence, sources, and checksums are",
              "in the sibling JSON. Character edit units include insertions/deletions; word and",
              "notation counts use separate denominators and must not be added to character fractions.",
              "All low-confidence/E3 units and at least 10% of other units require human review.",
              "Reviewed fractions are **unavailable**, not zero. Conflicts are not transcription scars.", "",
              "## Saved verifier audit", "",
              f"- Original content recovered by exact attestation hash: {c['original_content_recovered']}/{c['total']} verdicts.",
              f"- Recorded accepts: {c['accepted']}/{c['total']}; this is not an accuracy estimate.",
              f"- Strict-reference false rejections: {c['strict_reference_false_rejects']}/{c['strict_reference_positive_denominator']} known-positive synthetic references.",
              f"- Strict-reference false acceptances: {c['strict_reference_false_accepts']}/{c['strict_reference_negative_denominator']} known-negative references; rate is undefined with zero denominator.",
              f"- Uncertain/unavailable reconstruction labels: {c['unknown_reconstruction_labels']}/{c['total']}.",
              "- Historical-policy false acceptance/rejection rates: **not estimable** (0 independently labeled verdicts).",
              "", "Only the two predeclared synthetic control runs get reference-identity labels.",
              "Recorded per-episode model usage is preserved; served identity and exact historical",
              "prompt/policy version are missing. The observed contract is not a recovered policy",
              "version. No verdict is reassigned to a changed rendering or treated as a current-provider rate.", "",
              "## Calibration and next measurement", "",
              f"The runtime-only calibration file has {audit['packet_count']} cases: thirteen constructed",
              "controls and four unreviewed historical candidates. Construction labels and sources",
              "are in a separate grading file. Intelligibility labels are author-provisional;",
              "ambiguous cases explicitly retain null labels. No human review is fabricated.",
              "The matched DOG/DOG pair has identical reader input but opposite reconstruction",
              "labels. It demonstrates an information limit, not a model failure: text-only",
              "verification cannot establish cipher correctness even when a reading is fluent.",
              "", f"The source-review queue contains {len(audit['review_queue'])} required units.",
              "See `docs/reliability_r3_outcome_contract.md` for outcome axes, review requirements,",
              "and the bounded prospective experiment. No prospective run is authorized or executed.", ""]
    return "\n".join(lines)


def write_new(path, content):
    """Reproducible output; never overwrite changed audit evidence."""
    path = Path(path)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Refusing changed output: {path}; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent / "cipher_benchmark/benchmark")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    runtime, grading, audit = build(args.benchmark_root)
    payloads = {args.out / "runtime/candidates.jsonl": "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in runtime),
                args.out / "grading/labels.json": json.dumps(grading, ensure_ascii=False, indent=2)+"\n",
                args.report.with_suffix(".json"): json.dumps(audit, ensure_ascii=False, indent=2)+"\n",
                args.report: markdown(audit)}
    # Check all targets before writing any of them.
    for path, content in payloads.items():
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing changed output: {path}")
    for path, content in payloads.items():
        write_new(path, content)
    print(json.dumps({"report": str(args.report), "cases": len(runtime),
                      "review_units": len(audit["review_queue"]), "verdicts": audit["verifier_audit"]["counts"]}))


if __name__ == "__main__":
    main()
