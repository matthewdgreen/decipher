"""Hash-bound partial preferences and bounded candidate retention (no grading data)."""
from __future__ import annotations

import math
from typing import Any

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel, _score_branch_for_panel
from investigation.state import attestation_is_positive, latest_attestation_for_hash

PORTFOLIO_LIMIT = 6


def active(state, name: str) -> bool:
    if not state.workspace.has_branch(name):
        return False
    branch = state.workspace.get_branch(name)
    card = state.hypothesis_board.get(name) or {}
    return (branch.metadata.get("mode_status") not in {"rejected", "superseded"}
            and card.get("mode_status") not in {"rejected", "superseded"}
            and "rejected" not in branch.tags)


def content_hash(state, name: str) -> str:
    return _candidate_content_hash(_decoded_text_for_panel(state.workspace, name))


def fresh_compare_best_candidate(state) -> tuple[str, dict] | None:
    """Newest supported preference; legacy winner is a read-only migration.

    A null/unrankable comparison does not erase an older still-fresh partial.
    Every compared hash must match, not merely the preferred candidate's hash.
    """
    records = []
    for index, entry in enumerate(state.episode_ledger):
        if entry.get("kind") == "compare" and entry.get("status") == "ok":
            records.append((int(entry.get("turn") or entry.get("created_turn") or
                                (entry.get("comparison_binding") or {}).get("created_turn") or 0),
                            index, entry.get("comparison_binding") or {}, entry.get("result") or {}))
    for index, record in enumerate(state.comparison_records):
        records.append((int(record.get("created_turn") or 0), len(state.episode_ledger) + index,
                        {**record, "best_candidate": record.get("best_partial"),
                         "best_candidate_hash": record.get("best_partial_hash")}, {}))
    for _, _, binding, result in sorted(records, key=lambda row: (row[0], row[1]), reverse=True):
        legacy = "best_candidate" not in binding
        best = binding.get("winner") if legacy else binding.get("best_candidate")
        hashes = binding.get("branch_hashes") or {}
        best_hash = binding.get("winner_hash") if legacy else binding.get("best_candidate_hash")
        if not isinstance(best, str) or best not in hashes or not active(state, best):
            continue
        if any(not state.workspace.has_branch(name) or content_hash(state, name) != h
               for name, h in hashes.items()):
            continue
        if best_hash != hashes[best]:
            continue
        verdicts = result.get("verdicts") or []
        verdict = next((str(v.get("verdict") or "").lower() for v in verdicts if v.get("branch") == best), "")
        if "reject" in verdict or verdict in {"invalid", "not viable"}:
            continue
        if not legacy and result:
            ranking = result.get("ranking") or []
            if (len(ranking) != len(hashes) or set(ranking) != set(hashes)
                    or {v.get("branch") for v in verdicts} != set(hashes)):
                continue
        return best, {**binding, "best_candidate": best, "best_candidate_hash": best_hash,
                      "accepts_as_solution": False if legacy else binding.get("accepts_as_solution") is True,
                      "legacy_winner_read": legacy}
    return None


def _number(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else -math.inf
    except (ValueError, TypeError):
        return -math.inf


def candidate_portfolio(state, executor) -> list[dict[str, Any]]:
    """Pure view: six distinct hashes with explicit roles, never a solve gate.

    Existing scalar ordering selects within each family. Baseline, current
    preference, positive attestation and newest repair get reserved attention;
    remaining slots preserve family/refinement diversity before redundant rows.
    No branch is deleted when it leaves this attention portfolio.
    """
    ws = state.workspace
    rows = {}
    for name in ws.branch_names():
        if not active(state, name):
            continue
        branch = ws.get_branch(name)
        text = _decoded_text_for_panel(ws, name)
        if not any(ch.isalpha() for ch in text):
            continue
        h = _candidate_content_hash(text)
        meta = branch.metadata
        family = str(meta.get("cipher_mode") or "substitution")
        if meta.get("null_mask_selected") or meta.get("null_mask_finalist"):
            family += ":null_mask"
        elif branch.transform_pipeline:
            family += ":transform"
        att = latest_attestation_for_hash(state.verify_attestations, h)
        dr, quad = _score_branch_for_panel(ws, name, state.language, executor.word_set, executor._freq_rank)
        rows[name] = {"branch": name, "content_hash": h, "family": family,
                      "source": meta.get("decoded_text_source") or (meta.get("search_metadata") or {}).get("solver") or "workspace",
                      "created_turn": branch.created_iteration, "scores": {"dict_rate": dr, "quad": quad},
                      "verification": "positive" if attestation_is_positive(att) else "negative" if att else "missing",
                      "verification_priority": att is None,
                      "rank": (_number(dr), _number(quad), len(branch.key)),
                      "attestation": att}
    ranked = sorted(rows, key=lambda n: (*[-v for v in rows[n]["rank"]], n))
    chosen: dict[str, dict] = {}

    def add(name, role):
        if name not in rows:
            return
        source = rows[name]
        h = source["content_hash"]
        if h not in chosen:
            if len(chosen) >= PORTFOLIO_LIMIT:
                return
            chosen[h] = {k: v for k, v in source.items() if k not in {"rank", "attestation"}}
            chosen[h].update(roles=[], aliases=sorted(n for n in rows if rows[n]["content_hash"] == h))
        if role not in chosen[h]["roles"]:
            chosen[h]["roles"].append(role)

    baseline = None
    for name in ranked:
        if name == "automated_preflight" or "automated_preflight" in ws.get_branch(name).tags:
            baseline = name
            break
    if baseline is None:
        # CLI/MCP have no implicit preflight. Preserve the first explicitly
        # installed automated result instead; older records use installed_as.
        baseline = next((record.get("baseline_installed_as") or record.get("installed_as")
                         for record in state.experiment_queue
                         if record.get("type") == "automated_solver"
                         and record.get("status") == "completed"
                         and (record.get("baseline_installed_as") or record.get("installed_as")) in rows), None)
    add(baseline, "automated_baseline")
    best = fresh_compare_best_candidate(state)
    if best:
        add(best[0], "compare_best_partial")
    positives = [n for n in ranked if rows[n]["verification"] == "positive"]
    if positives:
        add(max(positives, key=lambda n: (_number(rows[n]["attestation"].get("semantic_recoverability")),
                                         _number(rows[n]["attestation"].get("target_language_confidence")),
                                         int(rows[n]["attestation"].get("created_turn") or 0), n)), "positive_verification")
    if ranked:
        add(ranked[0], "scalar_best")
    for tx in reversed(state.repair_transactions):
        if tx.get("status") == "installed" and tx.get("installed_branch") in rows:
            add(tx["installed_branch"], "newest_repair")
            break
    seen_families = set()
    for name in ranked:
        family = rows[name]["family"]
        if family not in seen_families:
            add(name, "family_finalist")
            seen_families.add(family)
    for name in ranked:
        if rows[name]["content_hash"] not in chosen:
            add(name, "additional_finalist")
    return list(chosen.values())


def refresh_portfolio(state, executor) -> list[dict]:
    """Persist attention roles; protect retained branch deletion until reranked.

    Keys remain editable: an intentional edit changes candidate identity and
    invalidates old evidence. Repair/search should fork when preserving both
    versions matters. A rejected branch loses protection on the next refresh.
    """
    state.candidate_portfolio = candidate_portfolio(state, executor)
    state.workspace.protected_branches = {row["branch"] for row in state.candidate_portfolio}
    return state.candidate_portfolio


def render_portfolio(state, executor) -> str:
    rows = candidate_portfolio(state, executor)
    if not rows:
        return ""
    lines = ["## Retained candidate portfolio (attention, not solved acceptance)"]
    for row in rows:
        lines.append(f"- {row['branch']} [{row['content_hash'][:12]}]: "
                     f"{', '.join(row['roles'])}; family={row['family']}; "
                     f"verification={row['verification']}; aliases={row['aliases']}" +
                     ("; fresh verification needed (explicit authorization only)" if row["verification_priority"] else ""))
    return "\n".join(lines)
