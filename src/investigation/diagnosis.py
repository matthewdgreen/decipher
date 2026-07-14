"""Ranked cipher-family diagnosis (INV-0 Part 6).

``diagnose(tokens, ...)`` embeds the real ``cipher_id`` fingerprint, runs the
nine-panel battery, scores every family with the executable weight formula, ranks
primaries (subtypes fold under their parent, modifiers are separate), and issues a
``confident``/``uncertain`` verdict with a recommended next discriminator.

INPUT FIREWALL: the signature accepts ciphertext-derived inputs only — no record
objects, no context text, no ground truth. Every number here is reproduced by
``scripts/research/calibrate_inv0_scoring.py`` FIXED mode.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from analysis.cipher_id import compute_cipher_fingerprint
from analysis.null_baseline import encode_tokens
from analysis.panels import PANEL_NAMES, run_battery
from investigation.families import (
    DISCRIMINATOR_REGISTRY,
    FAMILY_REGISTRY,
    MODIFIER_IDS,
    PRIMARY_IDS,
    SUBSTITUTION_PRIMARIES,
    SUBTYPE_IDS,
    SUSPICION_TO_FAMILY,
    family_discriminators,
)

# Verdict thresholds
_TOKEN_GATE = 60
_MARGIN_GATE = 0.15
_STRONG_SCORE = 0.70
_STRONG_MARGIN = 0.25
_MODERATE_SCORE = 0.45

_ALL_FAMILIES = tuple(PRIMARY_IDS) + tuple(SUBTYPE_IDS) + tuple(MODIFIER_IDS)


# ---------------------------------------------------------------------------
# View hash (D5 dedup key; INV-0 calls it with empty extras)
# ---------------------------------------------------------------------------

def view_hash(
    tokens,
    *,
    transform_pipeline: tuple = (),
    null_mask: tuple = (),
    segmentation: tuple = (),
) -> str:
    payload = encode_tokens(list(tokens)) + b"\x1f" + json.dumps(
        [list(transform_pipeline), list(null_mask), list(segmentation)],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FamilyDiagnosis:
    family: str
    score: float
    confidence: str                 # strong | moderate | weak | none
    evidence: list[str]
    counterevidence: list[str]      # mandatory; format renders "(none recorded)" when empty
    subtypes: list["FamilyDiagnosis"] = field(default_factory=list)
    pending_discriminators: list[str] = field(default_factory=list)
    solver_status: str = ""
    coverage_status: str = "untested"   # ledger is INV-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "score": round(self.score, 4),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counterevidence": self.counterevidence,
            "subtypes": [s.to_dict() for s in self.subtypes],
            "pending_discriminators": self.pending_discriminators,
            "solver_status": self.solver_status,
            "coverage_status": self.coverage_status,
        }


@dataclass
class DiagnosisReport:
    view_hash: str
    fingerprint: dict[str, Any]
    atoms: list[dict[str, Any]]
    ranked: list[FamilyDiagnosis]
    modifiers: list[FamilyDiagnosis]
    verdict: str                    # confident | uncertain
    battery_coverage: dict[str, str]
    recommended_next: list[dict[str, Any]]
    token_count: int
    language: str
    alphabet_class: str
    verdict_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_hash": self.view_hash,
            "fingerprint": self.fingerprint,
            "atoms": self.atoms,
            "ranked": [f.to_dict() for f in self.ranked],
            "modifiers": [m.to_dict() for m in self.modifiers],
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
            "battery_coverage": self.battery_coverage,
            "recommended_next": self.recommended_next,
            "token_count": self.token_count,
            "language": self.language,
            "alphabet_class": self.alphabet_class,
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rel_mult(reliability: str) -> float:
    return 1.0 if reliability == "high" else 0.4


def _score_families(atoms: list[dict[str, Any]]):
    raw = {f: 0.0 for f in _ALL_FAMILIES}
    ev = {f: [] for f in _ALL_FAMILIES}
    counter = {f: [] for f in _ALL_FAMILIES}
    high_support = {f: 0 for f in _ALL_FAMILIES}
    for a in atoms:
        w = a["weight"] * _rel_mult(a["reliability"])
        for f in a["supports"]:
            if f in raw:
                raw[f] += w
                ev[f].append(a["observation"])
                if a["reliability"] == "high":
                    high_support[f] += 1
        for f in a["weakens"]:
            if f in raw:
                raw[f] -= w
                counter[f].append(a["observation"])
    # Round to 6 decimals: clears float-summation noise (e.g. 0.35+0.30+0.20-0.35
    # yielding 0.4999999999999999) so the strict margin gate is stable at exact
    # boundaries. Never crosses a 0.01-scale threshold, so no fixture changes.
    score = {f: round(max(0.0, min(1.0, raw[f])), 6) for f in _ALL_FAMILIES}
    return score, ev, counter, high_support


def _confusable_primaries(family_id: str) -> set[str]:
    return set(FAMILY_REGISTRY[family_id].confusable_with) & set(PRIMARY_IDS)


def _confidence(family_id: str, score: dict[str, float], high_support: dict[str, int]):
    s = score[family_id]
    conf_set = _confusable_primaries(family_id)
    margin = s - max((score[c] for c in conf_set), default=0.0)
    if s >= _STRONG_SCORE and high_support[family_id] >= 2 and margin >= _STRONG_MARGIN:
        return "strong", margin
    if s >= _MODERATE_SCORE and high_support[family_id] >= 1:
        return "moderate", margin
    if s > 0:
        return "weak", margin
    return "none", margin


def _discriminator_status(disc_id: str, panel_status: dict[str, str]) -> str:
    disc = DISCRIMINATOR_REGISTRY[disc_id]
    if disc.status == "planned":
        return "unavailable"
    if all(panel_status.get(p) == "ok" for p in disc.depends_on_panels):
        return "run"
    return "pending"


def _pending_discriminators(family_id: str, panel_status: dict[str, str]) -> list[str]:
    return [
        d for d in family_discriminators(family_id)
        if _discriminator_status(d, panel_status) == "pending"
    ]


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def diagnose(
    tokens,
    *,
    alphabet_size: int,
    alphabet_class: str,
    language: str = "en",
    word_group_count: int = 0,
    line_lengths=None,
    numeric_values=None,
    letter_rendering: str | None = None,
    related_profile=None,
    max_period: int = 26,
    rng_namespace: str = "",
) -> DiagnosisReport:
    tokens = list(tokens)
    n = len(tokens)

    fp = compute_cipher_fingerprint(
        tokens, alphabet_size, max_period=max_period, language=language,
        word_group_count=word_group_count,
    )

    # ---- fingerprint priors (weight 0.5*suspicion, reliability low) ----
    atoms: list[dict[str, Any]] = []
    for key, susp in fp.suspicion_scores.items():
        fam = SUSPICION_TO_FAMILY.get(key)
        if fam is None or susp <= 0:
            continue
        atoms.append({
            "observation": f"fingerprint_prior:{fam}",
            "panel": "fingerprint",
            "weight": 0.5 * susp,
            "measurement": {"suspicion": susp},
            "baseline": None,
            "reliability": "low",
            "interpretation": f"cipher_id suspicion prior for {fam}.",
            "supports": [fam],
            "weakens": [],
        })

    # ---- panel battery ----
    battery = run_battery(
        tokens, alphabet_size=alphabet_size, alphabet_class=alphabet_class,
        language=language, word_group_count=word_group_count, line_lengths=line_lengths,
        numeric_values=numeric_values, letter_rendering=letter_rendering,
        related_profile=related_profile, max_period=max_period, rng_namespace=rng_namespace,
    )
    for name in PANEL_NAMES:
        atoms.extend(battery[name].atoms)

    panel_status = {name: battery[name].status for name in PANEL_NAMES}

    # ---- score ----
    score, ev, counter, high_support = _score_families(atoms)

    ranked_primaries = sorted(PRIMARY_IDS, key=lambda f: (-score[f], f))
    top1 = ranked_primaries[0]
    top2 = ranked_primaries[1] if len(ranked_primaries) > 1 else None

    # ---- verdict ----
    reasons: list[str] = []
    uncertain = False
    if n < _TOKEN_GATE:
        uncertain = True
        reasons.append("a:token_count<60")
    if top2 is not None and (score[top1] - score[top2]) < _MARGIN_GATE:
        uncertain = True
        reasons.append("b:margin<0.15")
    top1_discs = family_discriminators(top1)
    if top1_discs and all(
        _discriminator_status(d, panel_status) == "pending" for d in top1_discs
    ):
        uncertain = True
        reasons.append("c:all_discriminators_pending")
    if top2 is not None and top2 in _confusable_primaries(top1):
        covering = [
            d for d, spec in DISCRIMINATOR_REGISTRY.items()
            if frozenset(spec.splits) == frozenset({top1, top2})
        ]
        if covering and all(
            _discriminator_status(d, panel_status) != "run" for d in covering
        ):
            uncertain = True
            reasons.append("d:confusable_discriminator_not_run")
    verdict = "uncertain" if uncertain else "confident"

    # ---- recommended_next ----
    recommended_next: list[dict[str, Any]] = []
    if uncertain and top2 is not None:
        chosen = _recommend_discriminator(top1, top2, panel_status)
        if chosen is not None:
            ordered = [chosen] + [
                d for d in _related_discriminators(top1, top2)
                if d != chosen
            ]
            recommended_next = [_discriminator_dict(d) for d in ordered]

    # ---- build FamilyDiagnosis objects ----
    def _make(fam: str) -> FamilyDiagnosis:
        conf, _margin = _confidence(fam, score, high_support)
        return FamilyDiagnosis(
            family=fam,
            score=score[fam],
            confidence=conf,
            evidence=list(ev[fam]),
            counterevidence=list(counter[fam]),
            pending_discriminators=_pending_discriminators(fam, panel_status),
            solver_status=FAMILY_REGISTRY[fam].solver_status,
        )

    subtype_by_parent: dict[str, list[FamilyDiagnosis]] = {}
    for sub in SUBTYPE_IDS:
        parent = FAMILY_REGISTRY[sub].parent
        subtype_by_parent.setdefault(parent, []).append(_make(sub))
    for parent in subtype_by_parent:
        subtype_by_parent[parent].sort(key=lambda d: (-d.score, d.family))

    ranked: list[FamilyDiagnosis] = []
    for fam in ranked_primaries:
        fd = _make(fam)
        fd.subtypes = subtype_by_parent.get(fam, [])
        ranked.append(fd)

    modifiers = [_make(m) for m in MODIFIER_IDS]

    battery_coverage = {
        name: ("ran" if battery[name].status == "ok"
               else f"skipped({battery[name].reason})")
        for name in PANEL_NAMES
    }

    return DiagnosisReport(
        view_hash=view_hash(tokens),
        fingerprint=fp.to_dict(),
        atoms=atoms,
        ranked=ranked,
        modifiers=modifiers,
        verdict=verdict,
        battery_coverage=battery_coverage,
        recommended_next=recommended_next,
        token_count=n,
        language=language,
        alphabet_class=alphabet_class,
        verdict_reasons=reasons,
    )


def _recommend_discriminator(top1: str, top2: str, panel_status: dict[str, str]) -> str | None:
    """Choose the discriminator to run next for an uncertain top-2 (finding 5).

    Priority (post-third-review amendment #2): a covering discriminator whose
    splits == {top1, top2}; else the finding-5 fallback in the EXACT order the
    calibration uses — top2-runnable -> top2-any -> top1-runnable -> top1-any.
    """
    covering = [
        d for d, spec in DISCRIMINATOR_REGISTRY.items()
        if frozenset(spec.splits) == frozenset({top1, top2})
    ]
    if covering:
        return covering[0]

    def _runnable(d: str) -> bool:
        return _discriminator_status(d, panel_status) == "run"

    for fam in (top2, top1):
        cands = family_discriminators(fam)
        if not cands:
            continue
        runnable = [d for d in cands if _runnable(d)]
        pool = runnable or cands
        if pool:
            return pool[0]
    return None


def _related_discriminators(top1: str, top2: str) -> list[str]:
    """Discriminators naming top1 or top2, in registry order (chosen leads)."""
    out: list[str] = []
    for d, spec in DISCRIMINATOR_REGISTRY.items():
        if top1 in spec.splits or top2 in spec.splits:
            out.append(d)
    return out


def _discriminator_dict(disc_id: str) -> dict[str, Any]:
    spec = DISCRIMINATOR_REGISTRY[disc_id]
    return {
        "discriminator_id": spec.id,
        "description": spec.description,
        "tool": spec.tool,
        "splits": list(spec.splits),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_diagnosis(report: DiagnosisReport) -> str:
    lines: list[str] = []
    lines.append(
        f"Diagnosis ({report.token_count} tokens, class={report.alphabet_class}, "
        f"lang={report.language}) -> {report.verdict.upper()}"
    )
    if report.verdict_reasons:
        lines.append(f"  reasons: {', '.join(report.verdict_reasons)}")
    lines.append("")
    lines.append("Ranked families (evidence weight, not probability):")
    for fd in report.ranked:
        if fd.score <= 0 and fd.confidence == "none" and not fd.counterevidence:
            continue
        ev = ", ".join(fd.evidence) if fd.evidence else "(none)"
        counter = ", ".join(fd.counterevidence) if fd.counterevidence else "(none recorded)"
        lines.append(
            f"  {fd.family:30s} {fd.score:.2f} [{fd.confidence}] "
            f"solver={fd.solver_status}"
        )
        lines.append(f"      evidence: {ev}")
        lines.append(f"      counterevidence: {counter}")
        for sub in fd.subtypes:
            if sub.score > 0:
                sev = ", ".join(sub.evidence) if sub.evidence else "(none)"
                lines.append(f"      subtype {sub.family:22s} {sub.score:.2f} [{sub.confidence}] ({sev})")
    if report.modifiers:
        mods = ", ".join(f"{m.family}={m.score:.2f}" for m in report.modifiers)
        lines.append(f"  modifiers: {mods}")
    lines.append("")
    ran = [k for k, v in report.battery_coverage.items() if v == "ran"]
    skipped = [f"{k}:{v}" for k, v in report.battery_coverage.items() if v != "ran"]
    lines.append(f"Battery: ran={', '.join(ran)}")
    if skipped:
        lines.append(f"         skipped={', '.join(skipped)}")
    if report.recommended_next:
        rn = report.recommended_next[0]
        lines.append(
            f"Recommended next: {rn['discriminator_id']} "
            f"(splits {rn['splits'][0]} vs {rn['splits'][1]})"
        )
    text = "\n".join(lines)
    return text[:2500]
