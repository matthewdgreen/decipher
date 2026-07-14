"""Agent loop v3 — investigation state, context builder, sessions, lead loop.

This package implements Milestone M1 of the v3 redesign (see
``docs/specs/agent_v3_m1_spec.md`` and ``docs/specs/agent_v3_design.md``):

- ``state``    — ``InvestigationState``: the one serializable object owning
  everything durable about a run (cipher, workspace, evidence, budget,
  recent exchanges). Resume is the identity operation.
- ``context``  — ``build_lead_context``: a pure function rendering the lead's
  per-turn view from state (rebuilt every turn; the transcript is never
  load-bearing).
- ``sessions`` — the ``ModelSession`` seam: provider-native sessions that own
  their conversation format and expose a small event surface.
- ``loop_v3``  — ``run_v3``: the lead loop over the existing tool executor,
  with the declaration policy disabled (``NoGatesPolicy``).
"""
from investigation.state import (
    BudgetEntry,
    EvidenceEntry,
    InvestigationState,
)

__all__ = [
    "BudgetEntry",
    "EvidenceEntry",
    "InvestigationState",
]
