"""InvestigationRuntime: the per-investigation object graph (Part 4).

Reproduces exactly the object graph ``run_v3`` builds, minus the model session
and the turn loop. Every symbol is imported from the landed code; nothing is
re-implemented. The lead session is a null stub (the MCP server makes no lead
model calls); only server-side ``verify`` episodes may spend, through the host.
"""
from __future__ import annotations

from typing import Any

from agent.tools_v2 import AttestationPolicy, WorkspaceToolExecutor
from analysis import dictionary, pattern
from investigation.experiments import ExperimentQueue
from investigation.host import InvestigationHost
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState

# RES-5 gate surface: the ONLY v2/v3 tool names the MCP routing hands to
# ``host.handle_tool``. Any other name reaching the host is rejected
# ``lead_tool_not_available`` — defense in depth behind the server routing.
V3_MAPPED_TOOL_NAMES = {
    "observe_diagnosis", "decode_show",
    "workspace_create_hypothesis_branch", "workspace_update_hypothesis",
    "workspace_reject_hypothesis", "workspace_hypothesis_next_steps",
    "experiment_submit", "experiment_collect",
    "branch_adjudicate", "act_set_model_variant",
    "meta_declare_solution", "meta_declare_unsolved",
}


class _NullLeadSession:
    """Satisfies the ``ModelSession`` protocol with zero lead spend (4.2)."""

    capabilities = SessionCapabilities()
    model = "mcp-null"

    def send(self, blocks, tools=None, max_tokens: int = 8192):
        raise RuntimeError("the MCP server makes no lead model calls")

    def usage_entries(self):
        return []

    def export_transcript(self):
        return {"provider": "mcp", "messages": []}


class InvestigationRuntime:
    """One live investigation: state + executor + host + experiment queue."""

    def __init__(
        self,
        *,
        document: dict,
        emit,
        verify_provider: Any,
        verify_model: Any = None,
        max_cost_usd: float,
        synchronous_experiments: bool = False,
    ) -> None:
        self.meta: dict = document["meta"]
        self.records: dict = document.get("records") or {
            "comparisons": [], "repair_compiles": [],
        }
        self.records.setdefault("comparisons", [])
        self.records.setdefault("repair_compiles", [])
        self.verify_model = verify_model

        # 1. state
        state = InvestigationState.from_artifact_dict(document["state"])
        self.state = state
        self.workspace = state.workspace

        # 2. language resources (run_v3 lines 293-296)
        dict_path = dictionary.get_dictionary_path(state.language)
        word_set = dictionary.load_word_set(dict_path) if dict_path else set()
        word_list = pattern.load_word_list(dict_path) if dict_path else []
        pattern_dict = pattern.build_pattern_dictionary(word_list)
        self._word_set = word_set
        self._word_list = word_list
        self._pattern_dict = pattern_dict

        # 3. executor (byte-for-byte the run_v3 construction, benchmark_context=None,
        #    and NO set_max_iterations — MCP has no turn budget).
        executor = WorkspaceToolExecutor(
            workspace=state.workspace,
            language=state.language,
            word_set=word_set,
            word_list=word_list,
            pattern_dict=pattern_dict,
            benchmark_context=None,
            declaration_policy=AttestationPolicy(attestations=state.verify_attestations),
            repair_agenda=state.repair_agenda,
            hypothesis_board=state.hypothesis_board,
            finalist_sessions=state.finalist_sessions,
            model_variant=state.model_variant,
        )
        self.executor = executor

        # 4. experiment queue (None => env default preserves DECIPHER_EXPERIMENT_SYNC).
        self.queue = ExperimentQueue(synchronous=synchronous_experiments or None)

        # 5-6. provider + null lead session
        self.provider = verify_provider
        self.session = _NullLeadSession()

        # 7. host
        self.host = InvestigationHost(
            state=state, workspace=state.workspace, executor=executor,
            queue=self.queue, emit=emit, session=self.session,
            model_provider=self.provider, language=state.language,
            word_set=word_set, word_list=word_list, pattern_dict=pattern_dict,
            episode_models=None, max_cost_usd=max_cost_usd,
            prior_budget=list(state.budget_ledger),
        )
        # 8. RES-5 gate surface
        self.host.set_available_tools(V3_MAPPED_TOOL_NAMES)

    def persist_dict(self) -> dict:
        """Serialize the runtime after folding episode spend into the ledger."""
        self.host.sync_budget()
        return {
            "meta": self.meta,
            "state": self.state.to_artifact_dict(),
            "records": self.records,
        }

    def finalize(self, turn: int) -> None:
        """Server-shutdown finalize (4.4): mirrors run_v3 lines 890-898."""
        self.queue.poll(self.state, turn, promote=False)
        for record in self.state.experiment_queue:
            if record.get("status") in {"pending", "running"}:
                record["status"] = "orphaned"
                record["orphan_reason"] = "run_ended"
        self.queue.finalize_env_restore()
        self.host.sync_budget()
