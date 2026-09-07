# R1 — Investigation CLI acceptance

Date: 2026-09-06. Base revision: `c4554c5`; changes are in the working tree.
Scope: I-6 parity/onboarding and I-7 whole-interface review/README audit.
No paid provider calls, new solving algorithms, or declaration-gate weakening.
Final combined check: **227 passed in 27.50 seconds**, including the native
acceptance after both capstone fixes. R1 is complete; R2 is next.

## Evidence

- All 23 manifest operations preserve canonical nested JSON through inline,
  file, and stdin inputs. A dummy manifest entry reaches both projections
  without per-surface registration.
- Real CLI/MCP adapters continue one investigation across writer handoffs;
  a simultaneous CLI writer receives `writer_lease_held`. Existing experiment
  tests cover both collection directions and wait/detach interruption/recovery.
- The opt-in round-6 acceptance executes start → diagnose → real Rust Quagmire
  search → collect/install → decode → fresh invocation reload → keyless verify
  refusal → blocked declaration → authorized **local fake reader** → declaration.
  The fixed result matches all 566 reference characters in a post-hoc hash
  assertion. No answer/key enters any runtime operation. The native test passed
  in 7.29 seconds before capstone fixes and is rerun in the final check below.
- README was read end-to-end and corrected for the structured CLI, fresh-code
  behavior, keyed-column/composite support, incomplete blind periodic routing,
  dedicated Quagmire experiments, kernel/keyless distinctions, and historical
  rather than current model-performance claims. Onboarding has a CLI operator
  recipe covering revisions, collection, privacy, and recovery.

## Independent review and fixes

The user explicitly authorized a separate Astra reviewer in place of the
specification's named Fable reviewer. The reviewer was configured as
`gpt-6-astra`; trustworthy actual served-model metadata was unavailable.
Review scope included the entire landed CLI/service/manifest and associated
registry, experiment, verification, repair, tests, and documentation, not just
the new diff.

Two P2 findings were reproduced and fixed:

1. **Argparse errors escaped the JSON contract.** The actual entry point now
   uses `InvestigationArgumentParser` for the investigation namespace. Missing
   verbs, unknown flags, invalid values, and conflicting options return exactly
   one JSON error with exit 2, before touching registry/provider state. Other
   commands retain their existing parser behavior; explicit `--help` is the
   documented human-readable exception.
2. **Terminal state could change between initial read and lease acquisition.**
   The shared lease helper now rechecks current terminal state under the lease,
   before revision checks, turn updates, or domain execution. Deterministic
   tests cover solved/unsolved across CLI, MCP, wait, and detach notifications.

The reviewer rechecked both fixes: **89 interface tests passed in 2.01 seconds**,
both findings resolved, no new actionable findings in the fixes. No additional
concrete provider-consent or hash-attestation bypass was found in the review.

## Reproduction and limitations

```bash
DECIPHER_RUN_NATIVE_CLI_ACCEPTANCE=1 PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_reliability_packet.py tests/test_interface_parity.py \
  tests/test_investigation_cli.py tests/test_investigation_cli_native_acceptance.py \
  tests/test_v3_candidate_selection_report.py tests/test_mcp_onboarding.py \
  tests/test_mcp_tools.py tests/test_mcp_protocol.py tests/test_mcp_registry.py \
  tests/test_ground_truth_firewall.py -q
```

The real-solver workflow invokes CLI adapters in-process; executable subprocess
tests separately cover parser failures, help, detach, and signal handling.
Sleeping-stub lifecycle tests do **not** establish native process-pool child
cleanup under all signals. The fresh-clone test uses repository code with the
existing venv's third-party dependencies, not a clean dependency installation.
Fake verification proves gate composition, not independent language quality.
The full repository suite was not rerun for this slice; the earlier 1,933-test
baseline belongs to `c4554c5`, not this changed tree.

Follow-up (2026-09-07): the [fresh Astra solving handoff](../codex_astra_r1_handoff.md)
completed; see the [observation report](codex_astra_r1_observation.md).
It uses one development case, not a held-out case, and is a
qualitative CLI observation, not a matched model comparison or R4 routing run.
It recovered 501/501 letters post-hoc and closed honestly keyless/unsolved.
No Astra-versus-Sol solving improvement is established. Its terminal-status
and candidate-display findings are addressed by R2, not retroactively claimed
as covered by the original 227-test R1 check above.

R0 interpretation reminder: its replay selection gap compares the **historically
selected branch rendered by current code** with the best saved branch; it is
not a fresh run of today's selection policy. A third-place scalar ranking is
a ranking disagreement, not proof that the final delivered candidate was lost.
Missing generated menus and historical provenance remain unknown.
