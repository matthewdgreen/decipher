# I-5 implementation sub-spec — explicit verification authority

**Parent:** `docs/specs/investigation_cli_spec.md` §§3.1, 7.1, and 8.
Builds on the combined I-3/I-4 lifecycle (`b9353ae`).

## 0. Scope

Land the final domain operations on the structured CLI:

- `verify` / `request_independent_verification`;
- verifier-arbitrated `repair-transaction`; and
- solution and unsolved declarations through the unchanged host gates.

This slice changes transport authority, not epistemic policy. MCP keeps its
session-start provider resolution and all declaration/repair checks remain
byte-compatible.

## 1. CLI authority contract

The `investigation` namespace gains global options, before the verb:

```text
--verify-provider {anthropic,openai,gemini,openrouter,ollama,none}
--verify-model MODEL
--max-cost-usd N
--allow-external
```

An ambient API key never selects a provider. Resolution order is binding:

1. no explicit provider, or `none` -> `no_verification_provider`;
2. provider selected without `--allow-external` ->
   `external_call_not_authorized`;
3. provider selected and authorized but its required credential is absent ->
   `no_verification_provider`; and
4. only then may provider construction or an external send occur.

`no_verification_provider` is an `unavailable` domain outcome (exit 1).
`external_call_not_authorized` is a blocked policy outcome (exit 3). Neither
touches investigation state when verification is the requested operation.

## 2. Last-responsible-moment rule

`verify` is unconditionally external, so the CLI enforces the authority
contract before constructing `InvestigationService`, acquiring a lease, or
dispatching the domain operation.

`repair-transaction --verifier-arbitration` is conditionally external. It gets
a deferred provider whose resolver is invoked only if the mechanical repair
checks reach the arbitration send. Therefore:

- a mechanically accepted repair remains local and succeeds without provider
  selection or `--allow-external`;
- a repair requiring arbitration returns the typed authority refusal at the
  exact send boundary;
- a refused arbitration is not silently converted into an ordinary mechanical
  reject; and
- the failed invocation commits no partial transaction or turn bump.

The deferred-provider refusal is re-raised through the episode and host crash
guards. All ordinary provider/API failures retain their existing structured
episode-failure behavior.

## 3. Acceptance tests

All tests are local and use a scripted verifier; no API call is permitted.

1. Ambient provider keys without an explicit provider still refuse keylessly.
2. Explicit provider without `--allow-external` refuses before provider
   construction and leaves the revision unchanged.
3. Explicit authorized fake verification writes a fresh positive attestation;
   `declare-solution` then succeeds through DECL-1.
4. Keyless declaration remains blocked by `attestation_required`.
5. A mechanically accepted repair with arbitration requested never resolves a
   provider and installs normally.
6. A repair that actually needs arbitration returns, respectively,
   `no_verification_provider` or `external_call_not_authorized`, with no commit.
7. An explicitly authorized scripted arbitration retains the existing install
   semantics.
8. Friendly `verify` and canonical `call request_independent_verification`
   enforce the same authority policy.

Landing bar: focused CLI, repair-arbitration, MCP, registry, and state suites
green; then the full local suite. No paid verification run belongs to I-5.
