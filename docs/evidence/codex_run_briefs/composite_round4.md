# Codex run brief — composite_round4

Run in a FRESH Codex session opened in the decipher repository. This tests the
NEW composite substitution+transposition capability through the MCP surface.

## Freshness check (step 0 — this test needs current code)
Confirm the composite capability is live before starting:
- `git pull --ff-only` in this clone, then `sh scripts/bootstrap.sh` (short-circuits if nothing changed), then restart the Codex app.
- Verify: `investigation_list` `server_code.git_head` should match `git rev-parse --short HEAD` (>= 967bdbe). Also confirm `experiment_submit` advertises experiment type `composite_substitution_transposition`. If it does not, the server is stale — restart the app and re-check.

## Task
Crack the cipher below using ONLY the decipher MCP tools (no direct repo
scripts/solvers/Rust — this run measures the tool surface; record honestly if
the surface can't do something).

## Context you are permitted
- English plaintext (`en`). Single continuous letter stream, NO word boundaries.
- Nothing else — the cipher FAMILY is for you to diagnose.

## Instructions
1. `investigation_start` with the ciphertext below inline (language `en`).
2. Follow `docs/mcp_onboarding.md` §Investigation methodology, driving from `investigation_status`. Expect `observe_diagnosis` to help identify the family.
3. Honor WF-7: before stopping, `request_independent_verification` on your leading branch, then close with `meta_declare_solution` or `meta_declare_unsolved`.
4. Show the text: your final chat message must include the investigation id + verdict, the FULL `decode_show` text of the leading branch, its `candidate_list` signals, and the final attestation scalars, then a one-paragraph route account.
5. Do not read files under `docs/evidence/` or `artifacts/` during the run.

## Ciphertext
```
WCMLLWZFTTTTVRLMRLLTHTFWERCHHHKGTGTRIFTGHSLQLECFZWMWRTKIILITYTKGOKKJCJIMFFWKWMZYKMMSHJGXOFTFHKYHJLLHMMMCMLRAGVNVLAMAJMWKWLXWMFTWFWVKZYGFYWHLWGXELHWKGWYJTRFZAWCGRHTXLSVHISWLVHKOGSFCGWQRKWSRJCGCMMMRHMCLEBIQXTWZWCLRHNILFRLCKWCKVWVFGCWZFMMTGWFVLRHIWRCWFQLRTLRJWKIWGMWYFFHZFGCKLMTVCLSNFCLCRWX
```
