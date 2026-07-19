# Codex run brief — borg_single_B_borg_0109v

Run this in a FRESH Codex session opened in the decipher repository.

## Task

Crack the cipher below using the decipher MCP server.

## Context you are permitted

- The transcription is from a 17th-century Latin pharmaceutical manuscript (Borg cipher class).
- Plaintext language: Latin (`la`).
- Format: canonical S-token transcription; tokens are space-separated,
  ` | ` marks a word break.

## Instructions

1. If the `decipher` MCP server is not connected, run `sh scripts/bootstrap.sh`,
   reconnect, and retry.
2. Call `investigation_start` with the ciphertext below inline
   (language `la`), then follow `docs/mcp_onboarding.md`
   §Investigation methodology, driving from `investigation_status`.
3. Honor WF-7: do NOT end the session holding an unverified candidate.
   Before stopping — even short of a solve — run
   `request_independent_verification` on your leading branch, then close
   with `meta_declare_solution` or `meta_declare_unsolved`.
4. Do not read files under `docs/evidence/` or `artifacts/` during the run
   (they contain analyses of related experiments).
5. When finished, your final chat message MUST include, in this order:
   - the investigation id and terminal verdict;
   - the FULL current decode of your leading branch (use `decode_show` and
     paste its decoded text — even if partial or damaged; the reader should
     never have to open the registry to see the text);
   - that branch's scalar signals (from `candidate_list`) and the final
     attestation scalars (accepts / language confidence / coherence /
     damage scope);
   - a one-paragraph account of your route.

## Ciphertext

```
S006 S008 S009 S003 S005 | S025 S012 S018 S016 | S009 S007 | S008 S003 S004 S009 | S002 S012 S015
S011 S004 S009 S002 S003 S015 S006 | S012 | S012 S006 S004 | S003 S020 | S011 S012 S004 S004 S018
S006 S008 | S017 S015 S006 S012 S009 S008 S006 S015 | S012 S008 | S015 S004 S012 S015 S006 S019
S005 S003 S007 S006 S015 S006 S007 S008 | S012 S009 S012 S009 | S006 S008 | S010 S009 S006
S011 S015 S018 | S002 S006 S015 S008 S018 | S006 S013 S006 S015 | S004 S009 S017 S006
S017 S009 S008 S012 S015 | S019 S009 | S003 S012 S008 S006 S005 | S018 S005 S007 S006 S019
S011 S012 S004 S004 S009 | S005 S018 S015 S009 S006 S017 S003 S007 S008 S012 S015 | S006 S013
S019 S009 S005 S039 S004 S009 S008 S006 S015 | S005 S018 S015 S009 S006 S017 S003 S008 S012
S006 S008 | S025 S012 S009 S003 | S019 S006 S005 S006 S004 | S003 S011 S011 S004 S009 S019
S012 S009 | S004 | S011 S012 S004 S004 S047 S019 S014 | S025 S012 S009 | S018 S005 S007 S006
S005 S018 S015 S009 S006 S017 S003 S007 S008 S012 S015 | S006 S008 | S025 S012 S009 S003 | S039
S007 S018 S002 S008 S006 | S006 S015 S003 S008 | S007 S018 S007 | S011 S018 S008 S012 S009
S011 S004 S012 S015 S006 S019 | S011 S012 S004 S004 S018 S019 | S010 S003 S017 S006 S015
S006 S008 S019 S009 | S010 S003 S017 S012 S009 S019 S019 S006 S005 S012 S019 | S004
S017 S006 S015 S003 S019 S019 S006 S005 | S006 S012 S005 | S016 S006 S009 | S013 S015
S008 S009 S003 | S019 S006 S016 | S016 S006 S022 S006 S002 S008 S012 | S011 S012 S004 S004
S015 S012 S005 | S005 S018 S015 S009 S006 S017 S003 S008 S012 S015 S014 | S006 S008 | S009 S019 S008
S002 S012 S015 S003 | S006 S019 S008 | S005 S012 S004 S008 S012 S005 | S019 S012 S003 S012 S009
S006 S008 | S019 S009 S007 S006 | S016 S018 S004 S018 S015 S006 S020 | S006 S008 | S022
S015 S009 | S011 S018 S008 S006 S019 S008 S025 S012 S009 S017 S012 S019 | S006 S012 S003
```
