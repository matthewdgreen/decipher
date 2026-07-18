# Codex run brief — borg_single_B_borg_0045v

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
5. When finished, report: the investigation id, your terminal verdict, and
   a one-paragraph account of your route.

## Ciphertext

```
S025 S012 S006 S016 S003 S005 | S003 S007 S012 S019 | S005 S009 S010 S009 | S015 S006 S008
S019 S006 S019 S006 S009 S007 | S005 S012 S004 S008 S009 S019 | S006 S040 S011
S008 S003 S005 | S003 S015 S009 S019 S008 S018 S004 S018 S013 S050 S009 S006 | S011 S012 S004 S012
S015 S006 S005 | S006 S040 S010 S009 S017 S009 S008 S012 S005 S050 | S002 S012 S005 S012
S007 S018 | S012 S006 S004 | S003 S004 S009 S018 | S011 S018 S008 S012 | S003 S016 S011 S018
S016 S012 S019 | S011 S015 S018 | S012 S009 S002 S006 | S019 S008 S005
S002 S018 S007 S019 S008 S009 S008 S009 S019 S006 | S009 S004 S004 S012 S005 S022 S004
S040 S012 S005
S016 S018 S004 S018 S004 S009 S019 | S019 S006
S016 S003 S008 S009 S012 S012 S005 S009 S007 S002 S003 S004
S002 S012 S004 S018
S001 S020 | S016 S015 S003 S013 S003 S013
S013 S012 S005 S005 S009 | S003 S015 S003 S017 S009 S002 S009
S019 S011 S006 S015 S016 S009 S003 | S016 S015 S003 S005 S050 S003 S007 S008 S009 | S050
S013 S004 S009 S002 S009 S015 S009 S019 S006 | S003 S004 S047 S047 S050 S006 S007 S004
S018 S002 S012 S004 S018 S015 S012 S005 | S002 S003 S007 S002 S004 S002 S015 S012
S013 S012 S005 S005 S009 | S010 S006 S016 S006 S015 S006
S067 S003 S002 S002 S010 S003 S004 S009 | S043 S004 S017 S009
S006 S008 | S011 S012 S004 S012 S009 S019 | S019 S012 S017 S008 S009 S004 S050 S019 S019
S041 S042 S043 S044 S042 S045 S046 S047 S001 S048 | S005 S012 S019 S049
```
