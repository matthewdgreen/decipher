# Spec: OpenAI reasoning-item passback (v2 adapter, stateless)

Parent: fair 5.5-vs-5.6 comparison; first slice of the v3 native-session
direction (`docs/specs/agent_v3_design.md` C7 — this builds the
encrypted-content mechanics `OpenAISession` will reuse). Spec author:
Fable. Implementer: coding sub-agent. Entry gate: matrix + variance
re-run complete (orchestrator launches; do not start before being told
the tree is free).

## Problem

The Responses-API path (`_send_responses`, added in `94ef38e`) drops
`reasoning` items from `response.output`. Subsequent turns re-send
`function_call` items without their paired reasoning items. OpenAI
recommends passing reasoning items back between tool calls for
reasoning models; omission plausibly degrades gpt-5.6 tiers
(gpt-5.6-sol Borg: 91.0%/65.4% vs gpt-5.5's 95.9%/84.8%).

## Required behavior

1. Request: on the responses path, set `store=False` and
   `include=["reasoning.encrypted_content"]` so reasoning items return
   with re-sendable encrypted payloads. Verify the exact include-string
   and `store` interaction against the installed openai 2.32.0 SDK/types
   first; report if the SDK spells it differently.
2. Capture: `_openai_responses_response_to_model_response` additionally
   preserves `reasoning`-type output items (verbatim dicts, including
   `encrypted_content`) as an opaque block on the normalized response:
   `{"type": "provider_extra", "provider": "openai",
     "kind": "reasoning", "items": [...]}` — appended to
   `ModelResponse.content` alongside existing Text/ToolUse blocks.
3. Loop transparency: verify (and add tests proving) that
   `loop_v2`'s response handling ignores unknown block types safely when
   extracting text/tool_use, that the blocks land in `messages` history
   with the assistant turn, and that `_compress_history` does not stub
   them (it targets panels and old tool_results).
4. Re-emit: `_messages_to_openai_responses` converts `provider_extra`
   reasoning blocks back to their native items, preserving their
   position relative to sibling `function_call` items of the same
   assistant turn. Other providers' converters (chat, Anthropic path)
   silently drop `provider_extra` blocks — with a test per converter.
5. Artifacts: the blocks serialize with `artifact.messages` as-is (they
   are plain dicts). `scripts/inspect_artifact.py` must not crash on
   them (spot-check; add a tiny guard if needed).
6. Off-switch: env `DECIPHER_OPENAI_REASONING_PASSBACK=0` disables
   capture/re-emit (for A/B measurement); default on.

## Tests

- Fake responses-path round-trip: response with reasoning + function_call
  items → history → next request input contains the reasoning items
  verbatim, correctly positioned; with the env off, they are absent.
- Chat-path and mixed-provider histories: provider_extra blocks dropped
  without error.
- `_compress_history` leaves provider_extra blocks intact while stubbing
  old tool_results.
- Live acceptance (single cheap run): `decipher testgen --preset hardest
  --seed 6 --agentic --model gpt-5.6-sol --max-iterations 2` succeeds
  with store=False + include set (no 400).

## Acceptance

Full suite green (baseline 823 passed / 1 skipped); live acceptance run
clean. Report files changed, suite counts, SDK verification notes.
No commits.
