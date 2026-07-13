# Spec: OpenAI Responses-API support for GPT-5.6 tiers

Parent: unblocks the agentic model comparison (gpt-5.5 vs gpt-5.6-sol/
terra/luna). Spec author: Fable. Implementer: coding sub-agent.
Entry gate: start only after the Phase 1 commit lands (keep this diff
separate from Phase 1's).

## Problem (observed 2026-07-13)

`OpenAIModelProvider.send` (`src/agent/model_provider.py:138–190`) calls
`client.chat.completions.create(...)` with function tools and does NOT set
`reasoning_effort`. The GPT-5.6 tiers (`gpt-5.6-sol`, `gpt-5.6-terra`,
`gpt-5.6-luna`) apply a server-side default reasoning effort and reject
the combination:

> Error code: 400 — "Function tools with reasoning_effort are not
> supported for gpt-5.6-sol in /v1/chat/completions. To use function
> tools, use /v1/responses or set reasoning_effort to 'none'."

Setting effort to `'none'` would lobotomize the tiers and invalidate the
model comparison, so the fix is a Responses-API path. `gpt-5.5` works on
chat completions today and must be left byte-identical.

## Required behavior

1. Add a Responses-API request path to `OpenAIModelProvider`:
   - `client.responses.create(model=..., instructions=<system>,
     input=<converted messages>, tools=<responses-format tools>,
     max_output_tokens=<max_tokens>)`.
   - Message conversion: user/assistant text messages → role/content
     items; assistant `tool_use` blocks → `function_call` items
     (`call_id`, `name`, `arguments` JSON string); user `tool_result`
     blocks → `function_call_output` items (`call_id`, `output` string).
     Mirror the existing chat-format converter
     (`_messages_to_openai_chat`) as `_messages_to_openai_responses`.
   - Tool conversion: chat format nests the function under
     `{"type": "function", "function": {...}}`; responses format is flat
     `{"type": "function", "name", "description", "parameters"}`. Mirror
     `_tools_to_openai_chat`.
   - Response parsing: iterate `response.output`; `function_call` items →
     `ToolUseBlock(id=call_id, name, input=json.loads(arguments))`;
     `message` items → `TextBlock` per output_text content part. Map
     `response.usage.input_tokens/output_tokens` into `ModelUsage`
     consistently with the chat path.
   - Do NOT set any `reasoning` parameter explicitly in the first slice —
     let the model default apply. (A follow-up may expose effort as
     config; out of scope here.)
2. Routing: a module-level predicate `_requires_responses_api(model)`
   returning True for model ids starting with `"gpt-5.6"`, plus an env
   override `DECIPHER_OPENAI_API=responses|chat` that forces either path
   for any model (for experiments and future tiers). Chat-completions
   remains the default for everything else — gpt-5.5 requests must be
   unchanged.
3. IMPORTANT — verify against the installed SDK: `.venv` has a pinned
   `openai` package; confirm `client.responses` exists and the exact item
   shapes (`function_call`, `function_call_output`, `output_text`) against
   that version's types before coding the parser. If the installed SDK
   predates the Responses API, report that instead of upgrading anything.
4. Retry/error semantics: wrap in the same `ModelProviderError` pattern
   as the chat path. Preserve the existing `max_completion_tokens` /
   `max_tokens` fallback behavior on the chat path untouched.

## Tests (extend the existing provider test module if one exists, else new)

- Fake client capturing kwargs: gpt-5.6-sol routes to `responses.create`
  with flat tools and converted input items; gpt-5.5 still routes to
  `chat.completions.create` with unchanged kwargs.
- Round-trip: a messages list containing system + user text + assistant
  tool_use + user tool_result converts to the documented item sequence;
  a fake response with one function_call and one message item parses to
  ToolUseBlock + TextBlock with correct usage.
- Env override forces each path.
- Live smoke (optional, cheap, behind an env guard like
  `DECIPHER_LIVE_OPENAI_SMOKE=1`): one real gpt-5.6-sol call with a
  trivial tool, asserting a tool_use or text response arrives.

## Review follow-ups (non-blocking, from the Fable review — LAND AS-IS)

- Test hardening: clear `DECIPHER_OPENAI_API` via monkeypatch in the two
  send-routing tests; add a `reasoning`-item and empty-output case to the
  parser test; add a `ModelProviderError`-wrapping test for the responses
  path.
- `tools=None` serializes as `"tools": null` (same latent pattern as the
  chat path); omit the kwarg when absent.
- Comparison-fairness notes: `strict` is omitted (server default may
  differ between chat and responses paths), and reasoning items are not
  passed back between turns (OpenAI recommends it; may slightly understate
  gpt-5.6 tiers). Revisit the reasoning-item passback as a follow-up slice
  if 5.6 results look anomalously weak.

## Acceptance

Full suite green; then one real agentic iteration on gpt-5.6-sol succeeds
(e.g. `decipher testgen --preset hardest --seed 6 --agentic
--model gpt-5.6-sol --max-iterations 2`) without the 400 error.
Do not commit; report files changed, suite counts, and the SDK-version
verification result.
