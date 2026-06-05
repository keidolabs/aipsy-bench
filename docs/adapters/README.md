# Target adapters — connecting your bot to aipsy-bench

"The thing under test" is always the same shape: `message_history -> reply`.
Everything downstream (scoring, diagnostics, gate) is identical regardless of how
your bot is reached. Pick the lowest tier that fits.

| Tier | Use it when | How |
|---|---|---|
| **0 — model string** | ranking a bare foundational model | `aipsy-bench run --model openai/gpt-5.4-mini` |
| **1 — HTTP** | a bot behind a thin **stateless** OpenAI-chat-style proxy | `http_target(url, headers=...)` |
| **2 — Python callable** | **a real deployed app** (auth, sessions, streaming, rate limits) | `callable_target(fn)` |

## Recommended integration for a real app: a stateless `/eval` endpoint

Most adopters should take this path. Rather than benchmark *through* your
production session storage, rate limits, and SSE streaming, add one thin endpoint
that **reuses your real prompt assembly** but skips the operational machinery:

```
POST /internal/eval        {messages: [{role, content}, ...], context?: {...}}
                        ->  {reply: "..."}
```

Why this is the highest-leverage 30 lines you'll write:

- **You benchmark the variable you're tuning** (model + prompt), not Supabase /
  your queue / your CDN.
- **Runs are repeatable and unthrottled** — a 20-scenario × ~10-turn `gold` run is
  ~200 target calls and will die instantly against a free-tier rate limit.
- **No cross-language prompt drift** — the endpoint calls your real prompt builder
  (e.g. `buildCoachPrompt`), so the target keeps its own persona; the Python
  adapter never re-implements your prompt logic.

Then drive it with the Tier-1 HTTP adapter (if it speaks `{messages}->{reply}`) or
a trivial Tier-2 callable. See [`eval-endpoint.md`](./eval-endpoint.md).

## `conversation: stateless | session`

- **`stateless`** (default) — the harness owns history and replays the full
  transcript each turn. Use this for an `/eval` endpoint.
- **`session`** — the target owns history server-side. The harness opens a session
  once and sends only the *new* user turn each step; your function reconstructs
  context (including any server-side truncation window). The report records which
  mode produced the score — testing through a 10-message window vs. full history can
  change the Deepening/Resolution-phase results the diagnostics localize.

## Target failures are run failures, never low safety scores

If a turn errors, times out, rate-limits, returns empty, or is content-filtered,
aipsy-bench classifies it (`target_error | refusal | empty | truncated`) and marks
the **scenario a run failure** — it is *not* fed to the judge as if it were an
unsafe answer. A run with unresolved target failures cannot pass the gate or
produce a card/board row. Fix the plumbing and re-run; never ship a false-unsafe.

See [`callable.md`](./callable.md) for a worked auth + session + SSE example.
