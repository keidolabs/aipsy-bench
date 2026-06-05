# Worked example: benchmarking mojoe (a real deployed coach app)

mojoe's AI coach is the canonical "real app": **stateful** (Supabase, last-10-message
window), **streaming** (SSE), **auth-gated** (Supabase Bearer), **rate-limited**
(10 messages / free user — a 20×~10-turn run dies on scenario 1), reached over a
`start → message → stream` handshake. Prompt assembly is
`AICoachService.buildCoachPrompt` ("Alex from Manchester", gpt-4o).

Benchmarking *through* that production path is the wrong move (rate limits, SSE,
session storage are not what you're tuning). Take the recommended path: a stateless
`/eval` endpoint that **reuses `buildCoachPrompt`** — so the target keeps its Alex
persona — but skips session persistence, the rate limit, and SSE.

## 1. mojoe side — add `POST /api/ai-coach/eval` (≈30 lines)

> Cross-repo change in `mojoe-admin`. Do this with sign-off; it is a separate deploy.

`buildCoachPrompt` is `private static` today; expose a thin public wrapper, then:

```ts
// src/app/api/ai-coach/eval/route.ts  (Next.js App Router)
import { AICoachService } from "@/lib/services/ai-coach";
import { OpenAIService } from "@/lib/services/openai";

export async function POST(req: Request) {
  if (req.headers.get("x-eval-secret") !== process.env.EVAL_SECRET) {
    return new Response("unauthorized", { status: 401 });
  }
  const { messages, context } = await req.json();

  // aipsy-bench replays the full transcript each call (conversation: stateless),
  // so reconstruct mojoe's context string + current message from `messages`.
  const history = messages
    .slice(0, -1)
    .map((m: { role: string; content: string }) =>
      `${m.role === "user" ? "Athlete" : "Coach"}: ${m.content}`)
    .join("\n");
  const current = messages[messages.length - 1].content;

  const systemPrompt = AICoachService.buildCoachPromptPublic(context ?? null, history, current);
  const reply = await OpenAIService.generateCoachResponse(systemPrompt); // gpt-4o, Alex persona
  return Response.json({ reply });
}
```

This reuses the *real* prompt assembly (no cross-language prompt drift) and the real
coach model, but bypasses Supabase / the rate limit / SSE. Keep it internal (the
`x-eval-secret` header), since it intentionally has no user rate limit.

## 2. aipsy-bench side — Tier-1 HTTP adapter

```python
from aipsy_bench.targets import http_target
from aipsy_bench.task import aipsy_bench
from inspect_ai import eval as inspect_eval

target = http_target(
    "https://mojoe.app/api/ai-coach/eval",
    headers={"x-eval-secret": "<EVAL_SECRET>"},
    conversation="stateless",       # the /eval endpoint is stateless; we own history
    ref="mojoe-coach",
)
log = inspect_eval(aipsy_bench(judges="gold"), model=target.model)[0]
```

Or the CLI (preflight cost first):

```bash
aipsy-bench run --target mock --dry-run            # sanity-check call counts/cost
aipsy-bench run --model <ref> --judges gold --against-board   # vanilla mojoe via Tier-0, for a baseline
```

## 3. The cold-start loop, without us in the loop

1. **Score it** — `aipsy-bench run … --judges gold` against the endpoint.
2. **Read the card** — a phase-localized remediation card ("crisis_handling fails in
   the Pressure phase of s06, here's the failing turn + the fix").
3. **Fix** — tweak `buildCoachPrompt` (e.g. add an always-on crisis-resource hand-off).
4. **Re-run + diff** — `aipsy-bench compare base.eval cand.eval` shows the delta and
   fails CI if safety regressed.
5. **Reference frame** — `--against-board` overlays the score on the vanilla baselines.

## Correctness guarantee on a real, rate-limited target

mojoe's free tier caps at 10 messages — a full run *will* hit limits if you point at
the production path. Whatever the transport returns on failure (HTTP 429, timeout,
empty body), aipsy-bench classifies the turn (`target_error | refusal | empty |
truncated`) and marks the **scenario a run failure** — it is never fed to the judge
as an unsafe answer, and the run can't pass the gate or be carded. Scoring an infra
failure as `crisis_handling ≈ 1.0` would be a false-unsafe (§6). The `/eval` endpoint
exists precisely so you benchmark the model + prompt, not your rate limiter.

> All scores remain **PROVISIONAL** until the 014 human-validation gate lands (§0.3).
