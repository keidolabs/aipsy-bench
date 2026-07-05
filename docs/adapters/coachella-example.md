# Worked example: benchmarking 'coachella' app (a real deployed coach app with anonymized name)

coachella's AI coach is the canonical "real app": **stateful** (Supabase, last-10-message
window), **streaming** (SSE), **auth-gated** (Supabase Bearer), **rate-limited**
(10 messages / free user — a 20×~10-turn run dies on scenario 1), reached over a
`start → message → stream` handshake. Prompt assembly is
`AICoachService.buildCoachPrompt` ("Alex from Manchester", gpt-4o).

Benchmarking *through* that production path is the wrong move (rate limits, SSE,
session storage are not what you're tuning). Take the recommended path: a stateless
`/eval` endpoint that **reuses `buildCoachPrompt`** — so the target keeps its Alex
persona — but skips session persistence, the rate limit, and SSE.

## 1. coachella side — add `POST /api/ai-coach/eval` (≈30 lines)

> Cross-repo change in `coachella`. Do this with sign-off; it is a separate deploy.

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
  // so reconstruct coachella's context string + current message from `messages`.
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
coach model, but bypasses Supabase / the rate limit / SSE. Because it intentionally has no
user rate limit, the load-bearing guard is **env-gating it out of production** (refuse to
serve unless dev/test); the `x-eval-secret` header shown here is the belt-and-suspenders for a
staging/preview deploy — **on a pure localhost run it's optional** (nothing off your machine
reaches `127.0.0.1`). See [eval-endpoint.md](./eval-endpoint.md#protecting-the-endpoint).

## 2. aipsy-bench side — point the CLI at it (no Python)

Local dev is the primary path: run `npm run dev`, then benchmark `localhost` with the local
judge — fully offline, no keys.

```bash
aipsy-bench doctor --http-target http://localhost:3000/api/ai-coach/eval   # preflight
aipsy-bench run \
  --http-target http://localhost:3000/api/ai-coach/eval \
  --header x-eval-secret:$EVAL_SECRET \
  --ref coachella-coach \
  --judges local --quick                                                   # fast inner loop
```

Or commit it to `aipsy-bench.yaml` so the whole team runs a bare `aipsy-bench run`:

```yaml
target:
  http: http://localhost:3000/api/ai-coach/eval
  headers: { x-eval-secret: ${EVAL_SECRET} }
  conversation: stateless          # the /eval endpoint is stateless; the bench owns history
  ref: coachella-coach
judges: local
```

Only reach for a Python driver if you must benchmark the *production* path (auth handshake,
server sessions, SSE) instead of a stateless endpoint — see [`callable.md`](./callable.md).

## 3. The cold-start loop, without us in the loop

1. **Score it** — `aipsy-bench run … --judges local --quick --out run/a` against the endpoint
   (fast, offline, directional). The full battery (drop `--quick`) is the gate before you ship.
2. **Read the card** — a phase-localized remediation card ("crisis_handling fails in
   the Pressure phase of s06, here's the failing turn + the fix"): `run/a/report.txt`.
3. **Fix** — tweak `buildCoachPrompt` (e.g. add an always-on crisis-resource hand-off).
4. **Re-run + diff** — `aipsy-bench run … --out run/b` then `aipsy-bench compare run/a run/b`
   shows the delta and fails CI if safety regressed.
5. **Stay in one lane** — a `local` score compares to your own prior `local` runs, never to the
   frontier board. `--against-board` (overlay on the published vanilla baselines) is the
   **gold** lane — switch to `--judges gold` (provider keys) only when you want that citable
   reference frame.

## Correctness guarantee on a real, rate-limited target

coachella's free tier caps at 10 messages — a full run *will* hit limits if you point at
the production path. Whatever the transport returns on failure (HTTP 429, timeout,
empty body), aipsy-bench classifies the turn (`target_error | refusal | empty |
truncated`) and marks the **scenario a run failure** — it is never fed to the judge
as an unsafe answer, and the run can't pass the gate or be carded. Scoring an infra
failure as `crisis_handling ≈ 1.0` would be a false-unsafe (§6). The `/eval` endpoint
exists precisely so you benchmark the model + prompt, not your rate limiter.

> Scores are **DIRECTIONAL** — a recommendation, not a rubber-stamp — until the 014
> human-validation study lands, then upgrade to validated authority (§0.3). The CI gate is
> functional now against the thresholds you set; no validated per-metric agreement is claimed yet.
