# Cookbook: benchmark your own app via a stateless `/eval` endpoint

This is the path **most app developers should take**, and it's the fastest way to get a
psychological-safety read on your bot before a release. You add **one thin endpoint** to
your app that reuses your *real* prompt assembly but skips session persistence, rate
limiting, and streaming — then point the `aipsy-bench` CLI straight at it. **No Python,
no driver script.**

The whole loop runs against your **local dev server** (`npm run dev`, `uvicorn`, …) with
the **local judge**, so it's 100% offline: no API keys anywhere, nothing leaves your machine.

```
POST /internal/eval     {messages: [{role, content}, ...]}
                     ->  {reply: "..."}
```

---

## 1. Add the endpoint (your app, any language)

**The only contract is the shape:** `{messages: [{role, content}, …]}` in, `{reply: "…"}`
out (aipsy-bench also accepts the OpenAI `choices[0].message.content` response shape).
**Everything else is yours** — the host, port, and path (`/eval`, `/internal/eval`,
`/api/whatever`); pick any route your framework likes. Reuse `buildCoachPrompt` /
`assemblePrompt` / whatever you already have (that's the point — no cross-language prompt
drift; the target keeps its own persona).

**Next.js (App Router)**

```ts
// src/app/api/<your-path>/eval/route.ts
import { buildCoachPrompt } from "@/lib/coach";
import { openai } from "@/lib/openai";

export async function POST(req: Request) {
  // The load-bearing guard: this route must not exist in production (see "Protecting the
  // endpoint" below). On a localhost dev server this is all you need.
  if (process.env.NODE_ENV === "production") return new Response("not found", { status: 404 });
  // OPTIONAL second gate — only if the endpoint is reachable off-localhost (staging/preview):
  if (process.env.EVAL_SECRET && req.headers.get("x-eval-secret") !== process.env.EVAL_SECRET)
    return new Response("unauthorized", { status: 401 });

  const { messages } = await req.json();                 // [{role, content}, ...]
  const system = buildCoachPrompt();                     // your REAL prompt assembly
  const r = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "system", content: system }, ...messages],
  });
  return Response.json({ reply: r.choices[0].message.content });
}
```

**Express**

```js
app.post("/internal/eval", requireInternalAuth, async (req, res) => {
  const system = buildCoachPrompt();                     // your REAL prompt assembly
  const r = await llm.complete({ system, messages: req.body.messages });
  res.json({ reply: r.text });
});
```

**FastAPI (Python)**

```python
@app.post("/internal/eval")
def eval_endpoint(body: EvalBody, x_eval_secret: str = Header(...)):
    if x_eval_secret != os.environ["EVAL_SECRET"]:
        raise HTTPException(401)
    system = build_coach_prompt()                          # your REAL prompt assembly
    reply = llm.complete(system=system, messages=body.messages)
    return {"reply": reply}
```

### Protecting the endpoint

This route reuses your real model with **no rate limit and no user auth**, so the guarantee
that matters is: **it must not be reachable in production.** Two levers, cheapest first:

- **Env-gate the route** (the guard above) — don't serve it unless you're in dev/test, or put
  it behind an explicit `ENABLE_EVAL_ENDPOINT` flag. On a **localhost** dev server this is
  enough on its own: nothing off your machine can reach `127.0.0.1`, so **a secret is optional
  ceremony there.**
- **A shared secret / network ACL** — add the `x-eval-secret` check *only* when the endpoint is
  reachable off-localhost (a staging/preview deploy). It's a low bar (a static header), not a
  real security boundary — the env-gate is what actually keeps it out of prod.

Keep any secret in a **header, not the URL** — headers never land in `result.json` or a share
card; a query string would.

> **Reconstructing history.** aipsy-bench replays the **full transcript** on every call
> (`conversation: stateless`, the default), so `messages` already carries the whole
> conversation. If your prompt builder wants a formatted history string instead of an OpenAI
> `messages` array, map it inline — see [`mojoe.md`](./mojoe.md) for a worked `Athlete:/Coach:`
> reconstruction.

## 2. Point aipsy-bench at it — from the CLI, no Python

```bash
# fully offline: your local dev server + the local judge, no keys.
# The URL is yours — substitute your dev server's host, port, and path.
aipsy-bench run \
  --http-target http://localhost:3000/eval \
  --judges local --quick

# add --header ONLY if your endpoint is gated (e.g. a staging/preview deploy):
#   --http-target https://staging.example.app/internal/eval --header x-eval-secret:$EVAL_SECRET
```

- `--http-target URL` builds the Tier-1 HTTP adapter for you (request `{messages}` → `{reply}`;
  it also accepts the OpenAI `choices[0].message.content` shape).
- `--header NAME:VALUE` is repeatable and optional; the shell expands `$EVAL_SECRET` before it's
  sent. Omit it entirely for an ungated localhost endpoint.
- `--ref my-coach` sets a friendly label for the report/card (defaults to the URL).

Prefer to commit the config? Scaffold it with **`aipsy-bench init`** (interactive, or all
flags) — it writes a valid `aipsy-bench.yaml` and, for a gated endpoint, references the secret
as `${ENV}` so it never lands in git:

```bash
aipsy-bench init --http-target http://localhost:3000/eval            # ungated localhost
aipsy-bench init --http-target https://staging/eval \
  --secret-header x-eval-secret --secret-env EVAL_SECRET             # gated (writes ${EVAL_SECRET})
```

…or hand-write it, so a teammate just runs `aipsy-bench run`:

```yaml
target:
  http: http://localhost:3000/eval    # your dev server's host / port / path
  # headers are OPTIONAL — include only for a gated (off-localhost) endpoint:
  headers:
    x-eval-secret: ${EVAL_SECRET}     # ${ENV} expanded at load — the secret stays out of git
  conversation: stateless
judges: local
```

CLI flags always override the config, and `--header` merges over the config headers per key.

## 3. The pre-release loop (fully offline)

The recommended cadence for iterating on a system prompt — nothing billed, nothing leaves
your machine:

```bash
aipsy-bench judge pull                                   # one-time: fetch the local judge
aipsy-bench doctor --http-target http://localhost:3000/eval   # preflight: verifies data + local
#   judge, and PROBES the endpoint once (catches a wrong port/path/secret before the battery)

# --- inner loop: fast, directional ---
aipsy-bench run --http-target … --judges local --quick --out run/a   # smoke subset
#   → read run/a/report.txt: which turns failed, in which conversation phase, and the fix

# tweak buildCoachPrompt, re-run:
aipsy-bench run --http-target … --judges local --quick --out run/b
aipsy-bench compare run/a run/b                          # the delta; fails on a safety regression

# --- the gate: the full battery before you ship ---
aipsy-bench run --http-target … --judges local --out run/release      # all 20 scenarios
```

`--quick` (one scenario/domain + both crisis scenarios) is the fast directional inner loop;
the **full battery is the gate** (`--quick` results are directional only, never carded/compared
across runs). The local judge is slower than an API but free and offline — a `--quick` pass is a
handful of minutes, the full battery is longer (see [local-judge.md](../local-judge.md) hardware
notes). `compare` diffs two runs and **fails CI on a safety regression** (§7.1).

> **Which judge?** The local judge is the default and keeps the loop offline/no-key. If you
> already have provider keys and want faster wall-clock while iterating, `--judges single` is a
> fast frontier inner loop — but a `single` score is **not comparable** to a `local` or `gold`
> score. Pick one lane and stay in it for a given before/after comparison.

## Two things worth deciding

- **Context injection.** If your app injects a user profile / prior-conversation history from
  your DB, the `/eval` endpoint above runs **without it** — a clean baseline where the only
  variable is your model + prompt. That's the right default for a comparable number. If you want
  to test whether a *representative* injected profile changes safety behavior, inject a **fixed
  synthetic** profile inside the endpoint (never real user data) — but treat that as a separate,
  labeled experiment, not the baseline.
- **Stateless vs. session.** Stateless (default) means your endpoint is idempotent and the bench
  owns history — simplest, and what the recipe above assumes. Use `--conversation session` only
  if your target *must* own history server-side (then it receives only the new turn each step);
  the report records which mode produced the score.

## Target failures are run failures, never low safety scores

If a turn errors, times out, returns empty, or is content-filtered, aipsy-bench classifies it
(`target_error | refusal | empty | truncated`) and marks the **scenario a run failure** — it is
*not* fed to the judge as if it were an unsafe answer, and the run can't pass the gate or be
carded. Fix the plumbing and re-run; never ship a false-unsafe (§6).

## When you can't add a stateless endpoint

If your bot genuinely can't be reached statelessly — hard auth handshake, server-owned
sessions, SSE-only — drop to a **Tier-2 Python callable** that owns that lifecycle. See
[`callable.md`](./callable.md). Everything downstream (scoring, diagnostics, gate) is identical.
