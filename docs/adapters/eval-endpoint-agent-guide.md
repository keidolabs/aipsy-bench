# AI-agent guide: build your `/eval` endpoint (Claude Code / Codex)

**Purpose of this doc:** hand the prompt below to your AI coding agent and let it build the
`/eval` endpoint for you — no hand-writing required. (Want to write it yourself instead? The
[`/eval` cookbook](./eval-endpoint.md) has the hand-written framework snippets.)

The `/eval` endpoint is the **one piece of integration you write** to benchmark your own bot.
It's a thin route that reuses your app's *real brain* — system prompt, conversation history,
RAG/retrieval, tools, model — and strips the *plumbing* — auth, rate limits, sessions,
streaming. This guide hands you a **verbatim prompt** to give your coding agent so it builds
that route correctly, plus a **second prompt** to add a secret if the endpoint will ever be
reachable off your laptop.

It's **framework- and language-agnostic**: the prompt tells the agent to discover and reuse
*your* real chat code, whatever the stack (Next.js, Express, FastAPI, Rails, Go, …).

> **Where this fits.** This is the *build the endpoint* step. To **run** the benchmark once
> the endpoint exists (the offline `doctor` → `run` → `compare` loop with the local judge), see
> the [`/eval` cookbook](./eval-endpoint.md). For a real worked example (an auth-gated,
> stateful, streaming production app reduced to a stateless `/eval`), see
> [`coachella-example.md`](./coachella-example.md).

---

## The one hard rule: it must never reach production

The endpoint deliberately has **no auth and no rate limit** — that's the whole point, so you
measure the model + prompt, not your rate limiter. The load-bearing guarantee is therefore
simple: **it is not reachable in production.** Two postures, cheapest first:

- **Localhost (the default — keep it dead simple).** Env-gate the route so it 404s in
  production, and that's it. Nothing off your machine can reach `127.0.0.1`, so **no secret is
  needed.** Use **Prompt 1**.
- **Off-localhost (a staging/preview deploy).** If the route is reachable over a network, layer
  a **shared-secret header** on top of the env-gate. Use **Prompt 1**, then **Prompt 2**.

The env-gate is what actually keeps it out of prod; the secret is belt-and-suspenders for a
deploy. Keep any secret in a **header, never the URL** (a query string would leak into logs and
the benchmark's `result.json`).

## What goes in, what stays out

| Keep — your real brain (the thing under test) | Strip — plumbing (not what you're tuning) |
|---|---|
| System-prompt assembly, exactly as production builds it | User auth / login |
| Conversation-history handling | Rate limits / quotas |
| RAG / retrieval | Session persistence, DB reads/writes |
| Tool / function calls | Streaming / SSE (return the whole reply as one JSON) |
| The model + its params, the persona | Analytics, telemetry, logging side effects |

**Reuse your existing functions — don't re-implement them,** or you'll measure a copy that
drifts from your bot. Any per-user data your prompt normally injects (profile, account history)
becomes a **fixed synthetic value or `null`** — never a real user's data. The endpoint must be
idempotent.

## The contract (what aipsy-bench sends and expects)

```
POST <your-route>   { "messages": [ { "role": "user"|"assistant", "content": string }, ... ] }
                ->  { "reply": string }
```

- The benchmark **owns the conversation and replays the full transcript on every call**
  (`conversation: stateless`, the default), so `messages` already carries the whole dialogue.
  The **last message is always the new user turn.**
- Return the reply as **plain JSON, not streamed**. aipsy-bench also accepts the OpenAI
  `choices[0].message.content` response shape if that's what you already return.
- The **route (host / port / path) is yours** — pick anything; you'll point the CLI at exactly
  that URL.

---

## Prompt 1 — localhost (the default)

Hand this to Claude Code / Codex **from inside your app's repo** — **nothing to fill in**, just
copy-paste. The agent picks the route path and discovers the rest of your code. (Optional: add
one line naming your framework or a route path you'd prefer.)

```text
Add a stateless internal "eval" endpoint to this app so an external benchmark (aipsy-bench) can
score our chatbot's replies. Follow this exactly — the benchmark depends on the contract.

THE ONE HARD RULE — it must never run in production:
- This route bypasses our auth and rate limits, so guard it as the FIRST thing it does: return
  404 unless we are in dev/test (e.g. NODE_ENV / APP_ENV != "production"), OR gate it behind an
  explicit env flag like ENABLE_EVAL_ENDPOINT=1. In production with neither set, it 404s.
- Do NOT add auth or a rate limiter to this route. (A shared secret is a separate later step,
  only if we ever deploy it off localhost.)

CONTRACT (exact):
- Route: POST /internal/eval  (use this path; if it collides with an existing route, pick a
  sibling like /api/internal/eval instead, and tell me the final path in your reply).
- Request body:  { "messages": [ { "role": "user" | "assistant", "content": string }, ... ] }
- Response body: { "reply": string }   — plain JSON, NOT streamed.
- The benchmark owns the conversation and replays the FULL transcript on every call, so
  `messages` already contains the whole dialogue. The LAST message is always the new user turn.

REUSE OUR REAL BOT — do not re-implement any of its logic:
- Find the existing code that our normal chat path uses to build the system prompt and call the
  model, and call it. Feed it the SAME inputs it uses in production, rebuilt from `messages`:
  - System prompt: assemble it exactly as production does.
  - History: if our builder wants a formatted history string instead of a messages array,
    reconstruct it in that same format from messages[:-1], and pass messages[-1].content as the
    current user turn. If it takes a messages array directly, pass it through.
  - RAG / retrieval / tools: KEEP them — run the same retrieval and tool calls the real chat
    path runs. That behavior is part of what's being measured.
- If the real function is private, add a thin PUBLIC wrapper that forwards to it unchanged —
  do NOT alter its logic.

STRIP THE PLUMBING (this is why the endpoint exists — we measure the model + prompt):
- No login/auth, no rate limit/quota, no session or DB read/write, no streaming (return the
  whole reply as one JSON response), no analytics/logging side effects.
- Any per-user data our prompt normally injects (profile, account history, etc.): pass null, or
  a FIXED synthetic value — NEVER a real user's data. The endpoint must be idempotent.

Finally, when you're done, report:
- NO-DRIFT CHECK — prove the endpoint reuses the real brain, not a copy that will drift. For one
  sample conversation, call BOTH paths and print them SIDE BY SIDE in the terminal so I can
  compare: (a) the final system prompt + model + params (+ any RAG/retrieved context) the eval
  endpoint sends to the model, and (b) what our real production chat path assembles for the SAME
  input. They must match — identical system prompt and model/params; the only allowed differences
  are the stripped plumbing (auth/session/rate limit/streaming). If the prompt or model differs,
  you forked the logic instead of reusing it — fix the endpoint to call the SAME function, don't
  patch the copy. Also print both paths' final replies for that input so I can eyeball the voice
  (minor wording differences are fine if the model samples with temperature > 0; a different
  persona is not).
- The full endpoint URL to paste into the benchmark — host + port + the final route path, e.g.
  http://localhost:3000/internal/eval — I'll drop it straight into `aipsy-bench init --http-target <url>`.
- A curl command that POSTs a 2-message conversation to the route and prints the {reply}.
- Confirmation the route returns 404 when NODE_ENV/APP_ENV is "production".
```

## Prompt 2 — add a secret (only for an off-localhost deploy)

Skip this for a pure localhost run. Use it **only** when the endpoint will be reachable over a
network (staging/preview). Run it **after** Prompt 1, in the same repo:

```text
We're going to deploy the /eval endpoint somewhere reachable off localhost (a staging/preview
box). Add ONE more gate on top of the existing production env-gate — a shared secret — without
changing any of the endpoint's logic:

- If the env var EVAL_SECRET is set, require a request header `x-eval-secret` whose value equals
  EVAL_SECRET; otherwise return 401. If EVAL_SECRET is NOT set, skip this check (so a pure
  localhost run stays gate-free).
- Read the secret from the environment only — never hardcode it, never log it, never put it in
  the URL or a query string.
- Keep the production 404 env-gate exactly as it is. This secret is belt-and-suspenders for the
  off-localhost deploy, NOT a replacement for keeping the route out of production.

Then show me the curl again, now sending  -H "x-eval-secret: $EVAL_SECRET".
```

---

## After it builds: sanity-check, then point aipsy-bench at it

```bash
# 1. hit the endpoint directly (your app running locally) — substitute YOUR route:
curl -s -X POST http://localhost:3000/internal/eval \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"rough day — got any advice?"}]}'
# → {"reply":"..."}   a real reply from YOUR bot, in its own voice

# 2. let aipsy-bench probe it (catches a wrong port/path/secret before a full run), then run —
#    fully offline with the local judge, no API keys:
aipsy-bench doctor --http-target http://localhost:3000/internal/eval
aipsy-bench run --http-target http://localhost:3000/internal/eval --judges local --quick

# gated (off-localhost) endpoint? add the header — the shell expands the env var, so the
# secret never lands in the config or the report:
#   --http-target https://staging.example.app/internal/eval --header x-eval-secret:$EVAL_SECRET
```

The full offline loop — `init` to scaffold a config, the `--quick` inner loop, `compare` for
regressions, and the full-battery gate — is in the [`/eval` cookbook](./eval-endpoint.md).

> A target error, timeout, rate-limit, empty, or filtered reply is classified as a **run
> failure — never scored as an unsafe answer** — so a flaky endpoint can't produce a
> false-unsafe. Fix the plumbing and re-run.

## Before you commit or deploy — safety checklist

- [ ] The route returns **404 in production** — the env-gate is present and you've verified it.
- [ ] **No auth / rate-limit / session code** was added to the route (stripping that is the point).
- [ ] **No real user data** flows through it — per-user context is `null` or a fixed synthetic value.
- [ ] Reachable off localhost? **Prompt 2's `x-eval-secret` gate is in place** and the secret
      lives in an env var — never in the URL, a query string, or source.
- [ ] Ideally it isn't shipped to your production target **at all** — a dev/preview box is safest.
