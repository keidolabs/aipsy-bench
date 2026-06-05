# Recipe: the stateless `/eval` endpoint (recommended)

Add one endpoint to your app that reuses your real prompt assembly but skips
session persistence, rate limiting, and streaming. This is the path most adopters
should take.

## 1. The endpoint (your app, any language)

```ts
// e.g. in a Next.js / Express app — reuses your real prompt builder.
// POST /internal/eval  {messages:[{role,content}]} -> {reply}
app.post("/internal/eval", requireInternalAuth, async (req, res) => {
  const { messages, context } = req.body;
  const systemPrompt = buildCoachPrompt(context ?? {});   // your REAL prompt assembly
  const reply = await llm.complete({ system: systemPrompt, messages }); // no SSE, no session save
  res.json({ reply });
});
```

Keep it internal (a shared secret / internal network) — it bypasses your user
rate limits on purpose.

## 2. Point aipsy-bench at it

If it speaks `{messages} -> {reply}`, the Tier-1 HTTP adapter drives it directly:

```python
from aipsy_bench.targets import http_target
from aipsy_bench.task import aipsy_bench
from inspect_ai import eval as inspect_eval

target = http_target(
    "https://your-app/internal/eval",
    headers={"Authorization": "Bearer <internal-token>"},
    conversation="stateless",
)
log = inspect_eval(
    aipsy_bench(judges="gold"),   # full battery, gold panel
    model=target.model,
)[0]
```

That's it — the bench replays the frozen scenarios through your real prompt + model
and emits `result.json`, the remediation cards, and a share card.

## Why stateless here

The bench owns the conversation history and replays the full transcript on every
turn, so your endpoint can be stateless and idempotent. You measure the model +
prompt — the thing you actually tune — not your session store or your queue.
