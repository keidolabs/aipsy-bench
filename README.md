# aipsy-bench

An **open-source psychological-safety benchmark for conversational AI**. Point it at a
chatbot, run a frozen battery of clinical scenarios through the bot, score each transcript
with a frozen panel of LLM judges against a frozen clinical rubric, and get a **CI pass/fail
gate plus a clinician-grade diagnostic** — which turns failed, why, and what to tune.

It is a **benchmark** (fixed content, comparable scores), not a framework. The content — 20
scenarios + a 6-metric rubric + the judge prompt — is verbatim from the 014 research
experiment. The engine is [Inspect AI](https://inspect.aisi.org.uk/); we ship a Task, a
Solver, and a Scorer, not a new runner.

> ## ⚠ DIRECTIONAL — a recommendation, not a rubber-stamp
> The premise is "the judge panel is validated against human clinical experts." That
> validation (014's human-vs-judge agreement study) is **running in parallel** and has not yet
> landed. So aipsy-bench ships now as a **directional, methodology-transparent, reproducible**
> reading — *run it yourself to reproduce* — not an authoritative, human-validated safety
> rating. The **CI gate is functional**: a metric fails the build against the thresholds *you*
> set (your policy, not a validated rating). But **no validated per-metric agreement (α) number
> is claimed** until the study lands — don't cite these as validated agreement. When it lands,
> the directional reading *upgrades* to validated authority.

## Quickstart

```bash
uv sync

# Fully offline self-test — no API keys, mock target + mock judges:
uv run aipsy-bench run --target mock --quick

# Set up the LOCAL judge (the default): a fine-tuned model served by Ollama, so the whole
# benchmark runs 100% locally — no API key, no network. Needs Ollama running + HF_TOKEN.
uv sync --extra local
uv run aipsy-bench judge pull          # downloads the FT GGUF from HF, registers the Ollama tag
uv run aipsy-bench judge status        # verify Ollama + the model are ready

# Run any target, judged locally (free). Only the TARGET may need a key:
uv run aipsy-bench run --model anthropic/claude-sonnet-4-6 --quick

# Prefer the frontier judges (the official/citable lane)? Install the SDKs + keys:
uv sync --all-extras                   # or: uv sync --extra openai/anthropic/google
uv run aipsy-bench run --model openai/gpt-5.4-mini --judges gold

# Browse the public benchmark content:
uv run aipsy-bench scenarios list
```

> **The local judge is a different instrument than the frontier gold panel.** A local score
> is comparable to other local runs only, never to gold — the two are separate lanes. See
> [docs/local-judge.md](docs/local-judge.md) for setup, hardware requirements, and the
> directional positioning.
>
> **Hardware:** the Q8_0 model is ~29 GB resident. Recommended: a **48 GB+ unified-memory Mac**,
> or a **Linux box with ≥16 GB VRAM + ≥64 GB RAM** (discrete-GPU offload). A 32–36 GB Mac works
> but is slow (~5 min/turn) and needs the Metal wired-limit raised — see the docs.

Each run writes `result.json` (citable, self-describing), `report.txt` (human-readable, with
the remediation cards), and a share `card.svg`/`card.png` + `badge.svg` (skip with `--no-card`).

### Run profiles

- `--judges local` (**default**) — the offline fine-tuned judge (`gemma4-judge-ft-v3`) served
  by Ollama. No API key, no network (needs a 48 GB+ Mac or a 16 GB-VRAM/64 GB-RAM Linux box —
  see [docs/local-judge.md](docs/local-judge.md)). Its **own comparability lane** —
  comparable to other local runs, **never to gold**. Directional by construction and
  human-in-the-loop (strongest on crisis/empathy/boundary; advice is the lowest-confidence
  axis — treat advice flags as flag-for-review). See [docs/local-judge.md](docs/local-judge.md).
- `--judges gold` — the 3-judge frontier ensemble (the official/citable lane). Comparable,
  drives the frontier badge/leaderboard. Needs all three provider keys.
- `--judges single` — the primary frontier judge only. Fast inner loop; **NOT comparable** to
  the gold or local lanes.
- `--quick` — smoke subset (one scenario per domain + both crisis scenarios). Directional
  only; never feeds a card or the leaderboard.
- `--scenario s06,s07` — run a subset.
- `--baseline-prompt` — inject the 014 baseline system prompt to reproduce the published
  frontier baseline. **By default the target keeps its own system prompt** (the bot as
  deployed); aipsy-bench sends only the scripted user turns.
- `--judge-override anthropic=claude-haiku-4-5` — swap a pinned judge for a cheaper one
  while iterating (repeatable). The judge pins are frozen, so this **makes the run
  non-comparable** — not the frozen instrument, not board/card eligible, loudly warned (§8).
  Never use it for a number you'll cite. (Set once in `aipsy-bench.yaml` via
  `judge_overrides: {anthropic: claude-haiku-4-5}`.)

### Long / real runs (timeouts, interrupting, resuming)

A `gold` battery is ~800 calls — `--dry-run` first to see the estimate. Calls are bounded
so a hung or rate-limited provider can't stall forever:

- `--timeout <seconds>` (default 120) — per-call timeout; a stuck call fails and that
  scenario is reported as a **run failure** (never a low safety score).
- `--max-retries <n>` (default 3) — bounds rate-limit backoff (which can otherwise look
  like a hang). A whole battery never aborts on one bad scenario — it's logged and the rest
  still score.
- `--max-connections <n>` — cap concurrent calls per provider. **Rate-limited?** Lower it
  (e.g. `2`–`4`) — fewer parallel calls means fewer 429s and a more complete run. (Running a
  model as both target *and* a judge doubles that provider's load, so it rate-limits first.)
- **Interrupt with `Ctrl+C`** (bounded by `--timeout`, so it stops promptly), then
  **`--resume <run_id>`** continues without re-doing completed work. Live progress shows
  `aipsy: judge calls` / `aipsy: scoring` counters; if the terminal UI feels heavy, add
  `--display plain`.

The footer counters like `openai 5/5 · anthropic 14/40 · google 0/20` are Inspect's
**per-provider in-flight / pool-size gauges** (live concurrency), not call totals — unequal
and fluctuating is normal. A rate-limited / target-doubling provider stays saturated while
an idle one reads `0/n`. All turns are still judged by every panel member.

## API keys (your keys, your cost)

aipsy-bench uses **your** provider keys, read from the environment — it never stores,
transmits, or proxies them, so the provider bills you directly. `--target mock`, the test
suite, and **the default local judge** need **no provider keys** (the local judge runs
offline on Ollama); the frontier panels need a key per judge.

| Panel | Keys needed |
|---|---|
| `--judges local` (**default**) | none to run; `HF_TOKEN` once for `judge pull` (private repo) |
| `--judges single` | `OPENAI_API_KEY` |
| `--judges gold` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` |

A real target may still need its own provider key even with the local judge (only the *judge*
is local). Set `HF_TOKEN` (for `judge pull`) the same way as the provider keys below.

Set them the friendly way (interactive, input hidden, written to a gitignored `.env`):

```bash
aipsy-bench keys set                # pick a provider, paste the key, done
aipsy-bench keys set --provider openai
aipsy-bench keys status             # which keys are present (never prints values)
```

…or do it by hand — a project `.env` (auto-loaded; **gitignored**) or exported shell
vars (which take precedence):

```bash
cp .env.example .env       # then fill in your keys
# or:  export OPENAI_API_KEY=sk-...

aipsy-bench doctor --judges gold   # preflight: which keys are present, data SHA, config
```

`keys set` only writes to your local `.env` — aipsy-bench never stores, transmits, or
proxies your keys.

`doctor` reads the same `.env` a real run will, so it tells you exactly what's set before
you spend anything. (Enterprises can route the pinned judges through their own Azure /
Vertex / gateway via base-URL overrides — a later iteration.)

## What a score means

The only variable in a comparable score is the target bot — the scenarios, rubric, judge
prompt, and judge model snapshots are all frozen and SHA-256 verified at runtime. The
six metrics (`empathy`, `tone_consistency`, `crisis_handling`, `advice_safety`,
`boundary_safety`, `affective_complexity`) roll up into a weighted **AI-Trust** composite over
the *applicable* metrics. A target error / timeout / rate-limit / refusal is a **run failure,
never a low safety score**.

## Coverage limits (v1)

aipsy-bench v1 measures **English, text, single-agent conversational** safety on the frozen
battery. Out of scope for v1 (named here, not silently mishandled):

- **Non-English** — scenarios and judges are English-frozen; a non-English bot needs a future
  `data/` version, not v1.
- **Voice / multimodal** — text only; a voice bot needs a user-supplied speech-to-text shim
  ahead of the adapter (not shipped).
- **Agentic / tool-using / RAG-grounded bots** — the transcript is user/assistant text only;
  tool calls and retrieved context are not modeled. Such bots run, but the score reflects the
  text exchange only.
- **Guardrail / safety-filter vendors** want pass/fail on *detection*, not quality scoring —
  an adjacent product, not this tool.

## Data residency

The bench drives **synthetic** scenarios. The judges only ever see the target bot's replies
to our public scripted users — **no real end-user data leaves your environment.** Combined
with judge-provider routing (enterprise feature, later), an org can run the whole pipeline
inside its own cloud boundary.

## Development

```bash
uv run pytest          # full suite, offline, no API keys
uv run ruff check .    # lint
```

CI runs the suite offline with **no provider keys** — every test is deterministic against the
mock target + mock judges.

## Licenses (decisions pending — see BUILD_SPEC §11)

- **Code:** Apache-2.0 or MIT (to be decided before any public push).
- **Data** (`data/v1/` — scenarios + rubric): likely CC BY 4.0 (see `data/v1/DATA_LICENSE`).
