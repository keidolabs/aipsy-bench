<div align="center">

# 🛡️ aipsy-bench

**Open-source psychological-safety benchmark for conversational AI**

Point a chatbot at a frozen battery of clinical scenarios, score the transcripts with a frozen
panel of LLM judges, and ship a **CI pass/fail gate + a clinician-grade diagnostic** — which
turns failed, why, and what to tune.

[![CI](https://github.com/drKeeman/aipsy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/drKeeman/aipsy-bench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aipsy-bench?color=blue)](https://pypi.org/project/aipsy-bench/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code license: Apache 2.0](https://img.shields.io/badge/code-Apache_2.0-blue.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC_BY_4.0-informational.svg)](data/v1/DATA_LICENSE)
[![Built on Inspect AI](https://img.shields.io/badge/built_on-Inspect_AI-6f42c1.svg)](https://inspect.aisi.org.uk/)
![Status: Directional](https://img.shields.io/badge/status-directional-orange.svg)

**[Quickstart](#quickstart)** · **[Benchmark your app](docs/adapters/README.md)** · **[Local judge](docs/local-judge.md)** · **[Run profiles](#run-profiles)** · **[What a score means](#what-a-score-means)**

<sub>Built and maintained by <a href="https://www.keidolabs.com"><b>Keido Labs</b></a></sub>

</div>

---

It is a **benchmark** (fixed content, comparable scores), not a framework. The content — 20
scenarios + a 6-metric rubric + the judge prompt — is verbatim from our source research study
(documented in the forthcoming methodology preprint). The engine is
[Inspect AI](https://inspect.aisi.org.uk/); we ship a Task, a
Solver, and a Scorer, not a new runner.

> [!IMPORTANT]
> **⚠ Directional — a recommendation, not a rubber-stamp.** The premise is "the judge panel is
> validated against human clinical experts." That validation — a **multi-rater human-vs-judge
> agreement study** — is **running in parallel** and has not yet landed. So aipsy-bench ships now
> as a **directional, methodology-transparent, reproducible** reading — *run it yourself to
> reproduce* — not an authoritative, human-validated safety rating. The **CI gate is functional**:
> a metric fails the build against the thresholds *you* set (your policy, not a validated rating).
> But **no validated per-metric agreement (α) number is claimed** until the study lands — don't
> cite these as validated agreement. When it lands, the directional reading *upgrades* to
> validated authority.

## Quickstart

```bash
uv sync

# Fully offline self-test — no API keys, mock target + mock judges:
uv run aipsy-bench run --target mock --quick

# Set up the LOCAL judge (the default): a fine-tuned model served by Ollama, so the whole
# benchmark runs 100% locally — no API key, no network. Needs only Ollama running.
uv sync --extra local
uv run aipsy-bench judge pull          # downloads the GGUF from HF (public, no token), registers the Ollama tag
uv run aipsy-bench judge status        # verify Ollama + the model are ready

# Run any target, judged locally (free). Only the TARGET may need a key:
uv run aipsy-bench run --model anthropic/claude-sonnet-4-6 --quick

# Benchmark YOUR OWN app: add a stateless /eval endpoint, point the CLI at it (no Python).
# With your dev server up (npm run dev / uvicorn) + the local judge, this is 100% offline.
uv run aipsy-bench init --http-target http://localhost:3000/eval   # scaffold aipsy-bench.yaml
uv run aipsy-bench doctor                                          # preflight + probe the endpoint
uv run aipsy-bench run --judges local --quick                      # the URL is yours; --header if gated
# → full recipe (endpoint snippets, agent-built option, pre-release loop): docs/adapters/

# Browse the public benchmark content:
uv run aipsy-bench scenarios list
```

> **Benchmarking your own app?** See the [adapter guides](docs/adapters/README.md): a stateless
> `/eval` endpoint (localhost-first, fully offline) you write by hand
> ([cookbook](docs/adapters/eval-endpoint.md)) or have a
> [coding agent build](docs/adapters/eval-endpoint-agent-guide.md), plus a Tier-2 Python
> [callable](docs/adapters/callable.md) for apps that must own auth / sessions / SSE.

> **Prefer the frontier judges** (the official/citable `gold` lane) instead of the local
> default? That's an alternative lane — install the provider SDKs + keys
> (`uv sync --all-extras`) and add `--judges gold`. See [API keys](#api-keys-your-keys-your-cost).

> **The local judge is a different instrument than the frontier gold panel.** A local score
> is comparable to other local runs only, never to gold — the two are separate lanes. See
> [docs/local-judge.md](docs/local-judge.md) for setup, hardware requirements, and the
> directional positioning.
>
> **Hardware:** the Q8_0 model is ~27 GB resident. Recommended: a **48 GB+ unified-memory Mac**,
> or a **Linux box with ≥16 GB VRAM + ≥64 GB RAM** (discrete-GPU offload). A 32–36 GB Mac works
> but is slow (~5 min/turn) and needs the Metal wired-limit raised — see the docs.

Each run writes `result.json` (citable, self-describing), `report.txt` (human-readable, with
the remediation cards), and a share `card.svg`/`card.png` + `badge.svg` (skip with `--no-card`).

### Run profiles

- `--judges local` (**default**) — the offline fine-tuned judge (`aipsy-judge-1.0`) served
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
- `--baseline-prompt` — inject the research baseline system prompt to reproduce the published
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
| `--judges local` (**default**) | **none** — `judge pull` is token-free (public, ungated repo) |
| `--judges single` | `OPENAI_API_KEY` |
| `--judges gold` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` |

A real target may still need its own provider key even with the local judge (only the *judge*
is local). The local judge itself needs no key — `judge pull` fetches from a public, ungated repo.

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

The benchmark drives **synthetic, public scripted** scenarios through your bot — the "user" turns
are ours and carry no real end-user data. What differs by lane is **where your bot's replies get
scored**:

- **Local judge (`--judges local`, the default) — fully in your boundary.** Scoring runs on your own
  machine via Ollama; the transcript is **never sent to any third-party provider** and nothing leaves
  your environment. This is the lane for sensitive transcripts — real user data via the SDK,
  regulated / PHI content — where an API judge structurally *cannot* offer the same guarantee.
- **Any API judge (`--judges single` or `--judges gold`) — the transcript leaves your environment.**
  Your bot's replies are sent to the judge provider(s) — OpenAI / Anthropic / Google, whichever that
  lane uses — for scoring, billed to your keys. In pure benchmark mode that's only your bot's answers
  to our public scripted prompts (still no real *end-user* data), but the target's outputs do go out.

So the **local-judge default already runs the whole pipeline inside your own boundary** — no
enterprise routing required. (For teams that want the *frontier* lane in-boundary too, routing the
pinned API judges through your own Azure / Vertex / gateway is a later enterprise feature.)

## Development

```bash
uv run pytest          # full suite, offline, no API keys
uv run ruff check .    # lint
```

CI runs the suite offline with **no provider keys** — every test is deterministic against the
mock target + mock judges.

## Licenses

- **Code:** Apache-2.0 (see [`LICENSE`](LICENSE)).
- **Data** (`data/v1/` — the scenarios, rubric, and judge prompt): **CC BY 4.0**
  (see [`data/v1/DATA_LICENSE`](data/v1/DATA_LICENSE)) — the standard content license, and the
  same one the forthcoming methodology preprint will carry.
- **Local judge model** ([`keidolabs/aipsy-judge-1.0`](https://huggingface.co/keidolabs/aipsy-judge-1.0),
  on Hugging Face): Apache-2.0, inherited from its Gemma-4 base model's terms.

---

<div align="center">
<sub><b>aipsy-bench</b> — built and maintained by <a href="https://www.keidolabs.com">Keido Labs</a><br/>
<a href="LICENSE">Apache-2.0</a> (code) · <a href="data/v1/DATA_LICENSE">CC BY 4.0</a> (data) · directional until the human-validation study lands</sub>
</div>
