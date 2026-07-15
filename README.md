<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/keidolabs-logo-dark.svg">
  <img alt="Keido Labs" src="assets/keidolabs-logo.svg" width="190">
</picture>

# aipsy-bench

**Open-source psychological-safety benchmark for conversational AI**

Point a chatbot at a frozen battery of clinical scenarios, score the transcripts with a frozen
panel of LLM judges, and ship a **CI pass/fail gate + a clinician-grade diagnostic** — which
turns failed, why, and what to tune.

[![CI](https://github.com/keidolabs/aipsy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/keidolabs/aipsy-bench/actions/workflows/ci.yml)
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

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — gives you `uvx` (run with
   zero install). *(Prefer `pip`? You only need Python 3.12+.)*
2. **One way to score transcripts** — pick either:
   - 🖥️ **[Ollama](https://ollama.com/download)** — runs the **default local judge**, 100% offline,
     no API key (needs a **48 GB+ Mac** or a **16 GB-VRAM + 64 GB-RAM Linux box**).
   - 🔑 **A provider API key** — **OpenAI**, **Anthropic**, or **Google** — for the frontier judge
     lane (`--judges gold`/`single`; your key, your cost).

## Quickstart

**See what it catches — 10 seconds, nothing to install, fully offline** (no API keys, no
network, no setup):

```bash
uvx aipsy-bench demo
```

Replays two real recorded runs of the *same* coaching app on two different backing models — an
older one and a newer one, scored by the *same* judge — and renders the head-to-head. On a crisis
scenario, swapping the model under the app flips the safety gate from **FAIL** (AI-Trust 2.37) to
**PASS** (4.02). Same product, swap the LLM, safety moves — the kind of regression `aipsy-bench`
catches in CI. (Directional single-judge replay, so it's illustrative, not a comparable score.)

**Then run the engine yourself** (offline self-test: mock bot + mock judges):

```bash
uvx aipsy-bench run --target mock --quick
```

**Install once — then it's just `aipsy-bench …`** (no per-command prefixes):

```bash
uv tool install aipsy-bench             # one install — local judge + every provider, no extras to pick
aipsy-bench judge pull                  # one-time: fetch the local judge (needs Ollama)
aipsy-bench init --http-target http://localhost:3000/eval   # scaffold config for your bot
aipsy-bench run --quick                 # go
```

> **Frontier judges** instead of the local default? Nothing extra to install — set your provider key
> (`aipsy-bench keys set`, stored in a gitignored `.env`) and pass `--judges single` (or `gold`).
> Prefer classic pip? `pip install aipsy-bench` is identical.

Each run writes `result.json` (citable), `report.txt` (with the remediation cards), and a share
`card.svg`/`card.png` + `badge.svg` (skip with `--no-card`).

> **Benchmarking your own app?** The only integration you write is a thin, stateless `/eval`
> endpoint — see the [adapter guides](docs/adapters/README.md): a hand-written cookbook, an
> [AI-agent build prompt](docs/adapters/eval-endpoint-agent-guide.md), or a Tier-2 callable.

> **Which judge?** The **local** judge is the offline default; **frontier** (`--judges gold`/`single`,
> provider keys) is a *different instrument* — a local score compares to other local runs only,
> never to gold. Setup + hardware: [docs/local-judge.md](docs/local-judge.md).

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
