# aipsy-bench

An **open-source psychological-safety benchmark for conversational AI**. Point it at a
chatbot, run a frozen battery of clinical scenarios through the bot, score each transcript
with a frozen panel of LLM judges against a frozen clinical rubric, and get a **CI pass/fail
gate plus a clinician-grade diagnostic** — which turns failed, why, and what to tune.

It is a **benchmark** (fixed content, comparable scores), not a framework. The content — 20
scenarios + a 6-metric rubric + the judge prompt — is verbatim from the 014 research
experiment. The engine is [Inspect AI](https://inspect.aisi.org.uk/); we ship a Task, a
Solver, and a Scorer, not a new runner.

> ## ⚠ PROVISIONAL — instrument not yet human-validated
> The premise is "the judge panel is validated against human clinical experts." That
> validation (014's human-vs-judge agreement study) is **in progress**. Until its gate
> artifact lands, aipsy-bench runs and scores freely but every score is marked
> **provisional**: scores are *descriptive only*, **no per-metric agreement number is
> claimed**, and **no metric can fail the CI gate** (a metric gates the build only once 014
> has licensed it). Do not cite these as validated agreement.

## Quickstart

```bash
uv sync

# Fully offline self-test — no API keys, mock target + mock judges:
uv run aipsy-bench run --target mock --quick

# A real foundational model (needs the provider key, e.g. ANTHROPIC_API_KEY):
uv run aipsy-bench run --model anthropic/claude-sonnet-4-6 --judges single --quick

# The full, comparable run (3-judge gold panel, all 20 scenarios):
uv run aipsy-bench run --model openai/gpt-5.4-mini --judges gold

# Browse the public benchmark content:
uv run aipsy-bench scenarios list
```

Each run writes `result.json` (citable, self-describing), `report.txt` (human-readable, with
the remediation cards), and a share `card.svg`/`card.png` + `badge.svg` (skip with `--no-card`).

### Run profiles

- `--judges single` (default) — the primary judge only. Fast inner loop for iterating on a
  prompt. **NOT comparable to published gold numbers.**
- `--judges gold` — the 3-judge ensemble. Comparable, citable, drives the badge/leaderboard.
- `--quick` — smoke subset (one scenario per domain + both crisis scenarios). Directional
  only; never feeds a card or the leaderboard.
- `--scenario s06,s07` — run a subset.
- `--baseline-prompt` — inject the 014 baseline system prompt to reproduce the published
  frontier baseline. **By default the target keeps its own system prompt** (the bot as
  deployed); aipsy-bench sends only the scripted user turns.

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
