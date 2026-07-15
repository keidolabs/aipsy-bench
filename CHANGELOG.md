# Changelog

All notable changes to **aipsy-bench** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

> **Comparability note (benchmark-specific).** A score's meaning depends on the tool version, the
> `data/vN` content, **and** the judge instrument. Every release states its **`data_version`** and
> flags any change to the **frozen content** (scenarios / rubric / judge prompt) or the **judge
> instrument** (frontier pins / local-judge version) — those affect score comparability across
> versions, unlike tool/DX changes. A content change is a `data/` version bump by definition.

## [Unreleased]

### TODO (CI maintenance, non-blocking)
- Bump GitHub Actions off Node 20 before the runners drop it: `actions/checkout@v4` and
  `astral-sh/setup-uv@v5` are being force-run on Node 24 with a deprecation warning (seen in the
  0.1.4 `publish.yml` run). Move both to their latest majors in `ci.yml` + `publish.yml`; fold into
  the next release.

## [0.1.5] — 2026-07-16

Patch — a new offline `demo` command. **`data_version`: v1 (unchanged)**; no frozen-content or
judge-instrument change, and the score and gate are untouched, so results stay fully comparable to
0.1.2–0.1.4.

### Added
- **`aipsy-bench demo` — a fully offline, zero-setup first run.** `uvx aipsy-bench demo` replays two
  real recorded runs of the same app on two different backing models (an older and a newer one),
  scored by the same single frontier judge, and renders the head-to-head: on the s06 crisis
  scenario, swapping the model flips the safety gate **FAIL → PASS** (AI-Trust 2.37 → 4.02). No API
  keys, no network, no Ollama — it reads two bundled `.eval` logs through the same `report`/`compare`
  code a real run uses, so nothing is faked or hardcoded. It saves both `report.html` reports + a
  head-to-head card under `aipsy-run/demo/` and prints their paths. The demo is **fully generic**:
  the app and the two model names are undisclosed and the bundled logs are scrubbed of any identity;
  it shows the *mechanism*, not a named claim. Directional single-scenario/single-judge replay
  (correctly non-comparable, not board/card eligible), so `demo` always exits 0 — the gate FAIL is
  the story, not a process failure. The two ~37 KB demo logs live under `aipsy_bench/demo_assets/`
  (tool assets, **not** the frozen `data/v1` bundle — shipping them is not a `data/` bump).

## [0.1.4] — 2026-07-12

Patch — report clarity only. **`data_version`: v1 (unchanged)**; no frozen-content or
judge-instrument change, and **the score and the gate are untouched**, so results stay fully
comparable to 0.1.2/0.1.3.

### Added
- **Conservative read on the safety-critical axes.** The report (`report.txt`, `report.html`) and
  `result.json` (`safety_conservative`) now surface, for `crisis_handling` / `advice_safety` /
  `boundary_safety`, the **harshest single judge's** pooling (named) alongside the frozen
  equal-weight ensemble mean, flagging where the mean clears a critical threshold but the harshest
  judge would not. Gold panels only (needs ≥2 judges). This is **descriptive, never gated** — it
  reveals the safety-tail leniency that equal-weight pooling can hide (companion judge paper),
  while the gold score itself is unchanged. The remedy the analysis points to — conservative
  pooling (min, not mean) on these axes — moves the frozen numbers, so it is deferred to a future
  `data/` version rather than retrofitted into v1 (it would perturb the pre-registered instrument
  mid-validation). The printed `ensemble_mean` equals `scores.overall` exactly (it qualifies the
  frozen number, never restates a different one).

## [0.1.3] — 2026-07-05

Patch — DX only. **`data_version`: v1 (unchanged)**; no frozen-content or judge-instrument change,
so scores stay fully comparable to 0.1.2.

### Fixed
- `doctor` no longer exits non-zero — reading as a red "failure" in the terminal — when the
  **default** local judge isn't set up but the target is healthy and a usable lane exists (an API
  key is present, or the box is hardware-viable). It's now an advisory. An explicit `--judges local`
  that isn't ready still fails.

## [0.1.2] — 2026-07-05

Initial release. **`data_version`: v1** (SHA-256 verified at runtime) · **judges:** local FT
`aipsy-judge-1.0` (default, offline) + frozen frontier pins (openai / anthropic / google) ·
**posture:** DIRECTIONAL — a recommendation, not yet human-validated (§0.3); no per-metric
agreement (α) is claimed. Follows a rigorous internal QA pass across two hardware profiles.

### Added
- **One-install setup:** `uvx aipsy-bench` (zero-install run) or `uv tool install aipsy-bench` /
  `pip install aipsy-bench` — a single install includes the local judge and every frontier
  provider; no extras to choose. `doctor`/`init` steer you to the right lane at runtime.
- Psychological-safety benchmark on **Inspect AI**: 20 frozen clinical scenarios (s01–s20) + v2
  rubric + judge prompt, scored to an **AI-Trust** composite with a functional CI pass/fail gate.
- **Judge lanes:** `local` (offline FT judge via Ollama — the default), `single[:provider]` (one
  frontier judge, the key you hold; directional), `gold` (3-judge frontier ensemble — the
  comparable/citable panel). Lanes are namespaced and never cross-compare.
- **Targets:** Tier-0 model string, Tier-1 HTTP `/eval` endpoint (the zero-Python app-dev path),
  Tier-2 Python callable. A target failure is a RUN FAILURE, never a low safety score (§6).
- **`report.html`** — self-contained, offline, pytest-style report (alongside `result.json` +
  `report.txt`), with a prominent **SELF-JUDGING** alert when the judge and target share a provider.
- **Hardware-aware path recommendation** in `doctor` / `init` (local vs API), platform-aware
  (Apple-Silicon unified memory vs discrete-GPU VRAM).
- **Role-clear `doctor`** (separate Target + Judge readiness across all combinations) and
  role-aware `keys status`.
- Runs **accumulate** by default (`aipsy-run/<timestamp>/` + a `latest` pointer + a ready-to-paste
  `compare`), so you can diff old-vs-improved without flags.
- CLI: `init` scaffold, `doctor` preflight (+ endpoint connectivity probe), `keys` helper,
  `run`, `compare`, `explain`, share card/badge, `judge pull` / `warm` / `status`.

## [0.1.0] — unreleased (internal)

Internal development baseline — the full build plus an extensive pre-release QA + DX hardening pass
across two hardware profiles (model-string validation, target/judge key parity, run-failure
diagnostics, the hidden-and-saved key prompt, VRAM detection, and more). Never published;
`0.1.1` was a TestPyPI validation upload of the release candidate.

<!-- Release process: work accrues under [Unreleased]; on publish, rename it to the version +
     date and add compare/tag links below. -->
[Unreleased]: https://github.com/keidolabs/aipsy-bench/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/keidolabs/aipsy-bench/releases/tag/v0.1.5
[0.1.4]: https://github.com/keidolabs/aipsy-bench/releases/tag/v0.1.4
[0.1.3]: https://github.com/keidolabs/aipsy-bench/releases/tag/v0.1.3
[0.1.2]: https://github.com/keidolabs/aipsy-bench/releases/tag/v0.1.2
