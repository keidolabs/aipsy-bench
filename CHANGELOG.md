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

_Nothing yet._

## [0.1.2] — _pending (stamp date on PyPI publish)_

Initial release. **`data_version`: v1** (SHA-256 verified at runtime) · **judges:** local FT
`gemma4-judge-ft-v3` (default, offline) + frozen frontier pins (openai / anthropic / google) ·
**posture:** DIRECTIONAL — a recommendation, not yet human-validated (§0.3); no per-metric
agreement (α) is claimed. Follows a rigorous internal QA pass across two hardware profiles.

### Added
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
[Unreleased]: https://github.com/drKeeman/aipsy-bench/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/drKeeman/aipsy-bench/releases/tag/v0.1.2
