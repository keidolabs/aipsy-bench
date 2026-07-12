"""Judge transparency & trust surface (§15) — the anti-"rigged-judge" moat.

Turns the judge from a black box into the most-inspectable part of the tool, built
**deterministically** from material the Scorer already captured (per-judge scores +
reasoning + rubric anchors) — no new model call.

While 014 validation is PENDING (§0.3) nothing here prints a numeric agreement; the
judge-agreement field reads ``PENDING``. ``export --for-rating`` and the full
self-preference treatment are iteration 3 — this ships ``explain``, inter-judge
disagreement, and a minimal self-preference flag.
"""

from __future__ import annotations

from . import bundle, spec

SCORER_NAME = "clinical_judge_panel"
JUDGE_ALPHA_DISPLAY = "PENDING"

# Family tokens used to detect a judge/target self-preference confound.
_FAMILY_TOKENS = {
    "openai": ("openai", "gpt"),
    "anthropic": ("anthropic", "claude"),
    "google": ("google", "gemini"),
}


def _anchor_guides() -> dict[str, dict[str, str]]:
    return {d["id"]: d.get("scoring_guide", {}) for d in bundle.load_rubric()["dimensions"]}


def _find_turn(log, scenario_id: str, turn: int) -> tuple[dict, dict]:
    for s in log.samples:
        meta = s.scores[SCORER_NAME].metadata
        if meta["scenario_id"] != scenario_id:
            continue
        for pt in meta.get("per_turn", []):
            if pt["turn"] == turn:
                return meta, pt
    raise KeyError(f"no scored turn {turn} for scenario {scenario_id}")


def _fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "N/A"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _na_mean(values) -> float | None:
    nums = [v for v in values if _is_num(v)]
    return sum(nums) / len(nums) if nums else None


def _meta(sample) -> dict:
    return sample.scores[SCORER_NAME].metadata


def _judge_val(pt: dict, provider: str, metric: str) -> float | None:
    """One judge's numeric score for ``metric`` on this turn, or None (N/A / absent)."""
    for j in pt.get("per_judge", []):
        if j["judge"] == provider:
            v = j.get("scores", {}).get(metric)
            return v if _is_num(v) else None
    return None


def explain(log, scenario_id: str, turn: int) -> str:
    """Drill into one score: each judge's reasoning, the assigned rubric anchor band
    per metric, and the quoted target text. Deterministic from captured data."""
    meta, pt = _find_turn(log, scenario_id, turn)
    guides = _anchor_guides()
    phase = spec.PHASE_NAMES.get(pt["phase"], str(pt["phase"]))

    lines = [
        f"explain {scenario_id} turn {turn} ({phase} phase) · judge α={JUDGE_ALPHA_DISPLAY}",
        f'target: "{pt.get("ai_text", "")}"',
        "",
        "ensemble: " + " | ".join(
            f"AI_Trust {_fmt(pt.get('composite'))}" if i == 0 else f"{m} {_fmt(pt['ensemble'].get(m))}"
            for i, m in enumerate(("AI_Trust", *spec.METRICS))
        ),
        "",
        "rubric anchors (ensemble band):",
    ]
    for m in spec.METRICS:
        v = pt["ensemble"].get(m)
        if v is None:
            continue
        band = max(0, min(5, int(v)))
        anchor = guides.get(m, {}).get(str(band), "")
        if anchor:
            lines.append(f'  {m} {band}/5: "{anchor}"')

    lines += ["", "judges:"]
    for j in pt.get("per_judge", []):
        sc = j.get("scores", {})
        score_str = " ".join(f"{m}={_fmt(sc.get(m))}" for m in spec.METRICS)
        lines.append(f"  {j['judge']}: {score_str}")
        if j.get("reasoning"):
            lines.append(f"    reasoning: {j['reasoning']}")

    lines += ["", f"Reproduce: aipsy-bench run --scenario {scenario_id}"]
    return "\n".join(lines)


def judge_disagreement(samples, *, min_spread: float = 2.0, top_n: int = 10) -> list[dict]:
    """Per-metric / per-turn inter-judge spread (gold only). Surfacing the spread is
    the thing critics assume we hide — so we don't (§15)."""
    items: list[dict] = []
    for s in samples:
        meta = s.scores[SCORER_NAME].metadata
        for pt in meta.get("per_turn", []):
            for m in spec.METRICS:
                vals = [
                    j["scores"].get(m) for j in pt.get("per_judge", [])
                    if isinstance(j["scores"].get(m), (int, float)) and not isinstance(j["scores"].get(m), bool)
                ]
                if len(vals) < 2:
                    continue
                spread = max(vals) - min(vals)
                if spread >= min_spread:
                    items.append({
                        "scenario_id": meta["scenario_id"], "turn": pt["turn"],
                        "metric": m, "scores": vals, "spread": spread,
                    })
    items.sort(key=lambda x: x["spread"], reverse=True)
    return items[:top_n]


def conservative_safety_read(samples, *, thresholds: dict | None = None) -> list[dict]:
    """Descriptive harshest-judge read on the safety-critical axes (report layer, §0.3).

    The gold score pools the three frontier judges by equal-weight mean (§6.1). The
    companion judge paper shows that is not neutral on the failure tail: one judge is a
    systematic lenient outlier, so equal-weight pooling blends its leniency into crisis /
    advice / boundary, leaving the mean biased optimistic — a *passing* safety-axis mean is
    weaker evidence of safety than a failing one is of harm, so the gate can under-flag.

    This surfaces, per safety-critical axis, the **harshest single judge's** battery-level
    pooling (named) next to the frozen ensemble mean and the critical line, flagging where
    the mean clears the line but the harshest judge would not. It is **purely descriptive**:
    it never changes the score or the gate. The remedy the paper points to — pool
    conservatively (min, not mean) on these axes — moves the frozen numbers, so it is
    deferred to a future data version (it would perturb the pre-registered instrument
    mid-validation); here we only reveal what the mean already elides.

    Both the ensemble mean and each judge's number use the SAME nested aggregation the
    gated value uses (mean over scenarios of the mean over that scenario's turns, N/A-aware),
    so ``ensemble_mean`` matches ``scores.overall`` and the two poolings are comparable.
    Gold panels only (needs ≥2 judges); returns ``[]`` otherwise.
    """
    from . import scoring

    thresholds = thresholds if thresholds is not None else scoring.rubric_thresholds()

    # Judges present across the battery, ordered by spec.PROVIDERS for stable naming.
    seen = {j["judge"] for s in samples for pt in _meta(s).get("per_turn", []) for j in pt.get("per_judge", [])}
    providers = [p for p in spec.PROVIDERS if p in seen]
    if len(providers) < 2:
        return []

    out: list[dict] = []
    for m in spec.SAFETY_CRITICAL_METRICS:
        crit = thresholds.get(f"{m}_critical")

        # Each judge's battery mean, aggregated exactly like the ensemble (per-scenario
        # mean over turns, then mean over scenarios), N/A-aware.
        per_judge_battery: dict[str, float | None] = {}
        for p in providers:
            scenario_means = [
                sm for s in samples
                if (sm := _na_mean([_judge_val(pt, p, m) for pt in _meta(s).get("per_turn", [])])) is not None
            ]
            per_judge_battery[p] = _na_mean(scenario_means)

        # The frozen, gated ensemble mean (matches report._overall for this metric).
        ensemble_mean = _na_mean([
            sm for s in samples
            if (sm := _na_mean([pt["ensemble"].get(m) for pt in _meta(s).get("per_turn", [])])) is not None
        ])

        graded = [(p, v) for p, v in per_judge_battery.items() if v is not None]
        if not graded:
            out.append({"metric": m, "ensemble_mean": ensemble_mean, "harshest_judge": None,
                        "harshest_value": None, "critical": crit, "would_flag": False})
            continue
        harshest_judge, harshest_value = min(graded, key=lambda kv: kv[1])
        would_flag = (
            crit is not None and harshest_value < crit
            and (ensemble_mean is None or ensemble_mean >= crit)
        )
        out.append({
            "metric": m,
            "ensemble_mean": ensemble_mean,
            "harshest_judge": harshest_judge,
            "harshest_value": harshest_value,
            "critical": crit,
            "would_flag": would_flag,
        })
    return out


def self_preference(target_ref: str, judge_versions: dict) -> list[str]:
    """Flag judges whose model family intersects the target's family (a confound).

    Judges favor their own family; when the target and a judge share one, that row's
    score carries a self-preference confound (§15). Minimal flag here; the full honest
    treatment + blind re-rating sheet is iteration 3.
    """
    ref = (target_ref or "").lower()
    flagged = []
    for provider in judge_versions:
        tokens = _FAMILY_TOKENS.get(provider, ())
        if any(tok in ref for tok in tokens):
            flagged.append(provider)
    return flagged
