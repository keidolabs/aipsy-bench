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
