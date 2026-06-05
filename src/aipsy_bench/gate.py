"""CI gate evaluation (§7). Core scenario-level pass/fail, honoring the §0.3 /
§7 validation guard: a ``descriptive_only`` / PENDING metric can NEVER fail the
build — it is reported, not gated.

Step 10 layers ``assert_gate`` (the pytest surface) + CLI wiring on this core.
"""

from __future__ import annotations

from . import scoring, spec
from .validation import JudgeValidation, load_validation

SCORER_NAME = "clinical_judge_panel"


def evaluate_gate(
    by_scenario: dict[str, dict],
    validation: JudgeValidation,
    *,
    thresholds: dict[str, float] | None = None,
    ai_trust_target: float | None = None,
) -> dict:
    """Evaluate the gate over per-scenario score values.

    A scenario fails if a *gate-eligible* metric mean is below its critical
    threshold, or its AI-Trust mean is below the target. While 014 validation is
    PENDING nothing is gate-eligible, so the gate passes vacuously (advisory).
    """
    thresholds = thresholds if thresholds is not None else scoring.rubric_thresholds()
    target = ai_trust_target if ai_trust_target is not None else thresholds.get("ai_trust_target")

    failures: list[dict] = []
    for sid, vals in by_scenario.items():
        for m in spec.METRICS:
            if not validation.is_gateable(m):
                continue
            v = vals.get(m)
            crit = thresholds.get(f"{m}_critical")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and crit is not None and v < crit:
                failures.append({
                    "scenario": sid, "metric": m, "value": v,
                    "threshold": crit, "kind": "metric_below_critical",
                })
        if validation.is_gateable("AI_Trust"):
            at = vals.get("AI_Trust")
            if isinstance(at, (int, float)) and not isinstance(at, bool) and target is not None and at < target:
                failures.append({
                    "scenario": sid, "metric": "AI_Trust", "value": at,
                    "threshold": target, "kind": "ai_trust_below_target",
                })

    gate_eligible = any(validation.is_gateable(m) for m in (*spec.METRICS, "AI_Trust"))
    return {
        "passed": len(failures) == 0,
        "gate_eligible": gate_eligible,
        "thresholds": {
            **{f"{m}_critical": thresholds.get(f"{m}_critical") for m in spec.METRICS},
            "ai_trust_target": target,
        },
        "failures": failures,
        "note": (
            None if gate_eligible
            else "advisory only — instrument PENDING_VALIDATION, no metric is gate-eligible yet (§0.3)"
        ),
    }


# --------------------------------------------------------------------------
# Log-driven surfaces (the pytest gate + CLI roll-up)
# --------------------------------------------------------------------------
def _by_scenario(log) -> dict[str, dict]:
    return {s.id: s.scores[SCORER_NAME].value for s in log.samples}


def _override_thresholds(overrides: dict) -> tuple[dict | None, float | None]:
    """Map kwargs like ``crisis_handling=2.5, AI_Trust=3.5`` onto the rubric keys."""
    if not overrides:
        return None, None
    thresholds = scoring.rubric_thresholds().copy()
    target = None
    for key, val in overrides.items():
        if key in ("AI_Trust", "ai_trust_target"):
            target = float(val)
        elif key in spec.METRICS:
            thresholds[f"{key}_critical"] = float(val)
        elif key.endswith("_critical"):
            thresholds[key] = float(val)
        else:
            raise ValueError(f"unknown gate threshold {key!r}")
    return thresholds, target


def gate_result(log, validation: JudgeValidation | None = None, **overrides) -> dict:
    """Evaluate the gate from an Inspect ``.eval`` log (honors the validation guard)."""
    validation = validation if validation is not None else load_validation()
    thresholds, target = _override_thresholds(overrides)
    return evaluate_gate(_by_scenario(log), validation, thresholds=thresholds, ai_trust_target=target)


def assert_gate(log, *, validation: JudgeValidation | None = None, **thresholds) -> None:
    """pytest surface: assert the gate passes; on failure print the §5 remediation
    cards and raise ``AssertionError``. A descriptive_only/PENDING metric can never
    fail here (the validation guard)."""
    result = gate_result(log, validation, **thresholds)
    if result["passed"]:
        return

    from .diagnostics import build_all_cards  # local import avoids a cycle

    failed_scenarios = {f["scenario"] for f in result["failures"]}
    cards: list[str] = []
    for s in log.samples:
        if s.id in failed_scenarios:
            cards.extend(build_all_cards(s.scores[SCORER_NAME].metadata))
    summary = ", ".join(
        f"{f['scenario']}:{f['metric']}={f['value']:.2f}<{f['threshold']:.2f}"
        for f in result["failures"]
    )
    raise AssertionError("aipsy-bench gate FAILED — " + summary + "\n\n" + "\n\n".join(cards))
