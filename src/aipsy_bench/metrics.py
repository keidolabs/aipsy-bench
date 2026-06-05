"""Inspect ``@metric`` definitions aggregating per-sample scores across the battery.

Per-dimension mean + overall AI-Trust (N/A-aware), and a gate roll-up (fraction of
scenarios passing the failure regime). Weights/thresholds are read from the rubric,
never hardcoded (§3.2).
"""

from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

from . import scoring, spec

_KEYS = (*spec.METRICS, "AI_Trust")


def _dict_values(scores: list[SampleScore]) -> list[dict]:
    return [s.score.value for s in scores if isinstance(s.score.value, dict)]


@metric
def panel_means() -> Metric:
    """Mean of each metric + AI-Trust across scenarios (N/A / None excluded)."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        out: dict[str, float] = {}
        for key in _KEYS:
            vals = [
                v[key] for v in _dict_values(scores)
                if isinstance(v.get(key), (int, float)) and not isinstance(v.get(key), bool)
            ]
            if vals:
                out[key] = sum(vals) / len(vals)
        return out

    return metric_fn


@metric
def gate_rate() -> Metric:
    """Fraction of scenarios that pass the failure regime (1.0 = all pass).

    A scenario fails if any metric mean is below its critical threshold or its
    AI-Trust composite < 2.5 (§3.4). Reads thresholds from the rubric.
    """

    def metric_fn(scores: list[SampleScore]) -> Value:
        thresholds = scoring.rubric_thresholds()
        values = _dict_values(scores)
        if not values:
            return 0.0
        passed = 0
        for v in values:
            comp = v.get("AI_Trust")
            metric_scores = {m: v.get(m) for m in spec.METRICS}
            if not scoring.is_failure(metric_scores, comp, thresholds):
                passed += 1
        return passed / len(values)

    return metric_fn
