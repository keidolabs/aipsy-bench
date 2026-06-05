"""Flatten the immutable Inspect ``.eval`` log into a citable ``result.json`` +
a human-readable report (§4.3). The ``.eval`` log itself stays the source of
record — this is the derived, CI/human-friendly view.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from . import spec
from .gate import evaluate_gate
from .validation import JudgeValidation, provisional_banner

DATA_VERSION = "v1"
SCORER_NAME = "clinical_judge_panel"
_NUMERIC_KEYS = (*spec.METRICS, "AI_Trust")


def _na_mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None


def _scores(sample) -> dict:
    return sample.scores[SCORER_NAME].value


def _meta(sample) -> dict:
    return sample.scores[SCORER_NAME].metadata


def _overall(samples) -> dict:
    out = {k: _na_mean([_scores(s).get(k) for s in samples]) for k in _NUMERIC_KEYS}
    out["crisis_detected_any"] = any(_scores(s).get("crisis_detected_any") for s in samples)
    return out


def _by_domain(samples) -> dict:
    domains: dict[str, list] = {}
    for s in samples:
        domains.setdefault(_meta(s)["domain"], []).append(s)
    return {d: {k: _na_mean([_scores(s).get(k) for s in subs]) for k in _NUMERIC_KEYS}
            for d, subs in domains.items()}


def _by_phase(samples) -> dict:
    phases = ("1", "2", "3", "4")
    out: dict[str, dict] = {}
    for p in phases:
        rows = [_meta(s)["phase_breakdown"].get(p) for s in samples]
        rows = [r for r in rows if r]
        if not rows:
            continue
        out[p] = {
            "name": spec.PHASE_NAMES.get(int(p), p),
            "AI_Trust": _na_mean([r.get("composite") for r in rows]),
            **{m: _na_mean([r.get("metrics", {}).get(m) for r in rows]) for m in spec.METRICS},
        }
    return out


def _by_scenario(samples) -> dict:
    return {s.id: {k: _scores(s).get(k) for k in (*_NUMERIC_KEYS, "crisis_detected_any")}
            for s in samples}


def _target_block(log, override: dict | None) -> dict:
    if override:
        return override
    model = log.eval.model or ""
    adapter = "mock" if model.startswith("mockllm") else "model"
    return {"adapter": adapter, "ref": model, "model_snapshot": model}


def to_result_json(
    log,
    *,
    validation: JudgeValidation,
    target: dict | None = None,
    mode: str = "benchmark",
    gate: dict | None = None,
    extra_warnings: list[str] | None = None,
) -> dict:
    """Flatten an Inspect ``.eval`` log to the §4.3 result.json schema."""
    samples = log.samples
    panel = _meta(samples[0])["judge_panel"] if samples else "single"
    judge_versions = _meta(samples[0])["judge_versions"] if samples else {}
    by_scenario = {s.id: _scores(s) for s in samples}

    warnings: list[str] = list(extra_warnings or [])
    if panel == "single":
        warnings.append("single-judge panel — scores NOT comparable to published gold numbers")
    banner = provisional_banner(validation)
    if banner:
        warnings.append(banner)

    diagnostics = [
        {"scenario_id": s.id, "card": s.scores[SCORER_NAME].explanation}
        for s in samples
        if s.scores[SCORER_NAME].explanation and "no metric failed" not in s.scores[SCORER_NAME].explanation
    ]

    gate = gate if gate is not None else evaluate_gate(by_scenario, validation)

    return {
        "tool": "aipsy-bench",
        "tool_version": __version__,
        "data_version": DATA_VERSION,
        "mode": mode,
        "run_id": log.eval.run_id,
        "timestamp": log.eval.created,
        "target": _target_block(log, target),
        "judge_panel": panel,
        "judge_versions": judge_versions,
        "judge_validation": validation.model_dump(),
        "scores": {
            "overall": _overall(samples),
            "by_domain": _by_domain(samples),
            "by_phase": _by_phase(samples),
            "by_scenario": _by_scenario(samples),
        },
        "gate": gate,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def render_report(result_json: dict) -> str:
    """Human-readable text report; prints the §0.3 provisional banner prominently."""
    lines: list[str] = []
    banner = next((w for w in result_json["warnings"] if w.startswith("⚠ PROVISIONAL")), None)
    if banner:
        bar = "═" * 78
        lines += [bar, banner, bar, ""]

    t = result_json["target"]
    lines.append(f"aipsy-bench {result_json['tool_version']} · data {result_json['data_version']} · mode {result_json['mode']}")
    lines.append(f"target: {t['ref']} ({t['adapter']})  ·  judges: {result_json['judge_panel']}")
    lines.append("")

    lines.append("Overall scores:")
    overall = result_json["scores"]["overall"]
    for k in _NUMERIC_KEYS:
        lines.append(f"  {k:<22} {_fmt(overall.get(k))}")
    lines.append("")

    gate = result_json["gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    lines.append(f"Gate: {status}" + (f"  ({gate['note']})" if gate.get("note") else ""))
    for f in gate["failures"]:
        lines.append(f"  ✗ {f['scenario']} {f['metric']} {_fmt(f['value'])} < {_fmt(f['threshold'])}")
    lines.append("")

    if result_json["diagnostics"]:
        lines.append("Diagnostics:")
        lines.append("")
        for d in result_json["diagnostics"]:
            lines.append(d["card"])
            lines.append("")

    for w in result_json["warnings"]:
        if not w.startswith("⚠ PROVISIONAL"):
            lines.append(f"warning: {w}")
    return "\n".join(lines)


def write_artifacts(log, out_dir: str | Path, *, validation: JudgeValidation, **kwargs) -> dict:
    """Write ``result.json`` + ``report.txt`` to ``out_dir``; return the result dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = to_result_json(log, validation=validation, **kwargs)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    (out / "report.txt").write_text(render_report(result))
    return result
