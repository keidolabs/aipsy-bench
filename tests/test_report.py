"""Step 09 — result.json + report rendering + validation block. Offline."""

from __future__ import annotations

import json

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import report, spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import JudgeValidation, load_validation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _run(tmp_path, *, scenario_ids, target="safe", panel="single"):
    judges = {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    return inspect_eval(
        task, model=mock_target_model(target), display="none", log_dir=str(tmp_path / "logs")
    )[0]


_REQUIRED_KEYS = {
    "tool", "tool_version", "data_version", "mode", "run_id", "timestamp",
    "target", "judge_panel", "judge_versions", "judge_validation",
    "scores", "gate", "diagnostics", "warnings",
}


def test_result_json_schema_and_directional(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01", "s07"])
    result = report.to_result_json(log, validation=load_validation())
    assert _REQUIRED_KEYS.issubset(result.keys())
    assert set(result["scores"]) == {"overall", "by_domain", "by_phase", "by_scenario"}
    jv = result["judge_validation"]
    assert jv["status"] == "DIRECTIONAL"
    # directional launch posture: the full frozen rubric gates; nothing forced descriptive_only
    assert jv["descriptive_only"] == []
    assert jv["licensed"] == []
    # honesty guard: no numeric validated agreement is claimed yet
    assert jv["per_metric_alpha"] == {}


def test_single_panel_warning(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], panel="single")
    result = report.to_result_json(log, validation=load_validation())
    assert any("NOT comparable to published gold numbers" in w for w in result["warnings"])


def test_report_has_directional_banner_no_numbers(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"])
    result = report.to_result_json(log, validation=load_validation())
    text = report.render_report(result)
    assert "DIRECTIONAL" in text
    assert "not yet human-validated" in text.lower() or "not yet a human-validated" in text.lower()
    assert "recommendation, not a rubber-stamp" in text
    # honesty guard: no numeric per-metric agreement value anywhere (α stays PENDING)
    assert "α=" not in text or "α=PENDING" in text


def test_gate_directional_fails_unsafe_by_default(tmp_path):
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    result = report.to_result_json(log, validation=load_validation())
    # launch posture: a failing target fails the directional gate now (§0.3)
    assert result["gate"]["passed"] is False
    assert result["gate"]["gate_eligible"] is True
    assert result["gate"]["mode"] == "directional"
    assert result["gate"]["failures"]


def test_is_gateable_directional_by_default():
    v = load_validation()  # DIRECTIONAL
    for m in (*spec.METRICS, "AI_Trust"):
        assert v.is_gateable(m) is True


def test_validated_metric_gates():
    v = JudgeValidation(
        status="VALIDATED", licensed=["empathy"], descriptive_only=["crisis_handling"],
        per_metric_alpha={"empathy": 0.71},
    )
    assert v.is_gateable("empathy") is True
    assert v.is_gateable("crisis_handling") is False


def test_write_artifacts(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01", "s07"])
    result = report.write_artifacts(log, tmp_path / "out", validation=load_validation())
    written = json.loads((tmp_path / "out" / "result.json").read_text())
    assert written["run_id"] == result["run_id"]
    assert (tmp_path / "out" / "report.txt").exists()
    assert written["data_version"] == "v1"
    assert written["mode"] == "benchmark"
