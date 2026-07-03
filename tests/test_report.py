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
    assert (tmp_path / "out" / "report.html").exists()
    assert written["data_version"] == "v1"
    assert written["mode"] == "benchmark"


# ---- HTML report (§4.3) — a derived, offline, self-contained diagnostic view ----

def test_html_report_structure_and_content(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01", "s07"])
    result = report.to_result_json(log, validation=load_validation())
    doc = report.render_html(result)
    assert doc.startswith("<!doctype html>")
    assert doc.rstrip().endswith("</html>")
    # the directional honesty guard is carried over, prominently
    assert "DIRECTIONAL" in doc
    assert "recommendation, not a rubber-stamp" in doc
    # every metric is rendered
    for m in spec.METRICS:
        assert m in doc
    assert "AI-Trust" in doc


def test_html_report_no_validated_alpha_number(tmp_path):
    # honesty guard: α is never a number in HTML either (PENDING only, §0.3)
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    result = report.to_result_json(log, validation=load_validation())
    doc = report.render_html(result)
    assert "α=" not in doc or "α=PENDING" in doc
    # gate failed → the verdict banner says so
    assert "GATE FAILED" in doc


def test_html_report_is_self_contained_and_offline(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"])
    result = report.to_result_json(log, validation=load_validation())
    doc = report.render_html(result)
    # zero JS, no external assets, no telemetry — opening it must never phone home
    assert "<script" not in doc.lower()
    assert 'src=' not in doc
    assert "<link" not in doc.lower()
    assert "cdn" not in doc.lower()
    assert "googleapis" not in doc.lower()
    assert "http://" not in doc  # no insecure refs
    # the ONLY outbound URLs are the Keido Labs footer links (no CDN/telemetry)
    assert doc.count("https://") == doc.count("https://www.keidolabs.com")
    assert "https://www.keidolabs.com/contact" in doc


def test_html_report_has_subtle_keido_branding(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"])
    result = report.to_result_json(log, validation=load_validation())
    doc = report.render_html(result)
    assert "Built by" in doc and "Keido Labs" in doc
    # the CTA links out (never an auto-submit / tracking mechanic, §12/§13.6)
    assert "keidolabs.com" in doc
    # branding lives in the footer, separated from the verdict — not stamped on the score
    footer = doc[doc.index("<footer"):]
    assert "Keido Labs" in footer


def test_html_report_escapes_untrusted_target_ref(tmp_path):
    # target ref / judge reasoning are untrusted (§13.3) — must be HTML-escaped
    log = _run(tmp_path, scenario_ids=["s01"])
    xss = '<script>alert("pwn")</script>'
    result = report.to_result_json(
        log, validation=load_validation(),
        target={"adapter": "http", "ref": xss, "model_snapshot": "x"},
    )
    doc = report.render_html(result)
    assert xss not in doc
    assert "&lt;script&gt;" in doc


def test_html_report_deterministic(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01", "s07"])
    result = report.to_result_json(log, validation=load_validation())
    assert report.render_html(result) == report.render_html(result)
