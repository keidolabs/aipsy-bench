"""Step 10 — gate pass/fail + the validation guard. Offline."""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.gate import assert_gate, gate_result
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import JudgeValidation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _run(tmp_path, *, scenario_ids, target, panel="single"):
    judges = {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    return inspect_eval(
        task, model=mock_target_model(target), display="none", log_dir=str(tmp_path / "logs")
    )[0]


def _validated(licensed, descriptive_only=()):
    return JudgeValidation(status="VALIDATED", licensed=list(licensed),
                           descriptive_only=list(descriptive_only))


def test_directional_gate_fails_unsafe_by_default(tmp_path):
    # Launch posture: the gate is functional (directional). An unsafe target fails
    # the build now — no validation artifact needed (§0.3).
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    res = gate_result(log)
    assert res["passed"] is False
    assert res["gate_eligible"] is True
    assert res["mode"] == "directional"
    assert any(f["metric"] == "crisis_handling" for f in res["failures"])
    with pytest.raises(AssertionError):
        assert_gate(log)  # directional gate raises on an unsafe target


def test_directional_gate_passes_safe_by_default(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], target="safe")
    res = gate_result(log)
    assert res["passed"] is True
    assert res["gate_eligible"] is True
    assert res["mode"] == "directional"
    assert_gate(log)  # no raise


def test_licensed_below_threshold_fails(tmp_path):
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    v = _validated(["crisis_handling"])
    res = gate_result(log, v)
    assert res["passed"] is False
    assert any(f["metric"] == "crisis_handling" for f in res["failures"])
    with pytest.raises(AssertionError):
        assert_gate(log, validation=v)


def test_descriptive_only_metric_cannot_fail(tmp_path):
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    # the same failing crisis_handling, but only empathy is licensed (and empathy 2
    # is not below its 2.0 critical) → the guard means no gate failure
    v = _validated(["empathy"], descriptive_only=["crisis_handling", "advice_safety"])
    res = gate_result(log, v)
    assert res["passed"] is True


def test_assert_gate_prints_remediation_cards(tmp_path):
    log = _run(tmp_path, scenario_ids=["s07"], target="failing")
    v = _validated(["crisis_handling"])
    with pytest.raises(AssertionError) as exc:
        assert_gate(log, validation=v)
    assert "crisis-resource hand-off" in str(exc.value)  # the §5 card is in the message


def test_gate_passes_for_safe_licensed(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], target="safe")
    v = _validated(list(spec.METRICS) + ["AI_Trust"])
    assert gate_result(log, v)["passed"] is True


def test_quick_subset_is_one_per_domain_plus_crisis():
    from aipsy_bench import bundle
    from aipsy_bench.task import quick_scenario_ids

    ids = quick_scenario_ids()
    assert set(spec.CRISIS_SCENARIO_IDS).issubset(ids)
    assert len(ids) == 5  # 3 domain firsts + s06 + s07 (both mental_health)
    domains = {s.domain for s in bundle.load_scenarios() if s.id in ids}
    assert domains == {"mental_health", "companion", "coaching"}


def test_gold_panel_three_judges(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], target="safe", panel="gold")
    meta = log.samples[0].scores["clinical_judge_panel"].metadata
    assert len(meta["per_turn"][0]["per_judge"]) == 3
