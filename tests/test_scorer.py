"""Step 07 — clinical_judge_panel Scorer, full Solver→Scorer run. Offline."""

from __future__ import annotations

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue

_PANEL_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _run(tmp_path, *, scenario_ids, target="safe", panel="single"):
    judges = {p: mock_judge_model() for p in _PANEL_PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    logs = inspect_eval(
        task, model=mock_target_model(target), display="none", log_dir=str(tmp_path / "logs")
    )
    assert logs[0].status == "success"
    return {s.id: s for s in logs[0].samples}


def test_full_run_produces_scenario_score(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s01", "s07"])
    for sid in ("s01", "s07"):
        score = samples[sid].scores["clinical_judge_panel"]
        for m in spec.METRICS:
            assert m in score.value
        assert "AI_Trust" in score.value
        assert "crisis_detected_any" in score.value
        meta = score.metadata
        assert len(meta["per_turn"]) == 10
        assert set(meta["phase_breakdown"].keys()) == {"1", "2", "3", "4"}


def test_no_crisis_yields_na_crisis_handling(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s01"])
    score = samples["s01"].scores["clinical_judge_panel"]
    assert score.value["crisis_handling"] is None
    assert score.value["crisis_detected_any"] is False
    for pt in score.metadata["per_turn"]:
        assert pt["ensemble"]["crisis_handling"] is None


def test_crisis_yields_numeric_crisis_handling(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s07"])
    score = samples["s07"].scores["clinical_judge_panel"]
    assert score.value["crisis_detected_any"] is True
    assert score.value["crisis_handling"] is not None
    crisis_turns = [pt for pt in score.metadata["per_turn"] if pt["crisis_detected"]]
    assert crisis_turns
    assert all(pt["ensemble"]["crisis_handling"] is not None for pt in crisis_turns)


def test_single_panel_uses_one_judge(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s01"], panel="single")
    score = samples["s01"].scores["clinical_judge_panel"]
    assert score.metadata["judge_panel"] == "single"
    assert list(score.metadata["judge_versions"]) == ["openai"]
    assert len(score.metadata["per_turn"][0]["per_judge"]) == 1


def test_gold_panel_uses_three_judges(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s01"], panel="gold")
    score = samples["s01"].scores["clinical_judge_panel"]
    assert score.metadata["judge_panel"] == "gold"
    assert set(score.metadata["judge_versions"]) == set(spec.PROVIDERS)
    pj = score.metadata["per_turn"][0]["per_judge"]
    assert len(pj) == 3
    # provenance: resolved snapshot recorded per judge call (§8)
    assert all(j["model_snapshot"] for j in pj)


def test_failing_target_fails_gate(tmp_path):
    samples = _run(tmp_path, scenario_ids=["s01"], target="failing")
    score = samples["s01"].scores["clinical_judge_panel"]
    from aipsy_bench import scoring
    metric_scores = {m: score.value[m] for m in spec.METRICS}
    assert scoring.is_failure(metric_scores, score.value["AI_Trust"], scoring.rubric_thresholds())


def test_adversarial_target_not_inflated(tmp_path):
    # §13.7: an injection target must not produce 5/5 — scores reflect the rubric.
    samples = _run(tmp_path, scenario_ids=["s01"], target="adversarial")
    score = samples["s01"].scores["clinical_judge_panel"]
    numeric = [v for k, v in score.value.items() if isinstance(v, float)]
    assert numeric  # there are numeric metrics
    assert all(v < 5.0 for v in numeric)
