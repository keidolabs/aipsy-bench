"""Step 07 — clinical_judge_panel Scorer, full Solver→Scorer run. Offline."""

from __future__ import annotations

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from aipsy_bench import spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue


def _broken_judge():
    """A judge that always returns unparseable output (real models occasionally do)."""
    def _outputs(messages, tools, tool_choice, config):
        return ModelOutput.from_content(model="mockllm/judge", content="sorry, no JSON here at all.")
    return get_model("mockllm/model", custom_outputs=_outputs)


def _erroring_judge():
    """A judge whose call RAISES a provider error (e.g. a 400/timeout after retries)."""
    def _outputs(messages, tools, tool_choice, config):
        raise RuntimeError("400 BadRequest: temperature and top_p cannot both be specified")
    return get_model("mockllm/model", custom_outputs=_outputs)


def test_judge_config_omits_top_p_for_anthropic():
    # Deviation #4: Anthropic rejects temperature+top_p together for claude-sonnet-4-6.
    from aipsy_bench.scorer import _judge_config

    a = _judge_config("anthropic", 120, 3, None)
    assert a.top_p is None
    assert a.temperature == spec.JUDGE_TEMPERATURE
    assert _judge_config("openai", 120, 3, None).top_p == 1.0
    g = _judge_config("google", 120, 3, None)
    assert g.top_p == 1.0
    assert g.reasoning_tokens == spec.GEMINI_THINKING_BUDGET
    assert _judge_config("openai", 120, 3, 4).max_connections == 4


def test_judge_provider_error_degrades_not_crashes(tmp_path):
    # a judge whose call raises (provider 4xx/5xx/timeout) must degrade like a parse
    # failure — recorded, that turn dropped, battery continues, never crashes.
    task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="single", judges={"openai": _erroring_judge()}, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none",
                       log_dir=str(tmp_path / "logs"), fail_on_error=False)[0]
    assert log.status == "success"
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.metadata["judge_failures"]
    assert all(score.value[m] is None for m in spec.METRICS)

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


def test_na_reducer_avoids_convert_warning_and_preserves_na(tmp_path, monkeypatch):
    # the default epoch reducer floats N/A None → 0.0 and logs "Unable to convert value
    # to float: None". Our aipsy_na_mean reducer keeps N/A as None and never warns.
    import inspect_ai.scorer._metric as _metric

    from aipsy_bench.task import aipsy_bench

    warns: list[str] = []
    monkeypatch.setattr(_metric.logger, "warning", lambda msg, *a, **k: warns.append(str(msg)))

    log = inspect_eval(
        aipsy_bench(target="mock", judges="gold", scenario_ids=["s01"]),
        model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"),
    )[0]
    assert not [w for w in warns if "Unable to convert" in w]
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.value["crisis_handling"] is None  # N/A preserved, not coerced to 0.0
    assert isinstance(score.value["empathy"], float)


def test_judge_parse_failure_degrades_not_crashes(tmp_path):
    # the only judge always fails to parse → the run COMPLETES (no crash); failures are
    # recorded; the affected turns are unscored (None), never scored as unsafe (§ quality bar).
    task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="single", judges={"openai": _broken_judge()}, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success"  # did NOT crash the battery
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.metadata["judge_failures"]  # recorded
    assert all(score.value[m] is None for m in spec.METRICS)  # unscored, not low
    assert score.value["AI_Trust"] is None


def test_gold_degrades_when_one_judge_fails(tmp_path):
    # one broken judge + two good → the ensemble still scores from the two good ones
    judges = {"openai": mock_judge_model(), "anthropic": mock_judge_model(), "google": _broken_judge()}
    task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="gold", judges=judges, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.value["empathy"] is not None  # scored from the 2 working judges
    assert any(jf["judge"] == "google" for jf in score.metadata["judge_failures"])
    assert len(score.metadata["per_turn"][0]["per_judge"]) == 2  # broken judge dropped for the turn


def test_adversarial_target_not_inflated(tmp_path):
    # §13.7: an injection target must not produce 5/5 — scores reflect the rubric.
    samples = _run(tmp_path, scenario_ids=["s01"], target="adversarial")
    score = samples["s01"].scores["clinical_judge_panel"]
    numeric = [v for k, v in score.value.items() if isinstance(v, float)]
    assert numeric  # there are numeric metrics
    assert all(v < 5.0 for v in numeric)
