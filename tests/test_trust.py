"""Step 16 — explain + inter-judge disagreement + self-preference flag. Offline."""

from __future__ import annotations

import json

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from aipsy_bench import spec, trust
from aipsy_bench.cli import main
from aipsy_bench.dataset import build_dataset
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation


def _judge_with(empathy):
    """A judge that scores empathy at a fixed value (to create inter-judge spread)."""
    def _outputs(messages, tools, tool_choice, config):
        obj = {
            "context_analysis": {},
            "scores": {
                "empathy": empathy, "tone_consistency": 4,
                "crisis_detected": False, "crisis_severity": "none", "crisis_handling": "NA",
                "advice_safety": 4, "boundary_safety": 4, "affective_complexity": 4,
            },
            "reasoning": f"empathy judged {empathy} by this judge",
        }
        return ModelOutput.from_content(model="mockllm/judge", content="```json\n" + json.dumps(obj) + "\n```")
    return get_model("mockllm/model", custom_outputs=_outputs)


def _gold_divergent_log(tmp_path, scenario_ids=("s01",)):
    judges = {"openai": _judge_with(5), "anthropic": _judge_with(4), "google": _judge_with(1)}
    task = Task(
        dataset=build_dataset(scenario_ids=list(scenario_ids)),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="gold", judges=judges, cache=False),
    )
    return inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]


def test_explain_contains_reasoning_anchor_and_target(tmp_path):
    log = _gold_divergent_log(tmp_path)
    out = trust.explain(log, "s01", 5)
    assert "explain s01 turn 5" in out
    assert "judge α=PENDING" in out
    assert "empathy judged 5" in out and "empathy judged 1" in out  # each judge's reasoning
    assert "rubric anchor" in out
    assert "glad you told me" in out  # quoted target text
    assert "Reproduce: aipsy-bench run --scenario s01" in out


def test_explain_prints_no_numeric_agreement(tmp_path):
    log = _gold_divergent_log(tmp_path)
    out = trust.explain(log, "s01", 5)
    assert "α=PENDING" in out
    assert "α=0." not in out  # never a numeric agreement while PENDING (§0.3)


def test_disagreement_surfaced_in_result(tmp_path):
    log = _gold_divergent_log(tmp_path)
    result = to_result_json(log, validation=load_validation())
    dis = result["judge_disagreement"]
    assert dis
    top = dis[0]
    assert top["metric"] == "empathy"
    assert top["spread"] == 4.0  # 5 - 1
    assert sorted(top["scores"]) == [1.0, 4.0, 5.0]


def test_self_preference_flag():
    jv = {"openai": "gpt-5.4-mini", "anthropic": "claude-sonnet-4-6", "google": "gemini-2.5-flash"}
    assert trust.self_preference("openai/gpt-4o", jv) == ["openai"]
    assert trust.self_preference("anthropic/claude-3", jv) == ["anthropic"]
    assert trust.self_preference("mock", jv) == []


def test_self_preference_in_result_and_warning(tmp_path):
    judges = {p: mock_judge_model() for p in spec.PROVIDERS}
    task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="gold", judges=judges, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    # override the target ref to an openai-family model → openai judge is a confound
    result = to_result_json(log, validation=load_validation(),
                            target={"adapter": "model", "ref": "openai/gpt-4o", "model_snapshot": "gpt-4o"})
    assert "openai" in result["self_preference"]
    assert any("self-preference" in w for w in result["warnings"])


def test_cli_explain(tmp_path, capsys):
    main(["run", "--target", "mock", "--scenario", "s06", "--out", str(tmp_path / "r"),
          "--display", "none", "--no-card", "--judges", "single"])
    rc = main(["explain", "s06", "5", "--out", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "explain s06 turn 5" in out
