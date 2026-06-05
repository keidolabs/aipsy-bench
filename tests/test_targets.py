"""Step 13 — full target adapter: Tier-1 HTTP, Tier-2 callable, session mode,
target-failure-is-run-failure. Offline (HTTP via injected transport)."""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from aipsy_bench import leaderboard, spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.targets import callable_target, classify_outcome, http_target
from aipsy_bench.validation import load_validation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _run(tmp_path, target, *, scenario_ids, conversation="stateless", panel="single", judges=None):
    judge_models = judges or {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(conversation=conversation),
        scorer=clinical_judge_panel(panel=panel, judges=judge_models, cache=False),
    )
    return inspect_eval(task, model=target.model, display="none", log_dir=str(tmp_path / "logs"))[0]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def test_classify_outcome():
    assert classify_outcome(ModelOutput.from_content("m", "hi")) == "ok"
    assert classify_outcome(ModelOutput.from_content("m", "", error="boom")) == "target_error"
    assert classify_outcome(ModelOutput.from_content("m", "   ")) == "empty"
    assert classify_outcome(ModelOutput.from_content("m", "hi", stop_reason="max_tokens")) == "truncated"
    assert classify_outcome(ModelOutput.from_content("m", "x", stop_reason="content_filter")) == "refusal"


# --------------------------------------------------------------------------
# Tier 2 callable + session vs stateless
# --------------------------------------------------------------------------
def test_callable_target_drives_scenario(tmp_path):
    seen = []

    def fn(messages):
        seen.append(len(messages))
        return "I hear you — that sounds really hard, and I'm glad you reached out."

    log = _run(tmp_path, callable_target(fn), scenario_ids=["s01"])
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.value["empathy"] is not None  # scored normally
    assert seen[0] < seen[-1]  # stateless: full history grows each turn


def test_session_sends_only_new_turn(tmp_path):
    seen = []

    def fn(messages):
        seen.append(messages)
        return "thanks for telling me, I'm here."

    target = callable_target(fn, conversation="session")
    log = _run(tmp_path, target, scenario_ids=["s01"], conversation="session")
    assert all(len(m) == 1 and m[0]["role"] == "user" for m in seen)
    result = to_result_json(log, validation=load_validation())
    assert result["conversation"] == "session"


def test_stateless_records_mode(tmp_path):
    log = _run(tmp_path, callable_target(lambda m: "a kind reply"), scenario_ids=["s01"])
    result = to_result_json(log, validation=load_validation())
    assert result["conversation"] == "stateless"


# --------------------------------------------------------------------------
# Tier 1 HTTP (injected transport — offline)
# --------------------------------------------------------------------------
def test_http_target_drives_scenario(tmp_path):
    captured = {}

    def transport(url, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["n_messages"] = len(payload["messages"])
        return {"reply": "thank you for sharing that, it sounds genuinely tough."}

    target = http_target("https://bot/eval", headers={"Authorization": "Bearer x"}, transport=transport)
    log = _run(tmp_path, target, scenario_ids=["s01"])
    assert log.samples[0].scores["clinical_judge_panel"].value["empathy"] is not None
    assert captured["url"] == "https://bot/eval"
    assert captured["headers"]["Authorization"] == "Bearer x"


def test_http_parse_shapes():
    from aipsy_bench.targets import _parse_reply
    assert _parse_reply({"reply": "yo"}) == "yo"
    assert _parse_reply({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    with pytest.raises(ValueError):
        _parse_reply({"nope": 1})


# --------------------------------------------------------------------------
# Target failure = run failure (never a low safety score)
# --------------------------------------------------------------------------
def _exploding_judge():
    def _outputs(messages, tools, tool_choice, config):
        raise AssertionError("judge must NOT be called on a run-failure scenario")
    return get_model("mockllm/model", custom_outputs=_outputs)


def _raises(_messages):
    raise TimeoutError("rate limited at this turn")


@pytest.mark.parametrize("fn,status", [(_raises, "target_error"), (lambda m: "", "empty")])
def test_target_failure_is_run_failure(tmp_path, fn, status):
    target = callable_target(fn)
    # the judge would raise if called — proving it is NOT called on a failed scenario
    log = _run(tmp_path, target, scenario_ids=["s01"], judges={"openai": _exploding_judge()})
    assert log.status == "success"  # structured failure, not a crash
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.value["run_failure"] is True
    assert all(score.value[m] is None for m in spec.METRICS)  # NOT a low score
    assert score.value["AI_Trust"] is None
    assert score.metadata["per_turn"] == []  # judge never ran
    assert score.metadata["run_failure"]["status"] == status


def test_run_failure_fails_gate_and_excluded_from_board(tmp_path):
    target = callable_target(lambda m: "")  # empty → run failure
    log = _run(tmp_path, target, scenario_ids=["s01"], judges={"openai": _exploding_judge()})
    result = to_result_json(log, validation=load_validation())
    assert result["run_failures"]
    # a run failure fails the gate even while validation is PENDING (it is incompleteness)
    assert result["gate"]["passed"] is False
    assert any(f["kind"] == "run_failure" for f in result["gate"]["failures"])
    assert leaderboard.is_board_eligible(result) is False
