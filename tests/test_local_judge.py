"""Local Ollama judge — the offline, self-contained DEFAULT panel (exp 016).

Fully offline: the live `/api/chat` call is replaced by an injected `transport`, so
no Ollama server (and no network) is needed. These tests pin the 016 inference
contract (system+user roles, num_ctx/seed/top_p/num_predict, think=False) and the
`local` comparability lane (board eligibility, cross-lane refusal, card/banner).
"""

from __future__ import annotations

import json

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import card, leaderboard, spec
from aipsy_bench.compare import CompareError, compare
from aipsy_bench.dataset import build_dataset
from aipsy_bench.local_judge import (
    LocalJudgeUnavailable,
    _default_transport,
    local_judge_model,
)
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation


# --------------------------------------------------------------------------
# Helpers — a fake Ollama transport returning valid, parseable judge JSON
# --------------------------------------------------------------------------
def _safe_judge_response() -> dict:
    obj = {
        "context_analysis": {"note": "fake local judge"},
        "reasoning": "the reply is warm and safe",
        "scores": {
            "crisis_detected": False, "crisis_severity": "none",
            "empathy": 5, "tone_consistency": 5, "crisis_handling": "NA",
            "advice_safety": 5, "boundary_safety": 4, "affective_complexity": 4,
        },
    }
    return {"message": {"content": "```json\n" + json.dumps(obj) + "\n```"}}


def _capturing_transport(captured: list):
    def transport(url, payload, timeout):
        captured.append({"url": url, "payload": payload, "timeout": timeout})
        return _safe_judge_response()
    return transport


def _run_local(tmp_path, *, scenario_ids, transport=None, target="safe"):
    transport = transport or _capturing_transport([])
    judge = local_judge_model(transport=transport)
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="local", judges={"local": judge}, cache=False),
    )
    return inspect_eval(task, model=mock_target_model(target), display="none",
                        log_dir=str(tmp_path / "logs"))[0]


# --------------------------------------------------------------------------
# The 016 inference contract
# --------------------------------------------------------------------------
def test_request_shape_matches_016_contract(tmp_path):
    captured: list = []
    _run_local(tmp_path, scenario_ids=["s01"], transport=_capturing_transport(captured))
    assert captured, "the local judge transport was called"
    p = captured[-1]["payload"]

    assert p["model"] == spec.LOCAL_JUDGE_TAG
    assert p["think"] is False
    assert p["stream"] is False
    assert p["keep_alive"] == spec.LOCAL_JUDGE_KEEP_ALIVE  # stays resident across the battery
    # system + user roles (the gemma4 renderer handles the system message) — NOT merged
    assert [m["role"] for m in p["messages"]] == ["system", "user"]

    sys_msg, user_msg = p["messages"][0]["content"], p["messages"][1]["content"]
    assert len(sys_msg) > 1000                          # the full frozen rubric is intact
    assert "SECURITY NOTE" in sys_msg                   # the §13.7 injection fence is kept
    assert "TARGET MESSAGE TO EVALUATE" in user_msg     # the formatted conversation input

    opts = p["options"]
    assert opts["num_ctx"] == spec.LOCAL_JUDGE_NUM_CTX == 8192
    assert opts["seed"] == spec.LOCAL_JUDGE_SEED
    assert opts["top_p"] == 1.0
    assert opts["temperature"] == spec.JUDGE_TEMPERATURE
    assert opts["num_predict"] == spec.JUDGE_MAX_TOKENS


def test_default_transport_raises_clean_unavailable():
    # a closed local port → URLError → a friendly LocalJudgeUnavailable, not a raw stack
    with pytest.raises(LocalJudgeUnavailable):
        _default_transport("http://127.0.0.1:1/api/chat", {"model": "x", "messages": []}, 1)


# --------------------------------------------------------------------------
# The `local` comparability lane
# --------------------------------------------------------------------------
def test_local_panel_records_lane(tmp_path):
    log = _run_local(tmp_path, scenario_ids=["s01"])
    score = log.samples[0].scores["clinical_judge_panel"]
    assert score.metadata["judge_panel"] == "local"
    assert score.metadata["judge_versions"] == {"local": spec.LOCAL_JUDGE_VERSION}
    assert score.value["empathy"] == 5  # scored from the (fake) local judge, not mock-guessed


def test_result_json_has_local_banner(tmp_path):
    log = _run_local(tmp_path, scenario_ids=["s01"])
    result = to_result_json(log, validation=load_validation())
    assert result["judge_panel"] == "local"
    assert any("LOCAL JUDGE" in w for w in result["warnings"])
    assert result["gate"]["mode"] == "directional"


def test_local_full_battery_is_board_eligible(tmp_path):
    log = _run_local(tmp_path, scenario_ids=None)  # full s01–s20
    result = to_result_json(log, validation=load_validation())
    assert leaderboard.is_board_eligible(result) is True
    row = leaderboard.row_from_result(result)
    assert row.judge_panel == "local"  # the row carries the lane tag


def test_compare_refuses_local_vs_gold(tmp_path):
    local_log = _run_local(tmp_path / "l", scenario_ids=["s01"])
    gold_task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(
            panel="gold", judges={p: mock_judge_model() for p in spec.PROVIDERS}, cache=False),
    )
    gold_log = inspect_eval(gold_task, model=mock_target_model("safe"),
                            display="none", log_dir=str(tmp_path / "g"))[0]
    with pytest.raises(CompareError):
        compare(local_log, gold_log)


def test_against_board_refuses_cross_lane(tmp_path):
    log = _run_local(tmp_path, scenario_ids=["s01", "s06"])
    result = to_result_json(log, validation=load_validation())
    text = leaderboard.render_against_board(result)
    assert "No local-lane baselines" in text
    assert "gpt-5.4-mini" not in text  # the gold-lane rows are NOT overlaid on a local run


def test_card_and_badge_show_local(tmp_path):
    log = _run_local(tmp_path, scenario_ids=["s01"])
    result = to_result_json(log, validation=load_validation())
    svg, _png = card.render_card(result)
    assert "LOCAL JUDGE" in svg
    assert "--judges local" in card.reproduce_command(result)
    assert "local" in card.render_badge(result)


# --------------------------------------------------------------------------
# Status / onboarding (no server, no network)
# --------------------------------------------------------------------------
def test_status_shape_without_server(monkeypatch):
    from aipsy_bench import local_judge

    monkeypatch.setattr(local_judge, "_get_json", lambda *a, **k: None)
    st = local_judge.status()
    assert st["running"] is False and st["model_present"] is False
    assert st["tag"] == spec.LOCAL_JUDGE_TAG
    assert st["version"] == spec.LOCAL_JUDGE_VERSION
    assert st["quant"] == spec.LOCAL_JUDGE_QUANT


def test_ensure_model_falls_back_to_manual(monkeypatch):
    from aipsy_bench import local_judge

    monkeypatch.setattr(local_judge, "ollama_running", lambda: False)
    res = local_judge.ensure_model(log=lambda *_a, **_k: None)
    assert res["status"] == "manual"


def test_warm_up_sends_keep_alive_with_long_timeout(monkeypatch):
    from aipsy_bench import local_judge

    captured = {}

    def fake(url, payload, timeout):
        captured["payload"], captured["timeout"] = payload, timeout
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(local_judge, "_default_transport", fake)
    assert local_judge.warm_up() is True
    assert captured["payload"]["keep_alive"] == spec.LOCAL_JUDGE_KEEP_ALIVE
    assert captured["payload"]["options"]["num_ctx"] == spec.LOCAL_JUDGE_NUM_CTX
    assert captured["timeout"] == spec.LOCAL_JUDGE_TIMEOUT  # the generous cold-load floor


def test_warm_up_false_when_unavailable(monkeypatch):
    from aipsy_bench import local_judge

    def boom(url, payload, timeout):
        raise local_judge.LocalJudgeUnavailable("down")

    monkeypatch.setattr(local_judge, "_default_transport", boom)
    assert local_judge.warm_up() is False
