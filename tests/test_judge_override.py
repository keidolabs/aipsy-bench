"""Judge override (testing-only judge swap) — non-comparable, never on the board (§8)."""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import leaderboard, spec
from aipsy_bench.cli import _parse_judge_overrides, main
from aipsy_bench.dataset import build_dataset
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation


def test_parse_judge_overrides():
    assert _parse_judge_overrides(None) is None
    assert _parse_judge_overrides(["anthropic=claude-haiku-4-5"]) == {"anthropic": "claude-haiku-4-5"}
    assert _parse_judge_overrides(["anthropic=claude-haiku-4-5", "openai=gpt-x"]) == {
        "anthropic": "claude-haiku-4-5", "openai": "gpt-x"}
    for bad in (["anthropic"], ["bogus=x"], ["anthropic="]):
        with pytest.raises(ValueError):
            _parse_judge_overrides(bad)


def test_board_excludes_overridden_run():
    base = {
        "mode": "benchmark", "judge_panel": "gold", "run_failures": [], "judge_failures": [],
        "incomplete": False, "warnings": [],
        "scores": {"by_scenario": {f"s{n:02d}": {} for n in range(1, 21)}},
    }
    assert leaderboard.is_board_eligible(base) is True  # otherwise-eligible full gold run
    base["judge_overrides"] = {"anthropic": {"from": "claude-sonnet-4-6", "to": "claude-haiku-4-5"}}
    assert leaderboard.is_board_eligible(base) is False  # swap → not the frozen instrument


def test_override_surfaced_in_report(tmp_path):
    judges = {p: mock_judge_model() for p in spec.PROVIDERS}
    task = Task(
        dataset=build_dataset(scenario_ids=["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="gold", judges=judges, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    # the real get_model swap needs network/keys; simulate the recorded override metadata
    for s in log.samples:
        s.scores["clinical_judge_panel"].metadata["judge_overrides"] = {
            "anthropic": {"from": "claude-sonnet-4-6", "to": "claude-haiku-4-5"}}
    result = to_result_json(log, validation=load_validation())
    assert result["judge_overrides"]
    assert any("judge override active" in w for w in result["warnings"])
    assert leaderboard.is_board_eligible(result) is False


def test_cli_rejects_bad_override(tmp_path, capsys):
    rc = main(["run", "--target", "mock", "--scenario", "s01", "--judge-override", "bogus=x",
               "--out", str(tmp_path / "r"), "--display", "none", "--no-card"])
    assert rc == 2
    assert "judge-override" in capsys.readouterr().err
