"""Step 15 — run --against-board overlay + domain boards. Offline, no network."""

from __future__ import annotations

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import leaderboard
from aipsy_bench.cli import main
from aipsy_bench.dataset import build_dataset
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation


def _quick_result(tmp_path):
    task = Task(
        dataset=build_dataset(scenario_ids=["s01", "s06", "s07", "s09", "s15"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel="single", judges={"openai": mock_judge_model()}, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    return to_result_json(log, validation=load_validation())


def test_snapshot_is_marked_sample():
    snap = leaderboard.load_board_snapshot()
    assert snap["_sample"] is True
    assert "SAMPLE" in snap["_note"]
    assert {r["name"] for r in snap["rows"]} >= {"gpt-5.4-mini", "claude-sonnet-4-6", "gemini-2.5-flash"}


def test_overlay_table_and_caveat(tmp_path):
    result = _quick_result(tmp_path)
    text = leaderboard.render_against_board(result)
    assert "against-board" in text
    assert "YOU: mock" in text
    assert "gpt-5.4-mini" in text and "claude-sonnet-4-6" in text
    assert "INSTRUMENT" in text  # the §6 instrument-vs-conditions caveat


def test_domain_filter(tmp_path):
    result = _quick_result(tmp_path)
    text = leaderboard.render_against_board(result, domain="companion")
    assert "domain=companion" in text
    # the overlay still lists the vanilla baselines for that domain
    assert "claude-sonnet-4-6" in text


def test_cli_against_board_offline(tmp_path, capsys):
    rc = main([
        "run", "--target", "mock", "--quick", "--against-board",
        "--out", str(tmp_path / "run"), "--display", "none", "--no-card",
    ])
    out = capsys.readouterr().out
    assert "against-board" in out
    assert "YOU: mock" in out
    assert rc == 0
