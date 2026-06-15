"""Step 11 — share card, badge, head-to-head, board.json schema. Offline."""

from __future__ import annotations

from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import card, leaderboard, report, spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _result(tmp_path, *, scenario_ids=None, target="safe", panel="single", extra_warnings=None):
    judges = {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model(target), display="none", log_dir=str(tmp_path / "logs"))[0]
    return report.to_result_json(log, validation=load_validation(), extra_warnings=extra_warnings or [])


def test_render_card_svg_and_png(tmp_path):
    result = _result(tmp_path, scenario_ids=["s01", "s07"])
    svg, png = card.render_card(result)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    at = result["scores"]["overall"]["AI_Trust"]
    assert f"{at:.1f}" in svg
    assert "DIRECTIONAL" in svg  # directional banner until 014 lands (§0.3)
    assert "aipsy-bench run --model" in svg  # reproduce command
    assert "og:title" in svg  # OG meta present
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 100


def test_card_is_deterministic(tmp_path):
    result = _result(tmp_path, scenario_ids=["s01"])
    a_svg, a_png = card.render_card(result)
    b_svg, b_png = card.render_card(result)
    assert a_svg == b_svg
    assert a_png == b_png


def test_render_badge(tmp_path):
    result = _result(tmp_path, scenario_ids=["s01"])
    badge = card.render_badge(result)
    at = result["scores"]["overall"]["AI_Trust"]
    assert badge.startswith("<svg")
    assert f"AI-Trust {at:.1f}" in badge


def test_render_head_to_head(tmp_path):
    a = _result(tmp_path / "a", scenario_ids=["s01"], target="safe")
    b = _result(tmp_path / "b", scenario_ids=["s01"], target="failing")
    svg, png = card.render_head_to_head(a, b)
    assert svg.startswith("<svg")
    assert a["target"]["ref"] in svg or "mock" in svg
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_board_row_schema_and_eligibility(tmp_path):
    # a full-battery gold run is eligible
    gold = _result(tmp_path / "g", scenario_ids=None, panel="gold")
    assert leaderboard.is_board_eligible(gold) is True
    rows = leaderboard.board_from_results([gold])
    assert len(rows) == 1
    row = rows[0]
    assert set(row.scores) == {*spec.METRICS, "AI_Trust"}
    assert row.judge_panel == "gold"
    assert row.judge_validation_status == "DIRECTIONAL"


def test_single_and_partial_runs_excluded_from_board(tmp_path):
    single = _result(tmp_path / "s", scenario_ids=None, panel="single")
    assert leaderboard.is_board_eligible(single) is False

    partial = _result(
        tmp_path / "p", scenario_ids=["s01", "s07"], panel="gold",
        extra_warnings=["partial battery (quick/subset) — directional only"],
    )
    assert leaderboard.is_board_eligible(partial) is False
    assert leaderboard.board_from_results([single, partial]) == []
