"""Step 18 — opt-in publish-card (no network) + cite. Offline."""

from __future__ import annotations

import urllib.request

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import __version__, cite, leaderboard, spec
from aipsy_bench.cli import main
from aipsy_bench.dataset import build_dataset
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _result(tmp_path, *, scenario_ids=None, panel="gold", extra_warnings=None):
    judges = {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    log = inspect_eval(task, model=mock_target_model("safe"), display="none", log_dir=str(tmp_path / "logs"))[0]
    return to_result_json(log, validation=load_validation(), extra_warnings=extra_warnings or [])


def test_publish_eligible_makes_no_network_call(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("publish must NOT make a network call (§12)")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    result = _result(tmp_path)  # full gold battery
    out = leaderboard.publish_card_bundle(result, tmp_path / "pub")
    assert (tmp_path / "pub" / "card.svg").exists()
    assert (tmp_path / "pub" / "card.png").exists()
    assert (tmp_path / "pub" / "board_row.json").exists()
    assert (tmp_path / "pub" / "POST.md").exists()
    assert out["row"].judge_panel == "gold"


def test_publish_refuses_single_panel(tmp_path):
    result = _result(tmp_path, scenario_ids=["s01"], panel="single")
    with pytest.raises(leaderboard.PublishRefused):
        leaderboard.publish_card_bundle(result, tmp_path / "pub", render=False)


def test_publish_refuses_partial_battery(tmp_path):
    result = _result(tmp_path, scenario_ids=["s01"], panel="gold",
                     extra_warnings=["partial battery (quick/subset) — directional only"])
    with pytest.raises(leaderboard.PublishRefused):
        leaderboard.publish_card_bundle(result, tmp_path / "pub", render=False)


def test_cite_bibtex():
    bib = cite.bibtex()
    assert "4gu6d" in bib
    assert "aipsybench" in bib
    assert __version__ in bib
    assert spec.DATA_VERSION in bib


def test_cli_publish_card_full_gold(tmp_path):
    out = tmp_path / "r"
    main(["run", "--target", "mock", "--judges", "gold", "--out", str(out), "--display", "none", "--no-card"])
    rc = main(["publish-card", "--run", str(out)])
    assert rc == 0
    assert (out / "publish" / "card.svg").exists()


def test_cli_publish_card_refuses_quick(tmp_path):
    out = tmp_path / "r"
    main(["run", "--target", "mock", "--judges", "gold", "--quick", "--out", str(out), "--display", "none", "--no-card"])
    rc = main(["publish-card", "--run", str(out)])
    assert rc == 2


def test_cli_cite(capsys):
    rc = main(["cite"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "4gu6d" in out
    assert "@misc{aipsybench" in out
