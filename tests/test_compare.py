"""Step 14 — compare deltas + regression gate. Offline."""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import spec
from aipsy_bench.cli import main
from aipsy_bench.compare import CompareError, compare, regression_gate, render_compare_table
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import JudgeValidation, load_validation

_PROVIDERS = {"single": ["openai"], "gold": list(spec.PROVIDERS)}


def _log(tmp_path, target, *, scenario_ids=None, panel="single"):
    scenario_ids = scenario_ids or ["s01"]
    judges = {p: mock_judge_model() for p in _PROVIDERS[panel]}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    return inspect_eval(task, model=mock_target_model(target), display="none", log_dir=str(tmp_path))[0]


def test_compare_flags_regression(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "failing")
    diff = compare(base, cand)
    assert diff["overall"]["AI_Trust"]["delta"] < 0
    assert diff["overall"]["AI_Trust"]["direction"] == "↓"
    assert any(r["metric"] == "AI_Trust" for r in diff["regressions"])


def test_render_table_shows_regression(tmp_path):
    table = render_compare_table(compare(_log(tmp_path / "b", "safe"), _log(tmp_path / "c", "failing")))
    assert "AI_Trust" in table
    assert "REGRESSION" in table


def test_compare_refuses_panel_mismatch(tmp_path):
    base = _log(tmp_path / "b", "safe", panel="single")
    cand = _log(tmp_path / "c", "safe", panel="gold")
    with pytest.raises(CompareError):
        compare(base, cand)


def test_compare_refuses_data_version_mismatch(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "safe")
    cand.samples[0].scores["clinical_judge_panel"].metadata["data_version"] = "v2"
    with pytest.raises(CompareError):
        compare(base, cand)


def test_regression_gate_advisory_while_pending(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "failing")
    reg = regression_gate(base, cand, load_validation())  # PENDING
    assert reg["passed"] is True
    assert reg["gate_eligible"] is False


def test_regression_gate_fails_for_licensed_metric(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "failing")
    v = JudgeValidation(status="VALIDATED", licensed=["AI_Trust", "empathy"], descriptive_only=[])
    reg = regression_gate(base, cand, v, max_regression=0.3)
    assert reg["passed"] is False
    assert any(f["metric"] == "AI_Trust" for f in reg["failures"])


def test_regression_gate_passes_when_stable(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "safe")
    v = JudgeValidation(status="VALIDATED", licensed=["AI_Trust"], descriptive_only=[])
    assert regression_gate(base, cand, v)["passed"] is True


def test_cli_compare_with_card(tmp_path):
    base = _log(tmp_path / "b", "safe")
    cand = _log(tmp_path / "c", "failing")
    rc = main(["compare", base.location, cand.location, "--card", "--out", str(tmp_path / "cmp")])
    assert rc == 0  # PENDING → regression gate advisory passes
    assert (tmp_path / "cmp" / "head_to_head.svg").exists()


def test_cli_compare_accepts_run_dirs_with_multiple_logs(tmp_path):
    # Inspect appends a new .eval per run; re-running into the same --out accumulates
    # several. `compare <dir> <dir>` must pick the NEWEST in each, not break on a glob.
    base_dir, cand_dir = tmp_path / "base", tmp_path / "cand"
    _log(base_dir, "safe")
    _log(base_dir, "safe")  # second run → two accumulated .eval files in base_dir
    _log(cand_dir, "failing")
    assert len(list(base_dir.glob("*.eval"))) == 2
    rc = main(["compare", str(base_dir), str(cand_dir)])
    assert rc == 0  # resolves the newest .eval in each dir; no "unrecognized arguments"


def test_cli_compare_missing_log_errors(tmp_path):
    rc = main(["compare", str(tmp_path / "nope"), str(tmp_path / "also-nope")])
    assert rc == 2
