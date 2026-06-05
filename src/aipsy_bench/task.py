"""The Inspect ``@task`` wiring Dataset + Solver + Scorer into the benchmark.

For a ``mock`` target the judges are also the offline mocks, so
``aipsy-bench run --target mock`` is a fully offline self-test. For a real target
the pinned judge snapshots are resolved by the Scorer (needs provider keys).
"""

from __future__ import annotations

from inspect_ai import Task, task

from . import bundle, spec
from .dataset import build_dataset
from .mocks import mock_judge_model
from .scorer import clinical_judge_panel
from .solver import scripted_dialogue
from .targets import is_mock_ref

_PANEL_PROVIDERS = {"single": (spec.PRIMARY_JUDGE_PROVIDER,), "gold": spec.PROVIDERS}


def quick_scenario_ids() -> list[str]:
    """``--quick`` smoke subset: one scenario per domain + both crisis scenarios.

    Directional only — a subset is not the battery (§7.2); never feeds a card/board.
    """
    scenarios = bundle.load_scenarios()
    first_per_domain: dict[str, str] = {}
    for s in scenarios:
        first_per_domain.setdefault(s.domain, s.id)
    ids = set(first_per_domain.values()) | set(spec.CRISIS_SCENARIO_IDS)
    return sorted(ids)


@task
def aipsy_bench(
    *,
    target: str = "mock",
    judges: str = "single",
    scenario_ids: list[str] | None = None,
    quick: bool = False,
    baseline_prompt: bool = False,
) -> Task:
    """Assemble the benchmark task. ``target`` only decides whether to wire mock
    judges (offline self-test); the model under test is passed to ``eval()``."""
    if judges not in _PANEL_PROVIDERS:
        raise ValueError(f"judges must be 'single' or 'gold', got {judges!r}")

    if scenario_ids is None and quick:
        scenario_ids = quick_scenario_ids()

    mock = is_mock_ref(target)
    judge_models = (
        {p: mock_judge_model() for p in _PANEL_PROVIDERS[judges]} if mock else None
    )

    return Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(baseline_prompt=baseline_prompt),
        scorer=clinical_judge_panel(panel=judges, judges=judge_models, cache=not mock),
    )
