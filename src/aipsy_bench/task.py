"""The Inspect ``@task`` wiring Dataset + Solver + Scorer into the benchmark.

For a ``mock`` target the judges are also the offline mocks, so
``aipsy-bench run --target mock`` is a fully offline self-test. For a real target
the pinned judge snapshots are resolved by the Scorer (needs provider keys).
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig

from . import bundle, spec
from .dataset import build_dataset
from .mocks import mock_judge_model
from .scorer import aipsy_na_mean, clinical_judge_panel
from .solver import scripted_dialogue
from .targets import is_mock_ref

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
    conversation: str = "stateless",
    timeout: int | None = None,
    max_retries: int | None = None,
    max_connections: int | None = None,
    judge_overrides: dict[str, str] | None = None,
    local_num_ctx: int | None = None,
) -> Task:
    """Assemble the benchmark task. ``target`` only decides whether to wire mock
    judges (offline self-test); the model under test is passed to ``eval()``.
    ``conversation`` is recorded for the report (Tier-1/2 session targets set it)."""
    _, judge_providers = spec.parse_panel(judges)  # validates; raises on a bad panel

    if scenario_ids is None and quick:
        scenario_ids = quick_scenario_ids()

    mock = is_mock_ref(target)
    judge_models = (
        {p: mock_judge_model() for p in judge_providers} if mock else None
    )
    to = timeout if timeout is not None else spec.MODEL_TIMEOUT
    mr = max_retries if max_retries is not None else spec.MODEL_MAX_RETRIES
    target_cfg = dict(timeout=to, max_retries=mr)
    if max_connections is not None:
        target_cfg["max_connections"] = max_connections

    return Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(baseline_prompt=baseline_prompt, conversation=conversation),
        scorer=clinical_judge_panel(panel=judges, judges=judge_models, cache=not mock,
                                    timeout=to, max_retries=mr, max_connections=max_connections,
                                    judge_overrides=judge_overrides, local_num_ctx=local_num_ctx),
        # Bound a hung/rate-limited TARGET call so it fails (→ run failure) instead of
        # hanging forever; keeps cancellation responsive. max_connections caps concurrency
        # to ease rate limiting.
        config=GenerateConfig(**target_cfg),
        # N/A-aware epoch reducer: keeps situational-metric N/As as None instead of the
        # default reducer coercing them to 0.0 (which also logs "convert value to float: None").
        epochs=Epochs(1, reducer=aipsy_na_mean()),
    )
