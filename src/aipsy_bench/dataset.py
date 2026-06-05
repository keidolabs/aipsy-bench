"""Map the frozen scenarios onto Inspect ``Sample``s (one Sample per scenario).

Mechanical glue — no scoring, no model calls. The Solver (Step 06) replays the
scripted user turns from ``metadata["turns"]``; the Scorer (Step 07) reads phase
/ crisis / per-turn ``tests`` from the same metadata.
"""

from __future__ import annotations

from inspect_ai.dataset import MemoryDataset, Sample

from . import bundle
from .bundle import Scenario


def _sample_for(scenario: Scenario) -> Sample:
    turns = [
        {"turn": t.turn, "phase": t.phase, "message": t.message, "tests": t.tests}
        for t in scenario.turns
    ]
    return Sample(
        id=scenario.id,
        # input is turn 1's user message for viewer legibility; the Solver replays
        # the full ordered turn list from metadata["turns"], which is the source of
        # truth. persona is metadata only and is NEVER sent to the target (§3.1).
        input=scenario.turns[0].message,
        target=scenario.id,  # no gold answer — scoring is rubric-based (§4.2)
        metadata={
            "scenario_id": scenario.id,
            "domain": scenario.domain,
            "title": scenario.title,
            "crisis": scenario.crisis,
            "persona": scenario.persona,
            "turns": turns,
            "primary_metrics": list(scenario.primary_metrics),
        },
    )


def build_dataset(
    scenarios: tuple[Scenario, ...] | None = None,
    *,
    scenario_ids: list[str] | None = None,
    include_reserved: bool = False,
) -> MemoryDataset:
    """Build the Inspect dataset for the battery.

    ``scenario_ids`` subsets the battery (e.g. ``["s06", "s07"]``) — this is the
    machinery ``--quick`` and ``--scenario`` reuse later (§7.2); the subset lives
    here, not in the CLI.
    """
    if scenarios is None:
        scenarios = bundle.load_scenarios(include_reserved=include_reserved)

    selected = sorted(scenarios, key=lambda s: s.id)
    if scenario_ids is not None:
        wanted = set(scenario_ids)
        selected = [s for s in selected if s.id in wanted]
        missing = wanted - {s.id for s in selected}
        if missing:
            raise ValueError(f"unknown scenario ids: {sorted(missing)}")

    return MemoryDataset(samples=[_sample_for(s) for s in selected], name="aipsy-bench")
