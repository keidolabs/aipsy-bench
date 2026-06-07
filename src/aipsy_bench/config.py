"""Project config — ``aipsy-bench.yaml`` (§16). Committed = reproducible +
team-shareable; **CLI flags always override config.**

This is *project* config, distinct from the frozen ``spec.py`` constants: it can
set the target / judges / scenario subset / gate thresholds, but it can NEVER
change the frozen content or the judge pins (those keys are ignored).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_NAME = "aipsy-bench.yaml"


class ProjectConfig(BaseModel):
    # extra="ignore" → an attempt to set judge pins / data version in the yaml is
    # silently dropped; config cannot subvert the frozen invariants.
    model_config = ConfigDict(extra="ignore")

    target: str | None = None
    judges: str | None = None
    scenario: str | None = None        # comma-separated ids
    quick: bool = False
    out: str | None = None
    max_cost: float | None = None
    gate: dict[str, float] = Field(default_factory=dict)  # metric/AI_Trust threshold overrides
    judge_overrides: dict[str, str] = Field(default_factory=dict)  # provider→model (non-comparable)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load ``aipsy-bench.yaml`` from ``path`` (or the cwd); empty if absent."""
    p = Path(path) if path else Path(CONFIG_NAME)
    if not p.exists():
        return ProjectConfig()
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: config must be a mapping")
    return ProjectConfig(**data)
