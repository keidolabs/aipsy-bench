"""Project config — ``aipsy-bench.yaml`` (§16). Committed = reproducible +
team-shareable; **CLI flags always override config.**

This is *project* config, distinct from the frozen ``spec.py`` constants: it can
set the target / judges / scenario subset / gate thresholds, but it can NEVER
change the frozen content or the judge pins (those keys are ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_NAME = "aipsy-bench.yaml"


class HttpTargetSpec(BaseModel):
    """A Tier-1 HTTP ``/eval`` endpoint, declared in ``aipsy-bench.yaml`` so an
    app dev never writes Python to benchmark their own bot (§6). Shape:

    ```yaml
    target:
      http: http://localhost:3000/api/ai-coach/eval
      headers: { x-eval-secret: ${EVAL_SECRET} }   # ${ENV} expanded at load
      conversation: stateless                        # stateless (default) | session
    ```
    """

    model_config = ConfigDict(extra="ignore")

    http: str
    headers: dict[str, str] = Field(default_factory=dict)
    conversation: str = "stateless"
    ref: str | None = None

    @field_validator("headers")
    @classmethod
    def _expand_env(cls, headers: dict[str, str]) -> dict[str, str]:
        # ``${EVAL_SECRET}`` / ``$EVAL_SECRET`` in a *committed* yaml resolves from the
        # environment at load — so the secret stays out of the repo. Unknown vars pass
        # through verbatim (surfacing a clear "unauthorized" from the endpoint, not a crash).
        return {k: os.path.expandvars(v) for k, v in headers.items()}


class ProjectConfig(BaseModel):
    # extra="ignore" → an attempt to set judge pins / data version in the yaml is
    # silently dropped; config cannot subvert the frozen invariants.
    model_config = ConfigDict(extra="ignore")

    # A bare ref string ("mock", "openai/gpt-5.4-mini") OR an HTTP endpoint mapping.
    target: str | HttpTargetSpec | None = None
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
