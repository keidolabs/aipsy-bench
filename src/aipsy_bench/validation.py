"""Judge-validation provenance — the moat (§4.4 / §0.3).

Every score cites where the judge panel stands on human-agreement. 014's
human-vs-judge validation runs as a **parallel background track**, so until its
gate artifact lands the status is ``DIRECTIONAL``: the benchmark is a
recommendation, not a rubber-stamp. In that posture the CI gate is **functional**
— a metric gates the build against the developer's thresholds as a directional
recommendation — but the durable guards still hold:

* no numeric, *validated* per-metric agreement (α) is ever claimed (§0.3);
* a ``descriptive_only`` metric can NEVER fail the gate (§7) — custom/lab mode
  (§3.5) forces this for every metric, and a metric 014 explicitly declines lands
  here once the study completes;
* public/leaderboard authority ("validated against clinical experts · OSF DOI")
  is withheld until the study lands and is reproducible-by-design in the meantime.

When the artifact arrives the status becomes ``VALIDATED``: licensed metrics gate
with validated authority and their α is published. The artifact is parsed as
untrusted input.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import spec

# Launch posture: directional now, validated in parallel (the 2026-06-09 decision).
DIRECTIONAL = "DIRECTIONAL"
VALIDATED = "VALIDATED"
# Legacy artifact value — pre-decision logs/artifacts said this; map it to DIRECTIONAL.
_LEGACY_PENDING = "PENDING_VALIDATION"


class JudgeValidation(BaseModel):
    status: str = DIRECTIONAL
    source: str = "014 human-agreement study (in progress — parallel track)"
    per_metric_alpha: dict[str, float] = Field(default_factory=dict)
    licensed: list[str] = Field(default_factory=list)
    # Metrics that may NEVER gate, in any mode (the durable §7 guard): custom/lab
    # mode forces all metrics here; the 014 study moves any it declines here. Empty
    # by default so the directional gate covers the full frozen rubric.
    descriptive_only: list[str] = Field(default_factory=list)

    def is_gateable(self, metric: str) -> bool:
        """May this metric fail the build?

        * ``descriptive_only`` → never (the durable §7 guard, every mode).
        * ``VALIDATED`` → only when 014 has licensed the metric (validated authority).
        * ``DIRECTIONAL`` (launch default) → yes, as a **directional recommendation**:
          the gate is functional now, the thresholds are the developer's policy, and
          the score is explicitly not-yet-human-validated (§0.3).
        """
        if metric in self.descriptive_only:
            return False
        if self.status == VALIDATED:
            return metric in self.licensed
        return True


def load_validation(artifact_path: str | Path | None = None) -> JudgeValidation:
    """Load the 014 gate artifact, or return the DIRECTIONAL default if none exists."""
    if artifact_path is None:
        return JudgeValidation()
    path = Path(artifact_path)
    if not path.exists():
        return JudgeValidation()

    raw = json.loads(path.read_text())  # untrusted — validate at this boundary
    if not isinstance(raw, dict):
        raise ValueError("validation artifact must be a JSON object")
    status = str(raw.get("status", DIRECTIONAL))
    if status == _LEGACY_PENDING:
        status = DIRECTIONAL
    return JudgeValidation(
        status=status,
        source=str(raw.get("source", "014 gate artifact")),
        per_metric_alpha={k: float(v) for k, v in (raw.get("per_metric_alpha") or {}).items()},
        licensed=[str(m) for m in (raw.get("licensed") or [])],
        descriptive_only=[str(m) for m in (raw.get("descriptive_only") or [])],
    )


def custom_mode_validation() -> JudgeValidation:
    """Custom/lab mode (§3.5): force every metric to ``descriptive_only`` so a custom
    run can never gate or claim agreement — it is explicitly *not "a score."*"""
    return JudgeValidation(
        status=DIRECTIONAL,
        source="custom/lab mode — not the frozen instrument (§3.5)",
        descriptive_only=list(spec.METRICS) + ["AI_Trust"],
    )


def local_judge_banner() -> str:
    """The exp-016 §Fit-for-purpose positioning for the local FT judge — appended to a
    local-panel run's warnings (in addition to the directional banner). The local judge
    is a *different instrument* than the frontier gold panel and is directional by
    construction; it is a flag-for-review screen, not a machine-only certifier."""
    return (
        f"◆ LOCAL JUDGE — scored by the offline fine-tuned {spec.LOCAL_JUDGE_VERSION} (Ollama), a "
        "DIFFERENT instrument than the frontier gold panel: comparable to other local runs "
        "only, never to gold. Directional by construction and human-in-the-loop, not a "
        "machine-only gate — strongest on crisis/empathy/boundary; advice_safety is the "
        "lowest-confidence axis (treat advice flags as flag-for-review). Re-run it yourself."
    )


def directional_banner(validation: JudgeValidation) -> str | None:
    """The §0.3 directional banner, or None once validation lands (VALIDATED)."""
    if validation.status == VALIDATED:
        return None
    return (
        "⚠ DIRECTIONAL — a recommendation, not a rubber-stamp. The judge panel's "
        "human-agreement validation (014) is running in parallel and has not yet landed, "
        "so these scores are a directional, methodology-transparent, reproducible reading — "
        "not an authoritative, human-validated safety rating. Gate thresholds are your "
        "policy. No validated per-metric agreement (α) is claimed yet; re-run it yourself "
        "to reproduce."
    )
