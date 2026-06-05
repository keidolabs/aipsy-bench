"""Judge-validation provenance — the moat (§4.4 / §0.3).

Every score cites the judge's measured human-agreement. 014's human-vs-judge
validation is IN PROGRESS, so until its gate artifact lands the status is
``PENDING_VALIDATION``: all metrics are ``descriptive_only`` and NO numeric
per-metric agreement is ever emitted (the §0.3 HARD GATE). The artifact is parsed
as untrusted input when it eventually exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import spec

PENDING = "PENDING_VALIDATION"
VALIDATED = "VALIDATED"


class JudgeValidation(BaseModel):
    status: str = PENDING
    source: str = "014 gate artifact (pending)"
    per_metric_alpha: dict[str, float] = Field(default_factory=dict)
    licensed: list[str] = Field(default_factory=list)
    descriptive_only: list[str] = Field(default_factory=lambda: list(spec.METRICS))

    def is_gateable(self, metric: str) -> bool:
        """A metric may gate the build ONLY if 014 has validated+licensed it.

        While PENDING (or for any ``descriptive_only`` metric) this is False — the
        §7/§0.3 guard that stops an unvalidated metric from ever failing a build.
        ``AI_Trust`` (the composite) gates only when explicitly licensed.
        """
        return self.status == VALIDATED and metric in self.licensed and metric not in self.descriptive_only


def load_validation(artifact_path: str | Path | None = None) -> JudgeValidation:
    """Load the 014 gate artifact, or return the PENDING default if none exists."""
    if artifact_path is None:
        return JudgeValidation()
    path = Path(artifact_path)
    if not path.exists():
        return JudgeValidation()

    raw = json.loads(path.read_text())  # untrusted — validate at this boundary
    if not isinstance(raw, dict):
        raise ValueError("validation artifact must be a JSON object")
    return JudgeValidation(
        status=str(raw.get("status", PENDING)),
        source=str(raw.get("source", "014 gate artifact")),
        per_metric_alpha={k: float(v) for k, v in (raw.get("per_metric_alpha") or {}).items()},
        licensed=[str(m) for m in (raw.get("licensed") or [])],
        descriptive_only=[str(m) for m in (raw.get("descriptive_only") or list(spec.METRICS))],
    )


def provisional_banner(validation: JudgeValidation) -> str | None:
    """The §0.3 provisional banner, or None once validation lands."""
    if validation.status == PENDING:
        return (
            "⚠ PROVISIONAL — instrument NOT yet human-validated. 014's judge-vs-human "
            "validation is in progress; these scores are descriptive only and no metric "
            "is gate-eligible. Do not cite as validated agreement."
        )
    return None
