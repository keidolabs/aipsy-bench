"""Target adapter — the thing under test (§6).

Step 10 ships **Tier 0** (a bare Inspect model string — the launch-leaderboard
path) plus the offline ``mock`` target. Tier 1 (HTTP), Tier 2 (Python callable),
``conversation: session``, and target-failure classification are Step 13 — the
``resolve_target`` switch is the extension point for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import Model, get_model

from .mocks import mock_target_model

MOCK_PREFIX = "mock"


@dataclass
class ResolvedTarget:
    model: Model
    ref: str
    adapter: str          # "mock" | "model" (Tier 0)
    is_mock: bool
    mock_profile: str | None = None


def is_mock_ref(ref: str) -> bool:
    return ref == MOCK_PREFIX or ref.startswith(MOCK_PREFIX + ":")


def resolve_target(ref: str) -> ResolvedTarget:
    """Resolve a target ref to an Inspect model.

    - ``mock`` / ``mock:<profile>`` → the offline deterministic mock target.
    - anything else → a Tier-0 bare Inspect model string (``openai/gpt-…`` etc.).
    """
    if is_mock_ref(ref):
        profile = ref.split(":", 1)[1] if ":" in ref else "safe"
        return ResolvedTarget(
            model=mock_target_model(profile), ref=ref, adapter="mock",
            is_mock=True, mock_profile=profile,
        )
    return ResolvedTarget(model=get_model(ref), ref=ref, adapter="model", is_mock=False)
