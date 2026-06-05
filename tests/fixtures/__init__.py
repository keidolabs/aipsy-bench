"""Offline test substrate — re-exports the package's deterministic mock models.

The mocks live in ``aipsy_bench.mocks`` (shipped, so ``aipsy-bench run --target
mock`` is a real offline self-test); this package just re-exports them for tests.
"""

from __future__ import annotations

from aipsy_bench.mocks import (
    MARK_RE,
    build_marker,
    mock_judge_model,
    mock_target_model,
    parse_marker,
)

__all__ = [
    "MARK_RE",
    "build_marker",
    "mock_judge_model",
    "mock_target_model",
    "parse_marker",
]
