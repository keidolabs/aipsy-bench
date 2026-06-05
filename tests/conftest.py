"""Shared pytest fixtures — the offline mock target/judge models."""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model, mock_target_model


@pytest.fixture
def safe_target():
    return mock_target_model("safe")


@pytest.fixture
def failing_target():
    return mock_target_model("failing")


@pytest.fixture
def adversarial_target():
    return mock_target_model("adversarial")


@pytest.fixture
def judge_model():
    return mock_judge_model()
