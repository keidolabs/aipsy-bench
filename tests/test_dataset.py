"""Step 04 — scenarios → Inspect Samples. Offline."""

from __future__ import annotations

import pytest

from aipsy_bench import dataset, spec


def test_build_dataset_full_battery():
    ds = dataset.build_dataset()
    ids = [s.id for s in ds.samples]
    assert ids == [f"s{n:02d}" for n in range(1, 21)]
    for s in ds.samples:
        turns = s.metadata["turns"]
        assert len(turns) == 10
        for t in turns:
            assert t["phase"] == spec.PHASE_BY_TURN[t["turn"]]


def test_subset_by_ids():
    ds = dataset.build_dataset(scenario_ids=["s06", "s07"])
    assert [s.id for s in ds.samples] == ["s06", "s07"]
    assert all(s.metadata["crisis"] for s in ds.samples)


def test_unknown_id_raises():
    with pytest.raises(ValueError):
        dataset.build_dataset(scenario_ids=["s99"])


def test_persona_never_in_sent_fields():
    ds = dataset.build_dataset()
    for s in ds.samples:
        persona = s.metadata["persona"]
        assert persona  # persona exists as metadata
        assert persona not in str(s.input)
        for t in s.metadata["turns"]:
            assert persona not in t["message"]


def test_first_turn_is_input():
    ds = dataset.build_dataset(scenario_ids=["s01"])
    sample = ds.samples[0]
    assert sample.input == sample.metadata["turns"][0]["message"]
