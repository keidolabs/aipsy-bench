"""Step 01 — bundle loader + SHA-256 integrity. Offline, no network."""

from __future__ import annotations

import shutil

import pytest

from aipsy_bench import bundle, spec


def test_load_scenarios_default_battery():
    scenarios = bundle.load_scenarios()
    assert len(scenarios) == spec.N_SCENARIOS
    assert [s.id for s in scenarios] == [f"s{n:02d}" for n in range(1, 21)]
    for s in scenarios:
        assert len(s.turns) == spec.N_TURNS
        assert [t.turn for t in s.turns] == list(range(1, 11))


def test_crisis_scenarios_are_only_s06_s07():
    scenarios = bundle.load_scenarios()
    crisis_ids = sorted(s.id for s in scenarios if s.crisis)
    assert crisis_ids == ["s06", "s07"]


def test_default_excludes_reserved():
    ids = {s.id for s in bundle.load_scenarios()}
    assert ids.isdisjoint({f"s{n:02d}" for n in range(21, 31)})


def test_include_reserved_loads_extension():
    ids = {s.id for s in bundle.load_scenarios(include_reserved=True)}
    assert {f"s{n:02d}" for n in range(1, 21)}.issubset(ids)
    # reserved domains add scenarios beyond the frozen s01–s20 battery
    assert len(ids) > spec.N_SCENARIOS
    assert ids & {f"s{n:02d}" for n in range(21, 31)}


def test_verify_integrity_passes_on_shipped_bundle():
    digests = bundle.verify_integrity()
    assert "rubrics.yaml" in digests
    assert "scenarios/mental-health.yaml" in digests


def test_verify_integrity_detects_mutation(tmp_path):
    copy = tmp_path / "v1"
    shutil.copytree(spec.DATA_V1, copy)
    bundle.verify_integrity(copy)  # pristine copy verifies

    target = copy / "scenarios" / "mental-health.yaml"
    data = bytearray(target.read_bytes())
    data[0] ^= 0x01  # flip one bit of one byte
    target.write_bytes(bytes(data))

    with pytest.raises(bundle.BundleIntegrityError):
        bundle.verify_integrity(copy)


def test_persona_not_in_turn_messages():
    # persona is metadata only (§3.1) — it must never appear as a sent turn message
    for s in bundle.load_scenarios():
        for t in s.turns:
            assert s.persona not in t.message
