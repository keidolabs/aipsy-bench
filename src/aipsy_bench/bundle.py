"""Load the frozen ``data/v1`` content bundle and verify it against MANIFEST.

Ported from 014 ``harness/bundle.py``. Every pipeline entrypoint calls
:func:`verify_integrity` before doing anything, so a single changed byte in the
frozen bundle aborts the run loudly instead of silently producing an
off-benchmark score (§1.2 — a content change is a ``data/`` version bump).

Untrusted YAML is validated at this boundary via Pydantic models (CLAUDE.md
quality bar).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from . import spec


class BundleIntegrityError(RuntimeError):
    """Raised when a bundle file's SHA-256 does not match the MANIFEST."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest(manifest_path: Path) -> dict[str, str]:
    """Parse ``shasum -a 256`` format: '<hex>  <relative-path>' per line."""
    entries: dict[str, str] = {}
    for raw in manifest_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if not name:  # tolerate single-space / '*' binary marker
            digest, _, name = line.partition(" ")
            name = name.lstrip("*").strip()
        entries[name.strip()] = digest.strip().lower()
    return entries


def verify_integrity(bundle_dir: Path | None = None) -> dict[str, str]:
    """Verify every MANIFEST-listed file matches its recorded SHA-256.

    Returns the ``{relative-path: digest}`` map on success; raises
    :class:`BundleIntegrityError` on any mismatch or missing file. The manifest
    cannot hash itself — its own integrity is covered by version control.
    """
    bundle_dir = Path(bundle_dir or spec.DATA_V1)
    manifest_path = bundle_dir / spec.MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleIntegrityError(f"MANIFEST not found: {manifest_path}")

    expected = _parse_manifest(manifest_path)
    if not expected:
        raise BundleIntegrityError("MANIFEST is empty or unparseable")

    problems: list[str] = []
    for name, want in expected.items():
        fp = bundle_dir / name
        if not fp.exists():
            problems.append(f"missing: {name}")
            continue
        got = _sha256(fp)
        if got != want:
            problems.append(f"hash mismatch: {name}\n  want {want}\n  got  {got}")

    if problems:
        raise BundleIntegrityError(
            "Frozen bundle does not match MANIFEST.sha256 — the benchmark "
            "content has been altered. Refusing to proceed.\n  "
            + "\n  ".join(problems)
        )
    return expected


# --------------------------------------------------------------------------
# Typed views over the deposited files (Pydantic — boundary validation)
# --------------------------------------------------------------------------
class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn: int    # 1..10
    phase: int   # clinical arc phase 1..4 (014 prereg §5)
    message: str # the pre-scripted user message sent to the target, verbatim
    tests: str   # scenario-author annotation of what the turn probes (metadata)


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str       # s01..s20
    domain: str   # mental_health | companion | coaching
    title: str
    crisis: bool  # true only for s06, s07
    persona: str  # user persona — metadata only, NEVER sent to the target (§3.1)
    turns: tuple[Turn, ...]
    # ``primary_metrics`` is legacy v1 annotation, inert: the judge scores all six
    # v2 metrics regardless. Kept verbatim for provenance, never a scoring input.
    primary_metrics: tuple[str, ...]


def _scenario_from_yaml(s: dict[str, Any]) -> Scenario:
    turns = tuple(
        Turn(
            turn=int(t["turn"]),
            phase=int(t["phase"]),
            message=str(t["message"]),
            tests=str(t.get("tests", "")),
        )
        for t in sorted(s["turns"], key=lambda t: int(t["turn"]))
    )
    return Scenario(
        id=str(s["id"]),
        domain=str(s["domain"]),
        title=str(s["title"]),
        crisis=bool(s["crisis"]),
        persona=str(s["persona"]),
        turns=turns,
        primary_metrics=tuple(s.get("primary_metrics", ())),
    )


def _load_files(filenames: tuple[str, ...]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for fname in filenames:
        doc = yaml.safe_load((spec.DATA_V1 / fname).read_text())
        scenarios.extend(_scenario_from_yaml(s) for s in doc["scenarios"])
    return scenarios


@lru_cache(maxsize=1)
def load_rubric(verify: bool = True) -> dict:
    """Return the parsed ``rubrics.yaml`` (the authoritative v2 spec)."""
    if verify:
        verify_integrity()
    return yaml.safe_load((spec.DATA_V1 / "rubrics.yaml").read_text())


@lru_cache(maxsize=1)
def load_judge_prompt(verify: bool = True) -> str:
    """Return the verbatim ``judge_prompt.md`` evaluator prompt."""
    if verify:
        verify_integrity()
    return (spec.DATA_V1 / "judge_prompt.md").read_text()


@lru_cache(maxsize=4)
def load_scenarios(verify: bool = True, include_reserved: bool = False) -> tuple[Scenario, ...]:
    """Load the 20 confirmatory scenarios (s01–s20) from the 3 domain files.

    Reserved-extension domains (healthcare / customer-support, s21–s30) are
    excluded by default (§3.1). They load ONLY under ``include_reserved=True`` as
    an exploratory, non-validated extension; the default battery never pulls them.
    """
    if verify:
        verify_integrity()

    core = sorted(_load_files(spec.SCENARIO_FILES), key=lambda s: s.id)
    _validate_scenarios(core)
    if not include_reserved:
        return tuple(core)

    reserved = sorted(_load_files(spec.RESERVED_SCENARIO_FILES), key=lambda s: s.id)
    for s in reserved:  # structure-only checks; reserved is not the frozen battery
        _validate_turn_structure(s)
    return tuple(core) + tuple(reserved)


def _validate_turn_structure(s: Scenario) -> None:
    if len(s.turns) != spec.N_TURNS:
        raise BundleIntegrityError(f"{s.id}: expected {spec.N_TURNS} turns, got {len(s.turns)}")
    if [t.turn for t in s.turns] != list(range(1, spec.N_TURNS + 1)):
        raise BundleIntegrityError(f"{s.id}: turn numbering not 1..{spec.N_TURNS}")


def _validate_scenarios(scenarios: list[Scenario]) -> None:
    """Fail fast if the default battery deviates from the frozen design."""
    ids = [s.id for s in scenarios]
    expected_ids = [f"s{n:02d}" for n in range(1, spec.N_SCENARIOS + 1)]
    if ids != expected_ids:
        raise BundleIntegrityError(
            f"Expected exactly {spec.N_SCENARIOS} confirmatory scenarios "
            f"{expected_ids}, got {ids}"
        )
    for s in scenarios:
        _validate_turn_structure(s)
    crisis_ids = sorted(s.id for s in scenarios if s.crisis)
    if crisis_ids != sorted(spec.CRISIS_SCENARIO_IDS):
        raise BundleIntegrityError(
            f"Crisis-flagged scenarios {crisis_ids} != frozen design "
            f"{sorted(spec.CRISIS_SCENARIO_IDS)}"
        )
