"""Step 08 — remediation cards on real 014 golden transcripts. Offline.

Golden fixtures under tests/golden/ are built from REAL 014 transcripts + the 3
pinned judges' real scores/reasoning, transformed into the production Score.metadata
shape. The cards must match the frozen expected text byte-for-byte (deterministic,
no LLM, no timestamps).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipsy_bench import diagnostics

GOLDEN = Path(__file__).parent / "golden"
CASES = ["crisis_s06_openai_r3", "boundary_s01_openai_r3", "ceiling_s01_anthropic_r1"]


def _metadata(name):
    return json.loads((GOLDEN / f"{name}.metadata.json").read_text())


@pytest.mark.parametrize("name", CASES)
def test_card_matches_golden(name):
    md = _metadata(name)
    rendered = diagnostics.render_explanation(md)
    expected = (GOLDEN / f"{name}.cards.txt").read_text()
    assert rendered == expected


@pytest.mark.parametrize("name", CASES)
def test_cards_are_byte_stable(name):
    md = _metadata(name)
    assert diagnostics.render_explanation(md) == diagnostics.render_explanation(md)


def test_crisis_card_localizes_and_cites():
    md = _metadata("crisis_s06_openai_r3")
    card = diagnostics.build_card(md, metric="crisis_handling")
    assert "crisis_handling" in card
    assert "✗ FAIL" in card
    assert "phase" in card  # phase-localized
    assert "scenario s06 (crisis)" in card
    # quotes the failing turn's real target text
    worst = min(
        (pt for pt in md["per_turn"] if "crisis_handling" in pt["failed"]),
        key=lambda pt: pt["ensemble"]["crisis_handling"],
    )
    assert worst["ai_text"][:40] in card
    # cites a rubric anchor band + the crisis-resource fix + reproduce command
    assert "rubric anchor" in card
    assert "crisis-resource hand-off" in card
    assert "aipsy-bench run --scenario s06" in card


def test_judge_alpha_is_pending():
    md = _metadata("crisis_s06_openai_r3")
    assert "judge α=PENDING" in diagnostics.build_card(md, metric="crisis_handling")
    # never a numeric agreement while validation is PENDING (§0.3)
    assert diagnostics.JUDGE_ALPHA_DISPLAY == "PENDING"


def test_ceiling_has_no_cards():
    md = _metadata("ceiling_s01_anthropic_r1")
    assert diagnostics.build_all_cards(md) == []
    assert "no metric failed" in diagnostics.render_explanation(md)
