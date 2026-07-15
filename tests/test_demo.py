"""``aipsy-bench demo`` — the fully-offline guided head-to-head. Deterministic; no
network, no API keys, no Ollama (reads two bundled, scrubbed .eval logs).

Guards the two things a demo must never regress: the *story* (older model FAILs the
gate, newer model PASSes, on the same app + same judge) and the *hygiene* (nothing
identifying — the app or the backing model names — ships or renders, honoring the
fully-generic ship decision)."""

from __future__ import annotations

import json

import pytest

from aipsy_bench import demo

# Strings that must never appear in a bundled fixture or any rendered demo artifact.
# The demo was built fully-generic (no app name, no backing-model names).
_FORBIDDEN = ("mojoe", "gpt-4o", "gpt-4o-mini")


def test_fixtures_present_and_scrubbed():
    """Both bundled logs exist and carry no identifying strings (a re-bundle that forgets
    to scrub the app/model identity is a hygiene regression, not a cosmetic one)."""
    for run in (demo.OLDER, demo.NEWER):
        log = demo._load(run)
        blob = json.dumps(log.model_dump(), default=str).lower()
        for bad in _FORBIDDEN:
            assert bad not in blob, f"{run.asset} leaks {bad!r}"
        assert log.status == "success"


def test_render_demo_tells_the_story(tmp_path):
    res = demo.render_demo(tmp_path)

    # The whole point: same app + same judge, only the backing model differs → gate flips.
    assert res["older"]["gate"]["passed"] is False
    assert res["newer"]["gate"]["passed"] is True
    assert res["older"]["scores"]["overall"]["AI_Trust"] < 2.5
    assert res["newer"]["scores"]["overall"]["AI_Trust"] >= 3.5

    # Generic labels only — no app/model identity in the displayed refs.
    assert res["older"]["target"]["ref"] == "Coach app · older model"
    assert res["newer"]["target"]["ref"] == "Coach app · newer model"

    # Comparable lane (single/v1 on both) and every axis improves older → newer.
    overall = res["diff"]["overall"]
    assert all(overall[k]["delta"] >= 0 for k in overall)
    assert overall["AI_Trust"]["delta"] == pytest.approx(1.65, abs=0.01)


def test_written_artifacts_and_no_leak(tmp_path):
    res = demo.render_demo(tmp_path)
    for sub in ("older-model", "newer-model"):
        for name in ("result.json", "report.txt", "report.html"):
            assert (tmp_path / sub / name).exists(), f"{sub}/{name} not written"
    assert (tmp_path / "head_to_head.svg").exists()
    assert (tmp_path / "head_to_head.png").exists()

    # No identifying string in ANY written artifact, nor in the printed narrative/footer.
    for path in tmp_path.rglob("*"):
        if path.suffix in (".json", ".txt", ".html", ".svg"):
            text = path.read_text().lower()
            for bad in _FORBIDDEN:
                assert bad not in text, f"{path.name} leaks {bad!r}"
    printed = (res["narrative"] + res["footer"]).lower()
    for bad in _FORBIDDEN:
        assert bad not in printed


def test_footer_surfaces_report_html_paths(tmp_path):
    """The CLI must point the user at each saved report.html (the requested UX)."""
    res = demo.render_demo(tmp_path)
    footer = res["footer"]
    assert (tmp_path / "older-model" / "report.html").resolve().as_uri() in footer
    assert (tmp_path / "newer-model" / "report.html").resolve().as_uri() in footer
    assert "gate FAIL" in footer and "gate PASS" in footer


def test_reports_carry_directional_and_subset_notes(tmp_path):
    """A demo is a single-scenario, single-judge replay, so it must read as directional and
    non-comparable — never as an authoritative score."""
    res = demo.render_demo(tmp_path)
    warnings = " ".join(res["older"]["warnings"])
    assert "single crisis scenario" in warnings          # the demo note
    assert "not comparable" in warnings                  # subset non-comparability
    assert any(w.startswith("⚠ DIRECTIONAL") for w in res["older"]["warnings"])


def test_no_card_skips_card(tmp_path):
    res = demo.render_demo(tmp_path, no_card=True)
    assert res["card"] is None
    assert not (tmp_path / "head_to_head.svg").exists()
    # Reports still render even with cards off.
    assert (tmp_path / "older-model" / "report.html").exists()


def test_cli_demo_exits_zero(tmp_path, capsys):
    """The gate FAIL is part of the story — `demo` itself always succeeds (exit 0)."""
    from aipsy_bench.cli import main

    rc = main(["demo", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fully offline" in out
    assert "Gate ✗ FAIL" in out and "Gate ✓ PASS" in out
    assert "report.html" in out  # the CLI prints the path to each saved report
