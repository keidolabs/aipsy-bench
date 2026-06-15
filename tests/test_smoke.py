"""Step 12 — end-to-end mock smoke: run → result.json → cards → gate, all offline."""

from __future__ import annotations

import json

from aipsy_bench.cli import main


def test_end_to_end_mock_quick(tmp_path):
    out = tmp_path / "run"
    exit_code = main([
        "run", "--target", "mock", "--judges", "single", "--quick",
        "--out", str(out), "--display", "none",
    ])
    # the default mock target is safe → it passes the (now functional) directional gate → exit 0
    assert exit_code == 0

    result = json.loads((out / "result.json").read_text())
    assert result["mode"] == "benchmark"
    assert result["data_version"] == "v1"
    assert result["judge_validation"]["status"] == "DIRECTIONAL"
    assert result["gate"]["mode"] == "directional"
    assert result["gate"]["gate_eligible"] is True
    assert sorted(result["scores"]["by_scenario"]) == ["s01", "s06", "s07", "s09", "s15"]

    # artifacts: report + share card + badge
    assert (out / "report.txt").exists()
    assert (out / "card.svg").exists()
    assert (out / "card.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (out / "badge.svg").exists()


def test_failing_target_produces_remediation_card(tmp_path):
    out = tmp_path / "run"
    main(["run", "--target", "mock:failing", "--scenario", "s07",
          "--out", str(out), "--display", "none"])
    result = json.loads((out / "result.json").read_text())
    assert result["diagnostics"], "a failing target must yield remediation cards"
    card = result["diagnostics"][0]["card"]
    assert "✗ FAIL" in card
    assert "aipsy-bench run --scenario" in card  # reproduce command


def test_no_card_flag_skips_cards(tmp_path):
    out = tmp_path / "run"
    main(["run", "--target", "mock", "--scenario", "s01", "--no-card",
          "--out", str(out), "--display", "none"])
    assert (out / "result.json").exists()
    assert not (out / "card.svg").exists()
