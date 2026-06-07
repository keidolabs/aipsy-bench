"""Step 17 — doctor / --dry-run / --max-cost / --resume / config / provenance. Offline."""

from __future__ import annotations

import json

from aipsy_bench import cli
from aipsy_bench.bundle import BundleIntegrityError
from aipsy_bench.cli import main
from aipsy_bench.config import load_config


def test_cli_reports_integrity_error_cleanly(tmp_path, monkeypatch, capsys):
    # a tampered frozen bundle is a user-facing guard (§1.2) — show a clean message,
    # exit 1, and NOT a raw Python traceback.
    def boom(*a, **k):
        raise BundleIntegrityError("hash mismatch: scenarios/companion.yaml")

    monkeypatch.setattr(cli, "aipsy_bench", boom)
    rc = cli.main(["run", "--target", "mock", "--scenario", "s09", "--out", str(tmp_path), "--display", "none"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "integrity check failed" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# doctor — no scored calls
# --------------------------------------------------------------------------
def test_doctor_mock_offline(capsys):
    rc = main(["doctor", "--target", "mock", "--judges", "gold"])
    out = capsys.readouterr().out
    assert "data/v1 integrity: OK" in out
    assert "offline mock judges" in out
    assert rc == 0


def test_doctor_reads_dotenv(tmp_path, monkeypatch, capsys):
    # a key set only in a project .env (not exported) must be seen by doctor, exactly
    # as a real run would see it (Inspect loads .env on the eval path). chdir to a clean
    # tmp so the test never reads a real .env in the repo; SDK presence isolated.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_sdk_installed", lambda p: True)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-fake-for-test\n")
    monkeypatch.chdir(tmp_path)
    rc = main(["doctor", "--target", "openai/gpt-5.4-mini", "--judges", "single"])
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY present" in out
    assert rc == 0


def test_doctor_reports_missing_keys(tmp_path, monkeypatch, capsys):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(cli, "_sdk_installed", lambda p: True)
    monkeypatch.chdir(tmp_path)  # clean dir → no real .env leaks the key in
    rc = main(["doctor", "--target", "openai/gpt-5.4-mini", "--judges", "single"])
    out = capsys.readouterr().out
    assert "data/v1 integrity: OK" in out
    assert "OPENAI_API_KEY MISSING" in out
    assert rc == 1  # a missing key for the selected panel is a non-zero preflight


# --------------------------------------------------------------------------
# dry-run + max-cost — no scored calls
# --------------------------------------------------------------------------
def test_required_providers():
    assert cli._required_providers("openai/gpt-5.4-mini", "gold") == ["anthropic", "google", "openai"]
    # single panel target on anthropic → primary judge (openai) + target (anthropic)
    assert cli._required_providers("anthropic/claude-sonnet-4-6", "single") == ["anthropic", "openai"]
    assert cli._required_providers("mock", "single") == ["openai"]  # mock skipped at call site


def test_run_reports_missing_sdk_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_sdk_installed", lambda p: False)
    rc = cli.main(["run", "--model", "openai/gpt-5.4-mini", "--quick",
                   "--out", str(tmp_path), "--display", "none"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "uv sync" in err
    assert "Traceback" not in err
    assert not (tmp_path / "result.json").exists()  # bailed before running


def test_doctor_reports_missing_sdk(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_sdk_installed", lambda p: False)
    rc = cli.main(["doctor", "--target", "openai/gpt-5.4-mini", "--judges", "single"])
    out = capsys.readouterr().out
    assert "SDK MISSING" in out
    assert rc == 1


def test_dry_run_no_scored_calls(tmp_path, capsys):
    out = tmp_path / "r"
    rc = main(["run", "--target", "mock", "--quick", "--dry-run", "--out", str(out)])
    text = capsys.readouterr().out
    assert "dry-run estimate" in text
    assert "target calls" in text and "est. cost" in text
    assert rc == 0
    assert not (out / "result.json").exists()  # nothing was actually run


def test_max_cost_aborts_before_running(tmp_path):
    out = tmp_path / "r"
    rc = main(["run", "--target", "mock", "--max-cost", "0.0001", "--out", str(out), "--display", "none"])
    assert rc == 2
    assert not (out / "result.json").exists()


# --------------------------------------------------------------------------
# config — loaded, overridden by flags, cannot subvert frozen invariants
# --------------------------------------------------------------------------
def test_config_ignores_frozen_keys(tmp_path):
    cfg = tmp_path / "aipsy-bench.yaml"
    cfg.write_text("target: mock\njudges: single\nscenario: s01\njudge_pins: {openai: hacked}\ndata_version: v9\n")
    c = load_config(cfg)
    assert c.target == "mock"
    assert c.scenario == "s01"
    assert not hasattr(c, "judge_pins")  # frozen keys silently dropped (extra=ignore)
    assert not hasattr(c, "data_version")


def test_config_used_and_flag_overrides(tmp_path):
    cfg = tmp_path / "aipsy-bench.yaml"
    cfg.write_text("target: mock\nscenario: s01\n")
    out = tmp_path / "r"
    main(["run", "--config", str(cfg), "--out", str(out), "--display", "none", "--no-card"])
    assert list(json.loads((out / "result.json").read_text())["scores"]["by_scenario"]) == ["s01"]

    out2 = tmp_path / "r2"
    main(["run", "--config", str(cfg), "--scenario", "s06", "--out", str(out2), "--display", "none", "--no-card"])
    assert list(json.loads((out2 / "result.json").read_text())["scores"]["by_scenario"]) == ["s06"]


# --------------------------------------------------------------------------
# resume — reuse a completed run (no re-judge)
# --------------------------------------------------------------------------
def test_resume_reuses_completed_run(tmp_path):
    out = tmp_path / "r"
    main(["run", "--target", "mock", "--scenario", "s01", "--out", str(out), "--display", "none", "--no-card"])
    rid = json.loads((out / "result.json").read_text())["run_id"]
    rc = main(["run", "--target", "mock", "--scenario", "s01", "--out", str(out),
               "--display", "none", "--no-card", "--resume", rid])
    assert rc == 0
    # the resumed result reuses the prior log (same run_id) rather than running fresh
    assert json.loads((out / "result.json").read_text())["run_id"] == rid


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def test_provenance(tmp_path, capsys):
    out = tmp_path / "r"
    main(["run", "--target", "mock", "--scenario", "s01", "--judges", "gold",
          "--out", str(out), "--display", "none", "--no-card"])
    rc = main(["provenance", "--out", str(out)])
    o = capsys.readouterr().out
    assert rc == 0
    assert "gpt-5.4-mini" in o and "claude-sonnet-4-6" in o and "gemini-2.5-flash" in o
    assert "data_version" in o and "git_sha" in o and "run_id" in o
    assert "mockllm/judge" in o  # resolved snapshot recorded
