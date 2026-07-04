"""Tier-0 model-string validation — a bad/gibberish ref is a clean error, NEVER a run-setup
crash (init/doctor/run). Offline.
"""

from __future__ import annotations

import pytest

from aipsy_bench.cli import main
from aipsy_bench.targets import TargetResolutionError, resolve_target, validate_model_ref


# ---- validate_model_ref (offline structural + soft provider check) ----

@pytest.mark.parametrize("ref", [
    "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-6", "google/gemini-2.5-flash",
    "ollama/llama3", "hf/org/model", "mock", "mock:failing",
])
def test_validate_ok_refs(ref):
    assert validate_model_ref(ref)["level"] == "ok"


@pytest.mark.parametrize("ref", [
    "gpt-4o",              # no provider prefix
    "just some gibberish",  # whitespace → not a model string
    "",                     # empty
    "   ",                  # blank
    "openai/",              # empty model
    "/gpt-4o",              # empty provider
])
def test_validate_structural_errors(ref):
    check = validate_model_ref(ref)
    assert check["level"] == "error" and check["ok"] is False


def test_validate_unknown_provider_is_soft_warn():
    # a provider we don't recognize is NOT blocked — Inspect may support it (open-weights tail)
    check = validate_model_ref("acme-labs/some-model")
    assert check["level"] == "warn" and check["ok"] is True
    assert "acme-labs" in check["message"]


# ---- resolve_target normalizes every failure to a clean TargetResolutionError ----

def test_resolution_error_is_a_valueerror():
    # subclassing ValueError → the run command's existing `except ValueError` catches it
    assert issubclass(TargetResolutionError, ValueError)


def test_resolve_structural_gibberish_clean_error():
    with pytest.raises(TargetResolutionError) as ei:
        resolve_target("just some gibberish")
    assert "model string" in str(ei.value)  # a helpful message, not a raw traceback


def test_resolve_unknown_provider_clean_error():
    with pytest.raises(TargetResolutionError) as ei:
        resolve_target("foo/bar")
    assert "not a recognized" in str(ei.value) and "foo" in str(ei.value)


def test_resolve_missing_key_clean_error(monkeypatch):
    # valid provider, no key → Inspect raises PrerequisiteError (NOT a ValueError) — the bug.
    # It must become a clean, actionable TargetResolutionError, not crash.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TargetResolutionError) as ei:
        resolve_target("openai/gpt-9000")
    assert "OPENAI_API_KEY" in str(ei.value) or "keys set" in str(ei.value)


def test_resolve_mock_still_works():
    rt = resolve_target("mock")
    assert rt.adapter == "mock" and rt.is_mock


# ---- CLI: bad model is a clean exit, and init never scaffolds a crashing config ----

def test_run_bad_model_exits_2_not_crash(tmp_path):
    rc = main(["run", "--model", "not-a-model", "--scenario", "s01", "--out", str(tmp_path / "o")])
    assert rc == 2  # clean config error, no traceback


def test_init_refuses_bad_model(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    rc = main(["init", "--model", "not-a-model", "--out", str(out)])
    assert rc == 2
    assert not out.exists()  # a config that would crash `run` was never written


def test_init_accepts_good_model(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    assert main(["init", "--model", "openai/gpt-5.4-mini", "--out", str(out)]) == 0
    assert "openai/gpt-5.4-mini" in out.read_text()


def test_init_accepts_unknown_provider_with_warning(tmp_path, capsys):
    out = tmp_path / "aipsy-bench.yaml"
    assert main(["init", "--model", "acme/model", "--out", str(out)]) == 0  # warn, not blocked
    assert "acme/model" in out.read_text()
    assert "warning" in capsys.readouterr().err.lower()


def test_doctor_flags_bad_model(capsys):
    rc = main(["doctor", "--model", "not-a-model", "--judges", "gold"])
    out = capsys.readouterr().out
    assert "target model: ✗ not-a-model" in out  # caught at preflight
    assert rc == 1
