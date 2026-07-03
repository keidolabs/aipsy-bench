"""CLI-native HTTP target (§6) — the zero-Python path for an app dev.

Resolution + preflight/dry-run wiring only; a live scored run over HTTP is covered by
test_targets.py / test_mojoe_acceptance.py (which inject a fake transport). These tests
make NO network call.
"""

from __future__ import annotations

import pytest

from aipsy_bench import cli, targets
from aipsy_bench.cli import main
from aipsy_bench.config import HttpTargetSpec, ProjectConfig, load_config
from aipsy_bench.targets import probe_endpoint


def _args(argv):
    return cli.build_parser().parse_args(argv)


# --- header parsing -------------------------------------------------------
def test_parse_headers_splits_on_first_colon():
    assert cli._parse_headers(["x-eval-secret:s", "Authorization:Bearer a:b"]) == {
        "x-eval-secret": "s",
        "Authorization": "Bearer a:b",
    }


def test_parse_headers_rejects_malformed():
    with pytest.raises(ValueError, match="NAME:VALUE"):
        cli._parse_headers(["no-colon"])


# --- config spec ----------------------------------------------------------
def test_config_http_target_spec_expands_env(monkeypatch):
    monkeypatch.setenv("EVAL_SECRET", "shh")
    cfg = ProjectConfig(target={"http": "http://localhost:3000/eval",
                                "headers": {"x-eval-secret": "${EVAL_SECRET}"}})
    assert isinstance(cfg.target, HttpTargetSpec)
    assert cfg.target.headers == {"x-eval-secret": "shh"}
    assert cfg.target.conversation == "stateless"


def test_config_bare_ref_stays_a_string():
    assert ProjectConfig(target="openai/gpt-5.4-mini").target == "openai/gpt-5.4-mini"


# --- resolution -----------------------------------------------------------
def test_display_model_name_reads_as_endpoint():
    from aipsy_bench.targets import _display_model_name, http_target
    assert _display_model_name("mojoe-coach") == "mojoe-coach"
    assert _display_model_name("http://localhost:3000/api/ai-coach/eval") == "localhost-3000-api-ai-coach-eval"
    # the wrapped model no longer displays as a bare 'model' (which reads like a mock)
    rt = http_target("http://localhost:3000/eval", transport=lambda *a: {"reply": "x"})
    assert rt.model.name == "localhost-3000-eval"
    rt2 = http_target("http://x/eval", ref="mojoe-coach", transport=lambda *a: {"reply": "x"})
    assert rt2.model.name == "mojoe-coach"


def test_flag_resolves_to_http_adapter():
    rt = cli._resolve_run_target(
        _args(["run", "--http-target", "http://localhost:3000/eval",
               "--header", "x-eval-secret:s", "--ref", "mojoe-coach"]),
        ProjectConfig(),
    )
    assert rt.adapter == "http"
    assert rt.ref == "mojoe-coach"
    assert rt.conversation == "stateless"
    assert rt.meta["url"] == "http://localhost:3000/eval"
    assert not rt.is_mock


def test_config_http_target_resolves(monkeypatch):
    monkeypatch.setenv("EVAL_SECRET", "from-env")
    cfg = ProjectConfig(target={"http": "http://localhost:3000/eval",
                                "headers": {"x-eval-secret": "${EVAL_SECRET}"},
                                "conversation": "session", "ref": "mojoe-coach"})
    # a bare `run` (no target flags) picks up the config-declared endpoint.
    rt = cli._resolve_run_target(_args(["run"]), cfg)
    assert rt.adapter == "http"
    assert rt.meta["url"] == "http://localhost:3000/eval"
    assert rt.ref == "mojoe-coach"
    assert rt.conversation == "session"
    # passing --header alongside a config endpoint is allowed (merges, does not error).
    assert cli._resolve_run_target(_args(["run", "--header", "x-eval-secret:override"]), cfg).adapter == "http"


def test_http_target_conflicts_with_model():
    with pytest.raises(ValueError, match="cannot be combined"):
        cli._resolve_run_target(
            _args(["run", "--http-target", "http://x/eval", "--model", "openai/gpt-5.4-mini"]),
            ProjectConfig(),
        )


def test_header_without_http_target_errors():
    with pytest.raises(ValueError, match="apply to an HTTP target"):
        cli._resolve_run_target(_args(["run", "--target", "mock", "--header", "a:b"]), ProjectConfig())


def test_no_target_resolves_none():
    assert cli._resolve_run_target(_args(["run"]), ProjectConfig()) is None


# --- dry-run makes no network call ---------------------------------------
def test_dry_run_http_target_no_call(capsys):
    rc = main(["run", "--http-target", "http://localhost:9/eval", "--header", "x-eval-secret:s",
               "--judges", "local", "--quick", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run estimate" in out
    assert "http://localhost:9/eval" in out
    assert "hit your server" in out  # honest cost note for a self-hosted endpoint


# --- doctor over an HTTP target (frontier lane; no Ollama call) -----------
def test_doctor_http_target_needs_no_target_key(capsys):
    # rc depends on whether frontier keys happen to be present, so assert the branch's
    # deterministic output: an HTTP target is reported as needing no provider key of its own.
    main(["doctor", "--http-target", "http://localhost:3000/eval", "--judges", "gold", "--no-probe"])
    out = capsys.readouterr().out
    assert "HTTP endpoint http://localhost:3000/eval" in out
    assert "no provider key needed" in out


# --- connectivity probe ---------------------------------------------------
@pytest.mark.parametrize("resp,ok,kind", [
    ((200, '{"reply": "hey, that sounds hard"}'), True, "ok"),
    ((200, '{"choices": [{"message": {"content": "hi"}}]}'), True, "ok"),
    ((404, ""), False, "not_found"),
    ((401, "unauthorized"), False, "unauthorized"),
    ((403, ""), False, "unauthorized"),
    ((500, "boom"), False, "http_error"),
    ((200, "not-json-at-all"), False, "bad_shape"),
])
def test_probe_classifies(resp, ok, kind):
    r = probe_endpoint("http://x/eval", {}, poster=lambda *a: resp)
    assert r["ok"] is ok
    assert r["kind"] == kind


def test_probe_unreachable_and_timeout():
    def refused(*a):
        raise ConnectionRefusedError("refused")

    def timed_out(*a):
        raise TimeoutError("timed out")

    assert probe_endpoint("http://x/eval", poster=refused)["kind"] == "unreachable"
    assert probe_endpoint("http://x/eval", poster=timed_out)["kind"] == "timeout"


def test_doctor_probe_reports_reachable(monkeypatch, capsys):
    monkeypatch.setattr(targets, "_raw_post", lambda *a: (200, '{"reply": "ok"}'))
    main(["doctor", "--http-target", "http://localhost:3000/eval", "--judges", "gold"])
    out = capsys.readouterr().out
    assert "probing http://localhost:3000/eval" in out
    assert "endpoint: OK" in out


def test_doctor_probe_failure_flips_readiness(monkeypatch, capsys):
    # a failed probe makes doctor non-ready (rc 1) even when frontier keys are present.
    monkeypatch.setattr(targets, "_raw_post", lambda *a: (_ for _ in ()).throw(ConnectionRefusedError()))
    rc = main(["doctor", "--http-target", "http://localhost:3000/eval", "--judges", "gold"])
    out = capsys.readouterr().out
    assert "endpoint: FAIL" in out
    assert rc == 1


def test_doctor_no_probe_skips_the_call(monkeypatch, capsys):
    monkeypatch.setattr(targets, "_raw_post",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not probe")))
    main(["doctor", "--http-target", "http://localhost:3000/eval", "--no-probe", "--judges", "gold"])
    assert "probing" not in capsys.readouterr().out


# --- init scaffold --------------------------------------------------------
def test_init_http_ungated(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    assert main(["init", "--http-target", "http://localhost:3000/eval", "--out", str(out)]) == 0
    cfg = load_config(out)
    assert isinstance(cfg.target, HttpTargetSpec)
    assert cfg.target.http == "http://localhost:3000/eval"
    assert cfg.target.headers == {}
    assert cfg.judges == "local"


def test_init_gated_writes_env_reference_not_value(tmp_path, monkeypatch):
    out = tmp_path / "aipsy-bench.yaml"
    assert main(["init", "--http-target", "https://staging/eval", "--secret-header", "x-eval-secret",
                 "--secret-env", "EVAL_SECRET", "--out", str(out)]) == 0
    text = out.read_text()
    assert "x-eval-secret: ${EVAL_SECRET}" in text   # a reference, never the value
    monkeypatch.setenv("EVAL_SECRET", "shh")
    assert load_config(out).target.headers == {"x-eval-secret": "shh"}


def test_init_model_target(tmp_path):
    out = tmp_path / "c.yaml"
    assert main(["init", "--model", "openai/gpt-5.4-mini", "--judges", "gold", "--out", str(out)]) == 0
    cfg = load_config(out)
    assert cfg.target == "openai/gpt-5.4-mini"
    assert cfg.judges == "gold"


def test_init_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    out.write_text("target: mock\n")
    assert main(["init", "--http-target", "http://x/eval", "--out", str(out)]) == 2
    assert out.read_text() == "target: mock\n"       # untouched
    assert main(["init", "--http-target", "http://x/eval", "--out", str(out), "--force"]) == 0
    assert isinstance(load_config(out).target, HttpTargetSpec)


def test_init_secret_header_requires_env(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    assert main(["init", "--http-target", "http://x/eval", "--secret-header", "x-eval-secret",
                 "--out", str(out)]) == 2
    assert not out.exists()
