"""Provider-selectable single-judge lane + hardware path recommendation. Offline.

`single` is now `single:<provider>` (the frontier key a dev holds becomes their default),
staying a NON-comparable directional lane. `doctor`/`init` recommend local vs API from the
machine's RAM against the 26B local-judge thresholds.
"""

from __future__ import annotations

import pytest
from fixtures import mock_judge_model, mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import card, local_judge, report, spec
from aipsy_bench.cli import main
from aipsy_bench.compare import CompareError, compare
from aipsy_bench.dataset import build_dataset
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.validation import load_validation


# ---- spec.parse_panel / panel_base / JUDGE_CHOICES ----

def test_parse_panel_canonicalizes_and_validates():
    assert spec.parse_panel("local") == ("local", ("local",))
    assert spec.parse_panel("gold") == ("gold", spec.PROVIDERS)
    assert spec.parse_panel("single") == ("single", ("openai",))
    # single:<primary> collapses to plain single (same lane / same instrument)
    assert spec.parse_panel("single:openai") == ("single", ("openai",))
    assert spec.parse_panel("single:anthropic") == ("single:anthropic", ("anthropic",))
    assert spec.parse_panel("single:google") == ("single:google", ("google",))
    for bad in ("single:bogus", "nonsense", "gold:openai", "local:x"):
        with pytest.raises(ValueError):
            spec.parse_panel(bad)


def test_panel_base_and_choices():
    assert spec.panel_base("single:anthropic") == "single"
    assert spec.panel_base("gold") == "gold"
    assert spec.panel_base("local") == "local"
    assert {"single:openai", "single:anthropic", "single:google"} <= set(spec.JUDGE_CHOICES)


# ---- local_judge.viability bands (the platform-aware 26B hardware gate) ----

@pytest.mark.parametrize("ram,band,viable", [
    (64.0, "ready", True),
    (48.0, "ready", True),
    (36.0, "slow", False),          # 32–48 GB unified: loads but unusably slow → not recommended
    (16.0, "insufficient", False),
    (None, "unknown", False),
])
def test_viability_mac_bands(monkeypatch, ram, band, viable):
    monkeypatch.setattr(local_judge, "detect_platform", lambda: "mac")
    monkeypatch.setattr(local_judge, "total_ram_gb", lambda: ram)
    monkeypatch.setattr(local_judge, "ollama_running", lambda: False)
    v = local_judge.viability()
    assert v["platform"] == "mac"
    assert v["band"] == band
    assert v["viable"] is viable


@pytest.mark.parametrize("vram,ram,band,viable", [
    (24.0, 128.0, "ready", True),
    (15.9, 128.0, "ready", True),       # a 16 GB card under-reports as ~15.9 → still qualifies
    (16.0, 32.0, "tight", True),        # VRAM ok but system RAM < 64 GB
    (12.0, 128.0, "insufficient", False),  # genuinely below the 16 GB tier
    (None, 128.0, "no_gpu", False),       # CPU-only → unusable for a 26B judge
])
def test_viability_linux_bands(monkeypatch, vram, ram, band, viable):
    monkeypatch.setattr(local_judge, "detect_platform", lambda: "linux")
    monkeypatch.setattr(local_judge, "gpu_vram_gb", lambda: vram)
    monkeypatch.setattr(local_judge, "total_ram_gb", lambda: ram)
    monkeypatch.setattr(local_judge, "ollama_running", lambda: False)
    v = local_judge.viability()
    assert v["band"] == band
    assert v["viable"] is viable


def test_viability_ollama_absence_is_not_a_blocker(monkeypatch):
    # a capable box with Ollama not yet installed stays VIABLE (install is a step, not a gate)
    monkeypatch.setattr(local_judge, "detect_platform", lambda: "mac")
    monkeypatch.setattr(local_judge, "total_ram_gb", lambda: 64.0)
    monkeypatch.setattr(local_judge, "ollama_running", lambda: False)
    v = local_judge.viability()
    assert v["viable"] is True
    assert v["ollama_running"] is False


# ---- scorer resolves single:<provider> + records the CANONICAL judge_panel ----

def _log(tmp_path, target="safe", *, panel="single", scenario_ids=None):
    _, providers = spec.parse_panel(panel)
    judges = {p: mock_judge_model() for p in providers}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids or ["s01"]),
        solver=scripted_dialogue(),
        scorer=clinical_judge_panel(panel=panel, judges=judges, cache=False),
    )
    return inspect_eval(task, model=mock_target_model(target), display="none", log_dir=str(tmp_path))[0]


def _panel_of(log):
    return log.samples[0].scores["clinical_judge_panel"].metadata["judge_panel"]


def test_single_provider_selectable_and_canonical(tmp_path):
    assert _panel_of(_log(tmp_path / "a", panel="single:anthropic")) == "single:anthropic"
    assert _panel_of(_log(tmp_path / "b", panel="single:openai")) == "single"  # canonicalized
    assert _panel_of(_log(tmp_path / "c", panel="single")) == "single"


def test_task_builder_accepts_single_provider():
    from aipsy_bench.task import aipsy_bench

    # the task wiring must validate + build mock judges for the chosen provider
    task = aipsy_bench(target="mock", judges="single:google", scenario_ids=["s01"])
    assert task is not None


# ---- report warning names the provider; card reproduces the exact lane ----

def test_report_and_card_surface_single_provider(tmp_path):
    log = _log(tmp_path, panel="single:anthropic")
    result = report.to_result_json(log, validation=load_validation())
    assert result["judge_panel"] == "single:anthropic"
    assert any("single-judge panel (anthropic" in w for w in result["warnings"])
    assert "NOT comparable to published gold numbers" in " ".join(result["warnings"])
    assert "--judges single:anthropic" in card.reproduce_command(result)


# ---- compare: different single providers are different instruments (separate lanes) ----

def test_compare_refuses_cross_single_provider(tmp_path):
    a = _log(tmp_path / "a", panel="single:anthropic")
    b = _log(tmp_path / "b", panel="single:openai")  # canonical "single"
    with pytest.raises(CompareError):
        compare(a, b)


def test_compare_allows_same_single_provider(tmp_path):
    a = _log(tmp_path / "a", "safe", panel="single:anthropic")
    b = _log(tmp_path / "b", "failing", panel="single:anthropic")
    diff = compare(a, b)  # same lane → comparable, no raise
    assert diff["overall"]["AI_Trust"]["direction"] in ("↑", "↓", "→")


# ---- flag-driven init writes the chosen lane as the run default ----

def test_init_writes_single_provider_default(tmp_path):
    out = tmp_path / "aipsy-bench.yaml"
    rc = main(["init", "--model", "openai/gpt-5.4-mini",
               "--judges", "single:anthropic", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "judges: single:anthropic" in text


# ---- doctor path recommendation (offline; RAM + Ollama monkeypatched) ----

def _run_doctor(monkeypatch, capsys, ram):
    monkeypatch.setattr(local_judge, "detect_platform", lambda: "mac")  # deterministic across CI
    monkeypatch.setattr(local_judge, "total_ram_gb", lambda: ram)
    monkeypatch.setattr(local_judge, "ollama_running", lambda: False)
    main(["doctor", "--target", "mock"])
    return capsys.readouterr().out


def test_doctor_recommends_local_when_viable(monkeypatch, capsys):
    out = _run_doctor(monkeypatch, capsys, 64.0)
    assert "Recommended path" in out
    assert "LOCAL (priority)" in out


def test_doctor_flags_slow_mac(monkeypatch, capsys):
    out = _run_doctor(monkeypatch, capsys, 36.0)  # 32–48 GB Mac: runs but unusably slow
    assert "LOCAL possible but SLOW" in out
    assert "single:anthropic" in out  # steered to API despite technically running


def test_doctor_recommends_api_when_insufficient(monkeypatch, capsys):
    out = _run_doctor(monkeypatch, capsys, 8.0)
    assert "LOCAL not viable" in out
    assert "single:anthropic" in out  # the chosen-provider option is surfaced
    assert "gold" in out               # the comparable panel is still offered


# ---- interactive init: hardware-aware lane selection (monkeypatched stdin) ----

def _fake_inputs(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


def _fake_getpass(monkeypatch, value=""):
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: value)


def _no_save(*a, **k):
    raise AssertionError("set_provider_key must NOT be called")


def _viability(monkeypatch, *, viable, band):
    from aipsy_bench import keys, local_judge
    monkeypatch.setattr(local_judge, "viability",
                        lambda: {"viable": viable, "band": band, "detail": "detail",
                                 "requirement": "48 GB unified memory",
                                 "ollama_running": False, "model_present": False})
    monkeypatch.setattr(keys, "current_keys", lambda: {p: False for p in spec.PROVIDERS})


def test_init_prompt_local_when_viable(monkeypatch):
    from aipsy_bench import cli
    _viability(monkeypatch, viable=True, band="ready")
    _fake_inputs(monkeypatch, ["1"])  # pick local
    assert cli._init_prompt_judges() == "local"


def test_init_prompt_single_provider_when_viable(monkeypatch):
    from aipsy_bench import cli
    _viability(monkeypatch, viable=True, band="ready")
    _fake_inputs(monkeypatch, ["2", "2"])  # single → anthropic
    _fake_getpass(monkeypatch, "")          # hidden key prompt: Enter → skip
    assert cli._init_prompt_judges() == "single:anthropic"


# ---- inline key-set during init: hidden prompt, never a visible confirm (the QA bug) ----

def test_maybe_set_key_saves_via_hidden_prompt(monkeypatch, tmp_path):
    from aipsy_bench import cli, keys
    saved = {}
    monkeypatch.setattr(keys, "set_provider_key",
                        lambda p, v, **k: saved.update(p=p, v=v) or (tmp_path / ".env"))
    _fake_getpass(monkeypatch, "sk-test-abc123")
    # a secret must NEVER be read via a visible input() prompt
    monkeypatch.setattr("builtins.input", _no_save)
    cli._maybe_set_key("openai")
    assert saved == {"p": "openai", "v": "sk-test-abc123"}


def test_maybe_set_key_skip_on_empty(monkeypatch):
    from aipsy_bench import cli, keys
    monkeypatch.setattr(keys, "set_provider_key", _no_save)
    _fake_getpass(monkeypatch, "")  # Enter → skip, no save, no raise
    cli._maybe_set_key("openai")


def test_maybe_set_key_rejects_pasted_path(monkeypatch, capsys):
    from aipsy_bench import cli, keys
    monkeypatch.setattr(keys, "set_provider_key", _no_save)  # a path is not a key → don't save it
    _fake_getpass(monkeypatch, "/Users/me/proj/.env")
    cli._maybe_set_key("openai")
    assert "not saved" in capsys.readouterr().out.lower()


def test_maybe_set_key_writes_env_end_to_end(monkeypatch, tmp_path):
    # the SECOND half of the QA bug: the pasted key must actually LAND in .env — exercise the
    # real keys.set_provider_key (no mock), with env_path() resolving to this dir's ./.env.
    from aipsy_bench import cli
    monkeypatch.chdir(tmp_path)
    _fake_getpass(monkeypatch, "sk-proj-realish-KEY-999")
    cli._maybe_set_key("openai")
    env = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY" in env and "sk-proj-realish-KEY-999" in env


def test_init_prompt_api_regime_when_not_viable(monkeypatch):
    from aipsy_bench import cli, keys, local_judge
    monkeypatch.setattr(local_judge, "viability",
                        lambda: {"viable": False, "band": "insufficient", "detail": "8 GB",
                                 "requirement": "48 GB unified memory",
                                 "ollama_running": False, "model_present": False})
    # anthropic key already present → no key-set prompt, just the provider pick
    monkeypatch.setattr(keys, "current_keys",
                        lambda: {"openai": False, "anthropic": True, "google": False})
    _fake_inputs(monkeypatch, ["2"])  # not viable → straight to provider pick → anthropic
    assert cli._init_prompt_judges() == "single:anthropic"
