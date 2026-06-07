"""Key-setup helper — writes only to the local .env, never echoes the value. Offline."""

from __future__ import annotations

import stat

import pytest
from dotenv import get_key

from aipsy_bench import cli, keys


def test_set_provider_key_upserts_and_preserves(tmp_path):
    env = tmp_path / ".env"
    env.write_text("EXISTING=keepme\n")
    keys.set_provider_key("openai", "sk-abc123def456", path=env)
    assert get_key(str(env), "OPENAI_API_KEY") == "sk-abc123def456"
    assert get_key(str(env), "EXISTING") == "keepme"  # unrelated entry preserved


def test_set_provider_key_restricts_perms(tmp_path):
    env = tmp_path / ".env"
    keys.set_provider_key("anthropic", "sk-ant-xyz", path=env)
    mode = stat.S_IMODE(env.stat().st_mode)
    assert mode == 0o600  # the secret file is owner-only


def test_set_provider_key_validates():
    with pytest.raises(ValueError):
        keys.set_provider_key("nope", "k")
    with pytest.raises(ValueError):
        keys.set_provider_key("openai", "   ")


def test_mask_never_reveals_full_key():
    assert keys.mask("sk-supersecretvalue1234") == "…1234"
    assert "supersecret" not in keys.mask("sk-supersecretvalue1234")
    assert keys.mask("short") == "(set)"


def test_current_keys_presence_only(tmp_path):
    env = tmp_path / ".env"
    keys.set_provider_key("openai", "sk-1", path=env)
    present = keys.current_keys(path=env)
    assert present["openai"] is True
    assert present["anthropic"] is False


def test_cli_keys_set_writes_dotenv_and_doctor_sees_it(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: "sk-from-getpass-7777")
    monkeypatch.setattr(cli, "_sdk_installed", lambda p: True)  # isolate the key round-trip

    rc = cli.main(["keys", "set", "--provider", "openai"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "saved OPENAI_API_KEY" in out
    assert "…7777" in out
    assert "sk-from-getpass-7777" not in out  # the full key is never printed
    assert (tmp_path / ".env").exists()

    # the key written by `keys set` is exactly what doctor (and a real run) will read
    rc2 = cli.main(["doctor", "--target", "openai/gpt-5.4-mini", "--judges", "single"])
    assert rc2 == 0
    assert "OPENAI_API_KEY present" in capsys.readouterr().out


def test_cli_keys_set_requires_provider_when_noninteractive(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = cli.main(["keys", "set"])
    assert rc == 2
    assert "--provider required" in capsys.readouterr().err


def test_cli_keys_status_and_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    keys.set_provider_key("openai", "sk-1", path=tmp_path / ".env")
    assert cli.main(["keys", "status"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY: present" in out
    assert cli.main(["keys", "path"]) == 0
