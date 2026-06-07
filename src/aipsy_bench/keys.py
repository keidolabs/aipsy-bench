"""Local API-key setup helper (``aipsy-bench keys ...``).

A convenience writer for your **local, gitignored ``.env``** — NOT a key store, vault,
or service. The key is written only to your ``.env``, read from the environment at run
time, and used to call the provider directly (the provider bills you). aipsy-bench never
transmits or proxies the key, and never prints the value back (§12).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, find_dotenv, set_key

from . import spec


def env_path() -> Path:
    """The ``.env`` that will be used: the one Inspect would load (search up from cwd),
    or ``./.env`` if none exists yet."""
    found = find_dotenv(usecwd=True)
    return Path(found) if found else Path.cwd() / ".env"


def mask(value: str) -> str:
    """A non-revealing preview for confirmation output (never the full key)."""
    v = value.strip()
    return f"…{v[-4:]}" if len(v) > 8 else "(set)"


def set_provider_key(provider: str, value: str, *, path: str | Path | None = None) -> Path:
    """Upsert one provider's key into the local ``.env`` (preserving other entries),
    then restrict the file to ``0600``. Returns the path written."""
    if provider not in spec.API_ENV_VARS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {list(spec.API_ENV_VARS)}")
    value = value.strip()
    if not value:
        raise ValueError("empty key")

    p = Path(path) if path else env_path()
    p.touch(exist_ok=True)
    set_key(str(p), spec.API_ENV_VARS[provider], value)
    try:
        p.chmod(0o600)  # the file now holds a secret — restrict to the owner
    except OSError:
        pass
    return p


def current_keys(path: str | Path | None = None) -> dict[str, bool]:
    """Which provider keys are present in the ``.env`` (presence only, never values)."""
    p = Path(path) if path else env_path()
    values = dotenv_values(str(p)) if p.exists() else {}
    return {prov: bool(values.get(var)) for prov, var in spec.API_ENV_VARS.items()}
