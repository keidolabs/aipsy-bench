"""Local judge — the offline, self-contained DEFAULT panel (exp 016-local-judge).

The fine-tuned ``gemma4-judge-ft-v3`` model served by a local Ollama server. We call
Ollama's NATIVE ``/api/chat`` over stdlib ``urllib`` (no extra runtime deps for
scoring), reproducing the 016/015 served inference contract byte-for-byte
(``open_judges.py`` ``OllamaProvider.complete``): system + user roles (the gemma4
renderer handles the system message), ``num_ctx`` 8192, ``seed`` 14, thinking
disabled, ``top_p`` 1.0, and temperature / max_tokens from the frozen ``JUDGE_*``
constants. The model is wrapped as an Inspect model via the same ``mockllm/model`` +
``custom_outputs`` shim that ``mocks``/``targets`` use, so the scorer's ensemble /
composite path is reused unchanged.

Onboarding (``aipsy-bench judge pull``) fetches the published Q8_0 GGUF + Modelfile
from the (private) HF repo using ``HF_TOKEN`` and registers the Ollama tag. Scoring
itself needs only a running Ollama with the tag present — no key, no network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model

from . import spec

# (url, json_payload, timeout_s) -> parsed JSON response dict. Injectable offline.
Transport = Callable[[str, dict, "float | None"], dict]


class LocalJudgeUnavailable(RuntimeError):
    """The local Ollama judge could not be reached / served (setup incomplete)."""


_SETUP_HINT = (
    f"Is Ollama running and the '{spec.LOCAL_JUDGE_TAG}' model installed?\n"
    "  setup:  aipsy-bench judge pull        check:  aipsy-bench doctor --judges local"
)


# --------------------------------------------------------------------------
# The judge model (native Ollama /api/chat)
# --------------------------------------------------------------------------
def _default_transport(url: str, payload: dict, timeout: float | None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — localhost endpoint
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as e:
        raise LocalJudgeUnavailable(
            f"could not reach the local Ollama judge at {url} ({e}).\n{_SETUP_HINT}"
        ) from e


def _ollama_options(config: GenerateConfig) -> dict:
    """Ollama call options — the frozen 016 inference contract (a call-time override
    of the Modelfile defaults). temperature/max_tokens flow from the frozen JUDGE_*
    constants via the GenerateConfig the scorer builds."""
    temperature = config.temperature if config.temperature is not None else spec.JUDGE_TEMPERATURE
    num_predict = config.max_tokens if config.max_tokens is not None else spec.JUDGE_MAX_TOKENS
    return {
        "temperature": temperature,
        "top_p": 1.0,                          # 016 inference (overrides the Modelfile's 0.95)
        "num_predict": num_predict,
        "num_ctx": spec.LOCAL_JUDGE_NUM_CTX,   # the rubric is ~5k tokens — never truncate it
        "seed": spec.LOCAL_JUDGE_SEED,         # reproducibility
    }


def local_judge_model(*, tag: str | None = None, transport: Transport | None = None) -> Model:
    """The local FT judge as an Inspect model (native Ollama ``/api/chat``).

    The scorer passes ``[system, user]`` messages + a GenerateConfig; we forward them
    to Ollama with the frozen options. ``transport`` is injectable so offline tests
    exercise the request/parse path with no server. A connection failure raises
    ``LocalJudgeUnavailable`` (caught by the scorer as a judge failure — degraded, not
    crashed); the run preflight checks Ollama up-front so this is rare in practice.
    """
    model_tag = tag or spec.LOCAL_JUDGE_TAG
    send: Transport = transport or _default_transport
    url = f"{spec.OLLAMA_BASE_URL}/api/chat"

    def _outputs(messages, tools, tool_choice, config):
        # Preserve system + user roles exactly (== open_judges: [{system}, *messages]).
        msgs = [{"role": m.role, "content": m.text} for m in messages]
        payload = {
            "model": model_tag,
            "messages": msgs,
            "stream": False,
            "think": False,                    # disable thinking (016 inference)
            "keep_alive": spec.LOCAL_JUDGE_KEEP_ALIVE,  # stay resident across the battery
            "options": _ollama_options(config),
        }
        base = config.timeout if config.timeout is not None else spec.MODEL_TIMEOUT
        timeout = max(base, spec.LOCAL_JUDGE_TIMEOUT)  # floor: absorb cold load + slow local gen
        body = send(url, payload, timeout)
        content = ((body or {}).get("message") or {}).get("content", "")
        return ModelOutput.from_content(model=f"ollama/{model_tag}", content=content)

    return get_model("mockllm/model", custom_outputs=_outputs)


# --------------------------------------------------------------------------
# Health / status (stdlib; no server dependency to import)
# --------------------------------------------------------------------------
def _get_json(path: str, timeout: float = 5.0) -> dict | None:
    url = f"{spec.OLLAMA_BASE_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost endpoint
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def ollama_running() -> bool:
    return _get_json("/api/version") is not None


def installed_tags() -> list[str]:
    data = _get_json("/api/tags") or {}
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def model_present(tag: str | None = None) -> bool:
    tag = tag or spec.LOCAL_JUDGE_TAG
    return any(n == tag or n.split(":", 1)[0] == tag for n in installed_tags())


def warm_up(*, tag: str | None = None, timeout: float | None = None) -> bool:
    """Force-load the model into memory so the first SCORED call doesn't hit a cold-load
    timeout (paging ~27 GB in can take minutes). Sends a 1-token generate with keep_alive
    so the model stays resident for the run. Returns True if the model responded."""
    tag = tag or spec.LOCAL_JUDGE_TAG
    url = f"{spec.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": tag,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "keep_alive": spec.LOCAL_JUDGE_KEEP_ALIVE,
        "options": {"num_predict": 1, "num_ctx": spec.LOCAL_JUDGE_NUM_CTX},
    }
    try:
        _default_transport(url, payload, timeout if timeout is not None else spec.LOCAL_JUDGE_TIMEOUT)
        return True
    except LocalJudgeUnavailable:
        return False


def total_ram_gb() -> float | None:
    """Best-effort physical RAM in GB (for the doctor memory warning)."""
    try:  # Linux
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip()) / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def status() -> dict:
    """Snapshot for ``doctor`` / ``judge status`` — no scored calls."""
    running = ollama_running()
    present = model_present() if running else False
    ram = total_ram_gb()
    return {
        "running": running,
        "model_present": present,
        "tag": spec.LOCAL_JUDGE_TAG,
        "version": spec.LOCAL_JUDGE_VERSION,
        "quant": spec.LOCAL_JUDGE_QUANT,
        "ram_gb": ram,
        "ram_tight": ram is not None and ram < spec.LOCAL_JUDGE_RAM_RECOMMENDED_GB,
    }


# --------------------------------------------------------------------------
# Onboarding — pull the published GGUF + register the Ollama tag
# --------------------------------------------------------------------------
def _manual_setup() -> str:
    return (
        "Manual setup:\n"
        "  1. Install + start Ollama (https://ollama.com):   ollama serve\n"
        f"  2. Download the GGUF + Modelfile (needs HF_TOKEN for the private repo):\n"
        f"       hf download {spec.LOCAL_JUDGE_HF_REPO} {spec.LOCAL_JUDGE_GGUF} "
        f"{spec.LOCAL_JUDGE_MODELFILE} --local-dir ./gemma4-judge-ft\n"
        f"  3. Register the tag (the published GGUF is already {spec.LOCAL_JUDGE_QUANT}):\n"
        f"       cd ./gemma4-judge-ft/gguf && ollama create {spec.LOCAL_JUDGE_TAG} -f Modelfile\n"
        "  4. Verify:   aipsy-bench judge status"
    )


def _materialize_modelfile(downloaded: Path, gguf_path: Path, dest: Path) -> Path:
    """Copy the published Modelfile, rewriting FROM to the absolute GGUF path and
    appending ``PARAMETER num_ctx`` if absent (the HF recipe omits it; a smaller
    served ctx would truncate the ~5k-token rubric)."""
    lines: list[str] = []
    saw_from = saw_ctx = False
    for line in downloaded.read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            lines.append(f"FROM {gguf_path}")
            saw_from = True
        else:
            lines.append(line)
            if "num_ctx" in line:
                saw_ctx = True
    if not saw_from:
        lines.insert(0, f"FROM {gguf_path}")
    if not saw_ctx:
        lines.append(f"PARAMETER num_ctx {spec.LOCAL_JUDGE_NUM_CTX}")
    out = dest / "Modelfile.aipsy"
    out.write_text("\n".join(lines) + "\n")
    return out


def ensure_model(
    *,
    token: str | None = None,
    force: bool = False,
    cache_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Pull + register the local FT judge in Ollama (idempotent).

    Never raises on a missing optional dependency / tool — prints the manual recipe
    and returns ``{"status": "manual", ...}`` instead. Returns ``"ready"`` if already
    installed, ``"created"`` on success.
    """
    tag = spec.LOCAL_JUDGE_TAG
    if not ollama_running():
        log(f"Ollama is not running — start it with:  ollama serve\n\n{_manual_setup()}")
        return {"status": "manual", "reason": "ollama_not_running"}
    if model_present(tag) and not force:
        log(f"'{tag}' is already installed in Ollama — nothing to do.")
        return {"status": "ready", "tag": tag}
    if shutil.which("ollama") is None:
        log(f"the 'ollama' CLI is not on PATH.\n\n{_manual_setup()}")
        return {"status": "manual", "reason": "no_ollama_cli"}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log("huggingface_hub is not installed — install the optional extra:\n"
            f"  uv sync --extra local\n\n{_manual_setup()}")
        return {"status": "manual", "reason": "no_hf_hub"}

    token = token or os.environ.get("HF_TOKEN")
    dest = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "aipsy-bench" / "local-judge"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        log(f"downloading {spec.LOCAL_JUDGE_GGUF} (~{spec.LOCAL_JUDGE_RAM_GB} GB) "
            f"from {spec.LOCAL_JUDGE_HF_REPO} … (resumable)")
        gguf_path = Path(hf_hub_download(
            spec.LOCAL_JUDGE_HF_REPO, spec.LOCAL_JUDGE_GGUF, token=token, local_dir=str(dest)))
        modelfile_dl = Path(hf_hub_download(
            spec.LOCAL_JUDGE_HF_REPO, spec.LOCAL_JUDGE_MODELFILE, token=token, local_dir=str(dest)))
    except Exception as e:  # noqa: BLE001 — auth/network/repo errors → manual guidance, don't crash
        log(f"download failed ({type(e).__name__}: {e}).\n\n{_manual_setup()}")
        return {"status": "manual", "reason": "download_failed"}

    modelfile = _materialize_modelfile(modelfile_dl, gguf_path, dest)
    log(f"registering Ollama tag '{tag}' ({spec.LOCAL_JUDGE_QUANT}) …")
    try:
        subprocess.run(["ollama", "create", tag, "-f", str(modelfile)],
                       check=True, cwd=str(modelfile.parent))
    except subprocess.CalledProcessError as e:
        log(f"`ollama create` failed (exit {e.returncode}).\n\n{_manual_setup()}")
        return {"status": "manual", "reason": "ollama_create_failed"}
    log(f"done — '{tag}' is ready. Verify: aipsy-bench judge status")
    return {"status": "created", "tag": tag, "gguf": str(gguf_path)}
