"""Local judge — the offline, self-contained DEFAULT panel (exp 016-local-judge).

The fine-tuned ``aipsy-judge-1.0`` model (public HF repo ``keidolabs/aipsy-judge-1.0``;
a LoRA fine-tune of a Gemma base) served by a local Ollama server. We call
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

import asyncio
import inspect as _inspect
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import httpx
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model

from . import spec

# (url, json_payload, timeout_s) -> response dict; sync OR async (awaited if a coroutine).
# Injectable offline for tests.
Transport = Callable[[str, dict, "float | None"], object]


class LocalJudgeUnavailable(RuntimeError):
    """The local Ollama judge could not be reached / served (setup incomplete)."""


_SETUP_HINT = (
    f"Is Ollama running and the '{spec.LOCAL_JUDGE_TAG}' model installed?\n"
    "  setup:  aipsy-bench judge pull        check:  aipsy-bench doctor --judges local"
)


# --------------------------------------------------------------------------
# The judge model (native Ollama /api/chat)
# --------------------------------------------------------------------------
async def _async_post(url: str, payload: dict, timeout: float | None) -> dict:
    """The SCORING transport — async so a judge call never blocks Inspect's event loop.
    A blocking call would freeze the live TUI for the whole (10–40 s) generation; awaiting
    httpx yields control so the UI stays responsive. httpx is an inspect-ai dependency."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, OSError) as e:
        raise LocalJudgeUnavailable(
            f"could not reach the local Ollama judge at {url} ({e}).\n{_SETUP_HINT}"
        ) from e


def _sync_post(url: str, payload: dict, timeout: float | None) -> dict:
    """The PREFLIGHT transport (``warm_up``) — sync urllib, runs in the CLI off the event loop."""
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


def _ollama_options(config: GenerateConfig, num_ctx: int) -> dict:
    """Ollama call options — the frozen 016 inference contract (a call-time override
    of the Modelfile defaults). temperature/max_tokens flow from the frozen JUDGE_*
    constants via the GenerateConfig the scorer builds; num_ctx is resolved by the
    caller (the default, or a --num-ctx override)."""
    temperature = config.temperature if config.temperature is not None else spec.JUDGE_TEMPERATURE
    num_predict = config.max_tokens if config.max_tokens is not None else spec.JUDGE_MAX_TOKENS
    return {
        "temperature": temperature,
        "top_p": 1.0,                          # 016 inference (overrides the Modelfile's 0.95)
        "num_predict": num_predict,
        "num_ctx": num_ctx,                    # rubric (~5k) + full history + output must fit
        "seed": spec.LOCAL_JUDGE_SEED,         # reproducibility
    }


def local_judge_model(
    *, tag: str | None = None, transport: Transport | None = None, num_ctx: int | None = None
) -> Model:
    """The local FT judge as an Inspect model (native Ollama ``/api/chat``).

    The scorer passes ``[system, user]`` messages + a GenerateConfig; we forward them to
    Ollama with the frozen options. The call is **async** (httpx) so it never blocks Inspect's
    event loop — the live TUI stays responsive while the model generates. Requests are
    **serialized** (one in flight at a time) to match the validated sequential 016 inference and
    avoid running multiple KV caches at once (OOM on a memory-tight box). ``transport`` is
    injectable for tests (sync or async); a connection failure raises ``LocalJudgeUnavailable``
    (caught by the scorer as a degraded judge failure — the run preflight checks Ollama up-front).
    """
    model_tag = tag or spec.LOCAL_JUDGE_TAG
    ctx = num_ctx or spec.LOCAL_JUDGE_NUM_CTX
    send: Transport = transport or _async_post
    url = f"{spec.OLLAMA_BASE_URL}/api/chat"
    sem = asyncio.Semaphore(1)  # one Ollama request in flight at a time (non-blocking)

    async def _outputs(messages, tools, tool_choice, config):
        # Preserve system + user roles exactly (== open_judges: [{system}, *messages]).
        msgs = [{"role": m.role, "content": m.text} for m in messages]
        payload = {
            "model": model_tag,
            "messages": msgs,
            "stream": False,
            "think": False,                    # disable thinking (016 inference)
            "keep_alive": spec.LOCAL_JUDGE_KEEP_ALIVE,  # stay resident across the battery
            "options": _ollama_options(config, ctx),
        }
        base = config.timeout if config.timeout is not None else spec.MODEL_TIMEOUT
        timeout = max(base, spec.LOCAL_JUDGE_TIMEOUT)  # floor: absorb cold load + slow local gen
        async with sem:
            result = send(url, payload, timeout)
            if _inspect.isawaitable(result):
                result = await result
        content = ((result or {}).get("message") or {}).get("content", "")
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


def warm_up(*, tag: str | None = None, timeout: float | None = None, num_ctx: int | None = None) -> bool:
    """Force-load the model into memory so the first SCORED call doesn't hit a cold-load
    timeout (paging ~27 GB in can take minutes). Sends a 1-token generate with keep_alive so
    the model stays resident for the run — loaded at the SAME num_ctx the run will use, so the
    scored calls don't trigger an Ollama reload. Returns True if the model responded."""
    tag = tag or spec.LOCAL_JUDGE_TAG
    ctx = num_ctx or spec.LOCAL_JUDGE_NUM_CTX
    url = f"{spec.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": tag,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "keep_alive": spec.LOCAL_JUDGE_KEEP_ALIVE,
        "options": {"num_predict": 1, "num_ctx": ctx},
    }
    try:
        _sync_post(url, payload, timeout if timeout is not None else spec.LOCAL_JUDGE_TIMEOUT)
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


def detect_platform() -> str:
    """``mac`` = Apple-Silicon unified memory (RAM is the accelerator) · ``linux`` /
    ``other`` = discrete-GPU box (VRAM is the gate). Drives the viability heuristic."""
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mac"
    return "linux" if platform.system() == "Linux" else "other"


def gpu_vram_gb() -> float | None:
    """Best-effort largest single-GPU VRAM in GB via ``nvidia-smi`` (NVIDIA/CUDA).
    ``None`` when no NVIDIA GPU or the tool is absent — i.e. CPU-only, which is
    unusable for a 26B judge. AMD/ROCm isn't probed (treated as no-GPU)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]  # MiB per GPU
        return max(vals) / 1024 if vals else None
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


def viability() -> dict:
    """Can THIS machine run the local judge at a USABLE speed? Pure hardware detection
    (no scored calls) that powers the ``doctor``/``init`` path recommendation. The local
    judge is a 26B model, so "fits in memory" isn't enough — the gate is platform-aware:

    * **Mac (Apple-Silicon unified memory)** — RAM is the accelerator. ``ready`` ≥48 GB;
      ``slow`` 32–48 GB (loads but unsurvivably slow → steer to API, not recommended);
      ``insufficient`` <32 GB.
    * **Linux / other (discrete GPU)** — VRAM is the gate (a 26B on CPU is unusable).
      ``ready`` ≥16 GB VRAM & ≥64 GB RAM; ``tight`` ≥16 GB VRAM but <64 GB RAM;
      ``insufficient`` GPU present but <16 GB VRAM; ``no_gpu`` no CUDA GPU detected.

    Ollama presence is reported separately and is NEVER a viability blocker — a missing
    Ollama/model is a one-time install step, not a reason to abandon the local path.
    """
    plat = detect_platform()
    ram = total_ram_gb()
    ram_str = f"{ram:.0f} GB" if ram else "unknown RAM"
    running = ollama_running()
    common = {
        "platform": plat,
        "ram_gb": ram,
        "ollama_running": running,
        "model_present": model_present() if running else False,
    }

    if plat == "mac":
        req = f"{spec.LOCAL_JUDGE_MAC_RAM_MIN_GB} GB unified memory (Apple Silicon)"
        if ram is None:
            band, detail = "unknown", "could not read system memory"
        elif ram >= spec.LOCAL_JUDGE_MAC_RAM_MIN_GB:
            band, detail = "ready", f"{ram_str} unified memory (≥ {spec.LOCAL_JUDGE_MAC_RAM_MIN_GB} GB)"
        elif ram >= spec.LOCAL_JUDGE_MAC_RAM_SLOW_GB:
            band, detail = "slow", (f"{ram_str} unified memory — loads but runs a 26B judge "
                                    f"unusably slowly ({spec.LOCAL_JUDGE_MAC_RAM_MIN_GB} GB+ recommended)")
        else:
            band, detail = "insufficient", f"{ram_str} unified memory < {spec.LOCAL_JUDGE_MAC_RAM_SLOW_GB} GB"
        return {**common, "vram_gb": None, "band": band, "viable": band == "ready",
                "requirement": req, "detail": detail}

    # discrete-GPU box (linux/other): VRAM gates; system RAM is secondary
    vram = gpu_vram_gb()
    req = f"{spec.LOCAL_JUDGE_GPU_VRAM_MIN_GB} GB VRAM + {spec.LOCAL_JUDGE_GPU_RAM_MIN_GB} GB RAM"
    vram_str = f"{vram:.1f} GB" if vram else None  # honest precision — a 16 GB card reads ~15.9
    # Tolerance: nvidia-smi under-reports the advertised tier, so a real 16 GB card ("~15.9")
    # must still qualify (else the exact-16 GB box gets misdiagnosed and pushed to the API lane).
    vram_ok = vram is not None and vram >= spec.LOCAL_JUDGE_GPU_VRAM_MIN_GB - spec.LOCAL_JUDGE_GPU_VRAM_TOLERANCE_GB
    if vram is None:
        band, detail, viable = "no_gpu", "no CUDA GPU detected — a 26B judge on CPU is unusably slow", False
    elif not vram_ok:
        band, detail, viable = "insufficient", f"GPU VRAM {vram_str} is below the {spec.LOCAL_JUDGE_GPU_VRAM_MIN_GB} GB needed", False
    elif ram is not None and ram < spec.LOCAL_JUDGE_GPU_RAM_MIN_GB:
        band, detail, viable = "tight", (f"{vram_str} VRAM (meets ~{spec.LOCAL_JUDGE_GPU_VRAM_MIN_GB} GB); "
                                         f"{ram_str} RAM < {spec.LOCAL_JUDGE_GPU_RAM_MIN_GB} GB (tight)"), True
    else:
        band, detail, viable = "ready", f"{vram_str} VRAM + {ram_str} RAM", True
    return {**common, "vram_gb": vram, "band": band, "viable": viable,
            "requirement": req, "detail": detail}


# --------------------------------------------------------------------------
# Onboarding — pull the published GGUF + register the Ollama tag
# --------------------------------------------------------------------------
def _manual_setup() -> str:
    return (
        "Manual setup:\n"
        "  1. Install + start Ollama (https://ollama.com):   ollama serve\n"
        f"  2. Download the GGUF + Modelfile (public repo — no token needed):\n"
        f"       hf download {spec.LOCAL_JUDGE_HF_REPO} {spec.LOCAL_JUDGE_GGUF} "
        f"{spec.LOCAL_JUDGE_MODELFILE} --local-dir ./aipsy-judge\n"
        f"  3. Register the tag (the published GGUF is already {spec.LOCAL_JUDGE_QUANT}):\n"
        f"       cd ./aipsy-judge/gguf && ollama create {spec.LOCAL_JUDGE_TAG} -f Modelfile\n"
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
        log("huggingface_hub is missing (it is a base dependency) — reinstall aipsy-bench, "
            f"or run `pip install huggingface-hub`.\n\n{_manual_setup()}")
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
