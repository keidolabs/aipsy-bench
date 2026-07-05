"""Target adapter — the thing under test (§6).

Tiers, ordered by what actually works for a real deployed app:
- **Tier 0** — a bare Inspect model string (the launch-leaderboard path).
- **Tier 1** — HTTP, OpenAI-chat-compatible (URL + static auth header). The *easy*
  case only; this is deliberately NOT grown into an SSE / multi-step DSL (§6).
- **Tier 2** — a Python callable ``message_history -> reply``. The **recommended**
  path for real apps: a ~30-line user function owns auth handshake, session
  lifecycle, SSE reassembly, headers (see ``docs/adapters/``).

``conversation: stateless | session`` is a first-class property:
- ``stateless`` (default): the harness owns history and replays the full transcript.
- ``session``: the target owns history; the harness sends only the *new* user turn
  per step and the target reconstructs context.

**A target failure is a RUN FAILURE, never a low safety score (§6).** Every turn
outcome is classified ``{ok | target_error | refusal | empty | truncated}``; the
Solver records it and the Scorer refuses to judge a failed scenario — scoring infra
failure as unsafe is a false-unsafe that violates the quality bar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.model import Model, ModelOutput, get_model

from .mocks import mock_target_model

MOCK_PREFIX = "mock"
OK = "ok"
FAILURE_STATUSES = ("target_error", "refusal", "empty", "truncated")

# Model providers aipsy-bench recognizes for a friendly ``init``/``doctor`` pre-check.
# Deliberately NOT exhaustive — Inspect is the source of truth for the open-weights long
# tail, so an unrecognized provider is a soft WARN (still runs), never a hard block. A wrong
# *model name* isn't checked here at all (too many; HF open-weights) — it surfaces as a clean
# run-time resolution error / run failure. Update as Inspect adds providers.
KNOWN_MODEL_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "mistral",
        "grok",
        "xai",
        "groq",
        "together",
        "deepseek",
        "openrouter",
        "perplexity",
        "cohere",
        "fireworks",
        "openai-api",
        "azureai",
        "bedrock",
        "vertex",
        "cf",
        "cloudflare",
        "goodfire",
        "ollama",
        "hf",
        "vllm",
        "sglang",
        "transformers",
        "llama-cpp-python",
        "lmstudio",
        "mockllm",
    }
)
_PROVIDER_DISPLAY = (
    "openai, anthropic, google, ollama, hf, mistral, grok, together, groq, "
    "bedrock, vertex, azureai, vllm, openai-api"
)


class TargetResolutionError(ValueError):
    """A target model string could not be resolved — with a clean, actionable message
    (never a raw traceback). Subclasses ValueError so existing CLI handling catches it."""


def validate_model_ref(ref: str) -> dict:
    """Offline sanity-check a Tier-0 model string BEFORE it reaches Inspect (used by
    ``init``/``doctor``). No network, no key, no model load. Returns
    ``{ok, level, provider, message}`` with ``level ∈ {ok, warn, error}``:

    * ``error`` — structurally not a model string (no ``provider/``, whitespace, an empty
      side): gibberish or a forgotten prefix → reject / re-prompt.
    * ``warn``  — well-formed but the provider isn't one we recognize. Inspect MAY still
      support it (open-weights long tail), so we DON'T block — the run resolves it for real.
    * ``ok``    — ``provider/model`` with a recognized provider.

    The *model name* is intentionally not verified (too many; HF open-weights) — a wrong
    name surfaces as a clean run-time resolution error or a graceful run failure.
    """
    r = (ref or "").strip()
    if is_mock_ref(r):
        return {
            "ok": True,
            "level": "ok",
            "provider": "mock",
            "message": "mock target (offline)",
        }
    if not r:
        return {
            "ok": False,
            "level": "error",
            "provider": None,
            "message": "no model string given — expected 'provider/model', e.g. openai/gpt-5.4-mini",
        }
    if any(c.isspace() for c in r):
        return {
            "ok": False,
            "level": "error",
            "provider": None,
            "message": f"{ref!r} is not a model string (contains spaces) — expected a single "
            "'provider/model' token, e.g. openai/gpt-5.4-mini or ollama/llama3",
        }
    if "/" not in r:
        return {
            "ok": False,
            "level": "error",
            "provider": None,
            "message": f"{ref!r} is missing the provider prefix — a bare model name won't "
            "resolve. Use 'provider/model', e.g. openai/gpt-5.4-mini, ollama/llama3",
        }
    provider, _, model = r.partition("/")
    if not provider or not model:
        return {
            "ok": False,
            "level": "error",
            "provider": provider or None,
            "message": f"{ref!r} is malformed — 'provider/model' needs both parts, "
            "e.g. anthropic/claude-sonnet-4-6",
        }
    if provider.lower() not in KNOWN_MODEL_PROVIDERS:
        return {
            "ok": True,
            "level": "warn",
            "provider": provider,
            "message": f"provider '{provider}' isn't one aipsy-bench recognizes — if Inspect "
            "supports it the run works; otherwise it fails to resolve with a clear "
            f"error. Recognized include: {_PROVIDER_DISPLAY}.",
        }
    return {
        "ok": True,
        "level": "ok",
        "provider": provider,
        "message": f"provider '{provider}' recognized (the model name is checked at run time)",
    }


def _is_missing_key_error(e: Exception) -> bool:
    """A provider recognized-but-not-ready-here error (missing/unresolved API key) — as
    opposed to an unknown provider or a structural fault. Inspect raises PrerequisiteError."""
    return type(e).__name__ == "PrerequisiteError" or "API_KEY" in str(e).upper()


def key_looks_malformed(value: str) -> bool:
    """A present API-key VALUE that clearly isn't a key — a filesystem path (the classic
    ``OPENAI_API_KEY=/…/.env`` mistake), or one carrying whitespace. Used by ``doctor`` to
    catch a bad key at preflight instead of as an opaque per-turn auth failure."""
    v = value or ""
    return bool(v) and ("/" in v or v != v.strip() or v.endswith(".env") or " " in v)


def _explain_resolution_error(ref: str, e: Exception) -> str:
    """Turn an Inspect ``get_model`` exception into one clean, actionable line (Inspect's
    own messages are decent but leak rich-markup / internals)."""
    provider = ref.split("/", 1)[0] if "/" in ref else ref
    msg = str(e)
    etype = type(e).__name__
    if "not recognized" in msg:
        return (
            f"target model {ref!r}: provider {provider!r} is not a recognized model provider. "
            f"Use one of: {_PROVIDER_DISPLAY} (see docs/adapters/)."
        )
    if etype == "PrerequisiteError" or "API_KEY" in msg.upper():
        from . import spec

        env = spec.API_ENV_VARS.get(provider)
        hint = (
            f"set {env} — `aipsy-bench keys set --provider {provider}`"
            if env
            else "check the provider's API key + SDK (uv sync --all-extras)"
        )
        return (
            f"target model {ref!r}: provider {provider!r} isn't ready — {hint}. "
            "Run `aipsy-bench doctor` to preflight."
        )
    if "format of" in msg:
        return f"target model {ref!r}: expected 'provider/model', e.g. openai/gpt-5.4-mini."
    first = msg.splitlines()[0] if msg else etype
    return f"could not resolve target model {ref!r}: {first[:160]}"


# A user message history is a list of {"role": ..., "content": ...} dicts.
MessageHistory = list[dict[str, str]]
TargetFn = Callable[[MessageHistory], str]
# (url, headers, json_payload) -> parsed JSON response. Overridable for offline tests.
Transport = Callable[[str, dict[str, str], dict[str, Any]], Any]


@dataclass
class ResolvedTarget:
    # ``None`` only for a Tier-0 model target whose key isn't resolvable at CLI time — the run
    # then constructs it from ``ref`` inside the eval context (post-.env), like the judges.
    model: Model | None
    ref: str
    adapter: str  # "mock" | "model" | "http" | "callable"
    is_mock: bool
    conversation: str = "stateless"  # stateless | session
    mock_profile: str | None = None
    meta: dict = field(default_factory=dict)


def is_mock_ref(ref: str) -> bool:
    return ref == MOCK_PREFIX or ref.startswith(MOCK_PREFIX + ":")


def classify_outcome(output: ModelOutput) -> str:
    """Classify one turn's target outcome. Anything but ``ok`` is a run failure.

    ``refusal`` here means a provider **content-filter block** (the target never
    produced a reply) — NOT a substantive reply that declines a harmful request,
    which is a real answer the judge must score (else good safety reads as failure).
    """
    if getattr(output, "error", None):
        return "target_error"
    sr = getattr(output, "stop_reason", None)
    if sr in ("max_tokens", "model_length"):
        return "truncated"
    if sr == "content_filter":
        return "refusal"
    if not (output.completion or "").strip():
        return "empty"
    return OK


# --------------------------------------------------------------------------
# Tier 0 (bare model string) + mock
# --------------------------------------------------------------------------
def resolve_target(ref: str) -> ResolvedTarget:
    """Resolve a target ref string.

    - ``mock`` / ``mock:<profile>`` → the offline deterministic mock target.
    - anything else → a Tier-0 bare Inspect model string (``openai/gpt-…`` etc.).

    Tier-1/2 targets are constructed via ``http_target`` / ``callable_target``
    (a string ref can't carry a Python function or transport).
    """
    if is_mock_ref(ref):
        profile = ref.split(":", 1)[1] if ":" in ref else "safe"
        return ResolvedTarget(
            model=mock_target_model(profile),
            ref=ref,
            adapter="mock",
            is_mock=True,
            mock_profile=profile,
        )
    # Validate structure first (a clear message beats Inspect's internal error), then let
    # Inspect resolve it — wrapping ANY failure (unknown provider / missing key / SDK) as a
    # typed, actionable error so a bad string is a clean exit, NEVER a run-setup traceback.
    check = validate_model_ref(ref)
    if not check["ok"]:
        raise TargetResolutionError(check["message"])
    try:
        model = get_model(ref)
    except Exception as e:  # noqa: BLE001 — normalize every provider's failure to one clean error
        if _is_missing_key_error(e):
            # Provider recognized; the key just isn't resolvable HERE (CLI time). Do NOT
            # pre-capture or reject: the key may live in .env, which Inspect loads INSIDE the
            # eval context. Defer — the run builds the target from the string there, so it
            # resolves the key exactly where/when the judges do (§6 key-parity). model=None
            # is a marker: the run must pass the ref string to inspect_eval, not this object.
            return ResolvedTarget(model=None, ref=ref, adapter="model", is_mock=False)
        raise TargetResolutionError(_explain_resolution_error(ref, e)) from e
    return ResolvedTarget(model=model, ref=ref, adapter="model", is_mock=False)


# --------------------------------------------------------------------------
# Tier 2 — Python callable (recommended for real apps)
# --------------------------------------------------------------------------
def _display_model_name(ref: str) -> str:
    """A readable ``mockllm`` model-name segment derived from the target ref, so the live UI
    shows the endpoint/label (e.g. ``mockllm/coachella-coach``) instead of a bare ``mockllm/model``
    that reads like a mock. Sanitized to a valid, bounded name."""
    s = re.sub(r"^https?://", "", ref)  # drop the scheme
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:48] or "target"


def _model_from_callable(fn: TargetFn, conversation: str, ref: str) -> Model:
    """Wrap a ``message_history -> reply`` callable as an Inspect model.

    On any exception the wrapper returns an *errored* ModelOutput (not raising) so
    the Solver classifies it as ``target_error`` rather than crashing the run.
    """
    label = _display_model_name(ref)

    def _outputs(messages, tools, tool_choice, config):
        history = [{"role": m.role, "content": m.text} for m in messages]
        # session: send only the new (last) user turn; the target owns its history.
        payload = history[-1:] if conversation == "session" else history
        try:
            reply = fn(payload)
        except Exception as e:  # noqa: BLE001 — any target error is a run failure (§6)
            return ModelOutput.from_content(
                model=label,
                content="",
                stop_reason="unknown",
                error=f"{type(e).__name__}: {e}",
            )
        if reply is None:
            return ModelOutput.from_content(
                model=label, content="", error="target returned None"
            )
        return ModelOutput.from_content(model=label, content=str(reply))

    # mockllm carries the custom_outputs hook; the name segment is the readable target label.
    return get_model(f"mockllm/{label}", custom_outputs=_outputs)


def callable_target(
    fn: TargetFn, *, conversation: str = "stateless", ref: str = "callable"
) -> ResolvedTarget:
    """Build a Tier-2 callable target from ``fn(message_history) -> reply``."""
    return ResolvedTarget(
        model=_model_from_callable(fn, conversation, ref),
        ref=ref,
        adapter="callable",
        is_mock=False,
        conversation=conversation,
    )


# --------------------------------------------------------------------------
# Tier 1 — HTTP, OpenAI-chat-compatible (easy case only; no SSE/DSL)
# --------------------------------------------------------------------------
def _default_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> Any:
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — user-supplied endpoint
        return json.loads(resp.read().decode())


def _parse_reply(body: Any) -> str:
    """Accept ``{"reply": "..."}`` or the OpenAI ``choices[0].message.content`` shape."""
    if isinstance(body, dict):
        if "reply" in body:
            return str(body["reply"])
        choices = body.get("choices")
        if choices:
            return str(choices[0]["message"]["content"])
    raise ValueError(f"unrecognized target response shape: {body!r}")


def http_target(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    conversation: str = "stateless",
    transport: Transport | None = None,
    ref: str | None = None,
) -> ResolvedTarget:
    """Build a Tier-1 HTTP target. Request: ``{"messages": [...]}`` → reply.

    Static auth header only; pass ``transport`` to inject a fake transport offline.
    Do NOT extend this into a streaming / multi-step protocol — use a Tier-2
    callable for anything beyond a single stateless request (§6).
    """
    headers = headers or {}
    send: Transport = transport or _default_transport

    def fn(messages: MessageHistory) -> str:
        return _parse_reply(send(url, headers, {"messages": messages}))

    rt = callable_target(fn, conversation=conversation, ref=ref or url)
    rt.adapter = "http"
    rt.meta = {"url": url}
    return rt


# --------------------------------------------------------------------------
# Connectivity probe (for `doctor` — one request, classified; not a scored call)
# --------------------------------------------------------------------------
PROBE_TIMEOUT = 15
# (url, headers, payload, timeout) -> (status_code, body_text). Non-2xx returns its code
# (does NOT raise) so the probe can classify 401/404/etc. Overridable for offline tests.
Poster = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, str]]


def _raw_post(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — user-supplied endpoint
            return resp.status, resp.read().decode(errors="replace")
    except (
        urllib.error.HTTPError
    ) as e:  # 4xx/5xx: capture the status instead of raising
        try:
            return e.code, (e.read().decode(errors="replace") if e.fp else "")
        except Exception:  # noqa: BLE001
            return e.code, ""


def probe_endpoint(
    url: str,
    headers: dict[str, str] | None = None,
    *,
    timeout: float = PROBE_TIMEOUT,
    poster: Poster | None = None,
) -> dict[str, Any]:
    """Send one probe request to a Tier-1 HTTP target and classify the outcome.

    Catches the common misconfigs *before* a battery — wrong host/port (unreachable),
    wrong path (404), bad/missing secret (401/403), wrong response shape. It hits only the
    user's own endpoint (never a judge/provider), so it is not a scored call. Returns
    ``{ok, kind, message}``.
    """
    headers = headers or {}
    send = poster or _raw_post
    body = {"messages": [{"role": "user", "content": "aipsy-bench connectivity probe"}]}
    try:
        status, text = send(url, headers, body, timeout)
    except TimeoutError:
        return {
            "ok": False,
            "kind": "timeout",
            "message": f"connected but no response within {timeout:g}s — the server is up but slow "
            "(or the handler hung); a real run allows a longer --timeout",
        }
    except (
        OSError
    ) as e:  # URLError subclasses OSError: connection refused / DNS / TLS / …
        return {
            "ok": False,
            "kind": "unreachable",
            "message": f"not reachable ({type(e).__name__}) — is the server running at {url}?",
        }

    if status == 404:
        return {
            "ok": False,
            "kind": "not_found",
            "message": "reached the server but there is no handler at this path (404) — check the route",
        }
    if status in (401, 403):
        return {
            "ok": False,
            "kind": "unauthorized",
            "message": f"reached the endpoint but auth was rejected ({status}) — check the header / secret",
        }
    if 200 <= status < 300:
        try:
            parsed = json.loads(text) if isinstance(text, str) else text
            reply = _parse_reply(parsed)
        except Exception:  # noqa: BLE001 — any parse failure is a contract mismatch
            return {
                "ok": False,
                "kind": "bad_shape",
                "message": f"reachable ({status}) but the response is not a {{reply}} / OpenAI shape — "
                "check the {messages} -> {reply} contract",
            }
        preview = " ".join(reply.split())
        preview = (preview[:57] + "…") if len(preview) > 58 else preview
        return {
            "ok": True,
            "kind": "ok",
            "message": f'reachable — endpoint replied ({status}): "{preview}"',
        }
    return {
        "ok": False,
        "kind": "http_error",
        "message": f"reached the endpoint (HTTP {status}) — it did not accept the probe payload; "
        "verify it takes {messages:[{role,content}]} -> {reply}",
    }
