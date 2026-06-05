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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.model import Model, ModelOutput, get_model

from .mocks import mock_target_model

MOCK_PREFIX = "mock"
OK = "ok"
FAILURE_STATUSES = ("target_error", "refusal", "empty", "truncated")

# A user message history is a list of {"role": ..., "content": ...} dicts.
MessageHistory = list[dict[str, str]]
TargetFn = Callable[[MessageHistory], str]
# (url, headers, json_payload) -> parsed JSON response. Overridable for offline tests.
Transport = Callable[[str, dict[str, str], dict[str, Any]], Any]


@dataclass
class ResolvedTarget:
    model: Model
    ref: str
    adapter: str                       # "mock" | "model" | "http" | "callable"
    is_mock: bool
    conversation: str = "stateless"    # stateless | session
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
            model=mock_target_model(profile), ref=ref, adapter="mock",
            is_mock=True, mock_profile=profile,
        )
    return ResolvedTarget(model=get_model(ref), ref=ref, adapter="model", is_mock=False)


# --------------------------------------------------------------------------
# Tier 2 — Python callable (recommended for real apps)
# --------------------------------------------------------------------------
def _model_from_callable(fn: TargetFn, conversation: str, label: str) -> Model:
    """Wrap a ``message_history -> reply`` callable as an Inspect model.

    On any exception the wrapper returns an *errored* ModelOutput (not raising) so
    the Solver classifies it as ``target_error`` rather than crashing the run.
    """
    def _outputs(messages, tools, tool_choice, config):
        history = [{"role": m.role, "content": m.text} for m in messages]
        # session: send only the new (last) user turn; the target owns its history.
        payload = history[-1:] if conversation == "session" else history
        try:
            reply = fn(payload)
        except Exception as e:  # noqa: BLE001 — any target error is a run failure (§6)
            return ModelOutput.from_content(
                model=label, content="", stop_reason="unknown",
                error=f"{type(e).__name__}: {e}",
            )
        if reply is None:
            return ModelOutput.from_content(model=label, content="", error="target returned None")
        return ModelOutput.from_content(model=label, content=str(reply))

    return get_model("mockllm/model", custom_outputs=_outputs)


def callable_target(fn: TargetFn, *, conversation: str = "stateless", ref: str = "callable") -> ResolvedTarget:
    """Build a Tier-2 callable target from ``fn(message_history) -> reply``."""
    return ResolvedTarget(
        model=_model_from_callable(fn, conversation, "callable/target"),
        ref=ref, adapter="callable", is_mock=False, conversation=conversation,
    )


# --------------------------------------------------------------------------
# Tier 1 — HTTP, OpenAI-chat-compatible (easy case only; no SSE/DSL)
# --------------------------------------------------------------------------
def _default_transport(url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
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
