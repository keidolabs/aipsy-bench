"""Scripted-dialogue Solver — replays each scenario's frozen user turns.

Ported from 014 ``generate.py`` (the flat user/assistant history accumulation),
with the critical §6 difference: the target keeps its OWN system prompt. aipsy-
bench injects nothing by default — the Solver sends only the scripted user turns.
``--baseline-prompt`` is the single opt-in exception (§6), reproducing the 014
frontier baseline by injecting ``spec.BASELINE_SYSTEM_PROMPT``.

This step handles only the *stateless* conversation model (harness owns history,
replays the full transcript). ``conversation: session`` and target-failure
classification land in Step 13.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from . import spec
from .targets import OK, classify_outcome


@solver
def scripted_dialogue(*, baseline_prompt: bool = False, conversation: str = "stateless") -> Solver:
    """Drive one scenario: inject each scripted user turn, generate, accumulate.

    The transcript is written to ``state.metadata["transcript"]`` as a list of
    ``{turn, phase, user_message, ai_text, status}`` for the Scorer. Each turn's
    outcome is classified (§6); a non-``ok`` turn is a RUN FAILURE — the loop stops
    and ``state.metadata["target_failure"]`` is set, so the Scorer never judges it as
    if it were an unsafe answer. ``conversation`` (stateless|session) is recorded for
    the report (it can change Deepening/Resolution results); the actual session
    forwarding lives in the target adapter.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        turns = sorted(state.metadata["turns"], key=lambda t: t["turn"])

        # The Solver is the source of truth for the conversation. Reset any
        # pre-populated input messages and rebuild from the scripted turns so
        # nothing but scripted user turns (and the opt-in baseline system) is sent.
        state.messages = []
        if baseline_prompt:
            state.messages.append(ChatMessageSystem(content=spec.BASELINE_SYSTEM_PROMPT))

        transcript: list[dict] = []
        target_failure: dict | None = None
        for t in turns:
            state.messages.append(ChatMessageUser(content=t["message"]))
            state = await generate(state)
            status = classify_outcome(state.output)
            transcript.append(
                {
                    "turn": t["turn"],
                    "phase": t["phase"],
                    "user_message": t["message"],
                    "ai_text": state.output.completion,
                    "status": status,
                }
            )
            if status != OK:
                target_failure = {
                    "turn": t["turn"],
                    "status": status,
                    "detail": state.output.error or "",
                }
                break  # a run failure aborts the scenario; do not score it

        state.metadata["transcript"] = transcript
        state.metadata["conversation"] = conversation
        if target_failure is not None:
            state.metadata["target_failure"] = target_failure
        return state

    return solve
