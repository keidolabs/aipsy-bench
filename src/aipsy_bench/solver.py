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


@solver
def scripted_dialogue(*, baseline_prompt: bool = False) -> Solver:
    """Drive one scenario: inject each scripted user turn, generate, accumulate.

    The transcript is written to ``state.metadata["transcript"]`` as a list of
    ``{turn, phase, user_message, ai_text}`` for the Scorer (Step 07). The Inspect
    ``messages`` list is kept intact as the running conversation history.
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
        for t in turns:
            state.messages.append(ChatMessageUser(content=t["message"]))
            state = await generate(state)
            ai_text = state.output.completion
            transcript.append(
                {
                    "turn": t["turn"],
                    "phase": t["phase"],
                    "user_message": t["message"],
                    "ai_text": ai_text,
                }
            )

        state.metadata["transcript"] = transcript
        return state

    return solve
