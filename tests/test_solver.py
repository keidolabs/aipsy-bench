"""Step 06 — scripted_dialogue Solver, run through real Inspect eval. Offline."""

from __future__ import annotations

from fixtures import mock_target_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import spec
from aipsy_bench.dataset import build_dataset
from aipsy_bench.solver import scripted_dialogue


def _run(tmp_path, *, scenario_ids, baseline_prompt=False, profile="safe"):
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(baseline_prompt=baseline_prompt),
    )
    logs = inspect_eval(
        task,
        model=mock_target_model(profile),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success"
    return logs[0]


def test_transcript_has_ten_turns_with_phases(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"])
    transcript = log.samples[0].metadata["transcript"]
    assert len(transcript) == 10
    assert [t["turn"] for t in transcript] == list(range(1, 11))
    for t in transcript:
        assert t["phase"] == spec.PHASE_BY_TURN[t["turn"]]
        assert "glad you told me" in t["ai_text"]  # mock canned safe reply


def test_scripted_user_messages_are_verbatim(tmp_path):
    from aipsy_bench import bundle

    s01 = next(s for s in bundle.load_scenarios() if s.id == "s01")
    log = _run(tmp_path, scenario_ids=["s01"])
    transcript = log.samples[0].metadata["transcript"]
    for t, turn in zip(transcript, s01.turns):
        assert t["user_message"] == turn.message


def test_no_system_message_by_default(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], baseline_prompt=False)
    roles = [m.role for m in log.samples[0].messages]
    assert "system" not in roles


def test_baseline_prompt_injects_exact_system(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01"], baseline_prompt=True)
    systems = [m for m in log.samples[0].messages if m.role == "system"]
    assert len(systems) == 1
    assert systems[0].text == spec.BASELINE_SYSTEM_PROMPT


def test_persona_never_sent_to_target(tmp_path):
    log = _run(tmp_path, scenario_ids=["s01", "s06"])
    for samp in log.samples:
        persona = samp.metadata["persona"]
        sent = " ".join(m.text for m in samp.messages if m.role in ("user", "system"))
        assert persona not in sent
