"""Step 19 — coachella acceptance, OFFLINE portion.

Walks the real-app path against a SIMULATED coachella ``/eval`` endpoint (a fake HTTP
transport): full-battery scoring, phase-localized crisis cards, the rate-limit →
run-failure guarantee (§6), compare delta, and --against-board. The LIVE run against
the deployed coachella coach + the coachella-repo `/eval` edit need user sign-off + keys
(see docs/adapters/coachella-example.md) and are intentionally not done here.
"""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai import eval as inspect_eval

from aipsy_bench import leaderboard, spec
from aipsy_bench.compare import compare
from aipsy_bench.dataset import build_dataset
from aipsy_bench.mocks import build_marker
from aipsy_bench.report import to_result_json
from aipsy_bench.scorer import clinical_judge_panel
from aipsy_bench.solver import scripted_dialogue
from aipsy_bench.targets import http_target
from aipsy_bench.validation import load_validation
from fixtures import mock_judge_model

_CRISIS = (
    "hurt myself",
    "cutting",
    "want to die",
    "not turning the wheel",
    "wasn't here",
    "not existing",
    "can feel it coming back",
)


def _coachella_transport(profile="failing_on_crisis", cap=None):
    """A fake coachella /eval endpoint: an 'Alex from Manchester' coach that mishandles
    crisis (failing_on_crisis) and rate-limits after `cap` calls (free-tier sim)."""
    state = {"n": 0}

    def transport(url, headers, payload):
        state["n"] += 1
        if cap is not None and state["n"] > cap:
            raise TimeoutError("429 rate limited (coachella free tier)")
        users = " ".join(
            m["content"].lower() for m in payload["messages"] if m["role"] == "user"
        )
        crisis = any(c in users for c in _CRISIS)
        if crisis and profile == "failing_on_crisis":
            body, marker = (
                "i'm sure it's not that bad, just sleep it off, what else is new?",
                build_marker("failing", True),
            )
        else:
            body, marker = (
                "right, small steps — one conversation at a time, yeah?",
                build_marker("safe", crisis),
            )
        return {"reply": f"{body} {marker}"}

    return transport, state


def _run(tmp_path, transport, *, scenario_ids):
    target = http_target(
        "https://coachella.app/api/ai-coach/eval",
        headers={"x-eval-secret": "s"},
        transport=transport,
        ref="coachella-coach",
    )
    judges = {p: mock_judge_model() for p in spec.PROVIDERS}
    task = Task(
        dataset=build_dataset(scenario_ids=scenario_ids),
        solver=scripted_dialogue(conversation="stateless"),
        scorer=clinical_judge_panel(panel="gold", judges=judges, cache=False),
    )
    return inspect_eval(
        task, model=target.model, display="none", log_dir=str(tmp_path / "logs")
    )[0]


def test_coachella_full_path_scores_and_localizes(tmp_path):
    transport, _ = _coachella_transport()
    log = _run(tmp_path, transport, scenario_ids=["s01", "s06", "s07"])
    result = to_result_json(log, validation=load_validation())
    assert not result["run_failures"]
    # the failing-on-crisis coach fails crisis_handling on the crisis scenarios, carded
    cards = " ".join(d["card"] for d in result["diagnostics"])
    assert "crisis_handling" in cards
    assert "phase" in cards  # phase-localized
    assert result["conversation"] == "stateless"


def test_rate_limit_is_run_failure_not_safety_failure(tmp_path):
    # cap at 15 calls → the run dies mid-battery (coachella free-tier reality)
    transport, _ = _coachella_transport(cap=15)
    log = _run(tmp_path, transport, scenario_ids=["s01", "s06"])
    result = to_result_json(log, validation=load_validation())

    assert result["run_failures"], "a rate-limit must surface as a run failure"
    failed = {rf["scenario_id"] for rf in result["run_failures"]}
    # the failed scenario is NOT scored as unsafe — its metrics are None, not ~1.0
    for sid in failed:
        vals = result["scores"]["by_scenario"][sid]
        assert vals.get("run_failure") is True
        assert all(vals[m] is None for m in spec.METRICS)
    # and the run cannot pass the gate or be carded
    assert result["gate"]["passed"] is False
    assert leaderboard.is_board_eligible(result) is False


def test_coachella_compare_shows_safety_delta(tmp_path):
    fixed, _ = _coachella_transport(profile="safe")  # a coach that handles crisis
    broken, _ = _coachella_transport(profile="failing_on_crisis")
    base = _run(tmp_path / "fixed", fixed, scenario_ids=["s06"])
    cand = _run(tmp_path / "broken", broken, scenario_ids=["s06"])
    diff = compare(base, cand)
    assert diff["overall"]["crisis_handling"]["delta"] < 0  # broke crisis handling


def test_coachella_against_board(tmp_path):
    transport, _ = _coachella_transport()
    result = to_result_json(
        _run(tmp_path, transport, scenario_ids=["s01", "s09", "s15"]),
        validation=load_validation(),
        target={
            "adapter": "http",
            "ref": "coachella-coach",
            "model_snapshot": "gpt-4o",
        },
    )
    text = leaderboard.render_against_board(result)
    assert "YOU: coachella-coach" in text
    assert "claude-sonnet-4-6" in text
