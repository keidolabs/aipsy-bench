"""``board.json`` row schema + writer (the data feed, §13.1).

Step 11 ships the schema + a builder only. ``submit``/PR flow, tier auto-assignment,
``attest``/run_hash signing, and ``paraphrase`` are iteration 3.

Eligibility guard: a row is comparable ONLY for a full-battery ``gold`` benchmark
run. ``single`` / ``--quick`` / ``custom`` runs are excluded — they are never
"a score" in the leaderboard sense (§7.2 / §3.5).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import spec


class BoardRow(BaseModel):
    name: str
    kind: str = "vanilla_model"  # vanilla_model | open_weights | commercial_app
    tier: int | None = None      # iteration 3 auto-assigns; None for now
    data_version: str = "v1"
    judge_panel: str = "gold"
    scores: dict[str, float | None] = Field(default_factory=dict)
    paraphrase_delta: float | None = None  # iteration 3
    run_hash: str = ""           # iteration 3 (attest); run_id placeholder for now
    date: str = ""
    judge_validation_status: str = "PENDING_VALIDATION"
    card: str = ""               # path/URL to the card SVG


def is_board_eligible(result_json: dict) -> bool:
    """Only a full-battery gold benchmark run is leaderboard-comparable."""
    if result_json.get("mode") != "benchmark":
        return False
    if result_json.get("judge_panel") != "gold":
        return False
    if any("partial battery" in w for w in result_json.get("warnings", [])):
        return False
    return len(result_json["scores"]["by_scenario"]) == spec.N_SCENARIOS


def row_from_result(result_json: dict, *, name: str | None = None, kind: str = "vanilla_model",
                    card: str = "") -> BoardRow:
    """Build a BoardRow from a result.json (caller ensures eligibility)."""
    overall = result_json["scores"]["overall"]
    return BoardRow(
        name=name or result_json["target"]["ref"],
        kind=kind,
        data_version=result_json["data_version"],
        judge_panel=result_json["judge_panel"],
        scores={k: overall.get(k) for k in (*spec.METRICS, "AI_Trust")},
        run_hash=result_json.get("run_id", ""),  # binds to run_hash once attest lands
        date=str(result_json.get("timestamp", "")),
        judge_validation_status=result_json["judge_validation"]["status"],
        card=card,
    )


def board_from_results(results: list[dict], **kwargs) -> list[BoardRow]:
    """Build board rows from result.jsons, EXCLUDING non-comparable runs."""
    return [row_from_result(r, **kwargs) for r in results if is_board_eligible(r)]
