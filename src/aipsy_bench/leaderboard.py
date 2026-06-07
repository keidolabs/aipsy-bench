"""``board.json`` row schema + writer (the data feed, §13.1).

Step 11 ships the schema + a builder only. ``submit``/PR flow, tier auto-assignment,
``attest``/run_hash signing, and ``paraphrase`` are iteration 3.

Eligibility guard: a row is comparable ONLY for a full-battery ``gold`` benchmark
run. ``single`` / ``--quick`` / ``custom`` runs are excluded — they are never
"a score" in the leaderboard sense (§7.2 / §3.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import spec


class PublishRefused(RuntimeError):
    """Raised when a run is not eligible to be published as a comparable card."""

_INSTRUMENT_CAVEAT = (
    "Comparable on the INSTRUMENT, not on generation conditions: the baselines are "
    "minimal-prompt frontier models; your bot is your configured bot (§6). And the "
    "snapshot is a SAMPLE — not a validated launch board (§0.3)."
)
_AGAINST_KEYS = ("AI_Trust", *spec.METRICS)


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
    """Only a full-battery gold benchmark run with no run failures is comparable."""
    if result_json.get("mode") != "benchmark":
        return False
    if result_json.get("judge_panel") != "gold":
        return False
    if result_json.get("run_failures"):  # a run failure means the run is incomplete (§6)
        return False
    if result_json.get("judge_failures"):  # degraded instrument — not a clean comparable run
        return False
    if result_json.get("judge_overrides"):  # a swapped judge is not the frozen instrument (§8)
        return False
    if result_json.get("incomplete"):
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


# --------------------------------------------------------------------------
# run --against-board (§13.6) — local overlay on the packaged vanilla snapshot.
# No submission, no network, no telemetry.
# --------------------------------------------------------------------------
def load_board_snapshot() -> dict:
    """Load the read-only packaged vanilla board snapshot (a SAMPLE, not validated)."""
    return json.loads((spec.DATA_V1 / "board_snapshot.json").read_text())


def _local_scores(result_json: dict, domain: str | None) -> dict:
    scores = result_json["scores"]
    if domain:
        return scores["by_domain"].get(domain, {})
    return scores["overall"]


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else " N/A"


def render_against_board(result_json: dict, *, domain: str | None = None, snapshot: dict | None = None) -> str:
    """Overlay the local run's scores on the published vanilla rows (§13.6)."""
    snapshot = snapshot if snapshot is not None else load_board_snapshot()
    rows = snapshot.get("rows", [])
    if domain:
        rows = [r for r in rows if domain in r.get("by_domain", {})]

    local = _local_scores(result_json, domain)
    me = result_json["target"]["ref"]
    scope = f" · domain={domain}" if domain else ""

    header = ["against-board" + scope + "  (SAMPLE snapshot — not validated)", ""]
    col = "  {:<22}" + "".join(f" {k[:8]:>8}" for k in _AGAINST_KEYS)
    lines = header + [col.format("model/bot", *_AGAINST_KEYS)]

    def row_line(label, scoremap):
        cells = [_fmt(scoremap.get(k)) for k in _AGAINST_KEYS]
        return "  {:<22}".format(label[:22]) + "".join(f" {c:>8}" for c in cells)

    lines.append(row_line(f"YOU: {me}", local))
    lines.append("  " + "-" * (22 + 9 * len(_AGAINST_KEYS)))
    for r in rows:
        sm = r.get("by_domain", {}).get(domain, {}) if domain else r.get("scores", {})
        lines.append(row_line(r["name"], sm))

    lines += ["", _INSTRUMENT_CAVEAT]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# publish-card (§13.6 lead capture) — the one opt-in, user-initiated mechanic.
# It writes a ready-to-post bundle LOCALLY. There is NO automatic network call,
# no telemetry (§12); the hand-raise is the user's deliberate action of posting it.
# --------------------------------------------------------------------------
def _ineligible_reason(result_json: dict) -> str | None:
    if result_json.get("mode") != "benchmark":
        return "custom/lab runs are not comparable"
    if result_json.get("judge_panel") != "gold":
        return "only a gold-panel run is comparable (single is directional)"
    if result_json.get("incomplete"):
        return "the run is incomplete"
    if result_json.get("run_failures"):
        return "the run had target failures"
    if any("partial battery" in w for w in result_json.get("warnings", [])):
        return "a quick/subset run is not the full battery"
    if len(result_json["scores"]["by_scenario"]) != spec.N_SCENARIOS:
        return "not a full s01–s20 battery"
    return None


def publish_card_bundle(result_json: dict, out_dir: str | Path, *, render=True) -> dict:
    """Prepare a ready-to-post bundle (card + board-row stub + post text) LOCALLY.

    Refuses a single / quick / custom / incomplete / failed run (§13.6/§0.3). NEVER
    makes a network call — the user posts the bundle themselves (the opt-in hand-raise).
    """
    reason = _ineligible_reason(result_json)
    if reason is not None:
        raise PublishRefused(f"cannot publish a comparable card — {reason}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = row_from_result(result_json, card="card.svg")

    if render:
        from . import card  # local import — keeps card deps off the import path otherwise

        svg, png = card.render_card(result_json)
        (out / "card.svg").write_text(svg)
        (out / "card.png").write_bytes(png)
        (out / "badge.svg").write_text(card.render_badge(result_json))

    (out / "board_row.json").write_text(json.dumps(row.model_dump(), indent=2, default=str))
    (out / "POST.md").write_text(_post_text(result_json, row))
    return {"out": str(out), "row": row}


def _post_text(result_json: dict, row: BoardRow) -> str:
    at = row.scores.get("AI_Trust")
    at_str = f"{at:.2f}" if isinstance(at, (int, float)) else "N/A"
    return (
        f"# aipsy-bench result — {row.name}\n\n"
        f"AI-Trust **{at_str}** (gold panel, data {row.data_version}).\n\n"
        "> PROVISIONAL — instrument not yet human-validated (§0.3). "
        "Scores are descriptive only.\n\n"
        "To publish on the public gallery, open a PR adding `board_row.json` + `card.svg` "
        "to the gallery repo. This bundle is local — nothing was sent anywhere.\n\n"
        f"Reproduce: `aipsy-bench run --model {result_json['target']['ref']} --judges gold`\n"
    )
