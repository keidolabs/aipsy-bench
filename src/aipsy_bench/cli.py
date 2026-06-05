"""``aipsy-bench`` CLI (Step 10 surface: ``run`` + ``scenarios list``).

Operability verbs (``doctor``, ``--dry-run``, ``provenance``) are Step 17;
``compare``/``history`` are Step 14. This wires Dataset+Solver+Scorer into one
end-to-end run that writes artifacts and drives a gate exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from . import bundle, report
from .targets import resolve_target
from .task import aipsy_bench
from .validation import load_validation


def _run(args: argparse.Namespace) -> int:
    ref = args.model or args.target
    if not ref:
        print("error: specify --target <ref> or --model <ref>", file=sys.stderr)
        return 2

    scenario_ids = [s.strip() for s in args.scenario.split(",")] if args.scenario else None
    resolved = resolve_target(ref)
    task = aipsy_bench(
        target=ref,
        judges=args.judges,
        scenario_ids=scenario_ids,
        quick=args.quick,
        baseline_prompt=args.baseline_prompt,
    )
    out = Path(args.out)
    logs = inspect_eval(task, model=resolved.model, display=args.display, log_dir=str(out / "logs"))
    log = logs[0]
    if log.status != "success":
        print(f"run did not complete (status={log.status})", file=sys.stderr)
        return 2

    extra_warnings = []
    if args.quick or scenario_ids:
        extra_warnings.append(
            "partial battery (quick/subset) — directional only, not comparable; not card/board eligible"
        )
    target_block = {"adapter": resolved.adapter, "ref": resolved.ref, "model_snapshot": log.eval.model}
    result = report.write_artifacts(
        log,
        out,
        validation=load_validation(args.validation_artifact),
        target=target_block,
        extra_warnings=extra_warnings,
    )
    if not args.no_card:
        _emit_cards(result, out)

    print(report.render_report(result))
    print(f"\nartifacts written to: {out}/result.json · {out}/report.txt")
    return 0 if result["gate"]["passed"] else 1


def _emit_cards(result: dict, out: Path) -> None:
    """Render share card + badge. Cards NEVER block the run (§4.5)."""
    try:
        from . import card

        svg, png = card.render_card(result)
        (out / "card.svg").write_text(svg)
        (out / "card.png").write_bytes(png)
        (out / "badge.svg").write_text(card.render_badge(result))
    except Exception as e:  # noqa: BLE001 — a card failure must not fail the run
        print(f"warning: card rendering skipped ({e})", file=sys.stderr)


def _scenarios_list(args: argparse.Namespace) -> int:
    for s in bundle.load_scenarios(include_reserved=args.include_reserved):
        tag = "crisis" if s.crisis else "      "
        print(f"{s.id}  {s.domain:<14} {tag}  {s.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aipsy-bench", description="psychological-safety benchmark for conversational AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the benchmark against a target")
    r.add_argument("--model", help="Tier-0 Inspect model string, e.g. openai/gpt-5.4-mini")
    r.add_argument("--target", help="target ref: 'mock', 'mock:failing', or an Inspect model string")
    r.add_argument("--judges", choices=["single", "gold"], default="single",
                   help="single = primary judge (fast inner loop); gold = 3-judge ensemble")
    r.add_argument("--quick", action="store_true", help="smoke subset: one scenario/domain + s06,s07")
    r.add_argument("--scenario", help="comma-separated scenario ids, e.g. s06,s07")
    r.add_argument("--baseline-prompt", action="store_true", help="inject the 014 baseline system prompt (§6 opt-in)")
    r.add_argument("--out", default="aipsy-run", help="output directory for result.json + report.txt")
    r.add_argument("--no-card", action="store_true", help="skip share-card rendering (cards arrive in Step 11)")
    r.add_argument("--validation-artifact", default=None, help="path to a 014 gate artifact (else PENDING)")
    r.add_argument("--display", default="plain", help="Inspect display mode (plain|rich|none)")
    r.set_defaults(func=_run)

    sc = sub.add_parser("scenarios", help="browse the public benchmark content")
    scsub = sc.add_subparsers(dest="scmd", required=True)
    lst = scsub.add_parser("list", help="list scenarios")
    lst.add_argument("--include-reserved", action="store_true")
    lst.set_defaults(func=_scenarios_list)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
