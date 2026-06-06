"""``aipsy-bench`` CLI: ``run``, ``compare``, ``explain``, ``doctor``,
``provenance``, ``scenarios``. Wires Dataset+Solver+Scorer into one end-to-end run
that writes artifacts and drives a gate exit code, with the §16 operability verbs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from . import __version__, bundle, report, spec
from .config import load_config
from .gate import gate_result
from .targets import is_mock_ref, resolve_target
from .task import aipsy_bench, quick_scenario_ids
from .validation import load_validation

# Rough per-call cost / throughput for --dry-run (NOT billing-accurate; a planning aid).
_EST_COST_PER_CALL = 0.0015
_EST_CALLS_PER_MIN = 60.0


def estimate(n_scenarios: int, judges: str) -> dict:
    target_calls = n_scenarios * spec.N_TURNS
    judge_calls = target_calls * (3 if judges == "gold" else 1)
    total = target_calls + judge_calls
    return {
        "scenarios": n_scenarios,
        "target_calls": target_calls,
        "judge_calls": judge_calls,
        "total_calls": total,
        "est_cost_usd": total * _EST_COST_PER_CALL,
        "est_minutes": total / _EST_CALLS_PER_MIN,
    }


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ref = args.model or args.target or cfg.target
    if not ref:
        print("error: specify --target <ref> or --model <ref> (or set it in aipsy-bench.yaml)", file=sys.stderr)
        return 2

    judges = args.judges or cfg.judges or "single"
    quick = args.quick or cfg.quick
    scenario = args.scenario or cfg.scenario
    scenario_ids = [s.strip() for s in scenario.split(",")] if scenario else None
    out = Path(args.out or cfg.out or "aipsy-run")
    max_cost = args.max_cost if args.max_cost is not None else cfg.max_cost

    n_scen = len(scenario_ids) if scenario_ids else (len(quick_scenario_ids()) if quick else spec.N_SCENARIOS)
    est = estimate(n_scen, judges)

    if args.dry_run:
        print(f"dry-run estimate ({judges} panel, {n_scen} scenarios × {spec.N_TURNS} turns):")
        print(f"  target calls: {est['target_calls']}   judge calls: {est['judge_calls']}   total: {est['total_calls']}")
        print(f"  est. cost: ~${est['est_cost_usd']:.2f}   est. wall-time: ~{est['est_minutes']:.1f} min")
        print("  (rough planning estimate — not billing-accurate; no scored calls made)")
        return 0

    if max_cost is not None and est["est_cost_usd"] > max_cost:
        print(f"aborting: estimated ~${est['est_cost_usd']:.2f} exceeds --max-cost ${max_cost:.2f}", file=sys.stderr)
        return 2

    resolved = resolve_target(ref)
    task = aipsy_bench(
        target=ref, judges=judges, scenario_ids=scenario_ids,
        quick=quick, baseline_prompt=args.baseline_prompt,
    )

    if args.resume:
        log = _resume(out, args.resume, args.display)
        if log is None:
            print(f"error: no run with id {args.resume!r} under {out}/logs", file=sys.stderr)
            return 2
    else:
        log = inspect_eval(task, model=resolved.model, display=args.display, log_dir=str(out / "logs"))[0]

    incomplete = log.status != "success"
    extra_warnings = []
    if quick or scenario_ids:
        extra_warnings.append(
            "partial battery (quick/subset) — directional only, not comparable; not card/board eligible"
        )

    validation = load_validation(args.validation_artifact)
    gate = gate_result(log, validation, **cfg.gate) if cfg.gate else None
    target_block = {"adapter": resolved.adapter, "ref": resolved.ref, "model_snapshot": log.eval.model}
    result = report.write_artifacts(
        log, out, validation=validation, target=target_block,
        extra_warnings=extra_warnings, gate=gate, incomplete=incomplete,
    )

    if not args.no_card:
        if incomplete or result["run_failures"]:
            print("note: cards skipped — run incomplete or had target failures (not card-eligible, §6/§16)", file=sys.stderr)
        else:
            _emit_cards(result, out)

    print(report.render_report(result))
    if args.against_board:
        from . import leaderboard
        print()
        print(leaderboard.render_against_board(result, domain=args.domain))
    print(f"\nartifacts written to: {out}/result.json · {out}/report.txt")

    if incomplete:
        print("\nrun INCOMPLETE — not gated, not carded.", file=sys.stderr)
        return 1

    gate_ok = result["gate"]["passed"]
    if args.gate_baseline:
        gate_ok = _apply_regression_gate(args, log, gate_ok)
    return 0 if gate_ok else 1


def _resume(out: Path, run_id: str, display: str):
    from inspect_ai import eval_retry
    from inspect_ai.log import read_eval_log

    prior_path = _find_log_by_run_id(out, run_id)
    if not prior_path:
        return None
    prior = read_eval_log(prior_path)
    if prior.status == "success":
        return prior  # complete — reuse, nothing to re-judge (lean on the existing log)
    return eval_retry(prior_path, display=display)[0]


def _find_log_by_run_id(out: Path, run_id: str) -> str | None:
    from inspect_ai.log import read_eval_log

    for p in sorted((out / "logs").glob("*.eval")):
        try:
            if read_eval_log(str(p), header_only=True).eval.run_id == run_id:
                return str(p)
        except Exception:  # noqa: BLE001 — skip unreadable logs
            continue
    return None


def _apply_regression_gate(args: argparse.Namespace, log, gate_ok: bool) -> bool:
    from inspect_ai.log import read_eval_log

    from .compare import regression_gate, render_compare_table

    base_path = _resolve_eval(args.gate_baseline)
    if not base_path:
        print(f"error: no .eval log found for --gate-baseline {args.gate_baseline!r}", file=sys.stderr)
        return gate_ok
    base = read_eval_log(base_path)
    reg = regression_gate(
        base, log, load_validation(args.validation_artifact), max_regression=args.max_regression
    )
    print()
    print(render_compare_table(reg["diff"]))
    status = "PASS" if reg["passed"] else "FAIL"
    print(f"\nRegression gate: {status}" + (f"  ({reg['note']})" if reg.get("note") else ""))
    for f in reg["failures"]:
        print(f"  ✗ {f['metric']} {f['kind']}")
    return gate_ok and reg["passed"]


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


def _compare(args: argparse.Namespace) -> int:
    from inspect_ai.log import read_eval_log

    from .compare import CompareError, compare, regression_gate, render_compare_table

    base_path, cand_path = _resolve_eval(args.base), _resolve_eval(args.cand)
    for label, raw, resolved in (("base", args.base, base_path), ("cand", args.cand, cand_path)):
        if not resolved:
            print(f"error: no .eval log found for {label} ({raw!r}) — pass a .eval file or a run dir",
                  file=sys.stderr)
            return 2
    base, cand = read_eval_log(base_path), read_eval_log(cand_path)
    try:
        diff = compare(base, cand)
    except CompareError as e:
        print(f"refused: {e}", file=sys.stderr)  # an intentional comparability guard (§7.1)
        return 2

    print(render_compare_table(diff))
    reg = regression_gate(base, cand, load_validation(args.validation_artifact), max_regression=args.max_regression)
    status = "PASS" if reg["passed"] else "FAIL"
    print(f"\nRegression gate: {status}" + (f"  ({reg['note']})" if reg.get("note") else ""))
    for f in reg["failures"]:
        print(f"  ✗ {f['metric']} {f['kind']}")

    if args.card:
        from . import card

        ra = report.to_result_json(base, validation=load_validation())
        rb = report.to_result_json(cand, validation=load_validation())
        svg, png = card.render_head_to_head(ra, rb)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "head_to_head.svg").write_text(svg)
        (out / "head_to_head.png").write_bytes(png)
        print(f"head-to-head card: {out}/head_to_head.svg")

    return 0 if reg["passed"] else 1


def _resolve_eval(path_str: str) -> str | None:
    """Resolve a CLI arg to a single .eval file.

    Accepts a .eval file directly, OR a directory — a run ``--out`` dir or its
    ``logs/`` subdir — in which case the NEWEST .eval is chosen. This avoids the
    fragile ``logs/*.eval`` shell glob (Inspect appends a new timestamped log per
    run, so the glob can match several and shift argparse positionals).
    """
    p = Path(path_str)
    if p.is_file():
        return str(p)
    candidates: list[Path] = []
    if p.is_dir():
        candidates = list(p.glob("*.eval")) or list(p.glob("logs/*.eval"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda x: x.stat().st_mtime))


def _find_latest_log(out_dir: str) -> str | None:
    return _resolve_eval(out_dir)


def _explain(args: argparse.Namespace) -> int:
    from inspect_ai.log import read_eval_log

    from . import trust

    log_path = _resolve_eval(args.log) if args.log else _find_latest_log(args.out)
    if not log_path:
        print(f"error: no .eval log found (pass --log or run in {args.out} first)", file=sys.stderr)
        return 2
    log = read_eval_log(log_path)
    try:
        print(trust.explain(log, args.scenario, args.turn))
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _doctor(args: argparse.Namespace) -> int:
    """Preflight: data SHA, judge keys for the panel, resolved config. No scored calls."""
    cfg = load_config(args.config)
    ref = args.model or args.target or cfg.target or "mock"
    judges = args.judges or cfg.judges or "single"

    print("aipsy-bench doctor (preflight — no scored calls)\n")
    try:
        bundle.verify_integrity()
        print("  data/v1 integrity: OK (SHA-256 verified)")
        data_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  data/v1 integrity: FAIL — {e}")
        data_ok = False

    providers = spec.PROVIDERS if judges == "gold" else (spec.PRIMARY_JUDGE_PROVIDER,)
    if is_mock_ref(ref):
        print("  judges: mock target → offline mock judges, no provider keys needed")
        keys_ok = True
    else:
        print(f"  judges ({judges} panel):")
        keys_ok = True
        for p in providers:
            env = spec.API_ENV_VARS[p]
            present = bool(os.environ.get(env))
            keys_ok = keys_ok and present
            print(f"    {p} ({spec.JUDGE_MODEL_PINS[p]}): {env} {'present' if present else 'MISSING'}")

    print(f"\n  resolved: target={ref}  judges={judges}  data_version={spec.DATA_VERSION}")
    return 0 if (data_ok and keys_ok) else 1


def _provenance(args: argparse.Namespace) -> int:
    from inspect_ai.log import read_eval_log

    log_path = _resolve_eval(args.log) if args.log else _find_latest_log(args.out)
    if not log_path:
        print(f"error: no .eval log found (pass --log or run in {args.out} first)", file=sys.stderr)
        return 2
    log = read_eval_log(log_path)
    meta = log.samples[0].scores["clinical_judge_panel"].metadata if log.samples else {}
    snapshots = {
        j["judge"]: j.get("model_snapshot", "")
        for s in log.samples for pt in s.scores["clinical_judge_panel"].metadata.get("per_turn", [])
        for j in pt.get("per_judge", [])
    }
    print("aipsy-bench provenance")
    print(f"  tool_version : {__version__}")
    print(f"  data_version : {meta.get('data_version', spec.DATA_VERSION)}")
    print(f"  git_sha      : {_git_sha()}")
    print(f"  run_id       : {log.eval.run_id}")
    print(f"  target       : {log.eval.model}")
    print(f"  judge_panel  : {meta.get('judge_panel')}")
    print("  judge pins   :")
    for p, pin in spec.JUDGE_MODEL_PINS.items():
        resolved = snapshots.get(p)
        line = f"    {p}: {pin}" + (f"  (resolved: {resolved})" if resolved else "")
        print(line)
    return 0


def _publish_card(args: argparse.Namespace) -> int:
    """Opt-in lead-capture: write a ready-to-post bundle LOCALLY (no network, §12)."""
    from . import leaderboard

    run_dir = Path(args.run)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        print(f"error: no result.json under {run_dir} (run first)", file=sys.stderr)
        return 2
    import json

    result = json.loads(result_path.read_text())
    try:
        bundle_out = leaderboard.publish_card_bundle(result, run_dir / "publish")
    except leaderboard.PublishRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"publish bundle ready (local — nothing was sent): {bundle_out['out']}")
    print("Open a PR adding board_row.json + card.svg to the public gallery to publish.")
    return 0


def _cite(args: argparse.Namespace) -> int:
    from . import cite

    print(cite.bibtex())
    return 0


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
    r.add_argument("--judges", choices=["single", "gold"], default=None,
                   help="single = primary judge (fast inner loop); gold = 3-judge ensemble")
    r.add_argument("--quick", action="store_true", help="smoke subset: one scenario/domain + s06,s07")
    r.add_argument("--scenario", help="comma-separated scenario ids, e.g. s06,s07")
    r.add_argument("--baseline-prompt", action="store_true", help="inject the 014 baseline system prompt (§6 opt-in)")
    r.add_argument("--out", default=None, help="output directory for result.json + report.txt")
    r.add_argument("--no-card", action="store_true", help="skip share-card rendering")
    r.add_argument("--dry-run", action="store_true", help="estimate call counts/cost/time; make NO scored calls (§16)")
    r.add_argument("--max-cost", type=float, default=None, help="abort if the --dry-run cost estimate exceeds this (USD)")
    r.add_argument("--resume", default=None, help="resume a prior run by run_id (lean on Inspect's cache)")
    r.add_argument("--config", default=None, help="path to aipsy-bench.yaml (default: ./aipsy-bench.yaml)")
    r.add_argument("--validation-artifact", default=None, help="path to a 014 gate artifact (else PENDING)")
    r.add_argument("--gate-baseline", default=None, help="a baseline .eval log; fail on safety regression vs it (§7.1)")
    r.add_argument("--max-regression", type=float, default=0.3, help="max allowed drop on a gated metric vs baseline")
    r.add_argument("--against-board", action="store_true", help="overlay your score on the published vanilla baselines (§13.6)")
    r.add_argument("--domain", default=None, choices=["mental_health", "companion", "coaching"],
                   help="restrict --against-board to one domain")
    r.add_argument("--display", default="plain", help="Inspect display mode (plain|rich|none)")
    r.set_defaults(func=_run)

    d = sub.add_parser("doctor", help="preflight: data SHA, judge keys, resolved config (no scored calls, §16)")
    d.add_argument("--model")
    d.add_argument("--target")
    d.add_argument("--judges", choices=["single", "gold"], default=None)
    d.add_argument("--config", default=None)
    d.set_defaults(func=_doctor)

    pv = sub.add_parser("provenance", help="print tool/data version, judge pins + resolved snapshots, git SHA (§16)")
    pv.add_argument("--log", default=None, help="path to a .eval log (default: newest under --out/logs)")
    pv.add_argument("--out", default="aipsy-run")
    pv.set_defaults(func=_provenance)

    c = sub.add_parser("compare", help="diff two .eval logs → deltas + regression gate (§7.1)")
    c.add_argument("base", help="baseline: a .eval file OR a run dir (newest .eval is used)")
    c.add_argument("cand", help="candidate: a .eval file OR a run dir (newest .eval is used)")
    c.add_argument("--card", action="store_true", help="render the head-to-head card")
    c.add_argument("--out", default="aipsy-compare", help="output directory for the head-to-head card")
    c.add_argument("--max-regression", type=float, default=0.3)
    c.add_argument("--validation-artifact", default=None)
    c.set_defaults(func=_compare)

    e = sub.add_parser("explain", help="drill into one score: judge reasoning + anchors + target text (§15)")
    e.add_argument("scenario", help="scenario id, e.g. s06")
    e.add_argument("turn", type=int, help="turn number, e.g. 5")
    e.add_argument("--log", default=None, help="path to a .eval log (default: newest under --out/logs)")
    e.add_argument("--out", default="aipsy-run", help="run dir to autodiscover the log from")
    e.set_defaults(func=_explain)

    pc = sub.add_parser("publish-card", help="opt-in: prepare a ready-to-post card bundle locally (no network, §13.6)")
    pc.add_argument("--run", default="aipsy-run", help="run dir containing result.json")
    pc.set_defaults(func=_publish_card)

    ci = sub.add_parser("cite", help="print BibTeX for the OSF registration + tool/data version")
    ci.set_defaults(func=_cite)

    sc = sub.add_parser("scenarios", help="browse the public benchmark content")
    scsub = sc.add_subparsers(dest="scmd", required=True)
    lst = scsub.add_parser("list", help="list scenarios")
    lst.add_argument("--include-reserved", action="store_true")
    lst.set_defaults(func=_scenarios_list)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except bundle.BundleIntegrityError as e:
        # An expected, user-facing guard (§1.2) — present it cleanly, not as a traceback.
        print(f"error: frozen bundle integrity check failed (§1.2) — content has been "
              f"altered; refusing to proceed.\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
