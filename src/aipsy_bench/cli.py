"""``aipsy-bench`` CLI: ``run``, ``compare``, ``explain``, ``doctor``,
``provenance``, ``scenarios``. Wires Dataset+Solver+Scorer into one end-to-end run
that writes artifacts and drives a gate exit code, with the §16 operability verbs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from . import __version__, bundle, report, spec
from .config import CONFIG_NAME, HttpTargetSpec, load_config
from .gate import gate_result
from .targets import ResolvedTarget, http_target, is_mock_ref, resolve_target
from .task import aipsy_bench, quick_scenario_ids
from .validation import load_validation

# Rough per-call cost / throughput for --dry-run (NOT billing-accurate; a planning aid).
_EST_COST_PER_CALL = 0.0015
_EST_CALLS_PER_MIN = 60.0
_EST_LOCAL_CALLS_PER_MIN = 12.0  # local FT judge is slower per call than a frontier API

# provider → (SDK import module, pyproject extra) for clean "install the extra" hints.
# The provider SDKs are optional extras (offline/mock runs need none); a real run
# needs the SDK for the target provider + every judge in the panel.
_PROVIDER_SDK = {"openai": "openai", "anthropic": "anthropic", "google": "google.genai"}
_PROVIDER_EXTRA = {"openai": "openai", "anthropic": "anthropic", "google": "google"}


def _sdk_installed(provider: str) -> bool:
    module = _PROVIDER_SDK.get(provider)
    if module is None:
        return True  # unknown/other provider — let Inspect validate it
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _judge_panel_providers(judges: str) -> set[str]:
    """Frontier providers a judge panel needs (empty for the local panel — its judge
    runs on the local Ollama server, no provider SDK/key). Handles single:<provider>."""
    base, providers = spec.parse_panel(judges)
    return set() if base == "local" else set(providers)


def _required_providers(ref: str, judges: str) -> list[str]:
    """SDK modules a real run needs: the judge panel + the target provider. An
    ``ollama/<model>`` TARGET is driven via Inspect's OpenAI-compatible client, so it needs
    the ``openai`` module (no API key) — map it so the preflight catches a missing client."""
    provs = _judge_panel_providers(judges)
    target_provider = ref.split("/", 1)[0]
    if target_provider == "ollama":
        provs.add("openai")
    elif target_provider in _PROVIDER_SDK:
        provs.add(target_provider)
    return sorted(provs)


def _missing_sdk_hint(missing: list[str]) -> str:
    extras = sorted({_PROVIDER_EXTRA[p] for p in missing})
    per = " ".join(f"--extra {e}" for e in extras)
    return (
        f"error: missing provider SDK(s) for {missing} — needed for a real run "
        "(offline/--target mock needs none). Install the optional extras:\n"
        f"  uv sync {per}\n"
        "  uv sync --all-extras        # all judge providers (openai, anthropic, google)"
    )


def _local_judge_preflight(num_ctx: int | None = None) -> str | None:
    """Block a real-target local-judge run until Ollama + the FT model are ready."""
    from . import local_judge

    if not local_judge.ollama_running():
        return (
            "error: the local judge needs a running Ollama server (none reachable at "
            f"{spec.OLLAMA_BASE_URL}).\n"
            "  start it:       ollama serve\n"
            "  set up the judge: aipsy-bench judge pull\n"
            "  or use a frontier panel: --judges single|gold"
        )
    if not local_judge.model_present():
        return (
            f"error: the local judge model '{spec.LOCAL_JUDGE_TAG}' is not installed in Ollama.\n"
            "  set it up:               aipsy-bench judge pull\n"
            "  or use a frontier panel: --judges single|gold"
        )
    # Warm the model so the first scored call doesn't hit a cold-load timeout (the ~27 GB
    # Q8_0 model can take minutes to page into memory on the first request). Flush so the
    # message shows immediately, and bracket the (silent, blocking) load so it never looks hung.
    print(f"loading the local judge '{spec.LOCAL_JUDGE_TAG}' into memory "
          f"(one-time; the ~27 GB model at num_ctx {num_ctx or spec.LOCAL_JUDGE_NUM_CTX} can take "
          "a minute or two to page in) …", file=sys.stderr, flush=True)
    if not local_judge.warm_up(num_ctx=num_ctx):
        return (
            "error: the local judge model failed to load — Ollama may have run out of memory "
            f"(Q8_0 is ~{spec.LOCAL_JUDGE_RAM_GB} GB resident). Close other apps and retry, or "
            "use --judges single|gold."
        )
    print("local judge ready — starting the battery. Local scoring is slower than an API, so the "
          "first results take a moment; the live UI shows per-scenario progress.",
          file=sys.stderr, flush=True)
    return None


def _parse_judge_overrides(items: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``--judge-override PROVIDER=MODEL`` flags into a dict."""
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--judge-override must be PROVIDER=MODEL, got {item!r}")
        provider, model = (s.strip() for s in item.split("=", 1))
        if provider not in spec.PROVIDERS:
            raise ValueError(f"--judge-override provider must be one of {list(spec.PROVIDERS)}, got {provider!r}")
        if not model:
            raise ValueError(f"--judge-override needs a model for {provider!r}")
        out[provider] = model
    return out


def _parse_headers(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--header NAME:VALUE`` flags into a dict (split on the first ``:``)."""
    out: dict[str, str] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"--header must be NAME:VALUE, got {item!r}")
        name, value = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--header needs a name, got {item!r}")
        out[name] = value.strip()
    return out


def _http_target_params(args: argparse.Namespace, cfg) -> tuple[str, dict[str, str], str, str] | None:
    """Merge flags + config into ``(url, headers, conversation, ref)`` for an HTTP target.

    Returns ``None`` when no HTTP endpoint is specified. CLI flags override a
    config-declared endpoint (headers merge, the flag winning per key; ``${ENV}`` in the
    config is already expanded). Uses ``getattr`` so it works from the ``doctor`` namespace
    too (which omits ``--header/--ref/--conversation``).
    """
    cfg_http = cfg.target if isinstance(cfg.target, HttpTargetSpec) else None
    url = getattr(args, "http_target", None) or (cfg_http.http if cfg_http else None)
    if not url:
        return None
    headers = {**(cfg_http.headers if cfg_http else {}), **_parse_headers(getattr(args, "header", None))}
    conversation = getattr(args, "conversation", None) or (cfg_http.conversation if cfg_http else None) or "stateless"
    ref = getattr(args, "ref", None) or (cfg_http.ref if cfg_http else None) or url
    return url, headers, conversation, ref


def _resolve_run_target(args: argparse.Namespace, cfg) -> ResolvedTarget | None:
    """Resolve the target from flags + config, preferring a Tier-1 HTTP endpoint.

    An ``--http-target`` (or a ``target: {http: …}`` block in the yaml) is the
    zero-Python path for an app dev: the CLI builds the HTTP adapter directly — no
    driver script. Returns ``None`` if no target is specified at all.
    """
    cfg_http = cfg.target if isinstance(cfg.target, HttpTargetSpec) else None
    have_http = bool(args.http_target or cfg_http)

    if args.http_target and (args.model or args.target):
        raise ValueError("--http-target cannot be combined with --model/--target")
    if (args.header or args.ref or args.conversation) and not have_http:
        raise ValueError(
            "--header/--ref/--conversation apply to an HTTP target — pass --http-target "
            "<url> or set target: {http: …} in aipsy-bench.yaml"
        )

    params = _http_target_params(args, cfg)
    if params:
        url, headers, conversation, ref = params
        return http_target(url, headers=headers, conversation=conversation, ref=ref)

    ref = args.model or args.target or (cfg.target if isinstance(cfg.target, str) else None)
    if not ref:
        return None
    return resolve_target(ref)


def estimate(n_scenarios: int, judges: str) -> dict:
    target_calls = n_scenarios * spec.N_TURNS
    judge_calls = target_calls * (3 if judges == "gold" else 1)
    total = target_calls + judge_calls
    local = judges == "local"
    # The local judge runs on your machine: free, but slower per call. Only the target
    # calls are billed (and only if the target is a paid API — mock/local targets cost $0).
    billed_calls = target_calls if local else total
    rate = _EST_LOCAL_CALLS_PER_MIN if local else _EST_CALLS_PER_MIN
    return {
        "scenarios": n_scenarios,
        "target_calls": target_calls,
        "judge_calls": judge_calls,
        "total_calls": total,
        "local_judge": local,
        "est_cost_usd": billed_calls * _EST_COST_PER_CALL,
        "est_minutes": total / rate,
    }


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    try:
        # Resolve up front (non-network: builds the adapter, makes no call) so the
        # dry-run note + preflight can branch on the real adapter.
        resolved = _resolve_run_target(args, cfg)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if resolved is None:
        print("error: specify --target <ref>, --model <ref>, or --http-target <url> "
              "(or set target: in aipsy-bench.yaml)", file=sys.stderr)
        return 2

    judges = args.judges or cfg.judges or "local"
    quick = args.quick or cfg.quick
    scenario = args.scenario or cfg.scenario
    scenario_ids = [s.strip() for s in scenario.split(",")] if scenario else None
    out = Path(args.out or cfg.out or "aipsy-run")
    max_cost = args.max_cost if args.max_cost is not None else cfg.max_cost

    try:
        flag_overrides = _parse_judge_overrides(args.judge_override)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    judge_overrides = flag_overrides or (cfg.judge_overrides or None)

    n_scen = len(scenario_ids) if scenario_ids else (len(quick_scenario_ids()) if quick else spec.N_SCENARIOS)
    est = estimate(n_scen, judges)

    if args.dry_run:
        print(f"dry-run estimate ({judges} panel, {n_scen} scenarios × {spec.N_TURNS} turns):")
        print(f"  target calls: {est['target_calls']}   judge calls: {est['judge_calls']}   total: {est['total_calls']}")
        if est["local_judge"]:
            print(f"  judge: local Ollama ({spec.LOCAL_JUDGE_VERSION}) — free, runs on your machine")
        if resolved.adapter in ("http", "callable"):
            print(f"  target: your own endpoint ({resolved.meta.get('url', resolved.ref)}) — those calls "
                  "hit your server, not a billed provider (the cost line below assumes a paid API)")
        print(f"  est. cost: ~${est['est_cost_usd']:.2f}   est. wall-time: ~{est['est_minutes']:.1f} min")
        print("  (rough planning estimate — not billing-accurate; no scored calls made)")
        return 0

    if max_cost is not None and est["est_cost_usd"] > max_cost:
        print(f"aborting: estimated ~${est['est_cost_usd']:.2f} exceeds --max-cost ${max_cost:.2f}", file=sys.stderr)
        return 2

    if not resolved.is_mock:  # a real run needs the provider SDKs (optional extras)
        # An http/callable target is reached without a provider SDK (plain HTTP), so only
        # the judge panel needs SDKs there; a Tier-0 model string also needs its own provider.
        required = (
            _required_providers(resolved.ref, judges) if resolved.adapter == "model"
            else sorted(_judge_panel_providers(judges))
        )
        missing = [p for p in required if not _sdk_installed(p)]
        if missing:
            print(_missing_sdk_hint(missing), file=sys.stderr)
            return 2
        if judges == "local":  # the local judge needs a running Ollama + the FT model
            err = _local_judge_preflight(args.num_ctx)
            if err:
                print(err, file=sys.stderr)
                return 2

    task = aipsy_bench(
        target=resolved.ref, judges=judges, scenario_ids=scenario_ids,
        quick=quick, baseline_prompt=args.baseline_prompt,
        conversation=resolved.conversation,
        timeout=args.timeout, max_retries=args.max_retries,
        max_connections=args.max_connections, judge_overrides=judge_overrides,
        local_num_ctx=args.num_ctx,
    )

    if args.resume:
        log = _resume(out, args.resume, args.display)
        if log is None:
            print(f"error: no run with id {args.resume!r} under {out}/logs", file=sys.stderr)
            return 2
    else:
        # The local judge is a single serialized GPU resource, so concurrent scenarios
        # advance in lockstep and all finish at the end (the task progress bar would sit at
        # 0/N then jump). Run scenarios sequentially so progress is meaningful and an
        # interrupt leaves COMPLETE scenarios — no throughput loss, since the judge is the
        # serialized bottleneck either way (a local target shares the same GPU).
        eval_kwargs = {"max_samples": 1} if judges == "local" else {}
        # fail_on_error=False → one bad scenario never aborts a long battery; it is
        # logged (run failure / incomplete) and the rest still score.
        log = inspect_eval(task, model=resolved.model, display=args.display,
                           log_dir=str(out / "logs"), fail_on_error=False, **eval_kwargs)[0]

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
        if incomplete or result["run_failures"] or result["judge_overrides"]:
            print("note: cards skipped — run incomplete / target failures / judge override "
                  "(not card-eligible, §6/§8/§16)", file=sys.stderr)
        else:
            _emit_cards(result, out)

    print(report.render_report(result))
    if args.against_board:
        from . import leaderboard
        print()
        print(leaderboard.render_against_board(result, domain=args.domain))
    report_uri = (out / "report.html").resolve().as_uri()  # file:// → terminals linkify it (pytest-style)
    print(f"\nartifacts written to: {out}/result.json · {out}/report.txt · {out}/report.html")
    print(f"open the report:      {report_uri}")

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
    cfg_http = cfg.target if isinstance(cfg.target, HttpTargetSpec) else None
    http_url = args.http_target or (cfg_http.http if cfg_http else None)
    judges = args.judges or cfg.judges or "local"
    ref = None if http_url else (
        args.model or args.target or (cfg.target if isinstance(cfg.target, str) else None) or "mock"
    )

    print("aipsy-bench doctor (preflight — no scored calls)\n")
    try:
        bundle.verify_integrity()
        print("  data/v1 integrity: OK (SHA-256 verified)")
        data_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  data/v1 integrity: FAIL — {e}")
        data_ok = False

    ready = True
    if judges == "local":
        # The headline default: report local-judge readiness regardless of target.
        ready = _print_local_judge_doctor()
        if http_url:
            print(f"    note: target is your HTTP endpoint ({http_url}) — no provider key needed for "
                  "the target; the block above is your local-judge readiness. Start the endpoint "
                  "before `run`.")
        elif is_mock_ref(ref):
            print("    note: --target mock uses offline mock judges (no Ollama needed); the "
                  "block above is your readiness for a REAL target.")
        else:  # a real target still needs ITS provider key/SDK
            ready = _print_provider_doctor([ref.split("/", 1)[0]], ref, judges) and ready
    elif http_url:  # frontier judges over an HTTP target: only the judge panel needs keys
        print(f"  target: HTTP endpoint {http_url} — no provider key needed; ensure it is reachable.")
        ready = _print_provider_doctor(sorted(_judge_panel_providers(judges)), "__http__", judges)
    elif is_mock_ref(ref):
        print("  judges: mock target → offline mock judges, no keys or SDKs needed")
    else:
        ready = _print_provider_doctor(_required_providers(ref, judges), ref, judges)

    # An HTTP target: actually reach it once and classify (catches wrong port/path/secret
    # before a battery). Not a scored call — hits only the user's own endpoint.
    if http_url and not args.no_probe:
        from .targets import probe_endpoint

        _, headers, _, _ = _http_target_params(args, cfg)
        print(f"  probing {http_url} (one request; not a scored call) …")
        res = probe_endpoint(http_url, headers)
        print(f"    endpoint: {'OK' if res['ok'] else 'FAIL'} — {res['message']}")
        ready = ready and res["ok"]

    _print_path_recommendation()

    print(f"\n  resolved: target={http_url or ref}  judges={judges}  data_version={spec.DATA_VERSION}")
    return 0 if (data_ok and ready) else 1


def _print_local_judge_doctor() -> bool:
    from . import local_judge

    st = local_judge.status()
    print(f"  local judge ({st['version']}, served {st['quant']} via Ollama):")
    print(f"    Ollama server : {'reachable' if st['running'] else 'NOT reachable (run: ollama serve)'}")
    tag = st["tag"]
    if st["running"]:
        print(f"    model '{tag}' : {'installed' if st['model_present'] else 'MISSING (run: aipsy-bench judge pull)'}")
    ram = st["ram_gb"]
    if ram is not None:
        warn = (f"  ⚠ tight — {spec.LOCAL_JUDGE_RAM_RECOMMENDED_GB} GB+ recommended; on a Mac raise "
                "iogpu.wired_limit_mb (docs/local-judge.md)") if st["ram_tight"] else ""
        print(f"    system RAM    : {ram:.0f} GB{warn}")
    return bool(st["running"] and st["model_present"])


def _print_provider_doctor(providers: list[str], ref: str, judges: str) -> bool:
    label = "target" if providers == [ref.split("/", 1)[0]] and judges == "local" else f"{judges} panel + target"
    print(f"  providers ({label}) — key + SDK:")
    ready = True
    for p in providers:
        if p == "ollama":  # ollama target: Inspect's openai client lib, no API key
            sdk_ok = _sdk_installed("openai")
            ready = ready and sdk_ok
            print(f"    ollama target: openai client lib "
                  f"{'installed' if sdk_ok else 'MISSING (uv sync --extra local)'}; no key needed")
            continue
        if p not in spec.API_ENV_VARS:  # other non-keyed provider — let Inspect handle it
            continue
        env = spec.API_ENV_VARS[p]
        key_ok = bool(os.environ.get(env))
        sdk_ok = _sdk_installed(p)
        ready = ready and key_ok and sdk_ok
        sdk_str = "installed" if sdk_ok else f"MISSING (uv sync --extra {_PROVIDER_EXTRA[p]})"
        pin = spec.JUDGE_MODEL_PINS.get(p, "target")
        print(f"    {p} ({pin}): {env} {'present' if key_ok else 'MISSING'}; SDK {sdk_str}")
    return ready


def _print_api_lane_options() -> None:
    """The frontier-judge menu shared by the doctor recommendation when local is out."""
    from . import keys

    present = keys.current_keys()
    for p in spec.PROVIDERS:
        mark = "key present ✓" if present.get(p) else f"set {spec.API_ENV_VARS[p]}"
        print(f"      • --judges single:{p:<9} ({spec.JUDGE_MODEL_PINS[p]}) — {mark}")
    print("      • --judges gold          (all three — the comparable/citable panel)")
    if not any(present.values()):
        print("      get started: `aipsy-bench keys set --provider <provider>`  then re-run doctor")


def _print_path_recommendation() -> None:
    """Hardware-aware steer between the two judge regimes (§0.3 local-first posture).

    Local (offline 26B FT judge) is a real hardware gate, and it is PLATFORM-AWARE:
    a Mac needs ~48 GB unified memory (32–48 GB loads but is unusably slow); a discrete-
    GPU Linux box needs ~16 GB VRAM + 64 GB RAM. Missing Ollama is a one-time install,
    NEVER a viability blocker."""
    from . import local_judge

    v = local_judge.viability()
    print("\n  Recommended path:")
    if v["viable"]:
        caveat = "  ⚠ tight — see docs/local-judge.md" if v["band"] == "tight" else ""
        print(f"    ✓ LOCAL (priority) — offline FT judge, no API keys. {v['detail']}.{caveat}")
        if not v["ollama_running"]:
            print("      one-time setup (not a blocker): install Ollama (ollama.com) + "
                  "`aipsy-bench judge pull`")
        elif not v["model_present"]:
            print("      one-time setup: `aipsy-bench judge pull`  (Ollama is running)")
        else:
            print("      ready: `aipsy-bench run --judges local`")
        return
    if v["band"] == "slow":  # Mac 32–48 GB: technically runs, but don't recommend it
        print(f"    ⚠ LOCAL possible but SLOW — {v['detail']}. Recommend the API lane; "
              "force with `--judges local` only if you accept the speed.")
    else:
        print(f"    ✗ LOCAL not viable — {v['detail']} (needs {v['requirement']}). Use the API lane:")
    _print_api_lane_options()


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
    panel = meta.get("judge_panel")
    print(f"  judge_panel  : {panel}")
    if panel == "local":
        resolved = snapshots.get("local")
        print("  local judge  :")
        print(f"    {spec.LOCAL_JUDGE_VERSION}  (Ollama tag {spec.LOCAL_JUDGE_TAG}, served {spec.LOCAL_JUDGE_QUANT})"
              + (f"  (resolved: {resolved})" if resolved else ""))
        print(f"    gguf sha256: {spec.LOCAL_JUDGE_GGUF_SHA256}")
        print(f"    weights    : hf.co/{spec.LOCAL_JUDGE_HF_REPO}")
    else:
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


def _judge_status(args: argparse.Namespace) -> int:
    """Check the local Ollama judge is ready (no scored calls)."""
    from . import local_judge

    st = local_judge.status()
    print(f"local judge: {st['version']}  (Ollama tag {st['tag']}, served {st['quant']})")
    print(f"  Ollama server : {'reachable' if st['running'] else 'NOT reachable — run: ollama serve'}")
    if st["running"]:
        present = "yes" if st["model_present"] else "no — run: aipsy-bench judge pull"
        print(f"  model present : {present}")
    if st["ram_gb"] is not None:
        tight = (f"  ⚠ tight — {spec.LOCAL_JUDGE_RAM_RECOMMENDED_GB} GB+ recommended "
                 "(see docs/local-judge.md)") if st["ram_tight"] else ""
        print(f"  system RAM    : {st['ram_gb']:.0f} GB{tight}")
    print(f"  weights       : hf.co/{spec.LOCAL_JUDGE_HF_REPO}")
    ok = bool(st["running"] and st["model_present"])
    if not ok:
        print("\nnot ready — run: aipsy-bench judge pull")
    return 0 if ok else 1


def _judge_pull(args: argparse.Namespace) -> int:
    """Download the FT GGUF from HF (via HF_TOKEN) + register the Ollama tag."""
    from . import local_judge

    res = local_judge.ensure_model(force=args.force)
    return 0 if res["status"] in ("ready", "created") else 1


def _judge_warm(args: argparse.Namespace) -> int:
    """Pre-load the model into memory (absorbs the one-time cold-load before a run)."""
    from . import local_judge

    if not local_judge.ollama_running():
        print("error: Ollama is not running — start it with: ollama serve", file=sys.stderr)
        return 1
    if not local_judge.model_present():
        print("error: model not installed — run: aipsy-bench judge pull", file=sys.stderr)
        return 1
    print(f"loading '{spec.LOCAL_JUDGE_TAG}' into memory (first load can take a few minutes) …")
    ok = local_judge.warm_up()
    print("ready — model is resident and warm." if ok
          else f"FAILED to load (out of memory? Q8_0 is ~{spec.LOCAL_JUDGE_RAM_GB} GB).")
    return 0 if ok else 1


def _prompt_provider() -> str | None:
    print("Select provider:")
    for i, p in enumerate(spec.PROVIDERS, 1):
        print(f"  [{i}] {p} ({spec.JUDGE_MODEL_PINS[p]})")
    choice = input("> ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(spec.PROVIDERS):
        return spec.PROVIDERS[int(choice) - 1]
    if choice in spec.PROVIDERS:
        return choice
    return None


def _keys_set(args: argparse.Namespace) -> int:
    """Interactively save a provider key to the local .env (input hidden, never via argv)."""
    import getpass

    from . import keys

    if args.provider:
        provider = args.provider
    elif sys.stdin.isatty():
        provider = _prompt_provider()
        if provider is None:
            print("error: invalid selection", file=sys.stderr)
            return 2
    else:
        print("error: --provider required (no interactive terminal)", file=sys.stderr)
        return 2

    var = spec.API_ENV_VARS[provider]
    try:
        key = getpass.getpass(f"Paste {var} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\naborted", file=sys.stderr)
        return 1
    if not key:
        print(f"{var}: nothing entered — not saved", file=sys.stderr)
        return 1

    path = keys.set_provider_key(provider, key)
    print(f"saved {var} → {path}  [{keys.mask(key)}]  (gitignored — never committed)")
    print("verify: aipsy-bench doctor --judges <single|gold>")
    return 0


def _keys_path(args: argparse.Namespace) -> int:
    from . import keys

    print(keys.env_path())
    return 0


def _keys_status(args: argparse.Namespace) -> int:
    from . import keys

    print(f".env: {keys.env_path()}")
    for provider, present in keys.current_keys().items():
        print(f"  {spec.API_ENV_VARS[provider]}: {'present' if present else 'not set'}")
    return 0


def _render_config(params: dict) -> str:
    """Render an ``aipsy-bench.yaml`` from init params. A secret is written as a ``${ENV}``
    *reference*, never the value — so the committed config never carries a secret."""
    lines = [
        "# aipsy-bench project config — committed & team-shareable. CLI flags override these.",
        "# docs: docs/adapters/eval-endpoint.md",
        "",
    ]
    if params["kind"] == "http":
        lines.append("target:")
        lines.append(f"  http: {params['http']}")
        if params.get("secret_header") and params.get("secret_env"):
            lines.append("  headers:")
            lines.append(f"    {params['secret_header']}: ${{{params['secret_env']}}}"
                         "   # ${ENV} expanded at load — the secret stays out of git")
        lines.append(f"  conversation: {params.get('conversation', 'stateless')}")
    else:
        lines.append(f"target: {params['model_ref']}")
    lines.append(f"judges: {params.get('judges', 'local')}")
    lines.append("")
    return "\n".join(lines)


def _maybe_set_key(provider: str) -> None:
    """Offer to save the chosen provider's key to the local .env inline during init
    (hidden input, never via argv). Skipping is fine — doctor re-surfaces it later."""
    import getpass

    from . import keys

    var = spec.API_ENV_VARS[provider]
    if input(f"Set {var} now? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        return
    try:
        key = getpass.getpass(f"  paste {var} (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("  skipped")
        return
    if key:
        path = keys.set_provider_key(provider, key)
        print(f"  saved {var} → {path}  [{keys.mask(key)}]  (gitignored)")
    else:
        print("  skipped (nothing entered)")


def _init_prompt_single_provider(present: dict[str, bool]) -> str:
    """Pick the frontier provider for the single lane — the key the dev holds becomes
    their default — or 'gold' for all three. Offers to set a missing key inline. Returns
    the canonical panel string (single:<primary> collapses to plain 'single')."""
    for i, p in enumerate(spec.PROVIDERS, 1):
        mark = "  ← key present" if present.get(p) else ""
        print(f"  [{i}] {p}  ({spec.JUDGE_MODEL_PINS[p]}){mark}")
    gold_choice = len(spec.PROVIDERS) + 1
    print(f"  [{gold_choice}] gold — all three (comparable/citable, needs 3 keys)")
    sel = (input("> ").strip() or "1")
    if sel == str(gold_choice):
        return "gold"
    provider = (spec.PROVIDERS[int(sel) - 1]
                if sel.isdigit() and 1 <= int(sel) <= len(spec.PROVIDERS)
                else spec.PRIMARY_JUDGE_PROVIDER)
    if not present.get(provider):
        _maybe_set_key(provider)
    return spec.parse_panel(f"single:{provider}")[0]


def _init_prompt_judges() -> str:
    """Recommend a judge lane from the machine's hardware (§0.3 local-first): local when
    the box can run the 26B FT judge at a usable speed (platform-aware), else steer to a
    single frontier judge the dev holds a key for. Interactive."""
    from . import keys, local_judge

    v = local_judge.viability()
    present = keys.current_keys()
    if v["viable"]:
        caveat = " (tight)" if v["band"] == "tight" else ""
        print(f"Judge lane — this machine can run the offline local judge ({v['detail']}{caveat}). "
              "[recommended]")
        print("  [1] local (offline, no keys)   [2] single (one frontier key)   "
              "[3] gold (three keys)")
        base = {"1": "local", "2": "single", "3": "gold"}.get(input("> ").strip() or "1", "local")
        return base if base != "single" else _init_prompt_single_provider(present)
    if v["band"] == "slow":
        print(f"Judge lane — the local judge would run unusably slowly here ({v['detail']}); "
              "using API judges (force local later with --judges local if you accept the speed).")
    else:
        print(f"Judge lane — the local judge is not viable here ({v['detail']}; "
              f"needs {v['requirement']}); using API judges.")
    print("Pick the frontier judge you have a key for (it becomes your run default):")
    return _init_prompt_single_provider(present)


def _init_prompt() -> dict | None:
    print("aipsy-bench init — scaffold aipsy-bench.yaml\n")
    print("What are you benchmarking?")
    print("  [1] your own app via an HTTP /eval endpoint (recommended)")
    print("  [2] a bare model string, e.g. openai/gpt-5.4-mini")
    if (input("> ").strip() or "1") == "2":
        ref = input("model string: ").strip()
        return {"kind": "model", "model_ref": ref, "judges": _init_prompt_judges()} if ref else None
    url = input("endpoint URL (e.g. http://localhost:3000/eval): ").strip()
    if not url:
        return None
    secret_header = secret_env = None
    if input("Is it gated by a secret header? [y/N]: ").strip().lower() == "y":
        secret_header = input("  header name [x-eval-secret]: ").strip() or "x-eval-secret"
        secret_env = input("  env var holding the secret [EVAL_SECRET]: ").strip() or "EVAL_SECRET"
    return {"kind": "http", "http": url, "secret_header": secret_header, "secret_env": secret_env,
            "conversation": "stateless", "judges": _init_prompt_judges()}


def _init(args: argparse.Namespace) -> int:
    """Scaffold ``aipsy-bench.yaml`` — interactively, or fully from flags (scriptable)."""
    out = Path(args.out or CONFIG_NAME)
    if out.exists() and not args.force:
        print(f"error: {out} already exists — pass --force to overwrite", file=sys.stderr)
        return 2
    if bool(args.secret_header) != bool(args.secret_env):
        print("error: provide both --secret-header and --secret-env (or neither)", file=sys.stderr)
        return 2

    if args.http_target or args.model:  # flag-driven (scriptable / non-interactive)
        if args.http_target:
            params = {"kind": "http", "http": args.http_target,
                      "secret_header": args.secret_header, "secret_env": args.secret_env,
                      "conversation": args.conversation or "stateless", "judges": args.judges or "local"}
        else:
            params = {"kind": "model", "model_ref": args.model, "judges": args.judges or "local"}
    elif sys.stdin.isatty():
        params = _init_prompt()
        if params is None:
            print("aborted — nothing written", file=sys.stderr)
            return 1
    else:
        print("error: --http-target <url> or --model <ref> required (no interactive terminal)", file=sys.stderr)
        return 2

    out.write_text(_render_config(params))
    print(f"wrote {out}")
    print(f"  target: {params.get('http') or params.get('model_ref')}   judges: {params.get('judges', 'local')}")

    print("\nnext steps:")
    step = 1
    if params.get("secret_env"):
        print(f"  {step}. put the secret in your .env (it's your endpoint's, not a provider key):")
        print(f"       echo '{params['secret_env']}=<your-secret>' >> .env")
        step += 1
    if params["kind"] == "http":
        print(f"  {step}. start your dev server, then preflight:  aipsy-bench doctor")
        step += 1
    jbase = spec.panel_base(params.get("judges", "local"))
    if jbase == "local":
        print(f"  {step}. set up the local judge:  aipsy-bench judge pull")
        step += 1
    elif jbase in ("single", "gold"):
        from . import keys

        present = keys.current_keys()
        need = [p for p in sorted(_judge_panel_providers(params["judges"])) if not present.get(p)]
        if need:
            print(f"  {step}. set the judge key(s):  "
                  + "  ".join(f"aipsy-bench keys set --provider {p}" for p in need))
            step += 1
    print(f"  {step}. run it:  aipsy-bench run --quick")
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
    r.add_argument("--http-target", default=None, metavar="URL",
                   help="Tier-1 HTTP /eval endpoint (POST {messages:[...]} -> {reply}) — the "
                        "zero-Python path for benchmarking your own app; see docs/adapters/")
    r.add_argument("--header", action="append", metavar="NAME:VALUE",
                   help="header for --http-target (repeatable), e.g. x-eval-secret:$EVAL_SECRET")
    r.add_argument("--ref", default=None,
                   help="label for an --http-target in the report/card (default: the URL)")
    r.add_argument("--conversation", choices=["stateless", "session"], default=None,
                   help="--http-target history mode: stateless (default; the bench replays the full "
                        "transcript) or session (the target owns history; only the new turn is sent)")
    r.add_argument("--judges", choices=list(spec.JUDGE_CHOICES), default=None, metavar="LANE",
                   help="local = offline FT judge via Ollama (default, no keys); "
                        "single[:provider] = one frontier judge (openai|anthropic|google — "
                        "the key you hold; directional); gold = 3-judge frontier ensemble")
    r.add_argument("--quick", action="store_true", help="smoke subset: one scenario/domain + s06,s07")
    r.add_argument("--scenario", help="comma-separated scenario ids, e.g. s06,s07")
    r.add_argument("--baseline-prompt", action="store_true", help="inject the 014 baseline system prompt (§6 opt-in)")
    r.add_argument("--out", default=None, help="output directory for result.json + report.txt")
    r.add_argument("--no-card", action="store_true", help="skip share-card rendering")
    r.add_argument("--dry-run", action="store_true", help="estimate call counts/cost/time; make NO scored calls (§16)")
    r.add_argument("--max-cost", type=float, default=None, help="abort if the --dry-run cost estimate exceeds this (USD)")
    r.add_argument("--resume", default=None, help="resume a prior run by run_id (lean on Inspect's cache)")
    r.add_argument("--timeout", type=int, default=None,
                   help=f"per-call timeout in seconds (default {spec.MODEL_TIMEOUT}) — bounds a hung/slow call")
    r.add_argument("--max-retries", type=int, default=None,
                   help=f"max retries per call (default {spec.MODEL_MAX_RETRIES}) — bounds rate-limit backoff")
    r.add_argument("--max-connections", type=int, default=None,
                   help="cap concurrent calls per provider — lower it (e.g. 2-4) to ease rate limiting")
    r.add_argument("--num-ctx", type=int, default=None,
                   help=f"local judge context window (default {spec.LOCAL_JUDGE_NUM_CTX}) — raise for "
                        "very verbose targets (deep-turn JudgeParseErrors), lower on a memory-tight box")
    r.add_argument("--judge-override", action="append", metavar="PROVIDER=MODEL",
                   help="swap a judge for testing, e.g. anthropic=claude-haiku-4-5 — makes the run "
                        "NON-comparable (not the frozen instrument, not board/card eligible, §8)")
    r.add_argument("--config", default=None, help="path to aipsy-bench.yaml (default: ./aipsy-bench.yaml)")
    r.add_argument("--validation-artifact", default=None, help="path to a 014 gate artifact (else PENDING)")
    r.add_argument("--gate-baseline", default=None, help="a baseline .eval log; fail on safety regression vs it (§7.1)")
    r.add_argument("--max-regression", type=float, default=0.3, help="max allowed drop on a gated metric vs baseline")
    r.add_argument("--against-board", action="store_true", help="overlay your score on the published vanilla baselines (§13.6)")
    r.add_argument("--domain", default=None, choices=["mental_health", "companion", "coaching"],
                   help="restrict --against-board to one domain")
    r.add_argument("--display", default=None,
                   help="Inspect display mode (default: auto — live full UI in a terminal, "
                        "plain when piped). Options: full|rich|plain|log|none")
    r.set_defaults(func=_run)

    d = sub.add_parser("doctor", help="preflight: data SHA, judge readiness, resolved config (no scored calls, §16)")
    d.add_argument("--model")
    d.add_argument("--target")
    d.add_argument("--http-target", default=None, metavar="URL", help="check readiness for an HTTP /eval target")
    d.add_argument("--no-probe", action="store_true",
                   help="skip the --http-target connectivity probe (which sends one request to your endpoint)")
    d.add_argument("--judges", choices=list(spec.JUDGE_CHOICES), default=None, metavar="LANE")
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

    j = sub.add_parser("judge", help="set up / check the local Ollama judge (the offline default)")
    jsub = j.add_subparsers(dest="jcmd", required=True)
    jstatus = jsub.add_parser("status", help="check Ollama + the FT model are ready (no scored calls)")
    jstatus.set_defaults(func=_judge_status)
    jpull = jsub.add_parser("pull", help="download the FT GGUF from HF (uses HF_TOKEN) + register the Ollama tag")
    jpull.add_argument("--force", action="store_true", help="re-create the tag even if already present")
    jpull.set_defaults(func=_judge_pull)
    jwarm = jsub.add_parser("warm", help="pre-load the model into memory (absorbs the one-time cold load)")
    jwarm.set_defaults(func=_judge_warm)

    k = sub.add_parser("keys", help="save provider API keys to your local .env (your keys, your cost)")
    ksub = k.add_subparsers(dest="kcmd", required=True)
    kset = ksub.add_parser("set", help="interactively save a provider key to .env (input hidden)")
    kset.add_argument("--provider", choices=list(spec.PROVIDERS),
                      help="provider to set (omit for an interactive picker)")
    kset.set_defaults(func=_keys_set)
    kstatus = ksub.add_parser("status", help="show which provider keys are present in .env (no values)")
    kstatus.set_defaults(func=_keys_status)
    kpath = ksub.add_parser("path", help="print the .env path that will be used")
    kpath.set_defaults(func=_keys_path)

    ini = sub.add_parser("init", help="scaffold aipsy-bench.yaml for your target (interactive, or via flags)")
    ini.add_argument("--http-target", default=None, metavar="URL", help="HTTP /eval endpoint to benchmark")
    ini.add_argument("--model", default=None, help="a bare Inspect model string instead of an HTTP target")
    ini.add_argument("--secret-header", default=None, metavar="NAME",
                     help="secret header name for a gated endpoint (with --secret-env)")
    ini.add_argument("--secret-env", default=None, metavar="ENVVAR",
                     help="env var the secret header reads — written to the config as ${ENVVAR}, never the value")
    ini.add_argument("--conversation", choices=["stateless", "session"], default=None)
    ini.add_argument("--judges", choices=list(spec.JUDGE_CHOICES), default=None, metavar="LANE",
                     help="default: hardware-recommended (local if viable, else single:<provider>)")
    ini.add_argument("--out", default=None, help=f"config path (default: ./{CONFIG_NAME})")
    ini.add_argument("--force", action="store_true", help="overwrite an existing config")
    ini.set_defaults(func=_init)

    sc = sub.add_parser("scenarios", help="browse the public benchmark content")
    scsub = sc.add_subparsers(dest="scmd", required=True)
    lst = scsub.add_parser("list", help="list scenarios")
    lst.add_argument("--include-reserved", action="store_true")
    lst.set_defaults(func=_scenarios_list)

    return p


def _load_dotenv() -> None:
    """Load a project ``.env`` into the environment (provider keys), matching what
    Inspect does on the eval path — so ``doctor``/``--dry-run`` see the same keys a
    real run will. ``override=False`` → an exported shell var wins. We never store or
    transmit keys; they're read from the environment and passed to the providers."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)


try:  # private path, but stable — used only as a clean-message backstop
    from inspect_ai._util.error import PrerequisiteError as _PrereqError
except Exception:  # noqa: BLE001
    _PrereqError = ()  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except bundle.BundleIntegrityError as e:
        # An expected, user-facing guard (§1.2) — present it cleanly, not as a traceback.
        print(f"error: frozen bundle integrity check failed (§1.2) — content has been "
              f"altered; refusing to proceed.\n{e}", file=sys.stderr)
        return 1
    except _PrereqError:
        # Backstop for a missing provider SDK not caught by the run preflight.
        print("error: a provider SDK is missing. Install the optional extras:\n"
              "  uv sync --all-extras   (or --extra openai / anthropic / google)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
