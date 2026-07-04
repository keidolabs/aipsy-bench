"""Flatten the immutable Inspect ``.eval`` log into a citable ``result.json`` +
a human-readable report (§4.3). The ``.eval`` log itself stays the source of
record — this is the derived, CI/human-friendly view.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from . import __version__
from . import spec
from .gate import evaluate_gate
from .validation import JudgeValidation, directional_banner, local_judge_banner

DATA_VERSION = spec.DATA_VERSION
SCORER_NAME = "clinical_judge_panel"
_NUMERIC_KEYS = (*spec.METRICS, "AI_Trust")


def _na_mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None


def _scores(sample) -> dict:
    return sample.scores[SCORER_NAME].value


def _meta(sample) -> dict:
    return sample.scores[SCORER_NAME].metadata


def _overall(samples) -> dict:
    out = {k: _na_mean([_scores(s).get(k) for s in samples]) for k in _NUMERIC_KEYS}
    out["crisis_detected_any"] = any(_scores(s).get("crisis_detected_any") for s in samples)
    return out


def _by_domain(samples) -> dict:
    domains: dict[str, list] = {}
    for s in samples:
        domains.setdefault(_meta(s)["domain"], []).append(s)
    return {d: {k: _na_mean([_scores(s).get(k) for s in subs]) for k in _NUMERIC_KEYS}
            for d, subs in domains.items()}


def _by_phase(samples) -> dict:
    phases = ("1", "2", "3", "4")
    out: dict[str, dict] = {}
    for p in phases:
        rows = [_meta(s)["phase_breakdown"].get(p) for s in samples]
        rows = [r for r in rows if r]
        if not rows:
            continue
        out[p] = {
            "name": spec.PHASE_NAMES.get(int(p), p),
            "AI_Trust": _na_mean([r.get("composite") for r in rows]),
            **{m: _na_mean([r.get("metrics", {}).get(m) for r in rows]) for m in spec.METRICS},
        }
    return out


def _by_scenario(samples) -> dict:
    out = {}
    for s in samples:
        v = _scores(s)
        row = {k: v.get(k) for k in (*_NUMERIC_KEYS, "crisis_detected_any")}
        if v.get("run_failure"):  # surface run failures distinctly from low scores (§6)
            row["run_failure"] = True
        out[s.id] = row
    return out


def _target_block(log, override: dict | None) -> dict:
    if override:
        return override
    model = log.eval.model or ""
    adapter = "mock" if model.startswith("mockllm") else "model"
    return {"adapter": adapter, "ref": model, "model_snapshot": model}


def to_result_json(
    log,
    *,
    validation: JudgeValidation,
    target: dict | None = None,
    mode: str = "benchmark",
    gate: dict | None = None,
    extra_warnings: list[str] | None = None,
    incomplete: bool = False,
) -> dict:
    """Flatten an Inspect ``.eval`` log to the §4.3 result.json schema."""
    # Tolerate samples that errored / were interrupted before scoring (no score attached).
    all_samples = log.samples or []
    samples = [s for s in all_samples if SCORER_NAME in s.scores]
    if len(samples) < len(all_samples):
        incomplete = True

    panel = _meta(samples[0])["judge_panel"] if samples else "single"
    judge_versions = _meta(samples[0])["judge_versions"] if samples else {}
    by_scenario = {s.id: _scores(s) for s in samples}

    warnings: list[str] = list(extra_warnings or [])
    if incomplete:
        warnings.append("run INCOMPLETE — partial results; not gated, not carded (§16)")
    if spec.panel_base(panel) == "single":
        prov = spec.parse_panel(panel)[1][0]  # name the frontier judge (single is provider-selectable)
        warnings.append(
            f"single-judge panel ({prov}: {spec.JUDGE_MODEL_PINS[prov]}) — directional; "
            "scores NOT comparable to published gold numbers"
        )
    elif panel == "local":
        warnings.append(local_judge_banner())
    banner = directional_banner(validation)
    if banner:
        warnings.append(banner)

    run_failures = [
        {"scenario_id": s.id, **_meta(s)["run_failure"]}
        for s in samples if _meta(s).get("run_failure")
    ]
    if run_failures:
        warnings.append(
            f"{len(run_failures)} scenario(s) had target failures — run incomplete; "
            "not gate-passable, not card/board eligible (§6)"
        )

    judge_overrides = _meta(samples[0]).get("judge_overrides", {}) if samples else {}
    if judge_overrides:
        swaps = ", ".join(f"{p}: {v['from']}→{v['to']}" for p, v in judge_overrides.items())
        warnings.append(
            f"judge override active ({swaps}) — NOT the frozen instrument; non-comparable, "
            "not board/card eligible, do not cite (§8)"
        )

    judge_failures = [
        {"scenario_id": s.id, **jf}
        for s in samples for jf in _meta(s).get("judge_failures", [])
    ]
    if judge_failures:
        warnings.append(
            f"{len(judge_failures)} judge call(s) failed (parse error or provider error such "
            "as rate limit) — those (turn, judge) scores were dropped (degraded, not crashed); "
            "not board eligible"
        )

    diagnostics = [
        {"scenario_id": s.id, "card": s.scores[SCORER_NAME].explanation}
        for s in samples
        if s.scores[SCORER_NAME].explanation
        and not _meta(s).get("run_failure")
        and "no metric failed" not in s.scores[SCORER_NAME].explanation
    ]

    conversation = _meta(samples[0]).get("conversation", "stateless") if samples else "stateless"
    gate = gate if gate is not None else evaluate_gate(by_scenario, validation)

    from . import trust

    target_block = _target_block(log, target)
    disagreement = trust.judge_disagreement(samples)
    self_pref = trust.self_preference(target_block["ref"], judge_versions)
    # "Self-judging" = EVERY judge shares the target's family (the single:<p> + <p>-target
    # case) — a stronger confound than gold's 1-of-3 partial overlap. Alert prominently with
    # an actionable fix; a partial overlap stays an informational confound note.
    self_judging = bool(self_pref) and set(self_pref) == set(judge_versions)
    if self_judging:
        other = next((p for p in spec.PROVIDERS if p not in self_pref), "anthropic")
        warnings.append(
            f"⚠ SELF-JUDGING — the judge and the target are the same provider "
            f"({', '.join(self_pref)}). A model tends to favor its own family, so these scores may "
            f"be falsely ELEVATED (self-preference bias, §15). For a less biased read, judge with a "
            f"different provider: --judges single:{other}  (or --judges gold, where 2 of 3 judges "
            "are independent)."
        )
    elif self_pref:
        warnings.append(
            f"self-preference confound — target family intersects judge(s) {self_pref}; "
            "those judges may favor the target (§15)"
        )

    return {
        "tool": "aipsy-bench",
        "tool_version": __version__,
        "data_version": DATA_VERSION,
        "mode": mode,
        "incomplete": incomplete,
        "conversation": conversation,
        "run_id": log.eval.run_id,
        "timestamp": log.eval.created,
        "target": target_block,
        "judge_panel": panel,
        "judge_versions": judge_versions,
        "judge_overrides": judge_overrides,
        "judge_validation": validation.model_dump(),
        "scores": {
            "overall": _overall(samples),
            "by_domain": _by_domain(samples),
            "by_phase": _by_phase(samples),
            "by_scenario": _by_scenario(samples),
        },
        "gate": gate,
        "diagnostics": diagnostics,
        "run_failures": run_failures,
        "judge_failures": judge_failures,
        "judge_disagreement": disagreement,
        "self_preference": self_pref,
        "self_judging": self_judging,
        "warnings": warnings,
    }


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _judge_models_str(result_json: dict) -> str:
    """The concrete judge model(s) behind the panel label, so the report names WHICH judge —
    e.g. ``(openai/gpt-5.4-mini)`` for single, all three for gold — not just ``single``."""
    versions = result_json.get("judge_versions") or {}
    if not versions:
        return ""
    parts = [m if p == "local" else f"{p}/{m}" for p, m in versions.items()]
    return "(" + ", ".join(parts) + ")"


# A target error's detail (from Inspect) embeds the full request payload — pull out the
# actual provider message / status so the report shows WHY, not a JSON dump.
_ERR_MSG_RE = re.compile(r"""['"]message['"]\s*:\s*(['"])(.+?)\1""")
_ERR_CODE_RE = re.compile(r"code:\s*\d{3}\b.*")


def _summarize_error(detail: str, limit: int = 200) -> str:
    """Concise, human-useful reason from a captured target-error ``detail`` — the provider's
    message (``'message': "…"``) or an HTTP-ish ``code: NNN …`` tail, stripped of the request
    payload Inspect embeds. Empty string if there's nothing meaningful."""
    if not detail:
        return ""
    etype = detail.split(":", 1)[0].strip()
    cleaned = detail.replace('\\"', '"').replace("\\'", "'")  # error text often repr-escapes quotes
    m = _ERR_MSG_RE.search(cleaned)
    if m:
        core = m.group(2)
    else:
        m2 = _ERR_CODE_RE.search(cleaned)
        core = m2.group(0) if m2 else re.split(r"\n?Request:", cleaned, maxsplit=1)[0]
    core = " ".join(core.split())
    out = core if (not etype or core.startswith(etype)) else f"{etype}: {core}"
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _systematic_failure_hint(run_failures: list) -> str | None:
    """When every scenario dies identically on turn 1, it's a target/config problem, not
    flaky content — point the dev at the likely causes instead of leaving N opaque errors."""
    if len(run_failures) < 2:
        return None
    first = run_failures[0]
    if first.get("turn") == 1 and all(
        rf.get("turn") == 1 and rf.get("status") == first.get("status") for rf in run_failures
    ):
        return (
            f"all {len(run_failures)} scenarios failed on turn 1 with the same target error — "
            "this is a target config/access issue, not flaky content. Common causes: rate limit "
            "(retry, or lower concurrency with --max-connections 1), the model isn't accessible on "
            "your key, or quota. Run `aipsy-bench doctor` and check the reason above."
        )
    return None


def render_report(result_json: dict) -> str:
    """Human-readable text report; prints the §0.3 directional banner prominently."""
    lines: list[str] = []
    banner = next((w for w in result_json["warnings"] if w.startswith("⚠ DIRECTIONAL")), None)
    if banner:
        bar = "═" * 78
        lines += [bar, banner, bar, ""]
    self_judging = next((w for w in result_json["warnings"] if w.startswith("⚠ SELF-JUDGING")), None)
    if self_judging:
        bar = "─" * 78
        lines += [bar, self_judging, bar, ""]

    t = result_json["target"]
    judges = f"{result_json['judge_panel']} {_judge_models_str(result_json)}".rstrip()
    lines.append(f"aipsy-bench {result_json['tool_version']} · data {result_json['data_version']} · mode {result_json['mode']}")
    lines.append(f"target: {t['ref']} ({t['adapter']})  ·  judges: {judges}")
    lines.append("")

    lines.append("Overall scores:")
    overall = result_json["scores"]["overall"]
    for k in _NUMERIC_KEYS:
        lines.append(f"  {k:<22} {_fmt(overall.get(k))}")
    lines.append("")

    if result_json.get("run_failures"):
        rfs = result_json["run_failures"]
        lines.append(f"Run failures ({len(rfs)}):")
        for rf in rfs:
            summary = _summarize_error(rf.get("detail", ""))
            tail = f" — {summary}" if summary else ""
            lines.append(f"  ⚠ {rf['scenario_id']}: target {rf['status']} at turn {rf['turn']}{tail}")
        hint = _systematic_failure_hint(rfs)
        if hint:
            lines.append(f"  → {hint}")
        lines.append("")

    if result_json.get("judge_failures"):
        jf = result_json["judge_failures"]
        lines.append(f"Judge call failures ({len(jf)}) — dropped those (turn,judge) scores "
                     "(parse or provider error, e.g. rate limit); did not crash:")
        for f in jf[:5]:
            err = f.get("error", "").split(":", 1)[0]  # the error type (RateLimitError / JudgeParseError)
            lines.append(f"  ⚠ {f['scenario_id']} t{f['turn']} judge={f['judge']} ({err})")
        lines.append("")

    gate = result_json["gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    lines.append(f"Gate: {status}" + (f"  ({gate['note']})" if gate.get("note") else ""))
    for f in gate["failures"]:
        if f["kind"] == "run_failure":
            lines.append(f"  ✗ {f['scenario']} RUN FAILURE")
        else:
            lines.append(f"  ✗ {f['scenario']} {f['metric']} {_fmt(f['value'])} < {_fmt(f['threshold'])}")
    lines.append("")

    if result_json.get("judge_disagreement"):
        lines.append("Judge disagreement (inter-judge spread — surfaced, not hidden, §15):")
        for d in result_json["judge_disagreement"][:5]:
            scores = "/".join(_fmt(v) for v in d["scores"])
            lines.append(f"  {d['scenario_id']} t{d['turn']} {d['metric']}: {scores} (spread {_fmt(d['spread'])})")
        lines.append("")

    if result_json["diagnostics"]:
        lines.append("Diagnostics:")
        lines.append("")
        for d in result_json["diagnostics"]:
            lines.append(d["card"])
            lines.append("")

    for w in result_json["warnings"]:
        if not w.startswith(("⚠ DIRECTIONAL", "⚠ SELF-JUDGING")):
            lines.append(f"warning: {w}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTML report — a pytest-html-style browser view of the SAME result.json (§4.3).
# It is a *derived diagnostic view*, a sibling of report.txt — NOT a new eval-log
# viewer (Inspect owns that, §12). Self-contained & offline by construction:
# inline CSS, zero JS (native <details>), no CDN/fonts/images/telemetry — the only
# outbound URL is the Keido Labs footer link. Deterministic on fixed result.json.
# All dynamic strings (target ref, judge reasoning, warnings) are HTML-escaped —
# they are untrusted input (§13.3).
# --------------------------------------------------------------------------

_HTML_CSS = """
:root{
  --bg:#ffffff;--panel:#f6f7f9;--panel2:#eef0f3;--text:#1a1d21;--muted:#6b7280;
  --border:#e5e7eb;--good:#1e7d32;--warn:#b26a00;--bad:#c62828;--na:#9aa0a6;
  --good-bg:#e7f4e8;--warn-bg:#fbf0dd;--bad-bg:#fbe4e4;--accent:#3730a3;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1117;--panel:#171a21;--panel2:#1d2129;--text:#e6e6e6;--muted:#9aa0a6;
  --border:#2a2f3a;--good:#4caf50;--warn:#f9a825;--bad:#ef5350;--na:#9aa0a6;
  --good-bg:#14301a;--warn-bg:#332600;--bad-bg:#3a1717;--accent:#a5b4fc;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px;}
h1{font-size:22px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  margin:36px 0 12px;font-weight:600}
a{color:var(--accent)}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 4px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.callout{border-radius:10px;padding:14px 16px;margin:16px 0;font-size:13.5px;
  border:1px solid var(--border)}
.callout.directional{background:var(--warn-bg);border-color:var(--warn)}
.callout.selfjudge{background:var(--bad-bg);border-color:var(--bad)}
.callout.local{background:var(--panel2)}
.callout .tag{font-weight:700;letter-spacing:.02em}
.verdict{display:flex;align-items:center;gap:16px;border-radius:12px;padding:18px 22px;
  margin:20px 0;border:1px solid var(--border);background:var(--panel)}
.verdict .badge{font-size:15px;font-weight:800;letter-spacing:.04em;padding:8px 14px;
  border-radius:8px;white-space:nowrap}
.verdict.pass .badge{background:var(--good-bg);color:var(--good)}
.verdict.fail .badge{background:var(--bad-bg);color:var(--bad)}
.verdict.incomplete .badge{background:var(--panel2);color:var(--muted)}
.verdict .at{margin-left:auto;text-align:right}
.verdict .at b{font-size:30px;font-weight:800;display:block;line-height:1}
.verdict .at span{font-size:12px;color:var(--muted)}
.at.good{color:var(--good)}.at.warn{color:var(--warn)}.at.bad{color:var(--bad)}.at.na{color:var(--na)}
.bar-row{display:flex;align-items:center;gap:12px;margin:7px 0}
.bar-label{width:150px;color:var(--muted);font-size:13.5px}
.bar-track{flex:1;height:20px;background:var(--panel2);border-radius:5px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px}
.bar-fill.good{background:var(--good)}.bar-fill.warn{background:var(--warn)}
.bar-fill.bad{background:var(--bad)}.bar-fill.na{background:var(--na)}
.bar-val{width:44px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.bar-val.good{color:var(--good)}.bar-val.warn{color:var(--warn)}
.bar-val.bad{color:var(--bad)}.bar-val.na{color:var(--na)}
details.sc{border:1px solid var(--border);border-radius:10px;margin:8px 0;background:var(--panel);
  overflow:hidden}
details.sc.fail{border-color:var(--bad)}
details.sc.rf{border-color:var(--na)}
details.sc>summary{cursor:pointer;list-style:none;padding:12px 16px;display:flex;
  align-items:center;gap:10px;flex-wrap:wrap}
details.sc>summary::-webkit-details-marker{display:none}
.sc .sid{font-weight:700;font-family:ui-monospace,Menlo,monospace}
.sc .st{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;letter-spacing:.03em}
.st.ok{background:var(--good-bg);color:var(--good)}
.st.bad{background:var(--bad-bg);color:var(--bad)}
.st.rf{background:var(--panel2);color:var(--muted)}
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-left:auto}
.chip{font-size:11.5px;padding:2px 7px;border-radius:5px;background:var(--panel2);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.chip.good{color:var(--good)}.chip.warn{color:var(--warn)}
.chip.bad{color:var(--bad);background:var(--bad-bg)}.chip.na{color:var(--na)}
.sc pre{margin:0;padding:14px 16px;background:var(--bg);border-top:1px solid var(--border);
  font-size:12.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
ul.warns{list-style:none;padding:0;margin:0}
ul.warns li{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:9px 13px;margin:6px 0;font-size:13px;color:var(--muted)}
.fails{margin:8px 0 0;padding-left:18px;color:var(--bad);font-size:13.5px}
.fails li{margin:2px 0}
footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--border);
  font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 18px;
  align-items:baseline;justify-content:space-between}
footer .repro{font-family:ui-monospace,Menlo,monospace}
footer .cta{width:100%;color:var(--muted)}
footer a{text-decoration:none}
footer a:hover{text-decoration:underline}
"""

_METRIC_SHORT = {
    "empathy": "empathy",
    "tone_consistency": "tone",
    "crisis_handling": "crisis",
    "advice_safety": "advice",
    "boundary_safety": "boundary",
    "affective_complexity": "affective",
}


def _h(s) -> str:
    return html.escape(str(s), quote=True)


def _band(v) -> str:
    """CSS band class, mirroring card.py's green/amber/red thresholds (§5 bands)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "na"
    if v >= spec.BAND_CEILING_LOW:
        return "good"
    if v >= spec.BAND_MID_LOW:
        return "warn"
    return "bad"


def _bar_row(label: str, v) -> str:
    cls = _band(v)
    pct = (v / 5.0 * 100) if cls != "na" else 0.0
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{_h(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="bar-val {cls}">{_h(_fmt(v))}</div>'
        "</div>"
    )


def _scenario_details(result_json: dict) -> str:
    by_scenario = result_json["scores"]["by_scenario"]
    diag_by_sid = {d["scenario_id"]: d["card"] for d in result_json.get("diagnostics", [])}
    out = []
    for sid in sorted(by_scenario):
        row = by_scenario[sid]
        card = diag_by_sid.get(sid)
        run_fail = row.get("run_failure")
        chips = "".join(
            f'<span class="chip {_band(row.get(m))}" title="{_h(m)}">'
            f"{_h(_METRIC_SHORT.get(m, m))} {_h(_fmt(row.get(m)))}</span>"
            for m in spec.METRICS
        )
        if run_fail:
            cls, st_cls, st_txt = "rf", "rf", "RUN FAILURE"
            open_attr = ""
        elif card:
            cls, st_cls, st_txt = "fail", "bad", "issues"
            open_attr = " open"
        else:
            cls, st_cls, st_txt = "ok", "ok", "ok"
            open_attr = ""
        at = row.get("AI_Trust")
        summary = (
            f'<summary><span class="sid">{_h(sid)}</span>'
            f'<span class="st {st_cls}">{st_txt}</span>'
            f'<span class="chip {_band(at)}">AI-Trust {_h(_fmt(at))}</span>'
            f'<span class="chips">{chips}</span></summary>'
        )
        body = ""
        if run_fail:
            body = ("<pre>target failure (error / timeout / rate-limit / refusal / truncation) — "
                    "a RUN FAILURE, not a low safety score (§6). Re-run this scenario.</pre>")
        elif card:
            body = f"<pre>{_h(card)}</pre>"
        out.append(f'<details class="sc {cls}"{open_attr}>{summary}{body}</details>')
    return "\n".join(out)


def render_html(result_json: dict) -> str:
    """Render the result.json as a self-contained, offline HTML report (§4.3).

    Sibling of ``render_report``: same data, browser-shaped. Zero JS, inline CSS,
    no external assets or telemetry; deterministic on fixed input.
    """
    from .card import reproduce_command

    t = result_json["target"]
    overall = result_json["scores"]["overall"]
    gate = result_json["gate"]
    warnings = result_json.get("warnings", [])
    incomplete = result_json.get("incomplete")

    directional = next((w for w in warnings if w.startswith("⚠ DIRECTIONAL")), None)
    local_judge = next((w for w in warnings if w.startswith("◆ LOCAL JUDGE")), None)
    self_judging = next((w for w in warnings if w.startswith("⚠ SELF-JUDGING")), None)
    other_warnings = [w for w in warnings if w not in (directional, local_judge, self_judging)]

    ai_trust = overall.get("AI_Trust")
    title = f"aipsy-bench · {t['ref']} · AI-Trust {_fmt(ai_trust)}"

    # Verdict banner (pytest-style pass/fail summary line).
    if incomplete:
        v_cls, v_badge, v_note = "incomplete", "RUN INCOMPLETE", "not gated — partial results (§16)"
    elif gate["passed"]:
        v_cls, v_badge = "pass", "✓ GATE PASSED"
        v_note = gate.get("note") or "all gated metrics meet your thresholds"
    else:
        v_cls, v_badge = "fail", "✗ GATE FAILED"
        v_note = gate.get("note") or f"{len(gate['failures'])} threshold breach(es)"

    parts: list[str] = []
    parts.append('<div class="wrap">')
    parts.append("<h1>aipsy-bench · psychological-safety report</h1>")
    parts.append(
        f'<p class="sub">target <b>{_h(t["ref"])}</b> <span class="mono">({_h(t["adapter"])})</span>'
        f' · judges <b>{_h(result_json["judge_panel"])}</b>'
        f' <span class="mono">{_h(_judge_models_str(result_json))}</span>'
        f' · mode {_h(result_json["mode"])}</p>'
    )
    parts.append(
        f'<p class="sub mono">aipsy-bench {_h(result_json["tool_version"])}'
        f' · data {_h(result_json["data_version"])}'
        f' · run {_h(result_json.get("run_id", ""))}'
        f' · {_h(result_json.get("timestamp", ""))}</p>'
    )

    if directional:
        parts.append(f'<div class="callout directional">{_h(directional)}</div>')
    if self_judging:
        parts.append(f'<div class="callout selfjudge">{_h(self_judging)}</div>')
    if local_judge:
        parts.append(f'<div class="callout local">{_h(local_judge)}</div>')

    at_band = _band(ai_trust)
    parts.append(
        f'<div class="verdict {v_cls}"><span class="badge">{_h(v_badge)}</span>'
        f'<span class="sub" style="margin:0">{_h(v_note)}</span>'
        f'<span class="at"><b class="at {at_band}">{_h(_fmt(ai_trust))}</b>'
        f"<span>AI-Trust</span></span></div>"
    )

    # Gate failures list (directional recommendation against the dev's thresholds).
    if not incomplete and gate["failures"]:
        rows = []
        for f in gate["failures"]:
            if f["kind"] == "run_failure":
                rows.append(f'<li>{_h(f["scenario"])} — RUN FAILURE</li>')
            else:
                rows.append(
                    f'<li>{_h(f["scenario"])} · {_h(f["metric"])} '
                    f'{_h(_fmt(f["value"]))} &lt; {_h(_fmt(f["threshold"]))}</li>'
                )
        parts.append(f'<ul class="fails">{"".join(rows)}</ul>')

    parts.append("<h2>Overall scores</h2>")
    parts.append("".join(_bar_row(m, overall.get(m)) for m in spec.METRICS))

    parts.append("<h2>Scenarios</h2>")
    parts.append(_scenario_details(result_json))

    if result_json.get("run_failures"):
        rfs = result_json["run_failures"]
        rows = []
        for rf in rfs:
            summary = _summarize_error(rf.get("detail", ""))
            tail = f" — {_h(summary)}" if summary else ""
            rows.append(
                f'<li>⚠ {_h(rf["scenario_id"])}: target {_h(rf["status"])} '
                f'at turn {_h(rf["turn"])}{tail}</li>'
            )
        hint = _systematic_failure_hint(rfs)
        if hint:
            rows.append(f'<li><b>{_h(hint)}</b></li>')
        parts.append("<h2>Run failures</h2>")
        parts.append(f'<ul class="warns">{"".join(rows)}</ul>')

    if other_warnings:
        parts.append("<h2>Notes</h2>")
        items = "".join(f"<li>{_h(w)}</li>" for w in other_warnings)
        parts.append(f'<ul class="warns">{items}</ul>')

    repro = reproduce_command(result_json)
    parts.append(
        "<footer>"
        '<span>Built by <a href="https://www.keidolabs.com">Keido Labs</a>'
        " · AI Psychology Lab</span>"
        f'<span class="repro">reproduce: {_h(repro)}</span>'
        '<span class="cta">Want help closing these gaps? Keido Labs runs clinician-guided '
        'AI-safety engagements — <a href="https://www.keidolabs.com/contact">keidolabs.com/contact</a>.</span>'
        "</footer>"
    )
    parts.append("</div>")

    body = "\n".join(parts)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_h(title)}</title>"
        f"<style>{_HTML_CSS}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def write_artifacts(log, out_dir: str | Path, *, validation: JudgeValidation, **kwargs) -> dict:
    """Write ``result.json`` + ``report.txt`` + ``report.html`` to ``out_dir``; return the result dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = to_result_json(log, validation=validation, **kwargs)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    (out / "report.txt").write_text(render_report(result))
    (out / "report.html").write_text(render_html(result))
    return result
