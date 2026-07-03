"""Share card + badge + head-to-head — rendered in Python (locked decision), so
every run emits a shareable artifact with no external dependency (§4.5).

SVG is the primary artifact (full text card). The PNG is a deterministic,
dependency-free raster of the six metric dials (a tiny stdlib PNG encoder — no
native libs, so CI stays offline). A future Next.js site consumes these + board.json.

Until the 014 study lands the card shows the directional banner — a recommendation,
never a validated stamp or an agreement number (§0.3).
"""

from __future__ import annotations

import struct
import zlib
from xml.sax.saxutils import escape

from . import spec
from .validation import VALIDATED

_W, _H = 1200, 630  # OpenGraph card size
_BG = "#0f1117"
_FG = "#e6e6e6"
_MUTED = "#9aa0a6"


def _band_color(v: float | None) -> str:
    if v is None:
        return _MUTED
    if v >= spec.BAND_CEILING_LOW:
        return "#2e7d32"  # green
    if v >= spec.BAND_MID_LOW:
        return "#f9a825"  # amber
    return "#c62828"      # red


def _fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "N/A"


def _esc(s: str) -> str:
    return escape(str(s))


def reproduce_command(result_json: dict) -> str:
    panel = result_json.get("judge_panel", "gold")
    # panel_base so a provider-selected single lane (e.g. single:anthropic) still emits
    # its --judges flag and reproduces exactly.
    suffix = f" --judges {panel}" if spec.panel_base(panel) in ("gold", "local", "single") else ""
    return f"aipsy-bench run --model {result_json['target']['ref']}{suffix}"


def _judge_stamp(result_json: dict, directional: bool) -> str:
    """The bottom-bar provenance stamp — panel-aware so a local card never reads as
    the frontier-gold or validated instrument."""
    if result_json.get("judge_panel") == "local":
        return "◆ LOCAL JUDGE · gemma4-judge-ft-v3 (offline) — directional, flag-for-review"
    if directional:
        return "⚠ DIRECTIONAL — recommendation, not yet human-validated (014 study in parallel)"
    return "validated against clinical experts · OSF DOI"


def brutal_line(result_json: dict) -> str:
    """The single most brutal diagnostic line (§4.5) — lowest scenario metric."""
    worst = None  # (value, scenario, metric)
    for sid, vals in result_json["scores"]["by_scenario"].items():
        for m in spec.METRICS:
            v = vals.get(m)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if worst is None or v < worst[0]:
                    worst = (v, sid, m)
    if worst is None:
        return "no scored metrics"
    from .diagnostics import _PATTERN
    return f"{worst[2]} {worst[0]:.1f} @ {worst[1]} — {_PATTERN.get(worst[2], 'below threshold')}"


def og_meta_tags(result_json: dict) -> str:
    """OpenGraph / Twitter-card meta tags so links unfurl into rich previews (§4.5)."""
    at = _fmt(result_json["scores"]["overall"].get("AI_Trust"))
    title = f"aipsy-bench · {result_json['target']['ref']} · AI-Trust {at}"
    desc = "Psychological-safety benchmark for conversational AI — directional (recommendation; not yet human-validated)."
    return "\n".join([
        f'<meta property="og:title" content="{_esc(title)}" />',
        f'<meta property="og:description" content="{_esc(desc)}" />',
        '<meta name="twitter:card" content="summary_large_image" />',
        f'<meta name="twitter:title" content="{_esc(title)}" />',
    ])


def render_card(result_json: dict) -> tuple[str, bytes]:
    """Render the share card → (svg, png). Deterministic on fixed input."""
    overall = result_json["scores"]["overall"]
    ai_trust = overall.get("AI_Trust")
    ref = result_json["target"]["ref"]
    directional = result_json["judge_validation"]["status"] != VALIDATED

    rows = []
    y = 250
    for m in spec.METRICS:
        v = overall.get(m)
        bar_w = int((v / 5.0) * 560) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
        rows.append(
            f'<text x="60" y="{y + 18}" fill="{_MUTED}" font-size="22">{_esc(m)}</text>'
            f'<rect x="400" y="{y}" width="560" height="26" rx="4" fill="#1d2129"/>'
            f'<rect x="400" y="{y}" width="{bar_w}" height="26" rx="4" fill="{_band_color(v)}"/>'
            f'<text x="980" y="{y + 20}" fill="{_FG}" font-size="22">{_fmt(v)}</text>'
        )
        y += 46

    stamp = _judge_stamp(result_json, directional)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" viewBox="0 0 {_W} {_H}" font-family="system-ui, sans-serif">
  <metadata>
{_esc("OpenGraph/Twitter meta (for embedding HTML):")}
og:title aipsy-bench · {_esc(ref)} · AI-Trust {_fmt(ai_trust)}
twitter:card summary_large_image
  </metadata>
  <rect width="{_W}" height="{_H}" fill="{_BG}"/>
  <text x="60" y="80" fill="{_FG}" font-size="40" font-weight="bold">aipsy-bench · psychological safety</text>
  <text x="60" y="130" fill="{_MUTED}" font-size="28">{_esc(ref)}</text>
  <text x="60" y="210" fill="{_FG}" font-size="30">AI-Trust</text>
  <text x="400" y="215" fill="{_band_color(ai_trust)}" font-size="64" font-weight="bold">{_fmt(ai_trust)}</text>
  {''.join(rows)}
  <text x="60" y="{y + 40}" fill="#ff8a80" font-size="22">{_esc(brutal_line(result_json))}</text>
  <rect x="0" y="{_H - 110}" width="{_W}" height="50" fill="#3a2a00"/>
  <text x="60" y="{_H - 77}" fill="#ffcc80" font-size="22">{_esc(stamp)}</text>
  <text x="60" y="{_H - 25}" fill="{_MUTED}" font-size="22" font-family="monospace">{_esc(reproduce_command(result_json))}</text>
</svg>"""

    png = _png_dials([overall.get(m) for m in spec.METRICS])
    return svg, png


def render_badge(result_json: dict) -> str:
    """shields-style badge SVG: ``psych-safety | AI-Trust X.X``.

    Binds to ``run_id`` for now; once attest.py lands (iteration 3) the badge
    resolves over ``run_hash`` so it cannot be faked by editing a number.
    """
    at = _fmt(result_json["scores"]["overall"].get("AI_Trust"))
    color = _band_color(result_json["scores"]["overall"].get("AI_Trust"))
    local = result_json.get("judge_panel") == "local"
    label, value = ("psych-safety·local" if local else "psych-safety"), f"AI-Trust {at}"
    lw, vw = (120 if local else 90), 110
    run_id = result_json.get("run_id", "")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="20" role="img" aria-label="{label}: {value}">
  <title>aipsy-bench badge (run {_esc(run_id)})</title>
  <rect width="{lw}" height="20" fill="#555"/>
  <rect x="{lw}" width="{vw}" height="20" fill="{color}"/>
  <g fill="#fff" font-family="system-ui, sans-serif" font-size="11">
    <text x="8" y="14">{label}</text>
    <text x="{lw + 8}" y="14">{_esc(value)}</text>
  </g>
</svg>"""


def render_head_to_head(result_a: dict, result_b: dict) -> tuple[str, bytes]:
    """Side-by-side of two runs (the launch artifact). Delta math is Step 14;
    this renders a comparison given two result.jsons."""
    def col(res, x):
        o = res["scores"]["overall"]
        lines = [
            f'<text x="{x}" y="120" fill="{_FG}" font-size="28" font-weight="bold">{_esc(res["target"]["ref"])}</text>',
            f'<text x="{x}" y="170" fill="{_band_color(o.get("AI_Trust"))}" font-size="44" font-weight="bold">AI-Trust {_fmt(o.get("AI_Trust"))}</text>',
        ]
        yy = 230
        for m in spec.METRICS:
            lines.append(f'<text x="{x}" y="{yy}" fill="{_MUTED}" font-size="22">{_esc(m)}: <tspan fill="{_band_color(o.get(m))}">{_fmt(o.get(m))}</tspan></text>')
            yy += 40
        return "".join(lines)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" viewBox="0 0 {_W} {_H}" font-family="system-ui, sans-serif">
  <rect width="{_W}" height="{_H}" fill="{_BG}"/>
  <text x="60" y="70" fill="{_FG}" font-size="34" font-weight="bold">aipsy-bench · head-to-head</text>
  <line x1="600" y1="100" x2="600" y2="{_H - 60}" stroke="#2a2f3a"/>
  {col(result_a, 60)}
  {col(result_b, 660)}
</svg>"""
    png = _png_dials([result_a["scores"]["overall"].get(m) for m in spec.METRICS])
    return svg, png


# --------------------------------------------------------------------------
# Tiny stdlib PNG encoder — a deterministic RGB raster of the metric dials.
# Color type 2 (truecolor), 8-bit. No native deps → CI-offline safe.
# --------------------------------------------------------------------------
def _png_dials(values: list, width: int = 600, height: int = 315) -> bytes:
    rgb = {"bg": (15, 17, 23), "track": (29, 33, 41),
           "green": (46, 125, 50), "amber": (249, 168, 37), "red": (198, 40, 40),
           "muted": (154, 160, 166)}

    def color_for(v):
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return rgb["muted"]
        if v >= spec.BAND_CEILING_LOW:
            return rgb["green"]
        if v >= spec.BAND_MID_LOW:
            return rgb["amber"]
        return rgb["red"]

    canvas = [bytearray(rgb["bg"] * width) for _ in range(height)]

    n = len(values)
    bar_h, gap = 26, 18
    x0, track_w = 40, width - 80
    top = (height - (n * bar_h + (n - 1) * gap)) // 2
    for i, v in enumerate(values):
        y = top + i * (bar_h + gap)
        fill_w = int((v / 5.0) * track_w) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
        c = color_for(v)
        for row in range(y, y + bar_h):
            line = canvas[row]
            for x in range(x0, x0 + track_w):
                px = (c if x < x0 + fill_w else rgb["track"])
                off = x * 3
                line[off:off + 3] = bytes(px)
    return _encode_png(width, height, canvas)


def _encode_png(width: int, height: int, rows: list[bytearray]) -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + bytes(r) for r in rows)              # filter byte per row
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
