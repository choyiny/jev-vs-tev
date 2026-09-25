"""README diagrams, generated from results/ so they stay in sync with the numbers.

    uv run python -m bench.charts            # write docs/diagrams/*.html and render docs/img/*.png

Styled after the diagram-design `default` skin (see .diagram-design). Each diagram is a
self-contained HTML file; PNGs are rendered with headless Chrome because GitHub strips
web fonts from SVG images.
"""

from __future__ import annotations

import math
import subprocess
from html import escape
from pathlib import Path

from bench.dataset import load_items
from bench.report import prepare, prompt_data
from bench.run import ROOT, load_env

DIAGRAMS = ROOT / "docs" / "diagrams"
IMG = ROOT / "docs" / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W = 880  # every diagram shares one width so type renders at the same size in the README

# default skin tokens
PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#7a8399"
ACCENT, ACCENT_TINT = "#eb6c36", "rgba(235,108,54,0.08)"
RULE, GRID, AXIS = "rgba(45,49,66,0.12)", "rgba(45,49,66,0.08)", "rgba(45,49,66,0.25)"
SERIES_1, SERIES_2 = "#7c8f6f", "#5e7a9b"
SANS, MONO, SERIF = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, monospace", "'Instrument Serif', Georgia, serif"
FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1"
         "&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap")
SHORT = {"tev": "TEV", "jev": "JEV", "glm": "GLM 5.3", "opus": "Opus 5.5"}


def text(x, y, s, size=12, fill=INK, family=SANS, weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="{family}" '
            f'font-weight="{weight}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def eyebrow(x, y, s, fill=MUTED, anchor="start"):
    return text(x, y, s.upper(), 12, fill, MONO, 500, anchor, 'letter-spacing="0.14em"')


def header(slug: str, title: str, desc: str, h: int, dy: int) -> list[str]:
    """Open the SVG. No visible heading: the README supplies context, so the drawing starts at the top.
    Content is laid out in the original coordinates and shifted up by `dy`; the SVG is `h` tall."""
    return [
        f'<svg viewBox="0 {dy} {W} {h}" width="{W}" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-labelledby="{slug}-title {slug}-desc">',
        f'<title id="{slug}-title">{escape(title)}</title>',
        f'<desc id="{slug}-desc">{escape(desc)}</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>'
        '<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker></defs>',
        f'<rect y="{dy}" width="100%" height="100%" fill="{PAPER}"/>',
    ]


def page(slug: str, svg: list[str]) -> str:
    body = "\n".join(svg + ["</svg>"])
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{slug}</title>'
            f'<link href="{FONTS}" rel="stylesheet"><style>*{{margin:0;padding:0}}'
            f'body{{background:{PAPER}}}svg{{display:block}}</style></head><body>\n{body}\n</body></html>\n')


# ---------------------------------------------------------------- 1. how it works


def how_it_works() -> tuple[str, int]:
    dy = 80
    h = 400 - dy
    s = header("how-it-works",
               "One small edit, four models, four scores",
               "Flow diagram: two halves of a contrastive pair differ by one word and have different correct "
               "answers; both go through the same prompt to TEV, JEV, GLM 5.3 and Opus 5.5, and each model is "
               "scored on accuracy, pair accuracy, cost per task and speed.", h, dy)
    top, bottom = 136, 364  # shared vertical span of the prompt, model and score columns
    cols = {"pair": (32, 216), "prompt": (280, 152), "models": (484, 168), "score": (704, 144)}
    for key, label in (("pair", "1 · contrastive pair"), ("prompt", "2 · same prompt"),
                       ("models", "3 · four models"), ("score", "4 · scored on")):
        s.append(eyebrow(cols[key][0], 120, label, SOFT))

    # arrows first so boxes paint over their ends
    px, pw = cols["prompt"]
    mx, mw = cols["models"]
    sx, sw = cols["score"]
    halves = [(172, "Half A", "…on flags-staging", "auto_approve"),
              (292, "Half B", "…on flags-prod", "require_human_review")]
    for cy, *_ in halves:
        s.append(f'<line x1="{32 + 216}" y1="{cy}" x2="{px - 2}" y2="{cy}" stroke="{MUTED}" stroke-width="1" marker-end="url(#arrow)"/>')
    model_rows = [(top + i * 60, m) for i, m in enumerate(("tev", "jev", "glm", "opus"))]
    for y, _ in model_rows:
        cy = y + 24
        s.append(f'<line x1="{px + pw}" y1="{cy}" x2="{mx - 2}" y2="{cy}" stroke="{MUTED}" stroke-width="1" marker-end="url(#arrow)"/>')
        s.append(f'<line x1="{mx + mw}" y1="{cy}" x2="{sx - 2}" y2="{cy}" stroke="{MUTED}" stroke-width="1" marker-end="url(#arrow)"/>')

    # 1. the pair
    for cy, name, state, gold in halves:
        y = cy - 40
        s += [f'<rect x="32" y="{y}" width="216" height="80" rx="6" fill="{PAPER}"/>',
              f'<rect x="32" y="{y}" width="216" height="80" rx="6" fill="rgba(79,93,117,0.10)" stroke="{SOFT}" stroke-width="1"/>',
              text(48, y + 24, name, 16, INK, SANS, 600),
              text(48, y + 44, state, 12, MUTED, MONO),
              text(48, y + 64, f"→ {gold}", 12, INK, MONO, 500)]
    s.append(text(140, 232, "one word flips the answer", 16, MUTED, SERIF, 400, "middle", 'font-style="italic"'))

    # 2. the prompt
    s += [f'<rect x="{px}" y="{top}" width="{pw}" height="{bottom - top}" rx="6" fill="#ffffff" stroke="{INK}" stroke-width="1"/>',
          text(px + 16, top + 28, "Same prompt", 16, INK, SANS, 600)]
    for i, line in enumerate(("system prompt", "state", "question", "options A–D")):
        s.append(text(px + 16, top + 60 + i * 24, line, 12, MUTED, MONO))
    s.append(text(px + 16, bottom - 36, "JEV gets it as a", 12, SOFT, SANS))
    s.append(text(px + 16, bottom - 20, "choice question", 12, SOFT, SANS))

    # 3. the models
    subs = {"tev": "$17 fine-tune", "jev": "TypeSafe, closed",
            "glm": "open-weights LLM", "opus": "frontier LLM"}
    for y, m in model_rows:
        frontier = m in ("glm", "opus")
        fill, stroke = ("rgba(45,49,66,0.03)", "rgba(45,49,66,0.30)") if frontier else ("#ffffff", INK)
        s += [f'<rect x="{mx}" y="{y}" width="{mw}" height="48" rx="6" fill="{PAPER}"/>',
              f'<rect x="{mx}" y="{y}" width="{mw}" height="48" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
              text(mx + 16, y + 20, SHORT[m], 16, INK, SANS, 600),
              text(mx + 16, y + 38, subs[m], 12, MUTED, MONO)]

    # 4. the scorecard (focal)
    s += [f'<rect x="{sx}" y="{top}" width="{sw}" height="{bottom - top}" rx="6" fill="{PAPER}"/>',
          f'<rect x="{sx}" y="{top}" width="{sw}" height="{bottom - top}" rx="6" fill="{ACCENT_TINT}" stroke="{ACCENT}" stroke-width="1.2"/>']
    for i, (name, sub) in enumerate((("Accuracy", "of 400 items"), ("Both halves", "right, per pair"),
                                     ("Cost per task", "tokens × price"), ("Speed", "p50 per call"))):
        y = top + 28 + i * 56
        s += [text(sx + 16, y, name, 16, INK, SANS, 600), text(sx + 16, y + 20, sub, 12, MUTED, MONO)]
    return page("how-it-works", s), h


# ---------------------------------------------------------------- 2. cost vs accuracy


def cost_vs_accuracy(summ: dict) -> tuple[str, int]:
    dy = 76
    h = 504 - dy
    per_k = {p: summ[p]["cost_task"] * 1000 for p in summ}  # USD per 1,000 tasks
    frontier = [p for p in ("glm", "opus") if p in summ]
    ratios = [per_k[p] / per_k["jev"] for p in frontier]
    gain = max(summ[p]["acc"] for p in frontier) - summ["jev"]["acc"]
    title = (f"Frontier LLMs add {100 * gain:.0f} points for {min(ratios):.0f}–{max(ratios):.0f}× the price"
             if frontier else "Cost vs accuracy")

    def money(usd):
        return f"{usd * 100:.0f}¢" if usd < 1 else f"${usd:.2f}"

    s = header("cost-vs-accuracy", title,
               "Scatter plot of accuracy against cost per million tasks on a log scale: TEV 90.0% at about $10, "
               "JEV 97.2% at about $20, GLM 5.3 99.0% at about $810, Opus 5.5 99.2% at about $1,708.", h, dy)
    x0, x1, y0, y1 = 112, 824, 416, 136  # plot box
    lo, hi = math.log10(5), math.log10(5000)
    a_lo, a_hi = 85.0, 100.0

    def X(cost_1m):
        return round((x0 + (math.log10(cost_1m) - lo) / (hi - lo) * (x1 - x0)) / 4) * 4

    def Y(acc):
        return round((y0 - (acc - a_lo) / (a_hi - a_lo) * (y0 - y1)) / 4) * 4

    s.append(text(24, 276, "ACCURACY", 12, MUTED, MONO, 500, "middle", 'letter-spacing="0.14em" transform="rotate(-90 24 276)"'))
    s.append(text((x0 + x1) // 2, 484, "COST PER 1M TASKS (LOG SCALE)", 12, MUTED, MONO, 500, "middle", 'letter-spacing="0.14em"'))
    for acc in (85, 90, 95, 100):
        y = Y(acc)
        s.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{GRID}" stroke-width="0.8"/>')
        s.append(text(x0 - 12, y + 4, f"{acc}%", 12, MUTED, MONO, 400, "end"))
    for c in (10, 100, 1000):
        x = X(c)
        s.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y0}" stroke="{GRID}" stroke-width="0.8"/>')
        s.append(text(x, y0 + 24, f"${c:,}", 12, MUTED, MONO, 400, "middle"))
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{AXIS}" stroke-width="1"/>')

    # label offsets tuned so labels clear each other and the points
    place = {"tev": (16, 4, "start"), "jev": (16, 4, "start"), "glm": (-16, 28, "end"), "opus": (0, -32, "middle")}
    for p in ("glm", "opus", "tev", "jev"):  # focal points drawn last
        if p not in summ:
            continue
        cost_1m, acc = summ[p]["cost_task"] * 1e6, summ[p]["acc"] * 100
        x, y = X(cost_1m), Y(acc)
        focal = p in ("tev", "jev")
        r, fill, stroke = (6, "rgba(235,108,54,0.15)", ACCENT) if focal else (5, "rgba(79,93,117,0.20)", MUTED)
        s += [f'<circle cx="{x}" cy="{y}" r="{r}" fill="{PAPER}"/>',
              f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>']
        dx, ly, anchor = place[p]
        s.append(text(x + dx, y + ly, SHORT[p], 16, INK, SANS, 600, anchor))
        s.append(text(x + dx, y + ly + 18, f"{acc:.1f}% · ${cost_1m:,.0f}", 12, MUTED, MONO, 400, anchor))

    cheap = " and ".join(f"{SHORT[p]} {money(per_k[p])}" for p in ("tev", "jev"))
    dear = " and ".join(f"{SHORT[p]} {money(per_k[p])}" for p in frontier)
    s.append(text(468, 268, f"Per 1,000 tasks: {cheap},", 16, MUTED, SERIF, 400, "middle", 'font-style="italic"'))
    if dear:
        s.append(text(468, 290, f"{dear}.", 16, MUTED, SERIF, 400, "middle", 'font-style="italic"'))
    return page("cost-vs-accuracy", s), h


# ---------------------------------------------------------------- 3. speed


def speed(summ: dict) -> tuple[str, int]:
    order = [p for p in ("tev", "jev", "glm", "opus") if p in summ]
    dy = 80
    h = 136 + 48 * len(order) + 48 - dy
    s = header("speed", "TEV answers in under a fifth of a second",
               "Horizontal bar chart of median latency per call: TEV 173 ms, JEV 459 ms, GLM 5.3 2,883 ms, "
               "Opus 5.5 2,477 ms.", h, dy)
    x0, x1, top = 168, 760, 112
    vmax = 3000
    scale = (x1 - x0) / vmax
    bottom = top + 48 * len(order)
    for v in range(0, vmax + 1, 1000):
        x = round(x0 + v * scale)
        s.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{GRID}" stroke-width="0.8"/>')
        s.append(text(x, bottom + 24, f"{v:,} ms", 12, MUTED, MONO, 400, "middle"))
    s.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{AXIS}" stroke-width="1"/>')
    for i, p in enumerate(order):
        v = summ[p]["p50"]
        y = top + 12 + i * 48
        w = max(4, round(min(v, vmax) * scale))
        focal = p == "tev"
        fill, stroke = ("rgba(235,108,54,0.12)", ACCENT) if focal else ("rgba(79,93,117,0.15)", MUTED)
        s += [text(x0 - 16, y + 18, SHORT[p], 16, INK, SANS, 600, "end"),
              f'<rect x="{x0}" y="{y}" width="{w}" height="24" fill="{PAPER}"/>',
              f'<rect x="{x0}" y="{y}" width="{w}" height="24" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
              text(x0 + w + 12, y + 17, f"{v:,.0f} ms", 12, ACCENT if focal else MUTED, MONO, 500)]
    s.append(text(x0, h + dy - 16, "Median wall-clock time per call, as seen by the caller. Lower is better.", 12, SOFT, SANS))
    return page("speed", s), h


# ---------------------------------------------------------------- 4. prompts matter


def prompts_matter(pdata: dict) -> tuple[str, int]:
    variants = [("careful", "+ “read carefully” line"), ("reversed", "options in reverse order"),
                ("generic_question", "generic question"), ("keys_only", "no option descriptions")]
    variants = [(v, d) for v, d in variants if ("tev", v) in pdata and ("jev", v) in pdata]
    row = 64
    top = 144
    dy = 88
    h = top + row * len(variants) + 116 - dy
    s = header("prompts-matter",
               "Rewording barely matters. Option descriptions do.",
               "Diverging bar chart of accuracy change from each model's default prompt, for TEV and JEV under four "
               "prompt changes. Removing option descriptions costs TEV 6.8 points and JEV 4.5; other changes move "
               "accuracy by under a point.", h, dy)
    x_lo, x_hi = -8.0, 2.0
    px0, px1 = 312, 712
    scale = (px1 - px0) / (x_hi - x_lo)

    def X(v):
        return round(px0 + (v - x_lo) * scale)

    bottom = top + row * len(variants)
    # highlight the row with the biggest drop (editorial accent)
    worst = min(variants, key=lambda vd: pdata[("tev", vd[0])]["delta"] + pdata[("jev", vd[0])]["delta"])[0]
    for i, (v, _) in enumerate(variants):
        if v == worst:
            s.append(f'<rect x="24" y="{top + i * row + 4}" width="{W - 48}" height="{row - 8}" rx="6" '
                     f'fill="{ACCENT_TINT}" stroke="rgba(235,108,54,0.50)" stroke-width="0.8" stroke-dasharray="4,4"/>')
    for pts in (-8, -6, -4, -2, 0, 2):
        x = X(pts)
        s.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{AXIS if pts == 0 else GRID}" '
                 f'stroke-width="{1 if pts == 0 else 0.8}"/>')
        s.append(text(x, bottom + 24, f"{pts:+d}" if pts else "0", 12, MUTED, MONO, 400, "middle"))
    s.append(text((px0 + px1) // 2, bottom + 48, "ACCURACY CHANGE VS DEFAULT PROMPT (POINTS)", 12, MUTED, MONO, 500,
                  "middle", 'letter-spacing="0.14em"'))
    s.append(eyebrow(W - 32, top - 16, "answers changed", SOFT, "end"))

    for i, (v, desc) in enumerate(variants):
        y = top + i * row
        s.append(text(40, y + 28, desc, 16, INK, SANS, 600))
        s.append(text(40, y + 48, v, 12, MUTED, MONO))
        changed = []
        for j, (m, color) in enumerate((("tev", SERIES_2), ("jev", SERIES_1))):
            d = pdata[(m, v)]["delta"] * 100
            by = y + 14 + j * 20
            xa, xb = sorted((X(0), X(d)))
            w = max(2, xb - xa)
            s += [f'<rect x="{xa}" y="{by}" width="{w}" height="16" fill="{PAPER}"/>',
                  f'<rect x="{xa}" y="{by}" width="{w}" height="16" fill="{color}" fill-opacity="0.30" stroke="{color}" stroke-width="1"/>']
            lx, anchor = (xa - 8, "end") if d < 0 else (xb + 8, "start")
            s.append(text(lx, by + 12, f"{SHORT[m]} {d:+.1f}", 12, INK, MONO, 500, anchor))
            changed.append(f"{pdata[(m, v)]['changed'] * 100:.0f}%")
        s.append(text(W - 40, y + 36, " / ".join(changed), 16, ACCENT if v == worst else MUTED, MONO, 500, "end"))

    # legend strip
    ly = h + dy - 20
    s.append(f'<line x1="32" y1="{ly - 20}" x2="{W - 32}" y2="{ly - 20}" stroke="{RULE}" stroke-width="0.8"/>')
    s.append(eyebrow(32, ly, "legend"))
    for k, (m, color) in enumerate((("tev", SERIES_2), ("jev", SERIES_1))):
        x = 136 + k * 128
        s += [f'<rect x="{x}" y="{ly - 11}" width="16" height="12" fill="{color}" fill-opacity="0.30" stroke="{color}" stroke-width="1"/>',
              text(x + 24, ly, SHORT[m], 12, MUTED, SANS, 500)]
    s.append(text(W - 32, ly, "answers changed = TEV / JEV answers that differ from the default prompt", 12, SOFT, SANS, 400, "end"))
    return page("prompts-matter", s), h


# ---------------------------------------------------------------- render


def render_png(html: Path, png: Path, height: int) -> None:
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
         f"--window-size={W},{height}", "--virtual-time-budget=8000", f"--screenshot={png}", html.as_uri()],
        check=True, capture_output=True,
    )


def main() -> None:
    load_env()
    items = load_items()
    providers = [p for p in ("tev", "jev", "glm", "opus") if (ROOT / "results" / f"{p}.jsonl").exists()]
    by_id, common, _, summ = prepare(providers, items)
    pdata = prompt_data(by_id, set(common))
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    for name, (html, h) in {
        "how-it-works": how_it_works(),
        "cost-vs-accuracy": cost_vs_accuracy(summ),
        "speed": speed(summ),
        "prompts-matter": prompts_matter(pdata),
    }.items():
        src = DIAGRAMS / f"{name}.html"
        src.write_text(html)
        render_png(src, IMG / f"{name}.png", h)
        print(f"docs/img/{name}.png")


if __name__ == "__main__":
    main()
