"""Score results/*.jsonl against the dataset and write the results section of README.md.

    uv run python -m bench.report                 # tev vs jev, update README
    uv run python -m bench.report --stdout        # print instead of writing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bench.dataset import Item, load_items
from bench.run import RESULTS_DIR, ROOT, load_env

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LABELS = {"tev": "TEV (Together)", "jev": "JEV (AI Space)", "oracle": "oracle"}


@dataclass
class Row:
    item: Item
    key: str | None
    probs: dict[str, float] | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def correct(self) -> bool:
        return self.key == self.item.gold


def load_rows(provider: str, items: dict[str, Item]) -> dict[str, Row]:
    path = RESULTS_DIR / f"{provider}.jsonl"
    rows: dict[str, Row] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["item_id"] in items:
            rows[r["item_id"]] = Row(
                items[r["item_id"]], r["key"], r.get("probs"), r["latency_ms"],
                r["input_tokens"], r["output_tokens"], r["model"],
            )
    return rows


# ---------------------------------------------------------------- statistics


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def pair_bootstrap_ci(rows: list[Row], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI for accuracy, resampling whole pairs since the two halves aren't independent."""
    by_pair: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_pair[r.item.pair_id].append(r.correct)
    groups = list(by_pair.values())
    rng = random.Random(seed)
    accs = []
    for _ in range(n):
        sample = [rng.choice(groups) for _ in groups]
        accs.append(sum(sum(g) for g in sample) / sum(len(g) for g in sample))
    return pct(accs, 0.025), pct(accs, 0.975)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def ece(rows: list[Row], bins: int = 10) -> float | None:
    """Expected calibration error of the top-choice probability."""
    pts = [(max(r.probs.values()), r.correct) for r in rows if r.probs and r.key]
    if len(pts) < len(rows) * 0.9:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(p, c) for p, c in pts if lo < p <= hi or (b == 0 and p == 0)]
        if bucket:
            conf = statistics.fmean(p for p, _ in bucket)
            acc = statistics.fmean(c for _, c in bucket)
            total += len(bucket) / len(pts) * abs(conf - acc)
    return total


def brier(rows: list[Row]) -> float | None:
    scored = [r for r in rows if r.probs]
    if len(scored) < len(rows) * 0.9:
        return None
    return statistics.fmean(
        sum((r.probs.get(k, 0.0) - (k == r.item.gold)) ** 2 for k in r.item.keys) for r in scored
    )


def price(provider: str) -> tuple[float, float] | None:
    p_in = os.environ.get(f"{provider.upper()}_PRICE_IN")
    p_out = os.environ.get(f"{provider.upper()}_PRICE_OUT") or "0"
    if not p_in:
        return None
    return float(p_in), float(p_out)


# ---------------------------------------------------------------- summary


def summarize(provider: str, rows: list[Row]) -> dict:
    n = len(rows)
    by_pair: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_pair[r.item.pair_id].append(r.correct)
    full_pairs = [v for v in by_pair.values() if len(v) == 2]
    lat = [r.latency_ms for r in rows if r.key]
    avg_in = statistics.fmean(r.input_tokens for r in rows)
    avg_out = statistics.fmean(r.output_tokens for r in rows)
    pr = price(provider)
    cost_1k = (avg_in * pr[0] + avg_out * pr[1]) / 1e6 * 1000 if pr else None
    return {
        "provider": provider,
        "model": statistics.mode(r.model for r in rows),
        "n": n,
        "acc": sum(r.correct for r in rows) / n,
        "ci": pair_bootstrap_ci(rows),
        "pair_acc": sum(all(v) for v in full_pairs) / len(full_pairs) if full_pairs else float("nan"),
        "unusable": sum(r.key is None for r in rows) / n,
        "p50": pct(lat, 0.5),
        "p95": pct(lat, 0.95),
        "avg_in": avg_in,
        "avg_out": avg_out,
        "cost_1k": cost_1k,
        "ece": ece(rows),
        "brier": brier(rows),
    }


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def fmt_num(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def render(providers: list[str], items: list[Item]) -> str:
    by_id = {it.id: it for it in items}
    all_rows = {p: load_rows(p, by_id) for p in providers}
    # Score only items every provider answered (or failed on) so the comparison is like for like.
    common = sorted(set.intersection(*(set(r) for r in all_rows.values())))
    rows = {p: [all_rows[p][i] for i in common] for p in providers}
    summ = {p: summarize(p, rows[p]) for p in providers}
    cats = sorted({by_id[i].category for i in common})
    n_pairs = len({by_id[i].pair_id for i in common})

    out: list[str] = []
    out.append(f"_Run on {date.today().isoformat()} · {len(common)} items / {n_pairs} contrastive pairs · "
               f"{len(cats)} task families._\n")

    # Headline table
    head = "| Metric | " + " | ".join(LABELS.get(p, p) for p in providers) + " |"
    out += [head, "|---|" + "---:|" * len(providers)]

    def line(name: str, f) -> None:
        out.append(f"| {name} | " + " | ".join(f(summ[p]) for p in providers) + " |")

    line("Model", lambda s: f"`{s['model']}`")
    line("Accuracy (95% CI)", lambda s: f"{fmt_pct(s['acc'])} ({fmt_pct(s['ci'][0])}–{fmt_pct(s['ci'][1])})")
    line("Pair accuracy (both halves right)", lambda s: fmt_pct(s["pair_acc"]))
    line("Unusable output", lambda s: fmt_pct(s["unusable"]))
    line("Latency p50", lambda s: f"{fmt_num(s['p50'], '.0f')} ms")
    line("Latency p95", lambda s: f"{fmt_num(s['p95'], '.0f')} ms")
    line("Avg tokens in / out", lambda s: f"{s['avg_in']:.0f} / {s['avg_out']:.1f}")
    line("Cost per 1k decisions", lambda s: "n/a" if s["cost_1k"] is None else f"${s['cost_1k']:.4f}")
    line("Calibration ECE ↓", lambda s: fmt_num(s["ece"]))
    line("Brier score ↓", lambda s: fmt_num(s["brier"]))
    out.append("")

    # Head-to-head significance for the first two providers
    if len(providers) >= 2:
        a, b = providers[:2]
        ra, rb = rows[a], rows[b]
        a_only = sum(x.correct and not y.correct for x, y in zip(ra, rb))
        b_only = sum(y.correct and not x.correct for x, y in zip(ra, rb))
        both = sum(x.correct and y.correct for x, y in zip(ra, rb))
        neither = len(common) - a_only - b_only - both
        p = mcnemar_exact(a_only, b_only)
        out += [
            "**Head to head.** "
            f"Both right on {both}, both wrong on {neither}. "
            f"{LABELS.get(a, a)} alone right on {a_only}; {LABELS.get(b, b)} alone right on {b_only}. "
            f"Exact McNemar p = {p:.3g}"
            + (" (significant at 0.05)." if p < 0.05 else " (not significant at 0.05)."),
            "",
        ]

    # Per-category
    out += ["**Accuracy by task family**", ""]
    out.append("| Task family | n | Majority baseline | " + " | ".join(LABELS.get(p, p) for p in providers) + " |")
    out.append("|---|---:|---:|" + "---:|" * len(providers))
    for cat in cats:
        golds = [by_id[i].gold for i in common if by_id[i].category == cat]
        majority = max(golds.count(g) for g in set(golds)) / len(golds)
        accs = []
        for p in providers:
            rs = [r for r in rows[p] if r.item.category == cat]
            accs.append(sum(r.correct for r in rs) / len(rs))
        best = max(accs)
        cells = [f"**{fmt_pct(x)}**" if x == best and accs.count(best) == 1 else fmt_pct(x) for x in accs]
        n = sum(1 for i in common if by_id[i].category == cat)
        out.append(f"| `{cat}` | {n} | {fmt_pct(majority)} | " + " | ".join(cells) + " |")
    out.append("")

    # Difficulty
    out += ["**Accuracy by difficulty**", ""]
    out.append("| Difficulty | n | " + " | ".join(LABELS.get(p, p) for p in providers) + " |")
    out.append("|---|---:|" + "---:|" * len(providers))
    for d in ("easy", "hard"):
        n = sum(1 for i in common if by_id[i].difficulty == d)
        if n:
            cells = [fmt_pct(sum(r.correct for r in rows[p] if r.item.difficulty == d) / n) for p in providers]
            out.append(f"| {d} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def write_readme(section: str, readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"README is missing {START} / {END} markers")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    readme.write_text(f"{pre}{START}\n{section}\n{END}{post}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", nargs="+", default=["tev", "jev"])
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    load_env()
    section = render(args.provider, load_items())
    if args.stdout:
        print(section)
    else:
        write_readme(section)
        print("README.md results section updated")


if __name__ == "__main__":
    main()
