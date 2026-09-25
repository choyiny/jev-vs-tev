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

from bench import spend
from bench.dataset import Item, load_items
from bench.run import RESULTS_DIR, ROOT, load_env

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
ROUTE_START, ROUTE_END = "<!-- ROUTING:START -->", "<!-- ROUTING:END -->"
STRONG = "glm"  # the model risky answers are re-asked to
LABELS = {
    "tev": "TEV (Together)",
    "jev": "JEV (AI Space)",
    "glm": "GLM 5.3 (AI Space)",
    "opus": "Claude Opus 5.5 (AI Space)",
    "oracle": "oracle",
}
DEFAULT_PROVIDERS = ["tev", "jev", "glm", "opus"]
VARIANT_LABELS = {
    "default": "`default`: vendor-recommended prompt",
    "careful": "`careful`: + one line of reading guidance",
    "reversed": "`reversed`: same options, reverse order",
    "generic_question": "`generic_question`: \"Which option best fits the input?\"",
    "keys_only": "`keys_only`: option keys, no descriptions",
}


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


def label_metrics(rows: list[Row]) -> dict[tuple[str, str], dict]:
    """One-vs-rest precision, recall and F1 for every answer label, keyed by (task family, label).

    Labels are scoped to their family, since the same key can mean different things in different families.
    A label is included if it is the gold answer or the prediction for at least one item. Unusable output
    counts as a miss (a false negative for the gold label, no false positive). Undefined precision or recall
    (no predictions, or no gold items) is 0, as in scikit-learn's default.
    """
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for r in rows:
        cat, gold = r.item.category, r.item.gold
        if r.key == gold:
            counts[(cat, gold)][0] += 1
        else:
            counts[(cat, gold)][2] += 1
            if r.key is not None:
                counts[(cat, r.key)][1] += 1
    out = {}
    for label, (tp, fp, fn) in counts.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[label] = {"support": tp + fn, "predicted": tp + fp, "precision": p, "recall": rec,
                      "f1": 2 * p * rec / (p + rec) if p + rec else 0.0}
    return out


def macro(metrics: dict[tuple[str, str], dict], category: str | None = None) -> dict[str, float]:
    """Unweighted mean of per-label precision, recall and F1, so rare labels count as much as common ones.

    Without a category, it is the mean of the per-family macros, so each family counts equally. A plain mean
    over all labels would be dominated by support_intent, which has 40 labels with 1–3 items each.
    """
    keys = ("precision", "recall", "f1")
    if category is None:
        fams = [macro(metrics, c) for c in sorted({c for c, _ in metrics})]
        return {k: statistics.fmean(f[k] for f in fams) for k in keys}
    ms = [m for (cat, _), m in metrics.items() if cat == category]
    return {k: statistics.fmean(m[k] for m in ms) for k in keys}


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
    pr = spend.price(provider)
    # Cost per task: each call's billed tokens at list price, averaged over tasks.
    cost_task = (
        statistics.fmean((r.input_tokens * pr[0] + r.output_tokens * pr[1]) / 1e6 for r in rows) if pr else None
    )
    per_label = label_metrics(rows)
    mac = macro(per_label)
    return {
        "provider": provider,
        "model": statistics.mode(r.model for r in rows),
        "n": n,
        "acc": sum(r.correct for r in rows) / n,
        "ci": pair_bootstrap_ci(rows),
        "pair_acc": sum(all(v) for v in full_pairs) / len(full_pairs) if full_pairs else float("nan"),
        "labels": per_label,
        "macro_p": mac["precision"],
        "macro_r": mac["recall"],
        "macro_f1": mac["f1"],
        "unusable": sum(r.key is None for r in rows) / n,
        "p50": pct(lat, 0.5),
        "p95": pct(lat, 0.95),
        "avg_in": avg_in,
        "avg_out": avg_out,
        "cost_task": cost_task,
        "ece": ece(rows),
        "brier": brier(rows),
    }


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def fmt_num(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def verdict(a: dict, b: dict) -> str:
    """One-sentence summary comparing the first two providers on cost, speed and accuracy."""
    parts = []
    hi, lo = (a, b) if a["acc"] >= b["acc"] else (b, a)
    parts.append(f"**{LABELS.get(hi['provider'])}** is more accurate by {100 * (hi['acc'] - lo['acc']):.1f} points")
    fast, slow = (a, b) if a["p50"] <= b["p50"] else (b, a)
    parts.append(f"**{LABELS.get(fast['provider'])}** is {slow['p50'] / fast['p50']:.1f}× faster at p50")
    if a["cost_task"] and b["cost_task"]:
        cheap, dear = (a, b) if a["cost_task"] <= b["cost_task"] else (b, a)
        parts.append(f"**{LABELS.get(cheap['provider'])}** is {dear['cost_task'] / cheap['cost_task']:.1f}× cheaper per task")
    return "; ".join(parts) + "."


def prepare(providers: list[str], items: list[Item]):
    by_id = {it.id: it for it in items}
    all_rows = {p: load_rows(p, by_id) for p in providers}
    # Score only items every provider answered (or failed on) so the comparison is like for like.
    common = sorted(set.intersection(*(set(r) for r in all_rows.values())))
    rows = {p: [all_rows[p][i] for i in common] for p in providers}
    summ = {p: summarize(p, rows[p]) for p in providers}
    return by_id, common, rows, summ


def render_summary(providers: list[str], items: list[Item]) -> str:
    """The short results block for README.md; everything else goes to RESULTS.md."""
    by_id, common, rows, summ = prepare(providers, items)
    out = [verdict(summ[providers[0]], summ[providers[1]]), ""] if len(providers) >= 2 else []
    out.append("| | " + " | ".join(LABELS.get(p, p) for p in providers) + " |")
    out.append("|---|" + "---:|" * len(providers))
    rows_spec = [
        ("Cost per task", lambda s: s["cost_task"], lambda v: f"${v:.7f}", True),
        ("Speed (p50)", lambda s: s["p50"], lambda v: f"{v:,.0f} ms", True),
        ("Accuracy", lambda s: s["acc"], fmt_pct, False),
        ("Both halves of a pair right", lambda s: s["pair_acc"], fmt_pct, False),
        ("Macro F1 (every answer weighted equally)", lambda s: s["macro_f1"], fmt_pct, False),
    ]
    for name, metric, fmt, low in rows_spec:
        vals = [metric(summ[p]) for p in providers]
        known = [v for v in vals if v is not None]
        win = (min if low else max)(known)
        out.append(f"| {name} | " + " | ".join(
            "n/a" if v is None else (f"**{fmt(v)}**" if v == win and known.count(win) == 1 else fmt(v)) for v in vals
        ) + " |")
    out.append("")
    n_pairs = len({by_id[i].pair_id for i in common})
    t = spend.totals()
    cost = f" The whole experiment cost **${sum(r['usd'] for r in t.values()):.2f}** in API calls." if t else ""
    out.append(f"{len(common)} held-out items ({n_pairs} contrastive pairs, 8 task families).{cost} "
               "Per-label precision and recall, per-family scores, prompt variants, calibration and significance: "
               "[RESULTS.md](RESULTS.md).")
    return "\n".join(out)


def render(providers: list[str], items: list[Item]) -> str:
    by_id, common, rows, summ = prepare(providers, items)
    cats = sorted({by_id[i].category for i in common})
    n_pairs = len({by_id[i].pair_id for i in common})

    out: list[str] = ["# Results", "", "Full numbers for the [JEV vs TEV benchmark](README.md). "
                      "How they're measured: [METHODOLOGY.md](METHODOLOGY.md).", ""]
    out.append(f"_Run on {date.today().isoformat()} · {len(common)} items / {n_pairs} contrastive pairs · "
               f"{len(cats)} task families._\n")

    def line(name: str, f) -> None:
        out.append(f"| {name} | " + " | ".join(f(summ[p]) for p in providers) + " |")

    def header(title: str) -> None:
        out.append(f"| {title} | " + " | ".join(LABELS.get(p, p) for p in providers) + " |")
        out.append("|---|" + "---:|" * len(providers))

    if len(providers) >= 2:
        out += [verdict(summ[providers[0]], summ[providers[1]]), ""]
    for p in providers[2:]:
        s_ = summ[p]
        cost = "" if s_["cost_task"] is None else f", ${s_['cost_task']:.7f} per task"
        out.append(f"For reference, {LABELS.get(p, p)} scores {fmt_pct(s_['acc'])} at {s_['p50']:.0f} ms p50{cost}.")
    if len(providers) > 2:
        out.append("")

    def best(name: str, metric, fmt, lower_is_better: bool) -> None:
        """A row where the winning value is bolded."""
        vals = {p: metric(summ[p]) for p in providers}
        known = [v for v in vals.values() if v is not None]
        win = (min if lower_is_better else max)(known) if len(known) > 1 else None
        cells = ["n/a" if v is None else (f"**{fmt(v)}**" if v == win and known.count(win) == 1 else fmt(v))
                 for v in vals.values()]
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    # Headline: cost, speed, accuracy
    header("Cost · Speed · Accuracy")
    best("Cost per task", lambda s: s["cost_task"], lambda v: f"${v:.7f}", True)
    best("Cost per 1M tasks", lambda s: s["cost_task"] and s["cost_task"] * 1e6, lambda v: f"${v:,.2f}", True)
    best("Speed: latency p50", lambda s: s["p50"], lambda v: f"{v:.0f} ms", True)
    best("Speed: latency p95", lambda s: s["p95"], lambda v: f"{v:.0f} ms", True)
    best("Accuracy", lambda s: s["acc"], fmt_pct, False)
    line("Accuracy 95% CI", lambda s: f"{fmt_pct(s['ci'][0])}–{fmt_pct(s['ci'][1])}")
    best("Pair accuracy (both halves right)", lambda s: s["pair_acc"], fmt_pct, False)
    best("Macro precision", lambda s: s["macro_p"], fmt_pct, False)
    best("Macro recall", lambda s: s["macro_r"], fmt_pct, False)
    best("Macro F1", lambda s: s["macro_f1"], fmt_pct, False)
    out.append("")

    out += ["**Details**", ""]
    header("Metric")
    line("Model version served", lambda s: f"`{s['model']}`")
    line("Billed tokens per task, in / out", lambda s: f"{s['avg_in']:.0f} / {s['avg_out']:.1f}")
    line("Unusable output", lambda s: fmt_pct(s["unusable"]))
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
    out += per_label_section(providers, summ, cats)
    out += routing_section(providers, rows, summ)
    out += prompt_sensitivity(by_id, set(common))
    out += spend_section()
    return "\n".join(out)


def per_label_section(providers: list[str], summ: dict, cats: list[str]) -> list[str]:
    """Macro precision / recall / F1 per family, then every label's precision and recall."""
    names = " | ".join(LABELS.get(p, p) for p in providers)
    out = ["**Precision and recall by answer label**", "",
           "Accuracy counts items, so it rewards getting the common answers right. Here every answer label is "
           "scored on its own: *precision* is how often the model is right when it gives that answer, *recall* "
           "is how often it gives that answer when it's the right one. The macro average weights every label "
           "equally within its family, and the overall macro averages the families equally, so a model that "
           "ignores rare answers scores lower.", ""]
    out.append("| Task family | Labels | " + " | ".join(f"{LABELS.get(p, p)} P / R / F1" for p in providers) + " |")
    out.append("|---|---:|" + "---:|" * len(providers))
    for cat in cats:
        n_labels = len({lab for p in providers for (c, lab) in summ[p]["labels"] if c == cat})
        cells = []
        for p in providers:
            m = macro(summ[p]["labels"], cat)
            cells.append(" / ".join(f"{100 * m[k]:.0f}" for k in ("precision", "recall", "f1")))
        out.append(f"| `{cat}` | {n_labels} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Per label, as precision / recall (%). *n* is how many items have that label as the right answer; "
               "a label with n = 0 was never right but some model picked it.")
    out.append("")
    for cat in cats:
        labels = {lab for p in providers for (c, lab) in summ[p]["labels"] if c == cat}
        support = {lab: max(summ[p]["labels"].get((cat, lab), {"support": 0})["support"] for p in providers)
                   for lab in labels}
        out += [f"<details><summary><code>{cat}</code> ({len(labels)} labels)</summary>", "",
                f"| Label | n | {names} |", "|---|---:|" + "---:|" * len(providers)]
        for lab in sorted(labels, key=lambda l: (-support[l], l)):
            cells = []
            for p in providers:
                m = summ[p]["labels"].get((cat, lab))
                cells.append("–" if m is None else f"{100 * m['precision']:.0f} / {100 * m['recall']:.0f}")
            out.append(f"| `{lab}` | {support[lab]} | " + " | ".join(cells) + " |")
        out += ["", "</details>", ""]
    return out


def spend_section() -> list[str]:
    """What the whole experiment cost, from results/spend.jsonl."""
    t = spend.totals()
    if not t:
        return []
    order = [p for p in ("tev", "jev", "glm", "opus") if p in t] + sorted(set(t) - {"tev", "jev", "glm", "opus"})
    out = ["**What this experiment cost.** Every billed API call, including prompt variants, warm-ups and retries, "
           "priced at list rates. Pre-ledger calls that left no result row are estimated from average tokens per call.", ""]
    out += ["| Model | Billed calls | Input tokens | Output tokens | USD |", "|---|---:|---:|---:|---:|"]
    for p in order:
        r = t[p]
        est = f" (≈${r['usd_estimated']:.4f} estimated)" if r["usd_estimated"] else ""
        out.append(f"| {LABELS.get(p, p)} | {r['calls']:,} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                   f"${r['usd']:.4f}{est} |")
    out.append(f"| **Total** | {sum(r['calls'] for r in t.values()):,} | "
               f"{sum(r['input_tokens'] for r in t.values()):,} | {sum(r['output_tokens'] for r in t.values()):,} | "
               f"**${sum(r['usd'] for r in t.values()):.2f}** |")
    out.append("")
    return out


def prompt_data(by_id: dict[str, Item], common: set[str]) -> dict:
    """{(model, variant): {"acc", "pair_acc", "delta", "changed"}} for TEV and JEV prompt variants."""
    from bench.providers.tev import VARIANTS

    data = {}
    for base in ("tev", "jev"):
        default = None
        for v in VARIANTS:
            spec = base if v == "default" else f"{base}.{v}"
            if not (RESULTS_DIR / f"{spec}.jsonl").exists():
                continue
            loaded = load_rows(spec, by_id)
            rows = [loaded[i] for i in sorted(common) if i in loaded]
            if len(rows) != len(common):
                continue
            s_ = summarize(spec, rows)
            if v == "default":
                default = rows
            changed = None if default is None else sum(a.key != b.key for a, b in zip(rows, default)) / len(rows)
            data[(base, v)] = {"acc": s_["acc"], "pair_acc": s_["pair_acc"], "changed": changed}
        if (base, "default") in data:
            for (b_, v), d in data.items():
                if b_ == base:
                    d["delta"] = d["acc"] - data[(base, "default")]["acc"]
    return data


def prompt_sensitivity(by_id: dict[str, Item], common: set[str]) -> list[str]:
    """Full prompt-variant table for RESULTS.md."""
    data = prompt_data(by_id, common)
    variants = [v for v in VARIANT_LABELS if any((b, v) in data for b in ("tev", "jev"))]
    bases = [b for b in ("tev", "jev") if (b, "default") in data]
    if len(variants) < 2:
        return []
    out = ["**Prompt sensitivity.** The same 400 items with five prompt versions for TEV and JEV. "
           "Δ is the accuracy change from the model's default prompt; *changed* is the share of answers "
           "that differ from the default prompt's answer.", ""]
    cols = [f"{LABELS[b]} {m}" for b in bases for m in ("accuracy", "Δ", "pair accuracy", "changed")]
    out.append("| Prompt | " + " | ".join(cols) + " |")
    out.append("|---|" + "---:|" * len(cols))
    for v in variants:
        cells = []
        for b in bases:
            d = data.get((b, v))
            if not d:
                cells += ["n/a"] * 4
                continue
            delta = "–" if v == "default" else f"{100 * d['delta']:+.1f}"
            changed = "–" if v == "default" else fmt_pct(d["changed"])
            cells += [fmt_pct(d["acc"]), delta, fmt_pct(d["pair_acc"]), changed]
        out.append(f"| {VARIANT_LABELS[v]} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def spend_section() -> list[str]:
    """What the whole experiment cost, from results/spend.jsonl."""
    t = spend.totals()
    if not t:
        return []
    order = [p for p in ("tev", "jev", "glm", "opus") if p in t] + sorted(set(t) - {"tev", "jev", "glm", "opus"})
    out = ["**What this experiment cost.** Every billed API call, including prompt variants, warm-ups and retries, "
           "priced at list rates. Pre-ledger calls that left no result row are estimated from average tokens per call.", ""]
    out += ["| Model | Billed calls | Input tokens | Output tokens | USD |", "|---|---:|---:|---:|---:|"]
    for p in order:
        r = t[p]
        est = f" (≈${r['usd_estimated']:.4f} estimated)" if r["usd_estimated"] else ""
        out.append(f"| {LABELS.get(p, p)} | {r['calls']:,} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                   f"${r['usd']:.4f}{est} |")
    out.append(f"| **Total** | {sum(r['calls'] for r in t.values()):,} | "
               f"{sum(r['input_tokens'] for r in t.values()):,} | {sum(r['output_tokens'] for r in t.values()):,} | "
               f"**${sum(r['usd'] for r in t.values()):.2f}** |")
    out.append("")
    return out


def prompt_data(by_id: dict[str, Item], common: set[str]) -> dict:
    """{(model, variant): {"acc", "pair_acc", "delta", "changed"}} for TEV and JEV prompt variants."""
    from bench.providers.tev import VARIANTS

    data = {}
    for base in ("tev", "jev"):
        default = None
        for v in VARIANTS:
            spec = base if v == "default" else f"{base}.{v}"
            if not (RESULTS_DIR / f"{spec}.jsonl").exists():
                continue
            loaded = load_rows(spec, by_id)
            rows = [loaded[i] for i in sorted(common) if i in loaded]
            if len(rows) != len(common):
                continue
            s_ = summarize(spec, rows)
            if v == "default":
                default = rows
            changed = None if default is None else sum(a.key != b.key for a, b in zip(rows, default)) / len(rows)
            data[(base, v)] = {"acc": s_["acc"], "pair_acc": s_["pair_acc"], "changed": changed}
        if (base, "default") in data:
            for (b_, v), d in data.items():
                if b_ == base:
                    d["delta"] = d["acc"] - data[(base, "default")]["acc"]
    return data


def prompt_sensitivity(by_id: dict[str, Item], common: set[str]) -> list[str]:
    """Full prompt-variant table for RESULTS.md."""
    data = prompt_data(by_id, common)
    variants = [v for v in VARIANT_LABELS if any((b, v) in data for b in ("tev", "jev"))]
    bases = [b for b in ("tev", "jev") if (b, "default") in data]
    if len(variants) < 2:
        return []
    out = ["**Prompt sensitivity.** The same 400 items with five prompt versions for TEV and JEV. "
           "Δ is the accuracy change from the model's default prompt; *changed* is the share of answers "
           "that differ from the default prompt's answer.", ""]
    cols = [f"{LABELS[b]} {m}" for b in bases for m in ("accuracy", "Δ", "pair accuracy", "changed")]
    out.append("| Prompt | " + " | ".join(cols) + " |")
    out.append("|---|" + "---:|" * len(cols))
    for v in variants:
        cells = []
        for b in bases:
            d = data.get((b, v))
            if not d:
                cells += ["n/a"] * 4
                continue
            delta = "–" if v == "default" else f"{100 * d['delta']:+.1f}"
            changed = "–" if v == "default" else fmt_pct(d["changed"])
            cells += [fmt_pct(d["acc"]), delta, fmt_pct(d["pair_acc"]), changed]
        out.append(f"| {VARIANT_LABELS[v]} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def prompt_summary(by_id: dict[str, Item], common: set[str]) -> list[str]:
    """Short 'prompts matter' block for README.md."""
    data = prompt_data(by_id, common)
    bases = [b for b in ("tev", "jev") if (b, "default") in data]
    variants = [v for v in VARIANT_LABELS if any((b, v) in data for b in bases)]
    if len(variants) < 2:
        return []
    out = ["| Prompt | " + " | ".join(f"{LABELS[b].split(' (')[0]} accuracy" for b in bases) + " | "
           + " | ".join(f"{LABELS[b].split(' (')[0]} answers changed" for b in bases) + " |",
           "|---|" + "---:|" * (2 * len(bases))]
    for v in variants:
        acc = [f"{fmt_pct(data[(b, v)]['acc'])}" + ("" if v == "default" else f" ({100 * data[(b, v)]['delta']:+.1f})")
               for b in bases]
        chg = ["–" if v == "default" else fmt_pct(data[(b, v)]["changed"]) for b in bases]
        out.append(f"| {VARIANT_LABELS[v]} | " + " | ".join(acc + chg) + " |")
    spread = [max(d["acc"] for (b_, _), d in data.items() if b_ == b) - min(d["acc"] for (b_, _), d in data.items() if b_ == b)
              for b in bases]
    worst = min((v for v in variants if v != "default"), key=lambda v: sum(data[(b, v)]["delta"] for b in bases))
    out += ["", "Best vs worst prompt: " + ", ".join(
        f"{LABELS[b].split(' (')[0]} {100 * sp:.1f} points" for b, sp in zip(bases, spread))
        + f". Biggest single effect: `{worst}`. Answers can change even when accuracy doesn't: "
        "a prompt can fix some items and break others."]
    return out


def spend_section() -> list[str]:
    """What the whole experiment cost, from results/spend.jsonl."""
    t = spend.totals()
    if not t:
        return []
    order = [p for p in ("tev", "jev", "glm", "opus") if p in t] + sorted(set(t) - {"tev", "jev", "glm", "opus"})
    out = ["**What this experiment cost.** Every billed API call, including prompt variants, warm-ups and retries, "
           "priced at list rates. Pre-ledger calls that left no result row are estimated from average tokens per call.", ""]
    out += ["| Model | Billed calls | Input tokens | Output tokens | USD |", "|---|---:|---:|---:|---:|"]
    for p in order:
        r = t[p]
        est = f" (≈${r['usd_estimated']:.4f} estimated)" if r["usd_estimated"] else ""
        out.append(f"| {LABELS.get(p, p)} | {r['calls']:,} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                   f"${r['usd']:.4f}{est} |")
    out.append(f"| **Total** | {sum(r['calls'] for r in t.values()):,} | "
               f"{sum(r['input_tokens'] for r in t.values()):,} | {sum(r['output_tokens'] for r in t.values()):,} | "
               f"**${sum(r['usd'] for r in t.values()):.2f}** |")
    out.append("")
    return out


def prompt_sensitivity(by_id: dict[str, Item], common: set[str]) -> list[str]:
    """Accuracy of TEV and JEV under each prompt variant, on the same items as the main table."""
    from bench.providers.tev import VARIANTS

    bases = [b for b in ("tev", "jev") if (RESULTS_DIR / f"{b}.jsonl").exists()]
    specs = {(b, v): b if v == "default" else f"{b}.{v}" for b in bases for v in VARIANTS}
    specs = {k: s for k, s in specs.items() if (RESULTS_DIR / f"{s}.jsonl").exists()}
    if len({v for _, v in specs}) < 2:
        return []
    summ = {}
    for (b, v), spec in specs.items():
        rows = load_rows(spec, by_id)
        rows = [rows[i] for i in sorted(common) if i in rows]
        if len(rows) == len(common):
            summ[(b, v)] = summarize(spec, rows)

    out = ["**Prompt sensitivity.** The same 400 items run with three prompt versions for TEV and JEV. "
           "Δ is the accuracy change from each model's default prompt.", ""]
    cols = [f"{LABELS[b]} {m}" for b in bases for m in ("accuracy", "Δ", "pair accuracy")]
    out.append("| Prompt | " + " | ".join(cols) + " |")
    out.append("|---|" + "---:|" * len(cols))
    for v in VARIANTS:
        cells = []
        for b in bases:
            s_, d_ = summ.get((b, v)), summ.get((b, "default"))
            if not s_:
                cells += ["n/a"] * 3
                continue
            delta = "–" if v == "default" or not d_ else f"{100 * (s_['acc'] - d_['acc']):+.1f}"
            cells += [fmt_pct(s_["acc"]), delta, fmt_pct(s_["pair_acc"])]
        out.append(f"| {VARIANT_LABELS[v]} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def routing(providers: list[str], rows: dict, summ: dict) -> dict:
    """Sweep hybrid setups (first-pass model × risk threshold × rule) and pick the recommended one.

    Returns {"configs": [...], "best": config} where each config has held-out accuracy, escalation rate and
    cost, and the recommended one is the cheapest whose held-out accuracy is within TARGET_GAP of STRONG.
    """
    from bench import route

    if STRONG not in providers:
        return {}
    g = summ[STRONG]
    configs = []
    for base in ("tev", "jev"):
        if base not in providers:
            continue
        for better in (False, True):
            for th in route.THRESHOLDS:
                ho = route.held_out(rows[base], rows[STRONG], th, better)
                configs.append({"base": base, "threshold": th, "only_if_better": better, **ho,
                                "cost_task": summ[base]["cost_task"] + ho["escalated"] * g["cost_task"]})
    ok = [c for c in configs if c["acc"] >= g["acc"] - route.TARGET_GAP]
    best = min(ok, key=lambda c: c["cost_task"]) if ok else max(configs, key=lambda c: c["acc"])
    best["risky"] = route.risky_labels(rows[best["base"]], best["threshold"],
                                       rows[STRONG] if best["only_if_better"] else None)
    best["in_sample"] = route.cascade(rows[best["base"]], rows[STRONG], best["risky"],
                                      summ[best["base"]]["cost_task"], g["cost_task"])
    return {"configs": configs, "best": best}


def short(p: str) -> str:
    return LABELS.get(p, p).split(" (")[0]


def rule_text(c: dict) -> str:
    return (f"precision < {100 * c['threshold']:.0f}%" + (f" and {short(STRONG)} does better" if c["only_if_better"] else ""))


def render_recommendation(providers: list[str], items: list[Item]) -> str:
    """README block: the cheapest hybrid that stays close to STRONG's accuracy, against STRONG alone."""
    from bench.route import TARGET_GAP

    _, _, rows, summ = prepare(providers, items)
    rt = routing(providers, rows, summ)
    if not rt:
        return "_Routing needs results for GLM 5.3._"
    c = rt["best"]
    base, b, g = c["base"], summ[c["base"]], summ[STRONG]
    B, S = short(base), short(STRONG)
    risky = sorted(c["risky"].items(), key=lambda kv: kv[1]["precision"])
    labels = ", ".join(f"`{lab}` ({100 * m['precision']:.0f}%)" for (_, lab), m in risky)
    top = max((x for x in rt["configs"] if x["base"] == base and x["only_if_better"] == c["only_if_better"]),
              key=lambda x: (x["acc"], -x["cost_task"]))
    saved = 1 - c["cost_task"] / g["cost_task"]
    tev = [x for x in rt["configs"] if x["base"] == "tev"]
    tev_note = ""
    if base != "tev" and tev:
        tb = max(tev, key=lambda x: x["acc"])
        tev_note = (f"With TEV first, the best hybrid reaches {fmt_pct(tb['acc'])} while sending "
                    f"{100 * tb['escalated']:.0f}% of tasks to {S}: TEV's mistakes are spread over too many answers. ")

    out = [f"**Use {B} for every task, and re-ask {S} only for the few answers {B} tends to get wrong.** "
           f"On tasks held out from tuning, this scores {fmt_pct(c['acc'])} against {S}'s {fmt_pct(g['acc'])}, "
           f"and costs {100 * saved:.0f}% less: ${c['cost_task'] * 1e6:,.0f} per million tasks instead of "
           f"${g['cost_task'] * 1e6:,.0f}.", ""]
    out += [f"| Setup | Accuracy | Sent to {S} | Cost per 1M tasks | vs {S} only | Latency p50 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {B} only | {fmt_pct(b['acc'])} | 0% | ${b['cost_task'] * 1e6:,.0f} | "
            f"{g['cost_task'] / b['cost_task']:.0f}× cheaper | {b['p50']:,.0f} ms |",
            f"| **Hybrid (recommended)** | **{fmt_pct(c['acc'])}** | {100 * c['escalated']:.0f}% | "
            f"**${c['cost_task'] * 1e6:,.0f}** | **{g['cost_task'] / c['cost_task']:.1f}× cheaper** | "
            f"{c['in_sample']['p50']:,.0f} ms |"]
    if top is not c:
        out.append(f"| Hybrid, escalate every answer {B} has missed | {fmt_pct(top['acc'])} | "
                   f"{100 * top['escalated']:.0f}% | ${top['cost_task'] * 1e6:,.0f} | "
                   f"{g['cost_task'] / top['cost_task']:.1f}× cheaper | – |")
    out += [f"| {S} only | {fmt_pct(g['acc'])} | 100% | ${g['cost_task'] * 1e6:,.0f} | – | {g['p50']:,.0f} ms |", ""]
    out.append(f"**The rule.** {B} answers first. If its answer is one of these {len(risky)}, send the same prompt "
               f"to {S} and use {S}'s answer: {labels}. The percentage is {B}'s precision on that answer, i.e. how "
               f"often it's right when it gives it. An answer is on the list when {rule_text(c)} on the labelled "
               f"tasks. Routing only looks at {B}'s answer, so it works at run time.")
    out += ["", f"**How it was chosen.** We tried {len(rt['configs'])} hybrids: TEV or JEV first, risk thresholds "
                f"from 80% to 100%, with or without requiring {S} to do better on that answer. Each was scored on "
                f"pairs it wasn't tuned on (5-fold cross-validation). The recommended one is the cheapest within "
                f"{100 * TARGET_GAP:.0f} point of {S} only. " + tev_note +
                f"Full sweep: [RESULTS.md](RESULTS.md#hybrid-routing-cheapest-setup-close-to-glm-53)."]
    out += ["", "The list is specific to these task families. For your own tasks, label a few hundred examples, "
                "run both models on them, and derive your own list the same way."]
    return "\n".join(out)


def routing_section(providers: list[str], rows: dict, summ: dict) -> list[str]:
    """RESULTS.md: the whole hybrid sweep, and the recommended setup's risky answers."""
    from bench.route import FOLDS, SEEDS, TARGET_GAP

    rt = routing(providers, rows, summ)
    if not rt:
        return []
    g, c = summ[STRONG], rt["best"]
    S = short(STRONG)
    out = ["## Hybrid routing: cheapest setup close to GLM 5.3", "",
           f"A first-pass model answers every task. If its answer is on a risky list, the same prompt goes to "
           f"{LABELS[STRONG]} and its answer is used. An answer is risky when the first model's precision on it is "
           f"below the threshold; with *{S} must help*, only answers where {S} is right more often on those tasks "
           f"are kept. Accuracy is held out: the list is built on {FOLDS - 1}/{FOLDS} of the pairs and scored on "
           f"the rest, averaged over {SEEDS} shuffles. Escalated tasks pay for both calls. Target: within "
           f"{100 * TARGET_GAP:.0f} point of {S} only ({fmt_pct(g['acc'])}, ${g['cost_task'] * 1e6:,.0f} per 1M tasks).",
           ""]
    out += [f"| First pass | Rule | Accuracy, held out | Sent to {S} | Cost per 1M tasks | vs {S} only |",
            "|---|---|---:|---:|---:|---:|"]
    for p in ("tev", "jev"):
        if p in summ:
            out.append(f"| {short(p)} only | – | {fmt_pct(summ[p]['acc'])} | 0% | ${summ[p]['cost_task'] * 1e6:,.0f} | "
                       f"{g['cost_task'] / summ[p]['cost_task']:.0f}× cheaper |")
    for x in rt["configs"]:
        name = rule_text(x).replace(f"and {S} does better", f"+ {S} must help")
        row = (f"{short(x['base'])} → {S} | {name} | {fmt_pct(x['acc'])} | {100 * x['escalated']:.1f}% | "
               f"${x['cost_task'] * 1e6:,.0f} | {g['cost_task'] / x['cost_task']:.1f}× cheaper")
        out.append(f"| **{row.replace(' | ', '** | **')}** |" if x is c else f"| {row} |")
    out.append(f"| {S} only | – | {fmt_pct(g['acc'])} | 100% | ${g['cost_task'] * 1e6:,.0f} | – |")
    out += ["", f"Bold is the recommended setup. The threshold is itself picked from this sweep, so its held-out "
                f"number is slightly optimistic.", ""]
    out += [f"**Risky answers in the recommended setup** ({short(c['base'])} first, {rule_text(c)}, built on all "
            f"items)", "",
            f"| Task family | Answer | Times {short(c['base'])} gave it | {short(c['base'])} precision | "
            f"{S} accuracy on those tasks |", "|---|---|---:|---:|---:|"]
    strong = {x.item.id: x for x in rows[STRONG]}
    for (cat, lab), m in sorted(c["risky"].items(), key=lambda kv: (kv[1]["precision"], kv[0])):
        hit = [strong[x.item.id].correct for x in rows[c["base"]] if x.item.category == cat and x.key == lab]
        out.append(f"| `{cat}` | `{lab}` | {m['predicted']} | {fmt_pct(m['precision'])} | {fmt_pct(sum(hit) / len(hit))} |")
    out.append("")
    return out

def write_readme(section: str, readme: Path = ROOT / "README.md", start: str = START, end: str = END) -> None:
    text = readme.read_text()
    if start not in text or end not in text:
        raise SystemExit(f"README is missing {start} / {end} markers")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    readme.write_text(f"{pre}{start}\n{section}\n{end}{post}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", nargs="+", help=f"columns to compare (default: {' '.join(DEFAULT_PROVIDERS)})")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    load_env()
    providers = args.provider or [p for p in DEFAULT_PROVIDERS if (RESULTS_DIR / f"{p}.jsonl").exists()]
    items = load_items()
    full, summary = render(providers, items), render_summary(providers, items)
    rec = render_recommendation(providers, items)
    if args.stdout:
        print(rec, "\n\n---\n", summary, "\n\n---\n", full, sep="\n")
    else:
        (ROOT / "RESULTS.md").write_text(full + "\n")
        write_readme(summary)
        write_readme(rec, start=ROUTE_START, end=ROUTE_END)
        print("README.md summary and RESULTS.md updated")


if __name__ == "__main__":
    main()
