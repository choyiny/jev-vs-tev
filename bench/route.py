"""Routing: answer with a cheap model, and re-ask a stronger one when the cheap answer is a risky label.

A label is risky for a model when the model's precision on it (how often it's right when it gives that
answer) is below RISK_THRESHOLD. Routing only looks at the cheap model's answer, which is known at run
time, never at the gold label.

The risky list is picked from the same items it's scored on, so the in-sample accuracy is optimistic.
`held_out` picks the list on 4/5 of the pairs and scores the other 1/5, repeated over several shuffles.
"""

from __future__ import annotations

import random
import statistics

from bench.report import Row, label_metrics, pct

RISK_THRESHOLD = 0.95
THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 1.00)  # swept by bench.report.routing
TARGET_GAP = 0.01  # recommend the cheapest hybrid within this much of the strong model's accuracy
FOLDS, SEEDS = 5, 3


def risky_labels(rows: list[Row], threshold: float = RISK_THRESHOLD,
                 strong: list[Row] | None = None) -> dict[tuple[str, str], dict]:
    """{(family, label): metrics} for labels the model gives with precision below `threshold`.

    With `strong`, keep only labels where the strong model is right more often than the cheap one on the
    tasks that got that answer; escalating the rest pays for a second call that fixes nothing.
    """
    risky = {k: m for k, m in label_metrics(rows).items() if m["predicted"] and m["precision"] < threshold}
    if strong is None:
        return risky
    by_id = {r.item.id: r for r in strong}
    keep = {}
    for k, m in risky.items():
        hit = [by_id[r.item.id].correct for r in rows if (r.item.category, r.key) == k]
        if sum(hit) / len(hit) > m["precision"]:
            keep[k] = m
    return keep


def escalates(r: Row, risky) -> bool:
    return r.key is None or (r.item.category, r.key) in risky


def cascade(base: list[Row], strong: list[Row], risky, base_cost: float, strong_cost: float) -> dict:
    """Score the routed system. Escalated tasks pay for both calls and wait for both."""
    by_id = {r.item.id: r for r in strong}
    esc = [escalates(r, risky) for r in base]
    correct = [by_id[r.item.id].correct if e else r.correct for r, e in zip(base, esc)]
    pairs: dict[str, list[bool]] = {}
    for r, c in zip(base, correct):
        pairs.setdefault(r.item.pair_id, []).append(c)
    rate = sum(esc) / len(base)
    return {
        "acc": sum(correct) / len(base),
        "pair_acc": sum(all(v) for v in pairs.values()) / len(pairs),
        "escalated": rate,
        "cost_task": base_cost + rate * strong_cost,
        "p50": pct([r.latency_ms + (by_id[r.item.id].latency_ms if e else 0) for r, e in zip(base, esc)], 0.5),
    }


def held_out(base: list[Row], strong: list[Row], threshold: float = RISK_THRESHOLD, only_if_better: bool = False) -> dict:
    """Accuracy and escalation rate when the risky list is chosen without seeing the scored pairs."""
    by_id = {r.item.id: r for r in strong}
    pair_ids = sorted({r.item.pair_id for r in base})
    accs, rates = [], []
    for seed in range(SEEDS):
        order = pair_ids[:]
        random.Random(seed).shuffle(order)
        correct = esc = 0
        for f in range(FOLDS):
            test = set(order[f::FOLDS])
            train = [r for r in base if r.item.pair_id not in test]
            risky = risky_labels(train, threshold, strong if only_if_better else None)
            for r in base:
                if r.item.pair_id in test:
                    e = escalates(r, risky)
                    esc += e
                    correct += by_id[r.item.id].correct if e else r.correct
        accs.append(correct / len(base))
        rates.append(esc / len(base))
    return {"acc": statistics.fmean(accs), "escalated": statistics.fmean(rates)}
