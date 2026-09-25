"""Track what the experiment has cost, across every billed API call.

The runner appends one line to results/spend.jsonl for each call that returned
billed tokens: scored tasks, warm-ups, and retries alike. Nothing here is ever
truncated, including by `bench.run --fresh`.

    uv run python -m bench.spend                # totals by model
    uv run python -m bench.spend --backfill     # one-off: rebuild pre-ledger spend (see BACKFILL_EXTRA)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "spend.jsonl"


def price(provider: str) -> tuple[float, float] | None:
    """USD per 1M (input, output) tokens from the environment; prompt variants share their model's price."""
    base = provider.split(".")[0].upper()
    p_in = os.environ.get(f"{base}_PRICE_IN")
    p_out = os.environ.get(f"{base}_PRICE_OUT") or "0"
    if not p_in:
        return None
    return float(p_in), float(p_out)


def cost(provider: str, input_tokens: int, output_tokens: int) -> float | None:
    pr = price(provider)
    return None if pr is None else (input_tokens * pr[0] + output_tokens * pr[1]) / 1e6


def record(provider: str, model: str, input_tokens: int, output_tokens: int, kind: str, item_id: str = "",
           note: str = "") -> None:
    if not (input_tokens or output_tokens) or provider == "oracle":
        return
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "provider": provider,
            "model": model,
            "kind": kind,  # task | warmup | backfill | backfill_estimate
            "item_id": item_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usd": cost(provider, input_tokens, output_tokens),
            "note": note,
        }) + "\n")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def totals() -> dict[str, dict]:
    """Per base model: calls, tokens, USD, and how much of the USD is estimated."""
    out: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                "usd": 0.0, "usd_estimated": 0.0})
    for r in load():
        t = out[r["provider"].split(".")[0]]
        t["calls"] += 1
        t["input_tokens"] += r["input_tokens"]
        t["output_tokens"] += r["output_tokens"]
        usd = r["usd"] or 0.0
        t["usd"] += usd
        if r["kind"] == "backfill_estimate":
            t["usd_estimated"] += usd
    return dict(out)


# Billed calls made before the ledger existed that left no row in results/:
# warm-ups (not written to results), a 20-item JEV smoke test later wiped by --fresh,
# and manual probe calls. Counted from the session log; tokens estimated from each
# model's average billed tokens per task.
BACKFILL_EXTRA = {
    "tev": 3 + 3 + 1 + 1 + 3 + 3,      # warm-ups: smoke, full run, variant smokes (1 each), variant runs
    "jev": 23 + 3 + 3 + 1 + 1 + 3 + 3,  # wiped 20-item smoke + its 3 warm-ups, then warm-ups as above
    "glm": 1 + 1 + 3,                   # manual probe, smoke warm-up, run warm-up
    "opus": 2 + 1 + 3,                  # manual probes, smoke warm-up, run warm-up
}


def backfill() -> None:
    if any(r["kind"].startswith("backfill") for r in load()):
        raise SystemExit("ledger already has backfill rows; refusing to double count")
    results = ROOT / "results"
    avg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for path in sorted(results.glob("*.jsonl")):
        if path == LEDGER:
            continue
        spec = path.stem
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["input_tokens"] or r["output_tokens"]:
                record(spec, r["model"], r["input_tokens"], r["output_tokens"], "backfill", r["item_id"],
                       "rebuilt from results file")
                a = avg[spec.split(".")[0]]
                a[0] += r["input_tokens"]
                a[1] += r["output_tokens"]
                a[2] += 1
    for base, n in BACKFILL_EXTRA.items():
        tin, tout, k = avg[base]
        if not k:
            continue
        for _ in range(n):
            record(base, base, round(tin / k), round(tout / k), "backfill_estimate",
                   note="pre-ledger call with no results row; tokens = model average")


def main() -> None:
    from bench.run import load_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    load_env()
    if args.backfill:
        backfill()
    grand = 0.0
    print(f"{'model':8s} {'calls':>6s} {'input tok':>10s} {'output tok':>10s} {'USD':>10s}")
    for base, t in sorted(totals().items()):
        grand += t["usd"]
        print(f"{base:8s} {t['calls']:6d} {t['input_tokens']:10d} {t['output_tokens']:10d} {t['usd']:10.4f}")
    print(f"{'total':8s} {'':6s} {'':10s} {'':10s} {grand:10.4f}")


if __name__ == "__main__":
    main()
