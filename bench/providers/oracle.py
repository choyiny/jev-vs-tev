"""Offline stand-in for smoke-testing the pipeline without API keys.

Answers correctly with probability ORACLE_ACCURACY (default 0.8), deterministically per item.
"""

from __future__ import annotations

import hashlib
import os

import httpx

from bench.dataset import Item
from bench.providers.base import Prediction


class OracleProvider:
    name = "oracle"
    model = "oracle"

    def __init__(self) -> None:
        self.accuracy = float(os.environ.get("ORACLE_ACCURACY", "0.8"))

    async def predict(self, client: httpx.AsyncClient, item: Item) -> Prediction:
        h = int(hashlib.sha256(item.id.encode()).hexdigest(), 16)
        correct = (h % 1000) / 1000 < self.accuracy
        wrong = [k for k in item.keys if k != item.gold]
        key = item.gold if correct else wrong[h % len(wrong)]
        probs = {k: (0.7 if k == key else 0.3 / (len(item.keys) - 1)) for k in item.keys}
        return Prediction(item.id, self.name, self.model, key, probs, latency_ms=float(h % 50), input_tokens=100)
