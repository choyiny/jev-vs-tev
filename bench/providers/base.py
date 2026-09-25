from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx

from bench.dataset import Item


@dataclass
class Prediction:
    item_id: str
    provider: str
    model: str
    key: str | None  # predicted option key; None when the output was unusable
    probs: dict[str, float] | None = None  # distribution over option keys, if the API exposes one
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class Provider(Protocol):
    name: str
    model: str

    async def predict(self, client: httpx.AsyncClient, item: Item) -> Prediction: ...
