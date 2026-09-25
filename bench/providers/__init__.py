from __future__ import annotations

from bench.providers.base import Prediction, Provider


def get_provider(name: str) -> Provider:
    if name == "tev":
        from bench.providers.tev import TevProvider

        return TevProvider()
    if name == "jev":
        from bench.providers.jev import JevProvider

        return JevProvider()
    if name == "oracle":
        from bench.providers.oracle import OracleProvider

        return OracleProvider()
    raise ValueError(f"unknown provider {name!r}")


__all__ = ["Prediction", "Provider", "get_provider"]
