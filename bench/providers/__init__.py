from __future__ import annotations

from bench.providers.base import Prediction, Provider

BASE_PROVIDERS = ("tev", "jev", "glm", "opus", "oracle")


def get_provider(spec: str) -> Provider:
    """`tev`, `jev`, `glm`, `opus`, `oracle`; TEV and JEV also take a prompt variant, e.g. `tev.careful`."""
    name, _, variant = spec.partition(".")
    variant = variant or "default"
    if name in ("tev", "jev"):
        from bench.providers.tev import VARIANTS

        if variant not in VARIANTS:
            raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")
        if name == "tev":
            from bench.providers.tev import TevProvider

            return TevProvider(variant)
        from bench.providers.jev import JevProvider

        return JevProvider(variant)
    if variant != "default":
        raise ValueError(f"prompt variants apply to tev and jev only, not {name!r}")
    if name in ("glm", "opus"):
        from bench.providers.frontier import FrontierProvider

        return FrontierProvider(name)
    if name == "oracle":
        from bench.providers.oracle import OracleProvider

        return OracleProvider()
    raise ValueError(f"unknown provider {spec!r}")


__all__ = ["BASE_PROVIDERS", "Prediction", "Provider", "get_provider"]
