"""Reuse provider instances selected by named application roles."""

from __future__ import annotations

from collections.abc import Mapping

from aculptoi.config import ProviderConfig
from aculptoi.models.openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    """Create at most one local HTTP client per configured provider name."""

    def __init__(self, providers: Mapping[str, ProviderConfig]) -> None:
        self._configs = dict(providers)
        self._instances: dict[str, OpenAICompatibleProvider] = {}

    def get(self, name: str) -> OpenAICompatibleProvider:
        """Return the cached provider for ``name`` or a clear configuration error."""
        if name not in self._configs:
            raise ValueError(f"Unknown model provider: {name}")
        if name not in self._instances:
            self._instances[name] = OpenAICompatibleProvider(self._configs[name])
        return self._instances[name]

    def close(self) -> None:
        """Release HTTP clients created by this registry."""
        for provider in self._instances.values():
            provider.close()
