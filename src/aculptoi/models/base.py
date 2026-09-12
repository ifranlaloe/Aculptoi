"""Interfaces for models used by the harness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict


class Message(TypedDict):
    """Minimal OpenAI-style chat message."""

    role: str
    content: object


class ModelProviderError(RuntimeError):
    """An endpoint, transport, or response failure from a model provider."""


class ModelProvider(Protocol):
    """A provider capable of returning a JSON object from chat messages."""

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]: ...
