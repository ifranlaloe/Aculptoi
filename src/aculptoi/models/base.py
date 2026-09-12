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


class ModelResponseError(ModelProviderError):
    """A response error that may retain local-only diagnostic content."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ModelProvider(Protocol):
    """A provider capable of returning a JSON object from chat messages."""

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]: ...
