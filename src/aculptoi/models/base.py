"""Interfaces for models used by the harness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Protocol, TypedDict

from aculptoi.reasoning import ReasoningEffort


class Message(TypedDict):
    """Minimal OpenAI-style chat message."""

    role: str
    content: object


@dataclass(frozen=True)
class ModelUsage:
    """Optional usage values returned by a model provider for one completion."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    @classmethod
    def from_response(cls, response: object) -> ModelUsage | None:
        """Read standard OpenAI-compatible usage values without inventing metrics."""
        if not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None

        def token_count(value: object) -> int | None:
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                else None
            )

        details = usage.get("completion_tokens_details")
        reasoning_tokens = (
            token_count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        )
        if reasoning_tokens is None:
            reasoning_tokens = token_count(usage.get("reasoning_tokens"))
        result = cls(
            prompt_tokens=token_count(usage.get("prompt_tokens")),
            completion_tokens=token_count(usage.get("completion_tokens")),
            reasoning_tokens=reasoning_tokens,
        )
        return result if any(value is not None for value in result.__dict__.values()) else None


@dataclass(frozen=True)
class ModelCompletion:
    """A parsed model value paired with optional provider-reported usage."""

    value: dict[str, object]
    usage: ModelUsage | None = None


class ModelProviderError(RuntimeError):
    """An endpoint, transport, or response failure from a model provider."""

    def __init__(self, message: str, *, usage: ModelUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class ModelResponseError(ModelProviderError):
    """A response error that may retain local-only diagnostic content."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        *,
        usage: ModelUsage | None = None,
    ) -> None:
        super().__init__(message, usage=usage)
        self.raw_response = raw_response


class ModelProvider(Protocol):
    """A provider capable of returning a JSON object from chat messages."""

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]: ...


class LegacyModelProvider(Protocol):
    """A provider implementing the interface before explicit thinking control."""

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]: ...


def complete_json_with_usage(
    provider: ModelProvider | LegacyModelProvider,
    messages: Sequence[Message],
    *,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ModelCompletion:
    """Use provider usage metadata when available without burdening existing providers."""
    complete = getattr(provider, "complete_json_with_usage", None)
    if callable(complete):
        result = _complete_with_optional_thinking(
            complete,
            messages,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        if isinstance(result, ModelCompletion):
            return result
        raise ModelProviderError("Provider returned an invalid completion envelope")
    result = _complete_with_optional_thinking(
        provider.complete_json,
        messages,
        max_tokens=max_tokens,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(result, dict):
        raise ModelProviderError("Provider returned a non-object JSON completion")
    return ModelCompletion(result)


def _complete_with_optional_thinking(
    complete: Callable[..., object],
    messages: Sequence[Message],
    *,
    max_tokens: int | None,
    thinking: bool | None,
    reasoning_effort: ReasoningEffort | None,
) -> object:
    """Call legacy providers safely when they do not yet accept thinking control."""
    arguments: dict[str, object] = {
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    if thinking is not None and _accepts_keyword(complete, "thinking"):
        arguments["thinking"] = thinking
    return complete(messages, **arguments)


def _accepts_keyword(callable_object: Callable[..., object], keyword: str) -> bool:
    """Detect an optional provider keyword without catching provider-raised TypeErrors."""
    try:
        parameters = signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )
