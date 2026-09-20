"""OpenAI-compatible client suitable for local llama.cpp servers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from aculptoi.config import ModelConfig
from aculptoi.models.base import (
    Message,
    ModelCompletion,
    ModelProviderError,
    ModelResponseError,
    ModelUsage,
)
from aculptoi.reasoning import ReasoningEffort


class OpenAICompatibleProvider:
    """Call a local `/v1/chat/completions` endpoint without requiring an API key."""

    def __init__(self, config: ModelConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def endpoint(self) -> str:
        return f"{self._config.base_url}/chat/completions"

    @property
    def supports_json_schema(self) -> bool:
        """Expose configured structured-output support to the generic provider layer."""
        return self._config.supports_json_schema

    def healthcheck(self) -> dict[str, object]:
        """Perform a low-cost endpoint check used by `aculptoi doctor`."""
        try:
            response = self._client.get(
                f"{self._config.base_url}/models",
                timeout=min(self._config.timeout_seconds, 10.0),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelProviderError(f"Could not reach {self._config.base_url}: {error}") from error
        body = response.json()
        return body if isinstance(body, dict) else {"response": body}

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, object] | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        """Request parsed JSON while preserving the legacy provider interface."""
        return self.complete_json_with_usage(
            messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        ).value

    def complete_json_with_usage(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, object] | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ModelCompletion:
        """Request strict JSON and retain OpenAI-compatible usage metadata when available."""
        response_format: dict[str, object] = {"type": "json_object"}
        if response_schema is not None and self.supports_json_schema:
            response_format["schema"] = dict(response_schema)
        body = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": 0,
            "response_format": response_format,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        template_kwargs: dict[str, object] = {}
        thinking_transport = (
            self._config.thinking_transport or self._config.reasoning_effort_transport
        )
        if thinking is not None:
            if thinking_transport == "chat_template_kwargs":
                template_kwargs["enable_thinking"] = thinking
            elif thinking_transport == "top_level":
                body["enable_thinking"] = thinking
        if reasoning_effort is not None:
            if self._config.reasoning_effort_transport == "chat_template_kwargs":
                template_kwargs["reasoning_effort"] = reasoning_effort
            elif self._config.reasoning_effort_transport == "top_level":
                body["reasoning_effort"] = reasoning_effort
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        try:
            response = self._client.post(self.endpoint, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelProviderError(f"Local model request failed: {error}") from error

        try:
            response_body = response.json()
        except json.JSONDecodeError as error:
            raise ModelProviderError(
                "Endpoint returned an invalid chat-completions response"
            ) from error
        usage = ModelUsage.from_response(response_body)
        try:
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelProviderError(
                "Endpoint returned an invalid chat-completions response", usage=usage
            ) from error
        if not isinstance(content, str):
            raise ModelProviderError("Endpoint returned non-text assistant content", usage=usage)
        try:
            value = self._parse_json(content)
        except ModelResponseError as error:
            error.usage = usage
            raise
        return ModelCompletion(value, usage)

    def close(self) -> None:
        """Close the reusable local HTTP connection pool."""
        self._client.close()

    @staticmethod
    def _parse_json(content: str) -> dict[str, object]:
        cleaned = content.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ModelResponseError("Model response was not valid JSON", content) from error
        if not isinstance(result, dict):
            raise ModelResponseError("Model response must be a JSON object", content)
        return cast(dict[str, object], result)
