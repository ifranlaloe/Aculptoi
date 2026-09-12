"""OpenAI-compatible client suitable for local llama.cpp servers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import httpx

from aculptoi.config import ModelConfig
from aculptoi.models.base import Message, ModelProviderError


class OpenAICompatibleProvider:
    """Call a local `/v1/chat/completions` endpoint without requiring an API key."""

    def __init__(self, config: ModelConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def endpoint(self) -> str:
        return f"{self._config.base_url}/chat/completions"

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
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        """Request strict JSON, then defensively parse the returned assistant content."""
        body = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        try:
            response = self._client.post(self.endpoint, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ModelProviderError(f"Local model request failed: {error}") from error

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                "Endpoint returned an invalid chat-completions response"
            ) from error
        if not isinstance(content, str):
            raise ModelProviderError("Endpoint returned non-text assistant content")
        return self._parse_json(content)

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
            raise ModelProviderError("Model response was not valid JSON") from error
        if not isinstance(result, dict):
            raise ModelProviderError("Model response must be a JSON object")
        return cast(dict[str, object], result)
