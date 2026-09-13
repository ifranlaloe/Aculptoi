from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from aculptoi.agent import VisionCritic
from aculptoi.config import ProviderConfig
from aculptoi.models import ModelResponseError, OpenAICompatibleProvider


def test_json_parser_accepts_a_fenced_json_object() -> None:
    assert OpenAICompatibleProvider._parse_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_json_parser_rejects_non_object() -> None:
    with pytest.raises(ModelResponseError, match="JSON object") as error:
        OpenAICompatibleProvider._parse_json("[]")

    assert error.value.raw_response == "[]"


def test_vision_request_uses_openai_multimodal_image_content(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"score": 80, "issues": []}'}}]},
        )

    image = tmp_path / "front.png"
    Image.new("RGB", (64, 32), color="white").save(image)
    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        critique = VisionCritic(provider).inspect("create a creature", [image])
    finally:
        provider.close()

    body = json.loads(requests[0].content)
    content = body["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert critique.score == 0.8
    assert body["model"] == "local-multimodal"
    assert body["max_tokens"] == 16_384
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert "reasoning_effort" not in body
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    ("transport", "expected_field", "expected_value"),
    [
        ("top_level", "reasoning_effort", "high"),
        ("omit", None, None),
    ],
)
def test_provider_reasoning_effort_encoding_is_explicit_and_non_competing(
    transport: str,
    expected_field: str | None,
    expected_value: str | None,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig.model_validate(
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
                "reasoning_effort_transport": transport,
            }
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        assert provider.complete_json(
            [{"role": "user", "content": "{}"}],
            max_tokens=16_384,
            reasoning_effort="high",
        ) == {"ok": True}
    finally:
        provider.close()

    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 16_384
    if expected_field == "reasoning_effort":
        assert "chat_template_kwargs" not in body
    else:
        assert "reasoning_effort" not in body
        assert "chat_template_kwargs" not in body
    if expected_field is not None:
        assert body[expected_field] == expected_value
