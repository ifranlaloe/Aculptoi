from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from aculptoi.agent import VisionCritic
from aculptoi.config import ProviderConfig
from aculptoi.models import ModelProviderError, OpenAICompatibleProvider


def test_json_parser_accepts_a_fenced_json_object() -> None:
    assert OpenAICompatibleProvider._parse_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_json_parser_rejects_non_object() -> None:
    with pytest.raises(ModelProviderError, match="JSON object"):
        OpenAICompatibleProvider._parse_json("[]")


def test_vision_request_uses_openai_multimodal_image_content(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"score": 0.8, "summary": "Good.", "issues": []}'}}
                ]
            },
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
    assert body["max_tokens"] == 768
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
