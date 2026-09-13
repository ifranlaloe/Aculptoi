from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from aculptoi.agent import VisionCritic
from aculptoi.config import ProviderConfig
from aculptoi.models import ModelResponseError, OpenAICompatibleProvider
from aculptoi.schemas.inspection import (
    AtlasLayout,
    InspectionAtlasManifest,
    InspectionAtlasTile,
    InspectionBounds,
    InspectionFraming,
)


def _manifest() -> InspectionAtlasManifest:
    layout = AtlasLayout(columns=1, rows=1, tile_dimension=64, width=64, height=64)
    return InspectionAtlasManifest(
        sensor_version="inspection-atlas-v1",
        lighting_rig="neutral-studio-v1",
        width=64,
        height=64,
        layout=layout,
        bounds=InspectionBounds(
            minimum=(-1.0, -1.0, -1.0),
            maximum=(1.0, 1.0, 1.0),
            center=(0.0, 0.0, 0.0),
            radius=1.8,
        ),
        framing=InspectionFraming(margin=1.15, distance=5.0, orthographic_scale=4.0),
        estimated_surface_coverage=0.9,
        tiles={
            "A1": InspectionAtlasTile(
                tile_id="A1",
                camera_id="anchor-front",
                source="shots/A1.png",
                pixel_bounds=(0, 0, 64, 64),
                azimuth_degrees=0,
                elevation_degrees=0,
                orientation="front",
                projection="orthographic",
                selection_kind="canonical_anchor",
                selection_reason="canonical_anchor",
                coverage_gain=0,
                information_gain=0,
            )
        },
    )


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
        critique = VisionCritic(provider).inspect("create a creature", image, _manifest())
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
