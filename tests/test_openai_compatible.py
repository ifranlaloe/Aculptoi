from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from PIL import Image

from aculptoi.agent import Actor, VisionCritic
from aculptoi.config import ProviderConfig
from aculptoi.inspection import InspectionReviewer
from aculptoi.models import (
    ModelProviderError,
    ModelResponseError,
    OpenAICompatibleProvider,
    complete_json_with_usage,
)
from aculptoi.models.base import Message
from aculptoi.reasoning import ReasoningEffort
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


def test_provider_exposes_openai_usage_without_making_up_missing_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "completion_tokens_details": {"reasoning_tokens": 34},
                },
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        completion = provider.complete_json_with_usage([{"role": "user", "content": "{}"}])
    finally:
        provider.close()

    assert completion.value == {"ok": True}
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 123
    assert completion.usage.completion_tokens == 45
    assert completion.usage.reasoning_tokens == 34


def test_provider_keeps_unavailable_usage_fields_unset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 123},
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        completion = provider.complete_json_with_usage([{"role": "user", "content": "{}"}])
    finally:
        provider.close()

    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 123
    assert completion.usage.completion_tokens is None
    assert completion.usage.reasoning_tokens is None


def test_provider_sends_schema_constrained_json_only_when_requested() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    try:
        provider.complete_json([{"role": "user", "content": "{}"}])
        provider.complete_json([{"role": "user", "content": "{}"}], response_schema=schema)
    finally:
        provider.close()

    generic, constrained = [json.loads(request.content) for request in requests]
    assert generic["response_format"] == {"type": "json_object"}
    assert constrained["response_format"] == {"type": "json_object", "schema": schema}


def test_provider_can_disable_schema_constrained_output_by_configuration() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(
            base_url="http://127.0.0.1:8080/v1",
            model="plain-json-endpoint",
            supports_json_schema=False,
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        assert provider.supports_json_schema is False
        provider.complete_json(
            [{"role": "user", "content": "{}"}],
            response_schema={"type": "object"},
        )
    finally:
        provider.close()

    assert json.loads(requests[0].content)["response_format"] == {"type": "json_object"}


def test_provider_accepts_top_level_reasoning_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"reasoning_tokens": 34},
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        completion = provider.complete_json_with_usage([{"role": "user", "content": "{}"}])
    finally:
        provider.close()

    assert completion.usage is not None
    assert completion.usage.reasoning_tokens == 34


def test_provider_preserves_usage_on_malformed_model_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "not json"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45},
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(ModelResponseError) as error:
            provider.complete_json_with_usage([{"role": "user", "content": "{}"}])
    finally:
        provider.close()

    assert error.value.usage is not None
    assert error.value.usage.prompt_tokens == 123
    assert error.value.usage.completion_tokens == 45


def test_provider_preserves_usage_on_invalid_completion_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 123},
            },
        )

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(ModelProviderError) as error:
            provider.complete_json_with_usage([{"role": "user", "content": "{}"}])
    finally:
        provider.close()

    assert error.value.usage is not None
    assert error.value.usage.prompt_tokens == 123


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
    assert body["max_tokens"] == 4_096
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_role_requests_send_explicit_effective_thinking_settings(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    responses = [
        {
            "reason": "Create one visible base object.",
            "items": [{"id": "base", "title": "Base", "objective": "Create a cube."}],
        },
        {"status": "accept", "confidence": 95, "problems": []},
        {"score": 80, "issues": [["base", "M", 90, ["A1"], "edges are uneven"]]},
        {
            "desc": "The visible edges are inconsistent.",
            "evidence": ["A1: edge spacing differs across the silhouette."],
            "cause": "The base was scaled unevenly.",
            "fix": "Restore equal dimensions.",
            "criteria": ["All visible base edges have equal spacing."],
            "confidence": 90,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    image = tmp_path / "atlas.png"
    Image.new("RGB", (64, 32), color="white").save(image)
    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="http://127.0.0.1:8080/v1", model="local-multimodal"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        Actor(provider).plan_iteration_messages(
            [
                {"role": "system", "content": "plan"},
                {"role": "user", "content": "{}"},
            ]
        )
        InspectionReviewer(provider).review(image, _manifest())
        critic = VisionCritic(provider)
        discovery = critic.discover("create a cube", image, _manifest())
        critic.analyze_issue("create a cube", discovery.issues[0], image, _manifest())
    finally:
        provider.close()

    actor, reviewer, discovery, analysis = [json.loads(request.content) for request in requests]
    assert actor["max_tokens"] == 16_384
    assert actor["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert reviewer["max_tokens"] == 4_096
    assert reviewer["chat_template_kwargs"] == {"enable_thinking": False}
    assert discovery["max_tokens"] == 4_096
    assert discovery["chat_template_kwargs"] == {"enable_thinking": False}
    assert analysis["max_tokens"] == 16_384
    assert analysis["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    for body in (actor, reviewer, discovery, analysis):
        assert body["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in reviewer["chat_template_kwargs"]
    assert "reasoning_effort" not in discovery["chat_template_kwargs"]


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
            thinking=True,
            reasoning_effort="high",
        ) == {"ok": True}
    finally:
        provider.close()

    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 16_384
    if expected_field == "reasoning_effort":
        assert "chat_template_kwargs" not in body
        assert body["enable_thinking"] is True
    else:
        assert "reasoning_effort" not in body
        assert "chat_template_kwargs" not in body
        assert "enable_thinking" not in body
    if expected_field is not None:
        assert body[expected_field] == expected_value


def test_provider_can_omit_explicit_thinking_metadata() -> None:
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
                "thinking_transport": "omit",
            }
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        assert provider.complete_json(
            [{"role": "user", "content": "{}"}],
            max_tokens=4_096,
            thinking=False,
        ) == {"ok": True}
    finally:
        provider.close()

    body = json.loads(requests[0].content)
    assert "chat_template_kwargs" not in body
    assert "enable_thinking" not in body


def test_legacy_provider_without_thinking_keyword_remains_compatible() -> None:
    class LegacyProvider:
        def __init__(self) -> None:
            self.reasoning_efforts: list[str | None] = []

        def complete_json(
            self,
            messages: Sequence[Message],
            *,
            max_tokens: int | None = None,
            reasoning_effort: ReasoningEffort | None = None,
        ) -> dict[str, object]:
            assert messages
            assert max_tokens == 4_096
            self.reasoning_efforts.append(reasoning_effort)
            return {"ok": True}

    provider = LegacyProvider()
    completion = complete_json_with_usage(
        provider,
        [{"role": "user", "content": "{}"}],
        max_tokens=4_096,
        thinking=False,
    )

    assert completion.value == {"ok": True}
    assert provider.reasoning_efforts == [None]
