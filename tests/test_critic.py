from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from aculptoi.agent import Actor, VisionCritic
from aculptoi.models import ModelResponseError
from aculptoi.models.base import Message
from aculptoi.schemas.construction import ConstructionPlan
from aculptoi.schemas.critique import VisualCritique


class RecordingProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []
        self.max_tokens: list[int | None] = []

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        return self._responses.pop(0)


def _write_render(path: Path, size: tuple[int, int] = (64, 32)) -> bytes:
    Image.new("RGB", size, color="white").save(path)
    return path.read_bytes()


def test_critique_is_read_only_structured_data() -> None:
    critique = VisualCritique.model_validate(
        {
            "score": 0.63,
            "summary": "Recognizable silhouette with proportion issues.",
            "issues": [
                {
                    "severity": "high",
                    "region": "neck",
                    "description": "Too short relative to torso.",
                    "suggestion": "Lengthen and taper it.",
                }
            ],
        }
    )
    assert critique.score == 0.63
    assert critique.issues[0].severity == "high"


def test_critique_rejects_executable_extras() -> None:
    with pytest.raises(ValidationError):
        VisualCritique.model_validate(
            {
                "score": 0.5,
                "summary": "Fine",
                "issues": [],
                "actions": [{"command": "object.delete", "object": "Cube"}],
            }
        )


def test_shared_provider_receives_separate_actor_and_critic_requests(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "reason": "Build one coherent body component.",
                "items": [
                    {
                        "id": "body",
                        "title": "Body",
                        "objective": "Create the creature body.",
                        "depends_on": [],
                    }
                ],
            },
            {"score": 0.7, "summary": "Recognizable.", "issues": []},
        ]
    )
    image = tmp_path / "front.png"
    original = _write_render(image, (1600, 800))

    plan = Actor(provider).plan_iteration(
        "create a creature",
        {"objects": []},
        None,
        iteration=1,
        max_actor_requests=100,
        max_actions=1000,
    )
    critique = VisionCritic(provider, max_image_dimension=1000).inspect(
        "create a creature", [image]
    )

    actor_messages, critic_messages = provider.calls
    actor_content = actor_messages[1]["content"]
    critic_content = critic_messages[1]["content"]
    assert plan.items[0].id == "body"
    assert critique.score == 0.7
    assert isinstance(actor_content, str)
    assert "image_url" not in actor_content
    assert isinstance(critic_content, list)
    assert provider.max_tokens == [1536, 768]
    image_url = next(
        part["image_url"]["url"] for part in critic_content if part["type"] == "image_url"
    )
    prepared = Image.open(BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1])))
    assert prepared.size == (1000, 500)
    assert image.read_bytes() == original


def test_separate_providers_receive_only_their_own_role_request(tmp_path: Path) -> None:
    actor_provider = RecordingProvider(
        [
            {
                "reason": "Build a body.",
                "items": [
                    {
                        "id": "body",
                        "title": "Body",
                        "objective": "Create a body.",
                        "depends_on": [],
                    }
                ],
            }
        ]
    )
    critic_provider = RecordingProvider([{"score": 0.4, "summary": "Needs work.", "issues": []}])
    image = tmp_path / "perspective.png"
    _write_render(image)

    Actor(actor_provider).plan_iteration(
        "create a creature",
        {"objects": []},
        None,
        iteration=1,
        max_actor_requests=100,
        max_actions=1000,
    )
    VisionCritic(critic_provider).inspect("create a creature", [image])

    assert len(actor_provider.calls) == 1
    assert len(critic_provider.calls) == 1
    assert actor_provider.max_tokens == [1536]
    assert critic_provider.max_tokens == [768]


def test_actor_response_still_passes_typed_action_validation() -> None:
    provider = RecordingProvider(
        [
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Attempt an unsafe escape hatch.",
                "actions": [{"command": "execute_bpy", "code": "bpy.ops.wm.save_as_mainfile()"}],
            }
        ]
    )
    construction_plan = ConstructionPlan.model_validate(
        {
            "reason": "Build a body.",
            "items": [
                {
                    "id": "body",
                    "title": "Body",
                    "objective": "Create a body.",
                    "depends_on": [],
                }
            ],
        }
    )

    with pytest.raises(ModelResponseError, match="work-item action schema") as error:
        Actor(provider).execute_work_item(
            "create a creature",
            {"objects": []},
            None,
            construction_plan,
            construction_plan.items[0],
            iteration=1,
            action_batch=1,
            completed_work_item_ids=[],
            completed_work_items=[],
            completion_criteria=None,
            remaining_actor_requests=98,
            remaining_actions=1000,
            recent_execution=None,
        )

    assert '"command": "execute_bpy"' in error.value.raw_response


def test_first_work_item_response_must_create_its_completion_criteria() -> None:
    provider = RecordingProvider(
        [
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create the body.",
                "actions": [{"command": "object.create", "name": "Body", "primitive": "cube"}],
            }
        ]
    )
    construction_plan = ConstructionPlan.model_validate(
        {
            "reason": "Build a body.",
            "items": [
                {
                    "id": "body",
                    "title": "Body",
                    "objective": "Create a body.",
                    "depends_on": [],
                }
            ],
        }
    )

    with pytest.raises(ModelResponseError, match="must define completion criteria"):
        Actor(provider).execute_work_item(
            "create a creature",
            {"objects": []},
            None,
            construction_plan,
            construction_plan.items[0],
            iteration=1,
            action_batch=1,
            completed_work_item_ids=[],
            completed_work_items=[],
            completion_criteria=None,
            remaining_actor_requests=98,
            remaining_actions=1000,
            recent_execution=None,
        )


def test_later_work_item_response_cannot_replace_completion_criteria() -> None:
    provider = RecordingProvider(
        [
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Finish the body.",
                "completion_criteria": ["A different definition."],
                "actions": [],
            }
        ]
    )
    construction_plan = ConstructionPlan.model_validate(
        {
            "reason": "Build a body.",
            "items": [
                {
                    "id": "body",
                    "title": "Body",
                    "objective": "Create a body.",
                    "depends_on": [],
                }
            ],
        }
    )

    with pytest.raises(ModelResponseError, match="must not replace completion criteria"):
        Actor(provider).execute_work_item(
            "create a creature",
            {"objects": []},
            None,
            construction_plan,
            construction_plan.items[0],
            iteration=1,
            action_batch=2,
            completed_work_item_ids=[],
            completed_work_items=[],
            completion_criteria=["A body object exists."],
            remaining_actor_requests=97,
            remaining_actions=999,
            recent_execution={"result": {"executed": []}},
        )
