from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from aculptoi.agent import Actor, VisionCritic
from aculptoi.models import ModelResponseError
from aculptoi.models.base import Message
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.construction import ConstructionPlan
from aculptoi.schemas.critique import (
    VisualCritique,
    VisualIssue,
    VisualIssueDetail,
    VisualIssueDetailWire,
    VisualIssueDiscovery,
    VisualIssueDiscoveryWire,
    VisualIssueSummary,
)


class RecordingProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []
        self.max_tokens: list[int | None] = []
        self.reasoning_efforts: list[ReasoningEffort | None] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        self.reasoning_efforts.append(reasoning_effort)
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
                    "id": "issue-001",
                    "title": "Neck too short",
                    "severity": "high",
                    "region": "neck",
                    "confidence": 0.9,
                    "evidence_views": ["right", "perspective"],
                    "observation": "too short relative to torso",
                    "detail_status": "detailed",
                    "description": "Too short relative to torso.",
                    "evidence": ["The right silhouette has almost no neck length."],
                    "suggested_correction": "Lengthen and taper it.",
                    "success_criteria": ["The neck reads clearly in the right view."],
                    "analysis_confidence": 0.88,
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


def test_issue_detail_rejects_blender_actions() -> None:
    with pytest.raises(ValidationError):
        VisualIssueDetail.model_validate(
            {
                "id": "issue-001",
                "description": "The wing intersects the torso.",
                "evidence": ["Visible in perspective."],
                "likely_cause": "The root is too far inward.",
                "suggested_correction": "Separate the root from the torso silhouette.",
                "success_criteria": ["The forms are visibly separate."],
                "confidence": 0.9,
                "actions": [{"command": "object.delete", "name": "Cube"}],
            }
        )


def test_compact_discovery_wire_expands_codes_percentages_and_deterministic_ids() -> None:
    discovery = VisualIssueDiscoveryWire.model_validate(
        {
            "score": 58,
            "issues": [
                ["left_wing", "C", 97, ["F", "P"], "intersects torso"],
                ["neck", "H", 95, ["R", "P"], "too short relative to torso"],
            ],
        }
    ).to_domain()

    first, second = discovery.issues
    assert discovery.score == 0.58
    assert discovery.summary == "2 visible issues identified; highest severity: critical."
    assert first.model_dump(mode="json") == {
        "id": "issue-001",
        "title": "Left Wing: intersects torso",
        "region": "left_wing",
        "severity": "critical",
        "confidence": 0.97,
        "evidence_views": ["front", "perspective"],
        "observation": "intersects torso",
    }
    assert second.id == "issue-002"
    assert second.severity == "high"
    assert second.evidence_views == ["right", "perspective"]
    assert second.confidence == 0.95


@pytest.mark.parametrize(
    ("issues", "match"),
    [
        ([["wing", "H", 90, ["F"]]], "Field required"),
        ([["wing", "X", 90, ["F"], "intersects torso"]], "literal_error"),
        ([["wing", "H", 90, ["X"], "intersects torso"]], "literal_error"),
    ],
)
def test_compact_discovery_wire_rejects_malformed_tuples_and_unknown_codes(
    issues: list[object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        VisualIssueDiscoveryWire.model_validate({"score": 58, "issues": issues})


def test_compact_discovery_wire_requires_the_complete_top_level_shape() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        VisualIssueDiscoveryWire.model_validate({"score": 58})


def test_compact_detail_wire_receives_its_id_from_application_context() -> None:
    detail = VisualIssueDetailWire.model_validate(
        {
            "desc": "Wing penetrates upper torso near the shoulder.",
            "evidence": ["F: contour disappears into torso"],
            "cause": "wing root too low and inward",
            "fix": "move root upward and outward",
            "criteria": ["no penetration outside attachment"],
            "confidence": 96,
        }
    ).to_domain("issue-007")

    assert detail.id == "issue-007"
    assert detail.description == "Wing penetrates upper torso near the shoulder."
    assert detail.suggested_correction == "move root upward and outward"
    assert detail.confidence == 0.96


def test_actor_receives_rich_critique_not_compact_wire_data() -> None:
    actor_provider = RecordingProvider(
        [
            {
                "reason": "Address the wing intersection.",
                "items": [
                    {
                        "id": "wing-root",
                        "title": "Wing root",
                        "objective": "Correct the wing root.",
                        "depends_on": [],
                    }
                ],
            }
        ]
    )
    critique = VisualCritique(
        score=0.58,
        summary="1 visible issue identified; highest severity: critical.",
        issues=[
            VisualIssue(
                id="issue-001",
                title="Left Wing: intersects torso",
                region="left_wing",
                severity="critical",
                confidence=0.97,
                evidence_views=["front", "perspective"],
                observation="intersects torso",
                detail_status="detailed",
                description="The wing penetrates the upper torso near its root.",
                evidence=["F: contour disappears into torso"],
                likely_cause="The root is too far inward.",
                suggested_correction="Move the root outward.",
                success_criteria=["The forms are visually separate."],
                analysis_confidence=0.96,
            )
        ],
    )

    Actor(actor_provider).plan_iteration(
        "create a dragon",
        {"objects": []},
        critique,
        iteration=2,
        max_actor_requests=100,
        max_actions=1000,
    )

    context = json.loads(actor_provider.calls[0][1]["content"])
    actor_issue = context["latest_critique"]["issues"][0]
    assert actor_issue["id"] == "issue-001"
    assert actor_issue["observation"] == "intersects torso"
    assert actor_issue["suggested_correction"] == "Move the root outward."
    assert "desc" not in actor_issue


def test_discovery_issue_limit_still_applies_to_compact_wire(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 50,
                "issues": [
                    ["body", "H", 90, ["F"], "too small"],
                    ["tail", "M", 80, ["P"], "missing"],
                ],
            }
        ]
    )
    image = tmp_path / "front.png"
    _write_render(image)

    with pytest.raises(ModelResponseError, match="max_discovered_issues"):
        VisionCritic(provider, max_discovered_issues=1).discover("create a creature", [image])


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
            {"score": 70, "issues": []},
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
    assert provider.max_tokens == [16_384, 16_384]
    assert provider.reasoning_efforts == ["medium", "medium"]
    image_url = next(
        part["image_url"]["url"] for part in critic_content if part["type"] == "image_url"
    )
    prepared = Image.open(BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1])))
    assert prepared.size == (1000, 500)
    assert image.read_bytes() == original


def test_actor_construction_and_work_item_requests_share_their_configured_settings() -> None:
    provider = RecordingProvider(
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
            },
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "The body is complete.",
                "completion_criteria": ["A body object exists."],
                "actions": [],
            },
        ]
    )
    actor = Actor(provider, max_output_tokens=16_384, reasoning_effort="high")
    plan = actor.plan_iteration(
        "create a body",
        {"objects": []},
        None,
        iteration=1,
        max_actor_requests=100,
        max_actions=1_000,
    )
    actor.execute_work_item(
        "create a body",
        {"objects": []},
        None,
        plan,
        plan.items[0],
        iteration=1,
        action_batch=1,
        completed_work_item_ids=[],
        completed_work_items=[],
        completion_criteria=None,
        remaining_actor_requests=99,
        remaining_actions=1_000,
        recent_execution=None,
    )

    assert provider.max_tokens == [16_384, 16_384]
    assert provider.reasoning_efforts == ["high", "high"]


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
    critic_provider = RecordingProvider([{"score": 40, "issues": []}])
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
    assert actor_provider.max_tokens == [16_384]
    assert critic_provider.max_tokens == [16_384]
    assert actor_provider.reasoning_efforts == ["medium"]
    assert critic_provider.reasoning_efforts == ["medium"]


def test_discovery_uses_all_views_then_analysis_uses_only_evidence_views(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 45,
                "issues": [["left_wing", "H", 94, ["F", "P"], "intersects torso"]],
            },
            {
                "desc": "The wing disappears into the torso near its root.",
                "evidence": ["The front view has no visible separation."],
                "cause": "The root is too far inward.",
                "fix": "Move the root laterally while preserving attachment.",
                "criteria": ["A visible gap remains outside the attachment area."],
                "confidence": 91,
            },
        ]
    )
    images = [tmp_path / f"{view}.png" for view in ("front", "right", "top", "perspective")]
    for image in images:
        _write_render(image)

    critique = VisionCritic(provider, max_output_tokens=16_384, reasoning_effort="low").inspect(
        "create a dragon", images
    )

    discovery_content = provider.calls[0][1]["content"]
    analysis_content = provider.calls[1][1]["content"]
    assert isinstance(discovery_content, list)
    assert isinstance(analysis_content, list)
    assert sum(part["type"] == "image_url" for part in discovery_content) == 4
    assert sum(part["type"] == "image_url" for part in analysis_content) == 2
    analysis_context = next(
        json.loads(part["text"])
        for part in analysis_content
        if part["type"] == "text" and part["text"].startswith("{")
    )
    assert analysis_context["issue"] == {
        "region": "left_wing",
        "severity": "H",
        "views": ["F", "P"],
        "observation": "intersects torso",
    }
    assert "id" not in analysis_context["issue"]
    assert provider.max_tokens == [16_384, 16_384]
    assert provider.reasoning_efforts == ["low", "low"]
    assert critique.issues[0].id == "issue-001"
    assert critique.issues[0].title == "Left Wing: intersects torso"
    assert critique.issues[0].observation == "intersects torso"
    assert critique.issues[0].confidence == 0.94
    assert critique.issues[0].detail_status == "detailed"
    assert critique.issues[0].suggested_correction is not None


def test_issue_analysis_failure_preserves_the_discovery_observation(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "desc": "Unrelated detail.",
                "evidence": ["A view."],
                "cause": None,
                "fix": "Do something.",
                "criteria": ["Something changes."],
                "confidence": 50,
                "id": "wrong-issue",
            }
        ]
    )
    image = tmp_path / "front.png"
    _write_render(image)
    summary = VisualIssueSummary.model_validate(
        {
            "id": "issue-001",
            "title": "Missing tail",
            "region": "tail",
            "severity": "medium",
            "confidence": 0.8,
            "evidence_views": ["front"],
            "observation": "missing tail",
        }
    )

    with pytest.raises(
        ModelResponseError, match="compact visual-issue detail wire schema"
    ) as error:
        VisionCritic(provider).analyze_issue("create a dragon", summary, [image])

    discovery = VisualIssueDiscovery(score=0.5, summary="Tail needs work.", issues=[summary])
    critique = VisionCritic.assemble_critique(discovery, {}, {summary.id: str(error.value)})
    issue = critique.issues[0]
    assert issue.id == summary.id
    assert issue.title == summary.title
    assert issue.region == summary.region
    assert issue.severity == summary.severity
    assert issue.evidence_views == summary.evidence_views
    assert issue.detail_status == "analysis_failed"
    assert issue.analysis_failure is not None


def test_focused_analysis_request_budget_leaves_remaining_summaries_intact(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 40,
                "issues": [
                    ["body", "H", 90, ["F"], "first issue"],
                    ["tail", "M", 80, ["P"], "second issue"],
                ],
            },
            {
                "desc": "The first issue is visible.",
                "evidence": ["Visible in front."],
                "cause": None,
                "fix": "Correct the first issue.",
                "criteria": ["The first issue is absent in front."],
                "confidence": 90,
            },
        ]
    )
    images = [tmp_path / "front.png", tmp_path / "perspective.png"]
    for image in images:
        _write_render(image)

    critique = VisionCritic(provider, max_issue_analysis_requests=1).inspect(
        "create a creature", images
    )

    assert len(provider.calls) == 2
    assert critique.issues[0].detail_status == "detailed"
    assert critique.issues[1].detail_status == "summary_only"


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
