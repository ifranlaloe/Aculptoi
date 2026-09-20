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
    VisualIssueDetailWire,
    VisualIssueDiscovery,
    VisualIssueDiscoveryWire,
    VisualIssueSummary,
)
from aculptoi.schemas.inspection import (
    AtlasLayout,
    InspectionAtlasManifest,
    InspectionAtlasTile,
    InspectionBounds,
    InspectionFraming,
)


class RecordingProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []
        self.max_tokens: list[int | None] = []
        self.thinking: list[bool | None] = []
        self.reasoning_efforts: list[ReasoningEffort | None] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        self.thinking.append(thinking)
        self.reasoning_efforts.append(reasoning_effort)
        return self._responses.pop(0)


class SchemaRecordingProvider(RecordingProvider):
    """Test provider that opts in to the generic structured-output capability."""

    supports_json_schema = True

    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(responses)
        self.response_schemas: list[dict[str, object] | None] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, object] | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        self.response_schemas.append(response_schema)
        return super().complete_json(
            messages,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )


def _write_atlas(path: Path, size: tuple[int, int] = (64, 64)) -> bytes:
    Image.new("RGB", size, color="white").save(path)
    return path.read_bytes()


def _manifest() -> InspectionAtlasManifest:
    layout = AtlasLayout(columns=2, rows=2, tile_dimension=32, width=64, height=64)
    bounds = InspectionBounds(
        minimum=(-1.0, -1.0, -1.0),
        maximum=(1.0, 1.0, 1.0),
        center=(0.0, 0.0, 0.0),
        radius=1.8,
    )
    common = {
        "source": "shots/source.png",
        "projection": "perspective",
        "selection_kind": "dynamic",
        "selection_reason": "new_surface_coverage",
        "coverage_gain": 0.1,
        "information_gain": 0.2,
    }
    return InspectionAtlasManifest(
        sensor_version="inspection-atlas-v1",
        lighting_rig="neutral-studio-v1",
        width=64,
        height=64,
        layout=layout,
        bounds=bounds,
        framing=InspectionFraming(margin=1.15, distance=5.0, orthographic_scale=4.0),
        estimated_surface_coverage=0.9,
        tiles={
            "A1": InspectionAtlasTile(
                tile_id="A1",
                camera_id="anchor-front",
                pixel_bounds=(0, 0, 32, 32),
                azimuth_degrees=0,
                elevation_degrees=0,
                orientation="front",
                **common,
            ),
            "A2": InspectionAtlasTile(
                tile_id="A2",
                camera_id="anchor-right",
                pixel_bounds=(32, 0, 32, 32),
                azimuth_degrees=90,
                elevation_degrees=0,
                orientation="right",
                **common,
            ),
            "B1": InspectionAtlasTile(
                tile_id="B1",
                camera_id="anchor-rear",
                pixel_bounds=(0, 32, 32, 32),
                azimuth_degrees=180,
                elevation_degrees=0,
                orientation="rear",
                **common,
            ),
            "B2": InspectionAtlasTile(
                tile_id="B2",
                camera_id="anchor-front-upper",
                pixel_bounds=(32, 32, 32, 32),
                azimuth_degrees=0,
                elevation_degrees=35,
                orientation="front-upper",
                **common,
            ),
        },
    )


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
                    "evidence_tiles": ["A2", "B2"],
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


def test_compact_discovery_wire_expands_percentages_and_deterministic_ids() -> None:
    discovery = VisualIssueDiscoveryWire.model_validate(
        {
            "score": 58,
            "issues": [
                ["left_wing", "C", 97, ["A1", "B2"], "intersects torso"],
                ["neck", "H", 95, ["A2", "B2"], "too short relative to torso"],
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
        "evidence_tiles": ["A1", "B2"],
        "observation": "intersects torso",
    }
    assert second.id == "issue-002"
    assert second.evidence_tiles == ["A2", "B2"]
    assert second.confidence == 0.95


@pytest.mark.parametrize(
    ("issues", "match"),
    [
        ([["wing", "H", 90, ["A1"]]], "Field required"),
        ([["wing", "X", 90, ["A1"], "intersects torso"]], "literal_error"),
        ([["wing", "H", 90, ["unknown"], "intersects torso"]], "string_pattern_mismatch"),
    ],
)
def test_compact_discovery_wire_rejects_malformed_tuples_and_unknown_tiles(
    issues: list[object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        VisualIssueDiscoveryWire.model_validate({"score": 58, "issues": issues})


def test_compact_detail_wire_receives_its_id_from_application_context() -> None:
    detail = VisualIssueDetailWire.model_validate(
        {
            "desc": "Wing penetrates upper torso near the shoulder.",
            "evidence": ["A1: contour disappears into torso"],
            "cause": "wing root too low and inward",
            "fix": "move root upward and outward",
            "criteria": ["no penetration outside attachment"],
            "confidence": 96,
        }
    ).to_domain("issue-007")

    assert detail.id == "issue-007"
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
                evidence_tiles=["A1", "B2"],
                observation="intersects torso",
                detail_status="detailed",
                description="The wing penetrates the upper torso near its root.",
                evidence=["A1: contour disappears into torso"],
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
    assert actor_issue["evidence_tiles"] == ["A1", "B2"]
    assert "desc" not in actor_issue


def test_discovery_issue_limit_still_applies_to_compact_wire(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 50,
                "issues": [
                    ["body", "H", 90, ["A1"], "too small"],
                    ["tail", "M", 80, ["B2"], "missing"],
                ],
            }
        ]
    )
    atlas = tmp_path / "atlas.png"
    _write_atlas(atlas)

    with pytest.raises(ModelResponseError, match="max_discovered_issues"):
        VisionCritic(provider, max_discovered_issues=1).discover(
            "create a creature", atlas, _manifest()
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
            {"score": 70, "issues": []},
        ]
    )
    atlas = tmp_path / "atlas.png"
    original = _write_atlas(atlas, (1600, 800))

    plan = Actor(provider).plan_iteration(
        "create a creature",
        {"objects": []},
        None,
        iteration=1,
        max_actor_requests=100,
        max_actions=1000,
    )
    critique = VisionCritic(provider, max_image_dimension=1000).inspect(
        "create a creature", atlas, _manifest()
    )

    actor_messages, critic_messages = provider.calls
    actor_content = actor_messages[1]["content"]
    critic_content = critic_messages[1]["content"]
    assert plan.items[0].id == "body"
    assert critique.score == 0.7
    assert isinstance(actor_content, str)
    assert isinstance(critic_content, list)
    assert provider.max_tokens == [16_384, 4_096]
    assert provider.thinking == [True, False]
    assert provider.reasoning_efforts == ["medium", None]
    image_url = next(
        part["image_url"]["url"] for part in critic_content if part["type"] == "image_url"
    )
    prepared = Image.open(BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1])))
    assert prepared.size == (1000, 500)
    assert atlas.read_bytes() == original


def test_actor_construction_and_work_item_requests_share_configured_settings() -> None:
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
    assert provider.thinking == [True, True]
    assert provider.reasoning_efforts == ["high", "high"]
    artifact = actor.work_item_request_artifact(
        [{"role": "user", "content": "{}"}],
        completion_criteria_established=False,
    )
    assert artifact["structured_output"] == {"mode": "json_object"}


def test_actor_uses_schema_constrained_output_when_provider_supports_it() -> None:
    provider = SchemaRecordingProvider(
        [
            {
                "kind": "observation_request",
                "work_item_id": "integrate-tail-fin",
                "reason": "Need a side profile to judge the current transition.",
                "completion_criteria": ["Tail transition can be judged from both sides."],
                "view": {
                    "orientation": "left",
                    "projection": "orthographic",
                    "framing": "whole_subject",
                },
            }
        ]
    )
    actor = Actor(provider)
    messages: list[Message] = [
        {"role": "system", "content": "work item"},
        {"role": "user", "content": "{}"},
    ]

    batch = actor.execute_work_item_messages(
        messages,
        expected_work_item_id="integrate-tail-fin",
        require_completion_criteria=True,
    )

    assert batch.kind == "observation_request"
    assert len(provider.response_schemas) == 1
    schema = provider.response_schemas[0]
    assert isinstance(schema, dict)
    artifact = actor.work_item_request_artifact(
        messages,
        completion_criteria_established=False,
    )
    structured_output = artifact["structured_output"]
    assert isinstance(structured_output, dict)
    assert structured_output["mode"] == "json_schema"
    assert structured_output["schema_id"] == "work-item-first-response-v1"
    assert len(str(structured_output["schema_sha256"])) == 64


def test_focused_analysis_keeps_the_complete_atlas_context(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 45,
                "issues": [["left_wing", "H", 94, ["A1", "B2"], "intersects torso"]],
            },
            {
                "desc": "The wing disappears into the torso near its root.",
                "evidence": ["A1: no visible separation."],
                "cause": "The root is too far inward.",
                "fix": "Move the root laterally while preserving attachment.",
                "criteria": ["A visible gap remains outside the attachment area."],
                "confidence": 91,
            },
        ]
    )
    atlas = tmp_path / "atlas.png"
    _write_atlas(atlas)

    critique = VisionCritic(provider, reasoning_effort="low").inspect(
        "create a dragon", atlas, _manifest()
    )

    discovery_content = provider.calls[0][1]["content"]
    analysis_content = provider.calls[1][1]["content"]
    assert isinstance(discovery_content, list)
    assert isinstance(analysis_content, list)
    assert sum(part["type"] == "image_url" for part in discovery_content) == 1
    assert sum(part["type"] == "image_url" for part in analysis_content) == 1
    analysis_context = json.loads(analysis_content[0]["text"])
    assert analysis_context["issue"] == {
        "region": "left_wing",
        "severity": "H",
        "tiles": ["A1", "B2"],
        "observation": "intersects torso",
    }
    assert len(analysis_context["inspection_atlas"]["tiles"]) == 4
    assert critique.issues[0].evidence_tiles == ["A1", "B2"]
    assert critique.issues[0].detail_status == "detailed"
    assert provider.max_tokens == [16_384, 16_384]
    assert provider.thinking == [True, True]
    assert provider.reasoning_efforts == ["low", "low"]


def test_critic_defaults_separate_breadth_first_and_focused_profiles(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 45,
                "issues": [["left_wing", "H", 94, ["A1"], "intersects torso"]],
            },
            {
                "desc": "The wing disappears into the torso near its root.",
                "evidence": ["A1: no visible separation."],
                "cause": "The root is too far inward.",
                "fix": "Move the root laterally while preserving attachment.",
                "criteria": ["A visible gap remains outside the attachment area."],
                "confidence": 91,
            },
        ]
    )
    atlas = tmp_path / "atlas.png"
    _write_atlas(atlas)

    VisionCritic(provider).inspect("create a dragon", atlas, _manifest())

    assert provider.max_tokens == [4_096, 16_384]
    assert provider.thinking == [False, True]
    assert provider.reasoning_efforts == [None, "medium"]


def test_issue_analysis_failure_preserves_the_discovery_observation(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "desc": "Unrelated detail.",
                "evidence": ["A1: a view."],
                "cause": None,
                "fix": "Do something.",
                "criteria": ["Something changes."],
                "confidence": 50,
                "id": "wrong-issue",
            }
        ]
    )
    atlas = tmp_path / "atlas.png"
    _write_atlas(atlas)
    summary = VisualIssueSummary.model_validate(
        {
            "id": "issue-001",
            "title": "Missing tail",
            "region": "tail",
            "severity": "medium",
            "confidence": 0.8,
            "evidence_tiles": ["A1"],
            "observation": "missing tail",
        }
    )

    with pytest.raises(
        ModelResponseError, match="compact visual-issue detail wire schema"
    ) as error:
        VisionCritic(provider).analyze_issue("create a dragon", summary, atlas, _manifest())

    discovery = VisualIssueDiscovery(score=0.5, summary="Tail needs work.", issues=[summary])
    critique = VisionCritic.assemble_critique(discovery, {}, {summary.id: str(error.value)})
    assert critique.issues[0].evidence_tiles == ["A1"]
    assert critique.issues[0].detail_status == "analysis_failed"


def test_focused_analysis_request_budget_leaves_remaining_summaries_intact(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            {
                "score": 40,
                "issues": [
                    ["body", "H", 90, ["A1"], "first issue"],
                    ["tail", "M", 80, ["B2"], "second issue"],
                ],
            },
            {
                "desc": "The first issue is visible.",
                "evidence": ["A1: visible."],
                "cause": None,
                "fix": "Correct the first issue.",
                "criteria": ["The first issue is absent in A1."],
                "confidence": 90,
            },
        ]
    )
    atlas = tmp_path / "atlas.png"
    _write_atlas(atlas)

    critique = VisionCritic(provider, max_issue_analysis_requests=1).inspect(
        "create a creature", atlas, _manifest()
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

    with pytest.raises(ModelResponseError, match="work-item action schema"):
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


@pytest.mark.parametrize(
    ("response", "expected_kind"),
    [
        (
            {
                "kind": "modeling_step",
                "work_item_id": "body",
                "reason": "Establish the primary mass.",
                "intent": "Create one primary mass.",
                "completion_criteria": ["A body object exists."],
                "actions": [
                    {
                        "command": "object.create",
                        "name": "Body",
                        "primitive": "uv_sphere",
                    }
                ],
            },
            "modeling_step",
        ),
        (
            {
                "kind": "observation_request",
                "work_item_id": "body",
                "reason": "A right-side view is needed first.",
                "completion_criteria": ["The current body can be judged from the right."],
                "view": {"orientation": "right", "framing": "whole_subject"},
            },
            "observation_request",
        ),
        (
            {
                "kind": "complete",
                "work_item_id": "body",
                "reason": "The observed scene already satisfies this item.",
                "completion_criteria": ["The existing body meets the item objective."],
            },
            "complete",
        ),
    ],
)
def test_first_work_item_response_allows_each_kind_with_criteria(
    response: dict[str, object], expected_kind: str
) -> None:
    provider = RecordingProvider([response])
    plan = ConstructionPlan.model_validate(
        {
            "reason": "Build a body.",
            "items": [
                {
                    "id": "body",
                    "title": "Body",
                    "objective": "Create a readable body.",
                    "depends_on": [],
                }
            ],
        }
    )

    batch = Actor(provider).execute_work_item(
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
        remaining_actions=100,
        recent_execution=None,
    )

    assert batch.kind == expected_kind
    assert batch.completion_criteria


@pytest.mark.parametrize(
    ("action_batch", "criteria", "match"),
    [
        (
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create body.",
                "actions": [],
            },
            None,
            "completion_criteria is required",
        ),
        (
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Finish body.",
                "completion_criteria": ["Replacement."],
                "actions": [],
            },
            ["Body exists."],
            "completion_criteria is immutable",
        ),
    ],
)
def test_completion_criteria_ownership_remains_unchanged(
    action_batch: dict[str, object], criteria: list[str] | None, match: str
) -> None:
    provider = RecordingProvider([action_batch])
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

    with pytest.raises(ModelResponseError, match=match):
        Actor(provider).execute_work_item(
            "create a creature",
            {"objects": []},
            None,
            construction_plan,
            construction_plan.items[0],
            iteration=1,
            action_batch=1 if criteria is None else 2,
            completed_work_item_ids=[],
            completed_work_items=[],
            completion_criteria=criteria,
            remaining_actor_requests=98,
            remaining_actions=1000,
            recent_execution=None,
        )
