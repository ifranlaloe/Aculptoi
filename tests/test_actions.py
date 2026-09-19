from __future__ import annotations

import pytest
from pydantic import ValidationError

from aculptoi.schemas.actions import (
    MeshSubdivide,
    ObjectCreate,
    ObjectScale,
    action_catalog,
    parse_action,
)
from aculptoi.schemas.construction import (
    MAX_COMPLETION_CRITERIA,
    MIN_COMPLETION_CRITERIA,
    ConstructionItem,
    ConstructionPlan,
    WorkItemActionBatch,
    work_item_response_requirements,
)


def test_work_item_response_requirements_are_compact_and_state_derived() -> None:
    first = work_item_response_requirements(completion_criteria_established=False)
    later = work_item_response_requirements(completion_criteria_established=True)

    assert first == {
        "allowed_kinds": ["modeling_step", "observation_request", "complete"],
        "completion_criteria": {
            "required": True,
            "type": "array[string]",
            "min_items": MIN_COMPLETION_CRITERIA,
            "max_items": MAX_COMPLETION_CRITERIA,
        },
    }
    assert later == {
        "allowed_kinds": ["modeling_step", "observation_request", "complete"],
        "completion_criteria": {"required": False, "must_be_omitted": True},
    }


def test_valid_work_item_batch_uses_discriminated_action_types() -> None:
    batch = WorkItemActionBatch.model_validate(
        {
            "work_item_id": "dragon-body",
            "status": "complete",
            "reason": "Make the body more substantial.",
            "completion_criteria": ["DragonBody has the target proportions."],
            "actions": [
                {
                    "command": "object.scale",
                    "object": "DragonBody",
                    "scale": [1.15, 0.95, 1.05],
                }
            ],
        }
    )

    assert isinstance(batch.actions[0], ObjectScale)
    assert batch.actions[0].scale == (1.15, 0.95, 1.05)
    assert batch.completion_criteria == ["DragonBody has the target proportions."]


def test_continuing_work_item_requires_an_action() -> None:
    with pytest.raises(ValidationError, match="must include at least one action"):
        WorkItemActionBatch.model_validate(
            {
                "work_item_id": "dragon-body",
                "status": "continue",
                "reason": "More work remains.",
                "actions": [],
            }
        )


def test_complete_work_item_can_require_no_mutation() -> None:
    batch = WorkItemActionBatch.model_validate(
        {
            "work_item_id": "existing-body",
            "status": "complete",
            "reason": "The current scene already satisfies this item.",
            "actions": [],
        }
    )

    assert batch.actions == []


def test_modeling_step_wire_contract_requires_one_intent_and_actions() -> None:
    step = WorkItemActionBatch.model_validate(
        {
            "kind": "modeling_step",
            "work_item_id": "dragon-body",
            "reason": "Establish primary mass.",
            "intent": "Create one elongated body mass.",
            "actions": [{"command": "object.create", "name": "DragonBody"}],
        }
    )

    assert step.kind == "modeling_step"
    assert step.status == "continue"
    assert step.intent == "Create one elongated body mass."
    with pytest.raises(ValidationError, match="must include one semantic intent"):
        WorkItemActionBatch.model_validate(
            {
                "kind": "modeling_step",
                "work_item_id": "dragon-body",
                "reason": "Missing semantic intent.",
                "actions": [{"command": "object.create", "name": "DragonBody"}],
            }
        )


def test_object_create_legacy_default_is_isolated_from_actor_catalog_requirements() -> None:
    """Historic artifacts may omit a primitive; newly proposed Actor actions may not."""
    legacy = parse_action({"command": "object.create", "name": "LegacyBody"})
    create_catalog = next(
        entry for entry in action_catalog() if entry["command"] == "object.create"
    )

    assert isinstance(legacy, ObjectCreate)
    assert legacy.primitive == "cube"
    assert "primitive" in create_catalog["required_fields"]


def test_observation_and_complete_variants_cannot_carry_mutating_actions() -> None:
    observation = WorkItemActionBatch.model_validate(
        {
            "kind": "observation_request",
            "work_item_id": "dragon-body",
            "reason": "Need a top view.",
            "view": {"orientation": "top", "framing": "close"},
        }
    )
    complete = WorkItemActionBatch.model_validate(
        {
            "kind": "complete",
            "work_item_id": "dragon-body",
            "reason": "The observed body is complete.",
        }
    )

    assert observation.kind == "observation_request"
    assert observation.actions == []
    assert complete.kind == "complete"
    assert complete.status == "complete"
    with pytest.raises(ValidationError, match="cannot include mutating actions"):
        WorkItemActionBatch.model_validate(
            {
                "kind": "complete",
                "work_item_id": "dragon-body",
                "reason": "Invalid completion.",
                "actions": [{"command": "object.create", "name": "DragonBody"}],
            }
        )


def test_construction_plan_requires_unique_ordered_dependencies() -> None:
    plan = ConstructionPlan.model_validate(
        {
            "reason": "Build the cube in two logical layers.",
            "items": [
                {
                    "id": "base-layer",
                    "title": "Base layer",
                    "objective": "Create the lower cubies.",
                    "depends_on": [],
                },
                {
                    "id": "upper-layer",
                    "title": "Upper layer",
                    "objective": "Create the upper cubies.",
                    "depends_on": ["base-layer"],
                },
            ],
        }
    )

    assert [item.id for item in plan.items] == ["base-layer", "upper-layer"]
    assert [item.form_traits for item in plan.items] == [[], []]

    with pytest.raises(ValidationError, match="must refer to earlier items"):
        ConstructionPlan.model_validate(
            {
                "reason": "Invalid ordering.",
                "items": [
                    {
                        "id": "upper-layer",
                        "title": "Upper layer",
                        "objective": "Create the upper cubies.",
                        "depends_on": ["base-layer"],
                    },
                    {
                        "id": "base-layer",
                        "title": "Base layer",
                        "objective": "Create the lower cubies.",
                        "depends_on": [],
                    },
                ],
            }
        )


def test_construction_item_form_traits_are_optional_bounded_and_transferable() -> None:
    legacy_item = ConstructionItem.model_validate(
        {
            "id": "body",
            "title": "Body",
            "objective": "Create a readable body.",
            "depends_on": [],
        }
    )
    item = ConstructionItem.model_validate(
        {
            "id": "paired-appendages",
            "title": "Paired appendages",
            "objective": "Create mirrored thin appendages.",
            "depends_on": [],
            "form_traits": ["appendages", "bilateral_symmetry", "thin_features"],
        }
    )

    assert legacy_item.form_traits == []
    assert item.form_traits == ["appendages", "bilateral_symmetry", "thin_features"]

    with pytest.raises(ValidationError, match="work-item form_traits must be unique"):
        ConstructionItem.model_validate(
            {
                "id": "body",
                "title": "Body",
                "objective": "Create a readable body.",
                "depends_on": [],
                "form_traits": ["organic", "organic"],
            }
        )
    with pytest.raises(ValidationError):
        ConstructionItem.model_validate(
            {
                "id": "fish-body",
                "title": "Fish body",
                "objective": "Create a readable body.",
                "depends_on": [],
                "form_traits": ["fish"],
            }
        )


def test_construction_plan_remains_descriptive() -> None:
    with pytest.raises(ValidationError):
        ConstructionPlan.model_validate(
            {
                "reason": "Build a cube.",
                "items": [
                    {
                        "id": "base-cubie",
                        "title": "Base cubie",
                        "objective": "Create the base cubie.",
                        "depends_on": [],
                        "completion_criteria": ["The base cubie exists."],
                    }
                ],
            }
        )


def test_unsupported_action_is_explicitly_rejected() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        parse_action({"command": "execute_bpy", "code": "bpy.ops.wm.save_as_mainfile()"})


@pytest.mark.parametrize("scale", ([0, 1, 1], [-1, 1, 1], [101, 1, 1]))
def test_unsafe_scale_is_rejected(scale: list[float]) -> None:
    with pytest.raises(ValidationError):
        parse_action({"command": "object.scale", "object": "Cube", "scale": scale})


def test_names_cannot_contain_paths() -> None:
    with pytest.raises(ValidationError):
        parse_action({"command": "object.create", "name": "../../unsafe", "primitive": "cube"})


@pytest.mark.parametrize("cuts", [1, 3])
def test_mesh_subdivide_accepts_only_bounded_cut_counts(cuts: int) -> None:
    action = parse_action({"command": "mesh.subdivide", "object": "Body", "cuts": cuts})

    assert isinstance(action, MeshSubdivide)
    assert action.cuts == cuts


@pytest.mark.parametrize(
    "payload",
    [
        {"command": "mesh.subdivide", "object": "Body", "cuts": 0},
        {"command": "mesh.subdivide", "object": "Body", "cuts": 4},
        {"command": "mesh.subdivide", "object": "Body"},
        {"command": "mesh.subdivide", "object": "Body", "cuts": 1, "extra": True},
    ],
)
def test_mesh_subdivide_rejects_invalid_or_extra_payload_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_action(payload)
