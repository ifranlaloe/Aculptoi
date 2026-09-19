from __future__ import annotations

import pytest
from pydantic import ValidationError

from aculptoi.schemas.actions import ObjectScale, parse_action
from aculptoi.schemas.construction import ConstructionItem, ConstructionPlan, WorkItemActionBatch


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
