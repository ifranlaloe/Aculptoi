from __future__ import annotations

import pytest
from pydantic import ValidationError

from aculptoi.schemas.actions import ObjectScale, parse_action
from aculptoi.schemas.construction import ConstructionPlan, WorkItemActionBatch


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
