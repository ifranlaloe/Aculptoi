from __future__ import annotations

import pytest
from pydantic import ValidationError

from aculptoi.schemas.actions import ActionPlan, ObjectScale, parse_action


def test_valid_action_plan_uses_discriminated_action_types() -> None:
    plan = ActionPlan.model_validate(
        {
            "reason": "Make the body more substantial.",
            "actions": [
                {
                    "command": "object.scale",
                    "object": "DragonBody",
                    "scale": [1.15, 0.95, 1.05],
                }
            ],
        }
    )

    assert isinstance(plan.actions[0], ObjectScale)
    assert plan.actions[0].scale == (1.15, 0.95, 1.05)
    assert plan.ready_for_inspection is True


def test_action_plan_can_request_another_construction_batch() -> None:
    plan = ActionPlan.model_validate(
        {
            "reason": "Continue construction before visual inspection.",
            "ready_for_inspection": False,
            "actions": [{"command": "object.create", "name": "Cubie", "primitive": "cube"}],
        }
    )

    assert plan.ready_for_inspection is False


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
