"""Actor/planner role: it proposes typed actions but cannot execute them."""

from __future__ import annotations

import json
from typing import Any

from aculptoi.models.base import Message, ModelProvider
from aculptoi.schemas.actions import ActionPlan
from aculptoi.schemas.critique import VisualCritique


class Actor:
    """Generate a constrained plan from goal, scene state, and prior critique."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    def plan(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
    ) -> ActionPlan:
        """Request JSON and validate it at the trust boundary."""
        context = {
            "goal": goal,
            "scene": scene,
            "previous_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
        }
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are the planning component for a local Blender agent. "
                    "Return only a JSON object with `reason` and `actions`. "
                    "Actions must use only: object.create (cube|uv_sphere|cylinder|cone), "
                    "object.delete, object.translate, object.rotate, object.scale, "
                    "sculpt.voxel_remesh. Never produce shell commands, Python, bpy code, "
                    "filesystem paths, or unsupported commands. Prefer a small reversible plan."
                ),
            },
            {"role": "user", "content": json.dumps(context)},
        ]
        return ActionPlan.model_validate(self._provider.complete_json(messages))
