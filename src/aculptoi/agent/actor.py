"""Actor/planner role: it proposes typed actions but cannot execute them."""

from __future__ import annotations

import json
from typing import Any

from aculptoi.agent.prompts import ACTOR_SYSTEM_PROMPT
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
        *,
        iteration: int | None = None,
        recent_execution: dict[str, object] | None = None,
    ) -> ActionPlan:
        """Request JSON and validate it at the trust boundary."""
        context = {
            "goal": goal,
            "scene": scene,
            "latest_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
            "iteration": iteration,
            "recent_execution": recent_execution,
        }
        messages: list[Message] = [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, sort_keys=True)},
        ]
        return ActionPlan.model_validate(self._provider.complete_json(messages))
