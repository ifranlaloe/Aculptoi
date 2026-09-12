"""Actor/planner role: it proposes typed actions but cannot execute them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from aculptoi.agent.prompts import ACTOR_PROMPT_VERSION, ACTOR_SYSTEM_PROMPT
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.schemas.actions import ActionPlan
from aculptoi.schemas.critique import VisualCritique


class Actor:
    """Generate a constrained plan from goal, scene state, and prior critique."""

    def __init__(self, provider: ModelProvider, max_output_tokens: int = 1536) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    def plan(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        *,
        iteration: int | None = None,
        recent_execution: dict[str, object] | None = None,
    ) -> ActionPlan:
        """Build an Actor request, then validate its model response at the boundary."""
        return self.plan_messages(
            self.build_messages(
                goal,
                scene,
                previous_critique,
                iteration=iteration,
                recent_execution=recent_execution,
            )
        )

    def build_messages(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        *,
        iteration: int | None = None,
        recent_execution: dict[str, object] | None = None,
    ) -> list[Message]:
        """Build the exact text-only request for one planning iteration."""
        context = {
            "goal": goal,
            "scene": scene,
            "latest_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
            "iteration": iteration,
            "recent_execution": recent_execution,
        }
        return [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, sort_keys=True)},
        ]

    def request_artifact(self, messages: Sequence[Message]) -> dict[str, object]:
        """Return an inspectable representation of an Actor request without execution data."""
        return {
            "role": "actor",
            "prompt_version": ACTOR_PROMPT_VERSION,
            "max_output_tokens": self._max_output_tokens,
            "messages": list(messages),
        }

    def plan_messages(self, messages: Sequence[Message]) -> ActionPlan:
        """Request and validate a previously constructed Actor message sequence."""
        response = self._provider.complete_json(messages, max_tokens=self._max_output_tokens)
        try:
            return ActionPlan.model_validate(response)
        except ValidationError as error:
            raw_response = json.dumps(response, indent=2, sort_keys=True, default=str)
            raise ModelResponseError(
                "Model response did not satisfy the action-plan schema", raw_response
            ) from error
