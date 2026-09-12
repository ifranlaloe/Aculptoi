"""Actor role: plan iterations, then propose typed actions for one work item."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from aculptoi.agent.prompts import (
    CONSTRUCTION_PLAN_PROMPT_VERSION,
    CONSTRUCTION_PLAN_SYSTEM_PROMPT,
    WORK_ITEM_PROMPT_VERSION,
    WORK_ITEM_SYSTEM_PROMPT,
)
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.schemas.construction import (
    ConstructionItem,
    ConstructionPlan,
    WorkItemActionBatch,
)
from aculptoi.schemas.critique import VisualCritique


class Actor:
    """Create an iteration plan and execute it through bounded work-item responses."""

    def __init__(self, provider: ModelProvider, max_output_tokens: int = 1536) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    def plan_iteration(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        *,
        iteration: int,
        max_actor_requests: int,
        max_actions: int,
    ) -> ConstructionPlan:
        """Request and validate the immutable construction plan for one iteration."""
        return self.plan_iteration_messages(
            self.build_construction_plan_messages(
                goal,
                scene,
                previous_critique,
                iteration=iteration,
                max_actor_requests=max_actor_requests,
                max_actions=max_actions,
            )
        )

    def build_construction_plan_messages(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        *,
        iteration: int,
        max_actor_requests: int,
        max_actions: int,
    ) -> list[Message]:
        """Build the first Actor request for one visual-refinement iteration."""
        context = {
            "goal": goal,
            "scene": scene,
            "latest_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
            "iteration": iteration,
            "safety_budget": {
                "max_actor_requests": max_actor_requests,
                "max_actions": max_actions,
            },
        }
        return [
            {"role": "system", "content": CONSTRUCTION_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, sort_keys=True)},
        ]

    def construction_plan_request_artifact(self, messages: Sequence[Message]) -> dict[str, object]:
        """Return an inspectable representation of a construction-plan request."""
        return self._request_artifact(
            messages,
            request_type="construction_plan",
            prompt_version=CONSTRUCTION_PLAN_PROMPT_VERSION,
        )

    def plan_iteration_messages(self, messages: Sequence[Message]) -> ConstructionPlan:
        """Request and validate a previously constructed construction-plan request."""
        response = self._provider.complete_json(messages, max_tokens=self._max_output_tokens)
        try:
            return ConstructionPlan.model_validate(response)
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the construction-plan schema",
                self._raw_response(response),
            ) from error

    def execute_work_item(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        construction_plan: ConstructionPlan,
        work_item: ConstructionItem,
        *,
        iteration: int,
        action_batch: int,
        completed_work_item_ids: Sequence[str],
        completed_work_items: Sequence[dict[str, object]],
        completion_criteria: Sequence[str] | None,
        remaining_actor_requests: int,
        remaining_actions: int,
        recent_execution: dict[str, object] | None,
    ) -> WorkItemActionBatch:
        """Request and validate one action batch for the active work item."""
        return self.execute_work_item_messages(
            self.build_work_item_messages(
                goal,
                scene,
                previous_critique,
                construction_plan,
                work_item,
                iteration=iteration,
                action_batch=action_batch,
                completed_work_item_ids=completed_work_item_ids,
                completed_work_items=completed_work_items,
                completion_criteria=completion_criteria,
                remaining_actor_requests=remaining_actor_requests,
                remaining_actions=remaining_actions,
                recent_execution=recent_execution,
            ),
            expected_work_item_id=work_item.id,
            require_completion_criteria=completion_criteria is None,
        )

    def build_work_item_messages(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        construction_plan: ConstructionPlan,
        work_item: ConstructionItem,
        *,
        iteration: int,
        action_batch: int,
        completed_work_item_ids: Sequence[str],
        completed_work_items: Sequence[dict[str, object]],
        completion_criteria: Sequence[str] | None,
        remaining_actor_requests: int,
        remaining_actions: int,
        recent_execution: dict[str, object] | None,
    ) -> list[Message]:
        """Build a stateless request for one action batch of one construction item."""
        context = {
            "goal": goal,
            "scene": scene,
            "latest_critique": previous_critique.model_dump(mode="json")
            if previous_critique
            else None,
            "iteration": iteration,
            "construction_plan": construction_plan.model_dump(mode="json"),
            "active_work_item": work_item.model_dump(mode="json"),
            "completed_work_item_ids": list(completed_work_item_ids),
            "completed_work_items": list(completed_work_items),
            "completion_criteria": list(completion_criteria) if completion_criteria else None,
            "action_batch": action_batch,
            "recent_execution": recent_execution,
            "remaining_safety_budget": {
                "actor_requests": remaining_actor_requests,
                "actions": remaining_actions,
            },
        }
        return [
            {"role": "system", "content": WORK_ITEM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, sort_keys=True)},
        ]

    def work_item_request_artifact(self, messages: Sequence[Message]) -> dict[str, object]:
        """Return an inspectable representation of a work-item action request."""
        return self._request_artifact(
            messages,
            request_type="work_item_actions",
            prompt_version=WORK_ITEM_PROMPT_VERSION,
        )

    def execute_work_item_messages(
        self,
        messages: Sequence[Message],
        *,
        expected_work_item_id: str,
        require_completion_criteria: bool,
    ) -> WorkItemActionBatch:
        """Request and validate a previously constructed work-item action request."""
        response = self._provider.complete_json(messages, max_tokens=self._max_output_tokens)
        raw_response = self._raw_response(response)
        try:
            action_batch = WorkItemActionBatch.model_validate(response)
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the work-item action schema", raw_response
            ) from error
        if action_batch.work_item_id != expected_work_item_id:
            raise ModelResponseError(
                "Model response targeted a different construction work item", raw_response
            )
        if require_completion_criteria and action_batch.completion_criteria is None:
            raise ModelResponseError(
                "First work-item response must define completion criteria", raw_response
            )
        if not require_completion_criteria and action_batch.completion_criteria is not None:
            raise ModelResponseError(
                "Later work-item responses must not replace completion criteria", raw_response
            )
        return action_batch

    def _request_artifact(
        self,
        messages: Sequence[Message],
        *,
        request_type: str,
        prompt_version: str,
    ) -> dict[str, object]:
        return {
            "role": "actor",
            "request_type": request_type,
            "prompt_version": prompt_version,
            "max_output_tokens": self._max_output_tokens,
            "messages": list(messages),
        }

    @staticmethod
    def _raw_response(response: object) -> str:
        return json.dumps(response, indent=2, sort_keys=True, default=str)
