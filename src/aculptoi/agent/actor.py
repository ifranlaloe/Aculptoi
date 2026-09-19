"""Actor role: plan iterations, then propose typed actions for one work item."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from aculptoi.agent.prompts import (
    CONSTRUCTION_PLAN_PROMPT_VERSION,
    CONSTRUCTION_PLAN_SYSTEM_PROMPT,
    TARGET_BRIEF_PROMPT_VERSION,
    TARGET_BRIEF_SYSTEM_PROMPT,
    WORK_ITEM_PROMPT_VERSION,
    WORK_ITEM_SYSTEM_PROMPT,
)
from aculptoi.inference import InferenceProfile
from aculptoi.models import ModelUsage, complete_json_with_usage
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.construction import (
    MAX_COMPLETION_CRITERIA,
    MIN_COMPLETION_CRITERIA,
    ConstructionItem,
    ConstructionPlan,
    WorkItemActionBatch,
    work_item_response_requirements,
)
from aculptoi.schemas.critique import VisualCritique
from aculptoi.schemas.execution import (
    ProposalValidationErrorDetail,
    ProposalValidationFeedback,
)
from aculptoi.schemas.target import TargetBrief
from aculptoi.schemas.viewport import ViewportObservation


class WorkItemProposalValidationError(ModelResponseError):
    """A typed Actor proposal was invalid before it could mutate Blender."""

    def __init__(
        self,
        message: str,
        raw_response: str,
        feedback: ProposalValidationFeedback,
    ) -> None:
        super().__init__(message, raw_response)
        self.feedback = feedback


class Actor:
    """Create an iteration plan and execute it through bounded work-item responses."""

    def __init__(
        self,
        provider: ModelProvider,
        max_output_tokens: int = 16_384,
        thinking: bool = True,
        reasoning_effort: ReasoningEffort | None = "medium",
        provider_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._profile = InferenceProfile(
            max_output_tokens=max_output_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        self._provider_name = provider_name

    @property
    def inference_profile(self) -> InferenceProfile:
        """Return the immutable model settings used for all Actor requests."""
        return self._profile

    @property
    def provider_name(self) -> str | None:
        """Return the configured provider identifier when the runtime supplies one."""
        return self._provider_name

    def plan_iteration(
        self,
        goal: str,
        scene: dict[str, Any],
        previous_critique: VisualCritique | None,
        *,
        iteration: int,
        max_actor_requests: int,
        max_actions: int,
        modeling_context: Mapping[str, object] | None = None,
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
                modeling_context=modeling_context,
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
        modeling_context: Mapping[str, object] | None = None,
    ) -> list[Message]:
        """Build the first Actor request for one visual-refinement iteration."""
        context: dict[str, object] = {
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
        if modeling_context is not None:
            context.update(modeling_context)
        return [
            {"role": "system", "content": CONSTRUCTION_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, sort_keys=True)},
        ]

    def construction_plan_request_artifact(
        self,
        messages: Sequence[Message],
        *,
        knowledge: Sequence[dict[str, object]] = (),
    ) -> dict[str, object]:
        """Return an inspectable representation of a construction-plan request."""
        return self._request_artifact(
            messages,
            request_type="construction_plan",
            prompt_version=CONSTRUCTION_PLAN_PROMPT_VERSION,
            knowledge=knowledge,
        )

    def build_target_brief_messages(self, goal: str) -> list[Message]:
        """Build the one-time Actor request that interprets, but never replaces, a goal."""
        return [
            {"role": "system", "content": TARGET_BRIEF_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"goal": goal}, sort_keys=True)},
        ]

    def target_brief_request_artifact(self, messages: Sequence[Message]) -> dict[str, object]:
        """Return the durable request evidence for one target-brief derivation."""
        return self._request_artifact(
            messages,
            request_type="target_brief",
            prompt_version=TARGET_BRIEF_PROMPT_VERSION,
        )

    def derive_target_brief_messages(
        self,
        messages: Sequence[Message],
        *,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> TargetBrief:
        """Request and validate the bounded immutable interpretation aid."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._profile.max_output_tokens,
            thinking=self._profile.thinking,
            reasoning_effort=self._profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
        try:
            return TargetBrief.model_validate(response)
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the target-brief schema",
                self._raw_response(response),
            ) from error

    def plan_iteration_messages(
        self,
        messages: Sequence[Message],
        *,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> ConstructionPlan:
        """Request and validate a previously constructed construction-plan request."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._profile.max_output_tokens,
            thinking=self._profile.thinking,
            reasoning_effort=self._profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
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
        modeling_context: Mapping[str, object] | None = None,
        viewport_observation: ViewportObservation | None = None,
        viewport_image_data_url: str | None = None,
        recent_proposal_validation: dict[str, object] | None = None,
    ) -> WorkItemActionBatch:
        """Request and validate one Modeling Step, observation request, or completion."""
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
                modeling_context=modeling_context,
                viewport_observation=viewport_observation,
                viewport_image_data_url=viewport_image_data_url,
                recent_proposal_validation=recent_proposal_validation,
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
        modeling_context: Mapping[str, object] | None = None,
        viewport_observation: ViewportObservation | None = None,
        viewport_image_data_url: str | None = None,
        recent_proposal_validation: dict[str, object] | None = None,
    ) -> list[Message]:
        """Build a stateless request for one semantic turn of one construction item."""
        context: dict[str, object] = {
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
            "completion_criteria": (
                list(completion_criteria) if completion_criteria is not None else None
            ),
            "modeling_step": action_batch,
            "recent_execution": recent_execution,
            "recent_proposal_validation": recent_proposal_validation,
            "actor_viewport_available": bool(
                viewport_observation is not None and viewport_observation.available
            ),
            "viewport_observation": viewport_observation.model_dump(mode="json")
            if viewport_observation is not None
            else None,
            "remaining_safety_budget": {
                "actor_requests": remaining_actor_requests,
                "actions": remaining_actions,
            },
        }
        if modeling_context is not None:
            context.update(modeling_context)
        # This must reflect harness-owned item state, never optional modeling context.
        context["response_requirements"] = work_item_response_requirements(
            completion_criteria_established=completion_criteria is not None
        )
        user_content: object = json.dumps(context, sort_keys=True)
        if viewport_image_data_url is not None:
            if not viewport_image_data_url.startswith("data:image/png;base64,"):
                raise ValueError("viewport_image_data_url must be an in-memory PNG data URL")
            user_content = [
                {"type": "text", "text": json.dumps(context, sort_keys=True)},
                {"type": "text", "text": "Current transient Actor viewport observation."},
                {"type": "image_url", "image_url": {"url": viewport_image_data_url}},
            ]
        return [
            {"role": "system", "content": WORK_ITEM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def work_item_request_artifact(
        self,
        messages: Sequence[Message],
        *,
        knowledge: Sequence[dict[str, object]] = (),
    ) -> dict[str, object]:
        """Return an inspectable representation of a work-item action request."""
        return self._request_artifact(
            messages,
            request_type="work_item_actions",
            prompt_version=WORK_ITEM_PROMPT_VERSION,
            knowledge=knowledge,
        )

    def execute_work_item_messages(
        self,
        messages: Sequence[Message],
        *,
        expected_work_item_id: str,
        require_completion_criteria: bool,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> WorkItemActionBatch:
        """Request and validate a previously constructed work-item action request."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._profile.max_output_tokens,
            thinking=self._profile.thinking,
            reasoning_effort=self._profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
        raw_response = self._raw_response(response)
        try:
            action_batch = WorkItemActionBatch.model_validate(response)
        except ValidationError as error:
            raise WorkItemProposalValidationError(
                "Model response did not satisfy the work-item action schema",
                raw_response,
                self._proposal_validation_feedback(error),
            ) from error
        if action_batch.work_item_id != expected_work_item_id:
            raise self._proposal_validation_error(
                "Model response targeted a different construction work item",
                raw_response,
                location=["work_item_id"],
                code="unexpected_work_item",
            )
        response_requirements = work_item_response_requirements(
            completion_criteria_established=not require_completion_criteria
        )
        criteria_requirement = response_requirements["completion_criteria"]
        if criteria_requirement["required"] and action_batch.completion_criteria is None:
            raise self._proposal_validation_error(
                "completion_criteria is required on the first response and must be an array "
                f"of {MIN_COMPLETION_CRITERIA} to {MAX_COMPLETION_CRITERIA} strings",
                raw_response,
                location=["completion_criteria"],
                code="missing_completion_criteria",
            )
        if (
            criteria_requirement.get("must_be_omitted")
            and action_batch.completion_criteria is not None
        ):
            raise self._proposal_validation_error(
                "completion_criteria is immutable and must be omitted after the first response",
                raw_response,
                location=["completion_criteria"],
                code="immutable_completion_criteria",
            )
        return action_batch

    @staticmethod
    def _proposal_validation_error(
        message: str,
        raw_response: str,
        *,
        location: list[str | int],
        code: str,
    ) -> WorkItemProposalValidationError:
        """Create bounded semantic feedback for non-Pydantic response checks."""
        feedback = ProposalValidationFeedback(
            errors=[
                ProposalValidationErrorDetail(
                    location=location,
                    code=code,
                    message=message,
                )
            ]
        )
        return WorkItemProposalValidationError(message, raw_response, feedback)

    @staticmethod
    def _proposal_validation_feedback(error: ValidationError) -> ProposalValidationFeedback:
        """Translate Pydantic details into compact safe context, never a raw trace."""
        details: list[ProposalValidationErrorDetail] = []
        for detail in error.errors(include_url=False)[:5]:
            raw_location = detail.get("loc", ())
            location = [item if isinstance(item, int) else str(item)[:100] for item in raw_location]
            if not location:
                location = ["response"]
            raw_code = str(detail.get("type", "validation_error"))
            code = re.sub(r"[^a-z0-9_]", "_", raw_code.lower()).strip("_")
            if not code or not code[0].isalpha():
                code = "validation_error"
            message = str(detail.get("msg", "invalid proposal"))[:300]
            if location == ["completion_criteria"] and code == "list_type":
                message = (
                    "completion_criteria must be an array of "
                    f"{MIN_COMPLETION_CRITERIA} to {MAX_COMPLETION_CRITERIA} strings"
                )
            details.append(
                ProposalValidationErrorDetail(
                    location=location,
                    code=code[:100],
                    message=message,
                )
            )
        if not details:
            details.append(
                ProposalValidationErrorDetail(
                    location=["response"],
                    code="validation_error",
                    message="work-item proposal did not satisfy the schema",
                )
            )
        return ProposalValidationFeedback(errors=details)

    def _request_artifact(
        self,
        messages: Sequence[Message],
        *,
        request_type: str,
        prompt_version: str,
        knowledge: Sequence[dict[str, object]] = (),
    ) -> dict[str, object]:
        artifact: dict[str, object] = {
            "role": "actor",
            "request_type": request_type,
            "prompt_version": prompt_version,
            "thinking": self._profile.thinking,
            "max_output_tokens": self._profile.max_output_tokens,
            "reasoning_effort": self._profile.reasoning_effort,
            "messages": self._artifact_messages(messages),
        }
        if knowledge:
            artifact["knowledge"] = list(knowledge)
        return artifact

    @staticmethod
    def _artifact_messages(messages: Sequence[Message]) -> list[Message]:
        """Keep durable prompt artifacts useful without storing transient image payloads."""
        redacted: list[Message] = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                redacted.append({"role": message["role"], "content": content})
                continue
            artifact_content: list[dict[str, object]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    artifact_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": "<transient-viewport-image-omitted>"},
                        }
                    )
                else:
                    artifact_content.append(dict(part))
            redacted.append({"role": message["role"], "content": artifact_content})
        return redacted

    @staticmethod
    def _raw_response(response: object) -> str:
        return json.dumps(response, indent=2, sort_keys=True, default=str)
