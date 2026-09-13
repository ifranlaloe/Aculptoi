"""Read-only technical assessment of whether an inspection atlas is usable."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import ValidationError

from aculptoi.agent.prompts import (
    INSPECTION_REVIEW_PROMPT_VERSION,
    INSPECTION_REVIEW_SYSTEM_PROMPT,
)
from aculptoi.inference import InferenceProfile
from aculptoi.models import ModelUsage, complete_json_with_usage
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.inspection import (
    InspectionAtlasManifest,
    InspectionReview,
    InspectionReviewWire,
)
from aculptoi.vision import prepare_render


class InspectionReviewer:
    """Review observation quality only; this role cannot request scene mutations."""

    def __init__(
        self,
        provider: ModelProvider,
        max_image_dimension: int = 4096,
        max_output_tokens: int = 4096,
        reasoning_effort: ReasoningEffort = "low",
        provider_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._max_image_dimension = max_image_dimension
        self._profile = InferenceProfile(
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._provider_name = provider_name

    @property
    def inference_profile(self) -> InferenceProfile:
        """Return the immutable model settings for technical atlas review."""
        return self._profile

    @property
    def provider_name(self) -> str | None:
        """Return the configured provider identifier when the runtime supplies one."""
        return self._provider_name

    def review(self, atlas: Path, manifest: InspectionAtlasManifest) -> InspectionReview:
        """Request a compact technical verdict for one full-resolution inspection atlas."""
        messages, _ = self.build_review_request(atlas, manifest)
        return self.review_messages(messages, manifest)

    def build_review_request(
        self, atlas: Path, manifest: InspectionAtlasManifest
    ) -> tuple[list[Message], dict[str, object]]:
        """Build one atlas-only model request plus a data-URL-free durable artifact."""
        prepared = prepare_render(atlas, self._max_image_dimension)
        context = {
            "task": "technical inspection survey review",
            "atlas": manifest.to_reviewer_context(),
        }
        content: list[dict[str, object]] = [
            {"type": "text", "text": json.dumps(context, sort_keys=True)},
            {
                "type": "text",
                "text": (
                    f"Inspection atlas ({prepared.width}x{prepared.height} pixels). "
                    "All tiles show the same unchanged Blender scene."
                ),
            },
            {"type": "image_url", "image_url": {"url": prepared.data_url}},
        ]
        messages: list[Message] = [
            {"role": "system", "content": INSPECTION_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return messages, {
            "role": "inspection_reviewer",
            "request_type": "inspection_atlas_review",
            "prompt_version": INSPECTION_REVIEW_PROMPT_VERSION,
            "max_output_tokens": self._profile.max_output_tokens,
            "reasoning_effort": self._profile.reasoning_effort,
            "system_prompt": INSPECTION_REVIEW_SYSTEM_PROMPT,
            "input": context,
            "atlas": {
                "source_path": str(prepared.source),
                "prepared_width": prepared.width,
                "prepared_height": prepared.height,
            },
        }

    def review_messages(
        self,
        messages: Sequence[Message],
        manifest: InspectionAtlasManifest,
        *,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> InspectionReview:
        """Validate one reviewer response without accepting creative or executable output."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._profile.max_output_tokens,
            reasoning_effort=self._profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
        raw_response = json.dumps(response, indent=2, sort_keys=True, default=str)
        try:
            review = InspectionReviewWire.model_validate(response).to_domain()
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the inspection-review schema", raw_response
            ) from error
        valid_tiles = set(manifest.tiles)
        unknown_tiles = sorted(
            {
                tile
                for problem in review.problems
                for tile in problem.tiles
                if tile not in valid_tiles
            }
        )
        if unknown_tiles:
            raise ModelResponseError(
                "Inspection Reviewer referenced unknown atlas tile ids: "
                + ", ".join(unknown_tiles),
                raw_response,
            )
        return review
