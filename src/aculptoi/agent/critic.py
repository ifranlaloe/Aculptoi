"""Vision role: it inspects rendered files and emits read-only critiques."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from aculptoi.agent.prompts import CRITIC_PROMPT_VERSION, CRITIC_SYSTEM_PROMPT
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.schemas.critique import VisualCritique
from aculptoi.vision import prepare_render


class VisionCritic:
    """Use a vision-capable provider; it has no Blender-worker dependency."""

    def __init__(
        self,
        provider: ModelProvider,
        max_image_dimension: int = 1280,
        max_output_tokens: int = 768,
    ) -> None:
        self._provider = provider
        self._max_image_dimension = max_image_dimension
        self._max_output_tokens = max_output_tokens

    def inspect(
        self,
        goal: str,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> VisualCritique:
        """Build a multi-image request and validate its read-only visual critique."""
        messages, _ = self.build_request(goal, images, previous_score=previous_score)
        return self.inspect_messages(messages)

    def build_request(
        self,
        goal: str,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> tuple[list[Message], dict[str, object]]:
        """Build request messages and a readable manifest without inline image data."""
        context = {
            "goal": goal,
            "previous_score": previous_score,
            "views": [image.stem for image in images],
        }
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(context, sort_keys=True),
            }
        ]
        views: list[dict[str, object]] = []
        for image in images:
            prepared = prepare_render(image, self._max_image_dimension)
            views.append(
                {
                    "name": prepared.source.stem,
                    "source_path": str(prepared.source),
                    "prepared_width": prepared.width,
                    "prepared_height": prepared.height,
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Inspection view: {prepared.source.stem} "
                        f"({prepared.width}x{prepared.height} pixels)."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": prepared.data_url},
                }
            )
        messages: list[Message] = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        artifact: dict[str, object] = {
            "role": "vision_critic",
            "prompt_version": CRITIC_PROMPT_VERSION,
            "max_output_tokens": self._max_output_tokens,
            "system_prompt": CRITIC_SYSTEM_PROMPT,
            "input": context,
            "views": views,
        }
        return messages, artifact

    def inspect_messages(self, messages: Sequence[Message]) -> VisualCritique:
        """Request and validate a previously constructed critic message sequence."""
        response = self._provider.complete_json(messages, max_tokens=self._max_output_tokens)
        try:
            return VisualCritique.model_validate(response)
        except ValidationError as error:
            raw_response = json.dumps(response, indent=2, sort_keys=True, default=str)
            raise ModelResponseError(
                "Model response did not satisfy the visual-critique schema", raw_response
            ) from error
