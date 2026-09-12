"""Vision role: it inspects rendered files and emits read-only critiques."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from aculptoi.agent.prompts import CRITIC_SYSTEM_PROMPT
from aculptoi.models.base import Message, ModelProvider
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
        """Send a multi-image OpenAI-compatible message and validate the critique."""
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "goal": goal,
                        "previous_score": previous_score,
                        "views": [image.stem for image in images],
                    },
                    sort_keys=True,
                ),
            }
        ]
        for image in images:
            prepared = prepare_render(image, self._max_image_dimension)
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
        return VisualCritique.model_validate(
            self._provider.complete_json(messages, max_tokens=self._max_output_tokens)
        )
