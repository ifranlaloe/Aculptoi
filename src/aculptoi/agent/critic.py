"""Vision role: it inspects rendered files and emits read-only critiques."""

from __future__ import annotations

import base64
from pathlib import Path

from aculptoi.models.base import Message, ModelProvider
from aculptoi.schemas.critique import VisualCritique


class VisionCritic:
    """Use an independent vision-capable provider; it has no worker dependency."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    def inspect(self, goal: str, images: list[Path]) -> VisualCritique:
        """Send a multi-image OpenAI-compatible message and validate the critique."""
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Assess these Blender inspection renders against the goal below. "
                    "Return only JSON: {score: 0..1, summary: string, "
                    "issues: [{severity, region, description, suggestion}]}. "
                    "You are read-only: do not suggest commands or claim execution. Goal: " + goal
                ),
            }
        ]
        for image in images:
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        messages: list[Message] = [
            {"role": "system", "content": "You are a precise, read-only 3D visual critic."},
            {"role": "user", "content": content},
        ]
        return VisualCritique.model_validate(self._provider.complete_json(messages))
