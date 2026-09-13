"""Typed inference settings shared by model-consuming application roles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aculptoi.reasoning import ReasoningEffort


class InferenceProfile(BaseModel):
    """Bounded generation settings for one logical model request stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thinking: bool
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int = Field(ge=128, le=65_536)

    @model_validator(mode="after")
    def reasoning_effort_matches_thinking(self) -> InferenceProfile:
        """Keep reasoning depth meaningful only when hidden thinking is enabled."""
        if self.thinking and self.reasoning_effort is None:
            raise ValueError("reasoning_effort is required when thinking is enabled")
        if not self.thinking and self.reasoning_effort is not None:
            raise ValueError("reasoning_effort must be omitted when thinking is disabled")
        return self
