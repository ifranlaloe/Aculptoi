"""Typed inference settings shared by model-consuming application roles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aculptoi.reasoning import ReasoningEffort


class InferenceProfile(BaseModel):
    """Bounded generation settings for one logical model request stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: ReasoningEffort
    max_output_tokens: int = Field(ge=128, le=65_536)
