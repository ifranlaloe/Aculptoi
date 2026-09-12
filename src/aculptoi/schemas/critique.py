"""Read-only structured output contract for the vision critic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CritiqueIssue(BaseModel):
    """One actionable observation from a vision model."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["low", "medium", "high"]
    region: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1_000)
    suggestion: str = Field(min_length=1, max_length=1_000)


class VisualCritique(BaseModel):
    """Critique data only; it contains no Blender execution capability."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2_000)
    issues: list[CritiqueIssue] = Field(default_factory=list, max_length=30)
