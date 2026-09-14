"""Bounded, reusable interpretation of one immutable raw user goal."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BriefText = Annotated[str, Field(min_length=1, max_length=500)]
FormTrait = Literal[
    "organic",
    "hard_surface",
    "continuous_form",
    "bilateral_symmetry",
    "radial_symmetry",
    "tapered_form",
    "elongated_form",
    "serpentine_form",
    "appendages",
    "thin_features",
    "repeated_geometry",
    "layered_forms",
]


class TargetBrief(BaseModel):
    """A compact interpretation aid; the raw goal remains authoritative."""

    model_config = ConfigDict(extra="forbid")

    subject: BriefText
    visual_priorities: list[BriefText] = Field(min_length=1, max_length=12)
    constraints: list[BriefText] = Field(default_factory=list, max_length=12)
    non_goals: list[BriefText] = Field(default_factory=list, max_length=12)
    form_traits: list[FormTrait] = Field(default_factory=list, max_length=12)

    @field_validator("subject")
    @classmethod
    def subject_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target brief subject must not be blank")
        return value

    @field_validator("visual_priorities", "constraints", "non_goals")
    @classmethod
    def text_lists_are_unique_and_nonblank(cls, values: list[str]) -> list[str]:
        normalized = [value.casefold().strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("target brief entries must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("target brief entries must be unique")
        return values

    @field_validator("form_traits")
    @classmethod
    def form_traits_are_unique(cls, values: list[FormTrait]) -> list[FormTrait]:
        if len(values) != len(set(values)):
            raise ValueError("target brief form_traits must be unique")
        return values

    @classmethod
    def legacy_fallback(cls, goal: str) -> TargetBrief:
        """Derive a deterministic bounded fallback without making a model request."""
        subject = goal.strip()[:500] or "unspecified modeling goal"
        return cls(
            subject=subject,
            visual_priorities=[subject],
            constraints=[],
            non_goals=[],
            form_traits=[],
        )


class TargetBriefArtifact(BaseModel):
    """The immutable run artifact binding a brief to the raw goal it interprets."""

    model_config = ConfigDict(extra="forbid")

    goal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    brief: TargetBrief
    prompt_version: str | None = Field(default=None, pattern=r"^v[0-9]+$")
    derivation: Literal["actor", "legacy_fallback"]
