"""Typed, read-only Actor viewport observation contracts.

These schemas intentionally describe a small semantic sensor API rather than
Blender UI state.  They cannot express scene mutations, arbitrary matrices,
operators, or input events.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ViewportOrientation = Literal[
    "front",
    "rear",
    "left",
    "right",
    "top",
    "bottom",
    "front_three_quarter",
    "rear_three_quarter",
]
ViewportProjection = Literal["orthographic", "perspective"]
ViewportFraming = Literal["whole_subject", "medium", "close"]


class ViewportView(BaseModel):
    """One bounded semantic request for Aculptoi's reserved 3D observation view."""

    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(default=None, min_length=1, max_length=64)
    orientation: ViewportOrientation = "front_three_quarter"
    projection: ViewportProjection = "orthographic"
    framing: ViewportFraming = "whole_subject"

    @field_validator("target")
    @classmethod
    def target_is_a_safe_blender_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        # Keep this in sync with the bounded object-name contract enforced in
        # Blender.  The worker independently validates it before lookup.
        if not value[0].isalpha() and value[0] != "_":
            raise ValueError("viewport target must start with a letter or underscore")
        if not all(character.isalnum() or character in "_. -" for character in value):
            raise ValueError("viewport target contains unsupported characters")
        return value


class ViewportObservation(BaseModel):
    """Compact metadata for a current Actor sensor reading; never image bytes."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    target: str | None = Field(default=None, min_length=1, max_length=64)
    orientation: ViewportOrientation | None = None
    projection: ViewportProjection | None = None
    framing: ViewportFraming | None = None
    width: int | None = Field(default=None, ge=1, le=1024)
    height: int | None = Field(default=None, ge=1, le=1024)
    error: str | None = Field(default=None, min_length=1, max_length=500)

    @classmethod
    def unavailable(cls, error: str) -> ViewportObservation:
        """Create an explicit structured-only fallback without image data."""
        return cls(available=False, error=error)


class ViewportObservationRequest(BaseModel):
    """A non-mutating Actor request for one additional semantic observation."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    reason: str = Field(min_length=1, max_length=4_000)
    view: ViewportView
