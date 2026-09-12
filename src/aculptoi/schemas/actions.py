"""The deliberately small, allowlisted V1 Blender action language."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

ObjectName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_. -]*$",
        description="A Blender object name; paths and control characters are disallowed.",
    ),
]
Vector3 = Annotated[tuple[float, float, float], Field(description="Three finite numeric values.")]


class ActionBase(BaseModel):
    """Shared action safety checks."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def finite_numbers(cls, value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        if (
            isinstance(value, tuple)
            and all(isinstance(item, float | int) for item in value)
            and not all(math.isfinite(float(item)) for item in value)
        ):
            raise ValueError("vector values must be finite")
        return value


class ObjectCreate(ActionBase):
    command: Literal["object.create"]
    name: ObjectName
    primitive: Literal["cube", "uv_sphere", "cylinder", "cone"] = "cube"
    location: Vector3 = (0.0, 0.0, 0.0)
    scale: Vector3 = (1.0, 1.0, 1.0)


class ObjectDelete(ActionBase):
    command: Literal["object.delete"]
    object: ObjectName


class ObjectTranslate(ActionBase):
    command: Literal["object.translate"]
    object: ObjectName
    offset: Vector3


class ObjectRotate(ActionBase):
    command: Literal["object.rotate"]
    object: ObjectName
    degrees: Vector3


class ObjectScale(ActionBase):
    command: Literal["object.scale"]
    object: ObjectName
    scale: Vector3

    @field_validator("scale")
    @classmethod
    def positive_scale(cls, value: Vector3) -> Vector3:
        if any(component <= 0 or component > 100 for component in value):
            raise ValueError("scale multipliers must be greater than 0 and at most 100")
        return value


class SculptVoxelRemesh(ActionBase):
    command: Literal["sculpt.voxel_remesh"]
    object: ObjectName
    voxel_size: float = Field(gt=0.001, le=1.0)


Action = Annotated[
    ObjectCreate | ObjectDelete | ObjectTranslate | ObjectRotate | ObjectScale | SculptVoxelRemesh,
    Field(discriminator="command"),
]
_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(payload: object) -> Action:
    """Validate one action before it ever crosses the Blender boundary."""
    return _ACTION_ADAPTER.validate_python(payload)
