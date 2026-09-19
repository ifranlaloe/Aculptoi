"""The deliberately small, typed, allowlisted Blender action language."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

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
NormalizedVector3 = Annotated[
    tuple[float, float, float],
    Field(description="Three finite normalized local-bound coordinates from -1 through 1."),
]

MAX_NORMALIZED_OFFSET = 2.0
MAX_REGION_SCALE = 4.0
MAX_JOIN_OBJECTS = 16


@dataclass(frozen=True)
class ActionCatalogEntry:
    """Compact, schema-owned guidance exposed to the Actor, never to visual roles."""

    command: str
    purpose: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    enum_values: dict[str, tuple[str, ...]] | None = None
    payload: dict[str, object] | None = None

    def to_context(self, *, include_payload: bool) -> dict[str, object]:
        """Produce a compact JSON-serializable description without a full JSON schema."""
        context: dict[str, object] = {
            "command": self.command,
            "purpose": self.purpose,
            "required_fields": list(self.required_fields),
        }
        if self.optional_fields:
            context["optional_fields"] = list(self.optional_fields)
        if self.enum_values:
            context["enum_values"] = {
                name: list(values) for name, values in self.enum_values.items()
            }
        if include_payload and self.payload is not None:
            context["payload_shape"] = self.payload
        return context


class ActionBase(BaseModel):
    """Shared action safety checks."""

    model_config = ConfigDict(extra="forbid")
    catalog_entry: ClassVar[ActionCatalogEntry]

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


class NormalizedRegion(ActionBase):
    """An inclusive local mesh AABB expressed relative to its current local bounds."""

    min: NormalizedVector3
    max: NormalizedVector3

    @field_validator("min", "max")
    @classmethod
    def normalized_coordinates_are_bounded(cls, value: NormalizedVector3) -> NormalizedVector3:
        if any(component < -1.0 or component > 1.0 for component in value):
            raise ValueError("normalized region coordinates must be between -1 and 1")
        return value

    @model_validator(mode="after")
    def region_is_nonempty(self) -> NormalizedRegion:
        if any(lower >= upper for lower, upper in zip(self.min, self.max, strict=True)):
            raise ValueError("normalized region min must be strictly less than max on every axis")
        return self

    @property
    def minimum(self) -> NormalizedVector3:
        """Provide a descriptive read-only alias without changing the wire shape."""
        return self.min

    @property
    def maximum(self) -> NormalizedVector3:
        """Provide a descriptive read-only alias without changing the wire shape."""
        return self.max


class ObjectCreate(ActionBase):
    command: Literal["object.create"]
    name: ObjectName
    primitive: Literal["cube", "uv_sphere", "cylinder", "cone"] = "cube"
    location: Vector3 = (0.0, 0.0, 0.0)
    scale: Vector3 = (1.0, 1.0, 1.0)
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.create",
        purpose="Create a primitive blockout mesh with a stable object name.",
        required_fields=("command", "name"),
        optional_fields=("primitive", "location", "scale"),
        enum_values={"primitive": ("cube", "uv_sphere", "cylinder", "cone")},
        payload={
            "command": "object.create",
            "name": "Name",
            "primitive": "uv_sphere",
            "location": [0, 0, 0],
            "scale": [1, 1, 1],
        },
    )

    @field_validator("scale")
    @classmethod
    def positive_scale(cls, value: Vector3) -> Vector3:
        if any(component <= 0 or component > 100 for component in value):
            raise ValueError("scale multipliers must be greater than 0 and at most 100")
        return value


class ObjectDelete(ActionBase):
    command: Literal["object.delete"]
    object: ObjectName
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.delete",
        purpose="Delete one existing object.",
        required_fields=("command", "object"),
        payload={"command": "object.delete", "object": "Name"},
    )


class ObjectTranslate(ActionBase):
    command: Literal["object.translate"]
    object: ObjectName
    offset: Vector3
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.translate",
        purpose="Translate one object in world-space units.",
        required_fields=("command", "object", "offset"),
        payload={"command": "object.translate", "object": "Name", "offset": [0, 0, 0]},
    )


class ObjectRotate(ActionBase):
    command: Literal["object.rotate"]
    object: ObjectName
    degrees: Vector3
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.rotate",
        purpose="Rotate one object by Euler degrees.",
        required_fields=("command", "object", "degrees"),
        payload={"command": "object.rotate", "object": "Name", "degrees": [0, 0, 0]},
    )


class ObjectScale(ActionBase):
    command: Literal["object.scale"]
    object: ObjectName
    scale: Vector3
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.scale",
        purpose="Multiply one object's transform scale.",
        required_fields=("command", "object", "scale"),
        payload={"command": "object.scale", "object": "Name", "scale": [1, 1, 1]},
    )

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
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="sculpt.voxel_remesh",
        purpose="Fuse overlapping mesh masses into one voxel-remeshed volume.",
        required_fields=("command", "object", "voxel_size"),
        payload={"command": "sculpt.voxel_remesh", "object": "Name", "voxel_size": 0.06},
    )


class ObjectJoin(ActionBase):
    """Join local editable mesh objects while preserving the target name."""

    command: Literal["object.join"]
    objects: list[ObjectName] = Field(min_length=2, max_length=MAX_JOIN_OBJECTS)
    target: ObjectName
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.join",
        purpose="Join two or more mesh objects; the named target survives.",
        required_fields=("command", "objects", "target"),
        payload={
            "command": "object.join",
            "objects": ["FishBody", "TailBlock"],
            "target": "FishBody",
        },
    )

    @field_validator("objects")
    @classmethod
    def objects_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("object.join objects must be unique")
        return value

    @model_validator(mode="after")
    def target_is_joined(self) -> ObjectJoin:
        if self.target not in self.objects:
            raise ValueError("object.join target must be included in objects")
        return self


class MeshTransformRegion(ActionBase):
    """Transform a bounded local mesh region without model-provided element IDs."""

    command: Literal["mesh.transform_region"]
    object: ObjectName
    region: NormalizedRegion
    translate: Vector3 = (0.0, 0.0, 0.0)
    scale: Vector3 = (1.0, 1.0, 1.0)
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="mesh.transform_region",
        purpose="Translate and/or scale vertices selected by a normalized local mesh region.",
        required_fields=("command", "object", "region"),
        optional_fields=("translate", "scale"),
        payload={
            "command": "mesh.transform_region",
            "object": "FishBody",
            "region": {"min": [0.45, -1, -1], "max": [1, 1, 1]},
            "translate": [0.15, 0, 0],
            "scale": [0.65, 0.75, 0.75],
        },
    )

    @field_validator("translate")
    @classmethod
    def translate_is_bounded(cls, value: Vector3) -> Vector3:
        if any(abs(component) > MAX_NORMALIZED_OFFSET for component in value):
            raise ValueError(
                f"mesh-region translation components must be within "
                f"[-{MAX_NORMALIZED_OFFSET}, {MAX_NORMALIZED_OFFSET}]"
            )
        return value

    @field_validator("scale")
    @classmethod
    def region_scale_is_bounded(cls, value: Vector3) -> Vector3:
        if any(component <= 0 or component > MAX_REGION_SCALE for component in value):
            raise ValueError(
                f"mesh-region scale multipliers must be greater than 0 and at most "
                f"{MAX_REGION_SCALE}"
            )
        return value


class MeshExtrudeRegion(ActionBase):
    """Extrude one connected face region selected from normalized local bounds."""

    command: Literal["mesh.extrude_region"]
    object: ObjectName
    region: NormalizedRegion
    offset: Vector3
    scale: Vector3 = (1.0, 1.0, 1.0)
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="mesh.extrude_region",
        purpose="Extrude one connected face region and move the new geometry locally.",
        required_fields=("command", "object", "region", "offset"),
        optional_fields=("scale",),
        payload={
            "command": "mesh.extrude_region",
            "object": "FishBody",
            "region": {"min": [0.6, -0.4, -0.4], "max": [1, 0.4, 0.4]},
            "offset": [0.5, 0, 0],
            "scale": [0.7, 0.7, 0.7],
        },
    )

    @field_validator("offset")
    @classmethod
    def offset_is_bounded(cls, value: Vector3) -> Vector3:
        if any(abs(component) > MAX_NORMALIZED_OFFSET for component in value):
            raise ValueError(
                f"mesh-region offset components must be within "
                f"[-{MAX_NORMALIZED_OFFSET}, {MAX_NORMALIZED_OFFSET}]"
            )
        return value

    @field_validator("scale")
    @classmethod
    def extruded_scale_is_bounded(cls, value: Vector3) -> Vector3:
        if any(component <= 0 or component > MAX_REGION_SCALE for component in value):
            raise ValueError(
                f"mesh-region scale multipliers must be greater than 0 and at most "
                f"{MAX_REGION_SCALE}"
            )
        return value


class MeshSmoothRegion(ActionBase):
    """Apply bounded deterministic smoothing to vertices in a normalized region."""

    command: Literal["mesh.smooth_region"]
    object: ObjectName
    region: NormalizedRegion
    factor: float = Field(ge=0.0, le=1.0)
    iterations: int = Field(ge=1, le=10)
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="mesh.smooth_region",
        purpose="Apply simultaneous bounded Laplacian smoothing to selected vertices.",
        required_fields=("command", "object", "region", "factor", "iterations"),
        payload={
            "command": "mesh.smooth_region",
            "object": "FishBody",
            "region": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            "factor": 0.4,
            "iterations": 3,
        },
    )


class ObjectShadeSmooth(ActionBase):
    """Enable smooth shading on every polygon of one mesh without changing topology."""

    command: Literal["object.shade_smooth"]
    object: ObjectName
    catalog_entry: ClassVar[ActionCatalogEntry] = ActionCatalogEntry(
        command="object.shade_smooth",
        purpose="Enable smooth shading for every polygon of one mesh.",
        required_fields=("command", "object"),
        payload={"command": "object.shade_smooth", "object": "FishBody"},
    )


ACTION_TYPES: tuple[type[ActionBase], ...] = (
    ObjectCreate,
    ObjectDelete,
    ObjectTranslate,
    ObjectRotate,
    ObjectScale,
    SculptVoxelRemesh,
    ObjectJoin,
    MeshTransformRegion,
    MeshExtrudeRegion,
    MeshSmoothRegion,
    ObjectShadeSmooth,
)


Action = Annotated[
    ObjectCreate
    | ObjectDelete
    | ObjectTranslate
    | ObjectRotate
    | ObjectScale
    | SculptVoxelRemesh
    | ObjectJoin
    | MeshTransformRegion
    | MeshExtrudeRegion
    | MeshSmoothRegion
    | ObjectShadeSmooth,
    Field(discriminator="command"),
]
_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def action_catalog(*, include_payload: bool = True) -> list[dict[str, object]]:
    """Return the complete compact Actor-facing catalog from the typed action classes."""
    return [
        action_type.catalog_entry.to_context(include_payload=include_payload)
        for action_type in ACTION_TYPES
    ]


def action_capability_summary() -> list[dict[str, object]]:
    """Return planning-level command capabilities without executable payload examples."""
    return action_catalog(include_payload=False)


def modeling_action_semantics() -> dict[str, object]:
    """Return compact execution meaning for the non-obvious modeling commands."""
    return {
        "normalized_mesh_regions": {
            "coordinate_space": "current mesh local-space axis-aligned bounds",
            "coordinate_range": [-1.0, 1.0],
            "bounds": "inclusive",
            "recomputed": (
                "region coordinates are resolved from current mesh bounds at the start "
                "of each action"
            ),
            "axis_meaning": (
                "-1 is the current local minimum and +1 is the current local maximum on that axis"
            ),
            "raw_element_ids": "not available",
        },
        "commands": {
            "mesh.transform_region": {
                "selection": "vertices inside the normalized region",
                "translation_units": (
                    "1.0 equals one current half-extent of the local mesh bounds on that axis"
                ),
                "scale_pivot": "centroid of selected vertices",
            },
            "mesh.extrude_region": {
                "selection": "faces whose centers are inside the normalized region",
                "requirements": [
                    "at least one selected face",
                    "one connected selected face region",
                ],
                "recoverable_failures": ["empty_region", "disconnected_region"],
                "offset_units": (
                    "1.0 equals one current half-extent of the local mesh bounds on that axis"
                ),
                "scale_pivot": "centroid of new extruded vertices",
            },
            "mesh.smooth_region": {
                "selection": "vertices inside the normalized region",
                "effect": (
                    "bounded geometry smoothing using the supplied factor and iteration "
                    "count; distinct from smooth shading"
                ),
            },
            "object.join": {
                "topology_effect": (
                    "combines mesh objects into one mesh but does not weld or fuse "
                    "overlapping surfaces"
                ),
            },
            "sculpt.voxel_remesh": {
                "topology_effect": (
                    "rebuilds a mesh as a voxel-remeshed volume, can fuse overlapping "
                    "masses, and changes topology"
                ),
            },
        },
    }


def parse_action(payload: object) -> Action:
    """Validate one action before it ever crosses the Blender boundary."""
    return _ACTION_ADAPTER.validate_python(payload)
