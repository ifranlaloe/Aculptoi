"""Typed contracts for deterministic visual inspection surveys."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CameraId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="A stable lowercase camera identifier owned by Aculptoi.",
    ),
]
AtlasTileId = Annotated[
    str,
    Field(
        min_length=2,
        max_length=12,
        pattern=r"^[A-Z]+[1-9][0-9]*$",
        description="A stable row-and-column atlas tile identifier, such as A1.",
    ),
]
FiniteVector3 = Annotated[tuple[float, float, float], Field(description="Three finite values.")]
CameraProjection = Literal["orthographic", "perspective"]
SelectionKind = Literal["canonical_anchor", "dynamic"]
InspectionProfile = Literal["standard", "retry"]
InspectionStopReason = Literal[
    "coverage_target",
    "min_view_gain",
    "max_views",
    "atlas_resolution",
    "candidate_exhausted",
]
InspectionProblemType = Literal[
    "coverage_gap",
    "lighting",
    "framing",
    "tile_resolution",
    "redundancy",
    "atlas_composition",
    "other",
]
ViewPreference = Literal["any", "upper", "lower", "front", "rear", "right", "left"]


class InspectionBounds(BaseModel):
    """World-space subject bounds calculated from the current Blender geometry."""

    model_config = ConfigDict(extra="forbid")

    minimum: FiniteVector3
    maximum: FiniteVector3
    center: FiniteVector3
    radius: float = Field(gt=0)

    @model_validator(mode="after")
    def minimum_must_not_exceed_maximum(self) -> InspectionBounds:
        if any(low > high for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("inspection bounds minimum must not exceed maximum")
        return self


class InspectionFraming(BaseModel):
    """Reproducible camera framing derived from the inspected scene bounds."""

    model_config = ConfigDict(extra="forbid")

    margin: float = Field(ge=1.0, le=3.0)
    distance: float = Field(gt=0)
    orthographic_scale: float = Field(gt=0)


class CameraCandidate(BaseModel):
    """One deterministic camera direction before or after selection."""

    model_config = ConfigDict(extra="forbid")

    camera_id: CameraId
    azimuth_degrees: float = Field(ge=-180.0, le=180.0)
    elevation_degrees: float = Field(ge=-89.0, le=89.0)
    orientation: str = Field(min_length=1, max_length=120)
    projection: CameraProjection
    canonical_anchor: bool = False


class CandidateDiagnostic(BaseModel):
    """Low-cost visibility measurements for one candidate viewpoint."""

    model_config = ConfigDict(extra="forbid")

    camera_id: CameraId
    frame_coverage: float = Field(ge=0.0, le=1.0)
    surface_sample_ids: list[int] = Field(default_factory=list, max_length=2048)
    surface_sample_count: int = Field(ge=0, le=2048)
    silhouette_signature: str = Field(min_length=1, max_length=512, pattern=r"^[0-9a-f]+$")

    @field_validator("surface_sample_ids")
    @classmethod
    def surface_sample_ids_are_unique_and_nonnegative(cls, value: list[int]) -> list[int]:
        if any(sample < 0 for sample in value):
            raise ValueError("surface sample ids must be nonnegative")
        if len(value) != len(set(value)):
            raise ValueError("surface sample ids must be unique")
        return value

    @model_validator(mode="after")
    def samples_must_fit_the_reported_population(self) -> CandidateDiagnostic:
        if any(sample >= self.surface_sample_count for sample in self.surface_sample_ids):
            raise ValueError("surface sample id exceeds the reported sample population")
        return self


class CandidateSurvey(BaseModel):
    """A worker-produced low-cost diagnostic survey for candidate cameras."""

    model_config = ConfigDict(extra="forbid")

    bounds: InspectionBounds
    diagnostics: list[CandidateDiagnostic] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def diagnostic_camera_ids_are_unique(self) -> CandidateSurvey:
        ids = [diagnostic.camera_id for diagnostic in self.diagnostics]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate diagnostic camera ids must be unique")
        return self


class SelectedCameraChoice(CameraCandidate):
    """A selected camera with explainable deterministic selection measurements."""

    selection_kind: SelectionKind
    selection_reason: str = Field(min_length=1, max_length=200)
    coverage_gain: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)


class AtlasLayout(BaseModel):
    """Deterministic rectangular placement for an inspection atlas."""

    model_config = ConfigDict(extra="forbid")

    columns: int = Field(ge=1, le=128)
    rows: int = Field(ge=1, le=128)
    tile_dimension: int = Field(ge=1, le=16_384)
    width: int = Field(ge=1, le=16_384)
    height: int = Field(ge=1, le=16_384)

    @model_validator(mode="after")
    def dimensions_must_match_grid(self) -> AtlasLayout:
        if self.width != self.columns * self.tile_dimension:
            raise ValueError("atlas width must equal columns times tile_dimension")
        if self.height != self.rows * self.tile_dimension:
            raise ValueError("atlas height must equal rows times tile_dimension")
        return self


class InspectionCamera(SelectedCameraChoice):
    """A selected view assigned to a visible atlas tile."""

    tile_id: AtlasTileId


class InspectionCameraPlan(BaseModel):
    """The exact camera plan given to Blender for one inspection round."""

    model_config = ConfigDict(extra="forbid")

    sensor_version: str = Field(min_length=1, max_length=100)
    lighting_rig: str = Field(min_length=1, max_length=100)
    round: int = Field(ge=1)
    profile: InspectionProfile = "standard"
    layout: AtlasLayout
    views: list[InspectionCamera] = Field(min_length=1, max_length=128)
    estimated_surface_coverage: float = Field(ge=0.0, le=1.0)
    selection_stop_reason: InspectionStopReason

    @model_validator(mode="after")
    def views_must_fit_unique_tiles(self) -> InspectionCameraPlan:
        camera_ids = [view.camera_id for view in self.views]
        tile_ids = [view.tile_id for view in self.views]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("inspection camera ids must be unique")
        if len(tile_ids) != len(set(tile_ids)):
            raise ValueError("inspection atlas tile ids must be unique")
        if len(self.views) > self.layout.columns * self.layout.rows:
            raise ValueError("inspection views exceed atlas layout capacity")
        return self


class InspectionShot(BaseModel):
    """One final full-quality source render from Blender."""

    model_config = ConfigDict(extra="forbid")

    camera_id: CameraId
    tile_id: AtlasTileId
    path: str = Field(min_length=1, max_length=1_000)
    width: int = Field(ge=1, le=16_384)
    height: int = Field(ge=1, le=16_384)


class InspectionRenderResult(BaseModel):
    """Final render metadata returned by the local Blender worker."""

    model_config = ConfigDict(extra="forbid")

    bounds: InspectionBounds
    framing: InspectionFraming
    shots: list[InspectionShot] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def shot_camera_ids_are_unique(self) -> InspectionRenderResult:
        camera_ids = [shot.camera_id for shot in self.shots]
        tile_ids = [shot.tile_id for shot in self.shots]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("inspection render shot camera ids must be unique")
        if len(tile_ids) != len(set(tile_ids)):
            raise ValueError("inspection render shot tile ids must be unique")
        return self


class InspectionAtlasTile(BaseModel):
    """A tile's visual and geometric provenance within one combined atlas."""

    model_config = ConfigDict(extra="forbid")

    tile_id: AtlasTileId
    camera_id: CameraId
    source: str = Field(min_length=1, max_length=1_000)
    pixel_bounds: tuple[int, int, int, int]
    azimuth_degrees: float = Field(ge=-180.0, le=180.0)
    elevation_degrees: float = Field(ge=-89.0, le=89.0)
    orientation: str = Field(min_length=1, max_length=120)
    projection: CameraProjection
    selection_kind: SelectionKind
    selection_reason: str = Field(min_length=1, max_length=200)
    coverage_gain: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)

    @field_validator("pixel_bounds")
    @classmethod
    def pixel_bounds_are_positive(
        cls, value: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if value[0] < 0 or value[1] < 0 or value[2] < 1 or value[3] < 1:
            raise ValueError("atlas pixel bounds must have a nonnegative origin and positive size")
        return value


class InspectionAtlasManifest(BaseModel):
    """Complete durable metadata for one rendered inspection atlas."""

    model_config = ConfigDict(extra="forbid")

    sensor_version: str = Field(min_length=1, max_length=100)
    lighting_rig: str = Field(min_length=1, max_length=100)
    width: int = Field(ge=1, le=16_384)
    height: int = Field(ge=1, le=16_384)
    layout: AtlasLayout
    bounds: InspectionBounds
    framing: InspectionFraming
    estimated_surface_coverage: float = Field(ge=0.0, le=1.0)
    tiles: dict[AtlasTileId, InspectionAtlasTile] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def manifest_matches_its_layout(self) -> InspectionAtlasManifest:
        if self.width != self.layout.width or self.height != self.layout.height:
            raise ValueError("atlas manifest dimensions must match its layout")
        if any(tile_id != tile.tile_id for tile_id, tile in self.tiles.items()):
            raise ValueError("atlas manifest tile keys must match tile ids")
        for tile in self.tiles.values():
            x, y, width, height = tile.pixel_bounds
            if x + width > self.width or y + height > self.height:
                raise ValueError("atlas tile bounds must be contained by the atlas")
        return self

    def to_reviewer_context(self) -> dict[str, object]:
        """Return concise technical metadata for the Inspection Reviewer."""
        return {
            "sensor_version": self.sensor_version,
            "lighting_rig": self.lighting_rig,
            "atlas_width": self.width,
            "atlas_height": self.height,
            "tile_dimension": self.layout.tile_dimension,
            "estimated_surface_coverage": self.estimated_surface_coverage,
            "tiles": [
                {
                    "id": tile.tile_id,
                    "orientation": tile.orientation,
                    "azimuth_degrees": tile.azimuth_degrees,
                    "elevation_degrees": tile.elevation_degrees,
                    "projection": tile.projection,
                    "selection_kind": tile.selection_kind,
                }
                for tile in self.tiles.values()
            ],
        }

    def to_critic_context(self) -> dict[str, object]:
        """Return interpretation metadata without source paths or worker details."""
        return {
            "sensor_version": self.sensor_version,
            "lighting_rig": self.lighting_rig,
            "atlas_width": self.width,
            "atlas_height": self.height,
            "tile_dimension": self.layout.tile_dimension,
            "tiles": [
                {
                    "id": tile.tile_id,
                    "orientation": tile.orientation,
                    "azimuth_degrees": tile.azimuth_degrees,
                    "elevation_degrees": tile.elevation_degrees,
                    "projection": tile.projection,
                }
                for tile in self.tiles.values()
            ],
        }


class InspectionReviewProblem(BaseModel):
    """One technical observation about an inspection atlas, never a modeling judgment."""

    model_config = ConfigDict(extra="forbid")

    type: InspectionProblemType
    reason: str = Field(min_length=1, max_length=500)
    region: str | None = Field(default=None, min_length=1, max_length=120)
    tiles: list[AtlasTileId] = Field(default_factory=list, max_length=32)
    view_preference: ViewPreference = "any"

    @field_validator("tiles")
    @classmethod
    def tiles_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("inspection review problem tile ids must be unique")
        return value


class InspectionReviewWire(BaseModel):
    """Compact model-response contract for the Inspection Reviewer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accept", "augment", "retry"]
    confidence: int = Field(ge=0, le=100)
    problems: list[InspectionReviewProblem] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def status_must_match_problem_inventory(self) -> InspectionReviewWire:
        if self.status == "accept" and self.problems:
            raise ValueError("an accepted inspection must not report unresolved problems")
        if self.status != "accept" and not self.problems:
            raise ValueError("augment and retry inspection reviews must report a problem")
        return self

    def to_domain(self) -> InspectionReview:
        """Convert compact integer confidence to the normal domain scale."""
        return InspectionReview(
            status=self.status,
            confidence=self.confidence / 100,
            problems=self.problems,
        )


class InspectionReview(BaseModel):
    """A read-only technical assessment of one inspection atlas."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accept", "augment", "retry"]
    confidence: float = Field(ge=0.0, le=1.0)
    problems: list[InspectionReviewProblem] = Field(default_factory=list, max_length=8)


class InspectionSummary(BaseModel):
    """Durable final outcome for a bounded inspection subsystem invocation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted"]
    rounds: int = Field(ge=1)
    total_views_rendered: int = Field(ge=1)
    views_in_accepted_atlas: int = Field(ge=1)
    estimated_surface_coverage: float = Field(ge=0.0, le=1.0)
    accepted_atlas: str = Field(min_length=1, max_length=1_000)
    accepted_manifest: str = Field(min_length=1, max_length=1_000)
    sensor_version: str = Field(min_length=1, max_length=100)
    lighting_rig: str = Field(min_length=1, max_length=100)
