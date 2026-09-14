"""Typed, inspectable state for one self-contained Aculptoi run."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aculptoi.schemas.construction import WorkItemId

RunStatus = Literal["created", "running", "recovering", "completed", "interrupted", "stopped"]
IterationPhase = Literal["actor", "inspection", "critic"]
TargetBriefState = Literal["absent", "pending", "ready", "legacy"]


class ActiveWorkItem(BaseModel):
    """The one work item whose partial scene mutations are disposable."""

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    ordinal: int = Field(ge=1)
    work_item_id: WorkItemId
    attempt: int = Field(default=1, ge=1)


class DurableWorkItem(BaseModel):
    """A completed item with an immutable known-good checkpoint."""

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    ordinal: int = Field(ge=1)
    work_item_id: WorkItemId
    checkpoint: str = Field(min_length=1, max_length=300)


class ActiveIterationPhase(BaseModel):
    """The post-construction phase that must finish before an iteration advances."""

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    phase: IterationPhase
    attempt: int = Field(default=1, ge=1)
    inspection_artifact_root: str | None = Field(default=None, min_length=1, max_length=300)


class RunState(BaseModel):
    """The recovery source of truth beside a run's canonical Blender file."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    goal: str | None = Field(default=None, max_length=20_000)
    target_brief_state: TargetBriefState = "absent"
    target_brief_artifact: str | None = Field(default=None, min_length=1, max_length=300)
    target_brief_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scene: str = "scene.blend"
    initial_scene: str = "initial-scene.blend"
    status: RunStatus = "created"
    iteration: int = Field(default=0, ge=0)
    active_item: ActiveWorkItem | None = None
    active_phase: ActiveIterationPhase | None = None
    latest_completed_item: DurableWorkItem | None = None
    latest_checkpoint: str | None = Field(default=None, max_length=300)
    durable_items: list[DurableWorkItem] = Field(default_factory=list, max_length=5_000)
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def target_brief_reference_is_coherent(self) -> RunState:
        """Keep immutable target-brief provenance explicit while accepting historic runs."""
        has_reference = (
            self.target_brief_artifact is not None and self.target_brief_artifact_sha256 is not None
        )
        if self.target_brief_state in {"ready", "legacy"}:
            if not has_reference:
                raise ValueError("ready target brief state requires an artifact reference and hash")
            if self.target_brief_artifact != "target-brief.json":
                raise ValueError("target brief artifact must use the fixed target-brief.json path")
        elif (
            self.target_brief_artifact is not None or self.target_brief_artifact_sha256 is not None
        ):
            raise ValueError("only ready target brief state may reference a target brief artifact")
        return self

    def with_updated_timestamp(self) -> RunState:
        """Return a copy carrying a fresh UTC update time for atomic persistence."""
        return self.model_copy(update={"updated_at": datetime.now(UTC).isoformat()})
