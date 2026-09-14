"""Structured, bounded outcomes for work-item Blender action batches."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FailureKind = Literal["validation_error", "execution_error", "worker_error"]


class ActionExecutionFailure(BaseModel):
    """A public worker failure safe to persist and include in a fresh Actor request."""

    model_config = ConfigDict(extra="forbid")

    failure_kind: FailureKind
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1, max_length=500)
    action_index: int | None = Field(default=None, ge=0, le=24)
    command: str | None = Field(default=None, min_length=1, max_length=64)
    executed_before_failure: int = Field(ge=0, le=25)
    recoverable: bool
    rolled_back: bool
    scene_restored: bool

    @model_validator(mode="after")
    def failure_semantics_are_coherent(self) -> ActionExecutionFailure:
        """Prevent a fatal worker fault from being treated as a model retry signal."""
        if self.failure_kind == "worker_error" and self.recoverable:
            raise ValueError("worker_error failures must not be recoverable")
        if self.rolled_back != self.scene_restored:
            raise ValueError("rollback and scene restoration must be reported together")
        return self


class WorkItemExecutionOutcome(BaseModel):
    """The immediate execution state supplied to a later fresh work-item request."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    action_batch: int = Field(ge=1)
    proposed_action_count: int = Field(ge=0, le=25)
    attempted_action_count: int = Field(ge=0, le=25)
    worker_called: bool
    canonical_scene: str = Field(min_length=1, max_length=300)
    worker_result: dict[str, object] | None = None
    failure: ActionExecutionFailure | None = None

    @model_validator(mode="after")
    def outcome_fields_match_status(self) -> WorkItemExecutionOutcome:
        """Keep success and failure data unambiguous for the Actor and artifacts."""
        if self.status == "succeeded":
            if self.failure is not None:
                raise ValueError("successful execution must not include a failure")
            if self.attempted_action_count != self.proposed_action_count:
                raise ValueError("successful execution must attempt every proposed action")
        elif self.failure is None:
            raise ValueError("failed execution must include structured failure details")
        return self
