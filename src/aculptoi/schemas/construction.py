"""Typed construction plans and per-work-item Actor responses."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .actions import Action

WorkItemId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="A stable, lowercase kebab-case work-item identifier.",
    ),
]
ShortText = Annotated[str, Field(min_length=1, max_length=500)]
CompletionCriteria = Annotated[list[ShortText], Field(min_length=1, max_length=10)]


class ConstructionItem(BaseModel):
    """One ordered, semantically coherent unit of construction or refinement."""

    model_config = ConfigDict(extra="forbid")

    id: WorkItemId
    title: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=2_000)
    depends_on: list[WorkItemId] = Field(default_factory=list, max_length=20)

    @field_validator("depends_on")
    @classmethod
    def dependencies_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("work-item dependencies must be unique")
        return value


class ConstructionPlan(BaseModel):
    """The immutable Actor-produced plan at the root of one visual iteration."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=4_000)
    items: list[ConstructionItem] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_ordered_dependencies(self) -> ConstructionPlan:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("construction-plan work-item ids must be unique")

        all_ids = set(ids)
        completed_before_item: set[str] = set()
        for item in self.items:
            unknown = set(item.depends_on) - all_ids
            if unknown:
                raise ValueError(
                    f"work item '{item.id}' has unknown dependencies: {sorted(unknown)}"
                )
            unavailable = set(item.depends_on) - completed_before_item
            if unavailable:
                raise ValueError(
                    f"work item '{item.id}' dependencies must refer to earlier items: "
                    f"{sorted(unavailable)}"
                )
            completed_before_item.add(item.id)
        return self


class WorkItemActionBatch(BaseModel):
    """One bounded set of actions for the currently active construction item."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: WorkItemId
    status: Literal["continue", "complete"]
    reason: str = Field(min_length=1, max_length=4_000)
    completion_criteria: CompletionCriteria | None = None
    actions: list[Action] = Field(default_factory=list, max_length=25)

    @model_validator(mode="after")
    def continuing_batches_must_make_progress(self) -> WorkItemActionBatch:
        if self.status == "continue" and not self.actions:
            raise ValueError("a continuing work item must include at least one action")
        return self
