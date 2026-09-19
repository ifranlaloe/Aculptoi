"""Typed construction plans and per-work-item Actor responses."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .actions import Action
from .target import FormTrait
from .viewport import ViewportView

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
MIN_COMPLETION_CRITERIA = 1
MAX_COMPLETION_CRITERIA = 10
CompletionCriteria = Annotated[
    list[ShortText],
    Field(min_length=MIN_COMPLETION_CRITERIA, max_length=MAX_COMPLETION_CRITERIA),
]
WorkItemResponseKind = Literal["modeling_step", "observation_request", "complete"]


class CompletionCriteriaResponseRequirement(TypedDict, total=False):
    """Compact conditional contract for one work-item Actor response."""

    required: bool
    type: str
    min_items: int
    max_items: int
    must_be_omitted: bool


class WorkItemResponseRequirements(TypedDict):
    """Model-facing response requirements derived from durable item state."""

    allowed_kinds: list[WorkItemResponseKind]
    completion_criteria: CompletionCriteriaResponseRequirement


def work_item_response_requirements(
    *, completion_criteria_established: bool
) -> WorkItemResponseRequirements:
    """Return the small, deterministic response contract for the current Actor turn.

    Pydantic owns field types; this helper owns only the conditional requirement that
    establishes immutable completion criteria on an item's first response.
    """
    allowed_kinds: list[WorkItemResponseKind] = [
        "modeling_step",
        "observation_request",
        "complete",
    ]
    if completion_criteria_established:
        return {
            "allowed_kinds": allowed_kinds,
            "completion_criteria": {
                "required": False,
                "must_be_omitted": True,
            },
        }
    return {
        "allowed_kinds": allowed_kinds,
        "completion_criteria": {
            "required": True,
            "type": "array[string]",
            "min_items": MIN_COMPLETION_CRITERIA,
            "max_items": MAX_COMPLETION_CRITERIA,
        },
    }


class ConstructionItem(BaseModel):
    """One ordered, semantically coherent unit of construction or refinement."""

    model_config = ConfigDict(extra="forbid")

    id: WorkItemId
    title: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=2_000)
    depends_on: list[WorkItemId] = Field(default_factory=list, max_length=20)
    form_traits: list[FormTrait] = Field(default_factory=list, max_length=12)

    @field_validator("depends_on")
    @classmethod
    def dependencies_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("work-item dependencies must be unique")
        return value

    @field_validator("form_traits")
    @classmethod
    def form_traits_are_unique(cls, value: list[FormTrait]) -> list[FormTrait]:
        if len(value) != len(set(value)):
            raise ValueError("work-item form_traits must be unique")
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
    """One Actor turn: a Modeling Step, observation request, or completion.

    ``WorkItemActionBatch`` remains the public compatibility name for historic
    artifacts and integrations.  New model-facing responses use ``kind`` and
    treat a Modeling Step—not an arbitrary sized batch—as the unit of work.
    Legacy ``status`` payloads are accepted only while reading older runs.
    """

    model_config = ConfigDict(extra="forbid")

    kind: WorkItemResponseKind
    work_item_id: WorkItemId
    reason: str = Field(min_length=1, max_length=4_000)
    intent: str | None = Field(default=None, min_length=1, max_length=1_000)
    completion_criteria: CompletionCriteria | None = None
    actions: list[Action] = Field(default_factory=list, max_length=25)
    view: ViewportView | None = None
    legacy_complete_after_actions: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def read_legacy_status_payloads(cls, value: object) -> object:
        """Read pre-Modeling-Step artifacts without emitting their old wire shape."""
        if not isinstance(value, dict) or "kind" in value:
            return value
        data = dict(value)
        status = data.pop("status", None)
        actions = data.get("actions", [])
        if status not in {"continue", "complete"}:
            return value
        data["kind"] = "modeling_step" if status == "continue" or actions else "complete"
        if status == "complete" and actions:
            data["legacy_complete_after_actions"] = True
        if data["kind"] == "modeling_step":
            data.setdefault("intent", data.get("reason"))
        return data

    @model_validator(mode="after")
    def response_variant_is_coherent(self) -> WorkItemActionBatch:
        """Make response variants mutually exclusive at the trust boundary."""
        if self.kind == "modeling_step":
            if not self.actions:
                raise ValueError("a modeling_step must include at least one action")
            if self.intent is None:
                raise ValueError("a modeling_step must include one semantic intent")
            if self.view is not None:
                raise ValueError("a modeling_step cannot include a viewport view request")
        elif self.kind == "observation_request":
            if self.actions:
                raise ValueError("an observation_request cannot include mutating actions")
            if self.intent is not None:
                raise ValueError("an observation_request cannot include a modeling intent")
            if self.view is None:
                raise ValueError("an observation_request must include one semantic viewport view")
        else:
            if self.actions:
                raise ValueError("a complete response cannot include mutating actions")
            if self.intent is not None or self.view is not None:
                raise ValueError("a complete response cannot include intent or viewport controls")
        return self

    @property
    def status(self) -> Literal["continue", "complete"]:
        """Compatibility status for old callers; new code should use ``kind``."""
        return "complete" if self.kind == "complete" else "continue"
