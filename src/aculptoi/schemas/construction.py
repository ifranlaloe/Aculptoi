"""Typed construction plans and per-work-item Actor responses."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Annotated, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .actions import ACTION_TYPES, Action
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
WORK_ITEM_FIRST_RESPONSE_SCHEMA_ID = "work-item-first-response-v2"
WORK_ITEM_LATER_RESPONSE_SCHEMA_ID = "work-item-later-response-v2"


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


def work_item_response_schema_id(*, completion_criteria_established: bool) -> str:
    """Return the stable provider-facing contract identifier for one Actor turn."""
    return (
        WORK_ITEM_LATER_RESPONSE_SCHEMA_ID
        if completion_criteria_established
        else WORK_ITEM_FIRST_RESPONSE_SCHEMA_ID
    )


def work_item_response_schema(*, completion_criteria_established: bool) -> dict[str, object]:
    """Build the compact, deterministic JSON Schema for one work-item response.

    Pydantic remains the host-side authority.  This is a deliberately filtered transport
    view of ``WorkItemActionBatch``: it preserves typed action and viewport constraints,
    while replacing its state-dependent response variants with an explicit discriminated
    union.  The shared ``work_item_response_requirements`` helper remains the single
    authority for whether immutable completion criteria are established on this turn.
    """
    requirements = work_item_response_requirements(
        completion_criteria_established=completion_criteria_established
    )
    source = _provider_json_schema(WorkItemActionBatch.model_json_schema())
    properties = _schema_object(source.get("properties"), "work-item response")
    definitions = _schema_object(source.get("$defs", {}), "work-item response definitions")
    _require_actor_action_fields(definitions)

    common = {
        "work_item_id": _property(properties, "work_item_id"),
        "reason": _property(properties, "reason"),
    }
    criteria = _non_null_property(properties, "completion_criteria")
    actions = _property(properties, "actions")
    actions["minItems"] = 1
    intent = _non_null_property(properties, "intent")
    view = _non_null_property(properties, "view")

    criteria_requirement = requirements["completion_criteria"]
    criteria_required = criteria_requirement["required"]

    variant_fields: dict[WorkItemResponseKind, dict[str, dict[str, object]]] = {
        "modeling_step": {"intent": intent, "actions": actions},
        "observation_request": {"view": view},
        "complete": {},
    }
    variants = [
        _response_variant(
            kind,
            common=common,
            fields=variant_fields[kind],
            completion_criteria=criteria if criteria_required else None,
        )
        for kind in requirements["allowed_kinds"]
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": definitions,
        "oneOf": variants,
        "discriminator": {"propertyName": "kind"},
    }


def work_item_response_schema_sha256(*, completion_criteria_established: bool) -> str:
    """Return a stable digest for compact artifact provenance without storing the schema."""
    import json

    serialized = json.dumps(
        work_item_response_schema(completion_criteria_established=completion_criteria_established),
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _provider_json_schema(value: object) -> dict[str, object]:
    """Return the grammar-friendly provider subset of the Pydantic schema.

    Pydantic remains the exact host-side contract.  Provider grammar backends need the
    response shape and small structural limits, but large string ``maxLength`` values can
    expand into impractical bounded-character grammar repetitions.  Omit them here while
    retaining Pydantic enforcement after generation.
    """
    compact = _strip_provider_schema_noise(value)
    return _schema_object(compact, "Pydantic work-item schema")


def _strip_provider_schema_noise(value: object) -> object:
    """Recursively remove non-structural or grammar-hostile transport keywords."""
    if isinstance(value, dict):
        return {
            key: _strip_provider_schema_noise(item)
            for key, item in value.items()
            if key not in {"default", "description", "maxLength", "title"}
        }
    if isinstance(value, list):
        return [_strip_provider_schema_noise(item) for item in value]
    return value


def _schema_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object schema for {name}")
    return cast(dict[str, object], value)


def _property(properties: dict[str, object], name: str) -> dict[str, object]:
    value = properties.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"Pydantic work-item schema is missing property '{name}'")
    return deepcopy(cast(dict[str, object], value))


def _non_null_property(properties: dict[str, object], name: str) -> dict[str, object]:
    value = _property(properties, name)
    alternatives = value.get("anyOf")
    if not isinstance(alternatives, list):
        raise RuntimeError(f"Pydantic work-item schema property '{name}' is not nullable")
    for alternative in alternatives:
        if isinstance(alternative, dict) and alternative.get("type") != "null":
            return deepcopy(cast(dict[str, object], alternative))
    raise RuntimeError(f"Pydantic work-item schema property '{name}' has no non-null branch")


def _require_actor_action_fields(definitions: dict[str, object]) -> None:
    """Apply the existing Actor-only action-catalog requirements to transport schema."""
    for action_type in ACTION_TYPES:
        definition = definitions.get(action_type.__name__)
        if not isinstance(definition, dict):
            raise RuntimeError(f"Pydantic action schema is missing {action_type.__name__}")
        required = definition.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise RuntimeError(
                f"Pydantic action schema has invalid requirements for {action_type.__name__}"
            )
        for field in action_type.catalog_entry.actor_required_fields:
            if field not in required:
                required.append(field)


def _response_variant(
    kind: WorkItemResponseKind,
    *,
    common: dict[str, dict[str, object]],
    fields: dict[str, dict[str, object]],
    completion_criteria: dict[str, object] | None,
) -> dict[str, object]:
    properties: dict[str, object] = {"kind": {"const": kind}}
    properties.update(deepcopy(common))
    properties.update(deepcopy(fields))
    required = ["kind", "work_item_id", "reason", *fields]
    if completion_criteria is not None:
        properties["completion_criteria"] = deepcopy(completion_criteria)
        required.append("completion_criteria")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
