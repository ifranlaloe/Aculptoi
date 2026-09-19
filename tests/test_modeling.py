from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from aculptoi.modeling import (
    KNOWLEDGE_CHARACTER_BUDGET,
    MAX_KNOWLEDGE_CARDS,
    ModelingContextCompiler,
    load_modeling_knowledge,
    select_modeling_knowledge,
)
from aculptoi.modeling.knowledge import (
    KnowledgeCardError,
    _validate_unique_card_ids,
    parse_modeling_knowledge_card,
)
from aculptoi.schemas.actions import (
    ACTION_TYPES,
    MeshSmoothRegion,
    MeshSubdivide,
    SculptVoxelRemesh,
    action_capability_summary,
    action_catalog,
    modeling_action_semantics,
    parse_action,
)
from aculptoi.schemas.construction import ConstructionItem
from aculptoi.schemas.target import TargetBrief


def _fish_brief() -> TargetBrief:
    return TargetBrief(
        subject="simple stylized fish",
        visual_priorities=["one coherent organic body", "recognizable fish silhouette"],
        constraints=["major anatomy remains readable from multiple viewpoints"],
        non_goals=["realistic scales"],
        form_traits=[
            "organic",
            "continuous_form",
            "bilateral_symmetry",
            "tapered_form",
            "elongated_form",
            "appendages",
        ],
    )


def _card(
    card_id: str,
    *,
    topics: tuple[str, ...] = ("organic",),
    roles: tuple[str, ...] = ("actor_plan", "actor_work_item"),
) -> bytes:
    return f"""+++
id = "{card_id}"
topics = {list(topics)!r}
roles = {list(roles)!r}
sources = []
+++

# Transferable form guidance

## Construction guidance

Build readable primary forms.

## Evaluation signals

Check silhouette clarity.

## Common failure modes

Avoid unrelated detail.
""".encode()


def test_target_brief_rejects_invalid_bounds_and_duplicate_traits() -> None:
    with pytest.raises(ValidationError, match="form_traits must be unique"):
        TargetBrief(
            subject="fish",
            visual_priorities=["body"],
            form_traits=["organic", "organic"],
        )
    with pytest.raises(ValidationError):
        TargetBrief(
            subject="fish",
            visual_priorities=["x" * 501],
        )


def test_target_brief_legacy_fallback_is_deterministic_and_bounded() -> None:
    brief = TargetBrief.legacy_fallback(" create a cube ")

    assert brief.model_dump(mode="json") == {
        "subject": "create a cube",
        "visual_priorities": ["create a cube"],
        "constraints": [],
        "non_goals": [],
        "form_traits": [],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "command": "object.join",
            "objects": ["FishBody", "TailBlock"],
            "target": "FishBody",
        },
        {
            "command": "mesh.transform_region",
            "object": "FishBody",
            "region": {"min": [0.45, -1, -1], "max": [1, 1, 1]},
            "translate": [0.15, 0, 0],
            "scale": [0.65, 0.75, 0.75],
        },
        {
            "command": "mesh.extrude_region",
            "object": "FishBody",
            "region": {"min": [0.6, -0.4, -0.4], "max": [1, 0.4, 0.4]},
            "offset": [0.5, 0, 0],
            "scale": [0.7, 0.7, 0.7],
        },
        {
            "command": "mesh.smooth_region",
            "object": "FishBody",
            "region": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            "factor": 0.4,
            "iterations": 3,
        },
        {"command": "mesh.subdivide", "object": "FishBody", "cuts": 1},
        {"command": "object.shade_smooth", "object": "FishBody"},
    ],
)
def test_new_semantic_modeling_actions_validate(payload: dict[str, object]) -> None:
    assert parse_action(payload).command == payload["command"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "command": "object.join",
            "objects": ["FishBody", "FishBody"],
            "target": "FishBody",
        },
        {
            "command": "object.join",
            "objects": ["FishBody", "TailBlock"],
            "target": "Head",
        },
        {
            "command": "mesh.transform_region",
            "object": "FishBody",
            "region": {"min": [0, 0, 0], "max": [0, 1, 1]},
        },
        {
            "command": "mesh.extrude_region",
            "object": "FishBody",
            "region": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            "offset": [float("nan"), 0, 0],
        },
        {
            "command": "mesh.smooth_region",
            "object": "FishBody",
            "region": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            "factor": float("inf"),
            "iterations": 3,
        },
        {
            "command": "mesh.transform_region",
            "object": "FishBody",
            "region": {"min": [-1.1, -1, -1], "max": [1, 1, 1]},
        },
    ],
)
def test_new_semantic_modeling_actions_reject_unsafe_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_action(payload)


def test_action_catalog_covers_each_typed_action_exactly_once() -> None:
    catalog = action_catalog()
    commands = [entry["command"] for entry in catalog]
    typed_commands = [
        get_args(action_type.model_fields["command"].annotation)[0] for action_type in ACTION_TYPES
    ]

    assert commands == typed_commands
    assert len(commands) == len(set(commands))
    assert action_capability_summary() == [
        {key: value for key, value in entry.items() if key != "payload_shape"} for entry in catalog
    ]
    create = next(entry for entry in catalog if entry["command"] == "object.create")
    assert create["enum_values"] == {"primitive": ["cube", "uv_sphere", "cylinder", "cone"]}
    assert create["required_fields"] == ["command", "name", "primitive"]
    assert create["constraints"] == {
        "scale": {"components": {"exclusive_minimum": 0.0, "maximum": 100.0}}
    }
    remesh = next(entry for entry in catalog if entry["command"] == "sculpt.voxel_remesh")
    assert remesh["constraints"] == {"voxel_size": {"exclusive_minimum": 0.001, "maximum": 1.0}}
    subdivide = next(entry for entry in catalog if entry["command"] == "mesh.subdivide")
    assert subdivide["constraints"] == {"cuts": {"minimum": 1, "maximum": 3}}


def test_catalog_numeric_constraints_match_authoritative_schema_bounds() -> None:
    """Keep compact model guidance aligned with Pydantic's actual validation metadata."""

    def bounds(model: type[BaseModel], field: str) -> dict[str, float | int]:
        model_fields = model.model_fields
        values: dict[str, float | int] = {}
        for metadata in model_fields[field].metadata:
            for name in ("gt", "ge", "le"):
                value = getattr(metadata, name, None)
                if value is not None:
                    values[name] = value
        return values

    catalog = {entry["command"]: entry for entry in action_catalog()}
    assert bounds(SculptVoxelRemesh, "voxel_size") == {"gt": 0.001, "le": 1.0}
    assert catalog["sculpt.voxel_remesh"]["constraints"] == {
        "voxel_size": {"exclusive_minimum": 0.001, "maximum": 1.0}
    }
    assert bounds(MeshSubdivide, "cuts") == {"ge": 1, "le": 3}
    assert catalog["mesh.subdivide"]["constraints"] == {"cuts": {"minimum": 1, "maximum": 3}}
    assert bounds(MeshSmoothRegion, "factor") == {"ge": 0.0, "le": 1.0}
    assert bounds(MeshSmoothRegion, "iterations") == {"ge": 1, "le": 10}


def test_packaged_cards_are_original_parseable_and_have_stable_hashes() -> None:
    cards = load_modeling_knowledge()
    repeated_load = load_modeling_knowledge()

    assert [card.id for card in cards] == sorted(card.id for card in cards)
    assert len(cards) == len({card.id for card in cards})
    assert all(len(card.sha256) == 64 for card in cards)
    assert all(card.sources == () for card in cards)
    assert [card.sha256 for card in cards] == [card.sha256 for card in repeated_load]
    tapering = next(card for card in cards if card.id == "tapering-forms")
    raw_tapering = files("aculptoi.modeling").joinpath("knowledge/tapering-forms.md").read_bytes()
    assert tapering.sha256 == hashlib.sha256(raw_tapering).hexdigest()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"not TOML front matter", "must begin with TOML front matter"),
        (
            _card("missing-sources").replace(b"sources = []\n", b""),
            "metadata must contain exactly",
        ),
        (
            _card("versioned-card-v2"),
            "invalid non-versioned card id",
        ),
        (
            _card("invalid-sources").replace(b"sources = []", b'sources = "not-an-array"'),
            "metadata 'sources' must be a string array",
        ),
        (
            _card("unexpected-section").replace(
                b"## Common failure modes", b"## Unsupported section"
            ),
            "sections must be exactly",
        ),
    ],
)
def test_knowledge_card_parser_rejects_invalid_metadata_and_sections(
    raw: bytes, message: str
) -> None:
    with pytest.raises(KnowledgeCardError, match=message):
        parse_modeling_knowledge_card(raw, "invalid.md")


def test_knowledge_card_parser_rejects_unsupported_roles_and_duplicate_ids() -> None:
    with pytest.raises(KnowledgeCardError, match="unsupported topics or roles"):
        parse_modeling_knowledge_card(
            _card("invalid-role", roles=("inspection_reviewer",)),
            "invalid.md",
        )
    card = parse_modeling_knowledge_card(_card("duplicate"), "duplicate.md")
    with pytest.raises(KnowledgeCardError, match="duplicate knowledge card ids"):
        _validate_unique_card_ids((card, card))


def test_knowledge_selection_is_role_specific_bounded_and_deterministic() -> None:
    cards = load_modeling_knowledge()
    brief = _fish_brief()
    first = select_modeling_knowledge(cards, role="actor_work_item", target_brief=brief)
    second = select_modeling_knowledge(cards, role="actor_work_item", target_brief=brief)
    critic = select_modeling_knowledge(cards, role="critic_discovery", target_brief=brief)

    assert first == second
    assert 1 <= len(first) <= MAX_KNOWLEDGE_CARDS["actor_work_item"]
    assert len(critic) <= MAX_KNOWLEDGE_CARDS["critic_discovery"]
    assert sum(len(card.render()) for card in first) <= KNOWLEDGE_CHARACTER_BUDGET
    assert "Construction Guidance" in first[0].render()
    assert "Evaluation Signals" not in first[0].render()
    assert "Evaluation Signals" in critic[0].render()
    assert "Construction Guidance" not in critic[0].render()


def test_knowledge_selection_uses_lexical_tiebreaking_and_never_truncates_cards() -> None:
    alpha = parse_modeling_knowledge_card(
        _card("alpha-taper", topics=("tapered_form",)), "alpha-taper.md"
    )
    beta = parse_modeling_knowledge_card(
        _card("beta-taper", topics=("tapered_form",)), "beta-taper.md"
    )
    critic_only = parse_modeling_knowledge_card(
        _card("critic-taper", roles=("critic_discovery",), topics=("tapered_form",)),
        "critic-taper.md",
    )
    brief = TargetBrief(
        subject="a tapered shape",
        visual_priorities=["readable taper"],
        form_traits=[],
    )

    selected = select_modeling_knowledge(
        (beta, critic_only, alpha),
        role="actor_plan",
        target_brief=brief,
    )

    assert [entry.card.id for entry in selected] == ["alpha-taper", "beta-taper"]
    assert [entry.rank for entry in selected] == [1, 2]
    assert selected[0].render() == alpha.render_for_role("actor_plan")
    assert "Avoid unrelated detail." in selected[0].render()


def test_work_item_traits_prioritize_current_modeling_problem() -> None:
    cards = load_modeling_knowledge()
    brief = _fish_brief()
    body = select_modeling_knowledge(
        cards,
        role="actor_work_item",
        target_brief=brief,
        priority_traits=("organic", "continuous_form", "elongated_form", "tapered_form"),
        relevant_text=("Establish one coherent elongated body with gradual taper.",),
    )
    appendages = select_modeling_knowledge(
        cards,
        role="actor_work_item",
        target_brief=brief,
        priority_traits=("appendages", "bilateral_symmetry", "thin_features"),
        relevant_text=("Establish paired side appendages with clear attachment and symmetry.",),
    )
    body_blockout = select_modeling_knowledge(
        cards,
        role="actor_work_item",
        target_brief=brief,
        priority_traits=("organic", "continuous_form"),
        relevant_text=("Establish one coherent primary body mass.",),
    )
    fallback = select_modeling_knowledge(
        cards,
        role="actor_work_item",
        target_brief=brief,
        relevant_text=("Establish one coherent elongated body with gradual taper.",),
    )
    explicit_empty_traits = select_modeling_knowledge(
        cards,
        role="actor_work_item",
        target_brief=brief,
        priority_traits=(),
        relevant_text=("Establish one coherent elongated body with gradual taper.",),
    )

    body_ids = [entry.card.id for entry in body]
    appendage_ids = [entry.card.id for entry in appendages]
    assert "tapering-forms" in body_ids
    assert body_ids[0] == "tapering-forms"
    assert "organic-blockout" in [entry.card.id for entry in body_blockout]
    assert "appendage-construction" in appendage_ids
    assert "bilateral-symmetry" in appendage_ids
    assert appendage_ids.index("appendage-construction") < appendage_ids.index("tapering-forms")
    assert fallback == explicit_empty_traits


def test_modeling_action_semantics_cover_real_nontrivial_actions() -> None:
    semantics = modeling_action_semantics()
    normalized_regions = semantics["normalized_mesh_regions"]
    commands = semantics["commands"]

    assert normalized_regions == {
        "coordinate_space": "current mesh local-space axis-aligned bounds",
        "coordinate_range": [-1.0, 1.0],
        "bounds": "inclusive",
        "recomputed": (
            "region coordinates are resolved from current mesh bounds at the start of each action"
        ),
        "axis_meaning": (
            "-1 is the current local minimum and +1 is the current local maximum on that axis"
        ),
        "raw_element_ids": "not available",
    }
    assert isinstance(commands, dict)
    assert commands["mesh.transform_region"] == {
        "selection": "vertices inside the normalized region",
        "translation_units": (
            "1.0 equals one current half-extent of the local mesh bounds on that axis"
        ),
        "scale_pivot": "centroid of selected vertices",
    }
    assert commands["mesh.extrude_region"] == {
        "selection": "faces whose centers are inside the normalized region",
        "requirements": ["at least one selected face", "one connected selected face region"],
        "recoverable_failures": ["empty_region", "disconnected_region"],
        "offset_units": (
            "1.0 equals one current half-extent of the local mesh bounds on that axis"
        ),
        "scale_pivot": "centroid of new extruded vertices",
    }
    smoothing = commands["mesh.smooth_region"]
    assert "Laplacian-style geometry smoothing" in smoothing["effect"]
    assert smoothing["topology_effect"] == "does not add or remove topology"
    assert "may shrink or flatten" in smoothing["volume_behavior"]
    assert "sparse topology" in smoothing["volume_behavior"]
    assert smoothing["distinct_from"] == "object.shade_smooth changes polygon shading only"
    assert "does not weld or fuse" in commands["object.join"]["topology_effect"]
    subdivide = commands["mesh.subdivide"]
    assert "increases mesh topology density" in subdivide["effect"]
    assert "not smoothing or voxel reconstruction" in subdivide["topology_effect"]
    remesh = commands["sculpt.voxel_remesh"]
    assert "reconstructs the mesh" in remesh["effect"]
    assert "can fuse overlapping masses" in remesh["topology_effect"]
    assert remesh["not_for"] == [
        "ordinary topology-density increase on an existing surface",
        "subdivision",
    ]
    assert "can remove thin features" in remesh["cautions"]

    typed_commands = {
        get_args(action_type.model_fields["command"].annotation)[0] for action_type in ACTION_TYPES
    }
    assert {
        "mesh.transform_region",
        "mesh.extrude_region",
        "mesh.smooth_region",
        "mesh.subdivide",
        "object.join",
        "sculpt.voxel_remesh",
    }.issubset(commands)
    assert set(commands).issubset(typed_commands)


def test_fish_work_item_contexts_prioritize_stage_guidance_and_share_semantics() -> None:
    compiler = ModelingContextCompiler(load_modeling_knowledge())
    body = ConstructionItem(
        id="establish-body-mass",
        title="Establish body mass",
        objective="Create one coherent elongated primary body volume with gradual taper.",
        depends_on=[],
        form_traits=["organic", "continuous_form", "elongated_form", "tapered_form"],
    )
    appendages = ConstructionItem(
        id="construct-paired-appendages",
        title="Construct paired appendages",
        objective="Create matching side appendages with readable attachment bases and symmetry.",
        depends_on=["establish-body-mass"],
        form_traits=["appendages", "bilateral_symmetry", "thin_features"],
    )
    brief = _fish_brief()
    body_context = compiler.compile(
        role="actor_work_item",
        target_brief=brief,
        priority_traits=tuple(body.form_traits),
        relevant_text=(body.title, body.objective),
    )
    appendage_context = compiler.compile(
        role="actor_work_item",
        target_brief=brief,
        priority_traits=tuple(appendages.form_traits),
        relevant_text=(appendages.title, appendages.objective),
    )
    body_fields = body_context.request_fields(include_action_catalog="full")
    appendage_fields = appendage_context.request_fields(include_action_catalog="full")

    body_ids = [entry["id"] for entry in body_context.knowledge_metadata]
    appendage_ids = [entry["id"] for entry in appendage_context.knowledge_metadata]
    assert "tapering-forms" in body_ids
    assert "silhouette-and-proportion" in body_ids
    assert "appendage-construction" in appendage_ids
    assert "bilateral-symmetry" in appendage_ids
    assert appendage_ids.index("appendage-construction") < appendage_ids.index("tapering-forms")
    assert body_fields["action_semantics"] == appendage_fields["action_semantics"]


def test_modeling_context_exposes_catalog_only_to_actor_roles() -> None:
    compiler = ModelingContextCompiler(load_modeling_knowledge())
    plan = compiler.compile(role="actor_plan", target_brief=_fish_brief())
    work_item = compiler.compile(
        role="actor_work_item",
        target_brief=_fish_brief(),
        priority_traits=("appendages", "bilateral_symmetry", "thin_features"),
    )
    critic = compiler.compile(role="critic_analysis", target_brief=_fish_brief())

    plan_fields = plan.request_fields(include_action_catalog="summary")
    work_item_fields = work_item.request_fields(include_action_catalog="full")
    critic_fields = critic.request_fields()
    assert "modeling_capabilities" in plan_fields
    assert "action_catalog" not in plan_fields
    assert "action_semantics" not in plan_fields
    assert "action_catalog" in work_item_fields
    assert work_item_fields["action_semantics"] == modeling_action_semantics()
    assert "modeling_capabilities" not in work_item_fields
    assert "action_catalog" not in critic_fields
    assert "action_semantics" not in critic_fields
    assert "modeling_capabilities" not in critic_fields
    assert "Construction Guidance" not in str(critic_fields["modeling_guidance"])
    assert "Evaluation Signals" in str(critic_fields["modeling_guidance"])
    assert critic.knowledge_metadata


def test_hard_surface_context_remains_bounded_and_does_not_select_organic_guidance() -> None:
    brief = TargetBrief(
        subject="a repeated geometric cube arrangement",
        visual_priorities=["uniform repeated cubies"],
        form_traits=["hard_surface", "repeated_geometry"],
    )
    context = ModelingContextCompiler(load_modeling_knowledge()).compile(
        role="actor_work_item",
        target_brief=brief,
        priority_traits=("hard_surface", "repeated_geometry"),
        relevant_text=("Create and position the next repeated cubie.",),
    )
    fields = context.request_fields(include_action_catalog="full")
    card_ids = [entry["id"] for entry in context.knowledge_metadata]

    assert len(card_ids) <= MAX_KNOWLEDGE_CARDS["actor_work_item"]
    assert "organic-blockout" not in card_ids
    assert any(entry["command"] == "object.create" for entry in fields["action_catalog"])
