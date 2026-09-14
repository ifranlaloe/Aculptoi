from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import get_args

import pytest
from pydantic import ValidationError

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
    action_capability_summary,
    action_catalog,
    parse_action,
)
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


def test_modeling_context_exposes_catalog_only_to_actor_roles() -> None:
    compiler = ModelingContextCompiler(load_modeling_knowledge())
    plan = compiler.compile(role="actor_plan", target_brief=_fish_brief())
    work_item = compiler.compile(role="actor_work_item", target_brief=_fish_brief())
    critic = compiler.compile(role="critic_analysis", target_brief=_fish_brief())

    plan_fields = plan.request_fields(include_action_catalog="summary")
    work_item_fields = work_item.request_fields(include_action_catalog="full")
    critic_fields = critic.request_fields()
    assert "modeling_capabilities" in plan_fields
    assert "action_catalog" not in plan_fields
    assert "action_catalog" in work_item_fields
    assert "modeling_capabilities" not in work_item_fields
    assert "action_catalog" not in critic_fields
    assert "modeling_capabilities" not in critic_fields
    assert "Construction Guidance" not in str(critic_fields["modeling_guidance"])
    assert "Evaluation Signals" in str(critic_fields["modeling_guidance"])
    assert critic.knowledge_metadata
