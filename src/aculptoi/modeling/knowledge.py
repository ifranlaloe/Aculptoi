"""Load and deterministically select compact, transferable modeling knowledge cards."""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from aculptoi.schemas.target import FormTrait, TargetBrief

KnowledgeRole = Literal["actor_plan", "actor_work_item", "critic_discovery", "critic_analysis"]
_KNOWLEDGE_ROLE_ADAPTER: TypeAdapter[KnowledgeRole] = TypeAdapter(KnowledgeRole)
_FORM_TRAITS_ADAPTER: TypeAdapter[FormTrait] = TypeAdapter(FormTrait)
_CARD_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSIONED_ID_RE = re.compile(r"-v[0-9]+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_REQUIRED_METADATA = {"id", "topics", "roles", "sources"}
_REQUIRED_SECTIONS = {
    "construction_guidance",
    "evaluation_signals",
    "common_failure_modes",
}
_ROLE_SECTIONS: dict[KnowledgeRole, tuple[str, str]] = {
    "actor_plan": ("construction_guidance", "common_failure_modes"),
    "actor_work_item": ("construction_guidance", "common_failure_modes"),
    "critic_discovery": ("evaluation_signals", "common_failure_modes"),
    "critic_analysis": ("evaluation_signals", "common_failure_modes"),
}
MAX_CARD_CHARACTERS = 4_000
KNOWLEDGE_CHARACTER_BUDGET = 12_000
MAX_KNOWLEDGE_CARDS: dict[KnowledgeRole, int] = {
    "actor_plan": 5,
    "actor_work_item": 5,
    "critic_discovery": 4,
    "critic_analysis": 4,
}


class KnowledgeCardError(ValueError):
    """A packaged modeling knowledge card is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ModelingKnowledgeCard:
    """One compact, original card with role-specific extractable sections."""

    id: str
    topics: frozenset[FormTrait]
    roles: frozenset[KnowledgeRole]
    sources: tuple[str, ...]
    sections: Mapping[str, str]
    sha256: str

    def sections_for_role(self, role: KnowledgeRole) -> tuple[str, str]:
        """Return only guidance appropriate to the logical recipient."""
        if role not in self.roles:
            raise KnowledgeCardError(f"knowledge card {self.id!r} does not permit role {role!r}")
        return _ROLE_SECTIONS[role]

    def render_for_role(self, role: KnowledgeRole) -> str:
        """Render complete permitted Markdown sections without arbitrary truncation."""
        sections = self.sections_for_role(role)
        rendered = [f"### {self.id.replace('-', ' ').title()}"]
        for section in sections:
            rendered.append(f"#### {section.replace('_', ' ').title()}")
            rendered.append(self.sections[section])
        return "\n\n".join(rendered)

    def metadata_for_role(self, role: KnowledgeRole, rank: int) -> dict[str, object]:
        """Record stable provenance for one request without copying the source card."""
        return {
            "id": self.id,
            "sha256": self.sha256,
            "sections": list(self.sections_for_role(role)),
            "role": role,
            "rank": rank,
        }


@dataclass(frozen=True)
class SelectedKnowledgeCard:
    """One selected card with a deterministic rank in a compiled request context."""

    card: ModelingKnowledgeCard
    role: KnowledgeRole
    rank: int

    def render(self) -> str:
        return self.card.render_for_role(self.role)

    def metadata(self) -> dict[str, object]:
        return self.card.metadata_for_role(self.role, self.rank)


def _tokens(values: Iterable[str]) -> set[str]:
    return {
        token
        for value in values
        for token in _TOKEN_RE.findall(value.casefold())
        if len(token) >= 3
    }


def _split_front_matter(text: str, source: str) -> tuple[dict[str, object], str]:
    if not text.startswith("+++\n"):
        raise KnowledgeCardError(f"{source} must begin with TOML front matter")
    closing = text.find("\n+++\n", len("+++\n"))
    if closing < 0:
        raise KnowledgeCardError(f"{source} has unterminated TOML front matter")
    try:
        metadata = tomllib.loads(text[len("+++\n") : closing])
    except tomllib.TOMLDecodeError as error:
        raise KnowledgeCardError(f"{source} has invalid TOML front matter: {error}") from error
    if not isinstance(metadata, dict):
        raise KnowledgeCardError(f"{source} TOML front matter must be a table")
    return metadata, text[closing + len("\n+++\n") :].strip()


def _sections(body: str, source: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## ([A-Za-z][A-Za-z ]+)\n", body, re.MULTILINE))
    if not matches:
        raise KnowledgeCardError(f"{source} has no role-specific Markdown sections")
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        identifier = match.group(1).casefold().replace(" ", "_")
        next_start = matches[index + 1].start() if index + 1 < len(matches) else None
        content = body[match.end() : next_start]
        content = content.strip()
        if not content:
            raise KnowledgeCardError(f"{source} section {identifier!r} must not be empty")
        if identifier in sections:
            raise KnowledgeCardError(f"{source} duplicates section {identifier!r}")
        sections[identifier] = content
    missing = _REQUIRED_SECTIONS - set(sections)
    unexpected = set(sections) - _REQUIRED_SECTIONS
    if missing or unexpected:
        raise KnowledgeCardError(
            f"{source} sections must be exactly {sorted(_REQUIRED_SECTIONS)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return sections


def _string_list(value: object, source: str, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise KnowledgeCardError(f"{source} metadata {field!r} must be a string array")
    if len(value) > maximum:
        raise KnowledgeCardError(f"{source} metadata {field!r} exceeds {maximum} entries")
    if any(not item.strip() for item in value):
        raise KnowledgeCardError(f"{source} metadata {field!r} must not contain blank entries")
    if len({item.casefold().strip() for item in value}) != len(value):
        raise KnowledgeCardError(f"{source} metadata {field!r} must be unique")
    return value


def parse_modeling_knowledge_card(raw: bytes, source: str) -> ModelingKnowledgeCard:
    """Parse one packaged card, retaining a content hash over its exact bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KnowledgeCardError(f"{source} must be UTF-8") from error
    metadata, body = _split_front_matter(text, source)
    if set(metadata) != _REQUIRED_METADATA:
        raise KnowledgeCardError(
            f"{source} metadata must contain exactly {sorted(_REQUIRED_METADATA)}"
        )
    card_id = metadata["id"]
    if (
        not isinstance(card_id, str)
        or not _CARD_ID_RE.fullmatch(card_id)
        or _VERSIONED_ID_RE.search(card_id)
    ):
        raise KnowledgeCardError(f"{source} has an invalid non-versioned card id")
    topics = _string_list(metadata["topics"], source, "topics", 12)
    roles = _string_list(metadata["roles"], source, "roles", 4)
    sources = _string_list(metadata["sources"], source, "sources", 12)
    try:
        parsed_topics = frozenset(_FORM_TRAITS_ADAPTER.validate_python(topic) for topic in topics)
        parsed_roles = frozenset(_KNOWLEDGE_ROLE_ADAPTER.validate_python(role) for role in roles)
    except ValidationError as error:
        raise KnowledgeCardError(f"{source} has unsupported topics or roles: {error}") from error
    if not parsed_topics:
        raise KnowledgeCardError(f"{source} must declare at least one transferable topic")
    if not parsed_roles:
        raise KnowledgeCardError(f"{source} must declare at least one permitted role")
    sections = _sections(body, source)
    if len(body) > MAX_CARD_CHARACTERS:
        raise KnowledgeCardError(
            f"{source} exceeds the {MAX_CARD_CHARACTERS}-character modeling-card limit"
        )
    return ModelingKnowledgeCard(
        id=card_id,
        topics=parsed_topics,
        roles=parsed_roles,
        sources=tuple(sources),
        sections=sections,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_modeling_knowledge() -> tuple[ModelingKnowledgeCard, ...]:
    """Load the package's full small card library in a deterministic order."""
    directory = files("aculptoi.modeling").joinpath("knowledge")
    cards = [
        parse_modeling_knowledge_card(resource.read_bytes(), resource.name)
        for resource in directory.iterdir()
        if resource.name.endswith(".md")
    ]
    cards.sort(key=lambda card: card.id)
    _validate_unique_card_ids(cards)
    return tuple(cards)


def _validate_unique_card_ids(cards: Sequence[ModelingKnowledgeCard]) -> None:
    """Reject ambiguous card identities before deterministic selection can begin."""
    duplicate_ids = {card.id for card in cards if sum(other.id == card.id for other in cards) > 1}
    if duplicate_ids:
        raise KnowledgeCardError(f"duplicate knowledge card ids: {sorted(duplicate_ids)}")


def select_modeling_knowledge(
    cards: Sequence[ModelingKnowledgeCard],
    *,
    role: KnowledgeRole,
    target_brief: TargetBrief,
    relevant_text: Sequence[str] = (),
) -> tuple[SelectedKnowledgeCard, ...]:
    """Select a small role-permitted guidance set using stable trait-first scoring."""
    permitted = [card for card in cards if role in card.roles]
    traits = set(target_brief.form_traits)
    lexical_context = _tokens(
        (
            target_brief.subject,
            *target_brief.visual_priorities,
            *target_brief.constraints,
            *target_brief.non_goals,
            *relevant_text,
        )
    )
    ranked: list[tuple[int, int, ModelingKnowledgeCard]] = []
    for card in permitted:
        trait_matches = len(traits & card.topics)
        card_tokens = _tokens(
            (card.id.replace("-", " "), *(topic.replace("_", " ") for topic in card.topics))
        )
        lexical_matches = len(lexical_context & card_tokens)
        if trait_matches == 0 and lexical_matches == 0:
            continue
        ranked.append((trait_matches, lexical_matches, card))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].id))

    selected: list[SelectedKnowledgeCard] = []
    used_characters = 0
    for _, _, card in ranked:
        rendered = card.render_for_role(role)
        if len(selected) >= MAX_KNOWLEDGE_CARDS[role]:
            break
        if used_characters + len(rendered) > KNOWLEDGE_CHARACTER_BUDGET:
            continue
        selected.append(SelectedKnowledgeCard(card=card, role=role, rank=len(selected) + 1))
        used_characters += len(rendered)
    return tuple(selected)
