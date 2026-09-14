"""Centralized compact request-context compilation for modeling-aware logical roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aculptoi.schemas.actions import action_capability_summary, action_catalog
from aculptoi.schemas.target import TargetBrief

from .knowledge import (
    KnowledgeRole,
    ModelingKnowledgeCard,
    SelectedKnowledgeCard,
    select_modeling_knowledge,
)


@dataclass(frozen=True)
class CompiledModelingContext:
    """One inspectable, bounded set of role-appropriate target and guidance context."""

    target_brief: TargetBrief
    role: KnowledgeRole
    selected_cards: tuple[SelectedKnowledgeCard, ...]

    @property
    def knowledge_text(self) -> str:
        """Render whole selected sections in deterministic rank order."""
        return "\n\n".join(card.render() for card in self.selected_cards)

    @property
    def knowledge_metadata(self) -> list[dict[str, object]]:
        """Expose card identity and exact contents provenance for artifacts."""
        return [card.metadata() for card in self.selected_cards]

    def request_fields(
        self, *, include_action_catalog: Literal["none", "summary", "full"] = "none"
    ) -> dict[str, object]:
        """Return compact model-facing JSON fields for the applicable logical role."""
        fields: dict[str, object] = {
            "target_brief": self.target_brief.model_dump(mode="json"),
            "modeling_guidance": self.knowledge_text or None,
        }
        if include_action_catalog == "summary":
            fields["modeling_capabilities"] = action_capability_summary()
        elif include_action_catalog == "full":
            fields["action_catalog"] = action_catalog()
        return fields


class ModelingContextCompiler:
    """Select deterministic role-specific guidance from one shared card library."""

    def __init__(self, cards: tuple[ModelingKnowledgeCard, ...]) -> None:
        self._cards = cards

    def compile(
        self,
        *,
        role: KnowledgeRole,
        target_brief: TargetBrief,
        relevant_text: tuple[str, ...] = (),
    ) -> CompiledModelingContext:
        """Compile context without accessing model output, mutable scenes, or history."""
        return CompiledModelingContext(
            target_brief=target_brief,
            role=role,
            selected_cards=select_modeling_knowledge(
                self._cards,
                role=role,
                target_brief=target_brief,
                relevant_text=relevant_text,
            ),
        )
