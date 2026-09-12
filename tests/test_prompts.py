from __future__ import annotations

from aculptoi.agent.prompts import (
    ACTOR_PROMPT_VERSION,
    ACTOR_SYSTEM_PROMPT,
    CRITIC_PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
)


def test_role_prompts_are_loaded_from_versioned_markdown_templates() -> None:
    assert ACTOR_PROMPT_VERSION == "v2"
    assert CRITIC_PROMPT_VERSION == "v1"
    assert "ready_for_inspection" in ACTOR_SYSTEM_PROMPT
    assert "Do **not** use an `args` wrapper" in ACTOR_SYSTEM_PROMPT
    assert "read-only 3D inspection role" in CRITIC_SYSTEM_PROMPT
