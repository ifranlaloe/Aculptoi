"""Load versioned, human-editable role prompts packaged with Aculptoi."""

from __future__ import annotations

import re
from importlib.resources import files

_VERSION_MARKER = re.compile(r"^<!-- aculptoi-prompt-version: (v[0-9]+) -->\n")


def _load_prompt(filename: str) -> tuple[str, str]:
    """Read one packaged Markdown prompt and validate its lightweight version marker."""
    source = files("aculptoi.agent").joinpath("prompt_templates", filename)
    text = source.read_text(encoding="utf-8")
    match = _VERSION_MARKER.match(text)
    if match is None:
        raise RuntimeError(f"Prompt template {filename} is missing its version marker")
    return match.group(1), text[match.end() :].strip()


ACTOR_PROMPT_VERSION, ACTOR_SYSTEM_PROMPT = _load_prompt("actor.md")
CRITIC_PROMPT_VERSION, CRITIC_SYSTEM_PROMPT = _load_prompt("vision_critic.md")
