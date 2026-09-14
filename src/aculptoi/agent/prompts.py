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


CONSTRUCTION_PLAN_PROMPT_VERSION, CONSTRUCTION_PLAN_SYSTEM_PROMPT = _load_prompt(
    "actor_construction_plan.md"
)
TARGET_BRIEF_PROMPT_VERSION, TARGET_BRIEF_SYSTEM_PROMPT = _load_prompt("actor_target_brief.md")
WORK_ITEM_PROMPT_VERSION, WORK_ITEM_SYSTEM_PROMPT = _load_prompt("actor_work_item.md")
ISSUE_DISCOVERY_PROMPT_VERSION, ISSUE_DISCOVERY_SYSTEM_PROMPT = _load_prompt(
    "vision_issue_discovery.md"
)
ISSUE_ANALYSIS_PROMPT_VERSION, ISSUE_ANALYSIS_SYSTEM_PROMPT = _load_prompt(
    "vision_issue_analysis.md"
)
INSPECTION_REVIEW_PROMPT_VERSION, INSPECTION_REVIEW_SYSTEM_PROMPT = _load_prompt(
    "inspection_reviewer.md"
)
