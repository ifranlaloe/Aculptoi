from __future__ import annotations

from aculptoi.agent.prompts import (
    CONSTRUCTION_PLAN_PROMPT_VERSION,
    CONSTRUCTION_PLAN_SYSTEM_PROMPT,
    ISSUE_ANALYSIS_PROMPT_VERSION,
    ISSUE_ANALYSIS_SYSTEM_PROMPT,
    ISSUE_DISCOVERY_PROMPT_VERSION,
    ISSUE_DISCOVERY_SYSTEM_PROMPT,
    WORK_ITEM_PROMPT_VERSION,
    WORK_ITEM_SYSTEM_PROMPT,
)


def test_role_prompts_are_loaded_from_versioned_markdown_templates() -> None:
    assert CONSTRUCTION_PLAN_PROMPT_VERSION == "v3"
    assert WORK_ITEM_PROMPT_VERSION == "v3"
    assert ISSUE_DISCOVERY_PROMPT_VERSION == "v2"
    assert ISSUE_ANALYSIS_PROMPT_VERSION == "v2"
    assert "Do not return Blender actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "Do not include `completion_criteria` or actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert 'status: "continue"' in WORK_ITEM_SYSTEM_PROMPT
    assert '"completion_criteria": [' in WORK_ITEM_SYSTEM_PROMPT
    assert "must never replace or modify established criteria" in WORK_ITEM_SYSTEM_PROMPT
    assert "`scene` is the live source of truth" in WORK_ITEM_SYSTEM_PROMPT
    assert "`completed_work_items` provides the semantic lineage" in WORK_ITEM_SYSTEM_PROMPT
    assert "Do **not** use an `args` wrapper" in WORK_ITEM_SYSTEM_PROMPT
    assert "read-only visual issue discovery critic" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "[region, severity, confidence, views, observation]" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "C critical, H high, M medium, L low" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "Do not generate issue IDs" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "Analyze only that known issue" in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert "Do not discover, create, or enumerate unrelated" in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert '"desc"' in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert "Do not return an issue ID" in ISSUE_ANALYSIS_SYSTEM_PROMPT
