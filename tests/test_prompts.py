from __future__ import annotations

from aculptoi.agent.prompts import (
    CONSTRUCTION_PLAN_PROMPT_VERSION,
    CONSTRUCTION_PLAN_SYSTEM_PROMPT,
    INSPECTION_REVIEW_PROMPT_VERSION,
    INSPECTION_REVIEW_SYSTEM_PROMPT,
    ISSUE_ANALYSIS_PROMPT_VERSION,
    ISSUE_ANALYSIS_SYSTEM_PROMPT,
    ISSUE_DISCOVERY_PROMPT_VERSION,
    ISSUE_DISCOVERY_SYSTEM_PROMPT,
    TARGET_BRIEF_PROMPT_VERSION,
    TARGET_BRIEF_SYSTEM_PROMPT,
    WORK_ITEM_PROMPT_VERSION,
    WORK_ITEM_SYSTEM_PROMPT,
)


def test_role_prompts_are_loaded_from_versioned_markdown_templates() -> None:
    assert CONSTRUCTION_PLAN_PROMPT_VERSION == "v4"
    assert TARGET_BRIEF_PROMPT_VERSION == "v1"
    assert WORK_ITEM_PROMPT_VERSION == "v5"
    assert ISSUE_DISCOVERY_PROMPT_VERSION == "v5"
    assert ISSUE_ANALYSIS_PROMPT_VERSION == "v4"
    assert INSPECTION_REVIEW_PROMPT_VERSION == "v1"
    assert "Do not return Blender actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "Do not include `completion_criteria` or actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "transferable geometric and form characteristics" in TARGET_BRIEF_SYSTEM_PROMPT
    assert 'status: "continue"' in WORK_ITEM_SYSTEM_PROMPT
    assert "Cube dimensions are [0.9, 0.9, 0.9]." in WORK_ITEM_SYSTEM_PROMPT
    assert "must never replace or modify established criteria" in WORK_ITEM_SYSTEM_PROMPT
    assert "`scene` is the live source of truth" in WORK_ITEM_SYSTEM_PROMPT
    assert "`completed_work_items` provides the semantic lineage" in WORK_ITEM_SYSTEM_PROMPT
    assert "Do **not** use an `args` wrapper" in WORK_ITEM_SYSTEM_PROMPT
    assert "`action_catalog`" in WORK_ITEM_SYSTEM_PROMPT
    assert "authoritative action language" in WORK_ITEM_SYSTEM_PROMPT
    assert "No mutation from that\nfailed batch remains applied" in WORK_ITEM_SYSTEM_PROMPT
    assert "read-only visual issue discovery critic" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "[region, severity, confidence, tiles, observation]" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "C critical, H high, M medium, L low" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "Do not generate issue IDs" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "Work breadth-first" in ISSUE_DISCOVERY_SYSTEM_PROMPT
    assert "Analyze only that known issue" in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert "Do not discover, create, or enumerate unrelated" in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert '"desc"' in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert "Do not return an issue ID" in ISSUE_ANALYSIS_SYSTEM_PROMPT
    assert "technical quality of the supplied inspection atlas" in INSPECTION_REVIEW_SYSTEM_PROMPT
    assert "Do not judge model quality" in INSPECTION_REVIEW_SYSTEM_PROMPT
