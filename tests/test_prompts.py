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
    assert CONSTRUCTION_PLAN_PROMPT_VERSION == "v7"
    assert TARGET_BRIEF_PROMPT_VERSION == "v1"
    assert WORK_ITEM_PROMPT_VERSION == "v8"
    assert ISSUE_DISCOVERY_PROMPT_VERSION == "v5"
    assert ISSUE_ANALYSIS_PROMPT_VERSION == "v4"
    assert INSPECTION_REVIEW_PROMPT_VERSION == "v1"
    assert "Do not return Blender actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "Do not include `completion_criteria` or actions" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "`form_traits`" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "transferable geometric and form characteristics" in TARGET_BRIEF_SYSTEM_PROMPT
    assert '"kind": "modeling_step"' in WORK_ITEM_SYSTEM_PROMPT
    assert '"kind": "observation_request"' in WORK_ITEM_SYSTEM_PROMPT
    assert '"kind": "complete"' in WORK_ITEM_SYSTEM_PROMPT
    assert "LOOK" in WORK_ITEM_SYSTEM_PROMPT
    assert "UNDERSTAND THE CURRENT FORM" in WORK_ITEM_SYSTEM_PROMPT
    assert "MAKE THAT CHANGE" in WORK_ITEM_SYSTEM_PROMPT
    assert "never replace them" in WORK_ITEM_SYSTEM_PROMPT
    assert "`args`\nwrapper" in WORK_ITEM_SYSTEM_PROMPT
    assert "`action_catalog`" in WORK_ITEM_SYSTEM_PROMPT
    assert "`action_semantics`" in WORK_ITEM_SYSTEM_PROMPT
    assert "Existing scene objects are possible modeling material, not obligations" in (
        WORK_ITEM_SYSTEM_PROMPT
    )
    assert "The default Cube has no\nspecial status" in WORK_ITEM_SYSTEM_PROMPT
    assert "Do not use smoothing as a substitute" in WORK_ITEM_SYSTEM_PROMPT
    assert "Do not use remeshing merely as a generic way" in WORK_ITEM_SYSTEM_PROMPT
    assert "First inspect what changed" in WORK_ITEM_SYSTEM_PROMPT
    assert "join+\nvoxel-remesh" not in WORK_ITEM_SYSTEM_PROMPT
    assert "transform+smooth" not in WORK_ITEM_SYSTEM_PROMPT
    assert "create+scale" not in WORK_ITEM_SYSTEM_PROMPT
    assert "Turn the default Cube into one cubie" not in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "default Cube\nhas no special status" in CONSTRUCTION_PLAN_SYSTEM_PROMPT
    assert "reshape it, replace it, or use a different suitable starting" in (
        CONSTRUCTION_PLAN_SYSTEM_PROMPT
    )
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
