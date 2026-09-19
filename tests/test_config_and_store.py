from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aculptoi.checkpoints import CheckpointStore, RunStateError
from aculptoi.cli.app import Runtime, _selected_blender_mode
from aculptoi.config import AcuConfig, BlenderConfig, load_config
from aculptoi.models import ProviderRegistry


def test_config_defaults_are_local_first(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml")
    assert config.actor.provider == "local"
    assert config.vision.provider == "local"
    assert config.provider_for("actor") is config.provider_for("vision")
    assert config.provider_for("actor").base_url == "http://127.0.0.1:8080/v1"
    assert config.provider_for("actor").timeout_seconds == 900.0
    assert config.actor.thinking is True
    assert config.actor.max_output_tokens == 16_384
    assert config.actor.reasoning_effort == "medium"
    assert config.vision.max_image_dimension == 4096
    assert config.vision.inspection_review.thinking is False
    assert config.vision.inspection_review.reasoning_effort is None
    assert config.vision.inspection_review.max_output_tokens == 4_096
    assert config.vision.discovery.thinking is False
    assert config.vision.discovery.reasoning_effort is None
    assert config.vision.discovery.max_output_tokens == 4_096
    assert config.vision.issue_analysis.thinking is True
    assert config.vision.issue_analysis.reasoning_effort == "medium"
    assert config.vision.issue_analysis.max_output_tokens == 16_384
    assert config.vision.max_discovered_issues == 12
    assert config.vision.max_issue_analysis_requests == 12
    assert config.inspection.min_views == 8
    assert config.inspection.max_views == 24
    assert config.inspection.candidate_views == 64
    assert config.inspection.coverage_target == 0.95
    assert config.inspection.min_tile_dimension == 768
    assert config.inspection.sensor_version == "inspection-atlas-v1"
    assert config.max_actor_requests_per_iteration == 100
    assert config.max_actions_per_iteration == 1000
    assert config.max_actor_observations_per_work_item == 12
    assert config.iteration_timeout_seconds == 3600.0
    assert config.blender.host == "127.0.0.1"
    assert config.blender.mode == "ui"


def test_actor_and_vision_can_share_one_named_provider() -> None:
    config = AcuConfig.model_validate(
        {
            "providers": {
                "local": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local-multimodal",
                }
            },
            "actor": {"provider": "local"},
            "vision": {"provider": "local"},
        }
    )
    registry = ProviderRegistry(config.providers)
    try:
        assert registry.get(config.actor.provider) is registry.get(config.vision.provider)
    finally:
        registry.close()


@pytest.mark.parametrize(
    "inspection",
    [
        {"min_views": 9, "max_views": 8},
        {"min_views": 9, "max_views": 10, "max_views_per_round": 8},
        {"min_views": 9, "max_views": 10, "max_total_views": 8},
        {"min_views": 9, "max_views": 10, "candidate_views": 8},
    ],
)
def test_inspection_configuration_rejects_incoherent_view_limits(
    inspection: dict[str, int],
) -> None:
    with pytest.raises(ValidationError):
        AcuConfig.model_validate({"inspection": inspection})


def test_inspection_configuration_rejects_unsupported_worker_protocol_values() -> None:
    with pytest.raises(ValidationError):
        AcuConfig.model_validate({"inspection": {"lighting_rig": "artistic-studio-v1"}})
    with pytest.raises(ValidationError):
        AcuConfig.model_validate({"inspection": {"sensor_version": "inspection atlas v1"}})


def test_shared_provider_role_output_limits_are_parsed() -> None:
    config = AcuConfig.model_validate(
        {
            "providers": {
                "local": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local-multimodal",
                }
            },
            "actor": {"provider": "local", "max_output_tokens": 16_384},
            "vision": {
                "provider": "local",
                "max_output_tokens": 16_384,
                "max_discovered_issues": 8,
                "max_issue_analysis_requests": 5,
            },
        }
    )

    assert config.actor.provider == config.vision.provider == "local"
    assert config.actor.max_output_tokens == 16_384
    assert config.vision.max_output_tokens == 16_384
    assert config.actor.reasoning_effort == "medium"
    assert config.vision.discovery.thinking is False
    assert config.vision.discovery.reasoning_effort is None
    assert config.vision.discovery.max_output_tokens == 4_096
    assert config.vision.issue_analysis.reasoning_effort == "medium"
    assert config.vision.issue_analysis.max_output_tokens == 16_384
    assert config.vision.max_discovered_issues == 8
    assert config.vision.max_issue_analysis_requests == 5
    assert config.inspection.max_rounds == 3


def test_actor_and_vision_can_select_separate_providers() -> None:
    config = AcuConfig.model_validate(
        {
            "providers": {
                "actor-model": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local-actor",
                },
                "vision-model": {
                    "base_url": "http://127.0.0.1:8081/v1",
                    "model": "local-vision",
                },
            },
            "actor": {"provider": "actor-model"},
            "vision": {"provider": "vision-model"},
        }
    )
    registry = ProviderRegistry(config.providers)
    try:
        assert registry.get(config.actor.provider) is not registry.get(config.vision.provider)
    finally:
        registry.close()


def test_legacy_separate_role_endpoint_configuration_still_loads(tmp_path: Path) -> None:
    config_path = tmp_path / "aculptoi.toml"
    config_path.write_text(
        """[actor]
base_url = "http://127.0.0.1:8080/v1"
model = "legacy-actor"

[vision]
base_url = "http://127.0.0.1:8081/v1"
model = "legacy-vision"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.actor.provider == "actor-legacy"
    assert config.vision.provider == "vision-legacy"
    assert config.provider_for("actor").model == "legacy-actor"
    assert config.provider_for("vision").model == "legacy-vision"


def test_shared_provider_example_is_valid() -> None:
    example = Path(__file__).parents[1] / "examples" / "simple_creature" / "aculptoi.toml"

    config = load_config(example)

    assert config.actor.provider == "local"
    assert config.vision.provider == "local"
    assert config.actor.max_output_tokens == 16_384
    assert config.actor.reasoning_effort == "medium"
    assert config.vision.discovery.max_output_tokens == 4_096
    assert config.vision.issue_analysis.max_output_tokens == 16_384
    assert config.max_iterations == 3


def test_legacy_execution_batch_limit_becomes_an_actor_request_safety_budget() -> None:
    config = AcuConfig.model_validate({"max_execution_batches_per_iteration": 4})

    assert config.max_actor_requests_per_iteration == 5


def test_role_output_token_limits_allow_small_and_modern_values() -> None:
    config = AcuConfig.model_validate(
        {
            "actor": {"max_output_tokens": 4_096},
            "vision": {"max_output_tokens": 16_384},
        }
    )

    assert config.actor.max_output_tokens == 4_096
    assert config.vision.max_output_tokens == 16_384


def test_vision_stage_profiles_accept_explicit_independent_values() -> None:
    config = AcuConfig.model_validate(
        {
            "actor": {"thinking": False},
            "vision": {
                "inspection_review": {
                    "thinking": True,
                    "reasoning_effort": "high",
                    "max_output_tokens": 512,
                },
                "discovery": {"thinking": False, "max_output_tokens": 1_024},
                "issue_analysis": {
                    "thinking": True,
                    "reasoning_effort": "xhigh",
                    "max_output_tokens": 32_768,
                },
            },
        }
    )

    assert config.actor.thinking is False
    assert config.actor.reasoning_effort is None
    assert config.vision.inspection_review.thinking is True
    assert config.vision.inspection_review.reasoning_effort == "high"
    assert config.vision.inspection_review.max_output_tokens == 512
    assert config.vision.discovery.thinking is False
    assert config.vision.discovery.reasoning_effort is None
    assert config.vision.discovery.max_output_tokens == 1_024
    assert config.vision.issue_analysis.reasoning_effort == "xhigh"
    assert config.vision.issue_analysis.max_output_tokens == 32_768


def test_actor_accepts_a_value_above_its_former_output_ceiling() -> None:
    config = AcuConfig.model_validate({"actor": {"max_output_tokens": 8_193}})

    assert config.actor.max_output_tokens == 8_193


def test_legacy_stage_profiles_without_thinking_use_current_stage_defaults() -> None:
    config = AcuConfig.model_validate(
        {
            "vision": {
                "discovery": {"reasoning_effort": "low", "max_output_tokens": 1_024},
                "issue_analysis": {"reasoning_effort": "high", "max_output_tokens": 8_192},
            }
        }
    )

    assert config.vision.discovery.thinking is False
    assert config.vision.discovery.reasoning_effort is None
    assert config.vision.discovery.max_output_tokens == 1_024
    assert config.vision.issue_analysis.thinking is True
    assert config.vision.issue_analysis.reasoning_effort == "high"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_role_reasoning_effort_accepts_supported_values(effort: str) -> None:
    config = AcuConfig.model_validate(
        {
            "actor": {"reasoning_effort": effort},
            "vision": {"reasoning_effort": effort},
        }
    )

    assert config.actor.reasoning_effort == effort
    assert config.vision.reasoning_effort == effort


def test_role_reasoning_effort_rejects_unsupported_values() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        AcuConfig.model_validate({"actor": {"reasoning_effort": "maximum"}})
    with pytest.raises(ValidationError, match="literal_error"):
        AcuConfig.model_validate(
            {"vision": {"discovery": {"reasoning_effort": "maximum", "max_output_tokens": 4_096}}}
        )
    with pytest.raises(ValidationError, match="must be omitted"):
        AcuConfig.model_validate({"actor": {"thinking": False, "reasoning_effort": "medium"}})
    with pytest.raises(ValidationError, match="must be omitted"):
        AcuConfig.model_validate(
            {
                "vision": {
                    "discovery": {
                        "thinking": False,
                        "reasoning_effort": "low",
                        "max_output_tokens": 4_096,
                    }
                }
            }
        )


@pytest.mark.parametrize("role", ["actor", "vision"])
def test_role_output_token_limit_enforces_the_shared_ceiling(role: str) -> None:
    with pytest.raises(ValidationError):
        AcuConfig.model_validate({role: {"max_output_tokens": 65_537}})


def test_worker_host_cannot_be_remote() -> None:
    with pytest.raises(ValidationError, match="localhost"):
        BlenderConfig(host="0.0.0.0")


def test_ui_mode_is_default_and_cli_flags_override_configuration(tmp_path: Path) -> None:
    runtime = Runtime(config=AcuConfig(), project_dir=tmp_path)

    assert _selected_blender_mode(runtime, ui=False, headless=False) == "ui"
    assert _selected_blender_mode(runtime, ui=True, headless=False) == "ui"
    assert _selected_blender_mode(runtime, ui=False, headless=True) == "headless"
    with pytest.raises(Exception, match="cannot be used together"):
        _selected_blender_mode(runtime, ui=True, headless=True)


def test_store_allocates_run_and_writes_inspectable_metadata(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run()
    metadata = store.save_metadata(run, "actor-plan-001.json", {"reason": "test"})

    assert run.id == 1
    assert metadata.relative_to(tmp_path) == Path(".aculptoi/runs/000001/actor-plan-001.json")
    assert json.loads(metadata.read_text())["reason"] == "test"


def test_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    with pytest.raises(ValueError):
        store.save_metadata(store.create_run(), "../escape.json", {})


def test_store_can_make_construction_plan_artifacts_immutable(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run()
    store.save_json_artifact(
        run,
        "iteration-001/construction-plan.json",
        {"plan": "original"},
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        store.save_json_artifact(
            run,
            "iteration-001/construction-plan.json",
            {"plan": "replacement"},
            overwrite=False,
        )


def test_runs_have_distinct_canonical_scenes_and_run_local_checkpoints(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    first = store.create_run("first")
    second = store.create_run("second")
    first_scene = store.canonical_scene_path(first)
    second_scene = store.canonical_scene_path(second)
    first_scene.write_bytes(b"first scene")
    second_scene.write_bytes(b"second scene")

    first_checkpoint = store.create_checkpoint(first, iteration=1, ordinal=1, work_item_id="body")
    second_checkpoint = store.create_checkpoint(second, iteration=1, ordinal=1, work_item_id="body")

    assert first_scene != second_scene
    assert first_checkpoint.checkpoint == "checkpoints/item-001-001-body.blend"
    assert (first.path / first_checkpoint.checkpoint).read_bytes() == b"first scene"
    assert (second.path / second_checkpoint.checkpoint).read_bytes() == b"second scene"


def test_restore_latest_checkpoint_preserves_partial_canonical_scene(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("recover")
    canonical = store.canonical_scene_path(run)
    canonical.write_bytes(b"durable scene")
    checkpoint = store.create_checkpoint(run, iteration=1, ordinal=1, work_item_id="body")
    state = store.load_run_state(run).model_copy(
        update={
            "latest_completed_item": checkpoint,
            "latest_checkpoint": checkpoint.checkpoint,
            "durable_items": [checkpoint],
        }
    )
    store.save_run_state(run, state)
    canonical.write_bytes(b"partial active item")

    restored = store.restore_latest_checkpoint(run)

    assert restored == canonical
    assert canonical.read_bytes() == b"durable scene"
    assert list((run.path / "recovery").glob("abandoned-*.blend"))


def test_missing_checkpoint_referenced_by_state_fails_safely(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("broken")
    store.save_run_state(
        run,
        store.load_run_state(run).model_copy(
            update={"latest_checkpoint": "checkpoints/missing.blend"}
        ),
    )

    with pytest.raises(RunStateError, match="missing"):
        store.restore_latest_checkpoint(run)
