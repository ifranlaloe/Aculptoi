from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aculptoi.checkpoints import CheckpointStore
from aculptoi.config import AcuConfig, BlenderConfig, load_config
from aculptoi.models import ProviderRegistry


def test_config_defaults_are_local_first(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml")
    assert config.actor.provider == "local"
    assert config.vision.provider == "local"
    assert config.provider_for("actor") is config.provider_for("vision")
    assert config.provider_for("actor").base_url == "http://127.0.0.1:8080/v1"
    assert config.provider_for("actor").timeout_seconds == 900.0
    assert config.actor.max_output_tokens == 1536
    assert config.vision.max_output_tokens == 8192
    assert config.vision.max_discovered_issues == 12
    assert config.vision.max_issue_analysis_requests == 12
    assert config.max_actor_requests_per_iteration == 100
    assert config.max_actions_per_iteration == 1000
    assert config.iteration_timeout_seconds == 3600.0
    assert config.blender.host == "127.0.0.1"


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


def test_shared_provider_vision_critique_limits_are_parsed() -> None:
    config = AcuConfig.model_validate(
        {
            "providers": {
                "local": {
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local-multimodal",
                }
            },
            "actor": {"provider": "local"},
            "vision": {
                "provider": "local",
                "max_output_tokens": 8192,
                "max_discovered_issues": 8,
                "max_issue_analysis_requests": 5,
            },
        }
    )

    assert config.actor.provider == config.vision.provider == "local"
    assert config.vision.max_output_tokens == 8192
    assert config.vision.max_discovered_issues == 8
    assert config.vision.max_issue_analysis_requests == 5


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
    assert config.max_iterations == 3


def test_legacy_execution_batch_limit_becomes_an_actor_request_safety_budget() -> None:
    config = AcuConfig.model_validate({"max_execution_batches_per_iteration": 4})

    assert config.max_actor_requests_per_iteration == 5


def test_worker_host_cannot_be_remote() -> None:
    with pytest.raises(ValidationError, match="localhost"):
        BlenderConfig(host="0.0.0.0")


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


def test_store_only_copies_worker_checkpoints_into_a_run_iteration(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run()
    store.checkpoints.mkdir(parents=True)
    worker_snapshot = store.checkpoints / "worker.blend"
    worker_snapshot.write_bytes(b"blend")

    copied = store.copy_checkpoint_to_iteration(run, 1, {"path": str(worker_snapshot)})

    assert copied.read_bytes() == b"blend"
    outside_snapshot = tmp_path / "outside.blend"
    outside_snapshot.write_bytes(b"not a worker checkpoint")
    with pytest.raises(ValueError, match="outside"):
        store.copy_checkpoint_to_iteration(run, 2, {"path": str(outside_snapshot)})


def test_store_scopes_work_item_checkpoints_to_the_item_directory(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run()
    store.checkpoints.mkdir(parents=True)
    worker_snapshot = store.checkpoints / "worker.blend"
    worker_snapshot.write_bytes(b"blend")

    copied = store.copy_checkpoint_to_work_item(
        run,
        iteration=1,
        ordinal=2,
        work_item_id="upper-layer",
        action_batch=3,
        snapshot={"path": str(worker_snapshot)},
    )

    assert copied.relative_to(run.path) == Path(
        "iteration-001/items/002-upper-layer/scene-003.blend"
    )
