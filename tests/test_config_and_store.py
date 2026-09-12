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
    assert config.provider_for("actor").timeout_seconds == 300.0
    assert config.actor.max_output_tokens == 1536
    assert config.vision.max_output_tokens == 768
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
