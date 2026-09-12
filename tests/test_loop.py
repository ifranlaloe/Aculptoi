from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from aculptoi.agent import Actor, RefinementLoop, VisionCritic
from aculptoi.checkpoints import CheckpointStore
from aculptoi.models import ModelResponseError
from aculptoi.models.base import Message
from aculptoi.schemas.actions import Action


class FakeProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        assert messages
        return self.response


class SequencedProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        assert messages
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeBlender:
    def __init__(self, checkpoint_directory: Path) -> None:
        self._checkpoint_directory = checkpoint_directory
        self.executions: list[Sequence[Action]] = []
        self.render_calls = 0

    def scene_inspect(self) -> dict[str, object]:
        return {"objects": []}

    def execute(self, actions: Sequence[Action]) -> dict[str, object]:
        self.executions.append(actions)
        return {"executed": [{"command": action.command, "status": "ok"} for action in actions]}

    def render_views(
        self, views: Sequence[str], output_dir: Path, object_name: str | None = None
    ) -> dict[str, object]:
        self.render_calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for view in views:
            path = output_dir / f"{view}.png"
            Image.new("RGB", (32, 32), color="white").save(path)
            paths.append(str(path))
        return {"paths": paths}

    def checkpoint_save(self, name: str) -> dict[str, object]:
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_directory / f"{name}.blend"
        path.write_bytes(b"fake blend")
        return {"name": name, "path": str(path)}


class InvalidJsonProvider:
    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        raise ModelResponseError("Model response was not valid JSON", "<think>unfinished</think>")


def test_refinement_loop_persists_an_inspectable_iteration(tmp_path: Path) -> None:
    actor = Actor(
        FakeProvider(
            {
                "reason": "Add a body primitive.",
                "actions": [{"command": "object.create", "name": "Body", "primitive": "uv_sphere"}],
            }
        )
    )
    critic = VisionCritic(FakeProvider({"score": 0.95, "summary": "Goal met.", "issues": []}))
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=actor,
        critic=critic,
        blender=FakeBlender(store.checkpoints),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=3,
        score_target=0.9,
    )

    result = loop.run("create a sphere creature")

    assert result.completed is True
    assert result.iterations == 1
    assert (result.run_directory / "user-prompt.txt").read_text() == "create a sphere creature"
    assert (result.run_directory / "actor-plan-001-batch-001.json").is_file()
    assert (result.run_directory / "critique-001.json").is_file()
    iteration = result.run_directory / "iteration-001"
    batch = iteration / "batch-001"
    assert (batch / "actor-prompt.json").is_file()
    assert (batch / "actor-plan.json").is_file()
    assert (batch / "actions.json").is_file()
    assert (batch / "checkpoint.json").is_file()
    assert (batch / "scene.blend").read_bytes() == b"fake blend"
    assert (iteration / "vision-prompt.json").is_file()
    assert (iteration / "vision-analysis.json").is_file()
    assert (iteration / "checkpoint.json").is_file()
    assert (iteration / "perspective.png").is_file()
    assert (iteration / "scene.blend").read_bytes() == b"fake blend"
    actor_prompt = json.loads((batch / "actor-prompt.json").read_text())
    vision_prompt = json.loads((iteration / "vision-prompt.json").read_text())
    assert actor_prompt["role"] == "actor"
    assert vision_prompt["role"] == "vision_critic"
    assert vision_prompt["views"][0]["name"] == "front"
    assert "data:image" not in (iteration / "vision-prompt.json").read_text()


def test_refinement_loop_records_raw_actor_failures_by_default(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(InvalidJsonProvider()),
        critic=VisionCritic(FakeProvider({"score": 1.0, "summary": "Unused.", "issues": []})),
        blender=FakeBlender(store.checkpoints),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(ModelResponseError, match="not valid JSON"):
        loop.run("create a creature")

    run = store.runs / "000001"
    assert (run / "user-prompt.txt").read_text() == "create a creature"
    assert (run / "actor-error-001-batch-001.json").is_file()
    assert (run / "iteration-001" / "batch-001" / "actor-response-raw.txt").read_text() == (
        "<think>unfinished</think>"
    )


def test_construction_batches_render_once_before_a_visual_refinement(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            {
                "reason": "Create the first half of the grid.",
                "ready_for_inspection": False,
                "actions": [
                    {
                        "command": "object.create",
                        "name": "CubieA",
                        "primitive": "cube",
                        "location": [-0.5, 0.0, 0.0],
                        "scale": [0.4, 0.4, 0.4],
                    }
                ],
            },
            {
                "reason": "Finish the grid before inspection.",
                "ready_for_inspection": True,
                "actions": [
                    {
                        "command": "object.create",
                        "name": "CubieB",
                        "primitive": "cube",
                        "location": [0.5, 0.0, 0.0],
                        "scale": [0.4, 0.4, 0.4],
                    }
                ],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender(store.checkpoints)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 0.95, "summary": "Goal met.", "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        max_execution_batches_per_iteration=3,
        score_target=0.9,
    )

    result = loop.run("create a two-cubie grid")

    iteration = result.run_directory / "iteration-001"
    second_context = json.loads(actor_provider.calls[1][1]["content"])
    assert result.completed is True
    assert result.iterations == 1
    assert result.execution_batches == 2
    assert len(blender.executions) == 2
    assert blender.render_calls == 1
    assert second_context["execution_batch"] == 2
    assert second_context["recent_execution"] is not None
    assert (iteration / "batch-001" / "scene.blend").is_file()
    assert (iteration / "batch-002" / "scene.blend").is_file()
    assert (iteration / "vision-analysis.json").is_file()


def test_batch_limit_forces_one_inspection_milestone(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    blender = FakeBlender(store.checkpoints)
    loop = RefinementLoop(
        actor=Actor(
            SequencedProvider(
                [
                    {
                        "reason": "Continue initial construction.",
                        "ready_for_inspection": False,
                        "actions": [
                            {"command": "object.create", "name": "Cubie", "primitive": "cube"}
                        ],
                    }
                ]
            )
        ),
        critic=VisionCritic(FakeProvider({"score": 0.95, "summary": "Goal met.", "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        max_execution_batches_per_iteration=1,
        score_target=0.9,
    )

    result = loop.run("create a cubie")

    checkpoint = json.loads(
        (result.run_directory / "iteration-001" / "checkpoint.json").read_text()
    )
    assert result.execution_batches == 1
    assert blender.render_calls == 1
    assert checkpoint["inspection_forced_by_batch_limit"] is True
