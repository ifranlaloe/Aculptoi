from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aculptoi.agent import Actor, RefinementLoop, VisionCritic
from aculptoi.checkpoints import CheckpointStore
from aculptoi.models.base import Message
from aculptoi.schemas.actions import Action


class FakeProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def complete_json(self, messages: Sequence[Message]) -> dict[str, object]:
        assert messages
        return self.response


class FakeBlender:
    def scene_inspect(self) -> dict[str, object]:
        return {"objects": []}

    def execute(self, actions: Sequence[Action]) -> dict[str, object]:
        return {"executed": [{"command": action.command, "status": "ok"} for action in actions]}

    def render_views(
        self, views: Sequence[str], output_dir: Path, object_name: str | None = None
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True)
        paths: list[str] = []
        for view in views:
            path = output_dir / f"{view}.png"
            path.write_bytes(b"fake png")
            paths.append(str(path))
        return {"paths": paths}

    def checkpoint_save(self, name: str) -> dict[str, object]:
        return {"name": name, "path": f"/fake/{name}.blend"}


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
    loop = RefinementLoop(
        actor=actor,
        critic=critic,
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=CheckpointStore(tmp_path),
        max_iterations=3,
        score_target=0.9,
    )

    result = loop.run("create a sphere creature")

    assert result.completed is True
    assert result.iterations == 1
    assert (result.run_directory / "actor-plan-001.json").is_file()
    assert (result.run_directory / "critique-001.json").is_file()
    assert (result.run_directory / "iteration-001" / "perspective.png").is_file()
