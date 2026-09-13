from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from aculptoi.agent import Actor, RefinementLoop, VisionCritic
from aculptoi.agent.loop import IterationBudgetExceeded
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


class MixedProvider:
    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self._responses = responses

    def complete_json(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        assert messages
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeBlender:
    def __init__(self, checkpoint_directory: Path) -> None:
        self._checkpoint_directory = checkpoint_directory
        self._objects: dict[str, dict[str, object]] = {}
        self.executions: list[Sequence[Action]] = []
        self.render_calls = 0

    def scene_inspect(self) -> dict[str, object]:
        return {"objects": list(self._objects.values())}

    def execute(self, actions: Sequence[Action]) -> dict[str, object]:
        self.executions.append(actions)
        for action in actions:
            payload = action.model_dump(mode="json")
            if action.command == "object.create":
                self._objects[payload["name"]] = {
                    "name": payload["name"],
                    "location": payload["location"],
                    "scale": payload["scale"],
                    "type": "MESH",
                }
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


def _one_item_plan(item_id: str = "body") -> dict[str, object]:
    return {
        "reason": "Build one logical component.",
        "items": [
            {
                "id": item_id,
                "title": item_id.replace("-", " ").title(),
                "objective": f"Complete {item_id}.",
                "depends_on": [],
            }
        ],
    }


def test_refinement_loop_persists_plan_first_item_artifacts(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create the body primitive.",
                "completion_criteria": ["A body object exists."],
                "actions": [{"command": "object.create", "name": "Body", "primitive": "uv_sphere"}],
            },
        ]
    )
    critic = VisionCritic(FakeProvider({"score": 0.95, "summary": "Goal met.", "issues": []}))
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=critic,
        blender=FakeBlender(store.checkpoints),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=3,
        score_target=0.9,
    )

    result = loop.run("create a sphere creature")

    assert result.completed is True
    assert result.iterations == 1
    assert result.execution_batches == 1
    assert (result.run_directory / "user-prompt.txt").read_text() == "create a sphere creature"
    assert (result.run_directory / "critique-001.json").is_file()
    iteration = result.run_directory / "iteration-001"
    item = iteration / "items" / "001-body"
    assert (iteration / "construction-plan-prompt.json").is_file()
    assert (iteration / "construction-plan.json").is_file()
    assert (item / "item.json").is_file()
    assert (item / "completion-criteria.json").is_file()
    assert (item / "actor-prompt-001.json").is_file()
    assert (item / "action-batch-001.json").is_file()
    assert (item / "action-result-001.json").is_file()
    assert (item / "checkpoint-001.json").is_file()
    assert (item / "scene-001.blend").read_bytes() == b"fake blend"
    assert (item / "summary.json").is_file()
    critic_directory = iteration / "critic"
    assert (critic_directory / "discovery-prompt.json").is_file()
    assert (critic_directory / "discovery.json").is_file()
    assert (iteration / "vision-analysis.json").is_file()
    assert (iteration / "iteration-summary.json").is_file()
    assert (iteration / "checkpoint.json").is_file()
    assert (iteration / "perspective.png").is_file()
    assert (iteration / "scene.blend").read_bytes() == b"fake blend"

    plan_prompt = json.loads((iteration / "construction-plan-prompt.json").read_text())
    item_prompt = json.loads((item / "actor-prompt-001.json").read_text())
    discovery_prompt = json.loads((critic_directory / "discovery-prompt.json").read_text())
    action_batch = json.loads((item / "action-batch-001.json").read_text())
    assert plan_prompt["request_type"] == "construction_plan"
    assert item_prompt["request_type"] == "work_item_actions"
    assert action_batch["construction_plan_id"] == "run-000001-iteration-001"
    assert action_batch["work_item_id"] == "body"
    assert action_batch["response"]["completion_criteria"] == ["A body object exists."]
    assert discovery_prompt["role"] == "vision_critic"
    assert discovery_prompt["request_type"] == "vision_issue_discovery"
    assert discovery_prompt["views"][0]["name"] == "front"
    assert "data:image" not in (critic_directory / "discovery-prompt.json").read_text()
    assert len(actor_provider.calls) == 2


def test_refinement_loop_records_raw_construction_plan_failures(tmp_path: Path) -> None:
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

    iteration = store.runs / "000001" / "iteration-001"
    assert (store.runs / "000001" / "user-prompt.txt").read_text() == "create a creature"
    assert (iteration / "construction-plan-error.json").is_file()
    assert (iteration / "construction-plan-response-raw.txt").read_text() == (
        "<think>unfinished</think>"
    )


def test_refinement_loop_keeps_other_issue_analyses_when_one_is_malformed(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create the body primitive.",
                "completion_criteria": ["A body object exists."],
                "actions": [{"command": "object.create", "name": "Body", "primitive": "cube"}],
            },
        ]
    )
    critic_provider = MixedProvider(
        [
            {
                "score": 0.95,
                "summary": "Two observations remain.",
                "issues": [
                    {
                        "id": "issue-001",
                        "title": "Left wing intersects torso",
                        "region": "left-wing",
                        "severity": "high",
                        "confidence": 0.9,
                        "evidence_views": ["front", "perspective"],
                    },
                    {
                        "id": "issue-002",
                        "title": "Neck is too short",
                        "region": "neck",
                        "severity": "medium",
                        "confidence": 0.8,
                        "evidence_views": ["right", "perspective"],
                    },
                ],
            },
            {
                "id": "issue-001",
                "description": "This payload improperly includes an action.",
                "evidence": ["Visible in front."],
                "likely_cause": None,
                "suggested_correction": "Separate the forms.",
                "success_criteria": ["The silhouettes are distinct."],
                "confidence": 0.9,
                "actions": [{"command": "object.delete"}],
            },
            {
                "id": "issue-002",
                "description": "The neck has little visible length before the head.",
                "evidence": ["The right silhouette compresses the neck."],
                "likely_cause": "The neck primitive is too short.",
                "suggested_correction": "Lengthen and taper the neck.",
                "success_criteria": ["The right silhouette shows a distinct neck."],
                "confidence": 0.88,
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(critic_provider),
        blender=FakeBlender(store.checkpoints),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a creature")

    iteration = result.run_directory / "iteration-001"
    critic_directory = iteration / "critic" / "issues"
    assembled = json.loads((iteration / "vision-analysis.json").read_text())
    assert result.completed is True
    assert (critic_directory / "issue-001" / "summary.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-prompt.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-error.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-response-raw.txt").is_file()
    assert (critic_directory / "issue-002" / "analysis.json").is_file()
    assert assembled["issues"][0]["detail_status"] == "analysis_failed"
    assert assembled["issues"][1]["detail_status"] == "detailed"


def test_work_items_can_use_multiple_action_batches_before_one_visual_inspection(
    tmp_path: Path,
) -> None:
    actor_provider = SequencedProvider(
        [
            {
                "reason": "Build a small Rubik-style grid by logical component.",
                "items": [
                    {
                        "id": "left-cubie",
                        "title": "Left cubie",
                        "objective": "Create and size the left cubie.",
                        "depends_on": [],
                    },
                    {
                        "id": "right-cubie",
                        "title": "Right cubie",
                        "objective": "Create the right cubie beside the left cubie.",
                        "depends_on": ["left-cubie"],
                    },
                ],
            },
            {
                "work_item_id": "left-cubie",
                "status": "continue",
                "reason": "Create the left cubie first.",
                "completion_criteria": ["The left cubie has its target size."],
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
                "work_item_id": "left-cubie",
                "status": "complete",
                "reason": "Finish the left cubie's proportions.",
                "actions": [
                    {"command": "object.scale", "object": "CubieA", "scale": [1.0, 1.0, 1.0]}
                ],
            },
            {
                "work_item_id": "right-cubie",
                "status": "complete",
                "reason": "Create the dependent right cubie.",
                "completion_criteria": ["Both cubies form a row."],
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
        score_target=0.9,
    )

    result = loop.run("create a two-cubie Rubik-style row")

    iteration = result.run_directory / "iteration-001"
    first_item = iteration / "items" / "001-left-cubie"
    second_item = iteration / "items" / "002-right-cubie"
    second_batch_context = json.loads(actor_provider.calls[2][1]["content"])
    second_item_context = json.loads(actor_provider.calls[3][1]["content"])
    assert result.completed is True
    assert result.execution_batches == 3
    assert len(blender.executions) == 3
    assert blender.render_calls == 1
    assert second_batch_context["active_work_item"]["id"] == "left-cubie"
    assert second_batch_context["recent_execution"] is not None
    assert second_batch_context["completion_criteria"] == ["The left cubie has its target size."]
    assert second_item_context["completed_work_item_ids"] == ["left-cubie"]
    assert second_item_context["completed_work_items"] == [
        {
            "affected_object_names": ["CubieA"],
            "completion_criteria": ["The left cubie has its target size."],
            "created_object_names": ["CubieA"],
            "id": "left-cubie",
            "objective": "Create and size the left cubie.",
            "title": "Left cubie",
        }
    ]
    assert second_item_context["scene"]["objects"] == [
        {
            "location": [-0.5, 0.0, 0.0],
            "name": "CubieA",
            "scale": [0.4, 0.4, 0.4],
            "type": "MESH",
        }
    ]
    assert (first_item / "action-batch-001.json").is_file()
    assert (first_item / "action-batch-002.json").is_file()
    assert (second_item / "action-batch-001.json").is_file()
    assert (iteration / "vision-analysis.json").is_file()


def test_actor_request_budget_stops_a_nonterminating_work_item(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan("cubie"),
            {
                "work_item_id": "cubie",
                "status": "continue",
                "reason": "Start the cubie but request more work.",
                "completion_criteria": ["A cubie object exists."],
                "actions": [{"command": "object.create", "name": "Cubie", "primitive": "cube"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender(store.checkpoints)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 0.95, "summary": "Unused.", "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
        max_actor_requests_per_iteration=2,
    )

    with pytest.raises(IterationBudgetExceeded, match="max_actor_requests_per_iteration"):
        loop.run("create a cubie")

    iteration = store.runs / "000001" / "iteration-001"
    budget = json.loads((iteration / "budget-exhausted.json").read_text())
    assert budget["actor_requests"] == 2
    assert len(blender.executions) == 1
    assert blender.render_calls == 0


def test_action_budget_stops_before_an_oversized_scene_mutation(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan("cubies"),
            {
                "work_item_id": "cubies",
                "status": "complete",
                "reason": "Create two cubies.",
                "completion_criteria": ["Two cubie objects exist."],
                "actions": [
                    {"command": "object.create", "name": "CubieA", "primitive": "cube"},
                    {"command": "object.create", "name": "CubieB", "primitive": "cube"},
                ],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender(store.checkpoints)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 0.95, "summary": "Unused.", "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
        max_actions_per_iteration=1,
    )

    with pytest.raises(IterationBudgetExceeded, match="max_actions_per_iteration"):
        loop.run("create two cubies")

    iteration = store.runs / "000001" / "iteration-001"
    assert (iteration / "items" / "001-cubies" / "action-batch-001.json").is_file()
    assert (iteration / "budget-exhausted.json").is_file()
    assert blender.executions == []
