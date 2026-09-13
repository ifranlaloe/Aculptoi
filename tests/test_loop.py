from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
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
    def __init__(self) -> None:
        self._objects: dict[str, dict[str, object]] = {}
        self.executions: list[Sequence[Action]] = []
        self.render_calls = 0
        self.canonical_saves = 0
        self.active_scene_path: Path | None = None
        self.active_run_id: int | None = None
        self.fail_next_save = False
        self._saved_objects: dict[bytes, dict[str, dict[str, object]]] = {b"initial fake blend": {}}

    def attach_run(
        self, scene_path: Path, run_id: int, *, reload: bool = False
    ) -> dict[str, object]:
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        if not scene_path.exists():
            scene_path.write_bytes(b"initial fake blend")
        if reload:
            self._objects = deepcopy(self._saved_objects[scene_path.read_bytes()])
        self.active_scene_path = scene_path
        self.active_run_id = run_id
        return {"scene_path": str(scene_path), "active_filepath": str(scene_path)}

    def save_canonical_scene(self, scene_path: Path) -> dict[str, object]:
        assert scene_path == self.active_scene_path
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("disk full")
        self.canonical_saves += 1
        contents = f"fake blend save {self.canonical_saves}".encode()
        scene_path.write_bytes(contents)
        self._saved_objects[contents] = deepcopy(self._objects)
        return {"scene_path": str(scene_path), "active_filepath": str(scene_path)}

    def release_run(self) -> dict[str, object]:
        previous = self.active_run_id
        self.active_run_id = None
        self.active_scene_path = None
        return {"released_run_id": previous}

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
    critic = VisionCritic(FakeProvider({"score": 95, "issues": []}))
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=critic,
        blender=FakeBlender(),  # type: ignore[arg-type]
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
    assert (item / "checkpoint.json").is_file()
    assert (item / "summary.json").is_file()
    critic_directory = iteration / "critic"
    assert (critic_directory / "discovery-prompt.json").is_file()
    assert (critic_directory / "discovery.json").is_file()
    assert (iteration / "vision-analysis.json").is_file()
    assert (iteration / "iteration-summary.json").is_file()
    assert (iteration / "checkpoint.json").is_file()
    assert (iteration / "perspective.png").is_file()
    assert (result.run_directory / "scene.blend").read_bytes() == b"fake blend save 1"
    assert (result.run_directory / "checkpoints" / "item-001-001-body.blend").read_bytes() == (
        b"fake blend save 1"
    )
    state = json.loads((result.run_directory / "run-state.json").read_text())
    assert state["latest_checkpoint"] == "checkpoints/item-001-001-body.blend"

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
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        blender=FakeBlender(),  # type: ignore[arg-type]
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
                "score": 95,
                "issues": [
                    ["left_wing", "H", 90, ["F", "P"], "intersects torso"],
                    ["neck", "M", 80, ["R", "P"], "too short"],
                ],
            },
            {
                "desc": "This payload improperly includes an action.",
                "evidence": ["Visible in front."],
                "cause": None,
                "fix": "Separate the forms.",
                "criteria": ["The silhouettes are distinct."],
                "confidence": 90,
                "actions": [{"command": "object.delete"}],
            },
            {
                "desc": "The neck has little visible length before the head.",
                "evidence": ["The right silhouette compresses the neck."],
                "cause": "The neck primitive is too short.",
                "fix": "Lengthen and taper the neck.",
                "criteria": ["The right silhouette shows a distinct neck."],
                "confidence": 88,
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(critic_provider),
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a creature")

    iteration = result.run_directory / "iteration-001"
    critic_directory = iteration / "critic" / "issues"
    assembled = json.loads((iteration / "vision-analysis.json").read_text())
    discovery = json.loads((iteration / "critic" / "discovery.json").read_text())
    detail = json.loads((critic_directory / "issue-002" / "analysis.json").read_text())
    assert result.completed is True
    assert (critic_directory / "issue-001" / "summary.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-prompt.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-error.json").is_file()
    assert (critic_directory / "issue-001" / "analysis-response-raw.txt").is_file()
    assert (critic_directory / "issue-002" / "analysis.json").is_file()
    assert discovery["score"] == 0.95
    assert discovery["issues"][0] == {
        "id": "issue-001",
        "title": "Left Wing: intersects torso",
        "region": "left_wing",
        "severity": "high",
        "confidence": 0.9,
        "evidence_views": ["front", "perspective"],
        "observation": "intersects torso",
    }
    assert detail["id"] == "issue-002"
    assert detail["description"] == "The neck has little visible length before the head."
    assert detail["suggested_correction"] == "Lengthen and taper the neck."
    assert detail["confidence"] == 0.88
    assert "desc" not in detail
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
    blender = FakeBlender()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
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
    assert blender.canonical_saves == 3
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
    assert sorted(path.name for path in (result.run_directory / "checkpoints").glob("*.blend")) == [
        "item-001-001-left-cubie.blend",
        "item-001-002-right-cubie.blend",
    ]


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
    blender = FakeBlender()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
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
    assert blender.canonical_saves == 1
    assert not list((store.runs / "000001" / "checkpoints").glob("*.blend"))
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
    blender = FakeBlender()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
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


def test_canonical_save_failure_does_not_mark_an_item_durable(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.fail_next_save = True
    loop = RefinementLoop(
        actor=Actor(
            SequencedProvider(
                [
                    _one_item_plan(),
                    {
                        "work_item_id": "body",
                        "status": "complete",
                        "reason": "Create body.",
                        "completion_criteria": ["Body exists."],
                        "actions": [{"command": "object.create", "name": "Body"}],
                    },
                ]
            )
        ),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="could not save the canonical scene"):
        loop.run("create a body")

    run = store.get_run(1)
    state = store.load_run_state(run)
    assert state.status == "interrupted"
    assert state.active_item is not None
    assert state.latest_checkpoint is None
    assert not list((run.path / "checkpoints").glob("*.blend"))
    assert list((run.path / "iteration-001" / "items" / "001-body").glob("canonical-save-error*"))


def test_checkpoint_failure_does_not_mark_an_item_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    loop = RefinementLoop(
        actor=Actor(
            SequencedProvider(
                [
                    _one_item_plan(),
                    {
                        "work_item_id": "body",
                        "status": "complete",
                        "reason": "Create body.",
                        "completion_criteria": ["Body exists."],
                        "actions": [{"command": "object.create", "name": "Body"}],
                    },
                ]
            )
        ),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    def fail_checkpoint(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("checkpoint volume unavailable")

    monkeypatch.setattr(store, "create_checkpoint", fail_checkpoint)

    with pytest.raises(OSError, match="checkpoint volume"):
        loop.run("create a body")

    run = store.get_run(1)
    state = store.load_run_state(run)
    assert state.status == "interrupted"
    assert state.active_item is not None
    assert state.latest_checkpoint is None
    assert blender.canonical_saves == 1


def test_resume_restores_last_durable_item_and_restarts_active_item(tmp_path: Path) -> None:
    plan = {
        "reason": "Build a durable base and then a wing.",
        "items": [
            {"id": "base", "title": "Base", "objective": "Create base.", "depends_on": []},
            {
                "id": "wing",
                "title": "Wing",
                "objective": "Create wing.",
                "depends_on": ["base"],
            },
        ],
    }
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    interrupted = RefinementLoop(
        actor=Actor(
            SequencedProvider(
                [
                    plan,
                    {
                        "work_item_id": "base",
                        "status": "complete",
                        "reason": "Base complete.",
                        "completion_criteria": ["Base exists."],
                        "actions": [{"command": "object.create", "name": "Base"}],
                    },
                    {
                        "work_item_id": "wing",
                        "status": "continue",
                        "reason": "Start the wing.",
                        "completion_criteria": ["Wing exists."],
                        "actions": [{"command": "object.create", "name": "PartialWing"}],
                    },
                    {},
                ]
            )
        ),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(ModelResponseError):
        interrupted.run("create a base and wing")

    run = store.get_run(1)
    before_resume = store.load_run_state(run)
    assert before_resume.active_item is not None
    assert before_resume.active_item.work_item_id == "wing"
    assert before_resume.latest_checkpoint == "checkpoints/item-001-001-base.blend"

    resumed = RefinementLoop(
        actor=Actor(
            SequencedProvider(
                [
                    {
                        "work_item_id": "wing",
                        "status": "complete",
                        "reason": "Rebuild wing from its known-good base.",
                        "completion_criteria": ["Final wing exists."],
                        "actions": [{"command": "object.create", "name": "FinalWing"}],
                    },
                ]
            )
        ),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = resumed.resume(run)

    assert result.completed is True
    assert "PartialWing" not in blender._objects
    assert "Base" in blender._objects
    assert "FinalWing" in blender._objects
    assert list((run.path / "recovery").glob("abandoned-*.blend"))
    assert (run.path / "checkpoints" / "item-001-002-wing.blend").is_file()
    assert (
        run.path / "iteration-001/items/002-wing/recovery-attempt-002/action-batch-001.json"
    ).is_file()
