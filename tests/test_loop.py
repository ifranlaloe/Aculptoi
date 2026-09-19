from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from aculptoi.agent import Actor, RefinementLoop, VisionCritic
from aculptoi.agent.loop import IterationBudgetExceeded
from aculptoi.blender.client import BlenderActionError, BlenderWorkerError, ViewportCapture
from aculptoi.checkpoints import CheckpointStore, RunStateError
from aculptoi.checkpoints.store import RunDirectory
from aculptoi.inspection.service import AcceptedInspection
from aculptoi.models import ModelResponseError
from aculptoi.models.base import Message
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.actions import Action
from aculptoi.schemas.execution import ActionExecutionFailure, FailureKind
from aculptoi.schemas.inspection import (
    AtlasLayout,
    InspectionAtlasManifest,
    InspectionAtlasTile,
    InspectionBounds,
    InspectionFraming,
    InspectionSummary,
)
from aculptoi.schemas.viewport import ViewportObservation, ViewportView


def _target_brief_response(messages: Sequence[Message]) -> dict[str, object] | None:
    system = messages[0].get("content")
    if not isinstance(system, str) or "# Aculptoi Actor: Target Brief" not in system:
        return None
    content = messages[1].get("content")
    if not isinstance(content, str):
        raise AssertionError("target brief request must contain structured user content")
    goal = json.loads(content)["goal"]
    return {
        "subject": goal,
        "visual_priorities": [goal],
        "constraints": [],
        "non_goals": [],
        "form_traits": [],
    }


class FakeProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        assert messages
        del max_tokens, thinking, reasoning_effort
        target_brief = _target_brief_response(messages)
        if target_brief is not None:
            return target_brief
        return self.response


class SequencedProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        assert messages
        del max_tokens, thinking, reasoning_effort
        target_brief = _target_brief_response(messages)
        if target_brief is not None:
            return target_brief
        self.calls.append(messages)
        return self._responses.pop(0)


class TargetBriefRecordingProvider(SequencedProvider):
    def __init__(
        self,
        responses: list[dict[str, object]],
        target_brief: dict[str, object],
    ) -> None:
        super().__init__(responses)
        self.target_brief = target_brief
        self.target_brief_calls: list[Sequence[Message]] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        target_brief = _target_brief_response(messages)
        if target_brief is not None:
            self.target_brief_calls.append(messages)
            return self.target_brief
        return super().complete_json(
            messages,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )


class MixedProvider:
    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self._responses = responses
        self.max_tokens: list[int | None] = []
        self.thinking: list[bool | None] = []
        self.reasoning_efforts: list[ReasoningEffort | None] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        assert messages
        target_brief = _target_brief_response(messages)
        if target_brief is not None:
            return target_brief
        self.max_tokens.append(max_tokens)
        self.thinking.append(thinking)
        self.reasoning_efforts.append(reasoning_effort)
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
        self.fail_next_reload = False
        self.execution_failures: list[BlenderWorkerError] = []
        self.scene_inspection_calls = 0
        self.reload_calls = 0
        self._saved_objects: dict[bytes, dict[str, dict[str, object]]] = {b"initial fake blend": {}}

    def attach_run(
        self, scene_path: Path, run_id: int, *, reload: bool = False
    ) -> dict[str, object]:
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        if not scene_path.exists():
            scene_path.write_bytes(b"initial fake blend")
        if reload:
            self.reload_calls += 1
            if self.fail_next_reload:
                self.fail_next_reload = False
                raise BlenderWorkerError("canonical scene reload failed")
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
        self.scene_inspection_calls += 1
        return {"objects": deepcopy(list(self._objects.values()))}

    def _apply_actions(self, actions: Sequence[Action]) -> None:
        for action in actions:
            payload = action.model_dump(mode="json")
            if action.command == "object.create":
                self._objects[payload["name"]] = {
                    "name": payload["name"],
                    "location": payload["location"],
                    "scale": payload["scale"],
                    "type": "MESH",
                }

    def execute(self, actions: Sequence[Action]) -> dict[str, object]:
        self.executions.append(actions)
        if self.execution_failures and (
            not isinstance(self.execution_failures[0], BlenderActionError)
            or (
                self.execution_failures[0].failure.action_index is not None
                and self.execution_failures[0].failure.action_index < len(actions)
                and actions[self.execution_failures[0].failure.action_index].command
                == self.execution_failures[0].failure.command
            )
        ):
            error = self.execution_failures.pop(0)
            executed_before_failure = (
                error.failure.executed_before_failure
                if isinstance(error, BlenderActionError)
                else 0
            )
            self._apply_actions(actions[:executed_before_failure])
            raise error
        self._apply_actions(actions)
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


class ObservingFakeBlender(FakeBlender):
    """Fake UI worker whose image remains in memory for one Actor request."""

    def __init__(self) -> None:
        super().__init__()
        self.viewport_views: list[ViewportView] = []

    def observe_viewport(self, view: ViewportView) -> ViewportCapture:
        self.viewport_views.append(view)
        return ViewportCapture(
            observation=ViewportObservation(
                available=True,
                target=view.target,
                orientation=view.orientation,
                projection=view.projection,
                framing=view.framing,
                width=64,
                height=48,
            ),
            image_data_url="data:image/png;base64,aW1hZ2U=",
        )


def _recoverable_execution_error(
    *,
    failure_kind: FailureKind = "execution_error",
    code: str = "disconnected_region",
    action_index: int = 0,
    command: str = "mesh.extrude_region",
    executed_before_failure: int = 0,
    message: str = "selected faces form multiple disconnected regions",
) -> BlenderActionError:
    return BlenderActionError(
        ActionExecutionFailure(
            failure_kind=failure_kind,
            code=code,
            message=message,
            action_index=action_index,
            command=command,
            executed_before_failure=executed_before_failure,
            recoverable=True,
            rolled_back=True,
            scene_restored=True,
        )
    )


def _worker_internal_error(command: str = "mesh.transform_region") -> BlenderActionError:
    return BlenderActionError(
        ActionExecutionFailure(
            failure_kind="worker_error",
            code="internal_worker_error",
            message="internal Blender worker failure",
            action_index=0,
            command=command,
            executed_before_failure=0,
            recoverable=False,
            rolled_back=True,
            scene_restored=True,
        )
    )


class InvalidJsonProvider:
    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        target_brief = _target_brief_response(messages)
        if target_brief is not None:
            return target_brief
        del max_tokens, thinking, reasoning_effort
        raise ModelResponseError("Model response was not valid JSON", "<think>unfinished</think>")


class FakeInspection:
    def __init__(self) -> None:
        self.calls = 0

    def inspect(
        self,
        run: RunDirectory,
        iteration: int,
        checkpoints: CheckpointStore,
        *,
        artifact_root: str | None = None,
        events: object | None = None,
    ) -> AcceptedInspection:
        del events
        self.calls += 1
        root = artifact_root or "inspection"
        path = run.path / f"iteration-{iteration:03d}" / root / "atlas.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color="white").save(path)
        layout = AtlasLayout(columns=1, rows=1, tile_dimension=32, width=32, height=32)
        manifest = InspectionAtlasManifest(
            sensor_version="inspection-atlas-v1",
            lighting_rig="neutral-studio-v1",
            width=32,
            height=32,
            layout=layout,
            bounds=InspectionBounds(
                minimum=(-1.0, -1.0, -1.0),
                maximum=(1.0, 1.0, 1.0),
                center=(0.0, 0.0, 0.0),
                radius=1.8,
            ),
            framing=InspectionFraming(margin=1.15, distance=5.0, orthographic_scale=4.0),
            estimated_surface_coverage=0.9,
            tiles={
                "A1": InspectionAtlasTile(
                    tile_id="A1",
                    camera_id="anchor-front",
                    source="shots/A1.png",
                    pixel_bounds=(0, 0, 32, 32),
                    azimuth_degrees=0,
                    elevation_degrees=0,
                    orientation="front",
                    projection="orthographic",
                    selection_kind="canonical_anchor",
                    selection_reason="canonical_anchor",
                    coverage_gain=0,
                    information_gain=0,
                )
            },
        )
        summary = InspectionSummary(
            status="accepted",
            rounds=1,
            total_views_rendered=1,
            views_in_accepted_atlas=1,
            estimated_surface_coverage=0.9,
            accepted_atlas="atlas.png",
            accepted_manifest="atlas-manifest.json",
            sensor_version="inspection-atlas-v1",
            lighting_rig="neutral-studio-v1",
        )
        relative_root = f"iteration-{iteration:03d}/{root}"
        checkpoints.save_json_artifact(
            run,
            f"{relative_root}/atlas-manifest.json",
            manifest.model_dump(mode="json"),
            overwrite=False,
        )
        checkpoints.save_json_artifact(
            run,
            f"{relative_root}/summary.json",
            summary.model_dump(mode="json"),
            overwrite=False,
        )
        return AcceptedInspection(atlas=path, manifest=manifest, summary=summary)


class FailingOnceInspection(FakeInspection):
    def __init__(self) -> None:
        super().__init__()
        self.roots: list[str] = []

    def inspect(
        self,
        run: RunDirectory,
        iteration: int,
        checkpoints: CheckpointStore,
        *,
        artifact_root: str | None = None,
        events: object | None = None,
    ) -> AcceptedInspection:
        del events
        root = artifact_root or "inspection"
        self.roots.append(root)
        if len(self.roots) == 1:
            raise RuntimeError("inspection renderer interrupted")
        return super().inspect(
            run,
            iteration,
            checkpoints,
            artifact_root=artifact_root,
        )


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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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
    iteration = result.run_directory / "iteration-001"
    actor_directory = iteration / "actor"
    item = actor_directory / "items" / "001-body"
    assert (actor_directory / "construction-plan-prompt.json").is_file()
    assert (actor_directory / "construction-plan.json").is_file()
    assert (item / "item.json").is_file()
    assert (item / "completion-criteria.json").is_file()
    assert (item / "actor-prompt-001.json").is_file()
    assert (item / "actor-response-001.json").is_file()
    assert (item / "action-result-001.json").is_file()
    assert (item / "checkpoint.json").is_file()
    assert (item / "summary.json").is_file()
    critic_directory = iteration / "critic"
    assert (critic_directory / "discovery-prompt.json").is_file()
    assert (critic_directory / "discovery.json").is_file()
    assert (critic_directory / "critique.json").is_file()
    assert (iteration / "iteration-summary.json").is_file()
    assert (iteration / "checkpoint.json").is_file()
    assert not (iteration / "front.png").exists()
    assert not (iteration / "right.png").exists()
    assert (result.run_directory / "scene.blend").read_bytes() == b"fake blend save 1"
    assert (result.run_directory / "checkpoints" / "item-001-001-body.blend").read_bytes() == (
        b"fake blend save 1"
    )
    state = json.loads((result.run_directory / "run-state.json").read_text())
    assert state["latest_checkpoint"] == "checkpoints/item-001-001-body.blend"

    plan_prompt = json.loads((actor_directory / "construction-plan-prompt.json").read_text())
    item_prompt = json.loads((item / "actor-prompt-001.json").read_text())
    discovery_prompt = json.loads((critic_directory / "discovery-prompt.json").read_text())
    action_batch = json.loads((item / "actor-response-001.json").read_text())
    assert plan_prompt["request_type"] == "construction_plan"
    assert item_prompt["request_type"] == "work_item_actions"
    assert action_batch["construction_plan_id"] == "run-000001-iteration-001"
    assert action_batch["work_item_id"] == "body"
    assert action_batch["response"]["completion_criteria"] == ["A body object exists."]
    assert discovery_prompt["role"] == "vision_critic"
    assert discovery_prompt["request_type"] == "vision_issue_discovery"
    assert plan_prompt["thinking"] is True
    assert plan_prompt["reasoning_effort"] == "medium"
    assert discovery_prompt["thinking"] is False
    assert discovery_prompt["reasoning_effort"] is None
    assert discovery_prompt["max_output_tokens"] == 4_096
    assert discovery_prompt["atlas"]["prepared_width"] == 32
    assert "data:image" not in (critic_directory / "discovery-prompt.json").read_text()
    assert len(actor_provider.calls) == 2


def test_recoverable_action_failure_retries_same_item_with_fresh_execution_feedback(
    tmp_path: Path,
) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "continue",
                "reason": "Attempt the initial body operation.",
                "completion_criteria": ["A body object exists."],
                "actions": [{"command": "object.create", "name": "Discarded"}],
            },
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Use a different safe body operation.",
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(
        _recoverable_execution_error(
            failure_kind="validation_error",
            code="object_already_exists",
            command="object.create",
            message="requested object name already exists",
        )
    )
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a body")

    item = result.run_directory / "iteration-001/actor/items/001-body"
    retry_context = json.loads(actor_provider.calls[2][1]["content"])
    failed_result = json.loads((item / "action-result-001.json").read_text())
    summary = json.loads(
        (result.run_directory / "iteration-001/iteration-summary.json").read_text()
    )

    assert result.completed is True
    assert result.execution_batches == 1
    assert len(actor_provider.calls) == 3
    assert blender.reload_calls == 1
    assert blender.scene_inspection_calls >= 4
    assert retry_context["active_work_item"]["id"] == "body"
    assert retry_context["completion_criteria"] == ["A body object exists."]
    assert retry_context["scene"]["objects"] == []
    assert "Traceback" not in actor_provider.calls[2][1]["content"]
    assert retry_context["recent_execution"] == {
        "modeling_step": 1,
        "attempted_action_count": 1,
        "canonical_scene": "scene.blend",
        "failure": {
            "action_index": 0,
            "code": "object_already_exists",
            "command": "object.create",
            "executed_before_failure": 0,
            "failure_kind": "validation_error",
            "message": "requested object name already exists",
            "recoverable": True,
            "rolled_back": True,
            "scene_restored": True,
        },
        "proposed_action_count": 1,
        "status": "failed",
        "worker_called": True,
        "worker_result": None,
    }
    assert len(actor_provider.calls[2]) == 2
    assert failed_result["fresh_scene"] == {"objects": []}
    assert (
        failed_result["outcome"]["action_batch"]
        == retry_context["recent_execution"]["modeling_step"]
    )
    assert "Traceback" not in json.dumps(failed_result)
    assert "Discarded" not in blender._objects
    assert (item / "actor-response-001.json").is_file()
    assert (item / "actor-response-002.json").is_file()
    assert (item / "action-result-002.json").is_file()
    assert summary["safety_budget"]["actor_requests"]["used"] == 3
    assert summary["safety_budget"]["actions"]["used"] == 2


def test_recoverable_partial_batch_failure_restores_pre_batch_scene_and_charges_actions(
    tmp_path: Path,
) -> None:
    actor_provider = SequencedProvider(
        [
            {
                "reason": "Build a durable base, then refine its detail.",
                "items": [
                    {
                        "id": "base",
                        "title": "Base",
                        "objective": "Create a base.",
                        "depends_on": [],
                    },
                    {
                        "id": "detail",
                        "title": "Detail",
                        "objective": "Refine the durable base.",
                        "depends_on": ["base"],
                    },
                ],
            },
            {
                "work_item_id": "base",
                "status": "complete",
                "reason": "Create a durable base.",
                "completion_criteria": ["Durable exists."],
                "actions": [{"command": "object.create", "name": "Durable"}],
            },
            {
                "work_item_id": "detail",
                "status": "continue",
                "reason": "Attempt an appendage refinement.",
                "completion_criteria": ["A final detail exists."],
                "actions": [
                    {"command": "object.create", "name": "Transient"},
                    {
                        "command": "mesh.transform_region",
                        "object": "Durable",
                        "region": {"min": [-1, -1, -1], "max": [1, 1, 1]},
                    },
                ],
            },
            {
                "work_item_id": "detail",
                "status": "complete",
                "reason": "Use a compatible refinement.",
                "actions": [{"command": "object.create", "name": "FinalDetail"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(
        _recoverable_execution_error(
            code="empty_region",
            action_index=1,
            command="mesh.transform_region",
            executed_before_failure=1,
        )
    )
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("refine a base")

    item = result.run_directory / "iteration-001/actor/items/002-detail"
    retry_context = json.loads(actor_provider.calls[3][1]["content"])
    failed_result = json.loads((item / "action-result-001.json").read_text())
    summary = json.loads(
        (result.run_directory / "iteration-001/iteration-summary.json").read_text()
    )

    assert result.completed is True
    assert "Durable" in blender._objects
    assert "Transient" not in blender._objects
    assert "FinalDetail" in blender._objects
    assert blender.reload_calls == 1
    assert retry_context["scene"]["objects"] == [
        {
            "location": [0.0, 0.0, 0.0],
            "name": "Durable",
            "scale": [1.0, 1.0, 1.0],
            "type": "MESH",
        }
    ]
    assert retry_context["recent_execution"]["failure"]["executed_before_failure"] == 1
    assert retry_context["recent_execution"]["failure"]["action_index"] == 1
    assert failed_result["fresh_scene"] == retry_context["scene"]
    assert summary["safety_budget"]["actions"]["used"] == 4
    assert (item / "action-result-001.json").is_file()
    assert (item / "action-result-002.json").is_file()


def test_worker_error_stops_without_an_actor_retry_after_restoration(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Attempt body transformation.",
                "completion_criteria": ["Body exists."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(_worker_internal_error(command="object.create"))
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="internal Blender worker failure"):
        loop.run("create a body")

    run = store.get_run(1)
    item = run.path / "iteration-001/actor/items/001-body"
    failed_result = json.loads((item / "action-result-001.json").read_text())
    assert len(actor_provider.calls) == 2
    assert blender.reload_calls == 1
    assert failed_result["outcome"]["failure"]["failure_kind"] == "worker_error"
    assert (item / "action-execution-error-001.json").is_file()
    assert store.load_run_state(run).active_item is not None


def test_unstructured_worker_error_persists_failed_outcome_without_actor_retry(
    tmp_path: Path,
) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Attempt body creation.",
                "completion_criteria": ["Body exists."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(BlenderWorkerError("worker connection closed"))
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="internal Blender worker failure"):
        loop.run("create a body")

    item = store.get_run(1).path / "iteration-001/actor/items/001-body"
    failed_result = json.loads((item / "action-result-001.json").read_text())
    assert len(actor_provider.calls) == 2
    assert blender.reload_calls == 1
    assert failed_result["outcome"]["failure"] == {
        "action_index": None,
        "code": "worker_transport_error",
        "command": None,
        "executed_before_failure": 0,
        "failure_kind": "worker_error",
        "message": "Blender worker failed without a structured action error",
        "recoverable": False,
        "rolled_back": True,
        "scene_restored": True,
    }
    assert failed_result["fresh_scene"] == {"objects": []}
    assert (item / "action-execution-error-001.json").is_file()


def test_rollback_failure_stops_without_an_actor_retry(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Attempt body creation.",
                "completion_criteria": ["Body exists."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(_recoverable_execution_error(command="object.create"))
    blender.fail_next_reload = True
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="could not be restored"):
        loop.run("create a body")

    item = store.get_run(1).path / "iteration-001/actor/items/001-body"
    assert len(actor_provider.calls) == 2
    failed_result = json.loads((item / "action-result-001.json").read_text())
    assert failed_result["outcome"]["failure"]["code"] == "rollback_failed"
    assert failed_result["outcome"]["failure"]["recoverable"] is False
    assert (item / "action-execution-restore-error-001.json").is_file()


def test_recoverable_retry_cannot_replace_completion_criteria(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "continue",
                "reason": "Try the first approach.",
                "completion_criteria": ["Body exists."],
                "actions": [{"command": "object.create", "name": "Discarded"}],
            },
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Try a replacement approach.",
                "completion_criteria": ["Different criterion."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(_recoverable_execution_error(command="object.create"))
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(ModelResponseError, match="must not replace completion criteria"):
        loop.run("create a body")

    assert len(actor_provider.calls) == 3
    assert len(blender.executions) == 1


def test_recoverable_failure_does_not_refund_action_budget(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "continue",
                "reason": "Try the first approach.",
                "completion_criteria": ["Body exists."],
                "actions": [{"command": "object.create", "name": "Discarded"}],
            },
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Try a replacement approach.",
                "actions": [{"command": "object.create", "name": "Body"}],
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    blender.execution_failures.append(_recoverable_execution_error(command="object.create"))
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
        max_actions_per_iteration=1,
    )

    with pytest.raises(IterationBudgetExceeded, match="max_actions_per_iteration"):
        loop.run("create a body")

    budget = json.loads((store.runs / "000001/iteration-001/budget-exhausted.json").read_text())
    assert len(actor_provider.calls) == 3
    assert budget["budget"] == "max_actions_per_iteration"
    assert budget["limit"] == 1
    assert budget["actions_executed"] == 1
    assert len(blender.executions) == 1


def test_target_brief_is_persisted_and_reused_after_interrupted_inspection(
    tmp_path: Path,
) -> None:
    target_brief = {
        "subject": "simple stylized fish",
        "visual_priorities": ["a readable tapered body", "clear bilateral fins"],
        "constraints": ["keep the silhouette legible from several views"],
        "non_goals": ["realistic scales"],
        "form_traits": ["organic", "continuous_form", "bilateral_symmetry", "appendages"],
    }
    actor_provider = TargetBriefRecordingProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create the body primitive.",
                "completion_criteria": ["A body object exists."],
                "actions": [{"command": "object.create", "name": "Body", "primitive": "uv_sphere"}],
            },
        ],
        target_brief,
    )
    store = CheckpointStore(tmp_path)
    inspection = FailingOnceInspection()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        inspection=inspection,  # type: ignore[arg-type]
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="inspection renderer interrupted"):
        loop.run("create a simple stylized fish")

    run = store.runs / "000001"
    prompt = json.loads((run / "target-brief-prompt.json").read_text())
    artifact = json.loads((run / "target-brief.json").read_text())
    state = json.loads((run / "run-state.json").read_text())
    plan_prompt = json.loads(
        (run / "iteration-001/actor/construction-plan-prompt.json").read_text()
    )
    work_item_prompt = json.loads(
        (run / "iteration-001/actor/items/001-body/actor-prompt-001.json").read_text()
    )

    assert len(actor_provider.target_brief_calls) == 1
    assert prompt["request_type"] == "target_brief"
    assert artifact["brief"] == target_brief
    assert artifact["derivation"] == "actor"
    assert state["target_brief_state"] == "ready"
    assert state["target_brief_artifact"] == "target-brief.json"
    assert len(state["target_brief_artifact_sha256"]) == 64
    plan_context = json.loads(plan_prompt["messages"][1]["content"])
    work_item_context = json.loads(work_item_prompt["messages"][1]["content"])
    assert plan_context["target_brief"] == target_brief
    assert "modeling_capabilities" in plan_context
    assert "action_catalog" not in plan_context
    assert "action_semantics" not in plan_context
    assert work_item_context["target_brief"] == target_brief
    assert "action_catalog" in work_item_context
    assert work_item_context["action_semantics"]["normalized_mesh_regions"]["bounds"] == "inclusive"

    result = loop.resume(RunDirectory(id=1, path=run))

    assert result.completed is True
    assert inspection.calls == 1
    assert len(actor_provider.target_brief_calls) == 1


def test_resume_rejects_missing_pending_target_brief_without_model_call(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    run = store.create_run("create a creature")
    blender.attach_run(store.canonical_scene_path(run), run.id)
    store.preserve_initial_scene(run)
    state = store.load_run_state(run)
    assert state.target_brief_state == "pending"
    store.save_run_state(
        run,
        state.model_copy(update={"status": "interrupted"}),
    )
    actor_provider = TargetBriefRecordingProvider(
        [_one_item_plan()],
        {
            "subject": "creature",
            "visual_priorities": ["silhouette"],
            "constraints": [],
            "non_goals": [],
            "form_traits": ["organic"],
        },
    )
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RunStateError, match="target brief artifact is missing or invalid"):
        loop.resume(run)

    assert actor_provider.target_brief_calls == []


def test_resume_creates_deterministic_legacy_target_brief_without_model_call(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    goal = "create a simple cube"
    run = store.create_run(goal)
    blender.attach_run(store.canonical_scene_path(run), run.id)
    store.preserve_initial_scene(run)
    state = store.load_run_state(run)
    store.save_run_state(
        run,
        state.model_copy(update={"status": "interrupted", "target_brief_state": "absent"}),
    )
    actor_provider = TargetBriefRecordingProvider(
        [
            _one_item_plan(),
            {
                "work_item_id": "body",
                "status": "complete",
                "reason": "Create a cube.",
                "completion_criteria": ["A cube exists."],
                "actions": [{"command": "object.create", "name": "Cube", "primitive": "cube"}],
            },
        ],
        {
            "subject": "should never be requested",
            "visual_priorities": ["unused"],
            "constraints": [],
            "non_goals": [],
            "form_traits": [],
        },
    )
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.resume(run)

    artifact = json.loads((run.path / "target-brief.json").read_text())
    state = json.loads((run.path / "run-state.json").read_text())
    assert result.completed is True
    assert actor_provider.target_brief_calls == []
    assert artifact["derivation"] == "legacy_fallback"
    assert artifact["brief"] == {
        "subject": goal,
        "visual_priorities": [goal],
        "constraints": [],
        "non_goals": [],
        "form_traits": [],
    }
    assert state["target_brief_state"] == "legacy"


def test_refinement_loop_records_raw_construction_plan_failures(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(InvalidJsonProvider()),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(ModelResponseError, match="not valid JSON"):
        loop.run("create a creature")

    iteration = store.runs / "000001" / "iteration-001"
    assert (store.runs / "000001" / "user-prompt.txt").read_text() == "create a creature"
    assert (iteration / "actor/construction-plan-error.json").is_file()
    assert (iteration / "actor/construction-plan-response-raw.txt").read_text() == (
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
                    ["left_wing", "H", 90, ["A1"], "intersects torso"],
                    ["neck", "M", 80, ["A1"], "too short"],
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
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a creature")

    iteration = result.run_directory / "iteration-001"
    critic_directory = iteration / "critic" / "issues"
    assembled = json.loads((iteration / "critic/critique.json").read_text())
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
        "evidence_tiles": ["A1"],
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
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a two-cubie Rubik-style row")

    iteration = result.run_directory / "iteration-001"
    first_item = iteration / "actor/items" / "001-left-cubie"
    second_item = iteration / "actor/items" / "002-right-cubie"
    second_batch_context = json.loads(actor_provider.calls[2][1]["content"])
    second_item_context = json.loads(actor_provider.calls[3][1]["content"])
    assert result.completed is True
    assert result.execution_batches == 3
    assert len(blender.executions) == 3
    assert blender.canonical_saves == 3
    assert blender.render_calls == 0
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
    assert (first_item / "actor-response-001.json").is_file()
    assert (first_item / "actor-response-002.json").is_file()
    assert (second_item / "actor-response-001.json").is_file()
    assert (iteration / "critic/critique.json").is_file()
    assert sorted(path.name for path in (result.run_directory / "checkpoints").glob("*.blend")) == [
        "item-001-001-left-cubie.blend",
        "item-001-002-right-cubie.blend",
    ]


def test_modeling_steps_observe_before_each_next_actor_turn_without_persisting_images(
    tmp_path: Path,
) -> None:
    """The Actor sees one current sensor image, never an accumulated screenshot history."""
    actor_provider = SequencedProvider(
        [
            _one_item_plan("body"),
            {
                "kind": "modeling_step",
                "work_item_id": "body",
                "reason": "Establish the primary mass.",
                "intent": "Create one body mass.",
                "completion_criteria": ["A body mass exists."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
            {
                "kind": "observation_request",
                "work_item_id": "body",
                "reason": "The top silhouette needs confirmation.",
                "view": {"orientation": "top", "projection": "orthographic", "framing": "close"},
            },
            {
                "kind": "modeling_step",
                "work_item_id": "body",
                "reason": "The observed mass is too narrow.",
                "intent": "Widen the body mass.",
                "actions": [{"command": "object.scale", "object": "Body", "scale": [1.2, 1, 1]}],
            },
            {
                "kind": "complete",
                "work_item_id": "body",
                "reason": "The observed body mass satisfies the criterion.",
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    blender = ObservingFakeBlender()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a body")

    item = result.run_directory / "iteration-001" / "actor/items/001-body"
    first_context = json.loads(actor_provider.calls[1][1]["content"][0]["text"])
    second_context = json.loads(actor_provider.calls[2][1]["content"][0]["text"])
    third_context = json.loads(actor_provider.calls[3][1]["content"][0]["text"])
    fourth_context = json.loads(actor_provider.calls[4][1]["content"][0]["text"])
    prompt_artifact = json.loads((item / "actor-prompt-002.json").read_text())

    assert result.completed is True
    assert len(blender.executions) == 2
    assert blender.canonical_saves == 3  # two steps plus the durable completion save
    assert len(blender.viewport_views) == 4  # initial, post-step, requested, post-step
    assert blender.viewport_views[2].orientation == "top"
    assert all(
        context["actor_viewport_available"] is True
        for context in (
            first_context,
            second_context,
            third_context,
            fourth_context,
        )
    )
    assert second_context["scene"]["objects"][0]["name"] == "Body"
    assert third_context["viewport_observation"]["orientation"] == "top"
    assert fourth_context["recent_execution"]["modeling_step"] == 2
    assert all(call[1]["content"].__class__ is list for call in actor_provider.calls[1:])
    assert all(
        sum(part.get("type") == "image_url" for part in call[1]["content"]) == 1
        for call in actor_provider.calls[1:]
    )
    assert "<transient-viewport-image-omitted>" in json.dumps(prompt_artifact)
    assert "data:image/png;base64" not in json.dumps(prompt_artifact)
    assert not list(item.glob("*.png"))
    assert (item / "viewport-observation-001.json").is_file()
    assert (item / "viewport-observation-004.json").is_file()


def test_observation_requests_consume_actor_budget_but_not_action_budget(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan("body"),
            {
                "kind": "observation_request",
                "work_item_id": "body",
                "reason": "Need a top view before establishing criteria.",
                "completion_criteria": ["A body exists."],
                "view": {"orientation": "top"},
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=ObservingFakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
        max_actor_observations_per_work_item=0,
    )

    with pytest.raises(IterationBudgetExceeded, match="max_actor_observations_per_work_item"):
        loop.run("create a body")

    budget = json.loads((store.runs / "000001/iteration-001/budget-exhausted.json").read_text())
    assert budget["actions_executed"] == 0
    assert budget["actor_requests"] == 2


def test_headless_or_legacy_worker_continues_with_explicit_structured_only_context(
    tmp_path: Path,
) -> None:
    actor_provider = SequencedProvider(
        [
            _one_item_plan("body"),
            {
                "kind": "modeling_step",
                "work_item_id": "body",
                "reason": "Create the body from structured state.",
                "intent": "Create one body mass.",
                "completion_criteria": ["A body exists."],
                "actions": [{"command": "object.create", "name": "Body"}],
            },
            {
                "kind": "complete",
                "work_item_id": "body",
                "reason": "Structured inspection confirms the body exists.",
            },
        ]
    )
    store = CheckpointStore(tmp_path)
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 100, "issues": []})),
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=FakeBlender(),  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = loop.run("create a body")
    first_context = json.loads(actor_provider.calls[1][1]["content"])

    assert result.completed is True
    assert first_context["actor_viewport_available"] is False
    assert first_context["viewport_observation"]["error"]
    assert not list((result.run_directory / "iteration-001/actor/items/001-body").glob("*.png"))


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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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
        inspection=FakeInspection(),  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
        max_actions_per_iteration=1,
    )

    with pytest.raises(IterationBudgetExceeded, match="max_actions_per_iteration"):
        loop.run("create two cubies")

    iteration = store.runs / "000001" / "iteration-001"
    assert (iteration / "actor/items" / "001-cubies" / "actor-response-001.json").is_file()
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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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
    assert list((run.path / "iteration-001/actor/items/001-body").glob("canonical-save-error*"))


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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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


def test_resume_restarts_inspection_without_replaying_durable_actor_work(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
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
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    inspection = FailingOnceInspection()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(FakeProvider({"score": 95, "issues": []})),
        inspection=inspection,  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(RuntimeError, match="inspection renderer interrupted"):
        loop.run("create a body")

    run = store.get_run(1)
    interrupted = store.load_run_state(run)
    assert interrupted.status == "interrupted"
    assert interrupted.active_item is None
    assert interrupted.active_phase is not None
    assert interrupted.active_phase.phase == "inspection"

    result = loop.resume(run)

    assert result.completed is True
    assert len(actor_provider.calls) == 2
    assert inspection.roots == ["inspection", "inspection/recovery-attempt-002"]
    assert (run.path / "iteration-001/inspection/recovery-attempt-002/summary.json").is_file()


def test_resume_restarts_critic_from_its_persisted_accepted_atlas(tmp_path: Path) -> None:
    actor_provider = SequencedProvider(
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
    interrupted_critic = MixedProvider([ModelResponseError("discovery response interrupted")])
    store = CheckpointStore(tmp_path)
    blender = FakeBlender()
    inspection = FakeInspection()
    loop = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(
            interrupted_critic,
            max_output_tokens=16_384,
            reasoning_effort="medium",
        ),
        inspection=inspection,  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    with pytest.raises(ModelResponseError, match="discovery response interrupted"):
        loop.run("create a body")

    run = store.get_run(1)
    interrupted = store.load_run_state(run)
    assert interrupted.active_phase is not None
    assert interrupted.active_phase.phase == "critic"
    (run.path / "run-events.jsonl").unlink()

    resumed_critic = MixedProvider([{"score": 95, "issues": []}])
    resumed = RefinementLoop(
        actor=Actor(actor_provider),
        critic=VisionCritic(resumed_critic),
        inspection=inspection,  # type: ignore[arg-type]
        blender=blender,  # type: ignore[arg-type]
        checkpoints=store,
        max_iterations=1,
        score_target=0.9,
    )

    result = resumed.resume(run)

    assert result.completed is True
    assert len(actor_provider.calls) == 2
    assert inspection.calls == 1
    assert resumed_critic.max_tokens == [4_096]
    assert resumed_critic.thinking == [False]
    assert resumed_critic.reasoning_efforts == [None]
    events = [json.loads(line) for line in (run.path / "run-events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_resumed",
        "stage_completed",
        "run_completed",
    ]
    assert events[1]["stage"] == "critic_discovery"
    assert events[1]["thinking"] is False
    assert "reasoning_effort" not in events[1]
    assert events[1]["max_output_tokens"] == 4_096
    assert (run.path / "iteration-001/critic/recovery-attempt-002/discovery.json").is_file()


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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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
        inspection=FakeInspection(),  # type: ignore[arg-type]
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
        run.path / "iteration-001/actor/items/002-wing/recovery-attempt-002/actor-response-001.json"
    ).is_file()
