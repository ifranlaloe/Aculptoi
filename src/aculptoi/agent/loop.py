"""Explicit plan-first orchestration for local Blender refinement."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from aculptoi.agent.actor import Actor
from aculptoi.agent.critic import VisionCritic
from aculptoi.blender.client import BlenderClient
from aculptoi.checkpoints import ActiveWorkItem, CheckpointStore, RunStateError
from aculptoi.checkpoints.store import RunDirectory
from aculptoi.models import ModelResponseError
from aculptoi.schemas.actions import Action
from aculptoi.schemas.construction import ConstructionPlan
from aculptoi.schemas.critique import VisualCritique, VisualIssueDetail

logger = logging.getLogger(__name__)
DEFAULT_VIEWS = ("front", "right", "top", "perspective")


class IterationBudgetExceeded(RuntimeError):
    """A safety backstop stopped an iteration before further scene mutation."""


@dataclass(frozen=True)
class RunResult:
    """Outcome and artifact location for one refinement session."""

    completed: bool
    iterations: int
    execution_batches: int
    final_score: float | None
    run_directory: Path


class RefinementLoop:
    """Coordinate plan-first Actor work and read-only visual critique."""

    def __init__(
        self,
        actor: Actor,
        critic: VisionCritic,
        blender: BlenderClient,
        checkpoints: CheckpointStore,
        max_iterations: int,
        score_target: float,
        max_actor_requests_per_iteration: int = 100,
        max_actions_per_iteration: int = 1_000,
        iteration_timeout_seconds: float = 3_600.0,
        capture_raw_model_responses: bool = True,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.blender = blender
        self.checkpoints = checkpoints
        self.max_iterations = max_iterations
        self.score_target = score_target
        self.max_actor_requests_per_iteration = max_actor_requests_per_iteration
        self.max_actions_per_iteration = max_actions_per_iteration
        self.iteration_timeout_seconds = iteration_timeout_seconds
        self.capture_raw_model_responses = capture_raw_model_responses
        self._clock = clock

    def _record_model_failure(
        self,
        run: RunDirectory,
        iteration: int,
        role: str,
        error: ModelResponseError,
        *,
        artifact_prefix: str,
        context: dict[str, object] | None = None,
    ) -> None:
        """Persist local debugging evidence without changing the validation path."""
        if not self.capture_raw_model_responses:
            return
        self.checkpoints.save_json_artifact(
            run,
            f"{artifact_prefix}-error.json",
            {
                "iteration": iteration,
                "role": role,
                "error": str(error),
                **(context or {}),
            },
        )
        if error.raw_response is not None:
            path = self.checkpoints.save_text_artifact(
                run,
                f"{artifact_prefix}-response-raw.txt",
                error.raw_response,
            )
            logger.debug("[debug] saved raw %s response to %s", role, path)

    def _raise_budget_exceeded(
        self,
        run: RunDirectory,
        iteration: int,
        *,
        budget: str,
        limit: int | float,
        actor_requests: int,
        actions_executed: int,
        started_at: float,
        detail: str,
    ) -> None:
        elapsed_seconds = self._clock() - started_at
        self.checkpoints.save_json_artifact(
            run,
            f"iteration-{iteration:03d}/budget-exhausted.json",
            {
                "iteration": iteration,
                "budget": budget,
                "limit": limit,
                "actor_requests": actor_requests,
                "actions_executed": actions_executed,
                "elapsed_seconds": elapsed_seconds,
                "detail": detail,
            },
            overwrite=False,
        )
        raise IterationBudgetExceeded(
            f"Iteration {iteration} exceeded its {budget} safety budget ({limit}): {detail}"
        )

    def _check_time_budget(
        self,
        run: RunDirectory,
        iteration: int,
        *,
        actor_requests: int,
        actions_executed: int,
        started_at: float,
    ) -> None:
        if self._clock() - started_at <= self.iteration_timeout_seconds:
            return
        self._raise_budget_exceeded(
            run,
            iteration,
            budget="iteration_timeout_seconds",
            limit=self.iteration_timeout_seconds,
            actor_requests=actor_requests,
            actions_executed=actions_executed,
            started_at=started_at,
            detail="the iteration deadline passed before the next operation",
        )

    def _check_actor_request_budget(
        self,
        run: RunDirectory,
        iteration: int,
        *,
        actor_requests: int,
        actions_executed: int,
        started_at: float,
    ) -> None:
        if actor_requests < self.max_actor_requests_per_iteration:
            return
        self._raise_budget_exceeded(
            run,
            iteration,
            budget="max_actor_requests_per_iteration",
            limit=self.max_actor_requests_per_iteration,
            actor_requests=actor_requests,
            actions_executed=actions_executed,
            started_at=started_at,
            detail="the Actor requested more work without completing the iteration plan",
        )

    def _check_action_budget(
        self,
        run: RunDirectory,
        iteration: int,
        *,
        proposed_actions: int,
        actor_requests: int,
        actions_executed: int,
        started_at: float,
    ) -> None:
        if actions_executed + proposed_actions <= self.max_actions_per_iteration:
            return
        self._raise_budget_exceeded(
            run,
            iteration,
            budget="max_actions_per_iteration",
            limit=self.max_actions_per_iteration,
            actor_requests=actor_requests,
            actions_executed=actions_executed,
            started_at=started_at,
            detail=(
                f"the next action batch contains {proposed_actions} actions with only "
                f"{self.max_actions_per_iteration - actions_executed} remaining"
            ),
        )

    @staticmethod
    def _action_object_names(actions: Sequence[Action]) -> tuple[list[str], list[str]]:
        """Extract trace-only object names from already validated action models."""
        created: set[str] = set()
        affected: set[str] = set()
        for action in actions:
            payload = action.model_dump(mode="json")
            name = payload.get("name")
            object_name = payload.get("object")
            if isinstance(name, str):
                created.add(name)
                affected.add(name)
            if isinstance(object_name, str):
                affected.add(object_name)
        return sorted(created), sorted(affected)

    def run(self, goal: str) -> RunResult:
        """Create a self-contained run and execute it against its canonical scene."""
        run = self.checkpoints.create_run(goal)
        self.checkpoints.save_text_artifact(run, "user-prompt.txt", goal)
        self.blender.attach_run(self.checkpoints.canonical_scene_path(run), run.id)
        self.checkpoints.preserve_initial_scene(run)
        state = self.checkpoints.load_run_state(run).model_copy(update={"status": "running"})
        self.checkpoints.save_run_state(run, state)
        try:
            return self._run_existing(run, goal)
        except Exception:
            current = self.checkpoints.load_run_state(run)
            if current.status not in {"completed", "stopped"}:
                self.checkpoints.save_run_state(
                    run, current.model_copy(update={"status": "interrupted"})
                )
            raise
        finally:
            self._release_worker_run()

    def resume(self, run: RunDirectory) -> RunResult:
        """Restore the latest durable state and restart an interrupted item from batch one."""
        state = self.checkpoints.load_run_state(run)
        if state.status == "completed":
            raise RunStateError(f"run {run.id:06d} is already complete")
        if not state.goal:
            raise RunStateError(f"run {run.id:06d} has no recoverable goal")
        active = state.active_item
        canonical = self.checkpoints.restore_recovery_base(run)
        self.blender.attach_run(canonical, run.id, reload=True)
        recovering = state.model_copy(update={"status": "recovering"})
        if active is not None:
            recovering = recovering.model_copy(
                update={"active_item": active.model_copy(update={"attempt": active.attempt + 1})}
            )
        self.checkpoints.save_run_state(run, recovering)
        try:
            return self._run_existing(
                run,
                state.goal,
                start_iteration=active.iteration if active else max(1, state.iteration + 1),
                resume_active=recovering.active_item,
            )
        except Exception:
            current = self.checkpoints.load_run_state(run)
            if current.status not in {"completed", "stopped"}:
                self.checkpoints.save_run_state(
                    run, current.model_copy(update={"status": "interrupted"})
                )
            raise
        finally:
            self._release_worker_run()

    def _release_worker_run(self) -> None:
        """Release ownership after a loop exits without hiding the observer scene."""
        try:
            self.blender.release_run()
        except Exception as error:
            logger.warning("[blender] could not release run ownership: %s", error)

    def _run_existing(
        self,
        run: RunDirectory,
        goal: str,
        *,
        start_iteration: int = 1,
        resume_active: ActiveWorkItem | None = None,
    ) -> RunResult:
        """Run work against an attached canonical scene and durable run state."""
        critique: VisualCritique | None = None
        execution_batches = 0

        for iteration in range(start_iteration, self.max_iterations + 1):
            state = self.checkpoints.load_run_state(run).model_copy(
                update={"status": "running", "iteration": iteration}
            )
            self.checkpoints.save_run_state(run, state)
            iteration_started_at = self._clock()
            iteration_directory = self.checkpoints.iteration_directory(run, iteration)
            actor_requests = 0
            actions_executed = 0
            iteration_action_batches = 0
            completed_work_item_ids: list[str] = []
            completed_work_items: list[dict[str, object]] = []
            work_item_records: list[dict[str, object]] = []
            plan_id = f"run-{run.id:06d}-iteration-{iteration:03d}"

            self._check_actor_request_budget(
                run,
                iteration,
                actor_requests=actor_requests,
                actions_executed=actions_executed,
                started_at=iteration_started_at,
            )
            self._check_time_budget(
                run,
                iteration,
                actor_requests=actor_requests,
                actions_executed=actions_executed,
                started_at=iteration_started_at,
            )
            is_resuming_active_iteration = (
                resume_active is not None and resume_active.iteration == iteration
            )
            if is_resuming_active_iteration:
                assert resume_active is not None
                plan_path = iteration_directory / "construction-plan.json"
                try:
                    plan_artifact = json.loads(plan_path.read_text(encoding="utf-8"))
                    construction_plan = ConstructionPlan.model_validate(plan_artifact["plan"])
                except (OSError, ValueError, KeyError) as error:
                    raise RunStateError(
                        "cannot resume: the active work item's immutable construction plan "
                        "is missing or invalid"
                    ) from error
                if not any(
                    item.id == resume_active.work_item_id for item in construction_plan.items
                ):
                    raise RunStateError(
                        "cannot resume: active work item is absent from its construction plan"
                    )
                logger.info(
                    "[recovery] restarting work item %s from its first batch",
                    resume_active.work_item_id,
                )
            else:
                logger.info("[actor] planning iteration %s", iteration)
                scene = self.blender.scene_inspect()
                plan_messages = self.actor.build_construction_plan_messages(
                    goal,
                    scene,
                    critique,
                    iteration=iteration,
                    max_actor_requests=self.max_actor_requests_per_iteration,
                    max_actions=self.max_actions_per_iteration,
                )
                self.checkpoints.save_json_artifact(
                    run,
                    f"iteration-{iteration:03d}/construction-plan-prompt.json",
                    self.actor.construction_plan_request_artifact(plan_messages),
                    overwrite=False,
                )
                actor_requests += 1
                try:
                    construction_plan = self.actor.plan_iteration_messages(plan_messages)
                except ModelResponseError as error:
                    self._record_model_failure(
                        run,
                        iteration,
                        "actor_construction_planner",
                        error,
                        artifact_prefix=f"iteration-{iteration:03d}/construction-plan",
                    )
                    raise

                plan_artifact = {
                    "id": plan_id,
                    "iteration": iteration,
                    "goal": goal,
                    "plan": construction_plan.model_dump(mode="json"),
                }
                self.checkpoints.save_json_artifact(
                    run,
                    f"iteration-{iteration:03d}/construction-plan.json",
                    plan_artifact,
                    overwrite=False,
                )
            if is_resuming_active_iteration and resume_active is not None:
                completed_before_resume = construction_plan.items[: resume_active.ordinal - 1]
                completed_work_item_ids = [item.id for item in completed_before_resume]
                completed_work_items = [
                    {
                        "id": item.id,
                        "title": item.title,
                        "objective": item.objective,
                        "completion_criteria": [],
                        "created_object_names": [],
                        "affected_object_names": [],
                    }
                    for item in completed_before_resume
                ]
            first_ordinal = (
                resume_active.ordinal if is_resuming_active_iteration and resume_active else 1
            )
            minimum_item_requests = len(construction_plan.items) - first_ordinal + 1
            remaining_requests = self.max_actor_requests_per_iteration - actor_requests
            if minimum_item_requests > remaining_requests:
                self._raise_budget_exceeded(
                    run,
                    iteration,
                    budget="max_actor_requests_per_iteration",
                    limit=self.max_actor_requests_per_iteration,
                    actor_requests=actor_requests,
                    actions_executed=actions_executed,
                    started_at=iteration_started_at,
                    detail=(
                        f"the construction plan has {minimum_item_requests} items but only "
                        f"{remaining_requests} Actor requests remain"
                    ),
                )

            for ordinal, work_item in enumerate(construction_plan.items, start=1):
                if ordinal < first_ordinal:
                    continue
                state = self.checkpoints.load_run_state(run).model_copy(
                    update={
                        "active_item": ActiveWorkItem(
                            iteration=iteration,
                            ordinal=ordinal,
                            work_item_id=work_item.id,
                            attempt=(
                                resume_active.attempt
                                if is_resuming_active_iteration
                                and resume_active is not None
                                and ordinal == resume_active.ordinal
                                else 1
                            ),
                        ),
                        "status": "running",
                    }
                )
                self.checkpoints.save_run_state(run, state)
                item_directory = self.checkpoints.work_item_directory(
                    run, iteration, ordinal, work_item.id
                )
                item_relative = item_directory.relative_to(run.path).as_posix()
                recovery_attempt = (
                    resume_active.attempt
                    if is_resuming_active_iteration
                    and resume_active is not None
                    and ordinal == resume_active.ordinal
                    else None
                )
                if recovery_attempt is not None:
                    item_relative = f"{item_relative}/recovery-attempt-{recovery_attempt:03d}"
                item_artifact = {
                    "construction_plan_id": plan_id,
                    "iteration": iteration,
                    "ordinal": ordinal,
                    "item": work_item.model_dump(mode="json"),
                }
                self.checkpoints.save_json_artifact(
                    run,
                    f"{item_relative}/item.json",
                    item_artifact,
                    overwrite=recovery_attempt is not None,
                )

                action_batch_number = 0
                item_actions_executed = 0
                completion_criteria: tuple[str, ...] | None = None
                created_object_names: set[str] = set()
                affected_object_names: set[str] = set()
                recent_execution: dict[str, object] | None = None
                action_batch_records: list[dict[str, object]] = []
                while True:
                    self._check_actor_request_budget(
                        run,
                        iteration,
                        actor_requests=actor_requests,
                        actions_executed=actions_executed,
                        started_at=iteration_started_at,
                    )
                    self._check_time_budget(
                        run,
                        iteration,
                        actor_requests=actor_requests,
                        actions_executed=actions_executed,
                        started_at=iteration_started_at,
                    )
                    action_batch_number += 1
                    logger.info(
                        "[actor] iteration %s item %s action batch %s",
                        iteration,
                        work_item.id,
                        action_batch_number,
                    )
                    scene = self.blender.scene_inspect()
                    work_item_messages = self.actor.build_work_item_messages(
                        goal,
                        scene,
                        critique,
                        construction_plan,
                        work_item,
                        iteration=iteration,
                        action_batch=action_batch_number,
                        completed_work_item_ids=completed_work_item_ids,
                        completed_work_items=completed_work_items,
                        completion_criteria=completion_criteria,
                        remaining_actor_requests=(
                            self.max_actor_requests_per_iteration - actor_requests - 1
                        ),
                        remaining_actions=self.max_actions_per_iteration - actions_executed,
                        recent_execution=recent_execution,
                    )
                    self.checkpoints.save_json_artifact(
                        run,
                        f"{item_relative}/actor-prompt-{action_batch_number:03d}.json",
                        self.actor.work_item_request_artifact(work_item_messages),
                        overwrite=False,
                    )
                    actor_requests += 1
                    try:
                        action_batch = self.actor.execute_work_item_messages(
                            work_item_messages,
                            expected_work_item_id=work_item.id,
                            require_completion_criteria=completion_criteria is None,
                        )
                    except ModelResponseError as error:
                        self._record_model_failure(
                            run,
                            iteration,
                            "actor_work_item",
                            error,
                            artifact_prefix=(
                                f"{item_relative}/action-batch-{action_batch_number:03d}"
                            ),
                            context={
                                "construction_plan_id": plan_id,
                                "work_item_id": work_item.id,
                                "action_batch": action_batch_number,
                            },
                        )
                        raise

                    action_batch_artifact = {
                        "construction_plan_id": plan_id,
                        "iteration": iteration,
                        "work_item_id": work_item.id,
                        "action_batch": action_batch_number,
                        "response": action_batch.model_dump(mode="json"),
                    }
                    action_batch_path = self.checkpoints.save_json_artifact(
                        run,
                        f"{item_relative}/action-batch-{action_batch_number:03d}.json",
                        action_batch_artifact,
                        overwrite=False,
                    )
                    if completion_criteria is None:
                        if action_batch.completion_criteria is None:
                            raise RuntimeError(
                                "Work-item response validation did not produce completion criteria"
                            )
                        completion_criteria = tuple(action_batch.completion_criteria)
                        self.checkpoints.save_json_artifact(
                            run,
                            f"{item_relative}/completion-criteria.json",
                            {
                                "construction_plan_id": plan_id,
                                "iteration": iteration,
                                "work_item_id": work_item.id,
                                "criteria": list(completion_criteria),
                            },
                            overwrite=False,
                        )
                    self._check_time_budget(
                        run,
                        iteration,
                        actor_requests=actor_requests,
                        actions_executed=actions_executed,
                        started_at=iteration_started_at,
                    )
                    self._check_action_budget(
                        run,
                        iteration,
                        proposed_actions=len(action_batch.actions),
                        actor_requests=actor_requests,
                        actions_executed=actions_executed,
                        started_at=iteration_started_at,
                    )
                    batch_created_names, batch_affected_names = self._action_object_names(
                        action_batch.actions
                    )
                    created_object_names.update(batch_created_names)
                    affected_object_names.update(batch_affected_names)

                    if action_batch.actions:
                        logger.info("[blender] executing %s actions", len(action_batch.actions))
                        execution = self.blender.execute(action_batch.actions)
                        execution_batches += 1
                    else:
                        execution = {"executed": []}
                    try:
                        canonical_save = self.blender.save_canonical_scene(
                            self.checkpoints.canonical_scene_path(run)
                        )
                    except Exception as error:
                        self.checkpoints.save_json_artifact(
                            run,
                            f"{item_relative}/canonical-save-error-{action_batch_number:03d}.json",
                            {
                                "construction_plan_id": plan_id,
                                "iteration": iteration,
                                "work_item_id": work_item.id,
                                "action_batch": action_batch_number,
                                "execution": execution,
                                "error": str(error),
                            },
                            overwrite=False,
                        )
                        raise RuntimeError(
                            "Blender executed an action batch but could not save the "
                            "canonical scene"
                        ) from error
                    batch_action_count = len(action_batch.actions)
                    actions_executed += batch_action_count
                    item_actions_executed += batch_action_count
                    iteration_action_batches += 1
                    action_result = {
                        "construction_plan_id": plan_id,
                        "iteration": iteration,
                        "work_item_id": work_item.id,
                        "action_batch": action_batch_number,
                        "worker_called": bool(action_batch.actions),
                        "result": execution,
                        "canonical_scene": canonical_save,
                    }
                    action_result_path = self.checkpoints.save_json_artifact(
                        run,
                        f"{item_relative}/action-result-{action_batch_number:03d}.json",
                        action_result,
                        overwrite=False,
                    )
                    recent_execution = action_result
                    action_batch_records.append(
                        {
                            "action_batch": action_batch_number,
                            "status": action_batch.status,
                            "reason": action_batch.reason,
                            "action_count": batch_action_count,
                            "created_object_names": batch_created_names,
                            "affected_object_names": batch_affected_names,
                            "action_batch_path": str(action_batch_path.relative_to(run.path)),
                            "action_result_path": str(action_result_path.relative_to(run.path)),
                            "canonical_scene": str(
                                self.checkpoints.canonical_scene_path(run).relative_to(run.path)
                            ),
                        }
                    )
                    if action_batch.status == "complete":
                        break

                durable_item = self.checkpoints.create_checkpoint(
                    run,
                    iteration=iteration,
                    ordinal=ordinal,
                    work_item_id=work_item.id,
                )
                checkpoint_path = self.checkpoints.save_json_artifact(
                    run,
                    f"{item_relative}/checkpoint.json",
                    {
                        "construction_plan_id": plan_id,
                        "iteration": iteration,
                        "ordinal": ordinal,
                        "work_item_id": work_item.id,
                        "canonical_scene": "scene.blend",
                        "checkpoint": durable_item.checkpoint,
                        "durability": "completed-work-item",
                    },
                    overwrite=False,
                )
                previous_state = self.checkpoints.load_run_state(run)
                updated_durable_items = [*previous_state.durable_items, durable_item]
                self.checkpoints.save_run_state(
                    run,
                    previous_state.model_copy(
                        update={
                            "active_item": None,
                            "latest_completed_item": durable_item,
                            "latest_checkpoint": durable_item.checkpoint,
                            "durable_items": updated_durable_items,
                            "status": "running",
                        }
                    ),
                )
                completed_work_item_ids.append(work_item.id)
                item_record: dict[str, object] = {
                    "construction_plan_id": plan_id,
                    "iteration": iteration,
                    "ordinal": ordinal,
                    "work_item_id": work_item.id,
                    "status": "complete",
                    "completion_criteria": list(completion_criteria),
                    "created_object_names": sorted(created_object_names),
                    "affected_object_names": sorted(affected_object_names),
                    "action_batches": action_batch_records,
                    "actions_executed": item_actions_executed,
                    "checkpoint": durable_item.checkpoint,
                    "checkpoint_metadata_path": str(checkpoint_path.relative_to(run.path)),
                }
                work_item_records.append(item_record)
                self.checkpoints.save_json_artifact(
                    run,
                    f"{item_relative}/summary.json",
                    item_record,
                    overwrite=False,
                )
                completed_work_items.append(
                    {
                        "id": work_item.id,
                        "title": work_item.title,
                        "objective": work_item.objective,
                        "completion_criteria": list(completion_criteria),
                        "created_object_names": sorted(created_object_names),
                        "affected_object_names": sorted(affected_object_names),
                    }
                )

            durable_state = self.checkpoints.load_run_state(run)
            if durable_state.latest_checkpoint is None:
                raise RuntimeError("No construction item produced a durable checkpoint")

            self._check_time_budget(
                run,
                iteration,
                actor_requests=actor_requests,
                actions_executed=actions_executed,
                started_at=iteration_started_at,
            )
            logger.info("[render] generating %s", "/".join(DEFAULT_VIEWS))
            rendered = self.blender.render_views(DEFAULT_VIEWS, iteration_directory)
            render_paths = rendered.get("paths")
            if not isinstance(render_paths, list):
                raise RuntimeError("Blender worker returned an invalid render path list")
            image_paths = [Path(path) for path in render_paths if isinstance(path, str)]
            if not image_paths:
                raise RuntimeError("Blender worker did not return render paths")

            self._check_time_budget(
                run,
                iteration,
                actor_requests=actor_requests,
                actions_executed=actions_executed,
                started_at=iteration_started_at,
            )
            logger.info("[vision] discovering issues across %s renders", len(image_paths))
            critic_relative = f"iteration-{iteration:03d}/critic"
            discovery_messages, discovery_prompt = self.critic.build_discovery_request(
                goal,
                image_paths,
                previous_score=critique.score if critique else None,
            )
            self.checkpoints.save_json_artifact(
                run,
                f"{critic_relative}/discovery-prompt.json",
                discovery_prompt,
                overwrite=False,
            )
            try:
                discovery = self.critic.discover_messages(
                    discovery_messages, available_views=[path.stem for path in image_paths]
                )
            except ModelResponseError as error:
                self._record_model_failure(
                    run,
                    iteration,
                    "vision_issue_discovery",
                    error,
                    artifact_prefix=f"{critic_relative}/discovery",
                )
                raise
            self.checkpoints.save_json_artifact(
                run,
                f"{critic_relative}/discovery.json",
                discovery.model_dump(mode="json"),
                overwrite=False,
            )

            details: dict[str, VisualIssueDetail] = {}
            failures: dict[str, str] = {}
            for issue_index, issue in enumerate(discovery.issues):
                issue_relative = f"{critic_relative}/issues/{issue.id}"
                self.checkpoints.save_json_artifact(
                    run,
                    f"{issue_relative}/summary.json",
                    issue.model_dump(mode="json"),
                    overwrite=False,
                )
                if issue_index >= self.critic.max_issue_analysis_requests:
                    continue

                selected_images = self.critic.select_images_for_issue(image_paths, issue)
                analysis_messages, analysis_prompt = self.critic.build_issue_analysis_request(
                    goal,
                    issue,
                    selected_images,
                    previous_score=critique.score if critique else None,
                )
                self.checkpoints.save_json_artifact(
                    run,
                    f"{issue_relative}/analysis-prompt.json",
                    analysis_prompt,
                    overwrite=False,
                )
                try:
                    details[issue.id] = self.critic.analyze_issue_messages(
                        analysis_messages, expected_issue_id=issue.id
                    )
                except ModelResponseError as error:
                    failures[issue.id] = str(error)
                    self._record_model_failure(
                        run,
                        iteration,
                        "vision_issue_analysis",
                        error,
                        artifact_prefix=f"{issue_relative}/analysis",
                        context={"issue_id": issue.id},
                    )
                    logger.warning(
                        "[vision] analysis failed for issue %s; keeping summary", issue.id
                    )
                    continue
                self.checkpoints.save_json_artifact(
                    run,
                    f"{issue_relative}/analysis.json",
                    details[issue.id].model_dump(mode="json"),
                    overwrite=False,
                )
            critique = self.critic.assemble_critique(discovery, details, failures)
            self.checkpoints.save_metadata(
                run, f"critique-{iteration:03d}.json", critique.model_dump(mode="json")
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/vision-analysis.json",
                critique.model_dump(mode="json"),
                overwrite=False,
            )
            iteration_record = {
                "construction_plan_id": plan_id,
                "iteration": iteration,
                "goal": goal,
                "canonical_scene": "scene.blend",
                "latest_checkpoint": durable_state.latest_checkpoint,
                "work_items": work_item_records,
                "safety_budget": {
                    "actor_requests": {
                        "used": actor_requests,
                        "limit": self.max_actor_requests_per_iteration,
                    },
                    "actions": {
                        "used": actions_executed,
                        "limit": self.max_actions_per_iteration,
                    },
                    "elapsed_seconds": self._clock() - iteration_started_at,
                    "timeout_seconds": self.iteration_timeout_seconds,
                },
                "render_paths": [str(path) for path in image_paths],
                "score": critique.score,
            }
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/iteration-summary.json",
                iteration_record,
                overwrite=False,
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/checkpoint.json",
                iteration_record,
                overwrite=False,
            )
            high_count = sum(issue.severity in {"critical", "high"} for issue in critique.issues)
            logger.info("[vision] score: %.2f; %s high-priority issues", critique.score, high_count)
            logger.info(
                "[checkpoint] iteration %s saved %s durable item checkpoints "
                "after %s action batches",
                iteration,
                len(work_item_records),
                iteration_action_batches,
            )
            if critique.score >= self.score_target:
                self.checkpoints.save_run_state(
                    run,
                    self.checkpoints.load_run_state(run).model_copy(
                        update={"status": "completed", "active_item": None}
                    ),
                )
                return RunResult(True, iteration, execution_batches, critique.score, run.path)
        self.checkpoints.save_run_state(
            run,
            self.checkpoints.load_run_state(run).model_copy(
                update={"status": "stopped", "active_item": None}
            ),
        )
        return RunResult(
            False,
            self.max_iterations,
            execution_batches,
            critique.score if critique else None,
            run.path,
        )
