"""A deliberately small state-machine implementation for V1 orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aculptoi.agent.actor import Actor
from aculptoi.agent.critic import VisionCritic
from aculptoi.blender.client import BlenderClient
from aculptoi.checkpoints import CheckpointStore
from aculptoi.checkpoints.store import RunDirectory
from aculptoi.models import ModelResponseError
from aculptoi.schemas.critique import VisualCritique

logger = logging.getLogger(__name__)
DEFAULT_VIEWS = ("front", "right", "top", "perspective")


@dataclass(frozen=True)
class RunResult:
    """Outcome and artifact location for one refinement session."""

    completed: bool
    iterations: int
    final_score: float | None
    run_directory: Path


class RefinementLoop:
    """Coordinate independent actor and critic roles through the safe worker API."""

    def __init__(
        self,
        actor: Actor,
        critic: VisionCritic,
        blender: BlenderClient,
        checkpoints: CheckpointStore,
        max_iterations: int,
        score_target: float,
        capture_raw_model_responses: bool = True,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.blender = blender
        self.checkpoints = checkpoints
        self.max_iterations = max_iterations
        self.score_target = score_target
        self.capture_raw_model_responses = capture_raw_model_responses

    def _record_model_failure(
        self, run: RunDirectory, iteration: int, role: str, error: ModelResponseError
    ) -> None:
        """Optionally persist local debugging evidence without changing control flow."""
        if not self.capture_raw_model_responses:
            return
        self.checkpoints.save_metadata(
            run,
            f"{role}-error-{iteration:03d}.json",
            {"iteration": iteration, "role": role, "error": str(error)},
        )
        if error.raw_response is not None:
            path = self.checkpoints.save_text_artifact(
                run,
                f"iteration-{iteration:03d}/{role}-response-raw.txt",
                error.raw_response,
            )
            logger.debug("[debug] saved raw %s response to %s", role, path)

    def run(self, goal: str) -> RunResult:
        """Run bounded iterations; all durable artifacts are stored in one run directory."""
        run = self.checkpoints.create_run()
        # Keep the exact human request separate from role-specific JSON prompts so
        # a run remains understandable without reconstructing any model context.
        self.checkpoints.save_text_artifact(run, "user-prompt.txt", goal)
        critique: VisualCritique | None = None
        recent_execution: dict[str, object] | None = None
        for iteration in range(1, self.max_iterations + 1):
            iteration_directory = self.checkpoints.iteration_directory(run, iteration)
            logger.info("[actor] planning iteration %s", iteration)
            scene = self.blender.scene_inspect()
            actor_messages = self.actor.build_messages(
                goal,
                scene,
                critique,
                iteration=iteration,
                recent_execution=recent_execution,
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/actor-prompt.json",
                self.actor.request_artifact(actor_messages),
            )
            try:
                plan = self.actor.plan_messages(actor_messages)
            except ModelResponseError as error:
                self._record_model_failure(run, iteration, "actor", error)
                raise
            self.checkpoints.save_metadata(
                run, f"actor-plan-{iteration:03d}.json", plan.model_dump(mode="json")
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/actor-plan.json",
                plan.model_dump(mode="json"),
            )

            logger.info("[blender] executing %s actions", len(plan.actions))
            execution = self.blender.execute(plan.actions)
            recent_execution = execution
            self.checkpoints.save_metadata(run, f"actions-{iteration:03d}.json", execution)
            self.checkpoints.save_json_artifact(
                run, f"iteration-{iteration:03d}/actions.json", execution
            )

            logger.info("[render] generating %s", "/".join(DEFAULT_VIEWS))
            rendered = self.blender.render_views(DEFAULT_VIEWS, iteration_directory)
            render_paths = rendered.get("paths")
            if not isinstance(render_paths, list):
                raise RuntimeError("Blender worker returned an invalid render path list")
            image_paths = [Path(path) for path in render_paths if isinstance(path, str)]
            if not image_paths:
                raise RuntimeError("Blender worker did not return render paths")

            logger.info("[vision] inspecting %s renders", len(image_paths))
            critic_messages, critic_prompt = self.critic.build_request(
                goal,
                image_paths,
                previous_score=critique.score if critique else None,
            )
            self.checkpoints.save_json_artifact(
                run, f"iteration-{iteration:03d}/vision-prompt.json", critic_prompt
            )
            try:
                critique = self.critic.inspect_messages(critic_messages)
            except ModelResponseError as error:
                self._record_model_failure(run, iteration, "critic", error)
                raise
            self.checkpoints.save_metadata(
                run, f"critique-{iteration:03d}.json", critique.model_dump(mode="json")
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/vision-analysis.json",
                critique.model_dump(mode="json"),
            )
            snapshot = self.blender.checkpoint_save(f"run-{run.id:06d}-iteration-{iteration:03d}")
            run_snapshot = self.checkpoints.copy_checkpoint_to_iteration(run, iteration, snapshot)
            self.checkpoints.save_metadata(
                run,
                f"checkpoint-{iteration:03d}.json",
                {
                    "iteration": iteration,
                    "goal": goal,
                    "scene_snapshot": snapshot,
                    "run_snapshot_path": str(run_snapshot.relative_to(run.path)),
                    "render_paths": [str(path) for path in image_paths],
                    "score": critique.score,
                },
            )
            self.checkpoints.save_json_artifact(
                run,
                f"iteration-{iteration:03d}/checkpoint.json",
                {
                    "iteration": iteration,
                    "goal": goal,
                    "scene_snapshot": snapshot,
                    "run_snapshot_path": str(run_snapshot.relative_to(run.path)),
                    "render_paths": [str(path) for path in image_paths],
                    "score": critique.score,
                },
            )
            high_count = sum(issue.severity == "high" for issue in critique.issues)
            logger.info("[vision] score: %.2f; %s high-priority issues", critique.score, high_count)
            logger.info("[checkpoint] iteration %s saved", iteration)
            if critique.score >= self.score_target:
                return RunResult(True, iteration, critique.score, run.path)
        return RunResult(False, self.max_iterations, critique.score if critique else None, run.path)
