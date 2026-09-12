"""A deliberately small state-machine implementation for V1 orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aculptoi.agent.actor import Actor
from aculptoi.agent.critic import VisionCritic
from aculptoi.blender.client import BlenderClient
from aculptoi.checkpoints import CheckpointStore
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
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.blender = blender
        self.checkpoints = checkpoints
        self.max_iterations = max_iterations
        self.score_target = score_target

    def run(self, goal: str) -> RunResult:
        """Run bounded iterations; all durable artifacts are stored in one run directory."""
        run = self.checkpoints.create_run()
        critique: VisualCritique | None = None
        for iteration in range(1, self.max_iterations + 1):
            logger.info("[actor] planning iteration %s", iteration)
            scene = self.blender.scene_inspect()
            plan = self.actor.plan(goal, scene, critique)
            self.checkpoints.save_metadata(
                run, f"actor-plan-{iteration:03d}.json", plan.model_dump(mode="json")
            )

            logger.info("[blender] executing %s actions", len(plan.actions))
            execution = self.blender.execute(plan.actions)
            self.checkpoints.save_metadata(run, f"actions-{iteration:03d}.json", execution)

            iteration_directory = run.path / f"iteration-{iteration:03d}"
            logger.info("[render] generating %s", "/".join(DEFAULT_VIEWS))
            rendered = self.blender.render_views(DEFAULT_VIEWS, iteration_directory)
            render_paths = rendered.get("paths")
            if not isinstance(render_paths, list):
                raise RuntimeError("Blender worker returned an invalid render path list")
            image_paths = [Path(path) for path in render_paths if isinstance(path, str)]
            if not image_paths:
                raise RuntimeError("Blender worker did not return render paths")

            logger.info("[vision] inspecting %s renders", len(image_paths))
            critique = self.critic.inspect(goal, image_paths)
            self.checkpoints.save_metadata(
                run, f"critique-{iteration:03d}.json", critique.model_dump(mode="json")
            )
            snapshot = self.blender.checkpoint_save(f"run-{run.id:06d}-iteration-{iteration:03d}")
            self.checkpoints.save_metadata(
                run,
                f"checkpoint-{iteration:03d}.json",
                {
                    "iteration": iteration,
                    "goal": goal,
                    "scene_snapshot": snapshot,
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
