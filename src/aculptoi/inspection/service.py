"""Bounded orchestration of camera planning, rendering, atlas review, and acceptance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Never

from pydantic import ValidationError

from aculptoi.blender import BlenderClient, BlenderWorkerError
from aculptoi.checkpoints import CheckpointStore
from aculptoi.checkpoints.store import RunDirectory
from aculptoi.config import InspectionConfig
from aculptoi.models import ModelProviderError, ModelResponseError
from aculptoi.schemas.inspection import (
    CandidateSurvey,
    InspectionAtlasManifest,
    InspectionCamera,
    InspectionCameraPlan,
    InspectionProfile,
    InspectionRenderResult,
    InspectionReview,
    InspectionSummary,
    SelectedCameraChoice,
)
from aculptoi.telemetry import RunEvents

from .atlas import compose_atlas
from .cameras import atlas_tile_id, generate_candidate_cameras, select_camera_views
from .reviewer import InspectionReviewer


class InspectionBudgetExceeded(RuntimeError):
    """A bounded inspection could not produce technically accepted evidence."""


@dataclass(frozen=True)
class AcceptedInspection:
    """The only inspection result which may be submitted to the Vision Critic."""

    atlas: Path
    manifest: InspectionAtlasManifest
    summary: InspectionSummary


class InspectionSubsystem:
    """Prepare a high-information atlas without mutating the working Blender scene."""

    def __init__(
        self,
        config: InspectionConfig,
        reviewer: InspectionReviewer,
        blender: BlenderClient,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._config = config
        self._reviewer = reviewer
        self._blender = blender
        self._clock = clock

    def inspect(
        self,
        run: RunDirectory,
        iteration: int,
        checkpoints: CheckpointStore,
        *,
        artifact_root: str | None = None,
        events: RunEvents | None = None,
    ) -> AcceptedInspection:
        """Run bounded technical review rounds and return only an accepted atlas."""
        started_at = self._clock()
        relative_root = (
            f"iteration-{iteration:03d}/{artifact_root}"
            if artifact_root
            else f"iteration-{iteration:03d}/inspection"
        )
        try:
            if events is None:
                candidates = generate_candidate_cameras(self._config.candidate_views)
                candidate_analysis = self._blender.analyze_inspection_candidates(candidates)
                survey = CandidateSurvey.model_validate(candidate_analysis)
            else:
                with events.stage(
                    "inspection_camera_selection",
                    iteration=iteration,
                ):
                    candidates = generate_candidate_cameras(self._config.candidate_views)
                    candidate_analysis = self._blender.analyze_inspection_candidates(candidates)
                    survey = CandidateSurvey.model_validate(candidate_analysis)
        except BlenderWorkerError as error:
            self._fail(
                run,
                checkpoints,
                relative_root,
                "candidate_analysis_failed",
                str(error),
            )
        except ValidationError as error:
            self._fail(
                run,
                checkpoints,
                relative_root,
                "candidate_analysis_invalid",
                f"Blender worker returned invalid candidate analysis: {error}",
            )
        self._check_timeout(run, checkpoints, relative_root, started_at)
        expected_ids = {candidate.camera_id for candidate in candidates}
        if {diagnostic.camera_id for diagnostic in survey.diagnostics} != expected_ids:
            self._fail(
                run,
                checkpoints,
                relative_root,
                "candidate_analysis_invalid",
                "Blender worker candidate diagnostics did not match the generated camera inventory",
            )

        selected_views: tuple[SelectedCameraChoice, ...] = ()
        previous_review: InspectionReview | None = None
        total_views_rendered = 0
        for round_number in range(1, self._config.max_rounds + 1):
            self._check_timeout(run, checkpoints, relative_root, started_at)
            trigger = "initial" if previous_review is None else previous_review.status
            preferences = (
                [problem.view_preference for problem in previous_review.problems]
                if previous_review is not None
                else []
            )
            required_additional = (
                1 if previous_review and previous_review.status == "augment" else 0
            )
            selection = select_camera_views(
                candidates,
                survey.diagnostics,
                min_views=self._config.min_views,
                max_views=min(self._config.max_views, self._config.max_views_per_round),
                coverage_target=self._config.coverage_target,
                min_view_gain=self._config.min_view_gain,
                atlas_max_dimension=self._config.atlas_max_dimension,
                min_tile_dimension=self._config.min_tile_dimension,
                existing_views=selected_views,
                required_additional_views=required_additional,
                preferred_orientations=preferences,
            )
            if required_additional and len(selection.views) <= len(selected_views):
                self._fail(
                    run,
                    checkpoints,
                    relative_root,
                    "augmentation_unavailable",
                    (
                        "Inspection Reviewer requested augmentation, but no additional "
                        "camera fits the limits"
                    ),
                )
            if total_views_rendered + len(selection.views) > self._config.max_total_views:
                self._fail(
                    run,
                    checkpoints,
                    relative_root,
                    "max_total_views",
                    (
                        f"rendering round {round_number} would use "
                        f"{total_views_rendered + len(selection.views)} views, exceeding "
                        f"inspection.max_total_views={self._config.max_total_views}"
                    ),
                )

            profile: InspectionProfile = (
                "retry" if previous_review and previous_review.status == "retry" else "standard"
            )
            plan = InspectionCameraPlan(
                sensor_version=self._config.sensor_version,
                lighting_rig=self._config.lighting_rig,
                round=round_number,
                profile=profile,
                layout=selection.layout,
                views=[
                    InspectionCamera(
                        **view.model_dump(),
                        tile_id=atlas_tile_id(index, selection.layout.columns),
                    )
                    for index, view in enumerate(selection.views)
                ],
                estimated_surface_coverage=selection.estimated_surface_coverage,
                selection_stop_reason=selection.stop_reason,
            )
            round_relative = f"{relative_root}/round-{round_number:03d}"
            checkpoints.save_json_artifact(
                run,
                f"{round_relative}/request.json",
                {
                    "round": round_number,
                    "trigger": trigger,
                    "previous_review": (
                        previous_review.model_dump(mode="json") if previous_review else None
                    ),
                    "candidate_camera_count": len(candidates),
                    "configured_limits": {
                        "max_views": self._config.max_views,
                        "max_views_per_round": self._config.max_views_per_round,
                        "max_total_views": self._config.max_total_views,
                    },
                },
                overwrite=False,
            )
            checkpoints.save_json_artifact(
                run,
                f"{round_relative}/camera-plan.json",
                {
                    "plan": plan.model_dump(mode="json"),
                    "candidate_diagnostics": [
                        diagnostic.model_dump(mode="json") for diagnostic in survey.diagnostics
                    ],
                },
                overwrite=False,
            )
            shots_directory = run.path / round_relative / "shots"
            try:
                if events is None:
                    rendered_data = self._blender.render_inspection_views(plan, shots_directory)
                    rendered = InspectionRenderResult.model_validate(rendered_data)
                else:
                    with events.stage(
                        "inspection_render",
                        iteration=iteration,
                        inspection_round=round_number,
                    ):
                        rendered_data = self._blender.render_inspection_views(plan, shots_directory)
                        rendered = InspectionRenderResult.model_validate(rendered_data)
            except BlenderWorkerError as error:
                self._fail(run, checkpoints, relative_root, "render_failed", str(error))
            except ValidationError as error:
                self._fail(
                    run,
                    checkpoints,
                    relative_root,
                    "render_result_invalid",
                    f"Blender worker returned invalid inspection render metadata: {error}",
                )
            try:
                if events is None:
                    self._validate_shots(run, rendered, plan)
                    manifest = compose_atlas(
                        plan,
                        rendered,
                        run.path / round_relative / "atlas.png",
                    )
                else:
                    with events.stage(
                        "inspection_atlas_build",
                        iteration=iteration,
                        inspection_round=round_number,
                    ):
                        self._validate_shots(run, rendered, plan)
                        manifest = compose_atlas(
                            plan,
                            rendered,
                            run.path / round_relative / "atlas.png",
                        )
            except (InspectionBudgetExceeded, ValueError) as error:
                self._fail(run, checkpoints, relative_root, "render_result_invalid", str(error))
            total_views_rendered += len(plan.views)
            atlas_path = run.path / round_relative / "atlas.png"
            checkpoints.save_json_artifact(
                run,
                f"{round_relative}/atlas-manifest.json",
                manifest.model_dump(mode="json"),
                overwrite=False,
            )
            self._check_timeout(run, checkpoints, relative_root, started_at)
            messages, review_prompt = self._reviewer.build_review_request(atlas_path, manifest)
            checkpoints.save_json_artifact(
                run,
                f"{round_relative}/review-prompt.json",
                review_prompt,
                overwrite=False,
            )
            try:
                if events is None:
                    review = self._reviewer.review_messages(messages, manifest)
                else:
                    with events.stage(
                        "inspection_review",
                        iteration=iteration,
                        provider=self._reviewer.provider_name,
                        profile=self._reviewer.inference_profile,
                        inspection_round=round_number,
                    ) as event:
                        review = self._reviewer.review_messages(
                            messages,
                            manifest,
                            usage_recorder=event.record_usage,
                        )
            except ModelProviderError as error:
                self._record_review_failure(
                    run, checkpoints, round_relative, iteration, error, round_number
                )
                raise
            checkpoints.save_json_artifact(
                run,
                f"{round_relative}/review.json",
                review.model_dump(mode="json"),
                overwrite=False,
            )
            if review.status == "accept":
                summary = InspectionSummary(
                    status="accepted",
                    rounds=round_number,
                    total_views_rendered=total_views_rendered,
                    views_in_accepted_atlas=len(plan.views),
                    estimated_surface_coverage=selection.estimated_surface_coverage,
                    accepted_atlas=f"round-{round_number:03d}/atlas.png",
                    accepted_manifest=f"round-{round_number:03d}/atlas-manifest.json",
                    sensor_version=plan.sensor_version,
                    lighting_rig=plan.lighting_rig,
                )
                checkpoints.save_json_artifact(
                    run,
                    f"{relative_root}/summary.json",
                    summary.model_dump(mode="json"),
                    overwrite=False,
                )
                return AcceptedInspection(atlas=atlas_path, manifest=manifest, summary=summary)
            selected_views = tuple(selection.views)
            previous_review = review

        self._fail(
            run,
            checkpoints,
            relative_root,
            "max_rounds",
            (
                f"Inspection Reviewer did not accept evidence within "
                f"inspection.max_rounds={self._config.max_rounds}"
            ),
        )

    def _check_timeout(
        self,
        run: RunDirectory,
        checkpoints: CheckpointStore,
        relative_root: str,
        started_at: float,
    ) -> None:
        elapsed_seconds = self._clock() - started_at
        if elapsed_seconds <= self._config.inspection_timeout_seconds:
            return
        self._fail(
            run,
            checkpoints,
            relative_root,
            "inspection_timeout",
            (
                f"inspection exceeded inspection_timeout_seconds="
                f"{self._config.inspection_timeout_seconds} after {elapsed_seconds:.3f} seconds"
            ),
        )

    @staticmethod
    def _validate_shots(
        run: RunDirectory, rendered: InspectionRenderResult, plan: InspectionCameraPlan
    ) -> None:
        expected = {(view.camera_id, view.tile_id) for view in plan.views}
        actual = {(shot.camera_id, shot.tile_id) for shot in rendered.shots}
        if actual != expected:
            raise InspectionBudgetExceeded(
                "Blender worker shots do not match the inspection camera plan"
            )
        for shot in rendered.shots:
            path = Path(shot.path).resolve()
            try:
                path.relative_to(run.path.resolve())
            except ValueError as error:
                raise InspectionBudgetExceeded(
                    "Blender worker returned an inspection shot outside the active run"
                ) from error
            if not path.is_file():
                raise InspectionBudgetExceeded(
                    f"Blender worker reported a missing inspection shot: {path}"
                )

    @staticmethod
    def _record_review_failure(
        run: RunDirectory,
        checkpoints: CheckpointStore,
        round_relative: str,
        iteration: int,
        error: ModelProviderError,
        round_number: int,
    ) -> None:
        checkpoints.save_json_artifact(
            run,
            f"{round_relative}/review-error.json",
            {
                "iteration": iteration,
                "round": round_number,
                "role": "inspection_reviewer",
                "error": str(error),
            },
            overwrite=False,
        )
        if isinstance(error, ModelResponseError) and error.raw_response is not None:
            checkpoints.save_text_artifact(
                run,
                f"{round_relative}/review-response-raw.txt",
                error.raw_response,
            )

    @staticmethod
    def _fail(
        run: RunDirectory,
        checkpoints: CheckpointStore,
        relative_root: str,
        reason: str,
        detail: str,
    ) -> Never:
        checkpoints.save_json_artifact(
            run,
            f"{relative_root}/failure.json",
            {"reason": reason, "detail": detail},
            overwrite=False,
        )
        raise InspectionBudgetExceeded(f"Inspection failed ({reason}): {detail}")
