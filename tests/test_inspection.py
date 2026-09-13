from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from aculptoi.checkpoints import CheckpointStore
from aculptoi.config import InspectionConfig
from aculptoi.inspection import (
    InspectionBudgetExceeded,
    InspectionReviewer,
    InspectionSubsystem,
    generate_candidate_cameras,
    select_camera_views,
)
from aculptoi.models.base import Message, ModelResponseError
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.inspection import (
    CameraCandidate,
    CandidateDiagnostic,
    InspectionCameraPlan,
)


class RecordingProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Message]] = []

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, object]:
        del max_tokens, reasoning_effort
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeInspectionBlender:
    def __init__(self) -> None:
        self.plans: list[InspectionCameraPlan] = []
        self.scene_rotation = (0.0, 0.0, 0.0)
        self.scene_lights = ("UserLight",)

    def analyze_inspection_candidates(
        self, candidates: Sequence[CameraCandidate]
    ) -> dict[str, object]:
        return {
            "bounds": {
                "minimum": [-1.0, -2.0, -0.5],
                "maximum": [1.0, 2.0, 0.5],
                "center": [0.0, 0.0, 0.0],
                "radius": 2.3,
            },
            "diagnostics": [
                {
                    "camera_id": candidate.camera_id,
                    "frame_coverage": 0.4,
                    "surface_sample_ids": [index % 16, (index + 4) % 16],
                    "surface_sample_count": 16,
                    "silhouette_signature": f"{1 << (index % 63):064x}",
                }
                for index, candidate in enumerate(candidates)
            ],
        }

    def render_inspection_views(
        self, plan: InspectionCameraPlan, output_dir: Path
    ) -> dict[str, object]:
        self.plans.append(plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        shots: list[dict[str, object]] = []
        for index, view in enumerate(plan.views):
            path = output_dir / f"{view.tile_id}.png"
            Image.new("RGB", (128, 128), color=(32 + index, 64, 96)).save(path)
            shots.append(
                {
                    "camera_id": view.camera_id,
                    "tile_id": view.tile_id,
                    "path": str(path),
                    "width": 128,
                    "height": 128,
                }
            )
        return {
            "bounds": {
                "minimum": [-1.0, -2.0, -0.5],
                "maximum": [1.0, 2.0, 0.5],
                "center": [0.0, 0.0, 0.0],
                "radius": 2.3,
            },
            "framing": {"margin": 1.15, "distance": 6.0, "orthographic_scale": 5.3},
            "shots": shots,
        }


def _diagnostics(candidates: Sequence[CameraCandidate]) -> list[CandidateDiagnostic]:
    return [
        CandidateDiagnostic(
            camera_id=candidate.camera_id,
            frame_coverage=0.5,
            surface_sample_ids=[index % 16, (index + 1) % 16],
            surface_sample_count=16,
            silhouette_signature=f"{1 << (index % 63):064x}",
        )
        for index, candidate in enumerate(candidates)
    ]


def _small_config(**updates: object) -> InspectionConfig:
    defaults: dict[str, object] = {
        "min_views": 4,
        "max_views": 8,
        "candidate_views": 8,
        "coverage_target": 0.0,
        "min_view_gain": 0.01,
        "atlas_max_dimension": 512,
        "min_tile_dimension": 128,
        "max_rounds": 3,
        "max_total_views": 24,
        "max_views_per_round": 8,
        "inspection_timeout_seconds": 60.0,
    }
    return InspectionConfig(**(defaults | updates))


def test_candidate_cameras_are_deterministic_and_keep_canonical_anchors() -> None:
    first = generate_candidate_cameras(16)
    second = generate_candidate_cameras(16)

    assert first == second
    assert [camera.camera_id for camera in first[:4]] == [
        "anchor-front",
        "anchor-right",
        "anchor-rear",
        "anchor-front-upper",
    ]
    assert all(camera.canonical_anchor for camera in first[:4])
    assert len(first) == 16


def test_selection_honors_minimum_views_even_when_candidate_gain_is_low() -> None:
    candidates = generate_candidate_cameras(8)
    diagnostics = [
        CandidateDiagnostic(
            camera_id=camera.camera_id,
            frame_coverage=0.0,
            surface_sample_ids=[0],
            surface_sample_count=16,
            silhouette_signature="0" * 64,
        )
        for camera in candidates
    ]

    selection = select_camera_views(
        candidates,
        diagnostics,
        min_views=8,
        max_views=8,
        coverage_target=1.0,
        min_view_gain=0.5,
        atlas_max_dimension=1024,
        min_tile_dimension=128,
    )

    assert len(selection.views) == 8
    assert selection.stop_reason == "max_views"
    assert all(view.selection_kind == "canonical_anchor" for view in selection.views[:4])


def test_selection_stops_after_its_coverage_target_is_met() -> None:
    candidates = generate_candidate_cameras(8)
    diagnostics = _diagnostics(candidates)
    for diagnostic in diagnostics[:4]:
        diagnostic.surface_sample_ids = list(range(16))

    selection = select_camera_views(
        candidates,
        diagnostics,
        min_views=4,
        max_views=8,
        coverage_target=1.0,
        min_view_gain=0.01,
        atlas_max_dimension=1024,
        min_tile_dimension=128,
    )

    assert len(selection.views) == 4
    assert selection.estimated_surface_coverage == 1.0
    assert selection.stop_reason == "coverage_target"


def test_selection_stops_when_remaining_information_gain_is_too_low() -> None:
    candidates = generate_candidate_cameras(8)
    diagnostics = [
        CandidateDiagnostic(
            camera_id=camera.camera_id,
            frame_coverage=0.0,
            surface_sample_ids=[0],
            surface_sample_count=16,
            silhouette_signature="0" * 64,
        )
        for camera in candidates
    ]

    selection = select_camera_views(
        candidates,
        diagnostics,
        min_views=4,
        max_views=8,
        coverage_target=1.0,
        min_view_gain=0.1,
        atlas_max_dimension=1024,
        min_tile_dimension=128,
    )

    assert len(selection.views) == 4
    assert selection.stop_reason == "min_view_gain"


def test_selection_stops_before_atlas_tiles_become_too_small() -> None:
    candidates = generate_candidate_cameras(8)

    selection = select_camera_views(
        candidates,
        _diagnostics(candidates),
        min_views=4,
        max_views=8,
        coverage_target=1.0,
        min_view_gain=0.0,
        atlas_max_dimension=400,
        min_tile_dimension=200,
    )

    assert len(selection.views) == 4
    assert selection.stop_reason == "atlas_resolution"
    assert selection.layout.tile_dimension == 200


def test_accepted_inspection_persists_shots_atlas_and_manifest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("inspection test")
    provider = RecordingProvider([{"status": "accept", "confidence": 95, "problems": []}])
    blender = FakeInspectionBlender()
    subsystem = InspectionSubsystem(
        _small_config(),
        InspectionReviewer(provider),
        blender,  # type: ignore[arg-type]
    )

    accepted = subsystem.inspect(run, 1, store)

    inspection = run.path / "iteration-001" / "inspection"
    manifest = accepted.manifest
    assert accepted.summary.status == "accepted"
    assert accepted.summary.views_in_accepted_atlas == 4
    assert (inspection / "summary.json").is_file()
    assert (inspection / "round-001" / "camera-plan.json").is_file()
    assert (inspection / "round-001" / "atlas.png").is_file()
    assert (inspection / "round-001" / "atlas-manifest.json").is_file()
    assert (inspection / "round-001" / "review-prompt.json").is_file()
    assert (inspection / "round-001" / "review.json").is_file()
    assert sorted(path.name for path in (inspection / "round-001" / "shots").glob("*.png")) == [
        "A1.png",
        "A2.png",
        "B1.png",
        "B2.png",
    ]
    assert list(manifest.tiles) == ["A1", "A2", "B1", "B2"]
    assert manifest.tiles["B2"].pixel_bounds == (256, 256, 256, 256)
    assert manifest.tiles["A1"].selection_kind == "canonical_anchor"
    assert provider.calls[0][1]["content"][2]["type"] == "image_url"
    assert blender.scene_rotation == (0.0, 0.0, 0.0)
    assert blender.scene_lights == ("UserLight",)


def test_augment_and_retry_cause_bounded_new_inspection_rounds(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("inspection rounds")
    augment_provider = RecordingProvider(
        [
            {
                "status": "augment",
                "confidence": 92,
                "problems": [
                    {
                        "type": "coverage_gap",
                        "region": "lower geometry",
                        "tiles": ["A1"],
                        "view_preference": "lower",
                        "reason": "The lower contour is not clearly exposed.",
                    }
                ],
            },
            {"status": "accept", "confidence": 94, "problems": []},
        ]
    )
    blender = FakeInspectionBlender()
    accepted = InspectionSubsystem(
        _small_config(),
        InspectionReviewer(augment_provider),
        blender,  # type: ignore[arg-type]
    ).inspect(run, 1, store)

    assert accepted.summary.rounds == 2
    assert len(blender.plans[1].views) > len(blender.plans[0].views)
    assert (run.path / "iteration-001/inspection/round-002/review.json").is_file()

    retry_run = store.create_run("inspection retry")
    retry_provider = RecordingProvider(
        [
            {
                "status": "retry",
                "confidence": 90,
                "problems": [
                    {
                        "type": "lighting",
                        "tiles": ["A1"],
                        "view_preference": "any",
                        "reason": "The visible face is too dark.",
                    }
                ],
            },
            {"status": "accept", "confidence": 94, "problems": []},
        ]
    )
    retry_blender = FakeInspectionBlender()
    InspectionSubsystem(
        _small_config(),
        InspectionReviewer(retry_provider),
        retry_blender,  # type: ignore[arg-type]
    ).inspect(retry_run, 1, store)

    assert len(retry_blender.plans) == 2
    assert retry_blender.plans[1].profile == "retry"
    assert len(retry_blender.plans[1].views) == len(retry_blender.plans[0].views)


def test_reviewer_rejects_tile_references_outside_the_atlas(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("invalid reviewer tile")
    provider = RecordingProvider(
        [
            {
                "status": "augment",
                "confidence": 90,
                "problems": [
                    {
                        "type": "coverage_gap",
                        "reason": "An unlisted angle is needed.",
                        "tiles": ["Z9"],
                        "view_preference": "any",
                    }
                ],
            }
        ]
    )

    with pytest.raises(
        ModelResponseError, match="Inspection Reviewer referenced unknown atlas tile ids: Z9"
    ):
        InspectionSubsystem(
            _small_config(),
            InspectionReviewer(provider),
            FakeInspectionBlender(),  # type: ignore[arg-type]
        ).inspect(run, 1, store)


@pytest.mark.parametrize(
    ("responses", "config", "reason"),
    [
        (
            [
                {
                    "status": "augment",
                    "confidence": 90,
                    "problems": [
                        {
                            "type": "coverage_gap",
                            "reason": "More views are needed.",
                            "tiles": [],
                            "view_preference": "any",
                        }
                    ],
                }
            ],
            _small_config(max_rounds=1),
            "max_rounds",
        ),
        (
            [
                {
                    "status": "retry",
                    "confidence": 90,
                    "problems": [
                        {
                            "type": "lighting",
                            "reason": "The atlas is too dark.",
                            "tiles": ["A1"],
                            "view_preference": "any",
                        }
                    ],
                }
            ],
            _small_config(max_total_views=4),
            "max_total_views",
        ),
    ],
)
def test_inspection_limits_preserve_rejected_artifacts_and_stop(
    tmp_path: Path,
    responses: list[dict[str, object]],
    config: InspectionConfig,
    reason: str,
) -> None:
    store = CheckpointStore(tmp_path)
    run = store.create_run("rejected inspection")
    subsystem = InspectionSubsystem(
        config,
        InspectionReviewer(RecordingProvider(responses)),
        FakeInspectionBlender(),  # type: ignore[arg-type]
    )

    with pytest.raises(InspectionBudgetExceeded, match=reason):
        subsystem.inspect(run, 1, store)

    assert (run.path / "iteration-001/inspection/round-001/atlas.png").is_file()
    assert (run.path / "iteration-001/inspection/failure.json").is_file()
