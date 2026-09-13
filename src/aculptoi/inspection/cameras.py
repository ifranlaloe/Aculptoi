"""Pure deterministic candidate generation, greedy selection, and atlas layout."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from aculptoi.schemas.inspection import (
    AtlasLayout,
    CameraCandidate,
    CandidateDiagnostic,
    InspectionStopReason,
    SelectedCameraChoice,
)

_ANCHORS: tuple[CameraCandidate, ...] = (
    CameraCandidate(
        camera_id="anchor-front",
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        orientation="front",
        projection="orthographic",
        canonical_anchor=True,
    ),
    CameraCandidate(
        camera_id="anchor-right",
        azimuth_degrees=90.0,
        elevation_degrees=0.0,
        orientation="right",
        projection="orthographic",
        canonical_anchor=True,
    ),
    CameraCandidate(
        camera_id="anchor-rear",
        azimuth_degrees=180.0,
        elevation_degrees=0.0,
        orientation="rear",
        projection="orthographic",
        canonical_anchor=True,
    ),
    CameraCandidate(
        camera_id="anchor-front-upper",
        azimuth_degrees=0.0,
        elevation_degrees=35.0,
        orientation="front-upper",
        projection="perspective",
        canonical_anchor=True,
    ),
)


@dataclass(frozen=True)
class CameraSelection:
    """Selection output with an explicitly labelled approximate coverage metric."""

    views: tuple[SelectedCameraChoice, ...]
    estimated_surface_coverage: float
    stop_reason: InspectionStopReason
    layout: AtlasLayout


def generate_candidate_cameras(candidate_views: int) -> list[CameraCandidate]:
    """Generate a stable world-space survey with four canonical anchors.

    Azimuth zero is the world-space front convention: a camera on negative Y
    looking toward the inspection center. Positive azimuth advances toward +X.
    The dynamic directions use a Fibonacci sphere so no object-class assumption
    is needed and the lower hemisphere remains observable.
    """
    if candidate_views < len(_ANCHORS):
        raise ValueError(f"candidate_views must be at least {len(_ANCHORS)}")

    dynamic_count = candidate_views - len(_ANCHORS)
    candidates = list(_ANCHORS)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(dynamic_count):
        fraction = (index + 0.5) / dynamic_count
        z = 1.0 - 2.0 * fraction
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        longitude = index * golden_angle
        x = math.cos(longitude) * radius
        y = math.sin(longitude) * radius
        azimuth = math.degrees(math.atan2(x, -y))
        elevation = math.degrees(math.asin(z))
        candidates.append(
            CameraCandidate(
                camera_id=f"candidate-{index + 1:03d}",
                azimuth_degrees=round(azimuth, 4),
                elevation_degrees=round(elevation, 4),
                orientation=_orientation_label(azimuth, elevation),
                projection="perspective",
            )
        )
    return candidates


def plan_atlas_layout(view_count: int, atlas_max_dimension: int) -> AtlasLayout:
    """Choose a compact grid that maximizes square tile resolution."""
    if view_count < 1:
        raise ValueError("view_count must be positive")
    if atlas_max_dimension < 1:
        raise ValueError("atlas_max_dimension must be positive")

    layouts: list[tuple[int, int, int, int, int]] = []
    for columns in range(1, view_count + 1):
        rows = math.ceil(view_count / columns)
        tile_dimension = atlas_max_dimension // max(columns, rows)
        if tile_dimension < 1:
            continue
        unused_tiles = columns * rows - view_count
        skew = abs(columns - rows)
        layouts.append((tile_dimension, -unused_tiles, -skew, columns, rows))
    if not layouts:
        raise ValueError("atlas dimensions cannot accommodate one tile")
    _, _, _, columns, rows = max(layouts)
    tile_dimension = atlas_max_dimension // max(columns, rows)
    return AtlasLayout(
        columns=columns,
        rows=rows,
        tile_dimension=tile_dimension,
        width=columns * tile_dimension,
        height=rows * tile_dimension,
    )


def atlas_tile_id(index: int, columns: int) -> str:
    """Return the visible spreadsheet-style tile identity for a zero-based index."""
    if index < 0 or columns < 1:
        raise ValueError("atlas tile index and columns must be positive")
    row, column = divmod(index, columns)
    letters = ""
    row += 1
    while row:
        row, remainder = divmod(row - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}{column + 1}"


def select_camera_views(
    candidates: Sequence[CameraCandidate],
    diagnostics: Sequence[CandidateDiagnostic],
    *,
    min_views: int,
    max_views: int,
    coverage_target: float,
    min_view_gain: float,
    atlas_max_dimension: int,
    min_tile_dimension: int,
    existing_views: Sequence[SelectedCameraChoice] = (),
    required_additional_views: int = 0,
    preferred_orientations: Iterable[str] = (),
) -> CameraSelection:
    """Greedily maximize estimated surface novelty without shrinking atlas tiles.

    Estimated coverage is the union of the worker's deterministic, front-facing
    mesh-surface sample identifiers. Information gain combines new sample
    coverage, low-resolution silhouette novelty, and useful screen occupancy.
    It is intentionally an observation-planning estimate, not geometric truth.
    """
    if min_views < len(_ANCHORS):
        raise ValueError(f"min_views must be at least {len(_ANCHORS)}")
    if max_views < min_views:
        raise ValueError("max_views must be at least min_views")
    if not 0.0 <= coverage_target <= 1.0:
        raise ValueError("coverage_target must be between zero and one")
    if not 0.0 <= min_view_gain <= 1.0:
        raise ValueError("min_view_gain must be between zero and one")
    if required_additional_views < 0:
        raise ValueError("required_additional_views must not be negative")

    candidate_by_id = {candidate.camera_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("candidate camera ids must be unique")
    diagnostic_by_id = {diagnostic.camera_id: diagnostic for diagnostic in diagnostics}
    if set(candidate_by_id) != set(diagnostic_by_id):
        raise ValueError("candidate diagnostics must match the candidate camera ids exactly")
    sample_counts = {diagnostic.surface_sample_count for diagnostic in diagnostics}
    if len(sample_counts) != 1:
        raise ValueError("candidate diagnostics must share one surface sample population")
    sample_count = sample_counts.pop()

    choices_by_id = {view.camera_id: view for view in existing_views}
    if len(choices_by_id) != len(existing_views):
        raise ValueError("existing selected camera ids must be unique")
    for anchor in _ANCHORS:
        choices_by_id.setdefault(
            anchor.camera_id,
            SelectedCameraChoice(
                **anchor.model_dump(),
                selection_kind="canonical_anchor",
                selection_reason="canonical_anchor",
                coverage_gain=0.0,
                information_gain=0.0,
            ),
        )
    if any(camera_id not in candidate_by_id for camera_id in choices_by_id):
        raise ValueError("existing selected camera is absent from the candidate inventory")

    choices = list(choices_by_id.values())
    if len(choices) > max_views:
        raise ValueError("existing selected views exceed max_views")
    _ensure_tile_capacity(
        len(choices),
        atlas_max_dimension=atlas_max_dimension,
        min_tile_dimension=min_tile_dimension,
    )
    preferred = {
        orientation.lower() for orientation in preferred_orientations if orientation != "any"
    }
    target_minimum = max(min_views, len(choices) + required_additional_views)
    added_count = 0

    while True:
        coverage = _surface_coverage(choices, diagnostic_by_id, sample_count)
        if (
            len(choices) >= target_minimum
            and added_count >= required_additional_views
            and coverage >= coverage_target
        ):
            return _selection(
                choices,
                coverage,
                "coverage_target",
                atlas_max_dimension,
            )
        if len(choices) >= max_views:
            return _selection(choices, coverage, "max_views", atlas_max_dimension)
        try:
            _ensure_tile_capacity(
                len(choices) + 1,
                atlas_max_dimension=atlas_max_dimension,
                min_tile_dimension=min_tile_dimension,
            )
        except ValueError:
            return _selection(choices, coverage, "atlas_resolution", atlas_max_dimension)

        chosen_ids = {view.camera_id for view in choices}
        ranked = [
            _candidate_gain(
                candidate,
                diagnostic_by_id[candidate.camera_id],
                choices,
                diagnostic_by_id,
                sample_count,
                preferred,
            )
            for candidate in candidates
            if candidate.camera_id not in chosen_ids
        ]
        if not ranked:
            return _selection(choices, coverage, "candidate_exhausted", atlas_max_dimension)
        information_gain, coverage_gain, candidate, preference_match = max(
            ranked,
            key=lambda item: (
                item[0],
                item[1],
                diagnostic_by_id[item[2].camera_id].frame_coverage,
                item[2].camera_id,
            ),
        )
        must_meet_floor = len(choices) < target_minimum or added_count < required_additional_views
        if information_gain < min_view_gain and not must_meet_floor:
            return _selection(choices, coverage, "min_view_gain", atlas_max_dimension)

        reason = "new_surface_coverage"
        if coverage_gain == 0.0:
            reason = "silhouette_novelty"
        if must_meet_floor and information_gain < min_view_gain:
            reason = "minimum_view_floor"
        if existing_views and added_count < required_additional_views:
            reason = "review_augmentation"
        if preference_match:
            reason = f"{reason}:preferred_orientation"
        choices.append(
            SelectedCameraChoice(
                **candidate.model_dump(),
                selection_kind="canonical_anchor" if candidate.canonical_anchor else "dynamic",
                selection_reason=reason,
                coverage_gain=round(coverage_gain, 6),
                information_gain=round(min(information_gain, 1.0), 6),
            )
        )
        added_count += 1


def _selection(
    views: Sequence[SelectedCameraChoice],
    coverage: float,
    stop_reason: InspectionStopReason,
    atlas_max_dimension: int,
) -> CameraSelection:
    return CameraSelection(
        views=tuple(views),
        estimated_surface_coverage=round(coverage, 6),
        stop_reason=stop_reason,
        layout=plan_atlas_layout(len(views), atlas_max_dimension),
    )


def _ensure_tile_capacity(
    view_count: int, *, atlas_max_dimension: int, min_tile_dimension: int
) -> None:
    if plan_atlas_layout(view_count, atlas_max_dimension).tile_dimension < min_tile_dimension:
        raise ValueError("atlas tiles would fall below min_tile_dimension")


def _candidate_gain(
    candidate: CameraCandidate,
    diagnostic: CandidateDiagnostic,
    selected: Sequence[SelectedCameraChoice],
    diagnostic_by_id: dict[str, CandidateDiagnostic],
    sample_count: int,
    preferred_orientations: set[str],
) -> tuple[float, float, CameraCandidate, bool]:
    covered_samples = {
        sample
        for view in selected
        for sample in diagnostic_by_id[view.camera_id].surface_sample_ids
    }
    new_samples = set(diagnostic.surface_sample_ids) - covered_samples
    coverage_gain = len(new_samples) / sample_count if sample_count else 0.0
    silhouette_novelty = _silhouette_novelty(diagnostic, selected, diagnostic_by_id)
    preference_match = bool(
        preferred_orientations
        and any(
            token in candidate.orientation.lower().split("-") for token in preferred_orientations
        )
    )
    information_gain = (
        0.70 * coverage_gain + 0.25 * silhouette_novelty + 0.05 * diagnostic.frame_coverage
    )
    if preference_match:
        information_gain *= 1.15
    return information_gain, coverage_gain, candidate, preference_match


def _surface_coverage(
    selected: Sequence[SelectedCameraChoice],
    diagnostic_by_id: dict[str, CandidateDiagnostic],
    sample_count: int,
) -> float:
    if sample_count == 0:
        return 0.0
    covered = {
        sample
        for view in selected
        for sample in diagnostic_by_id[view.camera_id].surface_sample_ids
    }
    return len(covered) / sample_count


def _silhouette_novelty(
    candidate: CandidateDiagnostic,
    selected: Sequence[SelectedCameraChoice],
    diagnostic_by_id: dict[str, CandidateDiagnostic],
) -> float:
    if not selected:
        return 1.0
    candidate_signature = int(candidate.silhouette_signature, 16)
    signature_width = max(1, len(candidate.silhouette_signature) * 4)
    distances = [
        (
            candidate_signature ^ int(diagnostic_by_id[view.camera_id].silhouette_signature, 16)
        ).bit_count()
        / signature_width
        for view in selected
    ]
    return min(distances)


def _orientation_label(azimuth: float, elevation: float) -> str:
    directions = (
        "front",
        "front-right",
        "right",
        "rear-right",
        "rear",
        "rear-left",
        "left",
        "front-left",
    )
    index = int(math.floor(((azimuth + 22.5) % 360.0) / 45.0))
    horizontal = directions[index]
    if elevation >= 20.0:
        return f"{horizontal}-upper"
    if elevation <= -20.0:
        return f"{horizontal}-lower"
    return horizontal
