"""Pillow-based durable composition of labeled inspection atlases."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

from aculptoi.schemas.inspection import (
    InspectionAtlasManifest,
    InspectionAtlasTile,
    InspectionCameraPlan,
    InspectionRenderResult,
)


def compose_atlas(
    plan: InspectionCameraPlan,
    rendered: InspectionRenderResult,
    output_path: Path,
) -> InspectionAtlasManifest:
    """Compose a deterministic labeled atlas from final worker-provided source shots."""
    shots_by_camera = {shot.camera_id: shot for shot in rendered.shots}
    if set(shots_by_camera) != {view.camera_id for view in plan.views}:
        raise ValueError("inspection render shots do not match the camera plan")

    layout = plan.layout
    atlas = Image.new("RGB", (layout.width, layout.height), color=(96, 96, 96))
    draw = ImageDraw.Draw(atlas)
    tiles: dict[str, InspectionAtlasTile] = {}
    for index, view in enumerate(plan.views):
        row, column = divmod(index, layout.columns)
        x = column * layout.tile_dimension
        y = row * layout.tile_dimension
        shot = shots_by_camera[view.camera_id]
        source = Path(shot.path)
        try:
            with Image.open(source) as image:
                image.load()
                tile = image.convert("RGB").resize(
                    (layout.tile_dimension, layout.tile_dimension),
                    Image.Resampling.LANCZOS,
                )
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"could not compose inspection shot {source}") from error
        atlas.paste(tile, (x, y))
        draw.rectangle(
            (
                x,
                y,
                min(x + layout.tile_dimension, x + 238),
                min(y + layout.tile_dimension, y + 49),
            ),
            fill=(18, 18, 18),
        )
        label = (
            f"{view.tile_id}  {view.orientation}\n"
            f"az {view.azimuth_degrees:+.0f} deg  el {view.elevation_degrees:+.0f} deg"
        )
        draw.multiline_text((x + 8, y + 6), label, fill=(245, 245, 245), spacing=2)
        tiles[view.tile_id] = InspectionAtlasTile(
            tile_id=view.tile_id,
            camera_id=view.camera_id,
            source=f"shots/{view.tile_id}.png",
            pixel_bounds=(x, y, layout.tile_dimension, layout.tile_dimension),
            azimuth_degrees=view.azimuth_degrees,
            elevation_degrees=view.elevation_degrees,
            orientation=view.orientation,
            projection=view.projection,
            selection_kind=view.selection_kind,
            selection_reason=view.selection_reason,
            coverage_gain=view.coverage_gain,
            information_gain=view.information_gain,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path, format="PNG", optimize=True)
    return InspectionAtlasManifest(
        sensor_version=plan.sensor_version,
        lighting_rig=plan.lighting_rig,
        width=layout.width,
        height=layout.height,
        layout=layout,
        bounds=rendered.bounds,
        framing=rendered.framing,
        estimated_surface_coverage=plan.estimated_surface_coverage,
        tiles=tiles,
    )
