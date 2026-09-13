"""Bound multimodal request image sizes without changing stored render artifacts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class PreparedRender:
    """An in-memory PNG suitable for an OpenAI-compatible image data URL."""

    source: Path
    width: int
    height: int
    data_url: str


def prepare_render(path: Path, max_dimension: int | None) -> PreparedRender:
    """Load and optionally downsize a PNG render without modifying the source file."""
    if max_dimension is not None and max_dimension < 1:
        raise ValueError("max_image_dimension must be positive")
    if path.suffix.lower() != ".png":
        raise ValueError(f"Inspection render must be a PNG: {path}")

    try:
        with Image.open(path) as source:
            source.load()
            prepared = source.convert("RGBA") if "A" in source.getbands() else source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read inspection render: {path}") from error

    if max_dimension is not None:
        prepared.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    prepared.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return PreparedRender(
        source=path,
        width=prepared.width,
        height=prepared.height,
        data_url=f"data:image/png;base64,{encoded}",
    )
