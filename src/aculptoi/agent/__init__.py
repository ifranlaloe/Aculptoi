"""Explicit actor → Blender → vision orchestration."""

from typing import Any

from .actor import Actor
from .critic import VisionCritic

__all__ = ["Actor", "RefinementLoop", "VisionCritic"]


def __getattr__(name: str) -> Any:
    """Defer loop loading so prompt-only roles do not create an import cycle."""
    if name == "RefinementLoop":
        from .loop import RefinementLoop

        return RefinementLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
