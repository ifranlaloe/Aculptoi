"""TOML configuration for independent actor, vision, and Blender providers."""

from __future__ import annotations

import os
import platform
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _default_blender_executable() -> str:
    """Return a useful platform default without requiring Blender on PATH."""
    if platform.system() == "Darwin":
        return "/Applications/Blender.app/Contents/MacOS/Blender"
    return "blender"


class ModelConfig(BaseModel):
    """OpenAI-compatible local model endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value


class BlenderConfig(BaseModel):
    """Local-only persistent Blender worker configuration."""

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(default_factory=_default_blender_executable)
    host: str = "127.0.0.1"
    port: int = Field(default=9876, ge=1024, le=65535)
    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)

    @field_validator("host")
    @classmethod
    def local_host_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Blender worker must bind to localhost only")
        return value


class AcuConfig(BaseModel):
    """Top-level Aculptoi runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    actor: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            base_url="http://localhost:8080/v1", model="local-actor"
        )
    )
    vision: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            base_url="http://localhost:8081/v1", model="local-vision"
        )
    )
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    max_iterations: int = Field(default=5, ge=1, le=100)
    score_target: float = Field(default=0.9, ge=0.0, le=1.0)


def default_config_path(project_dir: Path | None = None) -> Path:
    """Use an explicit environment override or the project-local TOML file."""
    override = os.environ.get("ACULPTOI_CONFIG")
    if override:
        return Path(override).expanduser()
    return (project_dir or Path.cwd()) / "aculptoi.toml"


def load_config(path: Path | None = None) -> AcuConfig:
    """Load TOML settings, or return safe local-first defaults when absent."""
    config_path = path or default_config_path()
    if not config_path.exists():
        return AcuConfig()
    with config_path.open("rb") as file:
        data = tomllib.load(file)
    return AcuConfig.model_validate(data)
