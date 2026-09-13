"""TOML configuration for local model providers, roles, and Blender."""

from __future__ import annotations

import os
import platform
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aculptoi.reasoning import ReasoningEffort


def _default_blender_executable() -> str:
    """Return a useful platform default without requiring Blender on PATH."""
    if platform.system() == "Darwin":
        return "/Applications/Blender.app/Contents/MacOS/Blender"
    return "blender"


DEFAULT_PROVIDER_NAME = "local"


class ProviderConfig(BaseModel):
    """One OpenAI-compatible model endpoint available to one or more roles."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    timeout_seconds: float = Field(default=900.0, gt=0, le=3600)
    reasoning_effort_transport: Literal["chat_template_kwargs", "top_level", "omit"] = (
        "chat_template_kwargs"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value


# Kept as a public alias for integrations using the original V1 name.
ModelConfig = ProviderConfig


class RoleConfig(BaseModel):
    """Select the named provider used by an application role."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default=DEFAULT_PROVIDER_NAME,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )


class ActorRoleConfig(RoleConfig):
    """Actor-specific generation limit for planning with bounded reasoning."""

    max_output_tokens: int = Field(default=16_384, ge=128, le=65_536)
    reasoning_effort: ReasoningEffort = "medium"


class VisionRoleConfig(RoleConfig):
    """Vision-role settings kept separate from its provider selection."""

    max_image_dimension: int = Field(default=1280, ge=128, le=4096)
    max_output_tokens: int = Field(default=16_384, ge=128, le=65_536)
    reasoning_effort: ReasoningEffort = "medium"
    max_discovered_issues: int = Field(default=12, ge=1, le=50)
    max_issue_analysis_requests: int = Field(default=12, ge=0, le=50)


class BlenderConfig(BaseModel):
    """Local-only persistent Blender worker configuration."""

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(default_factory=_default_blender_executable)
    host: str = "127.0.0.1"
    port: int = Field(default=9876, ge=1024, le=65535)
    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    mode: Literal["ui", "headless"] = "ui"

    @field_validator("host")
    @classmethod
    def local_host_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Blender worker must bind to localhost only")
        return value


class AcuConfig(BaseModel):
    """Top-level Aculptoi runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            DEFAULT_PROVIDER_NAME: ProviderConfig(
                base_url="http://127.0.0.1:8080/v1", model="local-multimodal"
            )
        }
    )
    actor: ActorRoleConfig = Field(default_factory=ActorRoleConfig)
    vision: VisionRoleConfig = Field(default_factory=VisionRoleConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    max_iterations: int = Field(default=5, ge=1, le=100)
    max_actor_requests_per_iteration: int = Field(default=100, ge=2, le=2_500)
    max_actions_per_iteration: int = Field(default=1_000, ge=1, le=25_000)
    iteration_timeout_seconds: float = Field(default=3_600.0, gt=0, le=86_400)
    score_target: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_execution_budget(cls, value: Any) -> Any:
        """Treat the old batch cap as an equivalent action-request safety budget."""
        if not isinstance(value, dict) or "max_execution_batches_per_iteration" not in value:
            return value
        data = dict(value)
        legacy = data.pop("max_execution_batches_per_iteration")
        if "max_actor_requests_per_iteration" in data:
            raise ValueError(
                "configure max_actor_requests_per_iteration instead of combining it with "
                "max_execution_batches_per_iteration"
            )
        data["max_actor_requests_per_iteration"] = (
            legacy + 1 if isinstance(legacy, int) and not isinstance(legacy, bool) else legacy
        )
        return data

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_role_endpoints(cls, value: Any) -> Any:
        """Accept V1 role-local endpoint TOML while favoring named providers.

        V1 used ``[actor]`` and ``[vision]`` sections with ``base_url`` and
        ``model`` fields. Loading those files remains safe and deterministic;
        each legacy section becomes a uniquely named provider.
        """
        if not isinstance(value, dict):
            return value

        data = dict(value)
        has_providers = "providers" in data
        raw_providers = data.get("providers", {})
        providers = dict(raw_providers) if isinstance(raw_providers, dict) else raw_providers
        if not isinstance(providers, dict):
            return data

        migrated = False
        for role in ("actor", "vision"):
            role_value = data.get(role)
            if not isinstance(role_value, dict) or "provider" in role_value:
                continue
            if not {"base_url", "model"}.issubset(role_value):
                continue
            provider_name = f"{role}-legacy"
            providers[provider_name] = {
                key: role_value[key]
                for key in ("base_url", "model", "timeout_seconds")
                if key in role_value
            }
            role_data: dict[str, Any] = {"provider": provider_name}
            for key in (
                "max_output_tokens",
                "reasoning_effort",
                "max_image_dimension",
                "max_discovered_issues",
                "max_issue_analysis_requests",
            ):
                if key in role_value:
                    role_data[key] = role_value[key]
            data[role] = role_data
            migrated = True

        if has_providers or migrated:
            data["providers"] = providers
        return data

    @model_validator(mode="after")
    def validate_role_providers(self) -> AcuConfig:
        """Reject role references to providers which have not been configured."""
        for role, role_config in (("actor", self.actor), ("vision", self.vision)):
            if role_config.provider not in self.providers:
                raise ValueError(
                    f"{role}.provider references unknown provider '{role_config.provider}'"
                )
        return self

    def provider_for(self, role: Literal["actor", "vision"]) -> ProviderConfig:
        """Return the validated endpoint configuration selected for one role."""
        role_config = self.actor if role == "actor" else self.vision
        return self.providers[role_config.provider]


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
