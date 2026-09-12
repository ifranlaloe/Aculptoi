"""A small, inspectable launcher specification for a local llama.cpp server."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class LlamaServeConfig(BaseModel):
    """Fixed local-only llama.cpp options for Aculptoi's multimodal topology."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    mmproj_path: Path
    context_size: int = Field(default=32_768, ge=512, le=131_072)
    port: int = Field(default=8080, ge=1024, le=65535)
    alias: str = Field(
        default="aculptoi",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )

    @property
    def host(self) -> str:
        """Keep the optional launcher local-only; remote binding is intentionally absent."""
        return "127.0.0.1"

    def command(self) -> list[str]:
        """Build an argument-vector equivalent to the documented ``llama serve`` command."""
        return [
            "llama",
            "serve",
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-c",
            str(self.context_size),
            "-np",
            "1",
            "-fa",
            "on",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-a",
            self.alias,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
