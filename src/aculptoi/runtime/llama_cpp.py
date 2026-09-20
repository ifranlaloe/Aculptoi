"""A small, inspectable launcher specification for a local llama.cpp server."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DAVIDAU_REPOSITORY_DIRECTORY = (
    "models--DavidAU--Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF"
)
DAVIDAU_MODEL_FILENAME = "Qwen3.8-27B-TTURBO-Fable-C-Fusion-709-L-Uncen-NM-DAU-NEO-MTP-Q4_K_M.gguf"
DAVIDAU_MMPROJ_FILENAME = "mmproj-BF16.gguf"


def default_davidau_artifacts(hf_home: Path) -> tuple[Path, Path]:
    """Find the selected DavidAU GGUF and projection in a Hugging Face cache.

    Snapshot revisions are intentionally discovered rather than hard-coded so a
    cache refresh can retain the same default artifact names under a new revision.
    """
    snapshots = hf_home / "hub" / DAVIDAU_REPOSITORY_DIRECTORY / "snapshots"
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            model_path = snapshot / DAVIDAU_MODEL_FILENAME
            mmproj_path = snapshot / DAVIDAU_MMPROJ_FILENAME
            if snapshot.is_dir() and model_path.is_file() and mmproj_path.is_file():
                return model_path, mmproj_path
    raise FileNotFoundError(
        "Could not find the default DavidAU TWIN-TURBO NEO MTP Q4_K_M GGUF and "
        "mmproj-BF16.gguf under "
        f"{snapshots}. Download them there or pass both --model and --mmproj."
    )


class LlamaServeConfig(BaseModel):
    """Fixed local-only llama.cpp options for Aculptoi's multimodal topology."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    mmproj_path: Path
    context_size: int = Field(default=65_536, ge=512, le=131_072)
    mtp_enabled: bool = False
    mtp_draft_tokens: int = Field(default=2, ge=1, le=4)
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
        command = [
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
        ]
        if self.mtp_enabled:
            command.extend(
                [
                    "--spec-type",
                    "draft-mtp",
                    "--spec-draft-n-max",
                    str(self.mtp_draft_tokens),
                ]
            )
        command.extend(
            [
                "-a",
                self.alias,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
        )
        return command
