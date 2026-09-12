"""Persist run artifacts and checkpoint metadata in the project directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunDirectory:
    """A numbered inspection directory for a single autonomous run."""

    id: int
    path: Path


class CheckpointStore:
    """Persist inspected artifacts and copy validated worker snapshots into runs."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = (project_dir or Path.cwd()).resolve()
        self.root = self.project_dir / ".aculptoi"
        self.runs = self.root / "runs"
        self.checkpoints = self.root / "checkpoints"

    def create_run(self) -> RunDirectory:
        """Allocate a monotonically numbered, inspectable run directory."""
        self.runs.mkdir(parents=True, exist_ok=True)
        ids = [
            int(path.name) for path in self.runs.iterdir() if path.is_dir() and path.name.isdigit()
        ]
        run_id = max(ids, default=0) + 1
        path = self.runs / f"{run_id:06d}"
        path.mkdir()
        return RunDirectory(id=run_id, path=path)

    def save_metadata(self, run: RunDirectory, name: str, data: dict[str, Any]) -> Path:
        """Write human-readable, deterministic JSON within the numbered run."""
        target = run.path / name
        if target.suffix != ".json" or name != Path(name).name:
            raise ValueError("metadata filename must be a simple .json filename")
        payload = {"timestamp": datetime.now(UTC).isoformat(), **data}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def iteration_directory(self, run: RunDirectory, iteration: int) -> Path:
        """Create and return the validated directory for one positive iteration number."""
        if iteration < 1:
            raise ValueError("iteration must be positive")
        target = run.path / f"iteration-{iteration:03d}"
        target.mkdir(exist_ok=True)
        return target

    def batch_directory(self, run: RunDirectory, iteration: int, batch: int) -> Path:
        """Create and return the validated artifact directory for one execution batch."""
        if batch < 1:
            raise ValueError("batch must be positive")
        target = self.iteration_directory(run, iteration) / f"batch-{batch:03d}"
        target.mkdir(exist_ok=True)
        return target

    def save_text_artifact(self, run: RunDirectory, relative_path: str, content: str) -> Path:
        """Write an inspectable text artifact without permitting traversal outside a run."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".txt":
            raise ValueError("text artifact path must be a relative .txt path inside the run")
        target = (run.path / relative).resolve()
        try:
            target.relative_to(run.path.resolve())
        except ValueError as error:
            raise ValueError("text artifact path escapes the run directory") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def save_json_artifact(self, run: RunDirectory, relative_path: str, data: object) -> Path:
        """Write a JSON artifact within a run without adding or removing payload fields."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
            raise ValueError("JSON artifact path must be a relative .json path inside the run")
        target = (run.path / relative).resolve()
        try:
            target.relative_to(run.path.resolve())
        except ValueError as error:
            raise ValueError("JSON artifact path escapes the run directory") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return target

    def copy_checkpoint_to_iteration(
        self, run: RunDirectory, iteration: int, snapshot: dict[str, object]
    ) -> Path:
        """Copy a worker-created checkpoint into the corresponding visual iteration."""
        self.iteration_directory(run, iteration)
        return self._copy_checkpoint(run, f"iteration-{iteration:03d}/scene.blend", snapshot)

    def copy_checkpoint_to_batch(
        self, run: RunDirectory, iteration: int, batch: int, snapshot: dict[str, object]
    ) -> Path:
        """Copy a worker-created checkpoint into the corresponding execution batch."""
        self.batch_directory(run, iteration, batch)
        return self._copy_checkpoint(
            run, f"iteration-{iteration:03d}/batch-{batch:03d}/scene.blend", snapshot
        )

    def _copy_checkpoint(
        self, run: RunDirectory, relative_path: str, snapshot: dict[str, object]
    ) -> Path:
        """Copy a validated worker snapshot to a fixed, run-local `.blend` artifact.

        The source must resolve below the worker checkpoint directory, so a model
        response cannot turn checkpoint metadata into arbitrary file access.
        """
        source_value = snapshot.get("path")
        if not isinstance(source_value, str):
            raise ValueError("checkpoint metadata did not contain a snapshot path")
        source = Path(source_value).resolve()
        try:
            source.relative_to(self.checkpoints.resolve())
        except ValueError as error:
            raise ValueError(
                "checkpoint source is outside the worker checkpoint directory"
            ) from error
        if source.suffix != ".blend" or not source.is_file():
            raise ValueError("checkpoint source must be an existing .blend file")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".blend":
            raise ValueError("checkpoint target must be a relative .blend path inside the run")
        target = (run.path / relative).resolve()
        try:
            target.relative_to(run.path.resolve())
        except ValueError as error:
            raise ValueError("checkpoint target escapes the run directory") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def list_checkpoints(self) -> list[dict[str, str]]:
        """List worker-created .blend snapshot files without opening them."""
        if not self.checkpoints.exists():
            return []
        return [
            {
                "name": path.stem,
                "path": str(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }
            for path in sorted(self.checkpoints.glob("*.blend"))
        ]
