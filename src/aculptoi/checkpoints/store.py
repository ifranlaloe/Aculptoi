"""Persist run artifacts and checkpoint metadata in the project directory."""

from __future__ import annotations

import json
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
    """Small filesystem adapter; blend snapshots remain the worker's responsibility."""

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
