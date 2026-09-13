"""Persist self-contained runs, canonical scenes, checkpoints, and artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .state import DurableWorkItem, RunState

_WORK_ITEM_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class RunDirectory:
    """A numbered, self-contained directory for a single autonomous run."""

    id: int
    path: Path


class RunStateError(RuntimeError):
    """A run's recovery metadata is absent, malformed, or inconsistent."""


class CheckpointStore:
    """Persist inspectable run state without a global checkpoint namespace.

    ``scene.blend`` is the mutable canonical working scene for a run. Immutable
    checkpoint copies live only beneath the same run's ``checkpoints/`` directory.
    """

    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = (project_dir or Path.cwd()).resolve()
        self.root = self.project_dir / ".aculptoi"
        self.runs = self.root / "runs"

    def create_run(self, goal: str | None = None) -> RunDirectory:
        """Allocate a numbered run and atomically create its typed initial state."""
        self.runs.mkdir(parents=True, exist_ok=True)
        ids = [
            int(path.name) for path in self.runs.iterdir() if path.is_dir() and path.name.isdigit()
        ]
        run_id = max(ids, default=0) + 1
        path = self.runs / f"{run_id:06d}"
        path.mkdir()
        run = RunDirectory(id=run_id, path=path)
        self.checkpoints_directory(run).mkdir()
        self.save_run_state(run, RunState(goal=goal))
        return run

    def get_run(self, run_id: int) -> RunDirectory:
        """Resolve an existing numeric run without creating anything."""
        if run_id < 1:
            raise ValueError("run id must be positive")
        path = self.runs / f"{run_id:06d}"
        if not path.is_dir():
            raise RunStateError(f"run {run_id:06d} does not exist")
        return RunDirectory(id=run_id, path=path)

    def canonical_scene_path(self, run: RunDirectory) -> Path:
        """Return the sole mutable canonical Blender path for ``run``."""
        return run.path / "scene.blend"

    def checkpoints_directory(self, run: RunDirectory) -> Path:
        """Return the run-local immutable checkpoint directory."""
        return run.path / "checkpoints"

    def run_state_path(self, run: RunDirectory) -> Path:
        """Return the typed recovery-state file path for ``run``."""
        return run.path / "run-state.json"

    def initial_scene_path(self, run: RunDirectory) -> Path:
        """Return the immutable initial-scene fallback used before any item completes."""
        return run.path / "initial-scene.blend"

    def preserve_initial_scene(self, run: RunDirectory) -> Path:
        """Copy the initialized canonical scene once, without calling it a checkpoint."""
        source = self.canonical_scene_path(run)
        target = self.initial_scene_path(run)
        if not source.is_file():
            raise RunStateError("canonical scene.blend is missing; cannot preserve initial scene")
        if not target.exists():
            self._atomic_copy(source, target)
        return target

    def load_run_state(self, run: RunDirectory) -> RunState:
        """Load typed state or fail safely rather than guessing recovery inputs."""
        path = self.run_state_path(run)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RunState.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise RunStateError(f"run {run.id:06d} has invalid run-state.json: {error}") from error

    def save_run_state(self, run: RunDirectory, state: RunState) -> Path:
        """Atomically persist typed recovery state after a successful transition."""
        if state.scene != "scene.blend" or state.initial_scene != "initial-scene.blend":
            raise RunStateError("run state must use the fixed canonical and initial scene paths")
        self._atomic_json(
            self.run_state_path(run), state.with_updated_timestamp().model_dump(mode="json")
        )
        return self.run_state_path(run)

    def latest_resumable_run(self) -> RunDirectory | None:
        """Return the newest actively interrupted/recoverable run, if any."""
        if not self.runs.exists():
            return None
        for path in sorted(self.runs.iterdir(), reverse=True):
            if not path.is_dir() or not path.name.isdigit():
                continue
            run = RunDirectory(id=int(path.name), path=path)
            try:
                state = self.load_run_state(run)
            except RunStateError:
                continue
            if state.status in {"created", "running", "recovering", "interrupted"}:
                return run
        return None

    def save_metadata(self, run: RunDirectory, name: str, data: dict[str, Any]) -> Path:
        """Write human-readable, deterministic JSON within the numbered run."""
        target = run.path / name
        if target.suffix != ".json" or name != Path(name).name:
            raise ValueError("metadata filename must be a simple .json filename")
        payload = {"timestamp": datetime.now(UTC).isoformat(), **data}
        self._atomic_json(target, payload)
        return target

    def iteration_directory(self, run: RunDirectory, iteration: int) -> Path:
        """Create and return the validated directory for one positive iteration number."""
        if iteration < 1:
            raise ValueError("iteration must be positive")
        target = run.path / f"iteration-{iteration:03d}"
        target.mkdir(exist_ok=True)
        return target

    def work_item_directory(
        self, run: RunDirectory, iteration: int, ordinal: int, work_item_id: str
    ) -> Path:
        """Create the artifact directory for one ordered construction-plan item."""
        if ordinal < 1:
            raise ValueError("work-item ordinal must be positive")
        if not _WORK_ITEM_ID_RE.fullmatch(work_item_id):
            raise ValueError("work-item id must be lowercase kebab-case")
        target = (
            self.iteration_directory(run, iteration) / "items" / f"{ordinal:03d}-{work_item_id}"
        )
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_text_artifact(self, run: RunDirectory, relative_path: str, content: str) -> Path:
        """Write an inspectable text artifact without permitting traversal outside a run."""
        target = self._artifact_target(run, relative_path, ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def save_json_artifact(
        self, run: RunDirectory, relative_path: str, data: object, *, overwrite: bool = True
    ) -> Path:
        """Write a JSON artifact within a run without adding or removing payload fields."""
        target = self._artifact_target(run, relative_path, ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and target.exists():
            raise FileExistsError(target)
        if overwrite:
            self._atomic_json(target, data)
        else:
            with target.open("x", encoding="utf-8") as file:
                file.write(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
        return target

    def create_checkpoint(
        self, run: RunDirectory, *, iteration: int, ordinal: int, work_item_id: str
    ) -> DurableWorkItem:
        """Atomically snapshot saved canonical state at a completed-item boundary."""
        if ordinal < 1 or iteration < 1 or not _WORK_ITEM_ID_RE.fullmatch(work_item_id):
            raise ValueError("checkpoint requires a valid iteration, ordinal, and work-item id")
        canonical = self.canonical_scene_path(run)
        if not canonical.is_file():
            raise RunStateError("canonical scene.blend is missing; cannot create a checkpoint")
        filename = f"item-{iteration:03d}-{ordinal:03d}-{work_item_id}.blend"
        target = self.checkpoints_directory(run) / filename
        if target.exists():
            raise RunStateError(
                f"immutable checkpoint already exists: {target.relative_to(run.path)}"
            )
        self._atomic_copy(canonical, target)
        return DurableWorkItem(
            iteration=iteration,
            ordinal=ordinal,
            work_item_id=work_item_id,
            checkpoint=str(target.relative_to(run.path)),
        )

    def resolve_checkpoint(self, run: RunDirectory, relative_path: str) -> Path:
        """Resolve a state-referenced run-local checkpoint without path guessing."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".blend":
            raise RunStateError("checkpoint metadata contains an invalid relative path")
        path = (run.path / relative).resolve()
        try:
            path.relative_to(self.checkpoints_directory(run).resolve())
        except ValueError as error:
            raise RunStateError(
                "checkpoint metadata escapes this run's checkpoints directory"
            ) from error
        if not path.is_file():
            raise RunStateError(f"checkpoint referenced by run state is missing: {relative}")
        return path

    def restore_latest_checkpoint(
        self, run: RunDirectory, *, preserve_partial: bool = True
    ) -> Path:
        """Restore the latest immutable checkpoint over the canonical working scene."""
        state = self.load_run_state(run)
        if state.latest_checkpoint is None:
            raise RunStateError("run has no completed-item checkpoint to restore")
        checkpoint = self.resolve_checkpoint(run, state.latest_checkpoint)
        canonical = self.canonical_scene_path(run)
        if preserve_partial and canonical.is_file():
            recovery = run.path / "recovery"
            recovery.mkdir(exist_ok=True)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            self._atomic_copy(canonical, recovery / f"abandoned-{timestamp}.blend")
        self._atomic_copy(checkpoint, canonical)
        return canonical

    def restore_recovery_base(self, run: RunDirectory, *, preserve_partial: bool = True) -> Path:
        """Restore the latest checkpoint, or the immutable initial scene if none exists."""
        state = self.load_run_state(run)
        if state.latest_checkpoint is not None:
            return self.restore_latest_checkpoint(run, preserve_partial=preserve_partial)
        initial = self.initial_scene_path(run)
        if not initial.is_file():
            raise RunStateError("run has no checkpoint and its immutable initial scene is missing")
        canonical = self.canonical_scene_path(run)
        if preserve_partial and canonical.is_file():
            recovery = run.path / "recovery"
            recovery.mkdir(exist_ok=True)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            self._atomic_copy(canonical, recovery / f"abandoned-{timestamp}.blend")
        self._atomic_copy(initial, canonical)
        return canonical

    def list_checkpoints(self, run: RunDirectory) -> list[dict[str, str]]:
        """List immutable checkpoints for one run without opening Blender files."""
        directory = self.checkpoints_directory(run)
        if not directory.exists():
            return []
        return [
            {
                "name": path.stem,
                "path": str(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }
            for path in sorted(directory.glob("item-*.blend"))
        ]

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        """Copy to a sibling temporary path then atomically replace the target."""
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_json(target: Path, data: object) -> None:
        """Atomically write readable JSON so a crash cannot half-write state."""
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _artifact_target(run: RunDirectory, relative_path: str, suffix: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != suffix:
            raise ValueError(f"artifact path must be a relative {suffix} path inside the run")
        target = (run.path / relative).resolve()
        try:
            target.relative_to(run.path.resolve())
        except ValueError as error:
            raise ValueError("artifact path escapes the run directory") from error
        return target
