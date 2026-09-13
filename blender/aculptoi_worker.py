"""Persistent localhost-only Aculptoi worker, executed by Blender's Python.

This module intentionally uses only Blender and Python standard-library APIs. It
does not receive or execute model-generated Python, shell commands, or paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import tempfile
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]{0,63}$")
RUN_ID_RE = re.compile(r"^[0-9]{6}$")
CAMERA_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TILE_ID_RE = re.compile(r"^[A-Z]+[1-9][0-9]*$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
ALLOWED_VIEWS = {"front", "right", "top", "perspective"}
INSPECTION_RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}
MAX_REQUEST_BYTES = 1_000_000


class WorkerError(ValueError):
    """A request did not meet the worker's independent safety requirements."""


class MainThreadDispatcher:
    """Marshal UI-worker requests onto Blender's required main thread."""

    def __init__(self) -> None:
        self._requests: queue.Queue[
            tuple[Callable[[], dict[str, Any]], threading.Event, dict[str, Any]]
        ] = queue.Queue()
        self.stopping = False

    def call(self, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        complete = threading.Event()
        result: dict[str, Any] = {}
        self._requests.put((operation, complete, result))
        if not complete.wait(timeout=130):
            raise WorkerError("worker operation timed out waiting for Blender's main thread")
        error = result.get("error")
        if isinstance(error, BaseException):
            raise error
        value = result.get("value")
        if not isinstance(value, dict):
            raise WorkerError("worker operation returned invalid data")
        return value

    def pump(self) -> float | None:
        """Run a bounded amount of work, then return Blender to its UI event loop."""
        for _ in range(4):
            try:
                operation, complete, result = self._requests.get_nowait()
            except queue.Empty:
                break
            try:
                result["value"] = operation()
            except BaseException as error:  # Report the normal worker error to the HTTP caller.
                result["error"] = error
            finally:
                complete.set()
        return None if self.stopping else 0.05


def _name(value: object) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise WorkerError("object and checkpoint names must be 1-64 safe identifier characters")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise WorkerError(f"{label} must be a finite number")
    return float(value)


def _vector(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise WorkerError(f"{label} must contain exactly three numbers")
    return tuple(_number(item, label) for item in value)  # type: ignore[return-value]


def _object(name: object) -> Any:
    return bpy.data.objects.get(_name(name))


def _require_object(name: object) -> Any:
    obj = _object(name)
    if obj is None:
        raise WorkerError(f"object not found: {name}")
    return obj


def _object_data(obj: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": obj.name,
        "type": obj.type,
        "location": [round(value, 6) for value in obj.location],
        "rotation_degrees": [round(math.degrees(value), 6) for value in obj.rotation_euler],
        "scale": [round(value, 6) for value in obj.scale],
        "dimensions": [round(value, 6) for value in obj.dimensions],
        "modifiers": [{"name": modifier.name, "type": modifier.type} for modifier in obj.modifiers],
    }
    if obj.type == "MESH":
        mesh = obj.data
        data["mesh"] = {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "polygons": len(mesh.polygons),
            "manifold": None,
            "validation": "not implemented in V1",
        }
    return data


class AculptoiWorker:
    """Operations available through a narrow, versioned local HTTP protocol."""

    def __init__(self, mode: str = "ui") -> None:
        self.project_root = Path.cwd().resolve()
        self.artifact_root = (self.project_root / ".aculptoi").resolve()
        self.mode = mode
        self.active_run_id: int | None = None
        self.active_scene_path: Path | None = None
        self._lock_path: Path | None = None
        self._lock_token: str | None = None

    def health(self) -> dict[str, Any]:
        return {
            "blender_version": bpy.app.version_string,
            "project_root": str(self.project_root),
            "mode": self.mode,
            "active_run_id": self.active_run_id,
            "canonical_scene": str(self.active_scene_path) if self.active_scene_path else None,
        }

    def _safe_scene_path(self, requested: object, run_id: object) -> tuple[int, Path]:
        """Accept only this project's numbered-run canonical ``scene.blend`` paths."""
        if (
            not isinstance(requested, str)
            or not isinstance(run_id, int)
            or isinstance(run_id, bool)
        ):
            raise WorkerError("run_id and scene_path are required")
        if run_id < 1:
            raise WorkerError("run_id must be positive")
        path = Path(requested).expanduser().resolve()
        expected = (self.artifact_root / "runs" / f"{run_id:06d}" / "scene.blend").resolve()
        if (
            path != expected
            or path.name != "scene.blend"
            or not RUN_ID_RE.fullmatch(path.parent.name)
        ):
            raise WorkerError(
                "scene_path must be this run's canonical .aculptoi/runs/<id>/scene.blend"
            )
        try:
            path.relative_to(self.artifact_root / "runs")
        except ValueError as error:
            raise WorkerError(
                "scene_path must stay within this project's runs directory"
            ) from error
        if not path.parent.is_dir():
            raise WorkerError("run directory does not exist")
        return run_id, path

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """Check a locally recorded owner PID without trusting arbitrary lock contents."""
        if pid < 1:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _acquire_run_lock(self, scene_path: Path, run_id: int) -> None:
        """Claim one run-local worker lock; stale locks are recoverable after a crash."""
        lock_path = scene_path.parent / ".aculptoi-worker.lock"
        token = uuid.uuid4().hex
        payload = {"pid": os.getpid(), "run_id": run_id, "token": token}
        for _ in range(2):
            try:
                with lock_path.open("x", encoding="utf-8") as file:
                    json.dump(payload, file, sort_keys=True)
                self._lock_path = lock_path
                self._lock_token = token
                return
            except FileExistsError:
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                    owner_pid = existing.get("pid") if isinstance(existing, dict) else None
                except (OSError, json.JSONDecodeError):
                    raise WorkerError(
                        "run has an unreadable worker lock; inspect it before removing it"
                    ) from None
                if (
                    isinstance(owner_pid, int)
                    and not isinstance(owner_pid, bool)
                    and not self._pid_is_alive(owner_pid)
                ):
                    lock_path.unlink(missing_ok=True)
                    continue
                raise WorkerError(
                    f"run {run_id:06d} is already owned by another Blender worker"
                ) from None
        raise WorkerError(f"could not acquire ownership lock for run {run_id:06d}")

    def _release_run_lock(self) -> None:
        """Remove only the lock token created by this worker instance."""
        if self._lock_path is None or self._lock_token is None:
            return
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("token") == self._lock_token:
                self._lock_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        finally:
            self._lock_path = None
            self._lock_token = None

    def _apply_observer_guard(self) -> None:
        """Make accidental scene editing awkward; this is not a hostile-user boundary."""
        if self.mode != "ui":
            return
        for obj in bpy.context.scene.objects:
            obj.hide_select = True
        bpy.context.scene["aculptoi_observer_mode"] = True
        bpy.context.scene["aculptoi_observer_notice"] = "Aculptoi controls scene edits"
        workspace = bpy.context.workspace
        if workspace is not None:
            workspace.name = "Aculptoi Observer"

    def _allow_worker_selection(self) -> None:
        for obj in bpy.context.scene.objects:
            obj.hide_select = False

    @staticmethod
    def _redraw_viewports() -> None:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

    def attach_run(self, payload: object) -> dict[str, Any]:
        """Load or initialize the only mutable scene owned by this worker."""
        if not isinstance(payload, dict):
            raise WorkerError("run attachment payload must be an object")
        run_id, path = self._safe_scene_path(payload.get("scene_path"), payload.get("run_id"))
        reload_scene = payload.get("reload", False)
        if not isinstance(reload_scene, bool):
            raise WorkerError("reload must be a boolean")
        if self.active_run_id is not None and self.active_run_id != run_id:
            raise WorkerError(
                f"worker already owns run {self.active_run_id:06d}; "
                "release it before attaching another run"
            )
        acquired_lock = self.active_run_id is None
        if acquired_lock:
            self._acquire_run_lock(path, run_id)
        try:
            if path.is_file() and (reload_scene or self.active_scene_path != path):
                bpy.ops.wm.open_mainfile(filepath=str(path))
            self.active_run_id = run_id
            self.active_scene_path = path
            if not path.is_file():
                bpy.ops.wm.save_as_mainfile(filepath=str(path))
        except Exception:
            self.active_run_id = None
            self.active_scene_path = None
            if acquired_lock:
                self._release_run_lock()
            raise
        self._apply_observer_guard()
        self._redraw_viewports()
        return {"run_id": run_id, "scene_path": str(path), "active_filepath": bpy.data.filepath}

    def save_canonical_scene(self, payload: object) -> dict[str, Any]:
        """Durably save the active in-memory scene without creating a checkpoint."""
        if (
            not isinstance(payload, dict)
            or self.active_run_id is None
            or self.active_scene_path is None
        ):
            raise WorkerError("no run owns this worker")
        run_id, path = self._safe_scene_path(payload.get("scene_path"), self.active_run_id)
        if run_id != self.active_run_id or path != self.active_scene_path:
            raise WorkerError("only the active run's canonical scene may be saved")
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        self._apply_observer_guard()
        return {"scene_path": str(path), "active_filepath": bpy.data.filepath}

    def release_run(self) -> dict[str, Any]:
        """Release ownership while preserving the live visible scene for inspection."""
        previous_run_id = self.active_run_id
        self.active_run_id = None
        self.active_scene_path = None
        self._release_run_lock()
        return {"released_run_id": previous_run_id}

    def scene_inspect(self) -> dict[str, Any]:
        scene = bpy.context.scene
        return {
            "name": scene.name,
            "object_count": len(scene.objects),
            "camera": scene.camera.name if scene.camera else None,
            "world": scene.world.name if scene.world else None,
            "render": {
                "engine": scene.render.engine,
                "resolution_x": scene.render.resolution_x,
                "resolution_y": scene.render.resolution_y,
                "resolution_percentage": scene.render.resolution_percentage,
                "film_transparent": scene.render.film_transparent,
            },
            "objects": [
                _object_data(obj) for obj in sorted(scene.objects, key=lambda item: item.name)
            ],
            "active_object": (
                bpy.context.view_layer.objects.active.name
                if bpy.context.view_layer.objects.active
                else None
            ),
            "units": scene.unit_settings.system,
        }

    def object_list(self) -> dict[str, Any]:
        return {
            "objects": [
                _object_data(obj)
                for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name)
            ]
        }

    def object_inspect(self, name: str) -> dict[str, Any]:
        return {"object": _object_data(_require_object(name))}

    @staticmethod
    def _activate(obj: Any) -> None:
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    def _validate_action(self, action: object) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise WorkerError("each action must be an object")
        command = action.get("command")
        if command not in {
            "object.create",
            "object.delete",
            "object.translate",
            "object.rotate",
            "object.scale",
            "sculpt.voxel_remesh",
        }:
            raise WorkerError(f"unsupported command: {command!r}")
        if command == "object.create":
            _name(action.get("name"))
            if action.get("primitive", "cube") not in {"cube", "uv_sphere", "cylinder", "cone"}:
                raise WorkerError("unsupported primitive")
            _vector(action.get("location", [0, 0, 0]), "location")
            scale = _vector(action.get("scale", [1, 1, 1]), "scale")
            if any(value <= 0 or value > 100 for value in scale):
                raise WorkerError("scale must be within (0, 100]")
        elif command in {
            "object.delete",
            "object.translate",
            "object.rotate",
            "object.scale",
            "sculpt.voxel_remesh",
        }:
            _name(action.get("object"))
            if command == "object.translate":
                _vector(action.get("offset"), "offset")
            elif command == "object.rotate":
                _vector(action.get("degrees"), "degrees")
            elif command == "object.scale":
                scale = _vector(action.get("scale"), "scale")
                if any(value <= 0 or value > 100 for value in scale):
                    raise WorkerError("scale must be within (0, 100]")
            elif command == "sculpt.voxel_remesh":
                voxel_size = _number(action.get("voxel_size"), "voxel_size")
                if not 0.001 < voxel_size <= 1.0:
                    raise WorkerError("voxel_size must be within (0.001, 1]")
        return action

    def execute(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
            raise WorkerError("actions payload must contain an actions array")
        actions = payload["actions"]
        if not 1 <= len(actions) <= 25:
            raise WorkerError("actions array must contain 1 to 25 actions")
        validated = [self._validate_action(action) for action in actions]
        self._preflight_actions(validated)
        executed: list[dict[str, str]] = []
        self._allow_worker_selection()
        try:
            for action in validated:
                command = action["command"]
                if command == "object.create":
                    name = _name(action["name"])
                    location = _vector(action.get("location", [0, 0, 0]), "location")
                    primitive = action.get("primitive", "cube")
                    if primitive == "cube":
                        bpy.ops.mesh.primitive_cube_add(location=location)
                    elif primitive == "uv_sphere":
                        bpy.ops.mesh.primitive_uv_sphere_add(location=location)
                    elif primitive == "cylinder":
                        bpy.ops.mesh.primitive_cylinder_add(location=location)
                    else:
                        bpy.ops.mesh.primitive_cone_add(location=location)
                    obj = bpy.context.view_layer.objects.active
                    obj.name = name
                    obj.scale = _vector(action.get("scale", [1, 1, 1]), "scale")
                elif command == "object.delete":
                    obj = _require_object(action["object"])
                    self._activate(obj)
                    bpy.ops.object.delete()
                elif command == "object.translate":
                    obj = _require_object(action["object"])
                    offset = _vector(action["offset"], "offset")
                    obj.location = tuple(obj.location[index] + offset[index] for index in range(3))
                elif command == "object.rotate":
                    obj = _require_object(action["object"])
                    degrees = _vector(action["degrees"], "degrees")
                    obj.rotation_euler = tuple(
                        obj.rotation_euler[index] + math.radians(degrees[index])
                        for index in range(3)
                    )
                elif command == "object.scale":
                    obj = _require_object(action["object"])
                    scale = _vector(action["scale"], "scale")
                    obj.scale = tuple(obj.scale[index] * scale[index] for index in range(3))
                elif command == "sculpt.voxel_remesh":
                    obj = _require_object(action["object"])
                    if obj.type != "MESH":
                        raise WorkerError("sculpt.voxel_remesh requires a mesh object")
                    self._activate(obj)
                    obj.data.remesh_voxel_size = _number(action["voxel_size"], "voxel_size")
                    bpy.ops.object.voxel_remesh()
                else:  # Kept for defensive completeness if the allowlist changes.
                    raise WorkerError(f"unsupported command: {command}")
                executed.append({"command": str(command), "status": "ok"})
        finally:
            self._apply_observer_guard()
            self._redraw_viewports()
        if bpy.ops.ed.undo_push.poll():
            bpy.ops.ed.undo_push(message="Aculptoi action batch")
        return {"executed": executed}

    @staticmethod
    def _preflight_actions(actions: list[dict[str, Any]]) -> None:
        """Reject obvious batch failures before the first Blender mutation occurs."""
        available = {obj.name for obj in bpy.context.scene.objects}
        for action in actions:
            command = action["command"]
            if command == "object.create":
                name = _name(action["name"])
                if name in available:
                    raise WorkerError(f"object already exists: {name}")
                available.add(name)
                continue
            name = _name(action["object"])
            if name not in available:
                raise WorkerError(f"object not found: {name}")
            if command == "object.delete":
                available.remove(name)

    def _safe_artifact_dir(self, requested: object) -> Path:
        if not isinstance(requested, str):
            raise WorkerError("output_dir must be a string")
        directory = Path(requested).expanduser().resolve()
        try:
            directory.relative_to(self.artifact_root)
        except ValueError as error:
            raise WorkerError(
                "output_dir must stay within this project's .aculptoi directory"
            ) from error
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _safe_active_run_artifact_dir(self, requested: object) -> Path:
        """Restrict inspection output to the run currently owned by this worker."""
        if self.active_scene_path is None or self.active_run_id is None:
            raise WorkerError("no run owns this worker")
        if not isinstance(requested, str):
            raise WorkerError("output_dir must be a string")
        directory = Path(requested).expanduser().resolve()
        try:
            directory.relative_to(self.active_scene_path.parent.resolve())
        except ValueError as error:
            raise WorkerError(
                "inspection output_dir must stay within the active run directory"
            ) from error
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _renderable_objects(scene: Any) -> list[Any]:
        return sorted(
            [
                obj
                for obj in scene.objects
                if obj.type in INSPECTION_RENDERABLE_TYPES and not obj.hide_render
            ],
            key=lambda obj: obj.name,
        )

    @staticmethod
    def _point_payload(point: Vector) -> list[float]:
        return [round(float(point.x), 6), round(float(point.y), 6), round(float(point.z), 6)]

    @staticmethod
    def _inspection_text(value: object, label: str, pattern: re.Pattern[str]) -> str:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise WorkerError(f"{label} is invalid")
        return value

    def _camera_candidate(self, value: object) -> dict[str, Any]:
        expected = {
            "camera_id",
            "azimuth_degrees",
            "elevation_degrees",
            "orientation",
            "projection",
            "canonical_anchor",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise WorkerError("inspection candidate has an invalid shape")
        camera_id = self._inspection_text(value["camera_id"], "camera_id", CAMERA_ID_RE)
        azimuth = _number(value["azimuth_degrees"], "azimuth_degrees")
        elevation = _number(value["elevation_degrees"], "elevation_degrees")
        if not -180 <= azimuth <= 180 or not -89 <= elevation <= 89:
            raise WorkerError("inspection camera angles are out of range")
        orientation = value["orientation"]
        if not isinstance(orientation, str) or not orientation.strip() or len(orientation) > 120:
            raise WorkerError("inspection orientation is invalid")
        projection = value["projection"]
        if projection not in {"orthographic", "perspective"}:
            raise WorkerError("inspection projection is invalid")
        canonical_anchor = value["canonical_anchor"]
        if not isinstance(canonical_anchor, bool):
            raise WorkerError("canonical_anchor must be a boolean")
        return {
            "camera_id": camera_id,
            "azimuth_degrees": azimuth,
            "elevation_degrees": elevation,
            "orientation": orientation,
            "projection": projection,
            "canonical_anchor": canonical_anchor,
        }

    def _inspection_view(self, value: object) -> dict[str, Any]:
        expected = {
            "camera_id",
            "azimuth_degrees",
            "elevation_degrees",
            "orientation",
            "projection",
            "canonical_anchor",
            "selection_kind",
            "selection_reason",
            "coverage_gain",
            "information_gain",
            "tile_id",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise WorkerError("inspection camera plan view has an invalid shape")
        candidate = self._camera_candidate(
            {
                key: value[key]
                for key in (
                    "camera_id",
                    "azimuth_degrees",
                    "elevation_degrees",
                    "orientation",
                    "projection",
                    "canonical_anchor",
                )
            }
        )
        tile_id = self._inspection_text(value["tile_id"], "tile_id", TILE_ID_RE)
        selection_kind = value["selection_kind"]
        if selection_kind not in {"canonical_anchor", "dynamic"}:
            raise WorkerError("inspection selection_kind is invalid")
        reason = value["selection_reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 200:
            raise WorkerError("inspection selection_reason is invalid")
        for label in ("coverage_gain", "information_gain"):
            numeric = _number(value[label], label)
            if not 0 <= numeric <= 1:
                raise WorkerError(f"{label} must be between zero and one")
            candidate[label] = numeric
        candidate["tile_id"] = tile_id
        candidate["selection_kind"] = selection_kind
        candidate["selection_reason"] = reason
        return candidate

    def _inspection_plan(self, value: object) -> dict[str, Any]:
        expected = {
            "sensor_version",
            "lighting_rig",
            "round",
            "profile",
            "layout",
            "views",
            "estimated_surface_coverage",
            "selection_stop_reason",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise WorkerError("inspection plan has an invalid shape")
        sensor_version = self._inspection_text(
            value["sensor_version"], "sensor_version", VERSION_RE
        )
        lighting_rig = self._inspection_text(value["lighting_rig"], "lighting_rig", VERSION_RE)
        if lighting_rig != "neutral-studio-v1":
            raise WorkerError("unsupported inspection lighting rig")
        round_number = value["round"]
        if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 1:
            raise WorkerError("inspection round must be a positive integer")
        profile = value["profile"]
        if profile not in {"standard", "retry"}:
            raise WorkerError("inspection profile is invalid")
        layout = value["layout"]
        if not isinstance(layout, dict) or set(layout) != {
            "columns",
            "rows",
            "tile_dimension",
            "width",
            "height",
        }:
            raise WorkerError("inspection atlas layout is invalid")
        layout_numbers: dict[str, int] = {}
        for label in ("columns", "rows", "tile_dimension", "width", "height"):
            raw = layout[label]
            if not isinstance(raw, int) or isinstance(raw, bool) or not 1 <= raw <= 16_384:
                raise WorkerError(f"inspection layout {label} is invalid")
            layout_numbers[label] = raw
        if (
            layout_numbers["width"] != layout_numbers["columns"] * layout_numbers["tile_dimension"]
            or layout_numbers["height"] != layout_numbers["rows"] * layout_numbers["tile_dimension"]
        ):
            raise WorkerError("inspection atlas layout dimensions do not match its grid")
        views = value["views"]
        if not isinstance(views, list) or not 1 <= len(views) <= 128:
            raise WorkerError("inspection plan must contain one to 128 views")
        parsed_views = [self._inspection_view(view) for view in views]
        if len(parsed_views) > layout_numbers["columns"] * layout_numbers["rows"]:
            raise WorkerError("inspection plan views exceed atlas layout capacity")
        if len({view["camera_id"] for view in parsed_views}) != len(parsed_views):
            raise WorkerError("inspection plan camera ids must be unique")
        if len({view["tile_id"] for view in parsed_views}) != len(parsed_views):
            raise WorkerError("inspection plan tile ids must be unique")
        coverage = _number(value["estimated_surface_coverage"], "estimated_surface_coverage")
        if not 0 <= coverage <= 1:
            raise WorkerError("estimated_surface_coverage must be between zero and one")
        if value["selection_stop_reason"] not in {
            "coverage_target",
            "min_view_gain",
            "max_views",
            "atlas_resolution",
            "candidate_exhausted",
        }:
            raise WorkerError("inspection selection_stop_reason is invalid")
        return {
            "sensor_version": sensor_version,
            "lighting_rig": lighting_rig,
            "round": round_number,
            "profile": profile,
            "layout": layout_numbers,
            "views": parsed_views,
            "estimated_surface_coverage": coverage,
            "selection_stop_reason": value["selection_stop_reason"],
        }

    @staticmethod
    def _inspection_bounds(
        objects: list[Any], depsgraph: Any
    ) -> tuple[dict[str, Any], Vector, float]:
        points: list[Vector] = []
        for obj in objects:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh() if obj.type == "MESH" else None
            try:
                source_points = (
                    [vertex.co for vertex in mesh.vertices]
                    if mesh is not None
                    else [Vector(corner) for corner in evaluated.bound_box]
                )
                for point in source_points:
                    world_point = evaluated.matrix_world @ point
                    if all(math.isfinite(component) for component in world_point):
                        points.append(world_point)
            finally:
                if mesh is not None:
                    evaluated.to_mesh_clear()
        if not points:
            raise WorkerError("no renderable inspection geometry has usable bounds")
        minimum = Vector(
            (
                min(point.x for point in points),
                min(point.y for point in points),
                min(point.z for point in points),
            )
        )
        maximum = Vector(
            (
                max(point.x for point in points),
                max(point.y for point in points),
                max(point.z for point in points),
            )
        )
        center = (minimum + maximum) / 2
        radius = max((maximum - minimum).length / 2, 0.001)
        return (
            {
                "minimum": AculptoiWorker._point_payload(minimum),
                "maximum": AculptoiWorker._point_payload(maximum),
                "center": AculptoiWorker._point_payload(center),
                "radius": round(radius, 6),
            },
            center,
            radius,
        )

    @staticmethod
    def _surface_samples(objects: list[Any], depsgraph: Any) -> list[tuple[Vector, Vector]]:
        samples: list[tuple[str, int, Vector, Vector]] = []
        for obj in objects:
            if obj.type != "MESH":
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                normal_matrix = evaluated.matrix_world.to_3x3()
                for triangle in mesh.loop_triangles:
                    vertices = [mesh.vertices[index].co for index in triangle.vertices]
                    center = evaluated.matrix_world @ (vertices[0] + vertices[1] + vertices[2]) / 3
                    normal = (normal_matrix @ triangle.normal).normalized()
                    if all(math.isfinite(component) for component in center) and normal.length:
                        samples.append((obj.name, triangle.index, center, normal))
            finally:
                evaluated.to_mesh_clear()
        samples.sort(key=lambda sample: (sample[0], sample[1]))
        if len(samples) > 2048:
            samples = [samples[index * len(samples) // 2048] for index in range(2048)]
        return [(center, normal) for _, _, center, normal in samples]

    @staticmethod
    def _camera_direction(azimuth_degrees: float, elevation_degrees: float) -> Vector:
        azimuth = math.radians(azimuth_degrees)
        elevation = math.radians(elevation_degrees)
        return Vector(
            (
                math.sin(azimuth) * math.cos(elevation),
                -math.cos(azimuth) * math.cos(elevation),
                math.sin(elevation),
            )
        )

    @staticmethod
    def _framing(camera: Any, radius: float, profile: str) -> dict[str, float]:
        margin = 1.25 if profile == "retry" else 1.15
        orthographic_scale = max(2 * radius * margin, 0.001)
        half_fov = max(min(camera.data.angle_x, camera.data.angle_y) / 2, math.radians(5))
        distance = max(radius * margin / math.sin(half_fov), radius * 2)
        camera.data.clip_start = max(0.001, distance - radius * 1.5)
        camera.data.clip_end = distance + radius * 2
        return {
            "margin": round(margin, 6),
            "distance": round(distance, 6),
            "orthographic_scale": round(orthographic_scale, 6),
        }

    @staticmethod
    def _new_inspection_light(
        collection: Any, name: str, location: Vector, energy: float, size: float
    ) -> Any:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        collection.objects.link(light)
        light.location = location
        return light

    @contextmanager
    def _inspection_scene(
        self,
        source_scene: Any,
        objects: list[Any],
        resolution: int,
        transparent: bool,
    ) -> Any:
        """Create a fully isolated observational scene sharing only source objects."""
        temporary_scene = bpy.data.scenes.new("AculptoiInspectionScene")
        subject_collection = bpy.data.collections.new("AculptoiInspectionSubject")
        rig_collection = bpy.data.collections.new("AculptoiInspectionRig")
        world: Any | None = None
        camera_data: Any | None = None
        private_objects: list[Any] = []
        private_lights: list[Any] = []
        try:
            temporary_scene.collection.children.link(subject_collection)
            temporary_scene.collection.children.link(rig_collection)
            for obj in objects:
                subject_collection.objects.link(obj)
            world = bpy.data.worlds.new("AculptoiInspectionWorld")
            world.use_nodes = True
            background = world.node_tree.nodes.get("Background")
            if background is not None:
                background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
                background.inputs["Strength"].default_value = 0.35
            temporary_scene.world = world
            camera_data = bpy.data.cameras.new("AculptoiInspectionCamera")
            camera_data.lens = 50
            camera = bpy.data.objects.new("AculptoiInspectionCamera", camera_data)
            rig_collection.objects.link(camera)
            private_objects.append(camera)
            temporary_scene.camera = camera
            temporary_scene.frame_set(source_scene.frame_current)
            temporary_scene.render.resolution_x = resolution
            temporary_scene.render.resolution_y = resolution
            temporary_scene.render.resolution_percentage = 100
            temporary_scene.render.image_settings.file_format = "PNG"
            temporary_scene.render.film_transparent = transparent
            for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
                try:
                    temporary_scene.render.engine = engine
                    break
                except TypeError:
                    continue
            else:
                temporary_scene.render.engine = source_scene.render.engine
            yield {
                "scene": temporary_scene,
                "camera": camera,
                "rig_collection": rig_collection,
                "private_objects": private_objects,
                "private_lights": private_lights,
            }
        finally:
            bpy.data.scenes.remove(temporary_scene, do_unlink=True)
            for obj in private_objects:
                if obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            for light in private_lights:
                if light.name in bpy.data.lights:
                    bpy.data.lights.remove(light, do_unlink=True)
            if camera_data is not None and camera_data.name in bpy.data.cameras:
                bpy.data.cameras.remove(camera_data, do_unlink=True)
            if subject_collection.name in bpy.data.collections:
                bpy.data.collections.remove(subject_collection)
            if rig_collection.name in bpy.data.collections:
                bpy.data.collections.remove(rig_collection)
            if world is not None and world.name in bpy.data.worlds:
                bpy.data.worlds.remove(world)

    def _configure_inspection_lighting(
        self, environment: dict[str, Any], center: Vector, radius: float, profile: str
    ) -> Any:
        rig = environment["rig_collection"]
        scale = max(radius, 0.1)
        lighting_scale = scale * scale
        key = self._new_inspection_light(
            rig,
            "AculptoiInspectionKey",
            center + Vector((3, -4, 5)) * scale,
            700 * lighting_scale,
            4 * scale,
        )
        fill = self._new_inspection_light(
            rig,
            "AculptoiInspectionFill",
            center + Vector((-4, -2, 2)) * scale,
            350 * lighting_scale,
            5 * scale,
        )
        rim = self._new_inspection_light(
            rig,
            "AculptoiInspectionRim",
            center + Vector((1, 4, 4)) * scale,
            500 * lighting_scale,
            3 * scale,
        )
        camera_fill = self._new_inspection_light(
            rig,
            "AculptoiInspectionCameraFill",
            center + Vector((0, -3, 1)) * scale,
            (180 if profile == "retry" else 100) * lighting_scale,
            3 * scale,
        )
        for light in (key, fill, rim, camera_fill):
            self._point_camera(light, light.location, center)
        environment["private_objects"].extend((key, fill, rim, camera_fill))
        environment["private_lights"].extend((key.data, fill.data, rim.data, camera_fill.data))
        return camera_fill

    def _configure_inspection_camera(
        self,
        camera: Any,
        camera_fill: Any,
        center: Vector,
        framing: dict[str, float],
        view: dict[str, Any],
    ) -> Vector:
        direction = self._camera_direction(view["azimuth_degrees"], view["elevation_degrees"])
        position = center + direction * framing["distance"]
        camera.data.type = "ORTHO" if view["projection"] == "orthographic" else "PERSP"
        camera.data.ortho_scale = framing["orthographic_scale"]
        self._point_camera(camera, position, center)
        camera_fill.location = center + direction * (framing["distance"] * 0.75)
        camera_fill.location.z += framing["orthographic_scale"] * 0.2
        self._point_camera(camera_fill, camera_fill.location, center)
        return position

    @staticmethod
    def _silhouette_diagnostic(image: Any) -> tuple[float, str]:
        if image.size[0] < 1 or image.size[1] < 1:
            raise WorkerError("inspection candidate render produced an empty image")
        width, height = image.size
        pixels = list(image.pixels)
        alpha = [pixels[index] for index in range(3, len(pixels), 4)]
        if len(alpha) != width * height:
            raise WorkerError("inspection candidate render has invalid alpha pixels")
        mask = [value > 0.1 for value in alpha]
        coverage = sum(mask) / len(mask)
        signature = 0
        for row in range(16):
            for column in range(16):
                x_start = column * width // 16
                x_end = max(x_start + 1, (column + 1) * width // 16)
                y_start = row * height // 16
                y_end = max(y_start + 1, (row + 1) * height // 16)
                if any(
                    mask[y * width + x]
                    for y in range(y_start, y_end)
                    for x in range(x_start, x_end)
                ):
                    signature |= 1 << (row * 16 + column)
        return round(coverage, 6), f"{signature:064x}"

    @staticmethod
    def _visible_surface_samples(
        scene: Any,
        depsgraph: Any,
        position: Vector,
        samples: list[tuple[Vector, Vector]],
        tolerance: float,
    ) -> list[int]:
        visible: list[int] = []
        for index, (point, normal) in enumerate(samples):
            direction = point - position
            distance = direction.length
            if distance <= 0 or normal.dot(-direction.normalized()) <= 0:
                continue
            hit = scene.ray_cast(
                depsgraph,
                position,
                direction.normalized(),
                distance=distance + tolerance,
            )
            if hit[0] and (hit[1] - point).length <= tolerance:
                visible.append(index)
        return visible

    def analyze_inspection_candidates(self, payload: object) -> dict[str, Any]:
        """Measure temporary low-resolution diagnostics for Python-owned camera candidates."""
        if not isinstance(payload, dict) or set(payload) != {"candidates"}:
            raise WorkerError("inspection candidate payload must contain only candidates")
        requested = payload["candidates"]
        if not isinstance(requested, list) or not 4 <= len(requested) <= 256:
            raise WorkerError("inspection candidates must contain four to 256 entries")
        candidates = [self._camera_candidate(candidate) for candidate in requested]
        if len({candidate["camera_id"] for candidate in candidates}) != len(candidates):
            raise WorkerError("inspection candidate camera ids must be unique")
        source_scene = bpy.context.scene
        objects = self._renderable_objects(source_scene)
        if not objects:
            raise WorkerError("no renderable inspection geometry exists")
        if self.active_scene_path is None or self.active_run_id is None:
            raise WorkerError("no run owns this worker")
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bounds, center, radius = self._inspection_bounds(objects, depsgraph)
        samples = self._surface_samples(objects, depsgraph)
        diagnostics: list[dict[str, Any]] = []
        with (
            tempfile.TemporaryDirectory(
                prefix=".inspection-candidates-", dir=self.active_scene_path.parent
            ) as temporary_directory,
            self._inspection_scene(source_scene, objects, 128, transparent=True) as environment,
        ):
            camera = environment["camera"]
            camera_fill = self._configure_inspection_lighting(
                environment, center, radius, profile="standard"
            )
            framing = self._framing(camera, radius, "standard")
            tolerance = max(radius * 0.001, 0.0001)
            for candidate in candidates:
                position = self._configure_inspection_camera(
                    camera, camera_fill, center, framing, candidate
                )
                target = Path(temporary_directory) / f"{candidate['camera_id']}.png"
                environment["scene"].render.filepath = str(target)
                bpy.ops.render.render(scene=environment["scene"].name, write_still=True)
                image = bpy.data.images.load(str(target), check_existing=False)
                try:
                    coverage, signature = self._silhouette_diagnostic(image)
                finally:
                    bpy.data.images.remove(image)
                diagnostics.append(
                    {
                        "camera_id": candidate["camera_id"],
                        "frame_coverage": coverage,
                        "surface_sample_ids": self._visible_surface_samples(
                            source_scene, depsgraph, position, samples, tolerance
                        ),
                        "surface_sample_count": len(samples),
                        "silhouette_signature": signature,
                    }
                )
        return {"bounds": bounds, "diagnostics": diagnostics}

    def render_inspection_views(self, payload: object) -> dict[str, Any]:
        """Render selected inspection cameras without touching the canonical scene."""
        if not isinstance(payload, dict) or set(payload) != {"plan", "output_dir"}:
            raise WorkerError("inspection render payload must contain plan and output_dir")
        plan = self._inspection_plan(payload["plan"])
        output_dir = self._safe_active_run_artifact_dir(payload["output_dir"])
        source_scene = bpy.context.scene
        objects = self._renderable_objects(source_scene)
        if not objects:
            raise WorkerError("no renderable inspection geometry exists")
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bounds, center, radius = self._inspection_bounds(objects, depsgraph)
        tile_dimension = plan["layout"]["tile_dimension"]
        shots: list[dict[str, Any]] = []
        with self._inspection_scene(
            source_scene, objects, tile_dimension, transparent=False
        ) as environment:
            camera = environment["camera"]
            camera_fill = self._configure_inspection_lighting(
                environment, center, radius, plan["profile"]
            )
            framing = self._framing(camera, radius, plan["profile"])
            for view in plan["views"]:
                self._configure_inspection_camera(camera, camera_fill, center, framing, view)
                target = output_dir / f"{view['tile_id']}.png"
                environment["scene"].render.filepath = str(target)
                bpy.ops.render.render(scene=environment["scene"].name, write_still=True)
                shots.append(
                    {
                        "camera_id": view["camera_id"],
                        "tile_id": view["tile_id"],
                        "path": str(target),
                        "width": tile_dimension,
                        "height": tile_dimension,
                    }
                )
        return {
            "bounds": bounds,
            "framing": framing,
            "shots": shots,
        }

    @staticmethod
    def _bounds(objects: list[Any]) -> tuple[Vector, float]:
        points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
        if not points:
            return Vector((0, 0, 0)), 4.0
        minimum = Vector(
            (
                min(point.x for point in points),
                min(point.y for point in points),
                min(point.z for point in points),
            )
        )
        maximum = Vector(
            (
                max(point.x for point in points),
                max(point.y for point in points),
                max(point.z for point in points),
            )
        )
        center = (minimum + maximum) / 2
        return center, max((maximum - minimum).length * 1.6, 3.0)

    @staticmethod
    def _point_camera(camera: Any, position: Vector, target: Vector) -> None:
        camera.location = position
        camera.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()

    def render_views(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WorkerError("render payload must be an object")
        requested_views = payload.get("views")
        if not isinstance(requested_views, list) or not 1 <= len(requested_views) <= 4:
            raise WorkerError("views must contain one to four named views")
        views = [view for view in requested_views if isinstance(view, str)]
        if (
            len(views) != len(requested_views)
            or len(set(views)) != len(views)
            or not set(views) <= ALLOWED_VIEWS
        ):
            raise WorkerError("views must be unique names from front,right,top,perspective")
        output_dir = self._safe_artifact_dir(payload.get("output_dir"))
        target_name = payload.get("object")
        if target_name is not None:
            objects = [_require_object(target_name)]
        else:
            objects = [
                obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.visible_get()
            ]
        center, distance = self._bounds(objects)
        scene = bpy.context.scene
        original_camera = scene.camera
        original_path = scene.render.filepath
        original_resolution = (
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.resolution_percentage,
        )
        bpy.ops.object.camera_add()
        camera = bpy.context.view_layer.objects.active
        camera.name = "AculptoiInspectionCamera"
        scene.camera = camera
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        positions = {
            "front": center + Vector((0, -distance, distance * 0.2)),
            "right": center + Vector((distance, 0, distance * 0.2)),
            "top": center + Vector((0, 0, distance)),
            "perspective": center + Vector((distance * 0.8, -distance * 0.8, distance * 0.6)),
        }
        paths: list[str] = []
        try:
            for view in views:
                self._point_camera(camera, positions[view], center)
                target_path = output_dir / f"{view}.png"
                scene.render.image_settings.file_format = "PNG"
                scene.render.filepath = str(target_path)
                bpy.ops.render.render(write_still=True)
                paths.append(str(target_path))
        finally:
            scene.camera = original_camera
            scene.render.filepath = original_path
            (
                scene.render.resolution_x,
                scene.render.resolution_y,
                scene.render.resolution_percentage,
            ) = original_resolution
            bpy.data.objects.remove(camera, do_unlink=True)
        return {"paths": paths, "views": views}


class Handler(BaseHTTPRequestHandler):
    """HTTP adapter with explicit routes and JSON-only responses."""

    server: Any

    def log_message(self, format: str, *args: object) -> None:
        print("[aculptoi-worker] " + format % args, flush=True)

    @property
    def worker(self) -> AculptoiWorker:
        return self.server.worker

    def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _payload(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise WorkerError("invalid Content-Length") from error
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise WorkerError("request body must be between 1 byte and 1 MB")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise WorkerError("request body must be valid JSON") from error

    def _handle(self, method: str, path: str, payload: object | None = None) -> dict[str, Any]:
        if method == "GET" and path == "/v1/health":
            return self.worker.health()
        if method == "GET" and path == "/v1/scene/inspect":
            return self.worker.scene_inspect()
        if method == "GET" and path == "/v1/objects":
            return self.worker.object_list()
        if method == "GET" and path.startswith("/v1/objects/"):
            return self.worker.object_inspect(unquote(path.removeprefix("/v1/objects/")))
        if method == "POST" and path == "/v1/actions/execute":
            return self.worker.execute(payload)
        if method == "POST" and path == "/v1/run/attach":
            return self.worker.attach_run(payload)
        if method == "POST" and path == "/v1/run/release":
            return self.worker.release_run()
        if method == "POST" and path == "/v1/scene/save":
            return self.worker.save_canonical_scene(payload)
        if method == "POST" and path == "/v1/render/views":
            return self.worker.render_views(payload)
        if method == "POST" and path == "/v1/inspection/candidates/analyze":
            return self.worker.analyze_inspection_candidates(payload)
        if method == "POST" and path == "/v1/inspection/render":
            return self.worker.render_inspection_views(payload)
        if method == "POST" and path == "/v1/shutdown":
            self.server.stopping = True  # type: ignore[attr-defined]
            return {"stopping": True}
        raise WorkerError("unknown route")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            payload = self._payload() if method == "POST" else None
            data = self.server.call_worker(  # type: ignore[attr-defined]
                lambda: self._handle(method, urlparse(self.path).path, payload)
            )
            self._send(HTTPStatus.OK, {"ok": True, "data": data})
        except WorkerError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        except Exception as error:  # Blender can throw context-specific runtime errors.
            print(f"[aculptoi-worker] internal error: {error!r}", flush=True)
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "worker operation failed"}
            )


def main() -> None:
    """Start the worker after Blender's `--` separator arguments."""
    arguments = __import__("sys").argv
    arguments = arguments[arguments.index("--") + 1 :] if "--" in arguments else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost", "::1"))
    parser.add_argument("--port", default=9876, type=int)
    parser.add_argument("--mode", default="ui", choices=("ui", "headless"))
    options = parser.parse_args(arguments)
    worker = AculptoiWorker(options.mode)
    server_type = ThreadingHTTPServer if options.mode == "ui" else HTTPServer
    server = server_type((options.host, options.port), Handler)
    server.worker = worker  # type: ignore[attr-defined]
    server.stopping = False  # type: ignore[attr-defined]
    print(f"[aculptoi-worker] listening on http://{options.host}:{options.port}", flush=True)
    if options.mode == "headless":
        server.call_worker = lambda operation: operation()  # type: ignore[attr-defined]
        server.timeout = 0.5
        try:
            while not server.stopping:  # type: ignore[attr-defined]
                server.handle_request()
        finally:
            server.server_close()
            worker.release_run()
            bpy.ops.wm.quit_blender()
        return

    dispatcher = MainThreadDispatcher()
    server.call_worker = dispatcher.call  # type: ignore[attr-defined]
    worker._apply_observer_guard()
    thread = threading.Thread(target=server.serve_forever, name="aculptoi-http", daemon=True)
    thread.start()

    def pump_ui_requests() -> float | None:
        dispatcher.pump()
        if not server.stopping:  # type: ignore[attr-defined]
            return 0.05
        dispatcher.stopping = True
        server.shutdown()
        server.server_close()
        worker.release_run()
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(pump_ui_requests, first_interval=0.05, persistent=True)


if __name__ == "__main__":
    main()
