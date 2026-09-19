"""Persistent localhost-only Aculptoi worker, executed by Blender's Python.

This module intentionally uses only Blender and Python standard-library APIs. It
does not receive or execute model-generated Python, shell commands, or paths.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
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

import bmesh  # type: ignore[import-not-found]
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
MAX_ACTIONS_PER_BATCH = 25
MAX_JOIN_OBJECTS = 16
MAX_AFFECTED_ELEMENTS = 20_000
MAX_RESULT_VERTICES = 100_000
MAX_RESULT_POLYGONS = 100_000
MAX_NORMALIZED_OFFSET = 2.0
MAX_REGION_SCALE = 4.0
MAX_SMOOTH_ITERATIONS = 10
ACTOR_VIEWPORT_MAX_DIMENSION = 768
MAX_VIEWPORT_IMAGE_BYTES = 1_500_000
REGION_EPSILON = 1e-6
logger = logging.getLogger(__name__)


class WorkerError(ValueError):
    """A request did not meet the worker's independent safety requirements."""


class WorkerActionError(WorkerError):
    """A deliberate bounded action failure that can safely reach the Actor as feedback."""

    def __init__(
        self,
        failure_kind: str,
        code: str,
        message: str,
        *,
        action_index: int | None = None,
        command: str | None = None,
        executed_before_failure: int = 0,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.code = code
        self.message = message
        self.action_index = action_index
        self.command = command
        self.executed_before_failure = executed_before_failure
        self.rolled_back = False
        self.scene_restored = False

    @classmethod
    def validation(cls, code: str, message: str) -> WorkerActionError:
        return cls("validation_error", code, message)

    @classmethod
    def execution(cls, code: str, message: str) -> WorkerActionError:
        return cls("execution_error", code, message)

    def with_action_context(
        self, *, action_index: int, command: str, executed_before_failure: int
    ) -> WorkerActionError:
        """Add the bounded batch context known only by the dispatcher."""
        self.action_index = action_index
        self.command = command
        self.executed_before_failure = executed_before_failure
        return self

    def mark_scene_restored(self, restored: bool) -> WorkerActionError:
        """Record rollback facts only after a canonical reload actually completed."""
        self.rolled_back = restored
        self.scene_restored = restored
        return self

    def error_payload(self) -> dict[str, Any]:
        return {
            "failure_kind": self.failure_kind,
            "code": self.code,
            "message": self.message,
            "action_index": self.action_index,
            "command": self.command,
            "executed_before_failure": self.executed_before_failure,
            "recoverable": True,
            "rolled_back": self.rolled_back,
            "scene_restored": self.scene_restored,
        }


class WorkerInternalError(WorkerError):
    """A trusted worker fault that must stop Actor mutation instead of requesting a retry."""

    def __init__(
        self,
        code: str = "internal_worker_error",
        message: str = "internal Blender worker failure",
        *,
        action_index: int | None = None,
        command: str | None = None,
        executed_before_failure: int = 0,
        scene_restored: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.action_index = action_index
        self.command = command
        self.executed_before_failure = executed_before_failure
        self.scene_restored = scene_restored

    def with_action_context(
        self, *, action_index: int, command: str, executed_before_failure: int
    ) -> WorkerInternalError:
        """Add bounded execution context without exposing the underlying exception."""
        self.action_index = action_index
        self.command = command
        self.executed_before_failure = executed_before_failure
        return self

    def mark_scene_restored(self, restored: bool) -> WorkerInternalError:
        """Record whether trusted canonical-scene restoration completed."""
        self.scene_restored = restored
        return self

    def error_payload(self) -> dict[str, Any]:
        return {
            "failure_kind": "worker_error",
            "code": self.code,
            "message": self.message,
            "action_index": self.action_index,
            "command": self.command,
            "executed_before_failure": self.executed_before_failure,
            "recoverable": False,
            "rolled_back": self.scene_restored,
            "scene_restored": self.scene_restored,
        }


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


def _exact_action_fields(
    action: dict[str, Any], *, required: set[str], optional: set[str] | None = None
) -> None:
    """Reject unknown action fields independently of Python-side schema validation."""
    keys = set(action)
    missing = required - keys
    unexpected = keys - required - (optional or set())
    if missing:
        raise WorkerError(f"action is missing required fields: {sorted(missing)}")
    if unexpected:
        raise WorkerError(f"action contains unsupported fields: {sorted(unexpected)}")


def _normalized_region(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Validate a nonempty inclusive region in the local-bounds range [-1, 1]."""
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise WorkerError("region must contain exactly min and max vectors")
    minimum = _vector(value["min"], "region.min")
    maximum = _vector(value["max"], "region.max")
    if any(component < -1 or component > 1 for component in (*minimum, *maximum)):
        raise WorkerError("region coordinates must be within [-1, 1]")
    if any(lower >= upper for lower, upper in zip(minimum, maximum, strict=True)):
        raise WorkerError("region min must be strictly less than max on every axis")
    return minimum, maximum


def _bounded_normalized_vector(value: object, label: str) -> tuple[float, float, float]:
    """Validate a local-bounds relative vector without accepting unbounded displacement."""
    vector = _vector(value, label)
    if any(abs(component) > MAX_NORMALIZED_OFFSET for component in vector):
        raise WorkerError(
            f"{label} components must be within [-{MAX_NORMALIZED_OFFSET}, {MAX_NORMALIZED_OFFSET}]"
        )
    return vector


def _region_scale(value: object, label: str) -> tuple[float, float, float]:
    """Validate a bounded positive local-region scale."""
    scale = _vector(value, label)
    if any(component <= 0 or component > MAX_REGION_SCALE for component in scale):
        raise WorkerError(f"{label} multipliers must be within (0, {MAX_REGION_SCALE}]")
    return scale


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
        self._actor_viewport: tuple[Any, Any] | None = None

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

    def _select_actor_viewport(self) -> tuple[Any, Any, Any, Any]:
        """Reserve the largest deterministic VIEW_3D area for Actor observation.

        Blender's normal single-window layouts do not safely provide a second
        editor without reshaping the user's workspace.  The selected area is
        therefore reserved while a run is active and is reset explicitly before
        each capture; user navigation in it never becomes Actor state.
        """
        if self.mode != "ui":
            raise WorkerError("Actor viewport observation is unavailable in headless mode")
        choices: list[tuple[Any, Any]] = []
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    choices.append((window, area))
        if not choices:
            raise WorkerError("no VIEW_3D area is available for Actor observation")
        if self._actor_viewport in choices:
            window, area = self._actor_viewport
        else:
            window, area = max(
                choices,
                key=lambda item: (item[1].width * item[1].height, -item[1].y, -item[1].x),
            )
            self._actor_viewport = (window, area)
        space = area.spaces.active
        region = next((item for item in area.regions if item.type == "WINDOW"), None)
        if region is None or space is None or space.type != "VIEW_3D":
            self._actor_viewport = None
            raise WorkerError("reserved Actor viewport no longer has a 3D window region")
        return window, area, region, space

    @staticmethod
    def _viewport_view(payload: object) -> dict[str, object]:
        """Validate the tiny semantic viewport language independently in Blender."""
        if not isinstance(payload, dict) or set(payload) != {"view"}:
            raise WorkerError("viewport observation payload must contain only view")
        view = payload["view"]
        if not isinstance(view, dict):
            raise WorkerError("viewport view must be an object")
        allowed = {"target", "orientation", "projection", "framing"}
        if set(view) - allowed:
            raise WorkerError("viewport view contains unsupported fields")
        target = view.get("target")
        if target is not None:
            _name(target)
        orientation = view.get("orientation", "front_three_quarter")
        projection = view.get("projection", "orthographic")
        framing = view.get("framing", "whole_subject")
        if orientation not in {
            "front",
            "rear",
            "left",
            "right",
            "top",
            "bottom",
            "front_three_quarter",
            "rear_three_quarter",
        }:
            raise WorkerError("viewport orientation is unsupported")
        if projection not in {"orthographic", "perspective"}:
            raise WorkerError("viewport projection is unsupported")
        if framing not in {"whole_subject", "medium", "close"}:
            raise WorkerError("viewport framing is unsupported")
        return {
            "target": target,
            "orientation": orientation,
            "projection": projection,
            "framing": framing,
        }

    @staticmethod
    def _viewport_subject_bounds(target: str | None) -> tuple[Vector, float]:
        """Derive deterministic framing from bounded scene geometry, not user selection."""
        if target is not None:
            objects = [_require_object(target)]
        else:
            objects = [
                obj
                for obj in bpy.context.scene.objects
                if obj.type in INSPECTION_RENDERABLE_TYPES and not obj.hide_viewport
            ]
        points: list[Vector] = []
        for obj in objects:
            if not getattr(obj, "bound_box", None):
                continue
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        if not points:
            raise WorkerError("viewport target has no visible bounded geometry")
        minimum = Vector(min(point[index] for point in points) for index in range(3))
        maximum = Vector(max(point[index] for point in points) for index in range(3))
        extent = max((maximum - minimum).length, 0.25)
        return (minimum + maximum) / 2, extent

    @staticmethod
    def _configure_actor_viewport(
        window: Any,
        area: Any,
        region: Any,
        space: Any,
        view: dict[str, object],
    ) -> None:
        """Translate semantic framing into fixed, read-only VIEW_3D mechanics."""
        center, extent = AculptoiWorker._viewport_subject_bounds(view["target"])
        region_3d = space.region_3d
        framing_scale = {"whole_subject": 1.35, "medium": 0.85, "close": 0.5}[view["framing"]]
        region_3d.view_location = center
        region_3d.view_distance = max(0.1, extent * framing_scale)
        region_3d.view_perspective = "ORTHO" if view["projection"] == "orthographic" else "PERSP"
        space.shading.type = "SOLID"
        space.shading.light = "STUDIO"
        if hasattr(space.shading, "studiolight_rotate_z"):
            space.shading.studiolight_rotate_z = 0.0
        else:  # Blender 4.x compatibility.
            space.shading.studiolight_rotate = 0.0
        space.shading.background_type = "VIEWPORT"
        space.shading.background_color = (0.055, 0.055, 0.055)
        space.overlay.show_overlays = False
        space.show_gizmo = False
        base_orientation = {
            "front": "FRONT",
            "rear": "BACK",
            "left": "LEFT",
            "right": "RIGHT",
            "top": "TOP",
            "bottom": "BOTTOM",
            "front_three_quarter": "FRONT",
            "rear_three_quarter": "BACK",
        }[view["orientation"]]
        try:
            with bpy.context.temp_override(
                window=window,
                screen=window.screen,
                area=area,
                region=region,
                space_data=space,
                region_data=region_3d,
            ):
                bpy.ops.view3d.view_axis(type=base_orientation, align_active=False, relative=False)
                if view["orientation"] == "front_three_quarter":
                    for _ in range(3):
                        bpy.ops.view3d.view_orbit(type="ORBITLEFT")
                elif view["orientation"] == "rear_three_quarter":
                    for _ in range(3):
                        bpy.ops.view3d.view_orbit(type="ORBITRIGHT")
        except RuntimeError as error:
            raise WorkerError("could not configure the reserved Actor viewport") from error

    @staticmethod
    def _capture_actor_viewport(
        window: Any, area: Any, region: Any, space: Any
    ) -> tuple[bytes, int, int]:
        """Capture only the selected 3D editor and bound it before local transport.

        Blender's OpenGL viewport operator writes through its image backend.  A
        private OS-temporary file is therefore used only inside this worker,
        immediately read into memory, and removed in ``finally``.  No capture
        enters a run directory or persists after the request.
        """
        descriptor, raw_name = tempfile.mkstemp(prefix="aculptoi-viewport-", suffix=".png")
        os.close(descriptor)
        raw_path = Path(raw_name)
        resized_path = raw_path.with_name(f"{raw_path.stem}-bounded.png")
        scene = bpy.context.scene
        original_path = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        image: Any | None = None
        try:
            scene.render.filepath = str(raw_path)
            scene.render.image_settings.file_format = "PNG"
            with bpy.context.temp_override(
                window=window,
                screen=window.screen,
                area=area,
                region=region,
                space_data=space,
                region_data=space.region_3d,
            ):
                bpy.ops.render.opengl(write_still=True, view_context=True)
            image = bpy.data.images.load(str(raw_path), check_existing=False)
            width, height = image.size
            longest = max(width, height)
            selected_path = raw_path
            if longest > ACTOR_VIEWPORT_MAX_DIMENSION:
                scale = ACTOR_VIEWPORT_MAX_DIMENSION / longest
                width = max(1, round(width * scale))
                height = max(1, round(height * scale))
                image.scale(width, height)
                image.filepath_raw = str(resized_path)
                image.file_format = "PNG"
                image.save()
                selected_path = resized_path
            data = selected_path.read_bytes()
            if len(data) > MAX_VIEWPORT_IMAGE_BYTES:
                raise WorkerError("bounded Actor viewport image exceeds local transport limit")
            return data, int(width), int(height)
        except RuntimeError as error:
            raise WorkerError("could not capture the reserved Actor viewport") from error
        finally:
            scene.render.filepath = original_path
            scene.render.image_settings.file_format = original_format
            if image is not None:
                bpy.data.images.remove(image, do_unlink=True)
            raw_path.unlink(missing_ok=True)
            resized_path.unlink(missing_ok=True)

    def observe_viewport(self, payload: object) -> dict[str, Any]:
        """Apply a semantic view and return a transient PNG plus compact metadata."""
        view = self._viewport_view(payload)
        window, area, region, space = self._select_actor_viewport()
        self._configure_actor_viewport(window, area, region, space, view)
        self._redraw_viewports()
        pixels, width, height = self._capture_actor_viewport(window, area, region, space)
        return {
            "observation": {
                "available": True,
                "target": view["target"],
                "orientation": view["orientation"],
                "projection": view["projection"],
                "framing": view["framing"],
                "width": width,
                "height": height,
            },
            "image_data_url": "data:image/png;base64," + base64.b64encode(pixels).decode("ascii"),
        }

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
                self._actor_viewport = None
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

    @staticmethod
    def _require_editable_mesh(name: object) -> Any:
        """Return one local single-user mesh that safe semantic editing may change."""
        obj = _object(name)
        if obj is None:
            raise WorkerActionError.validation(
                "object_not_found", "referenced object does not exist"
            )
        if obj.type != "MESH":
            raise WorkerActionError.validation(
                "requires_mesh", "operation requires an editable mesh object"
            )
        if obj.library is not None or obj.data.library is not None:
            raise WorkerActionError.validation(
                "requires_local_mesh", "operation requires local editable mesh data"
            )
        if obj.data.users != 1:
            raise WorkerActionError.validation(
                "requires_single_user_mesh", "operation requires single-user mesh data"
            )
        return obj

    @staticmethod
    def _local_bounds(coordinates: list[Vector]) -> tuple[Vector, Vector]:
        """Return center and half-extents for non-degenerate local mesh coordinates."""
        if not coordinates:
            raise WorkerActionError.execution("empty_mesh", "mesh has no vertices")
        minimum = Vector(
            tuple(min(float(coordinate[index]) for coordinate in coordinates) for index in range(3))
        )
        maximum = Vector(
            tuple(max(float(coordinate[index]) for coordinate in coordinates) for index in range(3))
        )
        half_extent = (maximum - minimum) * 0.5
        if any(value <= REGION_EPSILON for value in half_extent):
            raise WorkerActionError.execution(
                "degenerate_local_bounds",
                "mesh local bounds must have non-zero extent on every axis",
            )
        return (minimum + maximum) * 0.5, half_extent

    @staticmethod
    def _is_in_region(
        coordinate: Vector,
        center: Vector,
        half_extent: Vector,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
    ) -> bool:
        normalized = tuple(
            (coordinate[index] - center[index]) / half_extent[index] for index in range(3)
        )
        return all(
            lower - REGION_EPSILON <= normalized[index] <= upper + REGION_EPSILON
            for index, (lower, upper) in enumerate(zip(minimum, maximum, strict=True))
        )

    @staticmethod
    def _assert_affected_count(count: int, label: str) -> None:
        if count < 1:
            raise WorkerActionError.execution("empty_region", f"{label} selection is empty")
        if count > MAX_AFFECTED_ELEMENTS:
            raise WorkerActionError.validation(
                "affected_element_limit",
                f"{label} selection exceeds maximum affected elements ({MAX_AFFECTED_ELEMENTS})",
            )

    @staticmethod
    def _assert_mesh_complexity(vertices: int, polygons: int) -> None:
        if vertices > MAX_RESULT_VERTICES:
            raise WorkerActionError.validation(
                "vertex_limit",
                f"resulting mesh exceeds maximum vertex count ({MAX_RESULT_VERTICES})",
            )
        if polygons > MAX_RESULT_POLYGONS:
            raise WorkerActionError.validation(
                "polygon_limit",
                f"resulting mesh exceeds maximum polygon count ({MAX_RESULT_POLYGONS})",
            )

    def _selected_mesh_vertices(
        self,
        obj: Any,
        region: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> tuple[list[Any], Vector, Vector]:
        mesh = obj.data
        center, half_extent = self._local_bounds([vertex.co.copy() for vertex in mesh.vertices])
        minimum, maximum = region
        vertices = [
            vertex
            for vertex in mesh.vertices
            if self._is_in_region(vertex.co, center, half_extent, minimum, maximum)
        ]
        self._assert_affected_count(len(vertices), "vertex")
        return vertices, center, half_extent

    @staticmethod
    def _mesh_result(
        obj: Any, *, affected_vertices: int = 0, affected_faces: int = 0
    ) -> dict[str, Any]:
        """Return compact post-operation topology metadata for the next Actor request."""
        mesh = obj.data
        return {
            "object": obj.name,
            "affected_vertex_count": affected_vertices,
            "affected_face_count": affected_faces,
            "resulting_vertex_count": len(mesh.vertices),
            "resulting_polygon_count": len(mesh.polygons),
        }

    def _execute_join(self, action: dict[str, Any]) -> dict[str, Any]:
        objects = [self._require_editable_mesh(name) for name in action["objects"]]
        target = self._require_editable_mesh(action["target"])
        if target not in objects:
            raise WorkerActionError.validation(
                "invalid_join_target", "object.join target must be included in objects"
            )
        affected_vertices = sum(len(obj.data.vertices) for obj in objects)
        affected_faces = sum(len(obj.data.polygons) for obj in objects)
        self._assert_affected_count(affected_vertices + affected_faces, "object.join")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = target
        if not bpy.ops.object.join.poll():
            raise WorkerActionError.execution(
                "join_unavailable", "object.join is unavailable for the current geometry"
            )
        bpy.ops.object.join()
        if bpy.data.objects.get(target.name) is not target:
            raise WorkerInternalError(
                "join_target_not_preserved",
                "internal Blender worker failure while preserving the join target",
            )
        self._assert_mesh_complexity(len(target.data.vertices), len(target.data.polygons))
        return self._mesh_result(
            target, affected_vertices=affected_vertices, affected_faces=affected_faces
        )

    def _execute_transform_region(self, action: dict[str, Any]) -> dict[str, Any]:
        obj = self._require_editable_mesh(action["object"])
        region = _normalized_region(action["region"])
        vertices, _, half_extent = self._selected_mesh_vertices(obj, region)
        self._assert_mesh_complexity(len(obj.data.vertices), len(obj.data.polygons))
        pivot = sum((vertex.co.copy() for vertex in vertices), Vector()) / len(vertices)
        translate = _bounded_normalized_vector(action.get("translate", [0, 0, 0]), "translate")
        scale = _region_scale(action.get("scale", [1, 1, 1]), "scale")
        displacement = Vector(tuple(translate[index] * half_extent[index] for index in range(3)))
        for vertex in vertices:
            vertex.co = pivot + (vertex.co - pivot) * Vector(scale) + displacement
        obj.data.update()
        return self._mesh_result(obj, affected_vertices=len(vertices))

    def _execute_extrude_region(self, action: dict[str, Any]) -> dict[str, Any]:
        obj = self._require_editable_mesh(action["object"])
        region = _normalized_region(action["region"])
        mesh = obj.data
        self._assert_mesh_complexity(len(mesh.vertices), len(mesh.polygons))
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            center, half_extent = self._local_bounds([vertex.co.copy() for vertex in bm.verts])
            minimum, maximum = region
            faces = [
                face
                for face in bm.faces
                if self._is_in_region(
                    face.calc_center_median(), center, half_extent, minimum, maximum
                )
            ]
            self._assert_affected_count(len(faces), "face")
            selected_indexes = {face.index for face in faces}
            connected_indexes: set[int] = set()
            pending = [faces[0]]
            while pending:
                face = pending.pop()
                if face.index in connected_indexes:
                    continue
                connected_indexes.add(face.index)
                for edge in face.edges:
                    for adjacent in edge.link_faces:
                        if (
                            adjacent.index in selected_indexes
                            and adjacent.index not in connected_indexes
                        ):
                            pending.append(adjacent)
            if connected_indexes != selected_indexes:
                raise WorkerActionError.execution(
                    "disconnected_region",
                    "selected faces form multiple disconnected regions",
                )

            original_vertices = set(bm.verts)
            extruded = bmesh.ops.extrude_face_region(bm, geom=faces)
            new_vertices = [
                element
                for element in extruded["geom"]
                if isinstance(element, bmesh.types.BMVert) and element not in original_vertices
            ]
            self._assert_affected_count(len(new_vertices), "extruded vertex")
            self._assert_mesh_complexity(len(bm.verts), len(bm.faces))
            offset = _bounded_normalized_vector(action["offset"], "offset")
            displacement = Vector(tuple(offset[index] * half_extent[index] for index in range(3)))
            for vertex in new_vertices:
                vertex.co += displacement
            scale = _region_scale(action.get("scale", [1, 1, 1]), "scale")
            pivot = sum((vertex.co.copy() for vertex in new_vertices), Vector()) / len(new_vertices)
            for vertex in new_vertices:
                vertex.co = pivot + (vertex.co - pivot) * Vector(scale)
            bm.normal_update()
            bm.to_mesh(mesh)
            mesh.update()
        finally:
            bm.free()
        return self._mesh_result(
            obj, affected_vertices=len(new_vertices), affected_faces=len(faces)
        )

    def _execute_smooth_region(self, action: dict[str, Any]) -> dict[str, Any]:
        obj = self._require_editable_mesh(action["object"])
        region = _normalized_region(action["region"])
        vertices, _, _ = self._selected_mesh_vertices(obj, region)
        self._assert_mesh_complexity(len(obj.data.vertices), len(obj.data.polygons))
        selected_indices = {vertex.index for vertex in vertices}
        neighbors: dict[int, set[int]] = {index: set() for index in selected_indices}
        for edge in obj.data.edges:
            first, second = edge.vertices
            if first in selected_indices:
                neighbors[first].add(second)
            if second in selected_indices:
                neighbors[second].add(first)
        if any(not adjacent for adjacent in neighbors.values()):
            raise WorkerActionError.execution(
                "isolated_vertex_region",
                "selected vertices must have adjacent edges for smoothing",
            )
        factor = _number(action["factor"], "factor")
        if not 0 <= factor <= 1:
            raise WorkerActionError.validation("invalid_factor", "factor must be within [0, 1]")
        iterations = action["iterations"]
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise WorkerActionError.validation(
                "invalid_iterations", "iterations must be an integer"
            )
        if not 1 <= iterations <= MAX_SMOOTH_ITERATIONS:
            raise WorkerActionError.validation(
                "invalid_iterations",
                f"iterations must be within [1, {MAX_SMOOTH_ITERATIONS}]",
            )
        mesh = obj.data
        for _ in range(iterations):
            coordinates = {vertex.index: vertex.co.copy() for vertex in mesh.vertices}
            updated: dict[int, Vector] = {}
            for index, adjacent in neighbors.items():
                average = sum((coordinates[neighbor] for neighbor in adjacent), Vector()) / len(
                    adjacent
                )
                updated[index] = coordinates[index].lerp(average, factor)
            for index, coordinate in updated.items():
                mesh.vertices[index].co = coordinate
        mesh.update()
        return self._mesh_result(obj, affected_vertices=len(vertices))

    def _execute_shade_smooth(self, action: dict[str, Any]) -> dict[str, Any]:
        obj = self._require_editable_mesh(action["object"])
        self._assert_mesh_complexity(len(obj.data.vertices), len(obj.data.polygons))
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()
        return self._mesh_result(obj, affected_faces=len(obj.data.polygons))

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
            "object.join",
            "mesh.transform_region",
            "mesh.extrude_region",
            "mesh.smooth_region",
            "object.shade_smooth",
        }:
            raise WorkerError(f"unsupported command: {command!r}")
        if command == "object.create":
            _exact_action_fields(
                action,
                required={"command", "name"},
                optional={"primitive", "location", "scale"},
            )
            _name(action.get("name"))
            if action.get("primitive", "cube") not in {"cube", "uv_sphere", "cylinder", "cone"}:
                raise WorkerError("unsupported primitive")
            _vector(action.get("location", [0, 0, 0]), "location")
            scale = _vector(action.get("scale", [1, 1, 1]), "scale")
            if any(value <= 0 or value > 100 for value in scale):
                raise WorkerError("scale must be within (0, 100]")
        elif command == "object.delete":
            _exact_action_fields(action, required={"command", "object"})
            _name(action.get("object"))
        elif command == "object.translate":
            _exact_action_fields(action, required={"command", "object", "offset"})
            _name(action.get("object"))
            _vector(action.get("offset"), "offset")
        elif command == "object.rotate":
            _exact_action_fields(action, required={"command", "object", "degrees"})
            _name(action.get("object"))
            _vector(action.get("degrees"), "degrees")
        elif command == "object.scale":
            _exact_action_fields(action, required={"command", "object", "scale"})
            _name(action.get("object"))
            scale = _vector(action.get("scale"), "scale")
            if any(value <= 0 or value > 100 for value in scale):
                raise WorkerError("scale must be within (0, 100]")
        elif command == "sculpt.voxel_remesh":
            _exact_action_fields(action, required={"command", "object", "voxel_size"})
            _name(action.get("object"))
            voxel_size = _number(action.get("voxel_size"), "voxel_size")
            if not 0.001 < voxel_size <= 1.0:
                raise WorkerError("voxel_size must be within (0.001, 1]")
        elif command == "object.join":
            _exact_action_fields(action, required={"command", "objects", "target"})
            objects = action.get("objects")
            if not isinstance(objects, list) or not 2 <= len(objects) <= MAX_JOIN_OBJECTS:
                raise WorkerError(
                    f"object.join objects must contain 2 to {MAX_JOIN_OBJECTS} object names"
                )
            names = [_name(name) for name in objects]
            if len(names) != len(set(names)):
                raise WorkerError("object.join objects must be unique")
            target = _name(action.get("target"))
            if target not in names:
                raise WorkerError("object.join target must be included in objects")
        elif command == "mesh.transform_region":
            _exact_action_fields(
                action,
                required={"command", "object", "region"},
                optional={"translate", "scale"},
            )
            _name(action.get("object"))
            _normalized_region(action.get("region"))
            _bounded_normalized_vector(action.get("translate", [0, 0, 0]), "translate")
            _region_scale(action.get("scale", [1, 1, 1]), "scale")
        elif command == "mesh.extrude_region":
            _exact_action_fields(
                action,
                required={"command", "object", "region", "offset"},
                optional={"scale"},
            )
            _name(action.get("object"))
            _normalized_region(action.get("region"))
            _bounded_normalized_vector(action.get("offset"), "offset")
            _region_scale(action.get("scale", [1, 1, 1]), "scale")
        elif command == "mesh.smooth_region":
            _exact_action_fields(
                action,
                required={"command", "object", "region", "factor", "iterations"},
            )
            _name(action.get("object"))
            _normalized_region(action.get("region"))
            factor = _number(action.get("factor"), "factor")
            if not 0 <= factor <= 1:
                raise WorkerError("factor must be within [0, 1]")
            iterations = action.get("iterations")
            if not isinstance(iterations, int) or isinstance(iterations, bool):
                raise WorkerError("iterations must be an integer")
            if not 1 <= iterations <= MAX_SMOOTH_ITERATIONS:
                raise WorkerError(f"iterations must be within [1, {MAX_SMOOTH_ITERATIONS}]")
        else:
            _exact_action_fields(action, required={"command", "object"})
            _name(action.get("object"))
        return action

    def _validate_actions(self, actions: list[object]) -> list[dict[str, Any]]:
        """Attach deterministic context to action-contract failures before mutation."""
        validated: list[dict[str, Any]] = []
        for action_index, action in enumerate(actions):
            command = (
                action.get("command")
                if isinstance(action, dict) and isinstance(action.get("command"), str)
                else "unknown"
            )
            try:
                validated.append(self._validate_action(action))
            except WorkerActionError as error:
                raise error.with_action_context(
                    action_index=action_index,
                    command=command,
                    executed_before_failure=0,
                ) from error
            except WorkerError as error:
                logger.debug("worker action validation failed: %s", error)
                raise WorkerActionError.validation(
                    "invalid_action", "action does not satisfy the worker contract"
                ).with_action_context(
                    action_index=action_index,
                    command=command,
                    executed_before_failure=0,
                ) from error
        return validated

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Perform one already validated allowlisted action."""
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
            return {"object": name}
        if command == "object.delete":
            obj = _require_object(action["object"])
            self._activate(obj)
            bpy.ops.object.delete()
            return {"object": obj.name}
        if command == "object.translate":
            obj = _require_object(action["object"])
            offset = _vector(action["offset"], "offset")
            obj.location = tuple(obj.location[index] + offset[index] for index in range(3))
            return {"object": obj.name}
        if command == "object.rotate":
            obj = _require_object(action["object"])
            degrees = _vector(action["degrees"], "degrees")
            obj.rotation_euler = tuple(
                obj.rotation_euler[index] + math.radians(degrees[index]) for index in range(3)
            )
            return {"object": obj.name}
        if command == "object.scale":
            obj = _require_object(action["object"])
            scale = _vector(action["scale"], "scale")
            obj.scale = tuple(obj.scale[index] * scale[index] for index in range(3))
            return {"object": obj.name}
        if command == "sculpt.voxel_remesh":
            obj = self._require_editable_mesh(action["object"])
            self._activate(obj)
            obj.data.remesh_voxel_size = _number(action["voxel_size"], "voxel_size")
            bpy.ops.object.voxel_remesh()
            self._assert_mesh_complexity(len(obj.data.vertices), len(obj.data.polygons))
            return self._mesh_result(obj)
        if command == "object.join":
            return self._execute_join(action)
        if command == "mesh.transform_region":
            return self._execute_transform_region(action)
        if command == "mesh.extrude_region":
            return self._execute_extrude_region(action)
        if command == "mesh.smooth_region":
            return self._execute_smooth_region(action)
        if command == "object.shade_smooth":
            return self._execute_shade_smooth(action)
        raise WorkerInternalError(
            "unsupported_validated_action",
            "internal Blender worker failure while dispatching a validated action",
        )

    def execute(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
            raise WorkerError("actions payload must contain an actions array")
        actions = payload["actions"]
        if not 1 <= len(actions) <= MAX_ACTIONS_PER_BATCH:
            raise WorkerError(f"actions array must contain 1 to {MAX_ACTIONS_PER_BATCH} actions")
        executed: list[dict[str, Any]] = []
        current_action_index: int | None = None
        current_command: str | None = None
        try:
            validated = self._validate_actions(actions)
            self._preflight_actions(validated)
            self._allow_worker_selection()
            for action_index, action in enumerate(validated):
                command = action["command"]
                current_action_index = action_index
                current_command = command
                result: dict[str, Any]
                try:
                    result = self._execute_action(action)
                except WorkerActionError as error:
                    raise error.with_action_context(
                        action_index=action_index,
                        command=command,
                        executed_before_failure=len(executed),
                    ) from error
                except WorkerInternalError as error:
                    raise error.with_action_context(
                        action_index=action_index,
                        command=command,
                        executed_before_failure=len(executed),
                    ) from error
                executed.append({"command": str(command), "status": "ok", **result})
            if bpy.ops.ed.undo_push.poll():
                bpy.ops.ed.undo_push(message="Aculptoi action batch")
            return {"executed": executed}
        except WorkerActionError as error:
            raise error.mark_scene_restored(self._restore_failed_batch()) from error
        except WorkerInternalError as error:
            raise error.mark_scene_restored(self._restore_failed_batch()) from error
        except Exception as error:
            logger.exception("internal Blender worker error during action execution")
            restored = self._restore_failed_batch()
            raise WorkerInternalError(
                action_index=current_action_index,
                command=current_command,
                executed_before_failure=len(executed),
                scene_restored=restored,
            ) from error
        finally:
            self._apply_observer_guard()
            self._redraw_viewports()

    def _restore_failed_batch(self) -> bool:
        """Reload the saved canonical scene after an execution-time active-run failure."""
        if self.active_scene_path is None or not self.active_scene_path.is_file():
            return False
        try:
            bpy.ops.wm.open_mainfile(filepath=str(self.active_scene_path))
        except Exception as error:
            logger.exception("failed to restore canonical scene after action failure")
            raise WorkerInternalError(
                "rollback_failed", "canonical scene could not be restored after action failure"
            ) from error
        return True

    @staticmethod
    def _preflight_actions(actions: list[dict[str, Any]]) -> None:
        """Reject obvious batch failures before the first Blender mutation occurs."""
        available = {obj.name for obj in bpy.context.scene.objects}
        for action_index, action in enumerate(actions):
            command = action["command"]
            if command == "object.create":
                name = _name(action["name"])
                if name in available:
                    raise WorkerActionError.validation(
                        "object_already_exists", "requested object name already exists"
                    ).with_action_context(
                        action_index=action_index,
                        command=command,
                        executed_before_failure=0,
                    )
                available.add(name)
                continue
            if command == "object.join":
                names = [_name(name) for name in action["objects"]]
                target = _name(action["target"])
                missing = sorted(name for name in names if name not in available)
                if missing:
                    raise WorkerActionError.validation(
                        "object_not_found", "referenced object does not exist"
                    ).with_action_context(
                        action_index=action_index,
                        command=command,
                        executed_before_failure=0,
                    )
                for name in names:
                    if name != target:
                        available.remove(name)
                continue
            name = _name(action["object"])
            if name not in available:
                raise WorkerActionError.validation(
                    "object_not_found", "referenced object does not exist"
                ).with_action_context(
                    action_index=action_index,
                    command=command,
                    executed_before_failure=0,
                )
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
        if method == "POST" and path == "/v1/viewport/observe":
            return self.worker.observe_viewport(payload)
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
        except WorkerActionError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": error.error_payload()})
        except WorkerInternalError as error:
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": error.error_payload()},
            )
        except WorkerError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        except Exception:  # Blender can throw context-specific runtime errors.
            logger.exception("unhandled Blender worker request failure")
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": WorkerInternalError().error_payload()},
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
