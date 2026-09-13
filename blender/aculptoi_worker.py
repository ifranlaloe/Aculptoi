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
import threading
import uuid
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]{0,63}$")
RUN_ID_RE = re.compile(r"^[0-9]{6}$")
ALLOWED_VIEWS = {"front", "right", "top", "perspective"}
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
