"""Persistent localhost-only Aculptoi worker, executed by Blender's Python.

This module intentionally uses only Blender and Python standard-library APIs. It
does not receive or execute model-generated Python, shell commands, or paths.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]{0,63}$")
ALLOWED_VIEWS = {"front", "right", "top", "perspective"}
MAX_REQUEST_BYTES = 1_000_000


class WorkerError(ValueError):
    """A request did not meet the worker's independent safety requirements."""


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

    def __init__(self) -> None:
        self.project_root = Path.cwd().resolve()
        self.artifact_root = (self.project_root / ".aculptoi").resolve()

    def health(self) -> dict[str, Any]:
        return {"blender_version": bpy.app.version_string, "project_root": str(self.project_root)}

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
        executed: list[dict[str, str]] = []
        for action in validated:
            command = action["command"]
            if command == "object.create":
                name = _name(action["name"])
                if _object(name) is not None:
                    raise WorkerError(f"object already exists: {name}")
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
                    obj.rotation_euler[index] + math.radians(degrees[index]) for index in range(3)
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
        if bpy.ops.ed.undo_push.poll():
            bpy.ops.ed.undo_push(message="Aculptoi action batch")
        return {"executed": executed}

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

    def checkpoint_save(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WorkerError("checkpoint payload must be an object")
        name = _name(payload.get("name"))
        self.artifact_root.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        path = self.artifact_root / "checkpoints" / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
        return {"name": name, "path": str(path)}

    def checkpoint_restore(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WorkerError("checkpoint payload must be an object")
        name = _name(payload.get("name"))
        path = self.artifact_root / "checkpoints" / f"{name}.blend"
        if not path.is_file():
            raise WorkerError(f"checkpoint not found: {name}")
        bpy.ops.wm.open_mainfile(filepath=str(path))
        return {"name": name, "path": str(path)}


class Handler(BaseHTTPRequestHandler):
    """HTTP adapter with explicit routes and JSON-only responses."""

    server: HTTPServer

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
        if method == "POST" and path == "/v1/render/views":
            return self.worker.render_views(payload)
        if method == "POST" and path == "/v1/checkpoints/save":
            return self.worker.checkpoint_save(payload)
        if method == "POST" and path == "/v1/checkpoints/restore":
            return self.worker.checkpoint_restore(payload)
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
            data = self._handle(method, urlparse(self.path).path, payload)
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
    options = parser.parse_args(arguments)
    server = HTTPServer((options.host, options.port), Handler)
    server.worker = AculptoiWorker()  # type: ignore[attr-defined]
    server.stopping = False  # type: ignore[attr-defined]
    server.timeout = 0.5
    print(f"[aculptoi-worker] listening on http://{options.host}:{options.port}", flush=True)
    try:
        while not server.stopping:  # type: ignore[attr-defined]
            server.handle_request()
    finally:
        server.server_close()
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
