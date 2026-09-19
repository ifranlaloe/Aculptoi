"""Headless Blender smoke coverage for the bounded semantic modeling action surface."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def _blender_executable() -> Path:
    configured = os.environ.get("ACULPTOI_BLENDER_EXECUTABLE")
    return Path(configured) if configured else DEFAULT_BLENDER


def test_semantic_modeling_actions_and_active_run_rollback(tmp_path: Path) -> None:
    """Exercise worker-only geometry semantics in Blender without a model endpoint."""
    blender = _blender_executable()
    if not blender.is_file():
        pytest.skip("set ACULPTOI_BLENDER_EXECUTABLE to run Blender worker integration tests")

    worker = PROJECT_ROOT / "blender" / "aculptoi_worker.py"
    project = tmp_path / "project"
    script = f"""
import importlib.util
from pathlib import Path
import os

worker_path = Path({str(worker)!r})
project = Path({str(project)!r})
project.mkdir()
os.chdir(project)
spec = importlib.util.spec_from_file_location("aculptoi_worker", worker_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
WorkerError = module.WorkerError
WorkerActionError = module.WorkerActionError
WorkerInternalError = module.WorkerInternalError
worker = module.AculptoiWorker("headless")
try:
    worker.observe_viewport({{"view": {{}}}})
    raise AssertionError("headless worker must not offer an Actor UI viewport")
except WorkerError:
    pass

worker.execute({{"actions": [
    {{"command": "object.create", "name": "FishBody", "primitive": "cube"}},
    {{"command": "object.create", "name": "TailBlock", "primitive": "cube", "location": [2, 0, 0]}}
]}})
join = worker.execute({{"actions": [{{
    "command": "object.join",
    "objects": ["FishBody", "TailBlock"],
    "target": "FishBody"
}}]}})
assert "TailBlock" not in __import__("bpy").data.objects
assert join["executed"][0]["object"] == "FishBody"
try:
    worker.execute({{"actions": [{{
        "command": "mesh.extrude_region",
        "object": "FishBody",
        "region": {{"min": [-1, -1, -1], "max": [1, 1, 1]}},
        "offset": [0.4, 0, 0]
    }}]}})
    raise AssertionError("disconnected face selection should fail")
except WorkerActionError as error:
    assert error.code == "disconnected_region"
    assert error.error_payload()["failure_kind"] == "execution_error"
    assert error.error_payload()["recoverable"] is True

transform = worker.execute({{"actions": [{{
    "command": "mesh.transform_region",
    "object": "FishBody",
    "region": {{"min": [0.25, -1, -1], "max": [1, 1, 1]}},
    "translate": [0.1, 0, 0],
    "scale": [0.7, 0.8, 0.8]
}}]}})
assert transform["executed"][0]["affected_vertex_count"] > 0
assert worker._is_in_region(
    __import__("mathutils").Vector((1, 0, 0)),
    __import__("mathutils").Vector((0, 0, 0)),
    __import__("mathutils").Vector((1, 2, 3)),
    (0.9, -1, -1),
    (1, 1, 1),
)

before_extrude = len(__import__("bpy").data.objects["FishBody"].data.vertices)
extrude = worker.execute({{"actions": [{{
    "command": "mesh.extrude_region",
    "object": "FishBody",
    "region": {{"min": [0.9, -1, -1], "max": [1, 1, 1]}},
    "offset": [0.4, 0, 0],
    "scale": [0.7, 0.7, 0.7]
}}]}})
assert extrude["executed"][0]["affected_face_count"] == 1
assert len(__import__("bpy").data.objects["FishBody"].data.vertices) > before_extrude

smooth = worker.execute({{"actions": [{{
    "command": "mesh.smooth_region",
    "object": "FishBody",
    "region": {{"min": [-1, -1, -1], "max": [1, 1, 1]}},
    "factor": 0.25,
    "iterations": 2
}}]}})
assert smooth["executed"][0]["affected_vertex_count"] > 0
worker.execute({{"actions": [{{"command": "object.shade_smooth", "object": "FishBody"}}]}})
assert all(face.use_smooth for face in __import__("bpy").data.objects["FishBody"].data.polygons)

scene_path = project / ".aculptoi" / "runs" / "000001" / "scene.blend"
scene_path.parent.mkdir(parents=True)
worker.attach_run({{"scene_path": str(scene_path), "run_id": 1}})
worker.execute({{"actions": [{{"command": "object.create", "name": "Durable"}}]}})
worker.save_canonical_scene({{"scene_path": str(scene_path)}})
try:
    worker.execute({{"actions": [
        {{"command": "object.create", "name": "Transient"}},
        {{
            "command": "mesh.transform_region",
            "object": "Durable",
            "region": {{"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}
        }}
    ]}})
    raise AssertionError("empty mesh region should fail")
except WorkerError:
    pass
assert "Durable" in __import__("bpy").data.objects
assert "Transient" not in __import__("bpy").data.objects

original_region_check = worker._is_in_region
def raise_vector_type_error(*args):
    raise TypeError("Vector must be divided by a float")
worker._is_in_region = raise_vector_type_error
try:
    worker.execute({{"actions": [{{
        "command": "mesh.transform_region",
        "object": "Durable",
        "region": {{"min": [-1, -1, -1], "max": [1, 1, 1]}}
    }}]}})
    raise AssertionError("internal vector failure should be reported as fatal")
except WorkerInternalError as error:
    assert error.code == "internal_worker_error"
    assert error.error_payload()["failure_kind"] == "worker_error"
    assert error.error_payload()["recoverable"] is False
finally:
    worker._is_in_region = original_region_check
"""
    completed = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-expr",
            script,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_object_delete_is_safe_transactional_and_observer_guarded(tmp_path: Path) -> None:
    """Exercise deletion against real Blender RNA objects, including observer mode."""
    blender = _blender_executable()
    if not blender.is_file():
        pytest.skip("set ACULPTOI_BLENDER_EXECUTABLE to run Blender worker integration tests")

    worker = PROJECT_ROOT / "blender" / "aculptoi_worker.py"
    project = tmp_path / "project"
    script = f"""
import importlib.util
from pathlib import Path
import os

worker_path = Path({str(worker)!r})
project = Path({str(project)!r})
project.mkdir()
os.chdir(project)
spec = importlib.util.spec_from_file_location("aculptoi_worker", worker_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bpy = __import__("bpy")
worker = module.AculptoiWorker("headless")

# A single delete must return the cached name without dereferencing removed RNA.
assert "Cube" in bpy.data.objects
single = worker.execute({{"actions": [{{"command": "object.delete", "object": "Cube"}}]}})
assert single["executed"] == [{{"command": "object.delete", "status": "ok", "object": "Cube"}}]
assert "Cube" not in bpy.data.objects

# Recreate the default name, then reproduce run 000008's create + delete transaction.
bpy.ops.mesh.primitive_cube_add()
bpy.context.view_layer.objects.active.name = "Cube"
pattern = worker.execute({{"actions": [
    {{
        "command": "object.create", "name": "FishBody", "primitive": "uv_sphere",
        "location": [0.0, 0.0, 0.0], "scale": [3.0, 1.0, 0.8]
    }},
    {{"command": "object.delete", "object": "Cube"}}
]}})
assert [entry["command"] for entry in pattern["executed"]] == ["object.create", "object.delete"]
assert pattern["executed"][1]["object"] == "Cube"
assert "FishBody" in bpy.data.objects
assert "Cube" not in bpy.data.objects

# An unavailable Blender context must be a bounded recoverable execution failure.
bpy.ops.mesh.primitive_cube_add()
bpy.context.view_layer.objects.active.name = "PollProbe"
bpy.ops.object.mode_set(mode="EDIT")
original_activate = worker._activate
worker._activate = lambda obj: None
try:
    worker.execute({{"actions": [{{"command": "object.delete", "object": "PollProbe"}}]}})
    raise AssertionError("object.delete should be unavailable in Edit Mode")
except module.WorkerActionError as error:
    assert error.code == "delete_unavailable"
    payload = error.error_payload()
    assert payload["failure_kind"] == "execution_error"
    assert payload["recoverable"] is True
finally:
    worker._activate = original_activate
    bpy.ops.object.mode_set(mode="OBJECT")
assert "PollProbe" in bpy.data.objects

# UI mode temporarily enables selection for mutation, then restores observer protection.
ui_worker = module.AculptoiWorker("ui")
ui_worker.execute({{"actions": [{{
    "command": "object.create", "name": "ObserverSentinel", "primitive": "cube"
}}]}})
ui_delete = ui_worker.execute({{"actions": [{{
    "command": "object.delete", "object": "FishBody"
}}]}})
assert ui_delete["executed"][0]["object"] == "FishBody"
assert "FishBody" not in bpy.data.objects
assert bpy.data.objects["ObserverSentinel"].hide_select is True
assert bpy.context.scene.get("aculptoi_observer_mode") is True
"""
    completed = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-expr",
            script,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_subdivision_and_voxel_reconstruction_behave_observably(tmp_path: Path) -> None:
    """Exercise density, fusion, and no-change handling in a real Blender process."""
    blender = _blender_executable()
    if not blender.is_file():
        pytest.skip("set ACULPTOI_BLENDER_EXECUTABLE to run Blender worker integration tests")

    worker = PROJECT_ROOT / "blender" / "aculptoi_worker.py"
    project = tmp_path / "project"
    script = f"""
import importlib.util
from pathlib import Path
import os

worker_path = Path({str(worker)!r})
project = Path({str(project)!r})
project.mkdir()
os.chdir(project)
spec = importlib.util.spec_from_file_location("aculptoi_worker", worker_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bpy = __import__("bpy")
worker = module.AculptoiWorker("headless")

worker.execute({{"actions": [
    {{"command": "object.create", "name": "Dense", "primitive": "cube"}}
]}})
dense = bpy.data.objects["Dense"]
before_counts = (len(dense.data.vertices), len(dense.data.polygons))
before_dimensions = tuple(dense.dimensions)
before_scale = tuple(dense.scale)
result = worker.execute({{"actions": [{{
    "command": "mesh.subdivide", "object": "Dense", "cuts": 1
}}]}})
assert result["executed"][0]["resulting_vertex_count"] > before_counts[0]
assert result["executed"][0]["resulting_polygon_count"] > before_counts[1]
assert tuple(dense.dimensions) == before_dimensions
assert tuple(dense.scale) == before_scale

old_vertex_limit = module.MAX_RESULT_VERTICES
module.MAX_RESULT_VERTICES = len(dense.data.vertices)
try:
    worker.execute({{"actions": [{{
        "command": "mesh.subdivide", "object": "Dense", "cuts": 1
    }}]}})
    raise AssertionError("subdivision beyond complexity limit should fail")
except module.WorkerActionError as error:
    assert error.code == "vertex_limit"
finally:
    module.MAX_RESULT_VERTICES = old_vertex_limit

worker.execute({{"actions": [
    {{"command": "object.create", "name": "Rebuilt", "primitive": "cube"}}
]}})
rebuilt = bpy.data.objects["Rebuilt"]
before_rebuild = (len(rebuilt.data.vertices), len(rebuilt.data.polygons))
worker.execute({{"actions": [{{
    "command": "sculpt.voxel_remesh", "object": "Rebuilt", "voxel_size": 0.2
}}]}})
assert len(rebuilt.data.vertices) > before_rebuild[0]
assert len(rebuilt.data.polygons) > before_rebuild[1]

worker.execute({{"actions": [
    {{
        "command": "object.create", "name": "FuseA", "primitive": "uv_sphere",
        "location": [-0.5, 0, 0]
    }},
    {{
        "command": "object.create", "name": "FuseB", "primitive": "uv_sphere",
        "location": [0.5, 0, 0]
    }}
]}})
worker.execute({{"actions": [{{
    "command": "object.join", "objects": ["FuseA", "FuseB"], "target": "FuseA"
}}]}})
fused = bpy.data.objects["FuseA"]
joined_polygon_count = len(fused.data.polygons)
worker.execute({{"actions": [{{
    "command": "sculpt.voxel_remesh", "object": "FuseA", "voxel_size": 0.2
}}]}})
assert len(fused.data.polygons) != joined_polygon_count
bm = module.bmesh.new()
try:
    bm.from_mesh(fused.data)
    remaining = set(bm.faces)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            face = pending.pop()
            for edge in face.edges:
                for linked in edge.link_faces:
                    if linked in remaining:
                        remaining.remove(linked)
                        pending.append(linked)
    assert components == 1
finally:
    bm.free()

original_fingerprint = worker._mesh_fingerprint
worker._mesh_fingerprint = lambda obj: "unchanged"
try:
    worker.execute({{"actions": [{{
        "command": "sculpt.voxel_remesh", "object": "Rebuilt", "voxel_size": 0.2
    }}]}})
    raise AssertionError("a voxel-remesh no-change must not report success")
except module.WorkerActionError as error:
    assert error.code == "no_topology_change"
    assert error.error_payload()["failure_kind"] == "execution_error"
finally:
    worker._mesh_fingerprint = original_fingerprint
"""
    completed = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-expr",
            script,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
