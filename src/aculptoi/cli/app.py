"""CLI-first interface for inspecting and safely operating the local worker."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from aculptoi import __version__
from aculptoi.agent import Actor, RefinementLoop, VisionCritic
from aculptoi.blender import BlenderClient, BlenderWorkerError
from aculptoi.checkpoints import CheckpointStore
from aculptoi.config import AcuConfig, default_config_path, load_config
from aculptoi.models import ModelProviderError, ProviderRegistry
from aculptoi.runtime import LlamaServeConfig

app = typer.Typer(
    name="aculptoi",
    help="Local AI that sees, builds, and refines in Blender.",
    no_args_is_help=True,
)
blender_app = typer.Typer(help="Manage the persistent local Blender worker.", no_args_is_help=True)
scene_app = typer.Typer(help="Inspect the current Blender scene.", no_args_is_help=True)
object_app = typer.Typer(help="List and inspect Blender objects.", no_args_is_help=True)
render_app = typer.Typer(help="Render inspection views.", no_args_is_help=True)
checkpoint_app = typer.Typer(help="Manage Blender scene checkpoints.", no_args_is_help=True)
config_app = typer.Typer(help="View configuration.", no_args_is_help=True)
model_app = typer.Typer(
    help="Optionally start a local llama.cpp server for user-supplied weights.",
    no_args_is_help=True,
)
app.add_typer(blender_app, name="blender")
app.add_typer(scene_app, name="scene")
app.add_typer(object_app, name="object")
app.add_typer(render_app, name="render")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")

JsonOption = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]


@dataclass
class Runtime:
    config: AcuConfig
    project_dir: Path

    @property
    def blender(self) -> BlenderClient:
        return BlenderClient(self.config.blender)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False)] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Set shared project configuration without making network calls."""
    try:
        context.obj = Runtime(load_config(config), Path.cwd())
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(f"Could not load configuration: {error}") from error
    _configure_logging(verbose)


def _runtime(context: typer.Context) -> Runtime:
    return context.ensure_object(Runtime)


def _emit(data: Any, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))
    elif isinstance(data, str):
        typer.echo(data)
    else:
        typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


def _require_hf_home(json_output: bool) -> Path:
    """Require an explicit Hugging Face cache root before launching llama.cpp."""
    value = os.environ.get("HF_HOME")
    if not value:
        message = (
            "HF_HOME is not set. Set it to your Hugging Face cache directory, "
            "for example: export HF_HOME=/absolute/path/to/huggingface; then rerun this command."
        )
        if json_output:
            _emit({"ok": False, "error": message}, True)
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1)
    path = Path(value).expanduser()
    if not path.is_dir():
        message = f"HF_HOME is not an existing directory: {path}"
        if json_output:
            _emit({"ok": False, "error": message}, True)
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1)
    return path


def _require_readable_file(path: Path, option: str, json_output: bool) -> Path:
    """Validate user-selected model artifacts without interpreting them as code."""
    if path.is_file() and os.access(path, os.R_OK):
        return path
    message = f"{option} must name a readable file: {path}"
    if json_output:
        _emit({"ok": False, "error": message}, True)
    else:
        typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _worker_call(json_output: bool, call: Any) -> None:
    try:
        _emit(call(), json_output)
    except BlenderWorkerError as error:
        if json_output:
            _emit({"ok": False, "error": str(error)}, True)
        else:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


def _pid_file(project_dir: Path) -> Path:
    return project_dir / ".aculptoi" / "blender-worker.json"


def _worker_script(project_dir: Path) -> Path:
    candidate = project_dir / "blender" / "aculptoi_worker.py"
    if candidate.exists():
        return candidate
    raise RuntimeError("Could not find blender/aculptoi_worker.py; run from an Aculptoi checkout.")


def _build_loop(runtime: Runtime) -> RefinementLoop:
    """Construct the explicit V1 loop with role-selected cached providers."""
    providers = ProviderRegistry(runtime.config.providers)
    return RefinementLoop(
        actor=Actor(providers.get(runtime.config.actor.provider)),
        critic=VisionCritic(
            providers.get(runtime.config.vision.provider),
            max_image_dimension=runtime.config.vision.max_image_dimension,
        ),
        blender=runtime.blender,
        checkpoints=CheckpointStore(runtime.project_dir),
        max_iterations=runtime.config.max_iterations,
        score_target=runtime.config.score_target,
    )


def _latest_checkpoint_context(store: CheckpointStore) -> tuple[str, str] | None:
    """Find the most recent recoverable goal and worker checkpoint metadata."""
    candidates = sorted(store.runs.glob("*/checkpoint-*.json"), reverse=True)
    for candidate in candidates:
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        snapshot = metadata.get("scene_snapshot")
        goal = metadata.get("goal")
        if isinstance(goal, str) and isinstance(snapshot, dict):
            name = snapshot.get("name")
            if isinstance(name, str):
                return goal, name
    return None


@app.command()
def doctor(context: typer.Context, json_output: JsonOption = False) -> None:
    """Check local runtime prerequisites without changing Blender state."""
    runtime = _runtime(context)
    blender_path = Path(runtime.config.blender.executable)
    blender_available = (
        blender_path.exists() or shutil.which(runtime.config.blender.executable) is not None
    )
    checks: dict[str, Any] = {
        "python": {
            "ok": True,
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "project_directory": {"ok": runtime.project_dir.exists(), "path": str(runtime.project_dir)},
        "blender": {"ok": blender_available, "executable": runtime.config.blender.executable},
    }
    providers = ProviderRegistry(runtime.config.providers)
    provider_checks: dict[str, dict[str, Any]] = {}
    for name, provider_config in runtime.config.providers.items():
        try:
            models = providers.get(name).healthcheck()
            provider_checks[name] = {
                "ok": True,
                "base_url": provider_config.base_url,
                "models": models,
            }
        except ModelProviderError as error:
            provider_checks[name] = {
                "ok": False,
                "base_url": provider_config.base_url,
                "error": str(error),
            }
    checks["providers"] = provider_checks
    for role in ("actor", "vision"):
        role_config = runtime.config.actor if role == "actor" else runtime.config.vision
        provider_check = provider_checks[role_config.provider]
        checks[role] = {
            "ok": provider_check["ok"],
            "provider": role_config.provider,
            "base_url": runtime.config.provider_for(role).base_url,
        }
    try:
        worker = runtime.blender.health()
        checks["blender_worker"] = {"ok": True, **worker}
    except BlenderWorkerError as error:
        checks["blender_worker"] = {"ok": False, "error": str(error)}
    checks["ok"] = all(
        item["ok"] for item in checks.values() if isinstance(item, dict) and "ok" in item
    )
    _emit(checks, json_output)
    if not checks["ok"]:
        raise typer.Exit(1)


@config_app.command("show")
def config_show(context: typer.Context, json_output: JsonOption = False) -> None:
    """Show effective configuration, including defaults when no TOML exists."""
    runtime = _runtime(context)
    _emit(
        {
            "config_path": str(default_config_path(runtime.project_dir)),
            **runtime.config.model_dump(mode="json"),
        },
        json_output,
    )


@model_app.command("serve")
def llama_serve(
    context: typer.Context,
    model: Annotated[
        Path,
        typer.Option("--model", "-m", file_okay=True, dir_okay=False),
    ],
    mmproj: Annotated[
        Path,
        typer.Option("--mmproj", file_okay=True, dir_okay=False),
    ],
    context_size: Annotated[
        int,
        typer.Option("--context-size", "-c", min=512, max=131_072),
    ] = 32_768,
    port: Annotated[int, typer.Option(min=1024, max=65535)] = 8080,
    alias: Annotated[str, typer.Option("--alias", "-a")] = "aculptoi",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the validated command without starting a server."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Start a foreground, localhost-only llama.cpp multimodal server.

    This is an operator command, not a capability available to model output.
    Child output moves to stderr under ``--json`` so stdout remains valid JSON.
    """
    del context
    hf_home = _require_hf_home(json_output)
    model = _require_readable_file(model, "--model", json_output)
    mmproj = _require_readable_file(mmproj, "--mmproj", json_output)
    try:
        server = LlamaServeConfig(
            model_path=model,
            mmproj_path=mmproj,
            context_size=context_size,
            port=port,
            alias=alias,
        )
    except ValidationError as error:
        if json_output:
            _emit({"ok": False, "error": str(error)}, True)
        else:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error

    command = server.command()
    details = {
        "command": command,
        "host": server.host,
        "port": server.port,
        "alias": server.alias,
        "hf_home": str(hf_home),
        "foreground": True,
    }
    if dry_run:
        _emit({"ok": True, **details, "dry_run": True}, json_output)
        return
    if shutil.which(command[0]) is None:
        message = "Could not find `llama` on PATH. Install llama.cpp or update PATH."
        if json_output:
            _emit({"ok": False, "error": message, **details}, True)
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1)

    if json_output:
        _emit({"ok": True, **details, "dry_run": False}, True)
    else:
        typer.echo(
            f"Starting llama.cpp on {server.host}:{server.port}; press Ctrl-C to stop.", err=True
        )
    try:
        result = subprocess.run(
            command,
            stdout=sys.stderr if json_output else None,
            check=False,
        )
    except OSError as error:
        message = f"Could not start llama.cpp: {error}"
        if json_output:
            _emit({"ok": False, "error": message, **details}, True)
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1) from error
    if result.returncode:
        raise typer.Exit(result.returncode)


@blender_app.command("start")
def blender_start(context: typer.Context, json_output: JsonOption = False) -> None:
    """Start one persistent, localhost-only Blender background worker."""
    runtime = _runtime(context)
    try:
        _emit({"already_running": True, **runtime.blender.health()}, json_output)
        return
    except BlenderWorkerError:
        pass
    executable = runtime.config.blender.executable
    if not (Path(executable).exists() or shutil.which(executable)):
        typer.echo(f"Error: Blender executable not found: {executable}", err=True)
        raise typer.Exit(1)
    try:
        script = _worker_script(runtime.project_dir)
    except RuntimeError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    state_file = _pid_file(runtime.project_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = state_file.parent / "blender-worker.log"
    with log_file.open("ab") as log:
        process = subprocess.Popen(
            [
                executable,
                "--background",
                "--python",
                str(script),
                "--",
                "--host",
                runtime.config.blender.host,
                "--port",
                str(runtime.config.blender.port),
            ],
            cwd=runtime.project_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state_file.write_text(
        json.dumps({"pid": process.pid, "log": str(log_file)}) + "\n", encoding="utf-8"
    )
    for _ in range(30):
        time.sleep(0.25)
        try:
            _emit({"started": True, "pid": process.pid, **runtime.blender.health()}, json_output)
            return
        except BlenderWorkerError:
            if process.poll() is not None:
                break
    typer.echo(f"Error: Worker failed to start; see {log_file}", err=True)
    raise typer.Exit(1)


@blender_app.command("status")
def blender_status(context: typer.Context, json_output: JsonOption = False) -> None:
    """Report worker health and the last locally recorded PID."""
    runtime = _runtime(context)
    recorded: dict[str, Any] = {}
    with suppress(OSError, json.JSONDecodeError):
        recorded = json.loads(_pid_file(runtime.project_dir).read_text(encoding="utf-8"))
    _worker_call(
        json_output, lambda: {"running": True, "recorded": recorded, **runtime.blender.health()}
    )


@blender_app.command("stop")
def blender_stop(context: typer.Context, json_output: JsonOption = False) -> None:
    """Ask the worker to shut down; it never exposes process-control to models."""
    runtime = _runtime(context)
    _worker_call(json_output, runtime.blender.shutdown)
    _pid_file(runtime.project_dir).unlink(missing_ok=True)


@scene_app.command("inspect")
def scene_inspect(context: typer.Context, json_output: JsonOption = False) -> None:
    """Return a structured scene summary from the persistent worker."""
    _worker_call(json_output, _runtime(context).blender.scene_inspect)


@object_app.command("list")
def object_list(context: typer.Context, json_output: JsonOption = False) -> None:
    """List scene objects in structured form."""
    _worker_call(json_output, _runtime(context).blender.object_list)


@object_app.command("inspect")
def object_inspect(context: typer.Context, name: str, json_output: JsonOption = False) -> None:
    """Inspect one named scene object."""
    _worker_call(json_output, lambda: _runtime(context).blender.object_inspect(name))


@render_app.command("views")
def render_views(
    context: typer.Context,
    object_name: Annotated[str | None, typer.Argument(help="Optional target object name.")] = None,
    views: Annotated[
        str, typer.Option(help="Comma-separated front,right,top,perspective views.")
    ] = "front,right,top,perspective",
    json_output: JsonOption = False,
) -> None:
    """Render a safe multi-view inspection set under `.aculptoi/runs`."""
    runtime = _runtime(context)
    view_names = tuple(view.strip() for view in views.split(",") if view.strip())
    store = CheckpointStore(runtime.project_dir)
    run = store.create_run()
    _worker_call(
        json_output,
        lambda: {"run": run.id, **runtime.blender.render_views(view_names, run.path, object_name)},
    )


@checkpoint_app.command("list")
def checkpoint_list(context: typer.Context, json_output: JsonOption = False) -> None:
    """List .blend snapshots saved by the local worker."""
    _emit(
        {"checkpoints": CheckpointStore(_runtime(context).project_dir).list_checkpoints()},
        json_output,
    )


@checkpoint_app.command("save")
def checkpoint_save(context: typer.Context, name: str, json_output: JsonOption = False) -> None:
    """Save a Blender .blend checkpoint with a validated simple name."""
    _worker_call(json_output, lambda: _runtime(context).blender.checkpoint_save(name))


@checkpoint_app.command("restore")
def checkpoint_restore(context: typer.Context, name: str, json_output: JsonOption = False) -> None:
    """Restore a previously saved Blender .blend checkpoint."""
    _worker_call(json_output, lambda: _runtime(context).blender.checkpoint_restore(name))


@app.command()
def run(context: typer.Context, goal: str, json_output: JsonOption = False) -> None:
    """Run the bounded actor → Blender → vision refinement loop."""
    runtime = _runtime(context)
    try:
        result = _build_loop(runtime).run(goal)
    except (
        BlenderWorkerError,
        ModelProviderError,
        ValidationError,
        OSError,
        RuntimeError,
    ) as error:
        if json_output:
            _emit({"ok": False, "error": str(error)}, True)
        else:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    _emit(
        {
            "completed": result.completed,
            "iterations": result.iterations,
            "final_score": result.final_score,
            "run_directory": str(result.run_directory),
        },
        json_output,
    )


@app.command()
def create(context: typer.Context, goal: str, json_output: JsonOption = False) -> None:
    """Convenience alias for `run` with a construction goal."""
    run(context, goal, json_output)


@app.command()
def refine(context: typer.Context, json_output: JsonOption = False) -> None:
    """Restore the latest recorded checkpoint and continue its goal."""
    runtime = _runtime(context)
    checkpoint = _latest_checkpoint_context(CheckpointStore(runtime.project_dir))
    if checkpoint is None:
        message = "No recoverable run checkpoint found under .aculptoi/runs."
        if json_output:
            _emit({"ok": False, "error": message}, True)
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1)
    goal, name = checkpoint
    try:
        runtime.blender.checkpoint_restore(name)
    except BlenderWorkerError as error:
        if json_output:
            _emit({"ok": False, "error": str(error)}, True)
        else:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    run(context, goal, json_output)


@app.command()
def version() -> None:
    """Print the installed Aculptoi version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
