"""Client for the local-only HTTP interface exposed inside Blender."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import httpx

from aculptoi.config import BlenderConfig
from aculptoi.schemas.actions import Action


class BlenderWorkerError(RuntimeError):
    """A worker rejected a request or returned an unexpected error."""


class BlenderWorkerUnavailable(BlenderWorkerError):
    """No persistent Blender worker is accepting local requests."""


class BlenderClient:
    """Use a versioned HTTP transport, keeping future socket transports possible."""

    def __init__(self, config: BlenderConfig) -> None:
        self._config = config
        self._base_url = f"http://{config.host}:{config.port}/v1"

    def _request(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        try:
            with httpx.Client(timeout=self._config.timeout_seconds) as client:
                response = client.request(method, f"{self._base_url}{path}", json=payload)
        except httpx.ConnectError as error:
            raise BlenderWorkerUnavailable(
                "Blender worker is not running. Start it with `aculptoi blender start`."
            ) from error
        except httpx.HTTPError as error:
            raise BlenderWorkerError(f"Blender worker request failed: {error}") from error
        try:
            body = response.json()
        except ValueError as error:
            raise BlenderWorkerError("Blender worker returned non-JSON output") from error
        if not isinstance(body, dict):
            raise BlenderWorkerError("Blender worker returned an invalid response")
        if response.is_error or not body.get("ok", False):
            raise BlenderWorkerError(
                str(body.get("error", f"Worker returned HTTP {response.status_code}"))
            )
        data = body.get("data", {})
        if not isinstance(data, dict):
            raise BlenderWorkerError("Blender worker returned invalid response data")
        return cast(dict[str, object], data)

    def health(self) -> dict[str, object]:
        return self._request("GET", "/health")

    def scene_inspect(self) -> dict[str, object]:
        return self._request("GET", "/scene/inspect")

    def object_list(self) -> dict[str, object]:
        return self._request("GET", "/objects")

    def object_inspect(self, name: str) -> dict[str, object]:
        return self._request("GET", f"/objects/{name}")

    def execute(self, actions: Sequence[Action]) -> dict[str, object]:
        return self._request(
            "POST",
            "/actions/execute",
            {"actions": [action.model_dump(mode="json") for action in actions]},
        )

    def attach_run(
        self, scene_path: Path, run_id: int, *, reload: bool = False
    ) -> dict[str, object]:
        """Give one worker exclusive ownership of one run's canonical scene."""
        return self._request(
            "POST",
            "/run/attach",
            {"scene_path": str(scene_path), "run_id": run_id, "reload": reload},
        )

    def save_canonical_scene(self, scene_path: Path) -> dict[str, object]:
        """Persist the active run's mutable canonical scene after a successful batch."""
        return self._request("POST", "/scene/save", {"scene_path": str(scene_path)})

    def release_run(self) -> dict[str, object]:
        """Release worker ownership without changing the visible scene."""
        return self._request("POST", "/run/release", {})

    def render_views(
        self, views: Sequence[str], output_dir: Path, object_name: str | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {"views": list(views), "output_dir": str(output_dir)}
        if object_name:
            payload["object"] = object_name
        return self._request("POST", "/render/views", payload)

    def shutdown(self) -> dict[str, object]:
        return self._request("POST", "/shutdown", {})
