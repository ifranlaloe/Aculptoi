"""Client for the local-only HTTP interface exposed inside Blender."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from pydantic import ValidationError

from aculptoi.config import BlenderConfig
from aculptoi.schemas.actions import Action
from aculptoi.schemas.execution import ActionExecutionFailure
from aculptoi.schemas.inspection import CameraCandidate, InspectionCameraPlan
from aculptoi.schemas.viewport import ViewportObservation, ViewportView


class BlenderWorkerError(RuntimeError):
    """A worker rejected a request or returned an unexpected error."""


class BlenderWorkerUnavailable(BlenderWorkerError):
    """No persistent Blender worker is accepting local requests."""


class BlenderActionError(BlenderWorkerError):
    """A worker-returned structured action failure, whether recoverable or fatal."""

    def __init__(self, failure: ActionExecutionFailure) -> None:
        self.failure = failure
        super().__init__(f"{failure.failure_kind} during {failure.command}: {failure.message}")


@dataclass(frozen=True)
class ViewportCapture:
    """One transient Actor observation returned in the local HTTP response."""

    observation: ViewportObservation
    image_data_url: str


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
            raise self._response_error(body, response.status_code)
        data = body.get("data", {})
        if not isinstance(data, dict):
            raise BlenderWorkerError("Blender worker returned invalid response data")
        return cast(dict[str, object], data)

    @staticmethod
    def _response_error(body: dict[str, object], status_code: int) -> BlenderWorkerError:
        """Decode only the bounded public error contract exposed by the worker."""
        error = body.get("error")
        if isinstance(error, dict):
            try:
                return BlenderActionError(ActionExecutionFailure.model_validate(error))
            except ValidationError:
                return BlenderWorkerError("Blender worker returned an invalid error response")
        if isinstance(error, str):
            return BlenderWorkerError(error[:500])
        return BlenderWorkerError(f"Worker returned HTTP {status_code}")

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

    def observe_viewport(self, view: ViewportView) -> ViewportCapture:
        """Capture only Aculptoi's reserved UI viewport, never the desktop or a render."""
        data = self._request("POST", "/viewport/observe", {"view": view.model_dump(mode="json")})
        try:
            observation = ViewportObservation.model_validate(data.get("observation"))
        except ValidationError as error:
            raise BlenderWorkerError("Blender worker returned invalid viewport metadata") from error
        image_data_url = data.get("image_data_url")
        if not isinstance(image_data_url, str) or not image_data_url.startswith(
            "data:image/png;base64,"
        ):
            raise BlenderWorkerError("Blender worker returned an invalid viewport image")
        return ViewportCapture(observation=observation, image_data_url=image_data_url)

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

    def analyze_inspection_candidates(
        self, candidates: Sequence[CameraCandidate]
    ) -> dict[str, object]:
        """Measure low-cost camera diagnostics without persisting candidate renders."""
        return self._request(
            "POST",
            "/inspection/candidates/analyze",
            {"candidates": [candidate.model_dump(mode="json") for candidate in candidates]},
        )

    def render_inspection_views(
        self, plan: InspectionCameraPlan, output_dir: Path
    ) -> dict[str, object]:
        """Render final selected inspection shots in an isolated worker-owned environment."""
        return self._request(
            "POST",
            "/inspection/render",
            {"plan": plan.model_dump(mode="json"), "output_dir": str(output_dir)},
        )

    def shutdown(self) -> dict[str, object]:
        return self._request("POST", "/shutdown", {})
