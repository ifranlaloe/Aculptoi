"""Unit coverage for the bounded structured Blender worker error transport."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aculptoi.blender.client import BlenderActionError, BlenderClient, BlenderWorkerError
from aculptoi.schemas.execution import ActionExecutionFailure


def _failure_payload(**updates: object) -> dict[str, object]:
    return {
        "failure_kind": "execution_error",
        "code": "disconnected_region",
        "message": "selected faces form multiple disconnected regions",
        "action_index": 1,
        "command": "mesh.extrude_region",
        "executed_before_failure": 1,
        "recoverable": True,
        "rolled_back": True,
        "scene_restored": True,
    } | updates


def test_client_decodes_structured_recoverable_action_error() -> None:
    error = BlenderClient._response_error({"error": _failure_payload()}, 400)

    assert isinstance(error, BlenderActionError)
    assert error.failure.failure_kind == "execution_error"
    assert error.failure.code == "disconnected_region"
    assert error.failure.action_index == 1
    assert error.failure.executed_before_failure == 1
    assert error.failure.recoverable is True


def test_client_keeps_fatal_worker_error_distinct_and_bounded() -> None:
    error = BlenderClient._response_error(
        {
            "error": _failure_payload(
                failure_kind="worker_error",
                code="internal_worker_error",
                message="internal Blender worker failure",
                recoverable=False,
                rolled_back=True,
                scene_restored=True,
            )
        },
        500,
    )

    assert isinstance(error, BlenderActionError)
    assert error.failure.failure_kind == "worker_error"
    assert error.failure.recoverable is False
    assert "Traceback" not in error.failure.message


def test_client_rejects_malformed_structured_error_payload() -> None:
    error = BlenderClient._response_error({"error": {"failure_kind": "execution_error"}}, 400)

    assert type(error) is BlenderWorkerError
    assert str(error) == "Blender worker returned an invalid error response"


@pytest.mark.parametrize(
    "updates",
    [
        {"failure_kind": "worker_error", "recoverable": True},
        {"rolled_back": False, "scene_restored": True},
    ],
)
def test_failure_schema_rejects_incoherent_recovery_metadata(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ActionExecutionFailure.model_validate(_failure_payload(**updates))
