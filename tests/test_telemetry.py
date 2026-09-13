from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from aculptoi.inference import InferenceProfile
from aculptoi.models import ModelProviderError, ModelUsage
from aculptoi.telemetry import RunEvents


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage_telemetry_appends_usage_and_timezone_aware_timing(tmp_path: Path) -> None:
    events = RunEvents(7, tmp_path)
    events.append_lifecycle("run_started", iteration=0)
    with events.stage(
        "critic_discovery",
        iteration=1,
        provider="local",
        profile=InferenceProfile(reasoning_effort="low", max_output_tokens=4_096),
    ) as stage:
        stage.record_usage(
            ModelUsage(prompt_tokens=14_119, completion_tokens=184, reasoning_tokens=730)
        )

    records = _events(events.path)
    assert [record["event"] for record in records] == ["run_started", "stage_completed"]
    stage_record = records[1]
    assert stage_record["stage"] == "critic_discovery"
    assert stage_record["outcome"] == "success"
    assert stage_record["duration_seconds"] >= 0
    assert stage_record["reasoning_effort"] == "low"
    assert stage_record["max_output_tokens"] == 4_096
    assert stage_record["prompt_tokens"] == 14_119
    assert stage_record["completion_tokens"] == 184
    assert stage_record["reasoning_tokens"] == 730
    assert datetime.fromisoformat(str(stage_record["timestamp"])).tzinfo is not None
    assert datetime.fromisoformat(str(stage_record["started_at"])).tzinfo is not None
    assert datetime.fromisoformat(str(stage_record["finished_at"])).tzinfo is not None


def test_stage_failure_is_appended_and_original_exception_propagates(tmp_path: Path) -> None:
    events = RunEvents(8, tmp_path)

    with (
        pytest.raises(ModelProviderError, match="timed out"),
        events.stage(
            "critic_discovery",
            iteration=1,
            provider="local",
            profile=InferenceProfile(reasoning_effort="low", max_output_tokens=4_096),
        ),
    ):
        raise ModelProviderError("Local model request timed out")

    record = _events(events.path)[0]
    assert record["event"] == "stage_failed"
    assert record["outcome"] == "timeout"
    assert record["error_type"] == "ModelProviderError"
    assert record["error"] == "Local model request timed out"
    assert "prompt_tokens" not in record
    assert "completion_tokens" not in record
    assert "reasoning_tokens" not in record


def test_telemetry_appends_existing_records_without_rewriting_them(tmp_path: Path) -> None:
    events = RunEvents(9, tmp_path)
    events.append_lifecycle("run_resumed", iteration=1, active_phase="critic")
    original = events.path.read_text(encoding="utf-8")
    events.append_lifecycle("run_interrupted", iteration=1, active_phase="critic")

    assert events.path.read_text(encoding="utf-8").startswith(original)
    records = _events(events.path)
    assert [record["event"] for record in records] == ["run_resumed", "run_interrupted"]
