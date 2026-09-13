"""Append-only, non-authoritative timing and model-usage telemetry for runs."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import fsync
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aculptoi.inference import InferenceProfile
from aculptoi.models import ModelProviderError, ModelUsage
from aculptoi.reasoning import ReasoningEffort

EventName = Literal[
    "run_started",
    "run_resumed",
    "run_completed",
    "run_interrupted",
    "run_stopped",
    "stage_completed",
    "stage_failed",
]
TelemetryStage = Literal[
    "actor_construction_plan",
    "actor_work_item",
    "inspection_camera_selection",
    "inspection_render",
    "inspection_atlas_build",
    "inspection_review",
    "critic_discovery",
    "critic_issue_analysis",
]
EventOutcome = Literal["success", "failure", "timeout"]


class RunEvent(BaseModel):
    """One independently valid, append-only observation about a run."""

    model_config = ConfigDict(extra="forbid")

    event: EventName
    run_id: int = Field(ge=1)
    timestamp: datetime
    stage: TelemetryStage | None = None
    iteration: int | None = Field(default=None, ge=0)
    outcome: EventOutcome | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    provider: str | None = Field(default=None, min_length=1, max_length=64)
    thinking: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = Field(default=None, ge=128, le=65_536)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    error_type: str | None = Field(default=None, min_length=1, max_length=120)
    error: str | None = Field(default=None, min_length=1, max_length=2_000)
    work_item_id: str | None = Field(default=None, min_length=1, max_length=120)
    inspection_round: int | None = Field(default=None, ge=1)
    issue_id: str | None = Field(default=None, min_length=1, max_length=120)
    request_number: int | None = Field(default=None, ge=1)
    active_phase: str | None = Field(default=None, min_length=1, max_length=40)


@dataclass
class RunEventStage(AbstractContextManager["RunEventStage"]):
    """Measure one expensive stage and append its result even when it fails."""

    events: RunEvents
    stage: TelemetryStage
    iteration: int
    provider: str | None
    profile: InferenceProfile | None
    context: dict[str, object]
    _started_at: datetime = field(init=False)
    _started_monotonic: float = field(init=False)
    _usage: ModelUsage | None = field(default=None, init=False)

    def __enter__(self) -> RunEventStage:
        self._started_at = datetime.now(UTC)
        self._started_monotonic = monotonic()
        return self

    def record_usage(self, usage: ModelUsage | None) -> None:
        """Associate provider-reported usage with this one measured request."""
        self._usage = usage

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception_type, traceback
        finished_at = datetime.now(UTC)
        if isinstance(exception, ModelProviderError):
            self.record_usage(exception.usage)
        event: dict[str, object] = {
            "event": "stage_completed" if exception is None else "stage_failed",
            "run_id": self.events.run_id,
            "timestamp": finished_at,
            "stage": self.stage,
            "iteration": self.iteration,
            "outcome": "success" if exception is None else _outcome_for(exception),
            "started_at": self._started_at,
            "finished_at": finished_at,
            "duration_seconds": monotonic() - self._started_monotonic,
            "provider": self.provider,
            **self.context,
        }
        if self.profile is not None:
            event["thinking"] = self.profile.thinking
            event["reasoning_effort"] = self.profile.reasoning_effort
            event["max_output_tokens"] = self.profile.max_output_tokens
        if self._usage is not None:
            event["prompt_tokens"] = self._usage.prompt_tokens
            event["completion_tokens"] = self._usage.completion_tokens
            event["reasoning_tokens"] = self._usage.reasoning_tokens
        if exception is not None:
            event["error_type"] = type(exception).__name__
            event["error"] = str(exception) or type(exception).__name__
        self.events.append(RunEvent.model_validate(event))
        return False


class RunEvents:
    """Write non-authoritative, durable run telemetry one event per JSONL line."""

    def __init__(self, run_id: int, run_path: Path) -> None:
        self.run_id = run_id
        self.path = run_path / "run-events.jsonl"

    def append_lifecycle(
        self,
        event: Literal[
            "run_started",
            "run_resumed",
            "run_completed",
            "run_interrupted",
            "run_stopped",
        ],
        *,
        iteration: int | None = None,
        active_phase: str | None = None,
    ) -> None:
        self.append(
            RunEvent(
                event=event,
                run_id=self.run_id,
                timestamp=datetime.now(UTC),
                iteration=iteration,
                active_phase=active_phase,
            )
        )

    def stage(
        self,
        stage: TelemetryStage,
        *,
        iteration: int,
        provider: str | None = None,
        profile: InferenceProfile | None = None,
        work_item_id: str | None = None,
        inspection_round: int | None = None,
        issue_id: str | None = None,
        request_number: int | None = None,
    ) -> RunEventStage:
        """Create one timing scope; event append occurs at normal or exceptional exit."""
        context: dict[str, object] = {
            "work_item_id": work_item_id,
            "inspection_round": inspection_round,
            "issue_id": issue_id,
            "request_number": request_number,
        }
        return RunEventStage(self, stage, iteration, provider, profile, context)

    def append(self, event: RunEvent) -> None:
        """Append and flush one standalone record without reading existing telemetry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            event.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(payload + "\n")
            file.flush()
            fsync(file.fileno())


def _outcome_for(exception: BaseException | None) -> EventOutcome:
    if exception is None:
        return "success"
    current: BaseException | None = exception
    while current is not None:
        if "timeout" in str(current).lower() or "timed out" in str(current).lower():
            return "timeout"
        current = current.__cause__
    return "failure"
