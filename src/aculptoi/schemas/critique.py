"""Read-only typed contracts for discovery, analysis, and assembled visual critique."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IssueId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="A stable lowercase kebab-case visual issue identifier.",
    ),
]
ViewName = Annotated[str, Field(min_length=1, max_length=64)]
Severity = Literal["critical", "high", "medium", "low"]
DetailStatus = Literal["detailed", "summary_only", "analysis_failed"]


class VisualIssueSummary(BaseModel):
    """A compact discovery observation, immutable for one critique cycle."""

    model_config = ConfigDict(extra="forbid")

    id: IssueId
    title: str = Field(min_length=1, max_length=160)
    region: str = Field(min_length=1, max_length=100)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_views: list[ViewName] = Field(min_length=1, max_length=8)


class VisualIssueDiscovery(BaseModel):
    """The broad, compact issue inventory produced from all inspection views."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2_000)
    issues: list[VisualIssueSummary] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def issue_ids_are_unique(self) -> VisualIssueDiscovery:
        ids = [issue.id for issue in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("discovered visual issue ids must be unique")
        return self


class VisualIssueDetail(BaseModel):
    """A focused analysis that may enrich exactly one discovered issue."""

    model_config = ConfigDict(extra="forbid")

    id: IssueId
    description: str = Field(min_length=1, max_length=2_000)
    evidence: list[str] = Field(min_length=1, max_length=12)
    likely_cause: str | None = Field(default=None, max_length=1_000)
    suggested_correction: str = Field(min_length=1, max_length=2_000)
    success_criteria: list[str] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    analysis_conflict: str | None = Field(default=None, max_length=1_000)


class VisualIssue(VisualIssueSummary):
    """An immutable discovery observation plus optional focused analysis."""

    detail_status: DetailStatus
    description: str | None = Field(default=None, max_length=2_000)
    evidence: list[str] | None = Field(default=None, max_length=12)
    likely_cause: str | None = Field(default=None, max_length=1_000)
    suggested_correction: str | None = Field(default=None, max_length=2_000)
    success_criteria: list[str] | None = Field(default=None, max_length=12)
    analysis_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    analysis_conflict: str | None = Field(default=None, max_length=1_000)
    analysis_failure: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def detail_status_matches_present_fields(self) -> VisualIssue:
        required_detail_fields = (
            self.description,
            self.evidence,
            self.suggested_correction,
            self.success_criteria,
            self.analysis_confidence,
        )
        all_analysis_fields = (*required_detail_fields, self.likely_cause, self.analysis_conflict)
        if self.detail_status == "detailed":
            if any(value is None for value in required_detail_fields):
                raise ValueError("detailed visual issues must contain a complete issue analysis")
            if self.analysis_failure is not None:
                raise ValueError("detailed visual issues cannot contain an analysis failure")
        elif self.detail_status == "analysis_failed":
            if not self.analysis_failure:
                raise ValueError("failed visual issue analysis must record a failure marker")
            if any(value is not None for value in all_analysis_fields):
                raise ValueError("failed visual issues cannot contain issue-analysis fields")
        elif (
            any(value is not None for value in all_analysis_fields)
            or self.analysis_failure is not None
        ):
            raise ValueError("summary-only visual issues cannot contain issue-analysis fields")
        return self


class VisualCritique(BaseModel):
    """Read-only final critique assembled from discovery plus issue analyses."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2_000)
    issues: list[VisualIssue] = Field(default_factory=list, max_length=50)
