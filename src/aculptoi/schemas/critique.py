"""Read-only wire and domain contracts for visual critique."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from .inspection import AtlasTileId

IssueId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="A stable lowercase kebab-case visual issue identifier.",
    ),
]
EvidenceReference = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^(?:[A-Z]+[1-9][0-9]*|front|right|top|perspective)$",
    ),
]
Severity = Literal["critical", "high", "medium", "low"]
DetailStatus = Literal["detailed", "summary_only", "analysis_failed"]

# Compact codes are used only in model responses.
WireSeverityCode = Literal["C", "H", "M", "L"]
WIRE_SEVERITY_TO_DOMAIN: dict[WireSeverityCode, Severity] = {
    "C": "critical",
    "H": "high",
    "M": "medium",
    "L": "low",
}
DOMAIN_SEVERITY_TO_WIRE: dict[Severity, WireSeverityCode] = {
    severity: code for code, severity in WIRE_SEVERITY_TO_DOMAIN.items()
}

WirePercentage = Annotated[StrictInt, Field(ge=0, le=100)]
WireRegion = Annotated[str, Field(min_length=1, max_length=100)]
WireObservation = Annotated[str, Field(min_length=1, max_length=320)]
WireEvidenceTiles = Annotated[list[AtlasTileId], Field(min_length=1, max_length=32)]
type VisualIssueTupleWire = tuple[
    WireRegion,
    WireSeverityCode,
    WirePercentage,
    WireEvidenceTiles,
    WireObservation,
]


def _require_nonblank(value: str, field_name: str) -> str:
    """Reject whitespace-only wire text instead of silently guessing its meaning."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _issue_title(region: str, observation: str) -> str:
    """Create a readable deterministic title without spending model output tokens."""
    readable_region = region.replace("_", " ").replace("-", " ").title()
    return f"{readable_region}: {observation}"[:160]


def _discovery_summary(issues: list[VisualIssueSummary]) -> str:
    """Derive a stable human-readable summary without an extra model request."""
    if not issues:
        return "No material visual issues identified in the supplied views."
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    highest = min(issues, key=lambda issue: severity_order[issue.severity]).severity
    noun = "issue" if len(issues) == 1 else "issues"
    return f"{len(issues)} visible {noun} identified; highest severity: {highest}."


class VisualIssueSummary(BaseModel):
    """A rich, immutable discovery observation used beyond the model boundary."""

    model_config = ConfigDict(extra="forbid")

    id: IssueId
    title: str = Field(min_length=1, max_length=160)
    region: str = Field(min_length=1, max_length=100)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_tiles: list[EvidenceReference] = Field(min_length=1, max_length=32)
    observation: str = Field(min_length=1, max_length=320)

    @model_validator(mode="before")
    @classmethod
    def preserve_historic_artifact_compatibility(cls, value: object) -> object:
        """Read artifacts written before observations and tile evidence existed."""
        if not isinstance(value, dict):
            return value
        compatible_value = dict(value)
        if "observation" not in compatible_value:
            title = compatible_value.get("title")
            if isinstance(title, str):
                compatible_value["observation"] = title
        if "evidence_tiles" not in compatible_value and "evidence_views" in compatible_value:
            compatible_value["evidence_tiles"] = compatible_value.pop("evidence_views")
        return compatible_value

    def to_critic_request_context(self) -> dict[str, object]:
        """Return the small non-identifying context needed for focused model analysis."""
        return {
            "region": self.region,
            "severity": DOMAIN_SEVERITY_TO_WIRE[self.severity],
            "tiles": self.evidence_tiles,
            "observation": self.observation,
        }


class VisualIssueDiscovery(BaseModel):
    """The human-readable discovery inventory persisted by the harness."""

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
    """A rich, focused analysis that may enrich exactly one discovered issue."""

    model_config = ConfigDict(extra="forbid")

    id: IssueId
    description: str = Field(min_length=1, max_length=2_000)
    evidence: list[str] = Field(min_length=1, max_length=12)
    likely_cause: str | None = Field(default=None, max_length=1_000)
    suggested_correction: str = Field(min_length=1, max_length=2_000)
    success_criteria: list[str] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    analysis_conflict: str | None = Field(default=None, max_length=1_000)


class VisualIssueDiscoveryWire(BaseModel):
    """Compact discovery JSON accepted only from the model response boundary.

    Each issue tuple is ``[region, severity, confidence, tiles, observation]``.
    """

    model_config = ConfigDict(extra="forbid")

    score: WirePercentage
    issues: list[VisualIssueTupleWire] = Field(max_length=50)

    @field_validator("issues")
    @classmethod
    def issue_tuple_text_is_nonblank(
        cls, issues: list[VisualIssueTupleWire]
    ) -> list[VisualIssueTupleWire]:
        for region, _, _, _, observation in issues:
            _require_nonblank(region, "issue region")
            _require_nonblank(observation, "issue observation")
        return issues

    def to_domain(self) -> VisualIssueDiscovery:
        """Expand compact tuples and assign deterministic IDs for this critique cycle."""
        issues = [
            VisualIssueSummary(
                id=f"issue-{index:03d}",
                title=_issue_title(region, observation),
                region=region,
                severity=WIRE_SEVERITY_TO_DOMAIN[severity],
                confidence=confidence / 100,
                evidence_tiles=list(views),
                observation=observation,
            )
            for index, (region, severity, confidence, views, observation) in enumerate(
                self.issues, start=1
            )
        ]
        return VisualIssueDiscovery(
            score=self.score / 100,
            summary=_discovery_summary(issues),
            issues=issues,
        )


class VisualIssueDetailWire(BaseModel):
    """Compact focused-analysis JSON accepted only from the model response boundary."""

    model_config = ConfigDict(extra="forbid")

    desc: str = Field(min_length=1, max_length=2_000)
    evidence: list[str] = Field(min_length=1, max_length=12)
    cause: str | None = Field(default=None, max_length=1_000)
    fix: str = Field(min_length=1, max_length=2_000)
    criteria: list[str] = Field(min_length=1, max_length=12)
    confidence: WirePercentage
    conflict: str | None = Field(default=None, max_length=1_000)

    @field_validator("desc", "fix")
    @classmethod
    def required_text_is_nonblank(cls, value: str) -> str:
        return _require_nonblank(value, "detail text")

    @field_validator("evidence", "criteria")
    @classmethod
    def list_text_is_nonblank(cls, values: list[str]) -> list[str]:
        for value in values:
            _require_nonblank(value, "detail list entry")
        return values

    @field_validator("cause", "conflict")
    @classmethod
    def optional_text_is_nonblank(cls, value: str | None) -> str | None:
        if value is not None:
            _require_nonblank(value, "optional detail text")
        return value

    def to_domain(self, issue_id: str) -> VisualIssueDetail:
        """Attach the application-owned issue ID and expand percentage confidence."""
        return VisualIssueDetail(
            id=issue_id,
            description=self.desc,
            evidence=self.evidence,
            likely_cause=self.cause,
            suggested_correction=self.fix,
            success_criteria=self.criteria,
            confidence=self.confidence / 100,
            analysis_conflict=self.conflict,
        )


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
