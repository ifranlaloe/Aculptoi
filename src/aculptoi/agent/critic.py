"""Read-only two-stage visual critique requests and final-critique assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from aculptoi.agent.prompts import (
    ISSUE_ANALYSIS_PROMPT_VERSION,
    ISSUE_ANALYSIS_SYSTEM_PROMPT,
    ISSUE_DISCOVERY_PROMPT_VERSION,
    ISSUE_DISCOVERY_SYSTEM_PROMPT,
)
from aculptoi.models.base import Message, ModelProvider, ModelResponseError
from aculptoi.reasoning import ReasoningEffort
from aculptoi.schemas.critique import (
    VisualCritique,
    VisualIssue,
    VisualIssueDetail,
    VisualIssueDetailWire,
    VisualIssueDiscovery,
    VisualIssueDiscoveryWire,
    VisualIssueSummary,
)
from aculptoi.vision import prepare_render


class VisionCritic:
    """Inspect renders through separate discovery and focused-analysis requests only."""

    def __init__(
        self,
        provider: ModelProvider,
        max_image_dimension: int = 1280,
        max_output_tokens: int = 16_384,
        reasoning_effort: ReasoningEffort = "medium",
        max_discovered_issues: int = 12,
        max_issue_analysis_requests: int = 12,
    ) -> None:
        self._provider = provider
        self._max_image_dimension = max_image_dimension
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._max_discovered_issues = max_discovered_issues
        self._max_issue_analysis_requests = max_issue_analysis_requests

    @property
    def max_issue_analysis_requests(self) -> int:
        """Return the harness-owned safety budget for focused analyses."""
        return self._max_issue_analysis_requests

    def inspect(
        self,
        goal: str,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> VisualCritique:
        """Convenience path for callers that do not need durable request artifacts."""
        discovery = self.discover(goal, images, previous_score=previous_score)
        details: dict[str, VisualIssueDetail] = {}
        failures: dict[str, str] = {}
        # The persistent refinement harness owns this policy in normal runs.
        # This convenience method follows the same simple V1 policy.
        for issue in discovery.issues[: self._max_issue_analysis_requests]:
            try:
                details[issue.id] = self.analyze_issue(
                    goal,
                    issue,
                    self.select_images_for_issue(images, issue),
                    previous_score=previous_score,
                )
            except ModelResponseError as error:
                failures[issue.id] = str(error)
        return self.assemble_critique(discovery, details, failures)

    def discover(
        self,
        goal: str,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> VisualIssueDiscovery:
        """Discover compact, independent visual issues from the complete render set."""
        messages, _ = self.build_discovery_request(goal, images, previous_score=previous_score)
        return self.discover_messages(messages, available_views=[image.stem for image in images])

    def build_discovery_request(
        self,
        goal: str,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> tuple[list[Message], dict[str, object]]:
        """Build the all-views discovery request and a data-URL-free artifact manifest."""
        context: dict[str, object] = {
            "goal": goal,
            "previous_score": previous_score,
            "views": [image.stem for image in images],
            "max_discovered_issues": self._max_discovered_issues,
        }
        return self._build_image_request(
            system_prompt=ISSUE_DISCOVERY_SYSTEM_PROMPT,
            prompt_version=ISSUE_DISCOVERY_PROMPT_VERSION,
            request_type="vision_issue_discovery",
            context=context,
            images=images,
        )

    def discover_messages(
        self,
        messages: Sequence[Message],
        *,
        available_views: Sequence[str] | None = None,
    ) -> VisualIssueDiscovery:
        """Request and validate one discovery response, including its configured issue cap."""
        response = self._provider.complete_json(
            messages,
            max_tokens=self._max_output_tokens,
            reasoning_effort=self._reasoning_effort,
        )
        raw_response = self._raw_response(response)
        try:
            discovery = VisualIssueDiscoveryWire.model_validate(response).to_domain()
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the compact visual-issue discovery wire schema",
                raw_response,
            ) from error
        if len(discovery.issues) > self._max_discovered_issues:
            raise ModelResponseError("Model response exceeded max_discovered_issues", raw_response)
        if available_views is not None:
            known_views = set(available_views)
            unknown_views = sorted(
                {
                    view
                    for issue in discovery.issues
                    for view in issue.evidence_views
                    if view not in known_views
                }
            )
            if unknown_views:
                raise ModelResponseError(
                    f"Model response referenced unavailable evidence views: {unknown_views}",
                    raw_response,
                )
        return discovery

    @staticmethod
    def select_images_for_issue(images: Sequence[Path], issue: VisualIssueSummary) -> list[Path]:
        """Use a discovery issue's evidence views, falling back to all images safely."""
        requested = set(issue.evidence_views)
        selected = [image for image in images if image.stem in requested]
        return selected or list(images)

    def analyze_issue(
        self,
        goal: str,
        issue: VisualIssueSummary,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> VisualIssueDetail:
        """Produce a detailed, read-only analysis for exactly one known issue."""
        messages, _ = self.build_issue_analysis_request(
            goal, issue, images, previous_score=previous_score
        )
        return self.analyze_issue_messages(messages, expected_issue_id=issue.id)

    def build_issue_analysis_request(
        self,
        goal: str,
        issue: VisualIssueSummary,
        images: Sequence[Path],
        *,
        previous_score: float | None = None,
    ) -> tuple[list[Message], dict[str, object]]:
        """Build one evidence-filtered issue-analysis request and its artifact manifest."""
        context: dict[str, object] = {
            "goal": goal,
            "previous_score": previous_score,
            "issue": issue.to_critic_request_context(),
        }
        messages, artifact = self._build_image_request(
            system_prompt=ISSUE_ANALYSIS_SYSTEM_PROMPT,
            prompt_version=ISSUE_ANALYSIS_PROMPT_VERSION,
            request_type="vision_issue_analysis",
            context=context,
            images=images,
        )
        artifact["requested_evidence_views"] = list(issue.evidence_views)
        return messages, artifact

    def analyze_issue_messages(
        self, messages: Sequence[Message], *, expected_issue_id: str
    ) -> VisualIssueDetail:
        """Request and validate a focused response without allowing issue identity drift."""
        response = self._provider.complete_json(
            messages,
            max_tokens=self._max_output_tokens,
            reasoning_effort=self._reasoning_effort,
        )
        raw_response = self._raw_response(response)
        try:
            detail = VisualIssueDetailWire.model_validate(response).to_domain(expected_issue_id)
        except ValidationError as error:
            raise ModelResponseError(
                "Model response did not satisfy the compact visual-issue detail wire schema",
                raw_response,
            ) from error
        return detail

    @staticmethod
    def assemble_critique(
        discovery: VisualIssueDiscovery,
        details: Mapping[str, VisualIssueDetail],
        failures: Mapping[str, str],
    ) -> VisualCritique:
        """Enrich immutable discovery observations without replacing their identity fields."""
        issues: list[VisualIssue] = []
        for summary in discovery.issues:
            detail = details.get(summary.id)
            if detail is not None:
                issues.append(
                    VisualIssue(
                        **summary.model_dump(mode="json"),
                        detail_status="detailed",
                        description=detail.description,
                        evidence=detail.evidence,
                        likely_cause=detail.likely_cause,
                        suggested_correction=detail.suggested_correction,
                        success_criteria=detail.success_criteria,
                        analysis_confidence=detail.confidence,
                        analysis_conflict=detail.analysis_conflict,
                    )
                )
            elif summary.id in failures:
                issues.append(
                    VisualIssue(
                        **summary.model_dump(mode="json"),
                        detail_status="analysis_failed",
                        analysis_failure=failures[summary.id],
                    )
                )
            else:
                issues.append(
                    VisualIssue(**summary.model_dump(mode="json"), detail_status="summary_only")
                )
        return VisualCritique(score=discovery.score, summary=discovery.summary, issues=issues)

    def _build_image_request(
        self,
        *,
        system_prompt: str,
        prompt_version: str,
        request_type: str,
        context: dict[str, object],
        images: Sequence[Path],
    ) -> tuple[list[Message], dict[str, object]]:
        """Create one generic OpenAI-compatible image request and inspectable manifest."""
        content: list[dict[str, object]] = [{"type": "text", "text": json.dumps(context)}]
        views: list[dict[str, object]] = []
        for image in images:
            prepared = prepare_render(image, self._max_image_dimension)
            views.append(
                {
                    "name": prepared.source.stem,
                    "source_path": str(prepared.source),
                    "prepared_width": prepared.width,
                    "prepared_height": prepared.height,
                }
            )
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Inspection view: {prepared.source.stem} "
                            f"({prepared.width}x{prepared.height} pixels)."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": prepared.data_url}},
                ]
            )
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return messages, {
            "role": "vision_critic",
            "request_type": request_type,
            "prompt_version": prompt_version,
            "max_output_tokens": self._max_output_tokens,
            "reasoning_effort": self._reasoning_effort,
            "system_prompt": system_prompt,
            "input": context,
            "views": views,
        }

    @staticmethod
    def _raw_response(response: object) -> str:
        return json.dumps(response, indent=2, sort_keys=True, default=str)
