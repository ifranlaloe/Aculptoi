"""Read-only two-stage visual critique over one accepted inspection atlas."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from aculptoi.agent.prompts import (
    ISSUE_ANALYSIS_PROMPT_VERSION,
    ISSUE_ANALYSIS_SYSTEM_PROMPT,
    ISSUE_DISCOVERY_PROMPT_VERSION,
    ISSUE_DISCOVERY_SYSTEM_PROMPT,
)
from aculptoi.inference import InferenceProfile
from aculptoi.models import ModelUsage, complete_json_with_usage
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
from aculptoi.schemas.inspection import InspectionAtlasManifest
from aculptoi.vision import prepare_render


class VisionCritic:
    """Discover and analyze visual issues using a technically accepted atlas only."""

    def __init__(
        self,
        provider: ModelProvider,
        max_image_dimension: int = 4096,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        max_discovered_issues: int = 12,
        max_issue_analysis_requests: int = 12,
        discovery_profile: InferenceProfile | None = None,
        issue_analysis_profile: InferenceProfile | None = None,
        provider_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._max_image_dimension = max_image_dimension
        legacy_profile = (
            InferenceProfile(
                max_output_tokens=max_output_tokens or 16_384,
                thinking=True,
                reasoning_effort=reasoning_effort or "medium",
            )
            if max_output_tokens is not None or reasoning_effort is not None
            else None
        )
        self._discovery_profile = (
            discovery_profile
            or legacy_profile
            or InferenceProfile(
                max_output_tokens=4_096,
                thinking=False,
            )
        )
        self._issue_analysis_profile = (
            issue_analysis_profile
            or legacy_profile
            or InferenceProfile(
                max_output_tokens=16_384,
                thinking=True,
                reasoning_effort="medium",
            )
        )
        self._provider_name = provider_name
        self._max_discovered_issues = max_discovered_issues
        self._max_issue_analysis_requests = max_issue_analysis_requests

    @property
    def discovery_profile(self) -> InferenceProfile:
        """Return the inexpensive breadth-first discovery settings."""
        return self._discovery_profile

    @property
    def issue_analysis_profile(self) -> InferenceProfile:
        """Return the deeper settings reserved for one known issue."""
        return self._issue_analysis_profile

    @property
    def provider_name(self) -> str | None:
        """Return the configured provider identifier when the runtime supplies one."""
        return self._provider_name

    @property
    def max_issue_analysis_requests(self) -> int:
        """Return the harness-owned safety budget for focused analyses."""
        return self._max_issue_analysis_requests

    def inspect(
        self,
        goal: str,
        atlas: Path,
        manifest: InspectionAtlasManifest,
        *,
        previous_score: float | None = None,
    ) -> VisualCritique:
        """Convenience path for callers that do not need durable request artifacts."""
        discovery = self.discover(goal, atlas, manifest, previous_score=previous_score)
        details: dict[str, VisualIssueDetail] = {}
        failures: dict[str, str] = {}
        for issue in discovery.issues[: self._max_issue_analysis_requests]:
            try:
                details[issue.id] = self.analyze_issue(
                    goal,
                    issue,
                    atlas,
                    manifest,
                    previous_score=previous_score,
                )
            except ModelResponseError as error:
                failures[issue.id] = str(error)
        return self.assemble_critique(discovery, details, failures)

    def discover(
        self,
        goal: str,
        atlas: Path,
        manifest: InspectionAtlasManifest,
        *,
        previous_score: float | None = None,
    ) -> VisualIssueDiscovery:
        """Discover compact, independently supported issues from one accepted atlas."""
        messages, _ = self.build_discovery_request(
            goal,
            atlas,
            manifest,
            previous_score=previous_score,
        )
        return self.discover_messages(messages, available_tiles=list(manifest.tiles))

    def build_discovery_request(
        self,
        goal: str,
        atlas: Path,
        manifest: InspectionAtlasManifest,
        *,
        previous_score: float | None = None,
    ) -> tuple[list[Message], dict[str, object]]:
        """Build an all-angle discovery request and a data-URL-free artifact."""
        context: dict[str, object] = {
            "goal": goal,
            "previous_score": previous_score,
            "inspection_atlas": manifest.to_critic_context(),
            "max_discovered_issues": self._max_discovered_issues,
        }
        return self._build_atlas_request(
            system_prompt=ISSUE_DISCOVERY_SYSTEM_PROMPT,
            prompt_version=ISSUE_DISCOVERY_PROMPT_VERSION,
            request_type="vision_issue_discovery",
            context=context,
            atlas=atlas,
            profile=self._discovery_profile,
        )

    def discover_messages(
        self,
        messages: Sequence[Message],
        *,
        available_tiles: Sequence[str] | None = None,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> VisualIssueDiscovery:
        """Request and validate one discovery response, including its configured cap."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._discovery_profile.max_output_tokens,
            thinking=self._discovery_profile.thinking,
            reasoning_effort=self._discovery_profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
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
        if available_tiles is not None:
            known_tiles = set(available_tiles)
            unknown_tiles = sorted(
                {
                    tile
                    for issue in discovery.issues
                    for tile in issue.evidence_tiles
                    if tile not in known_tiles
                }
            )
            if unknown_tiles:
                raise ModelResponseError(
                    f"Model response referenced unavailable evidence tiles: {unknown_tiles}",
                    raw_response,
                )
        return discovery

    def analyze_issue(
        self,
        goal: str,
        issue: VisualIssueSummary,
        atlas: Path,
        manifest: InspectionAtlasManifest,
        *,
        previous_score: float | None = None,
    ) -> VisualIssueDetail:
        """Produce one focused, read-only analysis with the full atlas retained."""
        messages, _ = self.build_issue_analysis_request(
            goal,
            issue,
            atlas,
            manifest,
            previous_score=previous_score,
        )
        return self.analyze_issue_messages(messages, expected_issue_id=issue.id)

    def build_issue_analysis_request(
        self,
        goal: str,
        issue: VisualIssueSummary,
        atlas: Path,
        manifest: InspectionAtlasManifest,
        *,
        previous_score: float | None = None,
    ) -> tuple[list[Message], dict[str, object]]:
        """Build a focused request with all atlas tiles available for comparison."""
        context: dict[str, object] = {
            "goal": goal,
            "previous_score": previous_score,
            "issue": issue.to_critic_request_context(),
            "inspection_atlas": manifest.to_critic_context(),
        }
        messages, artifact = self._build_atlas_request(
            system_prompt=ISSUE_ANALYSIS_SYSTEM_PROMPT,
            prompt_version=ISSUE_ANALYSIS_PROMPT_VERSION,
            request_type="vision_issue_analysis",
            context=context,
            atlas=atlas,
            profile=self._issue_analysis_profile,
        )
        artifact["requested_evidence_tiles"] = list(issue.evidence_tiles)
        return messages, artifact

    def analyze_issue_messages(
        self,
        messages: Sequence[Message],
        *,
        expected_issue_id: str,
        usage_recorder: Callable[[ModelUsage | None], None] | None = None,
    ) -> VisualIssueDetail:
        """Request and validate a focused response without allowing issue identity drift."""
        completion = complete_json_with_usage(
            self._provider,
            messages,
            max_tokens=self._issue_analysis_profile.max_output_tokens,
            thinking=self._issue_analysis_profile.thinking,
            reasoning_effort=self._issue_analysis_profile.reasoning_effort,
        )
        if usage_recorder is not None:
            usage_recorder(completion.usage)
        response = completion.value
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

    def _build_atlas_request(
        self,
        *,
        system_prompt: str,
        prompt_version: str,
        request_type: str,
        context: dict[str, object],
        atlas: Path,
        profile: InferenceProfile,
    ) -> tuple[list[Message], dict[str, object]]:
        """Create an atlas request without downsampling configured tile detail."""
        prepared = prepare_render(atlas, self._max_image_dimension)
        content: list[dict[str, object]] = [
            {"type": "text", "text": json.dumps(context)},
            {
                "type": "text",
                "text": (
                    f"Accepted inspection atlas: {prepared.source.stem} "
                    f"({prepared.width}x{prepared.height} pixels)."
                ),
            },
            {"type": "image_url", "image_url": {"url": prepared.data_url}},
        ]
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return messages, {
            "role": "vision_critic",
            "request_type": request_type,
            "prompt_version": prompt_version,
            "thinking": profile.thinking,
            "max_output_tokens": profile.max_output_tokens,
            "reasoning_effort": profile.reasoning_effort,
            "system_prompt": system_prompt,
            "input": context,
            "atlas": {
                "source_path": str(prepared.source),
                "prepared_width": prepared.width,
                "prepared_height": prepared.height,
            },
        }

    @staticmethod
    def _raw_response(response: object) -> str:
        return json.dumps(response, indent=2, sort_keys=True, default=str)
