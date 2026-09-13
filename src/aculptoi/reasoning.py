"""Small shared types for provider-native model reasoning controls."""

from typing import Literal

type ReasoningEffort = Literal["low", "medium", "high", "xhigh"]
"""Semantic reasoning levels forwarded to a compatible model provider."""
