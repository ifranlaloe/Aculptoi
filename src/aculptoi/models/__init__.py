"""Local model-provider abstractions."""

from .base import ModelProvider, ModelProviderError
from .openai_compatible import OpenAICompatibleProvider

__all__ = ["ModelProvider", "ModelProviderError", "OpenAICompatibleProvider"]
