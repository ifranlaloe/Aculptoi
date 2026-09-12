from __future__ import annotations

import pytest

from aculptoi.models import ModelProviderError, OpenAICompatibleProvider


def test_json_parser_accepts_a_fenced_json_object() -> None:
    assert OpenAICompatibleProvider._parse_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_json_parser_rejects_non_object() -> None:
    with pytest.raises(ModelProviderError, match="JSON object"):
        OpenAICompatibleProvider._parse_json("[]")
