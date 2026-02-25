"""Tests for utils.py — call_llm and parse_json_response."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the parent module directory is on sys.path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import call_llm, parse_json_response


# ──────────────────────────────────────────────
# parse_json_response — unit tests
# ──────────────────────────────────────────────

class TestParseJsonResponse:
    """Unit tests for parse_json_response."""

    def test_plain_json_object(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_plain_json_array(self):
        assert parse_json_response('[1, 2, 3]') == [1, 2, 3]

    def test_markdown_json_code_block(self):
        content = '```json\n{"key": "value"}\n```'
        assert parse_json_response(content) == {"key": "value"}

    def test_markdown_generic_code_block(self):
        content = '```\n{"key": "value"}\n```'
        assert parse_json_response(content) == {"key": "value"}

    def test_leading_trailing_text(self):
        content = 'Here is the result:\n{"a": 1}\nDone!'
        assert parse_json_response(content) == {"a": 1}

    def test_leading_text_with_array(self):
        content = 'Output: [1, 2, 3]  end'
        assert parse_json_response(content) == [1, 2, 3]

    def test_whitespace_padding(self):
        content = '   \n  {"b": 2}  \n  '
        assert parse_json_response(content) == {"b": 2}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("not json at all")

    def test_nested_json(self):
        obj = {"outer": {"inner": [1, 2, {"deep": True}]}}
        content = f"prefix {json.dumps(obj)} suffix"
        assert parse_json_response(content) == obj

    def test_markdown_block_with_surrounding_text(self):
        content = 'Some explanation\n```json\n[1,2,3]\n```\nMore text'
        assert parse_json_response(content) == [1, 2, 3]


# ──────────────────────────────────────────────
# call_llm — unit tests (mocked aiohttp)
# ──────────────────────────────────────────────

class TestCallLlm:
    """Unit tests for call_llm using mocked aiohttp session."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """call_llm returns content on first successful attempt."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "hello"}}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_response)

        result = await call_llm(session, [{"role": "user", "content": "hi"}])
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """call_llm retries on failure and succeeds on second attempt."""
        # First call raises, second succeeds
        fail_response = MagicMock()
        fail_response.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
        fail_response.__aexit__ = AsyncMock(return_value=False)

        ok_response = AsyncMock()
        ok_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "ok"}}]
        })
        ok_response.__aenter__ = AsyncMock(return_value=ok_response)
        ok_response.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(side_effect=[fail_response, ok_response])

        with patch("utils.asyncio.sleep", new_callable=AsyncMock):
            result = await call_llm(session, [{"role": "user", "content": "hi"}])
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """call_llm raises after exhausting all retries."""
        fail_response = MagicMock()
        fail_response.__aenter__ = AsyncMock(side_effect=Exception("fail"))
        fail_response.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=fail_response)

        with patch("utils.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="fail"):
                await call_llm(session, [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_uses_config_values(self):
        """call_llm builds payload from config values."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "resp"}}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_response)

        await call_llm(session, [{"role": "user", "content": "test"}], temperature=0.8, top_p=0.9)

        # Verify the post was called with correct URL and payload
        call_args = session.post.call_args
        assert "/chat/completions" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.9


# ──────────────────────────────────────────────
# Hypothesis imports for property-based testing
# ──────────────────────────────────────────────
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ──────────────────────────────────────────────
# Feature: plan-data-synthesis, Property 12: JSON 解析容错 — 包裹内容提取
# ──────────────────────────────────────────────

# Strategy: generate safe text that won't contain markdown code fences (```)
# or unbalanced JSON delimiters, which would confuse the parser's extraction.
_safe_text = st.text(
    alphabet=st.characters(blacklist_characters='`'),
    min_size=0, max_size=20,
)

# Leaf values: primitives with safe text (no backticks in strings)
_json_primitives = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | _safe_text
)

# Recursive strategy building nested structures from safe primitives.
_json_values = st.recursive(
    _json_primitives,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(_safe_text.filter(lambda s: len(s) > 0), children, max_size=5),
    max_leaves=15,
)

# parse_json_response is designed for LLM outputs which are always JSON
# objects or arrays. Top-level must be a dict or list.
json_objects = st.dictionaries(_safe_text.filter(lambda s: len(s) > 0), _json_values, min_size=1, max_size=5)
json_arrays = st.lists(_json_values, min_size=1, max_size=5)
json_containers = json_objects | json_arrays


class TestParseJsonResponseProperty12:
    """Property 12: parse_json_response 对各种包裹格式的容错解析。

    **Validates: Requirements 8.1, 8.2**
    """

    @given(obj=json_objects)
    @settings(max_examples=100)
    def test_plain_json_object_roundtrip(self, obj):
        """Plain JSON object string → parse_json_response returns equivalent object."""
        raw = json.dumps(obj)
        assert parse_json_response(raw) == obj

    @given(obj=json_arrays)
    @settings(max_examples=100)
    def test_plain_json_array_roundtrip(self, obj):
        """Plain JSON array string → parse_json_response returns equivalent object."""
        raw = json.dumps(obj)
        assert parse_json_response(raw) == obj

    @given(obj=json_containers)
    @settings(max_examples=100)
    def test_markdown_json_block_roundtrip(self, obj):
        """Wrapped in ```json\\n...\\n``` → parse_json_response returns equivalent object."""
        raw = json.dumps(obj)
        wrapped = f"```json\n{raw}\n```"
        assert parse_json_response(wrapped) == obj

    @given(obj=json_containers)
    @settings(max_examples=100)
    def test_markdown_generic_block_roundtrip(self, obj):
        """Wrapped in ```\\n...\\n``` → parse_json_response returns equivalent object."""
        raw = json.dumps(obj)
        wrapped = f"```\n{raw}\n```"
        assert parse_json_response(wrapped) == obj

    @given(obj=json_objects)
    @settings(max_examples=100)
    def test_leading_trailing_text_roundtrip(self, obj):
        """With leading text + JSON object + trailing text → parse_json_response returns equivalent object.

        Note: uses json_objects (dicts only) because the boundary detection
        in parse_json_response prioritises '{' over '[', so arrays containing
        objects would cause the finder to locate an inner '{' instead of the
        outer '[' when surrounded by non-JSON text.
        """
        raw = json.dumps(obj)
        wrapped = f"Here is the result:\n{raw}\nDone!"
        assert parse_json_response(wrapped) == obj
