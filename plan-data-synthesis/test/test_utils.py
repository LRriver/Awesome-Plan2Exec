"""Tests for utils.py — call_llm and parse_json_response."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the parent module directory is on sys.path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import call_llm, ensure_trailing_newline, parse_json_response


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

    def test_control_character_inside_json_string_is_sanitized(self):
        content = '{"query": "compare S&P 500\nand Nasdaq"}'
        assert parse_json_response(content) == {"query": "compare S&P 500 and Nasdaq"}


class TestEnsureTrailingNewline:
    def test_noops_for_missing_or_empty_file(self, tmp_path):
        missing = tmp_path / "missing.jsonl"
        ensure_trailing_newline(missing)
        assert not missing.exists()

        empty = tmp_path / "empty.jsonl"
        empty.write_bytes(b"")
        ensure_trailing_newline(empty)
        assert empty.read_bytes() == b""

    def test_adds_newline_when_last_line_is_incomplete(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_bytes(b'{"a": 1}')

        ensure_trailing_newline(path)

        assert path.read_bytes() == b'{"a": 1}\n'

    def test_keeps_existing_newline(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_bytes(b'{"a": 1}\n')

        ensure_trailing_newline(path)

        assert path.read_bytes() == b'{"a": 1}\n'


# ──────────────────────────────────────────────
# call_llm — unit tests (mocked OpenAI SDK)
# ──────────────────────────────────────────────

class TestCallLlm:
    """Unit tests for call_llm using mocked OpenAI async client."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """call_llm returns content on first successful attempt."""
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hello"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch("utils._get_openai_client", return_value=mock_client):
            result = await call_llm([{"role": "user", "content": "hi"}])
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """call_llm retries on failure and succeeds on second attempt."""
        ok_resp = MagicMock()
        ok_resp.choices = [MagicMock(message=MagicMock(content="ok"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[Exception("timeout"), ok_resp])

        with patch("utils.asyncio.sleep", new_callable=AsyncMock):
            with patch("utils._get_openai_client", return_value=mock_client):
                result = await call_llm([{"role": "user", "content": "hi"}])
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """call_llm raises after exhausting all retries."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("fail"))

        with patch("utils.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="fail"):
                with patch("utils._get_openai_client", return_value=mock_client):
                    await call_llm([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_uses_config_values(self):
        """call_llm passes expected args to OpenAI SDK."""
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="resp"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch("utils._get_openai_client", return_value=mock_client):
            await call_llm([{"role": "user", "content": "test"}], temperature=0.8, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"]
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["extra_body"] == {"reasoning_split": True}

    @pytest.mark.asyncio
    async def test_call_llm_accepts_model_override(self):
        """call_llm can route a single request to an explicit model."""
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="resp"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch("utils._get_openai_client", return_value=mock_client):
            await call_llm(
                [{"role": "user", "content": "test"}],
                model="DeepSeek-V4-Flash",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "DeepSeek-V4-Flash"

    def test_resolve_model_uses_named_profile(self):
        """Named model profiles keep stage routing out of call sites."""
        from utils import resolve_model

        with patch("utils.config.LLM_PROFILES", {"cheap": {"model": "DeepSeek-V4-Flash"}}):
            assert resolve_model(profile="cheap") == "DeepSeek-V4-Flash"

    def test_resolve_profile_max_concurrency_uses_named_profile(self):
        """Named profiles can set their own rate limits."""
        from utils import resolve_profile_max_concurrency

        with patch("utils.config.LLM_PROFILES", {"strong": {"max_concurrency": 7}}):
            assert resolve_profile_max_concurrency(profile="strong") == 7

    @pytest.mark.asyncio
    async def test_call_llm_enforces_profile_concurrency(self):
        """Profile max_concurrency limits concurrent requests to a model."""
        from utils import _PROFILE_SEMAPHORES

        _PROFILE_SEMAPHORES.clear()
        active = 0
        max_active = 0

        async def fake_create(**_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        profiles = {
            "strong": {
                "model": "gpt-5.4",
                "base_url": "http://example.test",
                "api_key": "key",
                "max_concurrency": 1,
            }
        }
        with patch("utils.config.LLM_PROFILES", profiles):
            with patch("utils._get_openai_client", return_value=mock_client):
                await asyncio.gather(
                    call_llm([{"role": "user", "content": "a"}], profile="strong"),
                    call_llm([{"role": "user", "content": "b"}], profile="strong"),
                )

        assert max_active == 1

    @pytest.mark.asyncio
    async def test_disable_reasoning_split(self):
        """call_llm omits extra_body when reasoning split is disabled."""
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="resp"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch("utils._get_openai_client", return_value=mock_client):
            with patch("utils.config.LLM_REASONING_SPLIT", False):
                await call_llm([{"role": "user", "content": "test"}], temperature=0.8, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs


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
