"""
Tests for plan_sampling.py — validate_plan_output, sample_plan, sample_plans_for_question,
and build_system_prompt (Property 5).
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from plan_sampling import (
    build_system_prompt,
    validate_plan_output,
    sample_plan,
    sample_plans_for_question,
)


# ── validate_plan_output ──────────────────────────────────────────

class TestValidatePlanOutput:
    """Unit tests for validate_plan_output."""

    def _make_valid_plan(self) -> dict:
        return {
            "fixed_question": "补全后的问题",
            "thought": "整体思考",
            "steps": [
                {
                    "thought": "步骤思考",
                    "title": "步骤标题",
                    "content": "步骤内容",
                    "tools": ["tool_a"],
                    "dependencies": None,
                }
            ],
        }

    def test_valid_plan_passes(self):
        assert validate_plan_output(self._make_valid_plan()) is True

    def test_missing_fixed_question(self):
        plan = self._make_valid_plan()
        del plan["fixed_question"]
        assert validate_plan_output(plan) is False

    def test_missing_thought(self):
        plan = self._make_valid_plan()
        del plan["thought"]
        assert validate_plan_output(plan) is False

    def test_empty_steps(self):
        plan = self._make_valid_plan()
        plan["steps"] = []
        assert validate_plan_output(plan) is False

    def test_steps_not_list(self):
        plan = self._make_valid_plan()
        plan["steps"] = "not a list"
        assert validate_plan_output(plan) is False

    def test_step_missing_title(self):
        plan = self._make_valid_plan()
        del plan["steps"][0]["title"]
        assert validate_plan_output(plan) is False

    def test_step_tools_null_ok(self):
        plan = self._make_valid_plan()
        plan["steps"][0]["tools"] = None
        assert validate_plan_output(plan) is True

    def test_step_dependencies_list_ok(self):
        plan = self._make_valid_plan()
        plan["steps"][0]["dependencies"] = ["步骤A"]
        assert validate_plan_output(plan) is True

    def test_step_tools_wrong_type(self):
        plan = self._make_valid_plan()
        plan["steps"][0]["tools"] = "not_a_list"
        assert validate_plan_output(plan) is False

    def test_step_dependencies_wrong_type(self):
        plan = self._make_valid_plan()
        plan["steps"][0]["dependencies"] = 123
        assert validate_plan_output(plan) is False

    def test_multiple_steps(self):
        plan = self._make_valid_plan()
        plan["steps"].append({
            "thought": "t2",
            "title": "t2",
            "content": "c2",
            "tools": None,
            "dependencies": ["步骤标题"],
        })
        assert validate_plan_output(plan) is True

    def test_step_not_dict(self):
        plan = self._make_valid_plan()
        plan["steps"] = ["not a dict"]
        assert validate_plan_output(plan) is False


# ── sample_plan ───────────────────────────────────────────────────

class TestSamplePlan:
    """Unit tests for sample_plan (mocks LLM call)."""

    def _valid_plan_json(self) -> str:
        return json.dumps({
            "fixed_question": "q",
            "thought": "t",
            "steps": [{"thought": "s", "title": "s", "content": "c", "tools": None, "dependencies": None}],
        })

    @pytest.mark.asyncio
    async def test_returns_valid_plan(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=self._valid_plan_json()):
            result = await sample_plan(session, semaphore, "sys", "query")
        assert result is not None
        assert result["fixed_question"] == "q"

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_structure(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        bad_json = json.dumps({"missing": "fields"})
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=bad_json):
            result = await sample_plan(session, semaphore, "sys", "query")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, side_effect=Exception("boom")):
            result = await sample_plan(session, semaphore, "sys", "query")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        wrapped = "```json\n" + self._valid_plan_json() + "\n```"
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=wrapped):
            result = await sample_plan(session, semaphore, "sys", "query")
        assert result is not None


# ── sample_plans_for_question ─────────────────────────────────────

class TestSamplePlansForQuestion:
    """Unit tests for sample_plans_for_question."""

    def _question_data(self) -> dict:
        return {
            "scenario": "test_scenario",
            "tools": {"tool_a": "desc_a"},
            "difficulty": "simple",
            "query": "test query",
        }

    def _valid_plan(self) -> dict:
        return {
            "fixed_question": "q",
            "thought": "t",
            "steps": [{"thought": "s", "title": "s", "content": "c", "tools": None, "dependencies": None}],
        }

    @pytest.mark.asyncio
    async def test_collects_all_successful_samples(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, return_value=self._valid_plan()):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(session, semaphore, self._question_data(), mock_writer)
        assert result["scenario"] == "test_scenario"
        assert result["difficulty"] == "simple"
        assert len(result["plans"]) == 5

    @pytest.mark.asyncio
    async def test_filters_out_none_results(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        side_effects = [self._valid_plan(), None, self._valid_plan(), None, None]
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, side_effect=side_effects):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(session, semaphore, self._question_data(), mock_writer)
        assert len(result["plans"]) == 2

    @pytest.mark.asyncio
    async def test_all_fail_returns_empty_plans(self):
        session = MagicMock()
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, return_value=None):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(session, semaphore, self._question_data(), mock_writer)
        assert result["plans"] == []
        assert result["query"] == "test query"

# Feature: plan-data-synthesis, Property 5: 规划采样 System Prompt 完整性
# **Validates: Requirements 3.2, 3.6, 3.7**

# Strategy: generate tool dictionaries with 1-5 tools.
# Keys use alphanumeric + underscore; values use printable text (letters, digits, spaces)
# to avoid json.dumps escaping control characters which would break the "in prompt" assertion.
_printable_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
)
tool_dict_strategy = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
        min_size=1,
        max_size=20,
    ),
    values=_printable_text,
    min_size=1,
    max_size=5,
)


class TestBuildSystemPromptProperty5:
    """Property 5: build_system_prompt() 包含工具集和三个示例。"""

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_all_tool_names(self, tools: dict):
        """System Prompt 应包含工具集中所有工具名。"""
        prompt = build_system_prompt(tools)
        for tool_name in tools:
            assert tool_name in prompt, f"Tool name '{tool_name}' not found in prompt"

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_all_tool_descriptions(self, tools: dict):
        """System Prompt 应包含工具集中所有工具描述。"""
        prompt = build_system_prompt(tools)
        for desc in tools.values():
            assert desc in prompt, f"Tool description '{desc}' not found in prompt"

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_tool_constraint(self, tools: dict):
        """System Prompt 应包含工具选择约束说明。"""
        prompt = build_system_prompt(tools)
        assert "只能从【当前可用工具集】中选取" in prompt

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_example_a(self, tools: dict):
        """System Prompt 应包含示例 A（多步骤，无依赖关系）。"""
        prompt = build_system_prompt(tools)
        assert "示例 A" in prompt
        assert "无依赖关系" in prompt

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_example_b(self, tools: dict):
        """System Prompt 应包含示例 B（无需工具，直接回答）。"""
        prompt = build_system_prompt(tools)
        assert "示例 B" in prompt
        assert "无需工具" in prompt

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_example_c(self, tools: dict):
        """System Prompt 应包含示例 C（多步骤，存在依赖关系）。"""
        prompt = build_system_prompt(tools)
        assert "示例 C" in prompt
        assert "存在依赖关系" in prompt

