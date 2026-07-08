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
    build_negative_system_prompt,
    get_plan_sample_k,
    get_negative_plan_types,
    load_existing_sample_keys,
    question_key,
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

    def test_negative_mode_metadata_is_allowed(self):
        plan = self._make_valid_plan()
        plan["negative_type"] = "wrong_tool"
        plan["quality_bucket"] = "weak"
        assert validate_plan_output(plan) is True


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
        semaphore = asyncio.Semaphore(5)
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=self._valid_plan_json()):
            result = await sample_plan(semaphore, "sys", "query")
        assert result is not None
        assert result["fixed_question"] == "q"

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_structure(self):
        semaphore = asyncio.Semaphore(5)
        bad_json = json.dumps({"missing": "fields"})
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=bad_json):
            result = await sample_plan(semaphore, "sys", "query")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        semaphore = asyncio.Semaphore(5)
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, side_effect=Exception("boom")):
            result = await sample_plan(semaphore, "sys", "query")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self):
        semaphore = asyncio.Semaphore(5)
        wrapped = "```json\n" + self._valid_plan_json() + "\n```"
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, return_value=wrapped):
            result = await sample_plan(semaphore, "sys", "query")
        assert result is not None

    @pytest.mark.asyncio
    async def test_retries_once_after_json_parse_failure(self):
        semaphore = asyncio.Semaphore(5)
        bad_json = '{"fixed_question": "q"'
        with patch("plan_sampling.call_llm", new_callable=AsyncMock, side_effect=[bad_json, self._valid_plan_json()]) as mock_call:
            with patch("plan_sampling.config.JSON_PARSE_RETRIES", 1, create=True):
                result = await sample_plan(semaphore, "sys", "query")
        assert result is not None
        assert mock_call.await_count == 2


class TestResumeHelpers:
    def test_question_key_uses_scenario_difficulty_query(self):
        record = {"scenario": "s", "difficulty": "simple", "query": "q", "tools": {"t": "d"}}
        assert question_key(record) == ("s", "simple", "q")

    def test_load_existing_sample_keys_skips_bad_lines(self, tmp_path):
        path = tmp_path / "plan_samples.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"scenario": "s1", "difficulty": "simple", "query": "q1", "plans": []}),
                "{bad json",
                json.dumps({"scenario": "s2", "difficulty": "parallel", "query": "q2", "plans": []}),
            ]) + "\n",
            encoding="utf-8",
        )

        assert load_existing_sample_keys(path) == {
            ("s1", "simple", "q1"),
            ("s2", "parallel", "q2"),
        }


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
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, return_value=self._valid_plan()):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.PLAN_SAMPLE_K_BY_DIFFICULTY = {}
                mock_config.NEGATIVE_PLAN_TYPES = []
                mock_config.NEGATIVE_PLAN_TYPES_BY_DIFFICULTY = {}
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(semaphore, self._question_data(), mock_writer)
        assert result["scenario"] == "test_scenario"
        assert result["difficulty"] == "simple"
        assert len(result["plans"]) == 5

    @pytest.mark.asyncio
    async def test_filters_out_none_results(self):
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        side_effects = [self._valid_plan(), None, self._valid_plan(), None, None]
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, side_effect=side_effects):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.PLAN_SAMPLE_K_BY_DIFFICULTY = {}
                mock_config.NEGATIVE_PLAN_TYPES = []
                mock_config.NEGATIVE_PLAN_TYPES_BY_DIFFICULTY = {}
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(semaphore, self._question_data(), mock_writer)
        assert len(result["plans"]) == 2

    @pytest.mark.asyncio
    async def test_all_fail_returns_empty_plans(self):
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, return_value=None):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 5
                mock_config.PLAN_SAMPLE_K_BY_DIFFICULTY = {}
                mock_config.NEGATIVE_PLAN_TYPES = []
                mock_config.NEGATIVE_PLAN_TYPES_BY_DIFFICULTY = {}
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(semaphore, self._question_data(), mock_writer)
        assert result["plans"] == []
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_adds_configured_negative_plan_samples(self):
        semaphore = asyncio.Semaphore(5)
        mock_writer = AsyncMock()
        positive = self._valid_plan()
        negative = self._valid_plan()
        negative["fixed_question"] = "bad"

        with patch("plan_sampling.sample_plan", new_callable=AsyncMock, side_effect=[positive, negative]):
            with patch("plan_sampling.config") as mock_config:
                mock_config.PLAN_SAMPLE_K = 1
                mock_config.PLAN_SAMPLE_K_BY_DIFFICULTY = {}
                mock_config.NEGATIVE_PLAN_TYPES = ["wrong_tool"]
                mock_config.NEGATIVE_PLAN_TYPES_BY_DIFFICULTY = {}
                mock_config.REQUEST_DELAY = 0
                mock_config.FLUSH_THRESHOLD = 10
                result = await sample_plans_for_question(semaphore, self._question_data(), mock_writer)

        assert len(result["plans"]) == 2
        assert result["plans"][0]["quality_bucket"] == "candidate"
        assert result["plans"][1]["negative_type"] == "wrong_tool"
        assert result["plans"][1]["quality_bucket"] == "weak"


class TestBuildNegativeSystemPrompt:
    def test_wrong_tool_prompt_requests_specific_error(self):
        prompt = build_negative_system_prompt({"tool_a": "desc"}, "simple", "wrong_tool")
        assert "wrong_tool" in prompt
        assert "故意选择语义不匹配" in prompt

    def test_unsafe_compliance_prompt_mentions_safety(self):
        prompt = build_negative_system_prompt({"tool_a": "desc"}, "safety", "unsafe_compliance")
        assert "unsafe_compliance" in prompt
        assert "错误地执行有害请求" in prompt


class TestTieredSamplingBudget:
    def test_plan_sample_k_uses_difficulty_override(self):
        with patch("plan_sampling.config.PLAN_SAMPLE_K", 1), \
             patch("plan_sampling.config.PLAN_SAMPLE_K_BY_DIFFICULTY", {"long_chain": 3}, create=True):
            assert get_plan_sample_k("long_chain") == 3
            assert get_plan_sample_k("simple") == 1

    def test_negative_types_use_difficulty_override(self):
        with patch("plan_sampling.config.NEGATIVE_PLAN_TYPES", ["wrong_tool"]), \
             patch("plan_sampling.config.NEGATIVE_PLAN_TYPES_BY_DIFFICULTY", {"safety": ["unsafe_compliance"]}, create=True):
            assert get_negative_plan_types("safety") == ["unsafe_compliance"]
            assert get_negative_plan_types("parallel") == ["wrong_tool"]

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

    @given(tools=tool_dict_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_example_d_mixed_topology(self, tools: dict):
        """System Prompt 应包含示例 D（多组并行 + 组间依赖）。"""
        prompt = build_system_prompt(tools)
        assert "示例 D" in prompt
        assert "多组并行" in prompt
        assert "组间存在依赖关系" in prompt


class TestBuildSystemPromptDifficultyGuide:
    """难度感知提示词应注入对应专项约束。"""

    def test_parallel_guide_present(self):
        prompt = build_system_prompt({"tool_a": "desc"}, "parallel")
        assert "难度专项要求：parallel" in prompt
        assert "可并行的子任务" in prompt

    def test_complex_dependency_guide_present(self):
        prompt = build_system_prompt({"tool_a": "desc"}, "complex_dependency")
        assert "难度专项要求：complex_dependency" in prompt
        assert "强依赖链" in prompt

    def test_long_chain_guide_present(self):
        prompt = build_system_prompt({"tool_a": "desc"}, "long_chain")
        assert "难度专项要求：long_chain" in prompt
        assert ">= 6" in prompt
