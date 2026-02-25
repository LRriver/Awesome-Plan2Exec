"""Tests for evaluate_plans.py — build_eval_prompt, compute_weighted_score, validate_evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate_plans import build_eval_prompt, compute_weighted_score, validate_evaluation


# ──────────────────────────────────────────────
# Feature: plan-data-synthesis, Property 9: 评分 Prompt 包含完整上下文
# Validates: Requirements 4.4, 4.5
# ──────────────────────────────────────────────

# Strategies: printable text for user queries, tool dicts, and plan dicts
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
)

_tool_dict_strategy = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
        min_size=1,
        max_size=20,
    ),
    values=_safe_text,
    min_size=1,
    max_size=5,
)

_plan_strategy = st.fixed_dictionaries({
    "fixed_question": _safe_text,
    "thought": _safe_text,
    "steps": st.lists(
        st.fixed_dictionaries({
            "thought": _safe_text,
            "title": _safe_text,
            "content": _safe_text,
            "tools": st.none() | st.lists(_safe_text, min_size=1, max_size=2),
            "dependencies": st.none() | st.lists(_safe_text, min_size=1, max_size=2),
        }),
        min_size=1,
        max_size=3,
    ),
})


class TestBuildEvalPromptProperty9:
    """Property 9: build_eval_prompt() 包含完整上下文（用户问题、工具集、规划、五维度评分规则）。"""

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_user_query(self, user_query: str, tools: dict, plan: dict):
        """**Validates: Requirements 4.4**
        评分 Prompt 应包含用户问题文本。
        """
        prompt = build_eval_prompt(user_query, tools, plan)
        assert user_query in prompt, f"User query '{user_query}' not found in prompt"

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_tool_names(self, user_query: str, tools: dict, plan: dict):
        """**Validates: Requirements 4.4**
        评分 Prompt 应包含工具集中所有工具名。
        """
        prompt = build_eval_prompt(user_query, tools, plan)
        for tool_name in tools:
            assert tool_name in prompt, f"Tool name '{tool_name}' not found in prompt"

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_plan_content(self, user_query: str, tools: dict, plan: dict):
        """**Validates: Requirements 4.4**
        评分 Prompt 应包含待评估规划的 fixed_question。
        """
        prompt = build_eval_prompt(user_query, tools, plan)
        assert plan["fixed_question"] in prompt, "Plan fixed_question not found in prompt"

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy)
    @settings(max_examples=100)
    def test_prompt_contains_five_dimensions(self, user_query: str, tools: dict, plan: dict):
        """**Validates: Requirements 4.4, 4.5**
        评分 Prompt 应包含五个评价维度的中文名称。
        """
        prompt = build_eval_prompt(user_query, tools, plan)
        assert "工具准确性" in prompt
        assert "依赖合理性" in prompt
        assert "任务完整性" in prompt
        assert "规划简洁性" in prompt
        assert "思维链质量" in prompt


# ──────────────────────────────────────────────
# Feature: plan-data-synthesis, Property 8: 加权总分计算正确性
# Validates: Requirements 4.3
# ──────────────────────────────────────────────

class TestComputeWeightedScoreProperty8:
    """Property 8: 加权总分计算正确性。"""

    @given(
        tool_accuracy=st.integers(min_value=1, max_value=10),
        dependency_logic=st.integers(min_value=1, max_value=10),
        completeness=st.integers(min_value=1, max_value=10),
        efficiency=st.integers(min_value=1, max_value=10),
        thought_quality=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_weighted_score_matches_manual_calculation(
        self, tool_accuracy, dependency_logic, completeness, efficiency, thought_quality
    ):
        """**Validates: Requirements 4.3**
        For any 5 dimension scores (1-10), compute_weighted_score should equal
        tool_accuracy*0.3 + dependency_logic*0.2 + completeness*0.2 + efficiency*0.15 + thought_quality*0.15
        within 0.01 tolerance.
        """
        dimensions = {
            "tool_accuracy": {"score": tool_accuracy, "reason": "r"},
            "dependency_logic": {"score": dependency_logic, "reason": "r"},
            "completeness": {"score": completeness, "reason": "r"},
            "efficiency": {"score": efficiency, "reason": "r"},
            "thought_quality": {"score": thought_quality, "reason": "r"},
        }
        expected = (
            tool_accuracy * 0.3
            + dependency_logic * 0.2
            + completeness * 0.2
            + efficiency * 0.15
            + thought_quality * 0.15
        )
        result = compute_weighted_score(dimensions)
        assert abs(result - expected) <= 0.01, (
            f"Expected ~{expected}, got {result}"
        )


# ──────────────────────────────────────────────
# Unit tests — validate_evaluation
# ──────────────────────────────────────────────

class TestValidateEvaluationUnit:
    """Unit tests for validate_evaluation."""

    def _make_valid_evaluation(self) -> dict:
        return {
            "dimensions": {
                "tool_accuracy": {"score": 8, "reason": "Good tool usage"},
                "dependency_logic": {"score": 9, "reason": "Correct deps"},
                "completeness": {"score": 7, "reason": "Covers most needs"},
                "efficiency": {"score": 8, "reason": "No redundancy"},
                "thought_quality": {"score": 7, "reason": "Clear reasoning"},
            },
            "total_score": 7.8,
            "reasoning": "Overall good plan.",
        }

    def test_valid_evaluation_passes(self):
        assert validate_evaluation(self._make_valid_evaluation()) is True

    def test_missing_dimensions_key(self):
        ev = self._make_valid_evaluation()
        del ev["dimensions"]
        assert validate_evaluation(ev) is False

    def test_missing_one_dimension(self):
        ev = self._make_valid_evaluation()
        del ev["dimensions"]["efficiency"]
        assert validate_evaluation(ev) is False

    def test_score_below_range(self):
        ev = self._make_valid_evaluation()
        ev["dimensions"]["tool_accuracy"]["score"] = 0
        assert validate_evaluation(ev) is False

    def test_score_above_range(self):
        ev = self._make_valid_evaluation()
        ev["dimensions"]["completeness"]["score"] = 11
        assert validate_evaluation(ev) is False

    def test_score_not_int(self):
        ev = self._make_valid_evaluation()
        ev["dimensions"]["dependency_logic"]["score"] = 7.5
        assert validate_evaluation(ev) is False

    def test_empty_reason(self):
        ev = self._make_valid_evaluation()
        ev["dimensions"]["thought_quality"]["reason"] = "   "
        assert validate_evaluation(ev) is False

    def test_missing_total_score(self):
        ev = self._make_valid_evaluation()
        del ev["total_score"]
        assert validate_evaluation(ev) is False

    def test_missing_reasoning(self):
        ev = self._make_valid_evaluation()
        del ev["reasoning"]
        assert validate_evaluation(ev) is False

    def test_not_a_dict(self):
        assert validate_evaluation("not a dict") is False
        assert validate_evaluation(None) is False
        assert validate_evaluation([]) is False

    def test_dimensions_not_dict(self):
        ev = self._make_valid_evaluation()
        ev["dimensions"] = "string"
        assert validate_evaluation(ev) is False


# ──────────────────────────────────────────────
# Unit tests — compute_weighted_score
# ──────────────────────────────────────────────

class TestComputeWeightedScoreUnit:
    """Unit tests for compute_weighted_score with specific examples."""

    def test_all_tens(self):
        dims = {k: {"score": 10, "reason": "r"} for k in [
            "tool_accuracy", "dependency_logic", "completeness", "efficiency", "thought_quality"
        ]}
        assert compute_weighted_score(dims) == 10.0

    def test_all_ones(self):
        dims = {k: {"score": 1, "reason": "r"} for k in [
            "tool_accuracy", "dependency_logic", "completeness", "efficiency", "thought_quality"
        ]}
        assert compute_weighted_score(dims) == 1.0

    def test_known_example(self):
        dims = {
            "tool_accuracy": {"score": 8, "reason": "r"},
            "dependency_logic": {"score": 9, "reason": "r"},
            "completeness": {"score": 7, "reason": "r"},
            "efficiency": {"score": 8, "reason": "r"},
            "thought_quality": {"score": 7, "reason": "r"},
        }
        # 8*0.3 + 9*0.2 + 7*0.2 + 8*0.15 + 7*0.15 = 2.4 + 1.8 + 1.4 + 1.2 + 1.05 = 7.85
        assert compute_weighted_score(dims) == 7.85


# ──────────────────────────────────────────────
# Unit tests — build_eval_prompt
# ──────────────────────────────────────────────

class TestBuildEvalPromptUnit:
    """Unit tests for build_eval_prompt with specific examples."""

    def test_basic_prompt_structure(self):
        prompt = build_eval_prompt(
            "查询股票价格",
            {"get_stock": "获取股票信息"},
            {"fixed_question": "查询600519股价", "thought": "需要查股票", "steps": []},
        )
        assert "查询股票价格" in prompt
        assert "get_stock" in prompt
        assert "查询600519股价" in prompt
        assert "工具准确性" in prompt

    def test_prompt_contains_scoring_rules(self):
        prompt = build_eval_prompt("q", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []})
        # Check deduction rules from requirements 4.5
        assert "扣 5 分" in prompt  # 使用不存在的工具直接扣 5 分
        assert "依赖" in prompt
        assert "遗漏" in prompt
        assert "冗余" in prompt
        assert "空洞" in prompt
