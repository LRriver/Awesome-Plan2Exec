"""Tests for evaluate_plans.py — build_eval_prompt, compute_weighted_score, validate_evaluation."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate_plans import (
    build_eval_prompt,
    build_rule_based_evaluation,
    compute_weighted_score,
    evaluate_all_plans,
    load_existing_evaluation_keys,
    question_key,
    select_llm_judge_indices,
    validate_evaluation,
    get_rubric_criteria,
    get_eval_sample_n,
)
import json
import config


# ──────────────────────────────────────────────
# Shared strategies
# ──────────────────────────────────────────────

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

_difficulty_strategy = st.sampled_from(config.DIFFICULTY_LEVELS)


# ──────────────────────────────────────────────
# Property tests: build_eval_prompt
# ──────────────────────────────────────────────

class TestBuildEvalPromptProperty:
    """build_eval_prompt() 包含完整上下文（用户问题、工具集、规划、10维度评分规则）。"""

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy, difficulty=_difficulty_strategy)
    @settings(max_examples=50)
    def test_prompt_contains_user_query(self, user_query, tools, plan, difficulty):
        prompt = build_eval_prompt(user_query, tools, plan, difficulty)
        assert user_query in prompt

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy, difficulty=_difficulty_strategy)
    @settings(max_examples=50)
    def test_prompt_contains_tool_names(self, user_query, tools, plan, difficulty):
        prompt = build_eval_prompt(user_query, tools, plan, difficulty)
        for tool_name in tools:
            assert tool_name in prompt

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy, difficulty=_difficulty_strategy)
    @settings(max_examples=50)
    def test_prompt_contains_plan_content(self, user_query, tools, plan, difficulty):
        prompt = build_eval_prompt(user_query, tools, plan, difficulty)
        assert plan["fixed_question"] in prompt

    @given(user_query=_safe_text, tools=_tool_dict_strategy, plan=_plan_strategy, difficulty=_difficulty_strategy)
    @settings(max_examples=50)
    def test_prompt_contains_ten_dimensions(self, user_query, tools, plan, difficulty):
        """评分 Prompt 应包含 10 个评价维度的中文名称。"""
        prompt = build_eval_prompt(user_query, tools, plan, difficulty)
        assert "工具存在性" in prompt
        assert "工具语义匹配" in prompt
        assert "依赖合理性" in prompt
        assert "无循环依赖" in prompt
        assert "数据流完整性" in prompt
        assert "显性需求覆盖" in prompt
        assert "隐性需求识别" in prompt
        assert "规划简洁性" in prompt
        assert "推理深度" in prompt
        assert "思维一致性" in prompt

    def test_safety_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("有害请求", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "safety")
        assert "安全类问题" in prompt
        assert "拒绝" in prompt

    def test_parallel_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("并行任务", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "parallel")
        assert "并行问题" in prompt
        assert "并行" in prompt
        assert "多组并行" in prompt

    def test_complex_dependency_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("依赖任务", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "complex_dependency")
        assert "强依赖问题" in prompt
        assert "依赖链" in prompt

    def test_ambiguous_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("模糊问题", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "ambiguous")
        assert "模糊问题" in prompt
        assert "歧义" in prompt

    def test_adversarial_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("对抗问题", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "adversarial")
        assert "对抗性问题" in prompt
        assert "误导" in prompt

    def test_long_chain_difficulty_adds_special_guide(self):
        prompt = build_eval_prompt("长链条", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "long_chain")
        assert "长链条问题" in prompt
        assert "4 步" in prompt

    def test_prompt_contains_parallel_group_dependency_checks(self):
        prompt = build_eval_prompt("q", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "parallel")
        assert "parallel_group" in prompt
        assert "组间依赖" in prompt

    def test_prompt_contains_mixed_topology_scoring_example(self):
        prompt = build_eval_prompt("q", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "long_chain")
        assert "混合拓扑判分示例" in prompt
        assert "组内并行、组间依赖" in prompt

    def test_prompt_contains_criterion_level_rubric_output(self):
        prompt = build_eval_prompt("q", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "parallel")
        assert "rubric_criteria" in prompt
        assert "criterion_id" in prompt
        assert "parallel.no_serialization" in prompt

    def test_prompt_strips_training_metadata_from_plan(self):
        plan = {
            "fixed_question": "q",
            "thought": "t",
            "negative_type": "wrong_tool",
            "quality_bucket": "weak",
            "steps": [],
        }
        prompt = build_eval_prompt("q", {"t": "d"}, plan, "simple")
        assert "negative_type" not in prompt
        assert "quality_bucket" not in prompt

    def test_safety_rubric_contains_refusal_criterion(self):
        criteria = get_rubric_criteria("safety")
        assert any(c["criterion_id"] == "safety.refuse_harmful_action" for c in criteria)


# ──────────────────────────────────────────────
# Property tests: compute_weighted_score (10 dimensions)
# ──────────────────────────────────────────────

class TestComputeWeightedScoreProperty:
    """加权总分计算正确性（10维度）。"""

    @given(scores=st.fixed_dictionaries({
        dim: st.integers(min_value=1, max_value=10)
        for dim in config.EVAL_WEIGHTS
    }))
    @settings(max_examples=100)
    def test_weighted_score_matches_manual(self, scores):
        dimensions = {dim: {"score": s, "reason": "r"} for dim, s in scores.items()}
        expected = sum(scores[dim] * config.EVAL_WEIGHTS[dim] for dim in config.EVAL_WEIGHTS)
        result = compute_weighted_score(dimensions)
        assert abs(result - round(expected, 2)) <= 0.01


# ──────────────────────────────────────────────
# Unit tests — validate_evaluation (10 dimensions)
# ──────────────────────────────────────────────

class TestValidateEvaluationUnit:

    def _make_valid_evaluation(self) -> dict:
        dims = {}
        for dim in config.EVAL_WEIGHTS:
            dims[dim] = {"score": 7, "reason": "Reasonable"}
        return {
            "dimensions": dims,
            "rubric_criteria": [
                {
                    "criterion_id": "tool.exists",
                    "score": 1,
                    "severity": "none",
                    "evidence": "All tools exist.",
                }
            ],
            "total_score": 7.0,
            "reasoning": "Overall assessment.",
        }

    def test_valid_evaluation_passes(self):
        assert validate_evaluation(self._make_valid_evaluation()) is True

    def test_missing_dimensions_key(self):
        ev = self._make_valid_evaluation()
        del ev["dimensions"]
        assert validate_evaluation(ev) is False

    def test_missing_one_dimension(self):
        ev = self._make_valid_evaluation()
        first_dim = list(config.EVAL_WEIGHTS.keys())[0]
        del ev["dimensions"][first_dim]
        assert validate_evaluation(ev) is False

    def test_score_below_range(self):
        ev = self._make_valid_evaluation()
        first_dim = list(config.EVAL_WEIGHTS.keys())[0]
        ev["dimensions"][first_dim]["score"] = 0
        assert validate_evaluation(ev) is False

    def test_score_above_range(self):
        ev = self._make_valid_evaluation()
        first_dim = list(config.EVAL_WEIGHTS.keys())[0]
        ev["dimensions"][first_dim]["score"] = 11
        assert validate_evaluation(ev) is False

    def test_float_score_accepted(self):
        """新版本允许 float 分数（中位数可能是小数）。"""
        ev = self._make_valid_evaluation()
        first_dim = list(config.EVAL_WEIGHTS.keys())[0]
        ev["dimensions"][first_dim]["score"] = 7.5
        assert validate_evaluation(ev) is True

    def test_empty_reason(self):
        ev = self._make_valid_evaluation()
        first_dim = list(config.EVAL_WEIGHTS.keys())[0]
        ev["dimensions"][first_dim]["reason"] = "   "
        assert validate_evaluation(ev) is False

    def test_missing_total_score(self):
        ev = self._make_valid_evaluation()
        del ev["total_score"]
        assert validate_evaluation(ev) is False

    def test_missing_reasoning(self):
        ev = self._make_valid_evaluation()
        del ev["reasoning"]
        assert validate_evaluation(ev) is False

    def test_invalid_rubric_criterion_is_rejected(self):
        ev = self._make_valid_evaluation()
        ev["rubric_criteria"][0]["severity"] = "catastrophic"
        assert validate_evaluation(ev) is False

    def test_not_a_dict(self):
        assert validate_evaluation("not a dict") is False
        assert validate_evaluation(None) is False
        assert validate_evaluation([]) is False


class TestResumeHelpers:
    def test_question_key_uses_scenario_difficulty_query(self):
        record = {"scenario": "s", "difficulty": "safety", "query": "q", "evaluated_plans": []}
        assert question_key(record) == ("s", "safety", "q")

    def test_load_existing_evaluation_keys_skips_bad_lines(self, tmp_path):
        path = tmp_path / "evaluated_plans.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"scenario": "s1", "difficulty": "simple", "query": "q1", "evaluated_plans": []}),
                "{bad json",
                json.dumps({"scenario": "s2", "difficulty": "safety", "query": "q2", "evaluated_plans": []}),
            ]) + "\n",
            encoding="utf-8",
        )

        assert load_existing_evaluation_keys(path) == {
            ("s1", "simple", "q1"),
            ("s2", "safety", "q2"),
        }


# ──────────────────────────────────────────────
# Unit tests — compute_weighted_score
# ──────────────────────────────────────────────

class TestComputeWeightedScoreUnit:

    def test_all_tens(self):
        dims = {k: {"score": 10, "reason": "r"} for k in config.EVAL_WEIGHTS}
        assert compute_weighted_score(dims) == 10.0

    def test_all_ones(self):
        dims = {k: {"score": 1, "reason": "r"} for k in config.EVAL_WEIGHTS}
        assert compute_weighted_score(dims) == 1.0

    def test_known_example(self):
        """已知输入的精确计算验证。"""
        dims = {dim: {"score": 5, "reason": "r"} for dim in config.EVAL_WEIGHTS}
        # 所有维度都是5，权重之和为1.0，所以结果应该是5.0
        assert compute_weighted_score(dims) == 5.0


# ──────────────────────────────────────────────
# Unit tests — build_eval_prompt
# ──────────────────────────────────────────────

class TestBuildEvalPromptUnit:

    def test_basic_prompt_structure(self):
        prompt = build_eval_prompt(
            "查询股票价格",
            {"get_stock": "获取股票信息"},
            {"fixed_question": "查询600519股价", "thought": "需要查股票", "steps": []},
            "simple",
        )
        assert "查询股票价格" in prompt
        assert "get_stock" in prompt
        assert "查询600519股价" in prompt
        assert "工具存在性" in prompt

    def test_prompt_contains_scoring_rules(self):
        prompt = build_eval_prompt("q", {"t": "d"}, {"fixed_question": "q", "thought": "t", "steps": []}, "simple")
        assert "扣 3 分" in prompt  # 工具不存在扣分
        assert "依赖" in prompt
        assert "遗漏" in prompt
        assert "冗余" in prompt
        assert "套话" in prompt


class TestTieredEvalBudget:
    def test_eval_sample_n_uses_difficulty_override(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "EVAL_SAMPLE_N", 1)
            mp.setattr(config, "EVAL_SAMPLE_N_BY_DIFFICULTY", {"safety": 2}, raising=False)
            assert get_eval_sample_n("safety") == 2
            assert get_eval_sample_n("simple") == 1


class TestRuleBasedEvaluation:
    def test_rule_based_evaluation_penalizes_fake_tool(self):
        plan = {
            "fixed_question": "q",
            "thought": "t",
            "steps": [
                {"thought": "s", "title": "s", "content": "c", "tools": ["fake_tool"], "dependencies": None}
            ],
        }
        evaluation = build_rule_based_evaluation("q", {"real_tool": "desc"}, plan, "simple")
        assert evaluation["dimensions"]["tool_existence"]["score"] < 10
        assert evaluation["total_score"] < 8
        assert evaluation["evaluation_source"] == "rule"

    def test_selective_judge_keeps_positive_and_limited_negatives(self):
        plans = [
            {"fixed_question": "p", "thought": "t", "steps": [], "negative_type": None},
            {"fixed_question": "n1", "thought": "t", "steps": [], "negative_type": "wrong_tool"},
            {"fixed_question": "n2", "thought": "t", "steps": [], "negative_type": "missing_dependency"},
        ]
        evaluations = [
            {"total_score": 9.0},
            {"total_score": 6.0},
            {"total_score": 4.0},
        ]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "EVAL_JUDGE_POLICY", "selective", raising=False)
            mp.setattr(config, "EVAL_MAX_LLM_NEGATIVES_BY_DIFFICULTY", {"simple": 1}, raising=False)
            assert select_llm_judge_indices(plans, evaluations, "simple") == {0, 1}

    @pytest.mark.asyncio
    async def test_evaluate_all_plans_uses_rule_fallback_for_unselected_candidates(self):
        plans = [
            {"fixed_question": "p", "thought": "t", "steps": [{"thought": "s", "title": "s", "content": "c", "tools": None, "dependencies": None}], "negative_type": None},
            {"fixed_question": "n1", "thought": "t", "steps": [{"thought": "s", "title": "s", "content": "c", "tools": ["fake_tool"], "dependencies": None}], "negative_type": "fake_tool"},
            {"fixed_question": "n2", "thought": "t", "steps": [{"thought": "s", "title": "s", "content": "c", "tools": None, "dependencies": None}], "negative_type": "missing_dependency"},
        ]
        question_data = {
            "scenario": "s",
            "tools": {"real_tool": "desc"},
            "difficulty": "simple",
            "query": "q",
            "plans": plans,
        }
        writer = AsyncMock()
        llm_result = {
            "plan": plans[0],
            "evaluation": {
                "dimensions": {dim: {"score": 9, "reason": "r"} for dim in config.EVAL_WEIGHTS},
                "rubric_criteria": [{"criterion_id": "tool.exists", "score": 2, "severity": "none", "evidence": "ok"}],
                "total_score": 9.0,
                "reasoning": "r",
                "evaluation_source": "llm",
            },
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "EVAL_JUDGE_POLICY", "selective", raising=False)
            mp.setattr(config, "EVAL_MAX_LLM_NEGATIVES_BY_DIFFICULTY", {"simple": 0}, raising=False)
            with patch("evaluate_plans.evaluate_plan", new_callable=AsyncMock, return_value=llm_result) as mock_eval:
                result = await evaluate_all_plans(None, question_data, writer)

        assert len(result["evaluated_plans"]) == 3
        assert mock_eval.await_count == 1
        sources = [item["evaluation"]["evaluation_source"] for item in result["evaluated_plans"]]
        assert sources.count("llm") == 1
        assert sources.count("rule") == 2
