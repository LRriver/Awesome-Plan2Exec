"""Tests for build_preference.py — build_preference_pair selection logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import config
from build_preference import build_preference_pair, build_ranked_candidates, assign_quality_bucket


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _make_evaluated_plan(score: float, tool_name: str = "tool_a") -> dict:
    """Build a minimal but structurally valid evaluated_plan entry.
    
    Different tool_name values ensure plans have different tool sequences,
    avoiding the plans_are_similar filter.
    """
    return {
        "plan": {
            "fixed_question": "q",
            "thought": "t",
            "steps": [{"thought": "s", "title": "s", "content": "c",
                        "tools": [tool_name], "dependencies": None}],
        },
        "evaluation": {
            "dimensions": {dim: {"score": 8, "reason": "r"} for dim in config.EVAL_WEIGHTS},
            "total_score": score,
            "reasoning": "r",
        },
    }


def _make_question_data(scores: list[float]) -> dict:
    """Build a mock question_data dict with evaluated_plans at given scores.
    
    Each plan uses a different tool name to avoid the plans_are_similar filter.
    """
    tool_names = [f"tool_{chr(ord('a') + i)}" for i in range(len(scores))]
    return {
        "scenario": "test_scenario",
        "tools": {"tool_a": "desc_a"},
        "difficulty": "simple",
        "query": "test query",
        "evaluated_plans": [_make_evaluated_plan(s, t) for s, t in zip(scores, tool_names)],
    }


# ──────────────────────────────────────────────
# Feature: plan-data-synthesis, Property 10: 偏好数据对有效性
# Validates: Requirements 5.2, 5.4, 5.5
# ──────────────────────────────────────────────


class TestPreferencePairValidityProperty10:
    """Property 10: 偏好数据对有效性 — hypothesis 属性测试"""

    @given(scores=st.lists(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=5, max_size=5))
    @settings(max_examples=100)
    def test_chosen_is_max_score(self, scores: list[float]):
        """**Validates: Requirements 5.2**
        When build_preference_pair returns a result, chosen_score must be the highest score."""
        question_data = _make_question_data(scores)
        result = build_preference_pair(question_data)
        if result is not None:
            assert result["chosen_score"] == max(scores)

    @given(scores=st.lists(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=5, max_size=5))
    @settings(max_examples=100)
    def test_rejected_less_than_chosen(self, scores: list[float]):
        """**Validates: Requirements 5.4**
        When build_preference_pair returns a result, rejected_score < chosen_score."""
        question_data = _make_question_data(scores)
        result = build_preference_pair(question_data)
        if result is not None:
            assert result["rejected_score"] < result["chosen_score"]

    @given(scores=st.lists(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=5, max_size=5))
    @settings(max_examples=100)
    def test_score_gap_meets_minimum(self, scores: list[float]):
        """**Validates: Requirements 5.5**
        When build_preference_pair returns a result, chosen - rejected >= MIN_SCORE_GAP."""
        question_data = _make_question_data(scores)
        result = build_preference_pair(question_data)
        if result is not None:
            assert result["chosen_score"] - result["rejected_score"] >= config.MIN_SCORE_GAP


# ──────────────────────────────────────────────
# Unit tests for build_preference_pair
# ──────────────────────────────────────────────


class TestBuildPreferencePairUnit:
    """Unit tests covering specific examples and edge cases."""

    def test_clear_score_gap_returns_valid_pair(self):
        """Scores [9.0, 7.0, 5.0, 3.0, 2.0] should produce a valid preference pair."""
        result = build_preference_pair(_make_question_data([9.0, 7.0, 5.0, 3.0, 2.0]))
        assert result is not None
        assert result["chosen_score"] == 9.0
        assert result["rejected_score"] < 9.0
        assert result["chosen_score"] - result["rejected_score"] >= config.MIN_SCORE_GAP

    def test_all_same_scores_returns_none(self):
        """All identical scores → no valid pair (gap < MIN_SCORE_GAP)."""
        result = build_preference_pair(_make_question_data([7.0, 7.0, 7.0, 7.0, 7.0]))
        assert result is None

    def test_score_gap_below_threshold_returns_none(self):
        """All scores within 0.4 of each other → gap < 0.5 → None."""
        result = build_preference_pair(_make_question_data([8.0, 7.8, 7.7, 7.6, 7.65]))
        assert result is None

    def test_fewer_than_two_plans_returns_none(self):
        """Fewer than 2 evaluated plans → None."""
        result = build_preference_pair(_make_question_data([9.0]))
        assert result is None
        result = build_preference_pair(_make_question_data([]))
        assert result is None

    def test_rejected_plan_has_valid_structure(self):
        """The rejected_plan in the output must pass PlanOutput structure validation."""
        result = build_preference_pair(_make_question_data([9.0, 7.0, 5.0, 3.0, 2.0]))
        assert result is not None
        rp = result["rejected_plan"]
        assert isinstance(rp["fixed_question"], str)
        assert isinstance(rp["thought"], str)
        assert isinstance(rp["steps"], list) and len(rp["steps"]) > 0
        step = rp["steps"][0]
        assert "thought" in step
        assert "title" in step
        assert "content" in step
        assert "tools" in step
        assert "dependencies" in step


class TestRankedCandidates:
    def test_assign_quality_bucket(self):
        assert assign_quality_bucket(8.7) == "strong"
        assert assign_quality_bucket(6.4) == "medium"
        assert assign_quality_bucket(3.2) == "weak"
        assert assign_quality_bucket(1.5) == "invalid"

    def test_build_ranked_candidates_preserves_score_order_and_metadata(self):
        question_data = _make_question_data([6.0, 9.0, 3.0])
        question_data["evaluated_plans"][2]["plan"]["negative_type"] = "wrong_tool"
        question_data["evaluated_plans"][2]["plan"]["quality_bucket"] = "weak"

        ranked = build_ranked_candidates(question_data)

        assert [item["score"] for item in ranked["candidates"]] == [9.0, 6.0, 3.0]
        assert ranked["candidates"][0]["quality_bucket"] == "strong"
        assert ranked["candidates"][1]["quality_bucket"] == "medium"
        assert ranked["candidates"][2]["quality_bucket"] == "weak"
        assert ranked["candidates"][2]["negative_type"] == "wrong_tool"
        assert "negative_type" not in ranked["candidates"][2]["plan"]
        assert "quality_bucket" not in ranked["candidates"][2]["plan"]

    def test_preference_pair_strips_plan_metadata(self):
        question_data = _make_question_data([9.0, 4.0])
        question_data["evaluated_plans"][1]["plan"]["negative_type"] = "wrong_tool"
        question_data["evaluated_plans"][1]["plan"]["quality_bucket"] = "weak"

        result = build_preference_pair(question_data)

        assert result is not None
        assert result["rejected_negative_type"] == "wrong_tool"
        assert "negative_type" not in result["rejected_plan"]
        assert "quality_bucket" not in result["rejected_plan"]
