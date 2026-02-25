"""Tests for build_preference.py — build_preference_pair selection logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import config
from build_preference import build_preference_pair


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _make_evaluated_plan(score: float) -> dict:
    """Build a minimal but structurally valid evaluated_plan entry."""
    return {
        "plan": {
            "fixed_question": "q",
            "thought": "t",
            "steps": [{"thought": "s", "title": "s", "content": "c", "tools": None, "dependencies": None}],
        },
        "evaluation": {
            "dimensions": {
                "tool_accuracy": {"score": 8, "reason": "r"},
                "dependency_logic": {"score": 8, "reason": "r"},
                "completeness": {"score": 8, "reason": "r"},
                "efficiency": {"score": 8, "reason": "r"},
                "thought_quality": {"score": 8, "reason": "r"},
            },
            "total_score": score,
            "reasoning": "r",
        },
    }


def _make_question_data(scores: list[float]) -> dict:
    """Build a mock question_data dict with evaluated_plans at given scores."""
    return {
        "scenario": "test_scenario",
        "tools": {"tool_a": "desc_a"},
        "difficulty": "simple",
        "query": "test query",
        "evaluated_plans": [_make_evaluated_plan(s) for s in scores],
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
        """All scores within 1.0 of each other → gap < 2.0 → None."""
        result = build_preference_pair(_make_question_data([8.0, 7.5, 7.0, 6.5, 6.1]))
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
