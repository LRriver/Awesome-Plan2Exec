"""Tests for generate_questions.py — build_question_prompt and load_scenarios."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the parent module directory is on sys.path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_questions import SELECTED_SCENARIOS, build_question_prompt, load_scenarios


# ──────────────────────────────────────────────
# Property 4: 问题生成 Prompt 包含场景与工具集
# Feature: plan-data-synthesis, Property 4: 问题生成 Prompt 包含场景与工具集
# Validates: Requirements 2.7
# ──────────────────────────────────────────────

class TestBuildQuestionPromptProperty:
    """Property-based tests for build_question_prompt."""

    @given(
        scenario=st.text(min_size=1),
        tools=st.dictionaries(
            st.text(min_size=1, alphabet=st.characters(whitelist_categories=("L", "N"))),
            st.text(min_size=1),
            min_size=1,
        ),
    )
    @settings(max_examples=100)
    def test_prompt_contains_scenario_and_tools(self, scenario: str, tools: dict):
        """**Validates: Requirements 2.7**

        For any scenario name and tools dict, the built prompt must contain
        the scenario name and every tool name from the dict.
        """
        prompt = build_question_prompt(scenario, tools)

        # Prompt must contain the scenario name
        assert scenario in prompt, f"Scenario '{scenario}' not found in prompt"

        # Prompt must contain each tool name
        for tool_name in tools:
            assert tool_name in prompt, f"Tool '{tool_name}' not found in prompt"


# ──────────────────────────────────────────────
# Property 1: 场景选取数量不变量
# Feature: plan-data-synthesis, Property 1: 场景选取数量不变量
# Validates: Requirements 1.2
# ──────────────────────────────────────────────

class TestLoadScenariosProperty:
    """Property-based test for load_scenarios count invariant."""

    @settings(max_examples=100)
    @given(
        extra_scenarios=st.lists(
            st.text(
                min_size=3,
                alphabet=st.characters(whitelist_categories=("L", "N")),
            ),
            min_size=38,
            max_size=38,
        ),
    )
    def test_load_scenarios_returns_12_to_13(self, extra_scenarios: list[str], tmp_path_factory):
        """**Validates: Requirements 1.2**

        Given a JSONL file containing all 13 selected scenarios plus extras
        (51 total), load_scenarios should return 12-13 items.
        """
        # Build 51 mock scenario records: 13 selected + 38 extras
        records = []
        for name in SELECTED_SCENARIOS:
            records.append(json.dumps({
                "scenario": name,
                "tools": {"tool_a": "desc_a"},
                "tools_count": 15,
            }, ensure_ascii=False))

        for name in extra_scenarios:
            # Ensure extras don't collide with selected names
            unique_name = f"extra_{name}"
            records.append(json.dumps({
                "scenario": unique_name,
                "tools": {"tool_b": "desc_b"},
                "tools_count": 12,
            }, ensure_ascii=False))

        # Write to a temp JSONL file
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp_file.write("\n".join(records) + "\n")
        tmp_file.close()

        # Patch config.INPUT_FILE to point to our temp file
        with patch("generate_questions.config") as mock_config:
            mock_config.INPUT_FILE = Path(tmp_file.name)
            result = load_scenarios()

        assert 12 <= len(result) <= 13, (
            f"Expected 12-13 scenarios, got {len(result)}"
        )

        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)


# ──────────────────────────────────────────────
# Unit tests — build_question_prompt
# ──────────────────────────────────────────────

class TestBuildQuestionPromptUnit:
    """Unit tests for build_question_prompt."""

    def test_basic_prompt_structure(self):
        prompt = build_question_prompt("旅游出行天气规划", {"get_weather": "获取天气"})
        assert "旅游出行天气规划" in prompt
        assert "get_weather" in prompt
        assert "获取天气" in prompt

    def test_multiple_tools_all_present(self):
        tools = {"tool_a": "desc_a", "tool_b": "desc_b", "tool_c": "desc_c"}
        prompt = build_question_prompt("测试场景", tools)
        for name, desc in tools.items():
            assert name in prompt
            assert desc in prompt

    def test_prompt_mentions_four_difficulties(self):
        prompt = build_question_prompt("场景", {"t": "d"})
        assert "简单" in prompt
        assert "并行" in prompt
        assert "复杂依赖" in prompt
        assert "闲聊" in prompt


# ──────────────────────────────────────────────
# Unit tests — load_scenarios
# ──────────────────────────────────────────────

class TestLoadScenariosUnit:
    """Unit tests for load_scenarios."""

    def test_loads_exactly_13_from_full_data(self):
        """With all 13 selected scenarios present, should return 13."""
        records = []
        for name in SELECTED_SCENARIOS:
            records.append(json.dumps({
                "scenario": name,
                "tools": {"t": "d"},
                "tools_count": 15,
            }, ensure_ascii=False))
        # Add some non-selected scenarios
        for i in range(38):
            records.append(json.dumps({
                "scenario": f"unselected_{i}",
                "tools": {"t": "d"},
                "tools_count": 10,
            }, ensure_ascii=False))

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp_file.write("\n".join(records) + "\n")
        tmp_file.close()

        with patch("generate_questions.config") as mock_config:
            mock_config.INPUT_FILE = Path(tmp_file.name)
            result = load_scenarios()

        assert len(result) == 13
        Path(tmp_file.name).unlink(missing_ok=True)

    def test_missing_file_exits(self):
        """Should sys.exit(1) when input file doesn't exist."""
        with patch("generate_questions.config") as mock_config:
            mock_config.INPUT_FILE = Path("/nonexistent/path/data.jsonl")
            with pytest.raises(SystemExit) as exc_info:
                load_scenarios()
            assert exc_info.value.code == 1
