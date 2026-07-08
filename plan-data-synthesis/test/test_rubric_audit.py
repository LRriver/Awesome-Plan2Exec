"""Tests for rubric audit reports."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rubric_audit import build_audit_report, write_audit_report


def test_build_audit_report_aggregates_negative_types_and_criteria():
    """Iteration reports summarize score distribution and rubric failure rates."""
    records = [
        {
            "difficulty": "parallel",
            "evaluated_plans": [
                {
                    "plan": {"negative_type": None},
                    "evaluation": {
                        "final_score": 9.0,
                        "evaluation_source": "llm",
                        "rubric_criteria": [
                            {"criterion_id": "tool_existence", "score": 1.0},
                            {"criterion_id": "dependency_logic", "score": 0.8},
                        ],
                    },
                },
                {
                    "plan": {"negative_type": "wrong_tool"},
                    "evaluation": {
                        "final_score": 4.0,
                        "evaluation_source": "rule",
                        "rubric_criteria": [
                            {"criterion_id": "tool_existence", "score": 0.0},
                            {"criterion_id": "dependency_logic", "score": 0.5},
                        ],
                    },
                },
            ],
        }
    ]

    report = build_audit_report(records)

    assert report["overall"]["candidate_count"] == 2
    assert report["by_negative_type"]["positive"]["avg_score"] == 9.0
    assert report["by_negative_type"]["wrong_tool"]["avg_score"] == 4.0
    assert report["by_evaluation_source"]["llm"] == 1
    assert report["rubric_criteria"]["tool_existence"]["failure_rate"] == 0.5


def test_write_audit_report_uses_configured_path(tmp_path, monkeypatch):
    """Audit report is an iteration artifact with an explicit path."""
    path = tmp_path / "rubric_audit_report.json"
    monkeypatch.setattr(config, "RUBRIC_AUDIT_REPORT_FILE", path, raising=False)

    write_audit_report([])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["overall"]["candidate_count"] == 0
