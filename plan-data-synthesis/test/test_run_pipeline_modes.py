"""Tests for run_pipeline strategy modes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from run_pipeline import apply_strategy_mode, build_run_manifest


def test_production_mode_freezes_strategy_checkpoint(tmp_path, monkeypatch):
    """Production mode records the fixed strategy checkpoint and disables audit flow."""
    monkeypatch.setattr(config, "EVAL_JUDGE_POLICY", "all", raising=False)
    monkeypatch.setattr(config, "ENABLE_RUBRIC_AUDIT", True, raising=False)

    apply_strategy_mode("production", strategy_checkpoint="controlled_v1", output_dir=tmp_path)

    assert config.SYNTHESIS_MODE == "production"
    assert config.STRATEGY_CHECKPOINT == "controlled_v1"
    assert config.EVAL_JUDGE_POLICY == "selective"
    assert config.EVAL_SAMPLE_N == 1
    assert config.ENABLE_RUBRIC_AUDIT is False
    assert config.QUESTIONS_FILE == tmp_path / "questions.jsonl"
    assert config.RUN_MANIFEST_FILE == tmp_path / "run_manifest.json"


def test_iteration_mode_enables_audit_without_changing_checkpoint(tmp_path, monkeypatch):
    """Iteration mode is a separate audit flow, not the default production extractor."""
    monkeypatch.setattr(config, "ENABLE_RUBRIC_AUDIT", False, raising=False)

    apply_strategy_mode("iteration", strategy_checkpoint="controlled_v1-dev", output_dir=tmp_path)

    assert config.SYNTHESIS_MODE == "iteration"
    assert config.STRATEGY_CHECKPOINT == "controlled_v1-dev"
    assert config.EVAL_JUDGE_POLICY == "selective"
    assert config.ENABLE_RUBRIC_AUDIT is True
    assert config.RUBRIC_AUDIT_SAMPLE_RATE > 0
    assert config.RUBRIC_AUDIT_REPORT_FILE == tmp_path / "rubric_audit_report.json"


def test_run_manifest_includes_mode_checkpoint_and_outputs(tmp_path, monkeypatch):
    """Run manifest makes production extraction reproducible."""
    apply_strategy_mode("production", strategy_checkpoint="controlled_v1", output_dir=tmp_path)
    monkeypatch.setattr(config, "RESUME", True, raising=False)

    manifest = build_run_manifest(start_stage=2)

    assert manifest["mode"] == "production"
    assert manifest["strategy_checkpoint"] == "controlled_v1"
    assert manifest["resume"] is True
    assert manifest["start_stage"] == 2
    assert manifest["output_files"]["questions"].endswith("questions.jsonl")
    assert manifest["settings"]["eval_judge_policy"] == "selective"
