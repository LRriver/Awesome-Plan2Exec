"""
Plan2Exec 数据合成流水线 — 入口脚本
串联四个阶段，支持从指定阶段开始执行。
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config


STAGE_NAMES = {
    1: "问题生成 (generate_questions)",
    2: "规划采样 (plan_sampling)",
    3: "自动评分 (evaluate_plans)",
    4: "偏好数据提取 (build_preference)",
}

DEFAULT_RUN_MODE_OVERRIDES = {
    "production": {
        "EVAL_JUDGE_POLICY": "selective",
        "EVAL_SAMPLE_N": 1,
        "ENABLE_RUBRIC_AUDIT": False,
        "RUBRIC_AUDIT_SAMPLE_RATE": 0.0,
    },
    "iteration": {
        "EVAL_JUDGE_POLICY": "selective",
        "EVAL_SAMPLE_N": 1,
        "ENABLE_RUBRIC_AUDIT": True,
        "RUBRIC_AUDIT_SAMPLE_RATE": 0.05,
    },
}


def get_stage_output_files() -> dict[int, Path]:
    """Return stage outputs from the current config."""
    return {
        1: config.QUESTIONS_FILE,
        2: config.PLAN_SAMPLES_FILE,
        3: config.EVALUATED_PLANS_FILE,
        4: config.PREFERENCE_DATA_FILE,
    }


def update_output_paths(output_dir: str | Path) -> None:
    """Move all pipeline outputs to a single directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = output_dir
    config.QUESTIONS_FILE = output_dir / "questions.jsonl"
    config.PLAN_SAMPLES_FILE = output_dir / "plan_samples.jsonl"
    config.EVALUATED_PLANS_FILE = output_dir / "evaluated_plans.jsonl"
    config.PREFERENCE_DATA_FILE = output_dir / "preference_data.jsonl"
    config.RANKED_CANDIDATES_FILE = output_dir / "ranked_candidates.jsonl"
    config.RUN_MANIFEST_FILE = output_dir / "run_manifest.json"
    config.RUBRIC_AUDIT_REPORT_FILE = output_dir / "rubric_audit_report.json"


def apply_strategy_mode(
    mode: str | None = None,
    strategy_checkpoint: str | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Apply a strategy mode before running the pipeline.

    production: fixed extraction checkpoint, no rubric-audit loop.
    iteration: pilot/audit mode for rubric and budget tuning.
    """
    mode = mode or getattr(config, "SYNTHESIS_MODE", "production")
    overrides_by_mode = getattr(config, "RUN_MODE_OVERRIDES", DEFAULT_RUN_MODE_OVERRIDES)
    if mode not in overrides_by_mode:
        valid_modes = ", ".join(sorted(overrides_by_mode))
        raise ValueError(f"unknown synthesis mode '{mode}', expected one of: {valid_modes}")

    if output_dir is not None:
        update_output_paths(output_dir)
    else:
        update_output_paths(getattr(config, "OUTPUT_DIR", Path("output")))

    config.SYNTHESIS_MODE = mode
    config.STRATEGY_CHECKPOINT = strategy_checkpoint or getattr(
        config,
        "STRATEGY_CHECKPOINT",
        "controlled_v1",
    )
    for key, value in overrides_by_mode[mode].items():
        setattr(config, key, value)


def build_run_manifest(start_stage: int) -> dict:
    """Build a reproducibility manifest for this pipeline run."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": getattr(config, "SYNTHESIS_MODE", "production"),
        "strategy_checkpoint": getattr(config, "STRATEGY_CHECKPOINT", "controlled_v1"),
        "resume": getattr(config, "RESUME", False),
        "start_stage": start_stage,
        "input_file": str(config.INPUT_FILE),
        "output_dir": str(config.OUTPUT_DIR),
        "output_files": {
            "questions": str(config.QUESTIONS_FILE),
            "plan_samples": str(config.PLAN_SAMPLES_FILE),
            "evaluated_plans": str(config.EVALUATED_PLANS_FILE),
            "preference_data": str(config.PREFERENCE_DATA_FILE),
            "ranked_candidates": str(getattr(config, "RANKED_CANDIDATES_FILE", "")),
        },
        "settings": {
            "scenario_limit": getattr(config, "SCENARIO_LIMIT", 0),
            "eval_judge_policy": getattr(config, "EVAL_JUDGE_POLICY", "all"),
            "eval_sample_n": getattr(config, "EVAL_SAMPLE_N", 1),
            "enable_rubric_audit": getattr(config, "ENABLE_RUBRIC_AUDIT", False),
            "rubric_audit_sample_rate": getattr(config, "RUBRIC_AUDIT_SAMPLE_RATE", 0.0),
            "question_model_profile": getattr(config, "QUESTION_MODEL_PROFILE", ""),
            "plan_model_profile": getattr(config, "PLAN_MODEL_PROFILE", ""),
            "negative_plan_model_profile": getattr(config, "NEGATIVE_PLAN_MODEL_PROFILE", ""),
            "eval_model_profile": getattr(config, "EVAL_MODEL_PROFILE", ""),
        },
    }


def write_run_manifest(start_stage: int) -> None:
    """Persist run metadata for reproducibility."""
    manifest_path = getattr(config, "RUN_MANIFEST_FILE", config.OUTPUT_DIR / "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_run_manifest(start_stage), f, ensure_ascii=False, indent=2)


async def run_pipeline(start_stage: int):
    """Run the pipeline from the specified stage."""
    write_run_manifest(start_stage)
    print(f"[INFO] 从阶段 {start_stage} 开始执行流水线...")
    print(
        "[INFO] mode="
        f"{getattr(config, 'SYNTHESIS_MODE', 'production')}, "
        f"strategy_checkpoint={getattr(config, 'STRATEGY_CHECKPOINT', 'controlled_v1')}"
    )
    if getattr(config, "RESUME", False):
        print("[INFO] resume=true，将跳过已完成的 LLM 记录并追加新结果")

    stage_output_files = get_stage_output_files()
    for stage in range(start_stage, 5):
        output_file = stage_output_files[stage]
        if output_file.exists():
            if getattr(config, "RESUME", False) and stage < 4:
                print(f"[INFO] 阶段 {stage} 的输出文件已存在，将断点续写: {output_file}")
            elif getattr(config, "RESUME", False) and stage == 4:
                print(f"[INFO] 阶段 {stage} 为本地构建，将从评分结果重新生成: {output_file}")
            else:
                print(f"[WARN] 阶段 {stage} 的输出文件已存在，将覆盖: {output_file}")

        print(f"\n{'='*60}")
        print(f"[INFO] 开始执行阶段 {stage}: {STAGE_NAMES[stage]}")
        print(f"{'='*60}\n")

        if stage == 1:
            from generate_questions import main as stage1_main
            await stage1_main()
        elif stage == 2:
            from plan_sampling import main as stage2_main
            await stage2_main()
        elif stage == 3:
            from evaluate_plans import main as stage3_main
            await stage3_main()
        elif stage == 4:
            from build_preference import main as stage4_main
            stage4_main()
            if getattr(config, "ENABLE_RUBRIC_AUDIT", False):
                from rubric_audit import main as audit_main
                audit_main()

        print(f"\n[INFO] 阶段 {stage} 完成: {STAGE_NAMES[stage]}")

    print(f"\n{'='*60}")
    print("[INFO] 流水线执行完毕！")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Plan2Exec 数据合成流水线")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="从指定阶段开始执行 (1=问题生成, 2=规划采样, 3=自动评分, 4=偏好数据提取)",
    )
    parser.add_argument(
        "--mode",
        choices=["production", "iteration"],
        default=getattr(config, "SYNTHESIS_MODE", "production"),
        help="production=固定策略 checkpoint 抽取; iteration=pilot/rubric 策略迭代",
    )
    parser.add_argument(
        "--strategy-checkpoint",
        default=getattr(config, "STRATEGY_CHECKPOINT", "controlled_v1"),
        help="写入 manifest 的策略 checkpoint 名称",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="覆盖输出目录，便于 production/iteration 分目录保存",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：阶段 1/2/3 跳过已有记录并追加，阶段 4 从完整评分结果重建",
    )
    args = parser.parse_args()
    apply_strategy_mode(
        args.mode,
        strategy_checkpoint=args.strategy_checkpoint,
        output_dir=args.output_dir,
    )
    config.RESUME = args.resume
    asyncio.run(run_pipeline(args.start_stage))


if __name__ == "__main__":
    main()
