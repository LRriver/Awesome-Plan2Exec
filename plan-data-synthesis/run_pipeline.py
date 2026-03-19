"""
Plan2Exec 数据合成流水线 — 入口脚本
串联四个阶段，支持从指定阶段开始执行。
"""
import argparse
import asyncio
import sys

import config


STAGE_NAMES = {
    1: "问题生成 (generate_questions)",
    2: "规划采样 (plan_sampling)",
    3: "自动评分 (evaluate_plans)",
    4: "偏好数据提取 (build_preference)",
}

STAGE_OUTPUT_FILES = {
    1: config.QUESTIONS_FILE,
    2: config.PLAN_SAMPLES_FILE,
    3: config.EVALUATED_PLANS_FILE,
    4: config.PREFERENCE_DATA_FILE,
}


async def run_pipeline(start_stage: int):
    """Run the pipeline from the specified stage."""
    print(f"[INFO] 从阶段 {start_stage} 开始执行流水线...")

    for stage in range(start_stage, 5):
        output_file = STAGE_OUTPUT_FILES[stage]
        if output_file.exists():
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
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.start_stage))


if __name__ == "__main__":
    main()
