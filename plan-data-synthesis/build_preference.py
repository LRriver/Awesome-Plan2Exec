"""
Plan2Exec 数据合成流水线 — 第四阶段：偏好数据构建器
根据评分结果筛选并组装 DPO 偏好数据对。
"""
import json
import sys

import config
from plan_sampling import validate_plan_output


def extract_tool_sequence(plan: dict) -> tuple:
    """提取计划的工具调用序列，用于去重比较。"""
    steps = plan.get("steps", [])
    tool_seq = []
    for step in steps:
        tools = step.get("tools") or []
        tool_seq.append(tuple(sorted(tools)))
    return tuple(tool_seq)


def plans_are_similar(plan1: dict, plan2: dict) -> bool:
    """判断两个计划是否实质相似（工具调用序列相同）。"""
    return extract_tool_sequence(plan1) == extract_tool_sequence(plan2)


def build_preference_pair(question_data: dict) -> dict | None:
    """从单个问题的评分结果中提取 DPO 偏好数据对。"""
    evaluated_plans = question_data.get("evaluated_plans", [])
    if len(evaluated_plans) < 2:
        print(f"[WARN] 评分结果不足 2 个，跳过: {question_data.get('query', '')[:50]}...")
        return None

    sorted_plans = sorted(
        evaluated_plans,
        key=lambda x: x["evaluation"]["total_score"],
        reverse=True,
    )

    chosen = sorted_plans[0]
    chosen_score = chosen["evaluation"]["total_score"]

    rejected = None
    for candidate in reversed(sorted_plans[1:]):
        candidate_score = candidate["evaluation"]["total_score"]
        if candidate_score >= chosen_score:
            continue
        if chosen_score - candidate_score < config.MIN_SCORE_GAP:
            continue
        if not validate_plan_output(candidate["plan"]):
            continue
        if plans_are_similar(chosen["plan"], candidate["plan"]):
            continue
        rejected = candidate
        break

    if rejected is None:
        print(f"[WARN] 无法找到合适的 rejected_plan，丢弃: {question_data.get('query', '')[:50]}...")
        return None

    return {
        "scenario": question_data["scenario"],
        "tools": question_data["tools"],
        "difficulty": question_data["difficulty"],
        "user_query": question_data["query"],
        "chosen_plan": chosen["plan"],
        "rejected_plan": rejected["plan"],
        "chosen_score": chosen_score,
        "rejected_score": rejected["evaluation"]["total_score"],
        "chosen_evaluation": chosen["evaluation"],
        "rejected_evaluation": rejected["evaluation"],
    }


def main():
    """入口：加载评分数据 → 筛选偏好对 → 流式写入"""
    if not config.EVALUATED_PLANS_FILE.exists():
        print(f"[ERROR] 评分结果文件不存在: {config.EVALUATED_PLANS_FILE}")
        sys.exit(1)

    evaluated_data = []
    with open(config.EVALUATED_PLANS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                evaluated_data.append(json.loads(line))

    if not evaluated_data:
        print("[ERROR] 评分文件为空，终止执行")
        return

    print(f"[INFO] 已加载 {len(evaluated_data)} 条问题的评分数据，开始构建偏好对...")

    # 流式写入，带缓冲
    buffer = []
    valid = 0
    discarded = 0

    with open(config.PREFERENCE_DATA_FILE, "w", encoding="utf-8") as f:
        for question_data in evaluated_data:
            pair = build_preference_pair(question_data)
            if pair is not None:
                buffer.append(json.dumps(pair, ensure_ascii=False) + "\n")
                valid += 1
                if len(buffer) >= config.FLUSH_THRESHOLD:
                    f.writelines(buffer)
                    buffer.clear()
            else:
                discarded += 1
        # 刷盘剩余
        if buffer:
            f.writelines(buffer)

    print(f"\n[INFO] 偏好数据构建完成！")
    print(f"  总问题数: {len(evaluated_data)}")
    print(f"  有效偏好对: {valid}")
    print(f"  丢弃: {discarded}")
    print(f"[INFO] 结果已写入 {config.PREFERENCE_DATA_FILE}")


if __name__ == "__main__":
    main()
