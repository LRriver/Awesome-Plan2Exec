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


def strip_plan_metadata(plan: dict) -> dict:
    """Remove training-only metadata from plan bodies used as answers."""
    if not isinstance(plan, dict):
        return plan
    blocked = {"negative_type", "quality_bucket"}
    return {key: value for key, value in plan.items() if key not in blocked}


def assign_quality_bucket(score: float) -> str:
    """按分数分桶，便于后续抽取 strong/medium/weak 训练视图。"""
    buckets = getattr(
        config,
        "QUALITY_BUCKETS",
        {"strong": 8.5, "medium": 6.0, "weak": 3.0},
    )
    if score >= buckets.get("strong", 8.5):
        return "strong"
    if score >= buckets.get("medium", 6.0):
        return "medium"
    if score >= buckets.get("weak", 3.0):
        return "weak"
    return "invalid"


def build_ranked_candidates(question_data: dict) -> dict:
    """构建同题多候选 ranked list，保留分数、分桶和负样本类型。"""
    evaluated_plans = sorted(
        question_data.get("evaluated_plans", []),
        key=lambda x: x["evaluation"]["total_score"],
        reverse=True,
    )
    candidates = []
    for item in evaluated_plans:
        plan = item["plan"]
        score = item["evaluation"]["total_score"]
        candidates.append({
            "plan": strip_plan_metadata(plan),
            "score": score,
            "quality_bucket": assign_quality_bucket(score),
            "negative_type": plan.get("negative_type"),
            "evaluation": item["evaluation"],
        })

    return {
        "scenario": question_data["scenario"],
        "tools": question_data["tools"],
        "difficulty": question_data["difficulty"],
        "query_style": question_data.get("query_style"),
        "difficulty_subtype": question_data.get("difficulty_subtype"),
        "user_query": question_data["query"],
        "candidates": candidates,
    }


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
        "chosen_plan": strip_plan_metadata(chosen["plan"]),
        "rejected_plan": strip_plan_metadata(rejected["plan"]),
        "chosen_score": chosen_score,
        "rejected_score": rejected["evaluation"]["total_score"],
        "chosen_quality_bucket": assign_quality_bucket(chosen_score),
        "rejected_quality_bucket": assign_quality_bucket(rejected["evaluation"]["total_score"]),
        "rejected_negative_type": rejected["plan"].get("negative_type"),
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
    ranked_buffer = []
    valid = 0
    discarded = 0

    with open(config.PREFERENCE_DATA_FILE, "w", encoding="utf-8") as f:
        ranked_path = getattr(config, "RANKED_CANDIDATES_FILE", None)
        ranked_file = open(ranked_path, "w", encoding="utf-8") if ranked_path else None
        try:
            for question_data in evaluated_data:
                ranked = build_ranked_candidates(question_data)
                if ranked_file:
                    ranked_buffer.append(json.dumps(ranked, ensure_ascii=False) + "\n")
                    if len(ranked_buffer) >= config.FLUSH_THRESHOLD:
                        ranked_file.writelines(ranked_buffer)
                        ranked_buffer.clear()

                pair = build_preference_pair(question_data)
                if pair is not None:
                    buffer.append(json.dumps(pair, ensure_ascii=False) + "\n")
                    valid += 1
                    if len(buffer) >= config.FLUSH_THRESHOLD:
                        f.writelines(buffer)
                        buffer.clear()
                else:
                    discarded += 1
        finally:
            if ranked_file:
                if ranked_buffer:
                    ranked_file.writelines(ranked_buffer)
                ranked_file.close()
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
