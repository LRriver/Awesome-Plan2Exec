"""
Plan2Exec 数据合成流水线 — 第四阶段：偏好数据构建器
根据评分结果筛选并组装 DPO 偏好数据对。
"""
import json
import sys

import config
from plan_sampling import validate_plan_output


def build_preference_pair(question_data: dict) -> dict | None:
    """从单个问题的评分结果中提取 DPO 偏好数据对。

    筛选逻辑：
    1. 按 total_score 降序排列所有 evaluated_plans
    2. 最高分作为 chosen_plan
    3. 从最低分向上遍历，找第一个满足以下条件的作为 rejected_plan：
       - total_score < chosen_score
       - 分差 >= config.MIN_SCORE_GAP
       - 通过 validate_plan_output 结构校验
    4. 找不到合适的 rejected_plan 则丢弃该条数据

    Args:
        question_data: 包含 scenario、tools、difficulty、query、evaluated_plans 的字典。
            evaluated_plans 是 [{"plan": {...}, "evaluation": {...}}, ...] 列表。

    Returns:
        包含 10 个字段的偏好数据字典，或 None（无法构建有效偏好对时）。
    """
    evaluated_plans = question_data.get("evaluated_plans", [])
    if len(evaluated_plans) < 2:
        print(f"[WARN] 评分结果不足 2 个，跳过: {question_data.get('query', '')[:50]}...")
        return None

    # 1. 按 total_score 降序排序
    sorted_plans = sorted(
        evaluated_plans,
        key=lambda x: x["evaluation"]["total_score"],
        reverse=True,
    )

    # 2. 最高分作为 chosen
    chosen = sorted_plans[0]
    chosen_score = chosen["evaluation"]["total_score"]

    # 3. 从最低分向上遍历，找合适的 rejected
    rejected = None
    for candidate in reversed(sorted_plans[1:]):
        candidate_score = candidate["evaluation"]["total_score"]
        # 必须低于 chosen_score
        if candidate_score >= chosen_score:
            continue
        # 分差必须 >= MIN_SCORE_GAP
        if chosen_score - candidate_score < config.MIN_SCORE_GAP:
            continue
        # 结构必须完整
        if not validate_plan_output(candidate["plan"]):
            continue
        rejected = candidate
        break

    if rejected is None:
        print(f"[WARN] 无法找到合适的 rejected_plan（分差不足或结构不完整），丢弃: {question_data.get('query', '')[:50]}...")
        return None

    # 4. 组装偏好数据对
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
    """入口：从 evaluated_plans.jsonl 加载评分数据 → 筛选偏好对 → 写入 preference_data.jsonl"""
    # 1. 检查输入文件是否存在
    if not config.EVALUATED_PLANS_FILE.exists():
        print(f"[ERROR] 评分结果文件不存在: {config.EVALUATED_PLANS_FILE}")
        print("[ERROR] 请先运行第三阶段 evaluate_plans.py 生成评分数据")
        sys.exit(1)

    # 2. 加载评分数据
    evaluated_data = []
    with open(config.EVALUATED_PLANS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evaluated_data.append(json.loads(line))

    if not evaluated_data:
        print("[ERROR] 评分文件为空，终止执行")
        return

    print(f"[INFO] 已加载 {len(evaluated_data)} 条问题的评分数据，开始构建偏好对...")

    # 3. 逐条构建偏好数据对
    preference_pairs = []
    discarded = 0
    for question_data in evaluated_data:
        pair = build_preference_pair(question_data)
        if pair is not None:
            preference_pairs.append(pair)
        else:
            discarded += 1

    # 4. 写入输出文件
    with open(config.PREFERENCE_DATA_FILE, "w", encoding="utf-8") as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # 5. 打印摘要
    total = len(evaluated_data)
    valid = len(preference_pairs)
    print(f"\n[INFO] 偏好数据构建完成！")
    print(f"  总问题数: {total}")
    print(f"  有效偏好对: {valid}")
    print(f"  丢弃: {discarded}")
    print(f"[INFO] 结果已写入 {config.PREFERENCE_DATA_FILE}")


if __name__ == "__main__":
    main()
