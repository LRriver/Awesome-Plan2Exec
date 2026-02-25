"""
Plan2Exec 数据合成流水线 — 第三阶段：LLM-as-Judge 评分
使用裁判大模型对多路采样的规划结果进行多维度打分。
"""
import asyncio
import json
import sys

import aiohttp

import config
from utils import call_llm, parse_json_response


def build_eval_prompt(user_query: str, tools: dict, plan: dict) -> str:
    """构建评分 Prompt，注入用户问题、工具集和待评估规划。

    Args:
        user_query: 用户原始问题文本
        tools: 可用工具集字典 {tool_name: tool_description}
        plan: 待评估的规划结果字典（含 fixed_question, thought, steps）

    Returns:
        完整的评分 Prompt 字符串
    """
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    generated_plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    return f"""你是一个严苛的智能体规划评价专家。请评估以下规划方案针对用户问题的解决能力。

【用户问题】
{user_query}

【可用工具集】
{tools_json}

【待评估的规划结果】
{generated_plan_json}

请根据以下 5 个维度对该规划结果打分（每个维度 1-10 分），并给出详细的扣分/加分理由：

1. 工具准确性 (Tool Accuracy)：
   - 使用的工具是否全部存在于可用工具集中？（使用不存在的工具直接扣 5 分）
   - 工具选择是否与步骤任务匹配？

2. 依赖合理性 (Dependency Logic)：
   - 步骤间的依赖关系是否真实存在？
   - 是否存在"未获取数据就进行分析"等逻辑错误？
   - 应该有依赖的步骤是否正确设置了 dependencies？

3. 任务完整性 (Completeness)：
   - 所有步骤组合起来能否完全满足用户问题的所有需求？
   - 是否遗漏了用户明确提出的子任务？

4. 规划简洁性 (Efficiency)：
   - 是否存在冗余、重复或可合并的步骤？
   - 步骤拆分粒度是否合理（不过粗也不过细）？

5. 思维链质量 (Thought Quality)：
   - fixed_question 是否准确补全了用户问题？
   - 整体 thought 是否体现了清晰的需求分解和工具匹配推理？
   - 每个步骤的 thought 是否与该步骤的实际内容一致？
   - 推理过程是否有逻辑性，而非空洞的套话？

输出严格按以下 JSON 格式：
{{
  "dimensions": {{
    "tool_accuracy": {{"score": 8, "reason": "..."}},
    "dependency_logic": {{"score": 9, "reason": "..."}},
    "completeness": {{"score": 7, "reason": "..."}},
    "efficiency": {{"score": 8, "reason": "..."}},
    "thought_quality": {{"score": 7, "reason": "..."}}
  }},
  "total_score": 7.8,
  "reasoning": "综合评价..."
}}

注意：total_score 为 5 个维度的加权平均，权重为：
工具准确性 0.3, 依赖合理性 0.2, 任务完整性 0.2, 规划简洁性 0.15, 思维链质量 0.15"""

REQUIRED_DIMENSIONS = list(config.EVAL_WEIGHTS.keys())


def compute_weighted_score(dimensions: dict) -> float:
    """按 config.EVAL_WEIGHTS 计算加权总分。

    Args:
        dimensions: 维度字典，每个键对应 {"score": int, "reason": str}

    Returns:
        加权总分，保留 2 位小数。
    """
    total = sum(
        dimensions[dim]["score"] * config.EVAL_WEIGHTS[dim]
        for dim in config.EVAL_WEIGHTS
    )
    return round(total, 2)


def validate_evaluation(evaluation: dict) -> bool:
    """校验评分结果结构是否合法。

    验证规则：
    - 包含 "dimensions" 字典，含全部 5 个维度键
    - 每个维度有 "score"（int, 1-10）和 "reason"（非空 str）
    - 包含 "total_score"（float/int）和 "reasoning"（str）

    Args:
        evaluation: 待校验的评分字典。

    Returns:
        True 表示结构合法，False 表示校验失败。
    """
    if not isinstance(evaluation, dict):
        return False

    # 校验 dimensions
    dims = evaluation.get("dimensions")
    if not isinstance(dims, dict):
        return False
    for dim_key in REQUIRED_DIMENSIONS:
        dim = dims.get(dim_key)
        if not isinstance(dim, dict):
            return False
        score = dim.get("score")
        if not isinstance(score, int) or score < 1 or score > 10:
            return False
        reason = dim.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return False

    # 校验 total_score 和 reasoning
    total = evaluation.get("total_score")
    if not isinstance(total, (int, float)):
        return False
    reasoning = evaluation.get("reasoning")
    if not isinstance(reasoning, str):
        return False

    return True


async def evaluate_plan(session, semaphore, user_query: str, tools: dict, plan: dict) -> dict | None:
    """评估单个规划方案。

    构建评分 Prompt，调用裁判模型打分，解析并校验评分结果，
    使用 compute_weighted_score 重新计算 total_score（不信任 LLM 的计算）。

    Args:
        session: aiohttp 客户端会话。
        semaphore: 异步信号量，控制并发。
        user_query: 用户问题文本。
        tools: 可用工具集字典。
        plan: 待评估的规划字典。

    Returns:
        {"plan": plan, "evaluation": evaluation} 成功时，失败返回 None。
    """
    prompt = build_eval_prompt(user_query, tools, plan)
    messages = [{"role": "user", "content": prompt}]
    try:
        async with semaphore:
            raw = await call_llm(session, messages, temperature=0.3)
        evaluation = parse_json_response(raw)
        if not isinstance(evaluation, dict) or not validate_evaluation(evaluation):
            print(f"[WARN] Evaluation validation failed for query: {user_query[:50]}...")
            return None
        # 用自己的权重重新计算 total_score，不信任 LLM 的计算
        evaluation["total_score"] = compute_weighted_score(evaluation["dimensions"])
        return {"plan": plan, "evaluation": evaluation}
    except Exception as e:
        print(f"[WARN] evaluate_plan failed: {e}")
        return None
    finally:
        await asyncio.sleep(config.REQUEST_DELAY)


async def evaluate_all_plans(session, semaphore, question_data: dict) -> dict:
    """评估单个问题的所有采样规划。

    遍历 question_data 中的 plans 列表，逐个调用 evaluate_plan，
    收集非 None 的结果。

    Args:
        session: aiohttp 客户端会话。
        semaphore: 异步信号量，控制并发。
        question_data: 包含 scenario、tools、difficulty、query、plans 的字典。

    Returns:
        包含 scenario、tools、difficulty、query、evaluated_plans 的结果字典。
    """
    user_query = question_data["query"]
    tools = question_data["tools"]
    plans = question_data.get("plans", [])

    evaluated_plans = []
    for i, plan in enumerate(plans):
        result = await evaluate_plan(session, semaphore, user_query, tools, plan)
        if result is not None:
            evaluated_plans.append(result)
        print(f"  Eval {i + 1}/{len(plans)} for [{question_data['difficulty']}] — {'OK' if result else 'FAIL'}")

    return {
        "scenario": question_data["scenario"],
        "tools": tools,
        "difficulty": question_data["difficulty"],
        "query": user_query,
        "evaluated_plans": evaluated_plans,
    }



async def main():
    """入口：从 plan_samples.jsonl 加载采样数据 → 逐条评分 → 写入 evaluated_plans.jsonl"""
    # 1. 检查输入文件是否存在
    if not config.PLAN_SAMPLES_FILE.exists():
        print(f"[ERROR] 规划采样文件不存在: {config.PLAN_SAMPLES_FILE}")
        print("[ERROR] 请先运行第二阶段 plan_sampling.py 生成采样数据")
        sys.exit(1)

    # 2. 加载采样数据
    plan_samples = []
    with open(config.PLAN_SAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            plan_samples.append(json.loads(line))

    if not plan_samples:
        print("[ERROR] 采样文件为空，终止执行")
        return

    print(f"[INFO] 已加载 {len(plan_samples)} 条问题的采样数据，开始评分...")

    # 3. 创建会话和信号量，逐条评分
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    results = []

    async with aiohttp.ClientSession() as session:
        for i, question_data in enumerate(plan_samples):
            plans_count = len(question_data.get("plans", []))
            print(f"\n[{i + 1}/{len(plan_samples)}] 场景: {question_data['scenario']} | 难度: {question_data['difficulty']} | 规划数: {plans_count}")
            result = await evaluate_all_plans(session, semaphore, question_data)
            results.append(result)
            evaluated_count = len(result["evaluated_plans"])
            print(f"  评分成功: {evaluated_count}/{plans_count}")

    # 4. 写入输出文件
    with open(config.EVALUATED_PLANS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. 打印摘要
    total_evaluated = sum(len(r["evaluated_plans"]) for r in results)
    total_plans = sum(len(ps.get("plans", [])) for ps in plan_samples)
    print(f"\n[INFO] 评分完成！共 {len(results)} 条问题，{total_evaluated}/{total_plans} 个规划评分成功")
    print(f"[INFO] 结果已写入 {config.EVALUATED_PLANS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
