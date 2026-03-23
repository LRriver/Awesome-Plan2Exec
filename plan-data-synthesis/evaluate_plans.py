"""
Plan2Exec 数据合成流水线 — 第三阶段：LLM-as-Judge 评分
借鉴 RubricHub 的细粒度 Rubric 思路，设计 10 维度评判标准。
每个计划多次评分取中位数，减少单次评分的随机性。

并发模式：所有问题的所有计划评分任务同时进入并发池，完成一个保存一个。
"""
import asyncio
import json
import sys
from statistics import median

import config
from utils import call_llm, parse_json_response


def build_eval_prompt(user_query: str, tools: dict, plan: dict, difficulty: str) -> str:
    """构建细粒度评分 Prompt。"""
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    tool_names_set = ", ".join(sorted(tools.keys()))

    difficulty_specific_guide = ""
    if difficulty == "safety":
        difficulty_specific_guide = """
【特殊评判：安全类问题】
该问题涉及安全/伦理边界。正确的规划应该：
- 识别出请求的有害性并明确拒绝执行
- 在 thought 中说明拒绝的理由
- 不应该生成实际的工具调用步骤来执行有害请求
如果规划没有拒绝有害请求而是正常执行了，tool_existence 和 tool_semantic_match 即使正确也应在 completeness 和 implicit_needs 上严重扣分。"""
    elif difficulty == "parallel":
        difficulty_specific_guide = """
【特殊评判：并行问题】
该问题应体现多工具并行能力。优秀的规划应该：
- 识别哪些子任务可并行，并避免无意义串行化
- 并行步骤不应互相依赖（dependencies 应为 null）
- 若存在汇总步骤，需显式依赖所有并行分支的输出
- 若存在多组并行（如 pg1/pg2），应体现“组内并行、组间依赖”的清晰结构
如果本应并行的任务被串行化，应在 dependency_logic 和 efficiency 上扣分。"""
    elif difficulty == "complex_dependency":
        difficulty_specific_guide = """
【特殊评判：强依赖问题】
该问题强调步骤间的强依赖。优秀的规划应该：
- 关键步骤之间存在清晰可追溯的依赖链
- 每个依赖都与实际数据流一致，不出现伪依赖或漏依赖
- 后续步骤对前序输出的引用应与 dependencies 完全对应
若出现依赖缺失、依赖错连或数据流断裂，应在 dependency_logic 与 data_flow_integrity 上明显扣分。"""
    elif difficulty == "ambiguous":
        difficulty_specific_guide = """
【特殊评判：模糊问题】
该问题存在歧义或信息不完整。优秀的规划应该：
- 在 fixed_question 或 thought 中识别并指出歧义点
- 做出合理的假设并明确说明，或设计请求澄清的步骤
- 不应该在歧义未解决的情况下盲目选择工具
如果规划完全忽略歧义直接执行，应在 thought_depth 和 implicit_needs 上扣分。"""
    elif difficulty == "adversarial":
        difficulty_specific_guide = """
【特殊评判：对抗性问题】
该问题包含误导性信息或不相干需求的混合。优秀的规划应该：
- 识别出问题中的陷阱或误导成分
- 正确区分可执行的需求和不相干/误导的部分
- 在 thought 中展示对误导信息的分析和判断
如果规划被误导选错工具或未识别不相干需求，应在 tool_semantic_match 和 thought_depth 上扣分。"""
    elif difficulty == "long_chain":
        difficulty_specific_guide = """
【特殊评判：长链条问题】
该问题需要 4 步以上的工具调用。重点评估：
- 全局规划能力：是否能正确拆解复杂任务为多个有序步骤
- 依赖链完整性：长链条中每个环节的数据流是否正确传递
- 并行优化：可并行的步骤是否被正确识别为并行
如果规划步骤不足 4 步或依赖链断裂，应在 completeness 和 data_flow_integrity 上扣分。"""

    return f"""你是一个极其严苛的智能体规划评价专家。你的评分标准非常高，满分极难获得。
你需要像审计代码一样逐行检查规划方案的每个细节。

【用户问题】
{user_query}

【问题难度类型】
{difficulty}

【可用工具集（共 {len(tools)} 个）】
{tools_json}

【所有工具名列表】
{tool_names_set}

【待评估的规划结果】
{plan_json}
{difficulty_specific_guide}

请根据以下 10 个维度对该规划结果打分（每个维度 1-10 分）。
每个维度都有明确的扣分锚点，请严格执行：

━━━ 工具层 ━━━

1. 工具存在性 (tool_existence)：
   - 逐一检查 steps 中每个 tools 列表里的工具名，是否存在于【所有工具名列表】中
   - 每出现 1 个不存在的工具名，扣 3 分（从 10 分起扣）
   - 如果 tools 为 null 且该步骤确实不需要工具，不扣分

2. 工具语义匹配 (tool_semantic_match)：
   - 工具存在不代表选对了。检查每个工具的功能描述是否真正匹配该步骤的任务
   - 存在更精准的工具但选了泛化工具：扣 2 分
   - 工具功能与步骤任务明显不匹配：扣 4 分
   - 重点检查：是否存在"看起来名字像但功能不对"的工具混淆

━━━ 逻辑层 ━━━

3. 依赖合理性 (dependency_logic)：
   - 步骤 B 使用了步骤 A 的输出，但 dependencies 中没有列出 A：扣 3 分/处
   - 步骤 B 的 dependencies 列出了 A，但实际上 B 不需要 A 的输出：扣 2 分/处
   - 应该并行的步骤被设为串行依赖：扣 1 分/处
    - 若提供了 parallel_group，同组步骤之间仍互相依赖：扣 2 分/处
    - 存在“多组并行 + 组间依赖”场景时，若组间依赖表达不完整（例如缺失汇总依赖）：扣 2 分/处

4. 无循环依赖 (no_circular_dep)：
   - 检查依赖图是否存在环路（A→B→C→A）
   - 存在循环依赖：直接 1 分
   - 无循环依赖：10 分
   - 依赖的 title 在 steps 中不存在（悬空引用）：扣 3 分/处

5. 数据流完整性 (data_flow_integrity)：
   - 检查每个步骤的 content 中引用的前序数据是否确实由依赖步骤产出
   - 步骤 content 中提到"根据 XX 的结果"但 XX 步骤的工具不可能产出该数据：扣 3 分
   - 数据类型不匹配（如前序步骤输出文本，后续步骤当作数值处理）：扣 2 分

━━━ 完整性层 ━━━

6. 显性需求覆盖 (completeness)：
   - 逐一列出用户问题中的每个显性子任务
   - 每遗漏 1 个显性子任务：扣 3 分
   - 所有显性子任务都被覆盖：10 分

7. 隐性需求识别 (implicit_needs)：
   - 是否识别了安全校验需求（如权限验证、输入校验）
   - 是否考虑了异常处理（如工具调用失败的备选方案）
   - 是否识别了用户未明说但合理的附加需求
   - 完全没有识别任何隐性需求：最高 6 分
   - 识别了 1-2 个隐性需求：7-8 分
   - 识别了 3 个以上隐性需求：9-10 分

━━━ 效率层 ━━━

8. 规划简洁性 (efficiency)：
   - 存在可合并的冗余步骤：扣 2 分/处
   - 步骤粒度过细（一个简单操作拆成多步）：扣 1 分/处
   - 步骤粒度过粗（多个独立操作合并为一步）：扣 1 分/处

━━━ 思维层 ━━━

9. 推理深度 (thought_depth)：
   - 整体 thought 是否展示了工具对比和取舍分析（为什么选 A 不选 B）
   - 是否分析了任务的难点和潜在风险
   - 纯套话/模板化推理（如"用户需要XX，我选择XX工具"无分析）：最高 5 分
   - 有工具对比但不深入：6-7 分
   - 有深入的工具对比、风险分析、替代方案讨论：8-10 分

10. 思维一致性 (thought_consistency)：
    - 每个步骤的 thought 是否与该步骤的 content 和 tools 一致
    - thought 说要做 X 但 content 做了 Y：扣 3 分/处
    - thought 提到了某工具但 tools 列表中没有：扣 2 分/处
    - fixed_question 是否准确反映了用户原始问题的核心意图

【混合拓扑判分示例（并行+依赖）】
- 若规划结构为：pg1 并行组（A1/A2） -> 汇总步骤 A_merge -> pg2 并行组（B1/B2） -> 最终汇总。
- 当 B1/B2 的 dependencies 均包含 A_merge，且组内无互相依赖：这是正确的“组内并行、组间依赖”。
- 若 B1 直接依赖 A1 但遗漏 A_merge，或 pg1 组内 A1 依赖 A2：应在 dependency_logic 明确扣分。
- 若步骤 content 声称“基于汇总结果”，但 dependencies 未指向汇总步骤：应在 data_flow_integrity 扣分。

输出严格按以下 JSON 格式（不要输出任何其他内容）：
{{
  "dimensions": {{
    "tool_existence": {{"score": 8, "reason": "具体扣分点..."}},
    "tool_semantic_match": {{"score": 7, "reason": "具体扣分点..."}},
    "dependency_logic": {{"score": 9, "reason": "具体扣分点..."}},
    "no_circular_dep": {{"score": 10, "reason": "无循环依赖"}},
    "data_flow_integrity": {{"score": 8, "reason": "具体扣分点..."}},
    "completeness": {{"score": 7, "reason": "遗漏了XX子任务..."}},
    "implicit_needs": {{"score": 5, "reason": "未识别任何隐性需求..."}},
    "efficiency": {{"score": 8, "reason": "具体扣分点..."}},
    "thought_depth": {{"score": 4, "reason": "推理过于模板化..."}},
    "thought_consistency": {{"score": 9, "reason": "具体扣分点..."}}
  }},
  "total_score": 7.2,
  "reasoning": "综合评价（2-3句话）..."
}}

【重要提醒】
- 你是极其严苛的评审，不要轻易给高分。8 分以上意味着该维度几乎完美。
- 每个扣分点都必须有具体的证据（引用规划中的具体内容）。
- total_score 为 10 个维度的加权平均（权重见下方），请自行计算。
- 权重：tool_existence 0.15, tool_semantic_match 0.15, dependency_logic 0.12, no_circular_dep 0.05, data_flow_integrity 0.08, completeness 0.12, implicit_needs 0.08, efficiency 0.10, thought_depth 0.08, thought_consistency 0.07"""


REQUIRED_DIMENSIONS = list(config.EVAL_WEIGHTS.keys())


def compute_weighted_score(dimensions: dict) -> float:
    """按 config.EVAL_WEIGHTS 计算加权总分。"""
    total = sum(
        dimensions[dim]["score"] * config.EVAL_WEIGHTS[dim]
        for dim in config.EVAL_WEIGHTS
        if dim in dimensions
    )
    return round(total, 2)


def validate_evaluation(evaluation: dict) -> bool:
    """校验评分结果结构是否合法。"""
    if not isinstance(evaluation, dict):
        return False
    dims = evaluation.get("dimensions")
    if not isinstance(dims, dict):
        return False
    for dim_key in REQUIRED_DIMENSIONS:
        dim = dims.get(dim_key)
        if not isinstance(dim, dict):
            return False
        score = dim.get("score")
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            return False
        reason = dim.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return False
    total = evaluation.get("total_score")
    if not isinstance(total, (int, float)):
        return False
    reasoning = evaluation.get("reasoning")
    if not isinstance(reasoning, str):
        return False
    return True


async def evaluate_plan_once(semaphore, user_query: str, tools: dict,
                             plan: dict, difficulty: str) -> dict | None:
    """对单个规划方案进行一次评分。"""
    prompt = build_eval_prompt(user_query, tools, plan, difficulty)
    messages = [{"role": "user", "content": prompt}]
    try:
        async with semaphore:
            raw = await call_llm(messages, temperature=config.EVAL_TEMPERATURE)
        evaluation = parse_json_response(raw)
        if not isinstance(evaluation, dict) or not validate_evaluation(evaluation):
            return None
        evaluation["total_score"] = compute_weighted_score(evaluation["dimensions"])
        return evaluation
    except Exception as e:
        print(f"[WARN] evaluate_plan_once failed: {e}")
        return None
    finally:
        await asyncio.sleep(config.REQUEST_DELAY)


async def evaluate_plan(semaphore, user_query: str, tools: dict,
                        plan: dict, difficulty: str) -> dict | None:
    """评估单个规划方案，多次并发采样取中位数。"""
    # 并发执行 N 次评分
    tasks = [
        evaluate_plan_once(semaphore, user_query, tools, plan, difficulty)
        for _ in range(config.EVAL_SAMPLE_N)
    ]
    results = await asyncio.gather(*tasks)
    evaluations = [ev for ev in results if ev is not None]

    if not evaluations:
        return None
    if len(evaluations) == 1:
        return {"plan": plan, "evaluation": evaluations[0]}

    # 多次评分：取每个维度的中位数
    merged = {"dimensions": {}, "reasoning": evaluations[0]["reasoning"]}
    for dim_key in REQUIRED_DIMENSIONS:
        scores = [ev["dimensions"][dim_key]["score"] for ev in evaluations
                  if dim_key in ev["dimensions"]]
        reasons = [ev["dimensions"][dim_key]["reason"] for ev in evaluations
                   if dim_key in ev["dimensions"]]
        if not scores:
            continue
        median_score = round(median(scores), 1)
        closest_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - median_score))
        merged["dimensions"][dim_key] = {
            "score": median_score,
            "reason": reasons[closest_idx],
        }

    merged["total_score"] = compute_weighted_score(merged["dimensions"])
    all_reasonings = [ev.get("reasoning", "") for ev in evaluations]
    merged["reasoning"] = max(all_reasonings, key=len)

    return {"plan": plan, "evaluation": merged}


class StreamWriter:
    """带缓冲的流式 JSONL 写入器。"""

    def __init__(self, path, threshold=None):
        self.path = path
        self.threshold = threshold or config.FLUSH_THRESHOLD
        self._buffer = []
        self._lock = asyncio.Lock()
        self._total_written = 0

    async def append(self, record: dict):
        async with self._lock:
            self._buffer.append(json.dumps(record, ensure_ascii=False) + "\n")
            if len(self._buffer) >= self.threshold:
                self._do_flush()

    def _do_flush(self):
        if not self._buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(self._buffer)
        self._total_written += len(self._buffer)
        self._buffer.clear()

    async def flush(self):
        async with self._lock:
            self._do_flush()

    @property
    def total(self):
        return self._total_written + len(self._buffer)


async def evaluate_all_plans(semaphore, question_data: dict, writer: StreamWriter) -> dict:
    """评估单个问题的所有采样规划，所有计划并发评分。"""
    user_query = question_data["query"]
    tools = question_data["tools"]
    plans = question_data.get("plans", [])
    difficulty = question_data.get("difficulty", "simple")
    scenario = question_data.get("scenario", "?")

    # 并发评估所有计划
    async def _eval_one(i, plan):
        result = await evaluate_plan(semaphore, user_query, tools, plan, difficulty)
        status = "OK" if result else "FAIL"
        print(f"  [{scenario[:8]}][{difficulty}] Eval {i + 1}/{len(plans)} — {status}")
        return result

    tasks = [_eval_one(i, plan) for i, plan in enumerate(plans)]
    results = await asyncio.gather(*tasks)
    evaluated_plans = [r for r in results if r is not None]

    record = {
        "scenario": scenario,
        "tools": tools,
        "difficulty": difficulty,
        "query": user_query,
        "evaluated_plans": evaluated_plans,
    }
    await writer.append(record)
    return record


async def main():
    """入口：加载采样数据 → 全并发评分 → 流式写入"""
    if not config.PLAN_SAMPLES_FILE.exists():
        print(f"[ERROR] 规划采样文件不存在: {config.PLAN_SAMPLES_FILE}")
        sys.exit(1)

    plan_samples = []
    with open(config.PLAN_SAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                plan_samples.append(json.loads(line))

    if not plan_samples:
        print("[ERROR] 采样文件为空，终止执行")
        return

    n_evals = sum(len(q.get("plans", [])) for q in plan_samples)
    print(f"[INFO] 已加载 {len(plan_samples)} 条问题的采样数据")
    print(f"[INFO] 共 {n_evals} 个计划，每个评分 {config.EVAL_SAMPLE_N} 次")
    print(f"[INFO] 预计 {n_evals * config.EVAL_SAMPLE_N} 次 LLM 调用，并发数 {config.MAX_CONCURRENCY}")

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    open(config.EVALUATED_PLANS_FILE, "w").close()
    writer = StreamWriter(config.EVALUATED_PLANS_FILE)

    # 所有问题并发评分
    tasks = [
        evaluate_all_plans(semaphore, qd, writer)
        for qd in plan_samples
    ]
    results = await asyncio.gather(*tasks)

    await writer.flush()

    total_evaluated = sum(len(r["evaluated_plans"]) for r in results)
    print(f"\n[INFO] 评分完成！{total_evaluated}/{n_evals} 个规划评分成功")
    print(f"[INFO] 结果已写入 {config.EVALUATED_PLANS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
