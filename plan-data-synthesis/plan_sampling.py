"""
Plan2Exec 数据合成流水线 — 第二阶段：规划采样器
将问题和工具集动态组装为 System Prompt，对每个问题多路采样生成结构化规划。
"""
import asyncio
import json
import sys

import aiohttp

import config
from utils import call_llm, parse_json_response


def build_system_prompt(tools: dict) -> str:
    """构建动态 System Prompt，将工具集注入模板。

    Args:
        tools: 工具名到描述的映射字典，如 {"get_weather": "获取天气信息", ...}

    Returns:
        包含工具集约束说明和三个参考示例的完整 System Prompt 字符串。
    """
    dynamic_loaded_tools_json = json.dumps(tools, ensure_ascii=False, indent=2)

    return f"""你是一个高级规划智能体（Planning Agent）。你的目标是根据用户的提问和提供的工具集，拆解并生成结构化的任务执行计划。

【当前可用工具集】
{dynamic_loaded_tools_json}

【规划规则】
1. 先输出 fixed_question：对用户原始问题进行补全和明确化，确保问题描述完整无歧义。
2. 思考过程 (thought)：先逐字分析用户需求，识别所有子任务；再逐一评估可用工具与子任务的匹配关系；最后规划步骤顺序和依赖关系。
3. 步骤拆解 (steps)：每个步骤包含 thought、title、content、tools、dependencies 五个字段。
4. 工具选择：tools 只能从【当前可用工具集】中选取，严禁使用不在列表中的工具名。如果某步骤不需要工具，设为 null。
5. 依赖关系 (dependencies)：如果步骤 B 需要步骤 A 的输出结果才能执行，则 B 的 dependencies 设为 ["步骤A的title"]。如果无依赖，设为 null。
6. 无匹配工具：如果问题可以直接回答，无需伪造工具调用，tools 设为 null，给出一个"直接回答"步骤即可。
7. 步骤内容中引用前序步骤输出时，使用【步骤title】格式标注。

【输出格式】
严格按以下 JSON 格式输出，不要输出任何其他内容：
{{
  "fixed_question": "补全/明确后的用户问题",
  "thought": "整体思考过程...",
  "steps": [
    {{
      "thought": "本步骤的思考过程...",
      "title": "步骤标题",
      "content": "步骤详细执行内容",
      "tools": ["tool_name"] 或 null,
      "dependencies": ["依赖的步骤title"] 或 null
    }}
  ]
}}

【参考示例】

示例 A：多步骤，无依赖关系
{{
  "fixed_question": "用户需要查询今天的天气情况，包括温度和天气状况；同时预订今天从天津到北京的车票。",
  "thought": "用户的任务涉及两个任务：查询天气和预订车票。...",
  "steps": [
    {{
      "thought": "...",
      "title": "查询目标城市天气",
      "content": "询问城市，并获取目标城市的天气预报...",
      "tools": ["get_weather"],
      "dependencies": null
    }},
    {{
      "thought": "...",
      "title": "查询高铁票",
      "content": "查询天津到北京的高铁票信息...",
      "tools": ["get_tickets"],
      "dependencies": null
    }}
  ]
}}

示例 B：无需工具，直接回答
{{
  "fixed_question": "用户请求讲一个笑话，这是一个可以直接回答的问题。",
  "thought": "用户的问题简单直接，无需使用任何工具...",
  "steps": [
    {{
      "thought": "...",
      "title": "直接回答",
      "content": "讲一个有趣的笑话来满足用户请求。",
      "tools": null,
      "dependencies": null
    }}
  ]
}}

示例 C：多步骤，存在依赖关系
{{
  "fixed_question": "用户需要获取指定股票代码的实时价格，计算其涨跌幅，并基于此给出投资建议。",
  "thought": "用户的任务是一个多步骤流程，存在明确依赖关系...",
  "steps": [
    {{
      "thought": "...",
      "title": "查询股票实时价格",
      "content": "获取指定股票代码的当前实时价格和相关交易数据。",
      "tools": ["get_search"],
      "dependencies": null
    }},
    {{
      "thought": "...",
      "title": "计算股票涨跌幅",
      "content": "根据【查询股票实时价格】的输出，计算该股票今天的涨跌幅和涨跌额。",
      "tools": ["caculator"],
      "dependencies": ["查询股票实时价格"]
    }},
    {{
      "thought": "...",
      "title": "分析涨跌幅并给出建议",
      "content": "根据【计算股票涨跌幅】的输出，分析股票的涨跌情况，并给出简要的投资建议。",
      "tools": null,
      "dependencies": ["计算股票涨跌幅"]
    }}
  ]
}}"""

def validate_plan_output(plan: dict) -> bool:
    """校验 Plan_Output 结构是否合法。

    验证顶层字段 fixed_question(str)、thought(str)、steps(非空 list)，
    以及每个 step 包含 thought(str)、title(str)、content(str)、
    tools(list|None)、dependencies(list|None)。

    Args:
        plan: 待校验的规划字典。

    Returns:
        True 表示结构合法，False 表示校验失败。
    """
    # 顶层字段校验
    if not isinstance(plan.get("fixed_question"), str):
        return False
    if not isinstance(plan.get("thought"), str):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return False

    # 每个 step 的字段校验
    for step in steps:
        if not isinstance(step, dict):
            return False
        if not isinstance(step.get("thought"), str):
            return False
        if not isinstance(step.get("title"), str):
            return False
        if not isinstance(step.get("content"), str):
            return False
        tools = step.get("tools")
        if tools is not None and not isinstance(tools, list):
            return False
        deps = step.get("dependencies")
        if deps is not None and not isinstance(deps, list):
            return False

    return True


async def sample_plan(session, semaphore, system_prompt: str, user_query: str) -> dict | None:
    """对单个问题进行一次规划采样。

    构建消息列表，调用 LLM 生成规划，解析并校验 Plan_Output 结构。
    采样使用 config 中的 PLAN_TEMPERATURE 和 PLAN_TOP_P 参数。

    Args:
        session: aiohttp 客户端会话。
        semaphore: 异步信号量，控制并发。
        system_prompt: 包含工具集的 System Prompt。
        user_query: 用户问题文本。

    Returns:
        校验通过的规划字典，失败时返回 None。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        async with semaphore:
            raw = await call_llm(
                session,
                messages,
                temperature=config.PLAN_TEMPERATURE,
                top_p=config.PLAN_TOP_P,
            )
        plan = parse_json_response(raw)
        if not isinstance(plan, dict) or not validate_plan_output(plan):
            print(f"[WARN] Plan validation failed for query: {user_query[:50]}...")
            return None
        return plan
    except Exception as e:
        print(f"[WARN] sample_plan failed: {e}")
        return None
    finally:
        await asyncio.sleep(config.REQUEST_DELAY)


async def sample_plans_for_question(session, semaphore, question_data: dict) -> dict:
    """对单个问题采样 K 次，收集有效规划。

    从 question_data 中提取工具集构建 System Prompt，
    循环调用 sample_plan 采样 config.PLAN_SAMPLE_K 次，
    过滤掉失败（None）的结果。

    Args:
        session: aiohttp 客户端会话。
        semaphore: 异步信号量，控制并发。
        question_data: 包含 scenario、tools、difficulty、query 的问题字典。

    Returns:
        包含 scenario、tools、difficulty、query、plans 字段的结果字典。
    """
    system_prompt = build_system_prompt(question_data["tools"])
    user_query = question_data["query"]

    plans = []
    for i in range(config.PLAN_SAMPLE_K):
        result = await sample_plan(session, semaphore, system_prompt, user_query)
        if result is not None:
            plans.append(result)
        print(f"  Sample {i + 1}/{config.PLAN_SAMPLE_K} for [{question_data['difficulty']}] — {'OK' if result else 'FAIL'}")

    return {
        "scenario": question_data["scenario"],
        "tools": question_data["tools"],
        "difficulty": question_data["difficulty"],
        "query": user_query,
        "plans": plans,
    }


async def main():
    """入口：从 questions.jsonl 加载问题 → 并发采样规划 → 写入 plan_samples.jsonl"""
    # 1. 检查输入文件是否存在
    if not config.QUESTIONS_FILE.exists():
        print(f"[ERROR] 问题文件不存在: {config.QUESTIONS_FILE}")
        print("[ERROR] 请先运行第一阶段 generate_questions.py 生成问题数据")
        sys.exit(1)

    # 2. 加载问题数据
    questions = []
    with open(config.QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    if not questions:
        print("[ERROR] 问题文件为空，终止执行")
        return

    print(f"[INFO] 已加载 {len(questions)} 条问题，开始规划采样...")

    # 3. 创建会话和信号量，逐条采样
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    results = []

    async with aiohttp.ClientSession() as session:
        for i, q in enumerate(questions):
            print(f"\n[{i + 1}/{len(questions)}] 场景: {q['scenario']} | 难度: {q['difficulty']}")
            result = await sample_plans_for_question(session, semaphore, q)
            results.append(result)
            print(f"  有效规划: {len(result['plans'])}/{config.PLAN_SAMPLE_K}")

    # 4. 写入输出文件
    with open(config.PLAN_SAMPLES_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. 打印摘要
    total_plans = sum(len(r["plans"]) for r in results)
    print(f"\n[INFO] 采样完成！共 {len(results)} 条问题，{total_plans} 个有效规划")
    print(f"[INFO] 结果已写入 {config.PLAN_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

