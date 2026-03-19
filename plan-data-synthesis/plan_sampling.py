"""
Plan2Exec 数据合成流水线 — 第二阶段：规划采样器
将问题和工具集动态组装为 System Prompt，对每个问题多路采样生成结构化规划。

并发模式：所有问题的所有采样任务同时进入并发池，完成一个保存一个。
"""
import asyncio
import json
import sys

import aiohttp

import config
from utils import call_llm, parse_json_response


def build_system_prompt(tools: dict) -> str:
    """构建动态 System Prompt，将工具集注入模板。"""
    dynamic_loaded_tools_json = json.dumps(tools, ensure_ascii=False, indent=2)

    return f"""你是一个高级规划智能体（Planning Agent）。你的目标是根据用户的提问和提供的工具集，拆解并生成结构化的任务执行计划。

【当前可用工具集】
{dynamic_loaded_tools_json}

【规划规则】
1. 先输出 fixed_question：对用户原始问题进行补全和明确化，确保问题描述完整无歧义。如果问题存在歧义，在 fixed_question 中明确你的理解和假设。
2. 思考过程 (thought)：
   a. 先逐字分析用户需求，识别所有显性和隐性子任务
   b. 逐一评估可用工具与子任务的匹配关系，说明为什么选择工具 A 而非工具 B
   c. 分析任务的潜在风险和边界情况
   d. 规划步骤顺序和依赖关系
3. 步骤拆解 (steps)：每个步骤包含 thought、title、content、tools、dependencies 五个字段。
4. 工具选择：tools 只能从【当前可用工具集】中选取，严禁使用不在列表中的工具名。如果某步骤不需要工具，设为 null。
5. 依赖关系 (dependencies)：如果步骤 B 需要步骤 A 的输出结果才能执行，则 B 的 dependencies 设为 ["步骤A的title"]。如果无依赖，设为 null。
6. 无匹配工具：如果问题可以直接回答，无需伪造工具调用，tools 设为 null，给出一个"直接回答"步骤即可。
7. 步骤内容中引用前序步骤输出时，使用【步骤title】格式标注。
8. 安全与伦理：如果用户请求涉及违法、有害或不道德的行为，必须在 thought 中明确识别风险，并在步骤中拒绝执行有害部分，给出安全警告。
9. 模糊问题处理：如果用户问题存在歧义或信息不完整，在 thought 中指出歧义点，做出合理假设并在 fixed_question 中明确。

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
    """校验 Plan_Output 结构是否合法。"""
    if not isinstance(plan.get("fixed_question"), str):
        return False
    if not isinstance(plan.get("thought"), str):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return False
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
    """对单个问题进行一次规划采样。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        async with semaphore:
            raw = await call_llm(
                session, messages,
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


class StreamWriter:
    """带缓冲的流式 JSONL 写入器，累积到阈值后批量写入磁盘。"""

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
        """强制刷盘（结束时调用）。"""
        async with self._lock:
            self._do_flush()

    @property
    def total(self):
        return self._total_written + len(self._buffer)


async def sample_plans_for_question(session, semaphore, question_data: dict, writer: StreamWriter) -> dict:
    """对单个问题并发采样 K 次，每个采样完成后立即汇总。

    所有 K 次采样作为独立协程并发执行，不再串行等待。
    """
    system_prompt = build_system_prompt(question_data["tools"])
    user_query = question_data["query"]
    difficulty = question_data["difficulty"]
    scenario = question_data["scenario"]

    async def _one_sample(idx):
        result = await sample_plan(session, semaphore, system_prompt, user_query)
        status = "OK" if result else "FAIL"
        print(f"  [{scenario[:8]}][{difficulty}] Sample {idx + 1}/{config.PLAN_SAMPLE_K} — {status}")
        return result

    # 并发执行所有 K 次采样
    tasks = [_one_sample(i) for i in range(config.PLAN_SAMPLE_K)]
    results = await asyncio.gather(*tasks)
    plans = [r for r in results if r is not None]

    record = {
        "scenario": scenario,
        "tools": question_data["tools"],
        "difficulty": difficulty,
        "query": user_query,
        "plans": plans,
    }
    await writer.append(record)
    print(f"  [{scenario[:8]}][{difficulty}] 有效规划: {len(plans)}/{config.PLAN_SAMPLE_K}")
    return record


async def main():
    """入口：加载问题 → 全并发采样 → 流式写入"""
    if not config.QUESTIONS_FILE.exists():
        print(f"[ERROR] 问题文件不存在: {config.QUESTIONS_FILE}")
        sys.exit(1)

    questions = []
    with open(config.QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if not questions:
        print("[ERROR] 问题文件为空，终止执行")
        return

    print(f"[INFO] 已加载 {len(questions)} 条问题，每条采样 {config.PLAN_SAMPLE_K} 次")
    print(f"[INFO] 预计 {len(questions) * config.PLAN_SAMPLE_K} 次 LLM 调用，并发数 {config.MAX_CONCURRENCY}")

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    # 清空输出文件
    open(config.PLAN_SAMPLES_FILE, "w").close()
    writer = StreamWriter(config.PLAN_SAMPLES_FILE)

    async with aiohttp.ClientSession() as session:
        # 所有问题并发执行
        tasks = [
            sample_plans_for_question(session, semaphore, q, writer)
            for q in questions
        ]
        await asyncio.gather(*tasks)

    await writer.flush()

    total_plans = writer.total
    print(f"\n[INFO] 采样完成！共 {len(questions)} 条问题，结果已写入 {config.PLAN_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
