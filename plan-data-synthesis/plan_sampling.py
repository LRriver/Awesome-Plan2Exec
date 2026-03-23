"""
Plan2Exec 数据合成流水线 — 第二阶段：规划采样器
将问题和工具集动态组装为 System Prompt，对每个问题多路采样生成结构化规划。

并发模式：所有问题的所有采样任务同时进入并发池，完成一个保存一个。
"""
import asyncio
import json
import sys

import config
from utils import call_llm, parse_json_response


def _build_difficulty_guidance(difficulty: str) -> str:
    """根据问题难度注入专项规划要求。"""
    if difficulty == "parallel":
        return """
【难度专项要求：parallel】
- 该问题应体现并行多工具执行：请把可并行的子任务拆成多条步骤，且这些并行步骤的 dependencies 设为 null。
- 对于同一批可并行步骤，请设置相同的 parallel_group（如 "pg1"）。
- 如果后续需要汇总并行结果，必须新增一个汇总步骤，并让该步骤 dependencies 同时依赖所有并行步骤 title。
- 不要把本可并行的步骤强行串行化。"""
    if difficulty == "complex_dependency":
        return """
【难度专项要求：complex_dependency】
- 该问题应体现强依赖链：后续步骤必须显式依赖前序关键步骤输出。
- 每一处“根据上一步结果继续处理”的语义，都应在 dependencies 中有对应 title。
- 避免伪依赖：若步骤实际不依赖前序输出，dependencies 应为 null。"""
    if difficulty == "long_chain":
        return """
【难度专项要求：long_chain】
- 该问题需要长链条执行，步骤数量通常应 >= 6，任务之间可能既有依赖关系，又有并行关系。
- 优先形成“并行组 + 串行汇总”的混合结构：并行组内部 dependencies 为 null，组间通过 dependencies 连接。
- 并行组建议使用 parallel_group 标注（如 "pg1"、"pg2"），便于区分多组并行。
- 尽量覆盖更多不同工具，避免无意义重复调用同一工具。"""
    if difficulty == "adversarial":
        return """
【难度专项要求：adversarial】
- 识别问题中的误导信息、无关需求和自相矛盾约束。
- 在 thought 中明确指出哪些需求可执行，哪些应拒绝或澄清。
- 避免被表面描述误导而选错工具。"""
    if difficulty == "safety":
        return """
【难度专项要求：safety】
- 识别有害/违法请求并拒绝执行有害部分。
- 不应给出可直接执行有害行为的工具调用步骤。
- 在 thought 与 steps 中都体现安全边界和替代性建议。"""
    if difficulty == "ambiguous":
        return """
【难度专项要求：ambiguous】
- 识别歧义点并在 fixed_question 中补全合理假设。
- 若缺少关键参数，应设计澄清步骤或显式声明假设。"""
    if difficulty == "chat":
        return """
【难度专项要求：chat】
- 该问题通常无需工具，优先输出直接回答方案（tools=null）。"""
    return """
【难度专项要求：simple】
- 该问题应尽量简洁，通常使用单工具即可完成。"""


def build_system_prompt(tools: dict, difficulty: str = "simple") -> str:
    """构建动态 System Prompt，将工具集注入模板。"""
    dynamic_loaded_tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    difficulty_guidance = _build_difficulty_guidance(difficulty)

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
10. 并行标注 (parallel_group)：
    - 若某步骤与其他步骤可并行执行，为这些步骤设置相同的 parallel_group（如 "pg1"）。
    - 若步骤不属于并行组，parallel_group 设为 null。
    - 同一 parallel_group 内步骤不应互相依赖（dependencies 应为 null）。
11. 混合拓扑（并行 + 依赖）：
    - 允许存在“多组并行，且组与组之间有依赖”的结构。
    - 推荐模式：先执行 pg1 并行组 -> 汇总步骤 A_merge -> 再执行 pg2 并行组 -> 汇总步骤 B_merge。
    - 组间依赖应通过明确的 dependencies 表达，避免跨组隐式依赖。

【当前问题难度】
{difficulty}

{difficulty_guidance}

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
            "dependencies": ["依赖的步骤title"] 或 null,
            "parallel_group": "pg1" 或 null
    }}
  ]
}}

示例 D：多组并行，组间存在依赖关系（并行+依赖混合）
{{
    "fixed_question": "用户需要先并行获取多个数据源信息并做第一轮汇总，再基于汇总结果并行执行后续分析，最后输出综合报告。",
    "thought": "任务可拆分为两层并行组：第一组并行采集，汇总后驱动第二组并行分析，最后统一汇总输出。",
    "steps": [
        {{
            "thought": "采集数据源A，与其他采集步骤互不依赖。",
            "title": "采集数据源A",
            "content": "调用数据源A工具获取原始数据。",
            "tools": ["tool_a"],
            "dependencies": null,
            "parallel_group": "pg1"
        }},
        {{
            "thought": "采集数据源B，可与采集A并行。",
            "title": "采集数据源B",
            "content": "调用数据源B工具获取原始数据。",
            "tools": ["tool_b"],
            "dependencies": null,
            "parallel_group": "pg1"
        }},
        {{
            "thought": "第一组并行结果汇总后才能进入下一阶段。",
            "title": "汇总第一组结果",
            "content": "整合【采集数据源A】与【采集数据源B】输出，生成统一中间结果。",
            "tools": ["merge_tool"],
            "dependencies": ["采集数据源A", "采集数据源B"],
            "parallel_group": null
        }},
        {{
            "thought": "基于第一轮汇总做质量分析，可与风险分析并行。",
            "title": "质量分析",
            "content": "使用【汇总第一组结果】产物进行质量评估。",
            "tools": ["quality_tool"],
            "dependencies": ["汇总第一组结果"],
            "parallel_group": "pg2"
        }},
        {{
            "thought": "基于第一轮汇总做风险分析，可与质量分析并行。",
            "title": "风险分析",
            "content": "使用【汇总第一组结果】产物进行风险评估。",
            "tools": ["risk_tool"],
            "dependencies": ["汇总第一组结果"],
            "parallel_group": "pg2"
        }},
        {{
            "thought": "最终步骤汇总第二组并行分析结果并输出。",
            "title": "输出综合报告",
            "content": "汇总【质量分析】和【风险分析】结果，形成最终报告。",
            "tools": null,
            "dependencies": ["质量分析", "风险分析"],
            "parallel_group": null
        }}
    ]
}}"

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
            "dependencies": null,
            "parallel_group": "pg1"
    }},
    {{
      "thought": "...",
      "title": "查询高铁票",
      "content": "查询天津到北京的高铁票信息...",
      "tools": ["get_tickets"],
            "dependencies": null,
            "parallel_group": "pg1"
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
            "dependencies": null,
            "parallel_group": null
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
            "dependencies": null,
            "parallel_group": null
    }},
    {{
      "thought": "...",
      "title": "计算股票涨跌幅",
      "content": "根据【查询股票实时价格】的输出，计算该股票今天的涨跌幅和涨跌额。",
      "tools": ["caculator"],
            "dependencies": ["查询股票实时价格"],
            "parallel_group": null
    }},
    {{
      "thought": "...",
      "title": "分析涨跌幅并给出建议",
      "content": "根据【计算股票涨跌幅】的输出，分析股票的涨跌情况，并给出简要的投资建议。",
      "tools": null,
            "dependencies": ["计算股票涨跌幅"],
            "parallel_group": null
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
        pg = step.get("parallel_group")
        if pg is not None and not isinstance(pg, str):
            return False
    return True


async def sample_plan(semaphore, system_prompt: str, user_query: str) -> dict | None:
    """对单个问题进行一次规划采样。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        async with semaphore:
            raw = await call_llm(
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


async def sample_plans_for_question(semaphore, question_data: dict, writer: StreamWriter) -> dict:
    """对单个问题并发采样 K 次，每个采样完成后立即汇总。

    所有 K 次采样作为独立协程并发执行，不再串行等待。
    """
    system_prompt = build_system_prompt(question_data["tools"], question_data.get("difficulty", "simple"))
    user_query = question_data["query"]
    difficulty = question_data["difficulty"]
    scenario = question_data["scenario"]

    async def _one_sample(idx):
        result = await sample_plan(semaphore, system_prompt, user_query)
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

    # 所有问题并发执行
    tasks = [
        sample_plans_for_question(semaphore, q, writer)
        for q in questions
    ]
    await asyncio.gather(*tasks)

    await writer.flush()

    total_plans = writer.total
    print(f"\n[INFO] 采样完成！共 {len(questions)} 条问题，结果已写入 {config.PLAN_SAMPLES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
