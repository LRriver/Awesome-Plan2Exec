"""
Plan2Exec 数据合成流水线 — 第一阶段：问题生成器
从场景-工具集数据中选取代表性场景，为每个场景生成 4 种难度的用户问题。
"""
import asyncio
import json
import sys

import aiohttp

import config
from utils import call_llm, parse_json_response


# 硬编码的代表性场景选取列表（13 个）
# 选取原则：
#   1. 覆盖不同领域（金融、医疗、旅游、项目管理、智能家居、文化遗产、营销、安全、地质、生态、健康、娱乐）
#   2. 工具数量差异性（范围 10-29，含 11 个不同值）
#   3. 排除语义高度重复的场景
SELECTED_SCENARIOS = [
    "股票历史价格查询",           # 金融 — 23 tools
    "医疗机器人健康监控与维护",     # 医疗/机器人 — 28 tools
    "旅游出行天气规划",           # 旅游/天气 — 19 tools
    "敏捷开发项目任务规划",        # 项目管理 — 23 tools
    "自动解析邮件推荐温度并调温",    # 智能家居/IoT — 12 tools
    "文化遗产空气污染风险评估",     # 文化遗产保护 — 18 tools
    "跨区域漫画营销策划",         # 营销/创意 — 19 tools
    "图书馆账户密码安全策略执行",    # 安全/权限管理 — 13 tools
    "地质样本分析与区域对比研究",    # 地质科学 — 11 tools
    "淡水生态系统入侵物种研究",     # 生态研究 — 17 tools
    "跨国家庭旅行预算规划",        # 家庭旅行/财务 — 29 tools
    "个性化健康饮食方案",         # 健康/饮食 — 21 tools
    "音乐榜单跨界合作提案",        # 娱乐/音乐 — 10 tools
]


def load_scenarios() -> list[dict]:
    """加载并选取 12-13 个代表性场景。

    从 config.INPUT_FILE 加载全部场景数据，
    按 SELECTED_SCENARIOS 硬编码列表过滤，返回选中的场景子集。

    Returns:
        选中场景的列表，每个元素为 {"scenario": str, "tools": dict, "tools_count": int}

    Raises:
        SystemExit: 输入文件不存在时输出错误信息并终止
    """
    input_path = config.INPUT_FILE
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    selected_set = set(SELECTED_SCENARIOS)
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data["scenario"] in selected_set:
                results.append(data)
                selected_set.discard(data["scenario"])

    if selected_set:
        print(f"[WARN] 以下场景在输入文件中未找到: {selected_set}")

    print(f"[INFO] 已加载 {len(results)} 个代表性场景")
    return results


def build_question_prompt(scenario: str, tools: dict) -> str:
    """构建问题生成 Prompt，注入场景名称和工具集描述。

    Args:
        scenario: 场景名称
        tools: 工具名-描述映射字典

    Returns:
        完整的 Prompt 字符串
    """
    tools_description_json = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""你是一个用户行为模拟器。现在有一个特定场景：【{scenario}】
在该场景下，你有以下可用工具集：
{tools_description_json}

请基于这些工具，生成 4 个不同复杂度的用户提问（User Query）：
1. [简单] 仅需使用 1 个工具就能解决的问题。问题要具体，包含明确的参数信息（如具体的名称、代码、日期等）。
2. [并行] 需要使用 2-3 个工具，但步骤之间没有先后依赖关系。问题应自然地包含多个并列需求。
3. [复杂依赖] 需要多步完成，且后一个步骤强烈依赖前一个步骤的查询结果。问题应体现出明确的因果链条。
4. [闲聊/基础] 与场景相关，但不需要使用任何工具，可以直接回答的问题。

要求：
- 问题要具体、真实，像真实用户会问的那样，避免过于笼统
- 问题中应包含具体的实体信息（如股票代码、城市名、日期等），而非泛泛而谈
- 每个问题独立，不要互相引用

输出格式要求为纯 JSON 数组：
[
  {{"difficulty": "simple", "query": "..."}},
  {{"difficulty": "parallel", "query": "..."}},
  {{"difficulty": "complex_dependency", "query": "..."}},
  {{"difficulty": "chat", "query": "..."}}
]"""


async def generate_questions_for_scenario(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    scenario_data: dict,
) -> list[dict]:
    """异步生成单个场景的 4 种难度用户问题。

    Args:
        session: aiohttp 会话
        semaphore: 并发控制信号量
        scenario_data: 场景数据，包含 scenario 和 tools 字段

    Returns:
        问题列表，每个元素包含 scenario、tools、difficulty、query 字段。
        解析失败时返回空列表（跳过该场景）。
    """
    scenario = scenario_data["scenario"]
    tools = scenario_data["tools"]
    prompt = build_question_prompt(scenario, tools)

    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        try:
            raw = await call_llm(session, messages)
        except Exception as e:
            print(f"[ERROR] 场景 '{scenario}' LLM 调用失败: {e}")
            return []

        await asyncio.sleep(config.REQUEST_DELAY)

    try:
        questions = parse_json_response(raw)
    except Exception as e:
        print(f"[ERROR] 场景 '{scenario}' JSON 解析失败: {e}")
        print(f"[DEBUG] 原始响应: {raw[:500]}")
        return []

    results = []
    for q in questions:
        results.append({
            "scenario": scenario,
            "tools": tools,
            "difficulty": q["difficulty"],
            "query": q["query"],
        })

    print(f"[INFO] 场景 '{scenario}' 生成 {len(results)} 个问题")
    return results


async def main():
    """入口：加载场景 → 并发生成问题 → 写入 JSONL"""
    scenarios = load_scenarios()
    if not scenarios:
        print("[ERROR] 没有加载到任何场景，终止执行")
        return

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_questions_for_scenario(session, semaphore, s)
            for s in scenarios
        ]
        results = await asyncio.gather(*tasks)

    all_questions = []
    for question_list in results:
        all_questions.extend(question_list)

    with open(config.QUESTIONS_FILE, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"[INFO] 共生成 {len(all_questions)} 条问题，已写入 {config.QUESTIONS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
