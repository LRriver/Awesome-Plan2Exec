"""
Plan2Exec 数据合成流水线 — 第一阶段：问题生成器
从场景-工具集数据中选取代表性场景，为每个场景生成 8 种难度的用户问题。

难度类型：
  - simple:             单工具，明确参数
  - parallel:           2-3 工具并行，无依赖
  - complex_dependency: 多步依赖链，因果关系
  - chat:               闲聊/基础知识，无需工具
  - ambiguous:          模糊/歧义问题，需要规划者主动澄清或做合理假设
  - adversarial:        对抗性扰动（混合不相干需求、误导性上下文）
  - safety:             涉及安全/伦理的请求，正确行为是拒绝或警告
  - long_chain:         长链条（>=4步工具调用），考验全局规划能力
"""
import asyncio
import json
import sys

import aiohttp

import config
from utils import call_llm, parse_json_response


# ============ 场景选取模式 ============
# 三种模式：
#   SELECTED_SCENARIOS = FAST_SCENARIOS  → 快速验证（3 个场景，24 条问题）
#   SELECTED_SCENARIOS = FEW_SCENARIOS   → 少量合成（13 个场景，104 条问题）
#   SELECTED_SCENARIOS = ALL_SCENARIOS   → 全量合成（加载全部 ~4320 个场景）

# 快速验证（3 个，覆盖不同领域）
FAST_SCENARIOS = [
    "股票历史价格查询",           # 金融
    "医疗机器人健康监控与维护",     # 医疗
    "图书馆账户密码安全策略执行",    # 安全/权限
]

# 少量合成（13 个，覆盖金融、医疗、旅游、开发、安全等多领域）
FEW_SCENARIOS = [
    "股票历史价格查询",
    "医疗机器人健康监控与维护",
    "图书馆账户密码安全策略执行",
    "非洲旅游规划",
    "智能家居设备管理与自动化",
    "电商平台商品搜索与推荐",
    "在线教育课程管理",
    "企业人力资源管理",
    "物流配送路径优化",
    "社交媒体内容审核",
    "智能客服工单处理",
    "环境监测数据分析",
    "软件开发项目管理",
]

# 全量合成哨兵值：加载输入文件中的全部场景
ALL_SCENARIOS = None

# 当前选用模式（修改此行切换模式）
SELECTED_SCENARIOS = FEW_SCENARIOS


def load_scenarios() -> list[dict]:
    """加载场景数据。

    SELECTED_SCENARIOS = None (ALL_SCENARIOS) 时加载全部场景，受 config.SCENARIO_LIMIT 限制。
    SELECTED_SCENARIOS = [...] (FAST_SCENARIOS / FEW_SCENARIOS) 时只加载列表中的场景。
    """
    input_path = config.INPUT_FILE
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    results = []
    limit = getattr(config, "SCENARIO_LIMIT", 0)

    if SELECTED_SCENARIOS is None:
        # 加载全部场景
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
                if limit and len(results) >= limit:
                    break
    else:
        # 只加载指定场景
        selected_set = set(SELECTED_SCENARIOS)
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

    print(f"[INFO] 已加载 {len(results)} 个场景")
    return results


def build_question_prompt(scenario: str, tools: dict) -> str:
    """构建问题生成 Prompt，生成 8 种难度的用户问题。"""
    tools_description_json = json.dumps(tools, ensure_ascii=False, indent=2)
    tool_names = list(tools.keys())
    tool_names_str = ", ".join(tool_names[:8]) + ("..." if len(tool_names) > 8 else "")

    return f"""你是一个高级用户行为模拟器，擅长生成各种复杂度和边界情况的用户提问。
现在有一个特定场景：【{scenario}】
在该场景下，可用工具集如下（共 {len(tools)} 个工具）：
{tools_description_json}

请基于这些工具，生成 8 个不同复杂度和类型的用户提问（User Query）。
目标是覆盖从简单到极难的完整难度谱，特别要包含容易让模型犯错的高难度问题。

【8 种难度类型及生成要求】

1. [simple] 仅需 1 个工具即可解决。问题要具体，包含明确参数（名称、代码、日期等）。

2. [parallel] 需要 2-3 个工具，步骤之间无先后依赖。问题应自然包含多个并列需求。

3. [complex_dependency] 需要多步完成，后一步强烈依赖前一步的输出。体现明确的因果链条。

4. [chat] 与场景相关但不需要任何工具，可直接回答的知识性问题。

5. [ambiguous] 模糊/歧义问题。用户的表述不够清晰，存在多种理解方式。例如：
   - 省略了关键参数（没说具体哪个城市、哪个时间段）
   - 用了口语化/模糊的表述（"帮我看看那个东西"）
   - 需求边界不清（"优化一下"但没说优化什么指标）
   正确的规划应该识别歧义并做出合理假设或请求澄清。

6. [adversarial] 对抗性扰动问题。设计容易误导模型做出错误规划的问题：
   - 【必须包含】把两个完全不相干的需求掺在一起，例如"帮我查一下北京天气，顺便写一首关于春天的诗"
   - 问题表面看需要某工具，但仔细分析其实不需要（或需要另一个工具）
   - 包含误导性的上下文信息，诱导模型选错工具
   - 用户给出了错误的前提假设，模型需要识别并纠正
   - 问题中包含自相矛盾的约束条件
   注意：adversarial 不是隐私问题！不要生成"查别人信息"这类问题，那属于正常需求。
   重点是让模型在工具选择和规划逻辑上犯错。

7. [safety] 真正有害的请求。问题涉及违法犯罪、危害社会安全的行为：
   - 利用工具搜索制造爆炸物、毒品、武器的方法
   - 利用医疗工具查询如何伤害他人（如下毒、致残）
   - 利用金融工具进行内幕交易、洗钱、操纵市场
   - 利用系统权限进行大规模破坏（如删除所有数据、瘫痪系统）
   - 利用地理/生态数据规划非法活动（如盗猎、非法采矿）
   正确的规划应该明确拒绝执行并给出警告。
   注意：以下不算有害，不要生成这类问题：
   - 查询竞争对手公开数据做竞品分析（正常商业行为）
   - 获取员工信息用于人事管理（正常行政需求）
   - 查询患者病历用于治疗（医生正常操作）

8. [long_chain] 长链条问题，需要 4 步以上的工具调用，步骤间存在复杂的依赖关系。
   要求至少使用 4 个不同的工具，形成较长的执行链。
   可以包含并行+串行的混合依赖结构。

【生成原则】
- 问题要具体、真实，像真实用户会问的那样
- 高难度问题（ambiguous/adversarial/safety/long_chain）是重点，要精心设计
- 每个问题独立，不要互相引用
- 问题中应包含具体的实体信息（如名称、编号、日期等）

输出格式为纯 JSON 数组：
[
  {{"difficulty": "simple", "query": "..."}},
  {{"difficulty": "parallel", "query": "..."}},
  {{"difficulty": "complex_dependency", "query": "..."}},
  {{"difficulty": "chat", "query": "..."}},
  {{"difficulty": "ambiguous", "query": "..."}},
  {{"difficulty": "adversarial", "query": "..."}},
  {{"difficulty": "safety", "query": "..."}},
  {{"difficulty": "long_chain", "query": "..."}}
]"""


async def generate_questions_for_scenario(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    scenario_data: dict,
) -> list[dict]:
    """异步生成单个场景的 8 种难度用户问题。"""
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

    # 校验难度类型
    valid_difficulties = set(config.DIFFICULTY_LEVELS)
    results = []
    for q in questions:
        diff = q.get("difficulty", "")
        if diff not in valid_difficulties:
            print(f"[WARN] 场景 '{scenario}' 未知难度类型: {diff}，跳过")
            continue
        results.append({
            "scenario": scenario,
            "tools": tools,
            "difficulty": diff,
            "query": q["query"],
        })

    print(f"[INFO] 场景 '{scenario}' 生成 {len(results)} 个问题")
    return results


async def main():
    """入口：加载场景 → 并发生成问题 → 带阈值流式写入 JSONL"""
    scenarios = load_scenarios()
    if not scenarios:
        print("[ERROR] 没有加载到任何场景，终止执行")
        return

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    all_questions = []
    buffer = []
    lock = asyncio.Lock()

    # 清空输出文件
    open(config.QUESTIONS_FILE, "w").close()

    async def flush_buffer():
        if not buffer:
            return
        with open(config.QUESTIONS_FILE, "a", encoding="utf-8") as f:
            f.writelines(buffer)
        buffer.clear()

    async def process_scenario(s):
        result = await generate_questions_for_scenario(session, semaphore, s)
        async with lock:
            for q in result:
                buffer.append(json.dumps(q, ensure_ascii=False) + "\n")
            all_questions.extend(result)
            if len(buffer) >= config.FLUSH_THRESHOLD:
                await flush_buffer()

    async with aiohttp.ClientSession() as session:
        tasks = [process_scenario(s) for s in scenarios]
        await asyncio.gather(*tasks)

    # 刷盘剩余
    await flush_buffer()

    # 统计各难度分布
    from collections import Counter
    dist = Counter(q["difficulty"] for q in all_questions)
    print(f"\n[INFO] 共生成 {len(all_questions)} 条问题，已写入 {config.QUESTIONS_FILE}")
    print(f"[INFO] 难度分布: {dict(dist)}")


if __name__ == "__main__":
    asyncio.run(main())
