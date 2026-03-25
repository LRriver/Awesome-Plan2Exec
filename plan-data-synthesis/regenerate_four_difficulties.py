"""
临时脚本：仅重生成 4 个难度的问题（parallel / complex_dependency / long_chain / adversarial）。

- 读取与 generate_questions.py 相同的场景输入
- 沿用相同并发/重试/解析逻辑（复用 utils.call_llm / parse_json_response）
- 输出到独立文件，避免覆盖已有 questions.jsonl

用法：
  python regenerate_four_difficulties.py
  python regenerate_four_difficulties.py --output output/questions_four_difficulties.jsonl --limit 0
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import config
from utils import call_llm, parse_json_response


TARGET_DIFFICULTIES = [
    "parallel",
    "complex_dependency",
    "long_chain",
    "adversarial",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate only four difficulty questions for all scenarios."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT_DIR / "questions_four_difficulties.jsonl",
        help="输出 JSONL 文件路径（默认: output/questions_four_difficulties.jsonl）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=getattr(config, "SCENARIO_LIMIT", 0),
        help="场景加载上限，0 表示不限制（默认沿用 config.SCENARIO_LIMIT）",
    )
    return parser.parse_args()


def load_all_scenarios(limit: int = 0) -> list[dict]:
    """加载全部场景，limit=0 表示不限制。"""
    input_path = config.INPUT_FILE
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
            if limit and len(results) >= limit:
                break

    print(f"[INFO] 已加载 {len(results)} 个场景")
    return results


def build_prompt_for_four_difficulties(scenario: str, tools: dict) -> str:
    """构建 4 难度问题生成 Prompt（与 generate_questions.py 风格保持一致）。"""
    tools_description_json = json.dumps(tools, ensure_ascii=False, indent=2)

    return f"""你是一个高级用户行为模拟器，擅长生成各种复杂度和边界情况的用户提问。
现在有一个特定场景：【{scenario}】
在该场景下，可用工具集如下（共 {len(tools)} 个工具）：
{tools_description_json}

请基于这些工具，仅生成 4 个不同复杂度和类型的用户提问（User Query）。
注意：只生成以下 4 种难度，各 1 个问题，不要输出其他难度。

【4 种难度类型及生成要求】

1. [parallel] 需要至少 2-7 个工具(如果提供的工具集数量很多,可以选取大于7个)，问题应自然包含多个并列需求，即多个工具可以并行使用;也可以拆分成几个前后任务，每个任务需要并行，即多次并行的情况;注意不要求所有工具都必须要参与并行，大部分工具需要并行即可。

2. [complex_dependency] 需要至少 2-7 个工具（(如果提供的工具集数量很多,可以选取大于7个)），需要多步完成，后一步强烈依赖前一步的输出。体现明确的因果链条。

3. [long_chain] 长链条问题，需要 4 步以上的工具调用，步骤间存在复杂的依赖关系。
   要求至少需要 6-10 个工具（(如果提供的工具集数量很多,可以选取大于10个)），建议同时包含并行+串行的混合依赖结构，即步骤之间既有并行又有依赖关系，体现复杂的执行结构。

4. [adversarial] 对抗性扰动问题。设计容易误导模型做出错误规划的问题：
   - 【必须包含】把2-5个完全不相干的需求掺在一起，例如"帮我查一下北京天气，顺便写一首关于春天的诗",可以参考以下建议：
   - 包含误导性的上下文信息，诱导模型选错工具，或者问题表面看需要某工具，但仔细分析其实不需要（或需要另一个工具）
   - 用户给出了错误的前提假设，模型需要识别并纠正
   - 问题中包含自相矛盾的约束条件
   重点是让模型在工具选择和规划逻辑上犯错。

【生成原则】
- 问题要具体、真实，像真实用户会问的那样
- 高难度问题（long_chain/adversarial）是重点，要精心设计
- 每个问题独立，不要互相引用
- 问题中应包含具体的实体信息（如名称、编号、日期等）

输出格式为纯 JSON 数组：
[
  {{"difficulty": "parallel", "query": "..."}},
  {{"difficulty": "complex_dependency", "query": "..."}},
  {{"difficulty": "long_chain", "query": "..."}},
  {{"difficulty": "adversarial", "query": "..."}}
]"""


async def generate_four_questions_for_scenario(
    semaphore: asyncio.Semaphore,
    scenario_data: dict,
) -> list[dict]:
    """异步为单场景生成 4 个指定难度问题。"""
    scenario = scenario_data["scenario"]
    tools = scenario_data["tools"]
    prompt = build_prompt_for_four_difficulties(scenario, tools)
    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        try:
            raw = await call_llm(messages)
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

    if not isinstance(questions, list):
        print(f"[WARN] 场景 '{scenario}' 响应不是 JSON 数组，跳过")
        return []

    valid = set(TARGET_DIFFICULTIES)
    results = []
    seen = set()
    for q in questions:
        if not isinstance(q, dict):
            continue
        diff = q.get("difficulty", "")
        query = q.get("query", "")
        if diff not in valid:
            continue
        if not isinstance(query, str) or not query.strip():
            continue
        if diff in seen:
            continue
        seen.add(diff)
        results.append(
            {
                "scenario": scenario,
                "tools": tools,
                "difficulty": diff,
                "query": query,
            }
        )

    missing = [d for d in TARGET_DIFFICULTIES if d not in seen]
    if missing:
        print(f"[WARN] 场景 '{scenario}' 缺少难度: {missing}，实际生成 {len(results)} 条")
    else:
        print(f"[INFO] 场景 '{scenario}' 生成 {len(results)} 条（4/4）")

    return results


async def main() -> None:
    args = parse_args()
    scenarios = load_all_scenarios(limit=max(args.limit, 0))
    if not scenarios:
        print("[ERROR] 没有加载到任何场景，终止执行")
        return

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    all_questions = []
    buffer = []
    lock = asyncio.Lock()

    # 清空目标文件
    open(output_path, "w", encoding="utf-8").close()

    async def flush_buffer() -> None:
        if not buffer:
            return
        with open(output_path, "a", encoding="utf-8") as f:
            f.writelines(buffer)
        buffer.clear()

    async def process_scenario(s: dict) -> None:
        result = await generate_four_questions_for_scenario(semaphore, s)
        async with lock:
            for q in result:
                buffer.append(json.dumps(q, ensure_ascii=False) + "\n")
            all_questions.extend(result)
            if len(buffer) >= config.FLUSH_THRESHOLD:
                await flush_buffer()

    tasks = [process_scenario(s) for s in scenarios]
    await asyncio.gather(*tasks)
    await flush_buffer()

    dist = Counter(q["difficulty"] for q in all_questions)
    print("\n[INFO] 四难度重生成完成")
    print(f"[INFO] 共生成 {len(all_questions)} 条，已写入: {output_path}")
    print(f"[INFO] 难度分布: {dict(dist)}")


if __name__ == "__main__":
    asyncio.run(main())
