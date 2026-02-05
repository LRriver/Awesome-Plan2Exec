#!/usr/bin/env python3
"""
合并工具集重复的场景，用LLM生成上层场景名
然后按工具数量分为两个文件
"""
import json
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import List
import time
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent.resolve()

LLM_BASE_URL = "http://127.0.0.1:6001/v1"
LLM_MODEL = "qwen3-30b"
INPUT_FILE = SCRIPT_DIR / "scenario_tools.jsonl"
OUTPUT_GTE10 = SCRIPT_DIR / "scenario_tools_gte10.jsonl"
OUTPUT_LT10 = SCRIPT_DIR / "scenario_tools_lt10.jsonl"

MAX_CONCURRENCY = 30
MAX_RETRIES = 3


class MergedScenario(BaseModel):
    """合并后的场景"""
    scenario: str = Field(
        description="合并后的上层场景名称，10-20字",
        min_length=3,
        max_length=50
    )


def build_merge_prompt(scenarios: List[str]) -> str:
    """构建合并场景的提示词"""
    scenarios_str = "\n".join(f"- {s}" for s in scenarios)
    return f"""请将以下多个相关的任务场景合并为一个更上层、更通用的场景名称，这个名称应该能够涵盖所有子场景。

待合并的场景：
<scenarios_str>
{scenarios_str}
</scenarios_str>

要求：
1. 合并后的场景名应简洁（10-20字）
2. 要能涵盖所有子场景的共同主题
3. 不要太宽泛，保持一定的具体性

请以JSON格式返回，格式如下：
{{"scenario": "合并后的场景名"}}"""


async def merge_scenarios(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    scenarios: List[str]
) -> str:
    """合并多个场景为一个上层场景"""
    async with semaphore:
        prompt = build_merge_prompt(scenarios)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("LLM返回内容为空")
                
                # 清理content，提取JSON
                content = content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                # 尝试找到JSON对象
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                
                # 解析JSON
                data = json.loads(content)
                
                # Pydantic验证
                validated = MergedScenario(**data)
                return validated.scenario
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.3)
                    continue
                else:
                    # 失败则用第一个场景
                    return scenarios[0]
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.3)
                    continue
                else:
                    return scenarios[0]
        
        return scenarios[0]


async def main():
    # 1. 加载数据
    print("加载数据...")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  总场景数: {len(data)}")
    
    # 2. 按工具集分组
    print("\n按工具集分组...")
    toolset_to_items = defaultdict(list)
    for item in data:
        tools_key = tuple(sorted(item['tools'].keys()))
        toolset_to_items[tools_key].append(item)
    
    unique_count = len(toolset_to_items)
    dup_groups = {k: v for k, v in toolset_to_items.items() if len(v) > 1}
    print(f"  唯一工具集数: {unique_count}")
    print(f"  需要合并的组数: {len(dup_groups)}")
    
    # 3. 用LLM合并重复场景
    print(f"\n开始合并重复场景 (并发={MAX_CONCURRENCY})...")
    
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="EMPTY")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    merged_results = []  # 最终结果
    
    start_time = time.time()
    processed = 0
    
    # 处理需要合并的组
    merge_tasks = []
    merge_items = []  # 对应的item（取第一个的tools）
    
    for tools_key, items in toolset_to_items.items():
        if len(items) == 1:
            # 不需要合并，直接加入结果
            merged_results.append(items[0])
        else:
            # 需要合并
            scenarios = [item['scenario'] for item in items]
            merge_tasks.append(merge_scenarios(client, semaphore, scenarios))
            merge_items.append(items[0])  # 保留第一个的tools
    
    # 并发执行合并
    if merge_tasks:
        merged_scenarios = await asyncio.gather(*merge_tasks)
        
        for merged_scenario, item in zip(merged_scenarios, merge_items):
            merged_results.append({
                "scenario": merged_scenario,
                "tools": item['tools'],
                "tools_count": item['tools_count']
            })
            processed += 1
            
            if processed % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  已合并: {processed}/{len(dup_groups)}, 耗时: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\n合并完成! 耗时: {elapsed:.1f}s")
    print(f"  合并后总数: {len(merged_results)}")
    
    # 4. 按工具数量分文件
    print("\n按工具数量分文件...")
    gte10 = [item for item in merged_results if item['tools_count'] >= 10]
    lt10 = [item for item in merged_results if item['tools_count'] < 10]
    
    print(f"  工具数>=10: {len(gte10)}")
    print(f"  工具数<10: {len(lt10)}")
    
    # 5. 保存
    with open(OUTPUT_GTE10, 'w', encoding='utf-8') as f:
        for item in gte10:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n保存: {OUTPUT_GTE10}")
    
    with open(OUTPUT_LT10, 'w', encoding='utf-8') as f:
        for item in lt10:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"保存: {OUTPUT_LT10}")
    
    # 6. 显示一些合并示例
    print("\n合并示例:")
    shown = 0
    for tools_key, items in list(dup_groups.items())[:5]:
        original_scenarios = [item['scenario'] for item in items]
        # 找到合并后的结果
        for result in merged_results:
            if tuple(sorted(result['tools'].keys())) == tools_key:
                print(f"\n原场景: {original_scenarios}")
                print(f"合并后: {result['scenario']}")
                shown += 1
                break
        if shown >= 5:
            break


if __name__ == "__main__":
    asyncio.run(main())
