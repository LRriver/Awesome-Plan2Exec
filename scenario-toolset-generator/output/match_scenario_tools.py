#!/usr/bin/env python3
"""
对每个场景，判断簇内哪些工具可能会用到
输入: cluster_scenarios.jsonl
输出: scenario_tools.jsonl (场景 -> 工具列表)
"""
import json
import asyncio
import aiohttp
from pathlib import Path
from pydantic import BaseModel
import time

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
# 上级目录 (graph_syn_datasets)
BASE_DIR = SCRIPT_DIR.parent

# ============ 配置 ============
LLM_BASE_URL = "http://127.0.0.1:6001/v1"
LLM_MODEL = "qwen3-30b"
INPUT_FILE = BASE_DIR / "generate" / "cluster_scenarios.jsonl"
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = OUTPUT_DIR / "scenario_tools.jsonl"

MAX_CONCURRENCY = 50
MAX_RETRIES = 4


class ToolMatch(BaseModel):
    """工具匹配结果"""
    relevant: bool


def build_prompt(scenario: str, tool_name: str, tool_desc: str) -> str:
    """构建提示词"""
    return f"""判断以下工具是否可能在给定的任务场景中用到。

任务场景: {scenario}

工具名称: {tool_name}
工具描述: {tool_desc}

要求：
- 不需要工具与场景直接相关，只要在该场景下可能会用到即可
- 考虑工具的功能是否能辅助完成该场景的任务

请以JSON格式返回，格式如下：
{{"relevant": true}} 或 {{"relevant": false}}"""


async def call_llm(session: aiohttp.ClientSession, prompt: str) -> str:
    """调用LLM"""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 100
    }
    
    async with session.post(
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30)
    ) as resp:
        result = await resp.json()
        return result["choices"][0]["message"]["content"]


def parse_response(content: str) -> ToolMatch:
    """解析LLM响应"""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]
    
    data = json.loads(content)
    return ToolMatch(**data)


async def check_tool_relevance(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    scenario: str,
    tool_name: str,
    tool_desc: str
) -> tuple:
    """检查单个工具是否与场景相关"""
    async with semaphore:
        prompt = build_prompt(scenario, tool_name, tool_desc)
        
        for attempt in range(MAX_RETRIES):
            try:
                content = await call_llm(session, prompt)
                result = parse_response(content)
                return (tool_name, tool_desc, result.relevant)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    # 默认不相关
                    return (tool_name, tool_desc, False)
                await asyncio.sleep(0.3)


async def process_scenario(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    scenario: str,
    tools: dict  # tool_name -> tool_desc
) -> dict:
    """处理单个场景"""
    # 并发检查所有工具
    tasks = [
        check_tool_relevance(session, semaphore, scenario, name, desc)
        for name, desc in tools.items()
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 收集相关工具
    relevant_tools = {}
    for tool_name, tool_desc, is_relevant in results:
        if is_relevant:
            relevant_tools[tool_name] = tool_desc
    
    return {
        "scenario": scenario,
        "tools": relevant_tools,
        "tools_count": len(relevant_tools)
    }


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    clusters = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            clusters.append(json.loads(line))
    print(f"  共 {len(clusters)} 个簇")
    
    # 统计
    total_scenarios = sum(len(c['scenarios']) for c in clusters)
    print(f"  共 {total_scenarios} 个场景")
    
    # 检查已处理的簇 (断点续传)
    processed_cluster_ids = set()
    if OUTPUT_FILE.exists():
        # 读取已有结果，统计已处理的簇
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                pass  # 只是检查文件存在
        print(f"  发现已有输出文件，将追加写入")
    
    # 处理
    print(f"\n开始处理 (并发={MAX_CONCURRENCY})...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    start_time = time.time()
    processed_scenarios = 0
    processed_clusters = 0
    
    async with aiohttp.ClientSession() as session:
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            scenarios = cluster['scenarios']
            
            # 收集该簇所有工具 (从所有toolsets的tools合并)
            cluster_tools = {}
            for ts in cluster['toolsets']:
                # 只收集called_tools_unique中的工具
                for tool_name in ts.get('called_tools_unique', []):
                    if tool_name in ts.get('tools', {}):
                        cluster_tools[tool_name] = ts['tools'][tool_name]
            
            if not cluster_tools:
                continue
            
            # 处理该簇的每个场景
            cluster_results = []
            for scenario in scenarios:
                result = await process_scenario(
                    session, semaphore, scenario, cluster_tools
                )
                cluster_results.append(result)
                processed_scenarios += 1
            
            # 每处理完一个簇就写入文件
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for result in cluster_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            processed_clusters += 1
            elapsed = time.time() - start_time
            print(f"  簇{cluster_id}完成, 已处理: {processed_clusters}/{len(clusters)}簇, {processed_scenarios}/{total_scenarios}场景, 耗时: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\n完成! 耗时: {elapsed:.1f}s")
    print(f"总场景数: {processed_scenarios}")
    
    # 统计
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        all_results = [json.loads(line) for line in f]
    
    with_tools = sum(1 for r in all_results if r['tools_count'] > 0)
    print(f"有工具的场景: {with_tools} ({with_tools/len(all_results)*100:.1f}%)")
    
    avg_tools = sum(r['tools_count'] for r in all_results) / len(all_results)
    print(f"平均每场景工具数: {avg_tools:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
