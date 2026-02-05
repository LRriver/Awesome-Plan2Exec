#!/usr/bin/env python3
"""
对每个簇提取中小任务场景
输入: clustered_toolsets.jsonl
输出: cluster_scenarios.jsonl
"""
import json
import asyncio
import aiohttp
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import time

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

LLM_BASE_URL = "http://127.0.0.1:6001/v1"
LLM_MODEL = "qwen3-30b"
INPUT_FILE = SCRIPT_DIR / "clustered_toolsets.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "cluster_scenarios.jsonl"

MAX_CONCURRENCY = 20
MAX_RETRIES = 3


class ScenarioResult(BaseModel):
    """场景提取结果"""
    scenarios: List[str] = Field(..., min_length=1, description="中小任务场景列表，至少1个")


def build_prompt(labels_summaries: List[dict]) -> str:
    """构建提示词"""
    content_lines = []
    for i, item in enumerate(labels_summaries, 1):
        content_lines.append(f"{i}. 标签: {', '.join(item['labels'])}")
        content_lines.append(f"   概要: {item['summary']}")
    
    content = "\n".join(content_lines)
    
    prompt = f"""你是一个任务场景分析专家。下面是同一个簇内多个工具集的标签和概要信息，它们在语义上相近。

请根据这些信息，总结出这个簇可能涵盖的中小任务场景。

要求：
1. 每个场景应该是具体的、可操作的任务描述，如"非洲旅游规划"、"股票投资分析"、"足球比赛数据查询"等
2. 场景粒度适中，不要太宽泛（如"数据分析"）也不要太细（如"查询某个具体API"）
3. 至少总结1个场景，如果内容丰富可以总结多个（一般1-10个）
4. 场景名称简洁，10-20个字为宜

工具集信息：
<content>
{content}
</content>

请以JSON格式返回，格式如下：
{{"scenarios": ["场景1", "场景2", ...]}}"""
    
    return prompt


async def call_llm(session: aiohttp.ClientSession, prompt: str) -> dict:
    """调用LLM"""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    async with session.post(
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        result = await resp.json()
        return result["choices"][0]["message"]["content"]


def parse_response(content: str) -> ScenarioResult:
    """解析LLM响应"""
    # 提取JSON
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
    
    data = json.loads(content)
    return ScenarioResult(**data)


async def process_cluster(
    session: aiohttp.ClientSession,
    cluster: dict,
    semaphore: asyncio.Semaphore
) -> dict:
    """处理单个簇"""
    async with semaphore:
        cluster_id = cluster["cluster_id"]
        toolsets = cluster["toolsets"]
        
        # 收集所有标签和概要
        labels_summaries = []
        for ts in toolsets:
            labels_summaries.append({
                "labels": ts.get("labels", []),
                "summary": ts.get("summary", "")
            })
        
        # 构建提示词
        prompt = build_prompt(labels_summaries)
        
        # 调用LLM并重试
        scenarios = []
        for attempt in range(MAX_RETRIES):
            try:
                content = await call_llm(session, prompt)
                result = parse_response(content)
                scenarios = result.scenarios
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  簇{cluster_id} 提取失败: {e}")
                    scenarios = ["未能提取场景"]
                await asyncio.sleep(0.5)
        
        # 构建输出数据
        output_toolsets = []
        for ts in toolsets:
            # 收集所有called_tools并去重
            all_called_tools = set()
            for conv in ts.get("conversations", []):
                for tool in conv.get("called_tools", []):
                    all_called_tools.add(tool)
            
            output_toolsets.append({
                "uid": ts.get("uid"),
                "labels": ts.get("labels", []),
                "summary": ts.get("summary", ""),
                "tools": ts.get("tools", {}),
                "called_tools_unique": list(all_called_tools)
            })
        
        return {
            "cluster_id": cluster_id,
            "scenarios": scenarios,
            "toolsets_count": len(toolsets),
            "toolsets": output_toolsets
        }


async def main():
    # 加载数据
    print("加载数据...")
    clusters = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            cluster = json.loads(line)
            # 跳过other
            if cluster["cluster_id"] == "other":
                continue
            clusters.append(cluster)
    print(f"  共 {len(clusters)} 个簇")
    
    # 处理
    print(f"\n开始处理 (并发={MAX_CONCURRENCY})...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    start_time = time.time()
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_cluster(session, c, semaphore) for c in clusters]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理: {i+1}/{len(clusters)}, 耗时: {elapsed:.1f}s")
    
    # 按cluster_id排序
    results.sort(key=lambda x: x["cluster_id"] if isinstance(x["cluster_id"], int) else 999999)
    
    # 保存
    print(f"\n保存到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    elapsed = time.time() - start_time
    print(f"\n完成! 耗时: {elapsed:.1f}s")
    
    # 统计
    total_scenarios = sum(len(r["scenarios"]) for r in results)
    print(f"总场景数: {total_scenarios}")
    print(f"平均每簇: {total_scenarios/len(results):.1f} 个场景")


if __name__ == "__main__":
    asyncio.run(main())
