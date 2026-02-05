#!/usr/bin/env python3
"""
对merged_by_toolset.jsonl中的每个工具集提取标签和概要
- 使用本地部署的LLM (6001端口)
- 并发调用
- Pydantic验证
- 每50条写入一次防止数据丢失
"""
import asyncio
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent.resolve()
MAX_CONCURRENCY = 50  # 最大并发数
LLM_BASE_URL = "http://0.0.0.0:6001/v1"
LLM_MODEL = "qwen3-30b"
BATCH_SIZE = 50  # 每处理50条写入一次

# 预定义的类别标签
PREDEFINED_CATEGORIES = [
    "Data Analysis", "Entertainment", "Travel", "Marketing", "FinServ",
    "Media", "IT/Sec", "Healthcare", "Education", "ESG", "E-comm", 
    "Web3", "Finance", "Engineering", "Hospitality", "Academia", "Gov", "Legal"
]

class ToolsetExtraction(BaseModel):
    """工具集提取结果"""
    labels: List[str] = Field(
        description="2-5个标签，描述该工具集的应用领域和场景",
        min_length=2,
        max_length=5
    )
    summary: str = Field(
        description="一句话概要，描述该工具集对话的任务场景",
        min_length=5,
        max_length=100
    )
    
    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v):
        if len(v) < 2 or len(v) > 5:
            raise ValueError(f"标签数量必须在2-5个之间，当前{len(v)}个")
        return v

MAX_PROMPT_LENGTH = 25000  # 最大prompt长度（字符），留一些余量

def build_extraction_prompt(toolset_data: dict, max_conversations: int = None) -> str:
    """构建提取prompt
    
    Args:
        toolset_data: 工具集数据
        max_conversations: 最大对话数，None表示不限制
    """
    
    # 构建工具列表
    tools_text = ""
    for name, desc in toolset_data['tools'].items():
        tools_text += f"- {name}: {desc[:100]}\n"
    
    # 构建对话内容
    conversations = toolset_data['conversations']
    if max_conversations and len(conversations) > max_conversations:
        conversations = conversations[:max_conversations]
    
    conversations_text = ""
    for i, conv in enumerate(conversations, 1):
        conversations_text += f"\n【对话{i}】\n"
        for turn in conv['dialogue']:
            role = turn['role']
            content = turn['content'][:200]  # 截断过长内容
            conversations_text += f"{role}: {content}\n"
        if conv['called_tools']:
            conversations_text += f"召回工具: {', '.join(conv['called_tools'])}\n"
    
    prompt = f"""你是一个专业的信息分类助手。请分析以下工具集和对话内容，提取标签和概要。

### 工具集
{tools_text}

### 对话内容
<conversation_content>
{conversations_text}
</conversation_content>

### 请基于上述<conversation_content>标签包裹的内容，完成以下提取任务：
1. 提取2-5个标签（labels），描述该工具集的应用领域和场景
2. 生成一句话概要（summary），描述这些对话的核心任务场景

### 标签要求
- 标签数量：2-5个
- 如果内容属于以下预定义类别，必须使用对应标签：
  {', '.join(PREDEFINED_CATEGORIES)}
- 可以添加其他具体场景标签，如：时间查询、天气预报、股票分析、旅游规划、音乐推荐等
- 标签应简洁明确，优先使用英文或中文短语

### 概要要求
- 一句话，10-50字
- 描述对话的核心任务，如：查询城市时间、假期出行规划旅游行程、分析股票走势、汇报PPT制作、调研报告生成等

### 输出格式（JSON）
{{
    "labels": ["标签1", "标签2", "标签3"],
    "summary": "一句话概要描述任务场景"
}}

请直接输出JSON，不要有其他内容。"""
    
    return prompt


async def extract_toolset_info(
    client: AsyncOpenAI,
    toolset_data: dict,
    toolset_idx: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3
) -> dict:
    """
    提取单个工具集的标签和概要
    """
    async with semaphore:
        max_conversations = None  # 初始不限制
        
        for attempt in range(max_retries):
            try:
                prompt = build_extraction_prompt(toolset_data, max_conversations)
                
                # 检查prompt长度，如果超过限制则截断对话
                if len(prompt) > MAX_PROMPT_LENGTH:
                    # 估算需要保留的对话数
                    total_convs = len(toolset_data['conversations'])
                    ratio = MAX_PROMPT_LENGTH / len(prompt)
                    max_conversations = max(5, int(total_convs * ratio * 0.8))
                    prompt = build_extraction_prompt(toolset_data, max_conversations)
                
                response = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                
                content = response.choices[0].message.content
                
                if not content or not content.strip():
                    raise ValueError("LLM返回内容为空")
                
                # 清理可能的markdown标记
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # 解析JSON
                data = json.loads(content)
                
                if "labels" not in data or "summary" not in data:
                    raise ValueError(f"缺少必需字段: {list(data.keys())}")
                
                # Pydantic验证
                validated = ToolsetExtraction(**data)
                
                # 返回结果，合并原数据
                result = toolset_data.copy()
                result['labels'] = validated.labels
                result['summary'] = validated.summary
                result['_extract_success'] = True
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    # 如果是最后一次重试前，尝试截断对话
                    if attempt == max_retries - 2 and max_conversations is None:
                        total_convs = len(toolset_data['conversations'])
                        if total_convs > 10:
                            max_conversations = min(10, total_convs // 2)
                    continue
                else:
                    result = toolset_data.copy()
                    result['labels'] = []
                    result['summary'] = ""
                    result['_extract_success'] = False
                    result['_extract_error'] = f"解析失败(重试{max_retries}次): {str(e)}"
                    return result
                    
            except Exception as e:
                result = toolset_data.copy()
                result['labels'] = []
                result['summary'] = ""
                result['_extract_success'] = False
                result['_extract_error'] = str(e)
                return result


async def process_batch(
    client: AsyncOpenAI,
    batch: List[tuple],  # [(idx, data), ...]
    semaphore: asyncio.Semaphore
) -> List[dict]:
    """处理一批数据"""
    tasks = [
        extract_toolset_info(client, data, idx, semaphore)
        for idx, data in batch
    ]
    results = await asyncio.gather(*tasks)
    return results


async def main_async(input_file: str, output_file: str, max_items: Optional[int] = None):
    """异步主函数"""
    
    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key="EMPTY"
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # 读取数据
    print(f"读取数据: {input_file}")
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    total = len(all_data)
    if max_items:
        all_data = all_data[:max_items]
        print(f"限制处理数量: {max_items}")
    
    print(f"总数据量: {total}, 待处理: {len(all_data)}")
    
    # 处理数据
    start_time = time.time()
    processed = 0
    success_count = 0
    fail_count = 0
    
    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        pass
    
    # 分批处理
    batch = []
    for idx, data in enumerate(all_data):
        batch.append((idx, data))
        
        if len(batch) >= BATCH_SIZE:
            # 批处理
            results = await process_batch(client, batch, semaphore)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    if result.get('_extract_success'):
                        success_count += 1
                    else:
                        fail_count += 1
            
            processed += len(batch)
            elapsed = time.time() - start_time
            print(f"已处理: {processed}/{len(all_data)}, 成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed:.1f}s")
            
            batch = []
    
    # 处理剩余的
    if batch:
        results = await process_batch(client, batch, semaphore)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                if result.get('_extract_success'):
                    success_count += 1
                else:
                    fail_count += 1
        
        processed += len(batch)
    
    # 统计
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n处理完成!")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  吞吐量: {processed/elapsed:.2f} 条/秒")
    print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='提取工具集标签和概要')
    parser.add_argument('--max', type=int, default=None, help='最大处理数量，不填则处理全部')
    args = parser.parse_args()
    
    input_file = SCRIPT_DIR / 'merged_by_toolset.jsonl'
    output_file = SCRIPT_DIR / 'merged_with_labels.jsonl'
    
    asyncio.run(main_async(str(input_file), str(output_file), args.max))


if __name__ == "__main__":
    main()
