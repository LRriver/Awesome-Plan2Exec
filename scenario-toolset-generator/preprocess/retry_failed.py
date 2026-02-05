#!/usr/bin/env python3
"""
重新处理失败的数据，并清理临时字段
"""
import asyncio
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent.resolve()

MAX_CONCURRENCY = 2  # 重试的降低并发
LLM_BASE_URL = "http://0.0.0.0:6001/v1"
LLM_MODEL = "qwen3-30b"
MAX_PROMPT_LENGTH = 25000  # 最大prompt长度

PREDEFINED_CATEGORIES = [
    "Data Analysis", "Entertainment", "Travel", "Marketing", "FinServ",
    "Media", "IT/Sec", "Healthcare", "Education", "ESG", "E-comm", 
    "Web3", "Finance", "Engineering", "Hospitality", "Academia", "Gov", "Legal"
]

class ToolsetExtraction(BaseModel):
    labels: List[str] = Field(min_length=2, max_length=5)
    summary: str = Field(min_length=5, max_length=100)

def build_extraction_prompt(toolset_data: dict, max_conversations: int = None) -> str:
    tools_text = ""
    for name, desc in toolset_data['tools'].items():
        tools_text += f"- {name}: {desc[:100]}\n"
    
    conversations = toolset_data['conversations']
    if max_conversations and len(conversations) > max_conversations:
        conversations = conversations[:max_conversations]
    
    conversations_text = ""
    for i, conv in enumerate(conversations, 1):
        conversations_text += f"\n【对话{i}】\n"
        for turn in conv['dialogue']:
            content = turn['content'][:200]
            conversations_text += f"{turn['role']}: {content}\n"
        if conv['called_tools']:
            conversations_text += f"召回工具: {', '.join(conv['called_tools'])}\n"
    
    return f"""你是一个专业的信息分类助手。请分析以下工具集和对话内容，提取标签和概要。

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
- 如果内容属于以下预定义类别，必须使用对应标签：
  {', '.join(PREDEFINED_CATEGORIES)}
- 可以添加其他具体场景标签

### 输出格式（JSON）
{{"labels": ["标签1", "标签2"], "summary": "一句话概要"}}

请直接输出JSON，不要有其他内容。"""

async def extract_single(client, data, semaphore, max_retries=5):
    async with semaphore:
        max_conversations = None
        
        for attempt in range(max_retries):
            try:
                prompt = build_extraction_prompt(data, max_conversations)
                
                # 检查prompt长度，如果超过限制则截断对话
                if len(prompt) > MAX_PROMPT_LENGTH:
                    total_convs = len(data['conversations'])
                    ratio = MAX_PROMPT_LENGTH / len(prompt)
                    max_conversations = max(5, int(total_convs * ratio * 0.8))
                    prompt = build_extraction_prompt(data, max_conversations)
                
                response = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                parsed = json.loads(content)
                validated = ToolsetExtraction(**parsed)
                return validated.labels, validated.summary, True
            except Exception as e:
                # 如果是长度超限错误，尝试截断
                if "maximum context length" in str(e) or "too long" in str(e).lower():
                    total_convs = len(data['conversations'])
                    if max_conversations is None:
                        max_conversations = min(10, total_convs // 2)
                    else:
                        max_conversations = max(3, max_conversations // 2)
                
                if attempt == max_retries - 1:
                    return [], "", False
                await asyncio.sleep(1)

async def main():
    input_file = SCRIPT_DIR / 'merged_with_labels_clean.jsonl'
    
    # 读取所有数据
    all_data = []
    failed_indices = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            all_data.append(data)
            # 检查labels为["Other"]的数据
            if data.get('labels') == ["Other"]:
                failed_indices.append(idx)
    
    print(f"总数据: {len(all_data)}, 待重试: {len(failed_indices)}")
    
    # 重新处理失败的
    if failed_indices:
        client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="EMPTY")
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        
        tasks = []
        for idx in failed_indices:
            tasks.append(extract_single(client, all_data[idx], semaphore))
        
        results = await asyncio.gather(*tasks)
        
        retry_success = 0
        for i, (labels, summary, success) in enumerate(results):
            idx = failed_indices[i]
            if success:
                all_data[idx]['labels'] = labels
                all_data[idx]['summary'] = summary
                retry_success += 1
            else:
                # 仍然失败，给默认值
                all_data[idx]['labels'] = ["Other"]
                all_data[idx]['summary'] = "未能提取概要"
        
        print(f"重试成功: {retry_success}/{len(failed_indices)}")
    
    # 写回原文件
    with open(input_file, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"已更新: {input_file}")

if __name__ == "__main__":
    asyncio.run(main())
