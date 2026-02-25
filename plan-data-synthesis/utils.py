"""
Plan2Exec 数据合成流水线 — 公共工具函数
包含 LLM 异步调用（含重试）和 JSON 解析容错。
"""
import asyncio
import json

import aiohttp

import config


async def call_llm(
    session: aiohttp.ClientSession,
    messages: list[dict],
    temperature: float = 0.3,
    top_p: float = 1.0,
) -> str:
    """调用 LLM，返回 content 字符串。

    使用 aiohttp 异步调用 OpenAI 格式 API，超时 120 秒，
    失败时采用指数退避策略重试最多 MAX_RETRIES 次。
    """
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < config.MAX_RETRIES:
                wait = config.RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"[WARN] LLM call failed (attempt {attempt + 1}): {e}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[ERROR] LLM call failed after {config.MAX_RETRIES + 1} attempts: {e}")
                raise


def parse_json_response(content: str) -> dict | list:
    """多层容错解析 LLM 返回的 JSON。

    处理策略：
    1. 提取 markdown 代码块（```json ... ``` 或 ``` ... ```）
    2. 定位 JSON 边界（首个 { 或 [ 到最后一个 } 或 ]）
    3. json.loads 解析
    """
    content = content.strip()
    # 1. 提取 markdown 代码块
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    # 2. 定位 JSON 边界 — 找到最早的 '{' 或 '['，用对应的闭合符号
    brace_pos = content.find("{")
    bracket_pos = content.find("[")
    if brace_pos >= 0 and (bracket_pos < 0 or brace_pos < bracket_pos):
        start = brace_pos
        end = content.rfind("}") + 1
    elif bracket_pos >= 0:
        start = bracket_pos
        end = content.rfind("]") + 1
    else:
        start = -1
        end = 0
    if start >= 0 and end > start:
        content = content[start:end]
    return json.loads(content)
