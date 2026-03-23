"""
Plan2Exec 数据合成流水线 — 公共工具函数
包含 LLM 异步调用（含重试）和 JSON 解析容错。
"""
import asyncio
import json

from openai import AsyncOpenAI

import config


_CLIENT: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """懒加载 OpenAI Async 客户端。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=120.0,
            # Avoid content-encoding compatibility issues across gateways.
            default_headers={"Accept-Encoding": "identity"},
        )
    return _CLIENT


def _extract_content(resp) -> str:
    """提取 chat.completions 响应中的文本内容。"""
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    return ""


async def call_llm(
    messages: list[dict],
    temperature: float = 0.3,
    top_p: float = 1.0,
) -> str:
    """调用 LLM，返回 content 字符串。

    使用 OpenAI Async SDK 调用 chat.completions，
    失败时采用指数退避策略重试最多 MAX_RETRIES 次。
    """
    client = _get_openai_client()
    request_kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if getattr(config, "LLM_REASONING_SPLIT", False):
        request_kwargs["extra_body"] = {"reasoning_split": True}

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            result = await client.chat.completions.create(**request_kwargs)
            return _extract_content(result)
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
