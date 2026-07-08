"""
Plan2Exec 数据合成流水线 — 公共工具函数
包含 LLM 异步调用（含重试）和 JSON 解析容错。
"""
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

import config


_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}
_PROFILE_SEMAPHORES: dict[tuple[str, str, int], asyncio.Semaphore] = {}


def _get_openai_client(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    """懒加载 OpenAI Async 客户端。"""
    base_url = base_url or config.LLM_BASE_URL
    api_key = api_key or config.LLM_API_KEY
    key = (base_url, api_key)
    if key not in _CLIENTS:
        _CLIENTS[key] = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
            # Avoid content-encoding compatibility issues across gateways.
            default_headers={"Accept-Encoding": "identity"},
        )
    return _CLIENTS[key]


def resolve_model(profile: str | None = None, model: str | None = None) -> str:
    """Resolve the model for a request from explicit override, profile, or default."""
    if model:
        return model
    if profile:
        profiles = getattr(config, "LLM_PROFILES", {})
        selected = profiles.get(profile)
        if selected and selected.get("model"):
            return selected["model"]
    return config.LLM_MODEL


def resolve_profile_connection(profile: str | None = None) -> tuple[str, str]:
    """Resolve base_url/api_key for an optional profile."""
    if profile:
        selected = getattr(config, "LLM_PROFILES", {}).get(profile, {})
        return (
            selected.get("base_url", config.LLM_BASE_URL),
            selected.get("api_key", config.LLM_API_KEY),
        )
    return config.LLM_BASE_URL, config.LLM_API_KEY


def resolve_profile_max_concurrency(profile: str | None = None) -> int:
    """Resolve per-profile concurrency, falling back to global config."""
    if profile:
        selected = getattr(config, "LLM_PROFILES", {}).get(profile, {})
        max_concurrency = selected.get("max_concurrency")
        if max_concurrency:
            return int(max_concurrency)
    return int(getattr(config, "MAX_CONCURRENCY", 5))


def _get_profile_semaphore(profile: str | None = None, model: str | None = None) -> asyncio.Semaphore:
    """Get a per-model semaphore so expensive models can be rate-limited."""
    limit = resolve_profile_max_concurrency(profile)
    key = (profile or "__default__", resolve_model(profile=profile, model=model), limit)
    if key not in _PROFILE_SEMAPHORES:
        _PROFILE_SEMAPHORES[key] = asyncio.Semaphore(limit)
    return _PROFILE_SEMAPHORES[key]


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


def _sanitize_control_chars_in_json_strings(content: str) -> str:
    """Replace raw control chars inside JSON strings with spaces.

    Some LLM gateways occasionally return unescaped newlines inside string
    values. JSON permits whitespace between tokens, but not raw control
    characters inside quoted strings.
    """
    result = []
    in_string = False
    escaped = False
    for ch in content:
        if in_string and ord(ch) < 32:
            result.append(" ")
            escaped = False
            continue
        result.append(ch)
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
    return "".join(result)


async def call_llm(
    messages: list[dict],
    temperature: float = 0.3,
    top_p: float = 1.0,
    model: str | None = None,
    profile: str | None = None,
) -> str:
    """调用 LLM，返回 content 字符串。

    使用 OpenAI Async SDK 调用 chat.completions，
    失败时采用指数退避策略重试最多 MAX_RETRIES 次。
    """
    base_url, api_key = resolve_profile_connection(profile)
    client = _get_openai_client(base_url=base_url, api_key=api_key)
    request_kwargs = {
        "model": resolve_model(profile=profile, model=model),
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if getattr(config, "LLM_REASONING_SPLIT", False):
        request_kwargs["extra_body"] = {"reasoning_split": True}

    semaphore = _get_profile_semaphore(profile=profile, model=model)
    async with semaphore:
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
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        if "Invalid control character" not in str(exc):
            raise
        return json.loads(_sanitize_control_chars_in_json_strings(content))


def ensure_trailing_newline(path) -> None:
    """Ensure appending to an existing JSONL file starts on a fresh line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, 2)
        last_byte = f.read(1)
    if last_byte != b"\n":
        with open(path, "ab") as f:
            f.write(b"\n")
