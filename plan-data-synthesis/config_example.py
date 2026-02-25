"""
Plan2Exec 数据合成流水线 — 集中配置文件
修改此文件即可适配不同的 LLM 服务和运行环境，无需改动业务代码。

使用方法：
1. 复制此文件为 config.py: cp config_example.py config.py
2. 修改 config.py 中的 LLM 配置
"""
from pathlib import Path

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_FILE = SCRIPT_DIR / ".." / "scenario-toolset-generator" / "output" / "scenario_tools_gte10.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ============ LLM 配置 ============
LLM_BASE_URL = "your-llm-api-url"   # 你的 LLM API 地址
LLM_MODEL = "your-model-name"       # 模型名
LLM_API_KEY = "your-api-key"        # API Key

# ============ 并发控制 ============
MAX_CONCURRENCY = 5          # Semaphore 最大并发数 (3-5)
REQUEST_DELAY = 0.5          # 请求间隔（秒）
MAX_RETRIES = 3              # 最大重试次数
RETRY_BACKOFF_BASE = 1       # 指数退避基数（秒），实际间隔: 1s, 2s, 4s

# ============ 采样参数 ============
PLAN_SAMPLE_K = 5            # 每个问题的采样次数
PLAN_TEMPERATURE = 0.8       # 规划采样温度
PLAN_TOP_P = 0.9             # 规划采样 Top-P

# ============ 评分权重 ============
EVAL_WEIGHTS = {
    "tool_accuracy": 0.3,
    "dependency_logic": 0.2,
    "completeness": 0.2,
    "efficiency": 0.15,
    "thought_quality": 0.15,
}

# ============ 偏好数据筛选 ============
MIN_SCORE_GAP = 2.0          # chosen 与 rejected 的最小分差

# ============ 输出文件路径 ============
QUESTIONS_FILE = OUTPUT_DIR / "questions.jsonl"
PLAN_SAMPLES_FILE = OUTPUT_DIR / "plan_samples.jsonl"
EVALUATED_PLANS_FILE = OUTPUT_DIR / "evaluated_plans.jsonl"
PREFERENCE_DATA_FILE = OUTPUT_DIR / "preference_data.jsonl"

# ============ 自动创建输出目录 ============
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
