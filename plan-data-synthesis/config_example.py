"""
Plan2Exec 数据合成流水线配置文件
1. 复制此文件为 config.py: cp config_example.py config.py
2. 修改 config.py 中的 LLM 配置
"""
from pathlib import Path

# 路径配置  
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_FILE = SCRIPT_DIR / ".." / "scenario-toolset-generator" / "output" / "scenario_tools_gte10.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "output"

#   LLM 配置  
LLM_BASE_URL = "your-llm-api-url"   # 你的 LLM API 地址
LLM_MODEL = "your-model-name"       # 模型名
LLM_API_KEY = "your-api-key"        # API Key
LLM_REASONING_SPLIT = True           # 若平台支持，将推理内容与正文拆分

#   并发控制  
MAX_CONCURRENCY = 5          # Semaphore 最大并发数 (3-5)
REQUEST_DELAY = 0.5          # 请求间隔（秒）
MAX_RETRIES = 3              # 最大重试次数
RETRY_BACKOFF_BASE = 1       # 指数退避基数（秒），实际间隔: 1s, 2s, 4s

#   采样参数  
PLAN_SAMPLE_K = 8            # 每个问题的采样次数
PLAN_TEMPERATURE = 1.0       # 规划采样温度
PLAN_TOP_P = 0.95            # 规划采样 Top-P

#   评分参数  
EVAL_TEMPERATURE = 1.0       # 评分采样温度（提高区分度）
EVAL_SAMPLE_N = 3            # 每个计划评分采样次数（多次取中位数）

#   评分权重（10维度）  
EVAL_WEIGHTS = {
    # 工具层 (0.30)
    "tool_existence":       0.15,   # 工具是否存在于工具集中
    "tool_semantic_match":  0.15,   # 工具语义是否匹配步骤任务
    # 逻辑层 (0.25)
    "dependency_logic":     0.12,   # 依赖关系是否正确
    "no_circular_dep":      0.05,   # 无循环依赖
    "data_flow_integrity":  0.08,   # 数据流完整性
    # 完整性层 (0.20)
    "completeness":         0.12,   # 是否覆盖所有显性子任务
    "implicit_needs":       0.08,   # 是否识别隐性需求
    # 效率层 (0.10)
    "efficiency":           0.10,   # 无冗余步骤，粒度合理
    # 思维层 (0.15)
    "thought_depth":        0.08,   # 推理深度
    "thought_consistency":  0.07,   # thought 与实际执行的一致性
}

#   流式写入  
FLUSH_THRESHOLD = 20         # 累积多少条结果后写入磁盘（减少 IO 次数）

#   偏好数据筛选  
MIN_SCORE_GAP = 0.5          # chosen 与 rejected 的最小分差

#   问题生成配置  
SCENARIO_LIMIT = 0               # 加载场景数上限，0 表示不限制（加载全部）
DIFFICULTY_LEVELS = [
    "chat",                  # 闲聊/无工具
    "simple",                # 单工具
    "parallel",              # 并行多工具
    "complex_dependency",    # 复杂依赖链
    "long_chain",            # 长链条（>=4步工具调用）
    "ambiguous",             # 模糊/歧义问题
    "adversarial",           # 对抗性扰动
    "safety",                # 安全/有害请求拒绝
]

#   输出文件路径  
QUESTIONS_FILE = OUTPUT_DIR / "questions.jsonl"
PLAN_SAMPLES_FILE = OUTPUT_DIR / "plan_samples.jsonl"
EVALUATED_PLANS_FILE = OUTPUT_DIR / "evaluated_plans.jsonl"
PREFERENCE_DATA_FILE = OUTPUT_DIR / "preference_data.jsonl"

#   自动创建输出目录  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
