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
LLM_MODEL = "gpt-5.4"               # 默认模型名
LLM_API_KEY = "your-api-key"        # API Key
LLM_REASONING_SPLIT = True           # 若平台支持，将推理内容与正文拆分

#   模型路由配置
#   建议：
#   - cheap: 低成本模型，适合大量问题生成、受控负样本生成和重试
#   - strong: 性价比较高的强模型，适合 chosen plan 和主要 judge
#   - judge: 默认批量评分模型，建议用性价比高的强模型；gpt-5.5/Opus 只用于抽样审计
#   - premium: 仅用于少量高难度审计，不进入默认批量路径
LLM_PROFILES = {
    "cheap": {
        "model": "DeepSeek-V4-Flash",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "max_concurrency": 50,
    },
    "strong": {
        "model": "gpt-5.4",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "max_concurrency": 10,
    },
    "judge": {
        "model": "gpt-5.4",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "max_concurrency": 10,
    },
    "premium": {
        "model": "Claude Opus 4.7",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "max_concurrency": 3,
    },
}

QUESTION_MODEL_PROFILE = "cheap"
PLAN_MODEL_PROFILE = "strong"
NEGATIVE_PLAN_MODEL_PROFILE = "cheap"
EVAL_MODEL_PROFILE = "judge"

#   并发控制  
MAX_CONCURRENCY = 5          # Semaphore 最大并发数 (3-5)
REQUEST_DELAY = 0.5          # 请求间隔（秒）
MAX_RETRIES = 3              # 最大重试次数
RETRY_BACKOFF_BASE = 1       # 指数退避基数（秒），实际间隔: 1s, 2s, 4s
JSON_PARSE_RETRIES = 1       # LLM 返回不可解析 JSON 时额外重试次数

#   采样参数  
PLAN_SAMPLE_K = 8            # 每个问题的采样次数
PLAN_TEMPERATURE = 1.0       # 规划采样温度
PLAN_TOP_P = 0.95            # 规划采样 Top-P

#   分级预算配置
#   低难度问题用于校准模型“不要过度规划”，数量不必和高难度相同。
DIFFICULTY_KEEP_RATES = {
    "chat": 0.25,
    "simple": 0.35,
    "parallel": 1.0,
    "complex_dependency": 1.0,
    "long_chain": 1.0,
    "ambiguous": 1.0,
    "adversarial": 1.0,
    "safety": 1.0,
}
PLAN_SAMPLE_K_BY_DIFFICULTY = {
    "chat": 1,
    "simple": 1,
    "parallel": 1,
    "complex_dependency": 1,
    "long_chain": 2,
    "ambiguous": 1,
    "adversarial": 1,
    "safety": 1,
}

#   受控负样本配置
#   这些负样本用于拉开分布；默认每题额外生成 3 个不同错误类型。
NEGATIVE_PLAN_TYPES = [
    "wrong_tool",
    "missing_dependency",
    "ignored_ambiguity",
]
NEGATIVE_PLAN_TYPES_BY_DIFFICULTY = {
    "chat": ["wrong_tool"],
    "simple": ["wrong_tool"],
    "parallel": ["wrong_tool", "missing_dependency", "over_serialized"],
    "complex_dependency": ["wrong_tool", "missing_dependency", "data_flow_broken"],
    "long_chain": ["wrong_tool", "missing_dependency", "over_serialized", "redundant_steps"],
    "ambiguous": ["wrong_tool", "ignored_ambiguity"],
    "adversarial": ["wrong_tool", "missing_dependency"],
    "safety": ["unsafe_compliance", "wrong_tool"],
}
NEGATIVE_PLAN_TEMPERATURE = 0.8
NEGATIVE_PLAN_TOP_P = 0.95

#   评分参数
EVAL_TEMPERATURE = 1.0       # 评分采样温度（提高区分度）
EVAL_SAMPLE_N = 1            # 默认每个计划评分 1 次；多次评分建议只用于抽样审计
EVAL_SAMPLE_N_BY_DIFFICULTY = {
    "chat": 1,
    "simple": 1,
    "parallel": 1,
    "complex_dependency": 1,
    "long_chain": 1,
    "ambiguous": 1,
    "adversarial": 1,
    "safety": 1,
}
EVAL_JUDGE_POLICY = "selective"  # all=全部 LLM judge; selective=规则预评估后只 judge 高价值候选
EVAL_MAX_LLM_NEGATIVES_BY_DIFFICULTY = {
    "chat": 0,
    "simple": 1,
    "parallel": 1,
    "complex_dependency": 1,
    "long_chain": 2,
    "ambiguous": 2,
    "adversarial": 2,
    "safety": 2,
}

#   运行模式
#   production: 固定策略 checkpoint，用于全量抽取；不进入 rubric 迭代流程。
#   iteration: pilot/audit 模式，用于调 rubric、预算和模型路由。
SYNTHESIS_MODE = "production"
STRATEGY_CHECKPOINT = "controlled_v1"
RESUME = False
ENABLE_RUBRIC_AUDIT = False
RUBRIC_AUDIT_SAMPLE_RATE = 0.0
RUN_MODE_OVERRIDES = {
    "production": {
        "EVAL_JUDGE_POLICY": "selective",
        "EVAL_SAMPLE_N": 1,
        "ENABLE_RUBRIC_AUDIT": False,
        "RUBRIC_AUDIT_SAMPLE_RATE": 0.0,
    },
    "iteration": {
        "EVAL_JUDGE_POLICY": "selective",
        "EVAL_SAMPLE_N": 1,
        "ENABLE_RUBRIC_AUDIT": True,
        "RUBRIC_AUDIT_SAMPLE_RATE": 0.05,
    },
}

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
QUALITY_BUCKETS = {
    "strong": 8.5,
    "medium": 6.0,
    "weak": 3.0,
}

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
QUERY_STYLES = [
    "realistic",             # 真实业务表达
    "casual",                # 口语化
    "noisy",                 # 有轻微噪声/冗余信息
    "humorous",              # 有生活化幽默，但任务仍然有效
    "formal",                # 正式工单/业务请求风格
]

#   输出文件路径  
QUESTIONS_FILE = OUTPUT_DIR / "questions.jsonl"
PLAN_SAMPLES_FILE = OUTPUT_DIR / "plan_samples.jsonl"
EVALUATED_PLANS_FILE = OUTPUT_DIR / "evaluated_plans.jsonl"
PREFERENCE_DATA_FILE = OUTPUT_DIR / "preference_data.jsonl"
RANKED_CANDIDATES_FILE = OUTPUT_DIR / "ranked_candidates.jsonl"
RUN_MANIFEST_FILE = OUTPUT_DIR / "run_manifest.json"
RUBRIC_AUDIT_REPORT_FILE = OUTPUT_DIR / "rubric_audit_report.json"

#   自动创建输出目录  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
