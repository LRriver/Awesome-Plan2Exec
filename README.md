# Awesome-Plan2Exec
[English](README_en.md) | [中文](README.md)

## 项目简介

本项目的最终目标是**训练规划智能体（Planning Agent）**，使其能够根据工具集和用户问题，生成结构化、合理的任务规划。

为此，我们设计了一条完整的数据构建流水线，分为两个递进阶段：

1. **场景-工具集构建**（`scenario-toolset-generator/`）：从对话数据中自动挖掘"任务场景 → 工具集"映射关系
2. **偏好数据合成**（`plan-data-synthesis/`）：基于上游场景-工具集，合成用于偏好训练的规划数据



## 项目结构

```
Awesome-Plan2Exec/
├── scenario-toolset-generator/    # 阶段1：场景-工具集生成器
│   ├── data/                      # 原始数据
│   ├── preprocess/                # 阶段1.1：预处理 - 合并、标注、嵌入
│   ├── embeddings/                # 阶段1.2：向量存储
│   ├── clustering/                # 阶段1.3：聚类结果
│   ├── generate/                  # 阶段1.4：场景生成
│   └── output/                    # 阶段1.5：生成"场景 → 工具集"数据集
├── plan-data-synthesis/           # 阶段2：偏好数据合成
│   ├── config.py                  # 集中配置（LLM、并发、采样）
│   ├── utils.py                   # 共享工具（LLM调用、JSON解析）
│   ├── generate_questions.py      # 阶段2.1：多难度问题生成
│   ├── plan_sampling.py           # 阶段2.2：多路径计划采样
│   ├── evaluate_plans.py          # 阶段2.3：LLM-as-Judge评估
│   ├── build_preference.py        # 阶段2.4：偏好数据提取
│   ├── run_pipeline.py            # 入口脚本（串联全部4个阶段）
│   ├── test/                      # 测试（pytest + hypothesis属性测试）
│   └── output/                    # 阶段输出文件
├── images/                        # 图像资源
├── requirements.txt               # Python依赖项
├── README_en.md
└── README.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

---

## 阶段一：场景-工具集构建

> 目录：`scenario-toolset-generator/`

### 目标

从对话数据中自动构建"任务场景 → 工具集"的映射关系，用于工具推荐、Agent 任务规划和多工具协同调用。

### 核心流程

1. **工具集聚合**：将使用相同工具集的对话合并
2. **语义标注**：LLM 提取领域标签和任务概要
3. **语义聚类**：Embedding + UMAP + HDBSCAN 发现相似工具集群
4. **场景生成**：LLM 从每个簇中提取具体任务场景
5. **工具匹配**：LLM 判断场景与工具的相关性，筛选工具子集

![数据合成设计图](images/数据合成设计图.png)

### 技术选型

| 组件 | 选择 | 用途 |
|------|------|------|
| LLM | Qwen3-30B-A3B-Instruct-2507 | 语义理解、标签提取、场景生成 |
| Embedding | Qwen3-Embedding-4B | 标签向量化 |
| 降维 | UMAP | 保留语义结构 |
| 聚类 | HDBSCAN | 自动发现簇数 |

### 运行

```bash
cd scenario-toolset-generator


# 1. 下载原始数据
wget -P data -O graphsyn.jsonl https://www.modelscope.cn/datasets/nanbeige/ToolMind/resolve/master/graph_syn_datasets/graphsyn.jsonl

# 2. 按工具集合并
python preprocess/merge_by_toolset.py

# 3. LLM提取标签 (需要本地LLM服务)
python preprocess/extract_labels.py

# 4. 标签嵌入 (需要本地Embedding服务)
python preprocess/embed_labels.py

# 5. 聚类
python clustering/cluster_labels.py

# 6. 按簇分组
python generate/group_by_cluster.py

# 7. 提取场景
python generate/extract_scenarios.py

# 8. 场景-工具匹配
python output/match_scenario_tools.py

# 9. 去重合并
python output/merge_duplicate_scenarios.py
```

### 输出

- `output/scenario_tools_gte10.jsonl` — 工具数≥10的场景（4,329条）
- `output/scenario_tools_lt10.jsonl` — 工具数<10的场景（169条）

**输出示例：**
```json
{
  "scenario": "非洲旅游规划",
  "tools": {
    "get_weather": "获取目的地天气信息",
    "book_hotel": "预订酒店",
    "search_flights": "搜索航班"
  },
  "tools_count": 15
}
```

### 数据统计

| 阶段 | 数据量 |
|------|--------|
| 原始对话 | 163,180 |
| 唯一工具集 | 23,183 |
| 聚类簇数 | 467 |
| 生成场景 | 4,701 |
| 最终输出(≥10工具) | 4,329 |

---

## 阶段二：偏好数据合成

> 目录：`plan-data-synthesis/`

### 目标

基于上游场景-工具集数据，经过"问题生成 → 规划采样 → LLM-as-Judge 评分 → 偏好数据提取"四个阶段，产出偏好训练数据，用于后续规划智能体的对齐训练。

### 前置条件

- 已完成阶段一，生成 `scenario-toolset-generator/output/scenario_tools_gte10.jsonl`
- 可用的 LLM API 服务（OpenAI 格式）

### 配置

修改 `plan-data-synthesis/config.py` 中的 LLM 配置：

```python
LLM_BASE_URL = "your-llm-api-url"  # 你的 LLM API 地址
LLM_MODEL = "your-model-name"      # 模型名
LLM_API_KEY = "your-api-key"       # API Key
```

### 运行

```bash
cd plan-data-synthesis

# 运行完整流水线
python run_pipeline.py

# 从指定阶段开始（断点续传）
python run_pipeline.py --start-stage 2

# 或单独运行某个阶段
python generate_questions.py    # 阶段1：问题生成
python plan_sampling.py         # 阶段2：规划采样
python evaluate_plans.py        # 阶段3：自动评分
python build_preference.py      # 阶段4：偏好数据提取
```

### 输出

| 文件 | 说明 |
|------|------|
| `output/questions.jsonl` | ~50条多难度用户问题 |
| `output/plan_samples.jsonl` | 每条问题5次采样的规划结果 |
| `output/evaluated_plans.jsonl` | 五维度评分结果 |
| `output/preference_data.jsonl` | 最终偏好训练数据 |

### 测试

```bash
cd plan-data-synthesis
python -m pytest test/ -v
```

---

## License

Apache-2.0
