# Awesome-Plan2Exec
[English](README_en.md) | [中文](README.md)
场景-工具集数据构建与Agent规划执行研究

## 项目简介

本项目旨在从对话数据中自动构建"任务场景 → 工具集"的映射关系，用于：
- 工具推荐
- Agent任务规划
- 多工具协同调用

**输出示例：**
```json
{
  "scenario": "非洲旅游规划",
  "tools": {
    "get_weather": "获取目的地天气信息",
    "book_hotel": "预订酒店",
    "search_flights": "搜索航班",
    ...
  },
  "tools_count": 15
}
```

---

## 系统架构

![数据合成设计图](images/数据合成设计图.png)

### 核心流程

1. **工具集聚合**：将使用相同工具集的对话合并
2. **语义标注**：LLM提取领域标签和任务概要
3. **语义聚类**：Embedding + UMAP + HDBSCAN发现相似工具集群
4. **场景生成**：LLM从每个簇中提取具体任务场景
5. **工具匹配**：LLM判断场景与工具的相关性，筛选工具子集

### 技术选型

| 组件 | 选择 | 用途 |
|------|------|------|
| LLM | Qwen3-30B | 语义理解、标签提取、场景生成 |
| Embedding | Qwen3-Embedding-4B | 标签向量化 |
| 降维 | UMAP | 保留语义结构 |
| 聚类 | HDBSCAN | 自动发现簇数 |

---

## 项目结构

```
Awesome-Plan2Exec/
├── scenario-toolset-generator/    # 场景-工具集生成器
│   ├── data/                      # 原始数据
│   ├── preprocess/                # 预处理：合并、标注、嵌入
│   ├── embeddings/                # 向量存储
│   ├── clustering/                # 聚类结果
│   ├── generate/                  # 场景生成
│   └── output/                    # 最终输出
├── images/                        # 图片资源
└── README.md
```

---

## 快速开始

### 环境依赖

```bash
pip install openai pydantic numpy umap-learn hdbscan
```

### 运行流程

```bash
cd scenario-toolset-generator

# 1. 下载原始数据
cd data
wget https://www.modelscope.cn/datasets/nanbeige/ToolMind/resolve/master/graph_syn_datasets/graphsyn.jsonl
cd ..

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

### 输出文件

- `output/scenario_tools_gte10.jsonl` - 工具数≥10的场景 (4,329条)
- `output/scenario_tools_lt10.jsonl` - 工具数<10的场景 (169条)

---

## 数据统计

| 阶段 | 数据量 |
|------|--------|
| 原始对话 | 163,180 |
| 唯一工具集 | 23,183 |
| 聚类簇数 | 467 |
| 生成场景 | 4,701 |
| 最终输出(≥10工具) | 4,329 |

---

## License

Apache-2.0