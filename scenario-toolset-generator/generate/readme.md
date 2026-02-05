# Generate 场景生成

## 数据文件

### clustered_toolsets.jsonl
按簇分组的工具集数据
- 记录数: 468 (467簇 + 1 other)
- 格式: `{"cluster_id": N, "size": N, "toolsets": [...]}`

### cluster_scenarios.jsonl
每个簇提取的场景列表
- 记录数: 467 (不含other)
- 格式: `{"cluster_id": N, "scenarios": [...], "toolsets": [...]}`
- 总场景数: ~4,700

### sample_complete_cluster.json
完整簇数据样本（调试用）

## 脚本文件

### group_by_cluster.py
将聚类结果映射回原始数据，按簇分组输出

### extract_scenarios.py
使用LLM对每个簇提取中小任务场景（1-10个）
