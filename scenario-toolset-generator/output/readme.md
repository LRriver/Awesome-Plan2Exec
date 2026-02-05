# Output 文件说明

## 数据文件

### scenario_tools.jsonl
原始场景-工具集映射（未去重）
- 记录数: 4,701
- 格式: `{"scenario": "场景名", "tools": {"工具名": "描述", ...}, "tools_count": N}`

### scenario_tools_gte10.jsonl
工具数≥10的场景（已去重合并）
- 记录数: 4,329
- 工具数: 10-428, 平均49.9

### scenario_tools_lt10.jsonl  
工具数<10的场景（已去重合并）
- 记录数: 169
- 工具数: 1-9, 平均6.9

## 脚本文件

### match_scenario_tools.py
对每个场景判断簇内哪些工具可能用到，生成scenario_tools.jsonl

### merge_duplicate_scenarios.py
合并工具集重复的场景（结合LLM生成上层场景名），按工具数分文件
