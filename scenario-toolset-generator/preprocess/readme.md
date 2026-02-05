# Preprocess 预处理

## 数据文件

### merged_by_toolset.jsonl
按工具集合并后的对话数据
- 记录数: 23,183 (唯一工具集)
- 格式: `{"tools": {...}, "conversations": [{"dialogue": [...], "called_tools": [...]}]}`

### merged_with_labels.jsonl
添加标签和概要后的数据（含处理状态字段）
- 记录数: 23,183

### merged_with_labels_clean.jsonl
清理后的最终版本（移除临时字段，添加uuid）
- 记录数: 23,183
- 格式: `{"tools": {...}, "conversations": [...], "labels": [...], "summary": "...", "uid": "..."}`

### sample_small.json / test_5_results.json
测试用小样本数据

## 脚本文件

### merge_by_toolset.py
将原始数据按工具集合并，输入data/graphsyn.jsonl

### extract_labels.py
使用LLM提取每个工具集的标签(2-5个)和一句话概要

### retry_failed.py
重试提取失败的记录

### embed_labels.py
对标签进行嵌入，输出到embeddings/
