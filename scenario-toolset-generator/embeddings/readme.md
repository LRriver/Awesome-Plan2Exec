# Embeddings 嵌入向量

## 数据文件

### embeddings.npy
标签嵌入向量矩阵
- 形状: (23,180, 1024)
- 模型: Qwen3-Embedding-4B
- 输入: 拼接后的labels字符串

### records.jsonl
与embeddings.npy行对齐的元数据
- 记录数: 23,180
- 格式: `{"uid": "...", "labels": [...], "summary": "..."}`
- 第i行对应embeddings.npy的第i行向量

## 说明

跳过了labels为["Other"]的4条失败记录，因此比preprocess少3条。
