# Clustering 聚类结果

## 数据文件

### cluster_labels.npy
每条记录的簇标签（与embeddings行对齐）
- 形状: (23,180,)
- 值: 簇ID (0-466) 或 -1 (噪声点)

### embeddings_umap.npy
UMAP降维后的向量
- 形状: (23,180, 10)
- 参数: n_neighbors=15, min_dist=0.0

### clusters_index.json
按簇分组的行号索引
- 格式: `{"簇ID": [行号列表], ...}`

### clustering_stats.json
聚类统计信息
- 簇数量: 467
- 噪声点: 约2,800 (12%)

### clusters_preview.json
每个簇的样本预览（前10条）

## 脚本文件

### cluster_labels.py
UMAP降维 + HDBSCAN聚类
- min_cluster_size=10, min_samples=5
