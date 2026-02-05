#!/usr/bin/env python3
"""
使用UMAP降维 + HDBSCAN对labels嵌入向量进行聚类
输出保持行号映射，方便追溯原始数据
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import numpy as np
from pathlib import Path
import umap
import hdbscan
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

EMBEDDINGS_FILE = BASE_DIR / "embeddings" / "embeddings.npy"
RECORDS_FILE = BASE_DIR / "embeddings" / "records.jsonl"
OUTPUT_DIR = BASE_DIR / "clustering"

# UMAP参数
UMAP_N_COMPONENTS = 10  # 降维目标维度
UMAP_N_NEIGHBORS = 15   # 邻居数
UMAP_MIN_DIST = 0.0     # 最小距离(聚类用0更好)

# HDBSCAN参数
MIN_CLUSTER_SIZE = 10  # 最小簇大小
MIN_SAMPLES = 5        # 核心点最小邻居数


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  shape: {embeddings.shape}")
    
    print("加载records...")
    with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    print(f"  数量: {len(records)}")
    
    # UMAP降维
    print(f"\nUMAP降维 ({embeddings.shape[1]}D -> {UMAP_N_COMPONENTS}D)...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42,
        n_jobs=-1
    )
    embeddings_reduced = reducer.fit_transform(embeddings)
    print(f"  降维后shape: {embeddings_reduced.shape}")
    
    # 保存降维后的向量
    np.save(OUTPUT_DIR / "embeddings_umap.npy", embeddings_reduced)
    
    # HDBSCAN聚类
    print(f"\n开始HDBSCAN聚类 (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embeddings_reduced)
    
    # 统计结果
    label_counts = Counter(labels)
    n_clusters = len([l for l in label_counts if l >= 0])
    n_noise = label_counts.get(-1, 0)
    
    print(f"\n聚类结果:")
    print(f"  簇数量: {n_clusters}")
    print(f"  噪声点数量: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"  有效聚类点: {len(labels) - n_noise} ({(len(labels)-n_noise)/len(labels)*100:.1f}%)")
    
    # 按簇大小排序显示前20个簇
    print(f"\n前20个最大的簇:")
    sorted_clusters = sorted([(l, c) for l, c in label_counts.items() if l >= 0], 
                            key=lambda x: -x[1])[:20]
    for cluster_id, count in sorted_clusters:
        print(f"  簇{cluster_id}: {count}条")
    
    # 保存结果
    # 1. 保存每行的cluster标签 (行号对齐)
    labels_file = OUTPUT_DIR / "cluster_labels.npy"
    np.save(labels_file, labels)
    print(f"\n保存cluster_labels.npy: {len(labels)}行")
    
    # 2. 保存按簇分组的行号索引
    clusters_index = {}
    for row_idx, cluster_id in enumerate(labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters_index:
            clusters_index[cluster_id] = []
        clusters_index[cluster_id].append(row_idx)
    
    clusters_file = OUTPUT_DIR / "clusters_index.json"
    with open(clusters_file, 'w', encoding='utf-8') as f:
        json.dump(clusters_index, f, ensure_ascii=False, indent=2)
    print(f"保存clusters_index.json: {len(clusters_index)}个簇")
    
    # 3. 保存聚类统计信息
    stats = {
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_sizes": {str(k): v for k, v in sorted(label_counts.items(), key=lambda x: -x[1])}
    }
    stats_file = OUTPUT_DIR / "clustering_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"保存clustering_stats.json")
    
    # 4. 保存每个簇的样本预览 (方便查看聚类效果)
    preview = {}
    for cluster_id, row_indices in clusters_index.items():
        if int(cluster_id) == -1:
            # 噪声只取前5个
            sample_indices = row_indices[:5]
        else:
            # 每个簇取前10个样本
            sample_indices = row_indices[:10]
        
        samples = []
        for idx in sample_indices:
            rec = records[idx]
            samples.append({
                "row_idx": idx,
                "uid": rec["uid"],
                "labels": rec["labels"],
                "summary": rec["summary"]
            })
        preview[cluster_id] = {
            "size": len(row_indices),
            "samples": samples
        }
    
    preview_file = OUTPUT_DIR / "clusters_preview.json"
    with open(preview_file, 'w', encoding='utf-8') as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
    print(f"保存clusters_preview.json")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
