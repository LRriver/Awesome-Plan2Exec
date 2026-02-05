#!/usr/bin/env python3
"""
将聚类结果映射回原始数据，按簇分组输出
"""
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

CLUSTER_LABELS_FILE = BASE_DIR / "clustering" / "cluster_labels.npy"
RECORDS_FILE = BASE_DIR / "embeddings" / "records.jsonl"
ORIGINAL_FILE = BASE_DIR / "preprocess" / "merged_with_labels_clean.jsonl"
OUTPUT_DIR = BASE_DIR / "generate"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载聚类标签
    print("加载聚类标签...")
    cluster_labels = np.load(CLUSTER_LABELS_FILE)
    print(f"  共 {len(cluster_labels)} 条")
    
    # 2. 加载records (获取uid)
    print("加载records...")
    with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    print(f"  共 {len(records)} 条")
    
    # 3. 加载原始数据，建立uid索引
    print("加载原始数据...")
    uid_to_data = {}
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid_to_data[data['uid']] = data
    print(f"  共 {len(uid_to_data)} 条")
    
    # 4. 按簇分组
    print("\n按簇分组...")
    clusters = {}  # cluster_id -> list of toolset data
    noise_data = []  # 噪声点 (cluster_id = -1)
    
    for row_idx, cluster_id in enumerate(cluster_labels):
        cluster_id = int(cluster_id)
        uid = records[row_idx]['uid']
        toolset_data = uid_to_data.get(uid)
        
        if toolset_data is None:
            print(f"  警告: uid {uid} 未找到原始数据")
            continue
        
        if cluster_id == -1:
            noise_data.append(toolset_data)
        else:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(toolset_data)
    
    # 5. 统计
    print(f"\n统计:")
    print(f"  有效簇数: {len(clusters)}")
    print(f"  噪声点数: {len(noise_data)}")
    total_in_clusters = sum(len(v) for v in clusters.values())
    print(f"  簇内数据: {total_in_clusters}")
    
    # 6. 输出为jsonl，每行一个簇
    output_file = OUTPUT_DIR / "clustered_toolsets.jsonl"
    print(f"\n写入 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 先写有效簇
        for cluster_id in sorted(clusters.keys()):
            toolsets = clusters[cluster_id]
            record = {
                "cluster_id": cluster_id,
                "size": len(toolsets),
                "toolsets": toolsets
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # 最后写噪声点
        other_record = {
            "cluster_id": "other",
            "size": len(noise_data),
            "toolsets": noise_data
        }
        f.write(json.dumps(other_record, ensure_ascii=False) + '\n')
    
    print(f"完成! 共 {len(clusters) + 1} 行 (467簇 + 1 other)")


if __name__ == "__main__":
    main()
