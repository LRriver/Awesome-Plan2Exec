#!/usr/bin/env python3
"""
对merged_with_labels_clean.jsonl中的labels进行嵌入
1. 先给每条数据添加uuid
2. 将labels拼接后嵌入
3. 存储为embeddings.npy + records.jsonl（行对齐）
"""
import json
import uuid
import numpy as np
from pathlib import Path
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

# ============ 配置 ============
EMBED_BASE_URL = "http://127.0.0.1:6002/v1"
EMBED_MODEL = "Qwen3-Embedding-4B"
EMBED_DIMENSION = 1024
BATCH_SIZE = 100  # 每批嵌入的数量

INPUT_FILE = BASE_DIR / "preprocess" / "merged_with_labels_clean.jsonl"
OUTPUT_DIR = BASE_DIR / "embeddings"


def add_uuid_to_data(input_file: str) -> list:
    """给每条数据添加uuid，并返回数据列表"""
    all_data = []
    updated_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 添加uuid
                if 'uid' not in data:
                    data['uid'] = str(uuid.uuid4())
                all_data.append(data)
                updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 写回文件
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"已为 {len(all_data)} 条数据添加uuid")
    return all_data


def prepare_embed_data(all_data: list) -> tuple:
    """准备需要嵌入的数据，过滤掉失败的"""
    to_embed = []
    
    for data in all_data:
        labels = data.get('labels', [])
        # 跳过失败的（labels为空或为["Other"]）
        if not labels or labels == ["Other"]:
            continue
        
        # 拼接labels为一个字符串
        labels_text = ", ".join(labels)
        
        to_embed.append({
            'uid': data['uid'],
            'labels_text': labels_text,
            'labels': labels,
            'summary': data.get('summary', '')
        })
    
    print(f"需要嵌入的数据: {len(to_embed)} 条")
    return to_embed


def embed_batch(client: OpenAI, texts: list) -> list:
    """批量嵌入"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSION
    )
    # 按index排序确保顺序正确
    sorted_data = sorted(resp.data, key=lambda x: x.index)
    return [d.embedding for d in sorted_data]


def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 添加uuid
    print("Step 1: 添加uuid...")
    all_data = add_uuid_to_data(INPUT_FILE)
    
    # Step 2: 准备嵌入数据
    print("\nStep 2: 准备嵌入数据...")
    to_embed = prepare_embed_data(all_data)
    
    # Step 3: 批量嵌入
    print(f"\nStep 3: 开始嵌入 (batch_size={BATCH_SIZE})...")
    client = OpenAI(base_url=EMBED_BASE_URL, api_key="EMPTY")
    
    all_embeddings = []
    records = []
    
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i+BATCH_SIZE]
        texts = [item['labels_text'] for item in batch]
        
        # 嵌入
        embeddings = embed_batch(client, texts)
        
        # 立即保存到列表（保证顺序）
        for j, item in enumerate(batch):
            all_embeddings.append(embeddings[j])
            records.append({
                'uid': item['uid'],
                'labels_text': item['labels_text'],
                'labels': item['labels'],
                'summary': item['summary']
            })
        
        processed = min(i + BATCH_SIZE, len(to_embed))
        print(f"  已处理: {processed}/{len(to_embed)}")
    
    # Step 4: 保存
    print("\nStep 4: 保存结果...")
    
    # 保存embeddings
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings_array)
    print(f"  embeddings.npy: shape={embeddings_array.shape}")
    
    # 保存records
    records_file = OUTPUT_DIR / "records.jsonl"
    with open(records_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  records.jsonl: {len(records)} 条")
    
    # 验证行对齐
    print(f"\n验证: embeddings行数={len(all_embeddings)}, records行数={len(records)}")
    assert len(all_embeddings) == len(records), "行数不匹配！"
    print("✓ 行对齐验证通过")


if __name__ == "__main__":
    main()
