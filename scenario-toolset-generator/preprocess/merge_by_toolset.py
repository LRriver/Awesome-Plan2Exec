#!/usr/bin/env python3
"""
将graphsyn.jsonl中工具集相同的对话合并
输出格式:
{
    "tools": {tool_name: description, ...},
    "conversations": [
        {
            "dialogue": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "called_tools": ["tool1", "tool2"]
        },
        ...
    ]
}
"""
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

def extract_tool_info(tool):
    """提取工具信息"""
    func = tool.get('function', tool)
    if isinstance(func, dict) and 'function' in func:
        func = func['function']
    name = func.get('name', '')
    desc = func.get('description', '')
    return name, desc

def extract_dialogue_and_tools(conversations):
    """从对话中提取user/assistant内容和召回的工具"""
    dialogue = []
    called_tools = set()
    
    for conv in conversations:
        role = conv.get('role', '')
        content = conv.get('content', '')
        
        # 提取召回的工具
        if 'tool_calls' in conv:
            for call in conv['tool_calls']:
                func = call.get('function', {})
                tool_name = func.get('name', '')
                if tool_name:
                    called_tools.add(tool_name)
        
        # 只保留user和assistant的对话内容
        if role in ['user', 'assistant'] and content:
            # 去掉<think>标签内容
            if '<think>' in content:
                think_end = content.find('</think>')
                if think_end != -1:
                    content = content[think_end + 8:].strip()
            if content:
                dialogue.append({'role': role, 'content': content})
    
    return dialogue, list(called_tools)

def main():
    # 输入
    input_file = BASE_DIR / 'data' / 'graphsyn.jsonl'
    output_file = SCRIPT_DIR / 'merged_by_toolset.jsonl'
    
    # 按工具集分组
    toolset_groups = defaultdict(lambda: {'tools': {}, 'conversations': []})
    
    print('读取数据...')
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # 提取工具集信息
            tools_dict = {}
            tool_names = []
            for tool in data.get('tools', []):
                name, desc = extract_tool_info(tool)
                if name:
                    tools_dict[name] = desc
                    tool_names.append(name)
            
            # 用排序后的工具名作为key
            toolset_key = tuple(sorted(tool_names))
            
            # 保存工具信息（只需保存一次）
            if not toolset_groups[toolset_key]['tools']:
                toolset_groups[toolset_key]['tools'] = tools_dict
            
            # 提取对话内容和召回的工具
            dialogue, called_tools = extract_dialogue_and_tools(data.get('conversations', []))
            
            if dialogue:
                toolset_groups[toolset_key]['conversations'].append({
                    'dialogue': dialogue,
                    'called_tools': called_tools
                })
            
            count += 1
            if count % 10000 == 0:
                print(f'  已处理 {count} 条...')
    
    print(f'总共处理 {count} 条数据')
    print(f'合并后得到 {len(toolset_groups)} 个工具集')
    
    # 保存结果
    print(f'保存到 {output_file}...')
    with open(output_file, 'w', encoding='utf-8') as f:
        for toolset_key, group in toolset_groups.items():
            f.write(json.dumps(group, ensure_ascii=False) + '\n')
    
    # 统计信息
    conv_counts = [len(g['conversations']) for g in toolset_groups.values()]
    tool_counts = [len(g['tools']) for g in toolset_groups.values()]
    
    print(f'\n统计信息:')
    print(f'  工具集数量: {len(toolset_groups)}')
    print(f'  每个工具集的对话数: min={min(conv_counts)}, max={max(conv_counts)}, avg={sum(conv_counts)/len(conv_counts):.1f}')
    print(f'  每个工具集的工具数: min={min(tool_counts)}, max={max(tool_counts)}, avg={sum(tool_counts)/len(tool_counts):.1f}')

if __name__ == '__main__':
    main()
