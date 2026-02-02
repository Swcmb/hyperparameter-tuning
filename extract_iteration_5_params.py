#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取第5次迭代的完整参数配置
"""

import json
from pathlib import Path

def extract_iteration_5_params():
    """提取第5次迭代的完整参数"""
    log_file = Path("logs/autodl_20260201_171815_20260201_171815/autodl_20260201_171815_structured.jsonl")
    
    iteration_count = 0
    current_params = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                tag = data.get('tag', '')
                message = data.get('message', '')
                
                # 检测迭代开始
                if tag == 'ITERATION' and message.isdigit():
                    iteration_count = int(message)
                    
                # 如果是第5次迭代的参数建议
                if iteration_count == 5 and tag == 'SUGGESTED_PARAMS' and 'structured_data' in data:
                    current_params = data['structured_data']
                    break
                    
            except json.JSONDecodeError:
                continue
    
    if current_params:
        print("=== 第5次迭代完整参数配置 ===\n")
        
        # 按类别组织参数
        network_params = {}
        optimization_params = {}
        moco_params = {}
        attention_params = {}
        other_params = {}
        
        for key, value in current_params.items():
            if key in ['dimensions', 'hidden1', 'hidden2', 'decoder1']:
                network_params[key] = value
            elif key in ['lr', 'dropout', 'weight_decay', 'batch']:
                optimization_params[key] = value
            elif key.startswith('moco_'):
                moco_params[key] = value
            elif key in ['gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy']:
                attention_params[key] = value
            elif key in ['alpha', 'beta', 'gamma']:
                other_params[key] = value
            else:
                other_params[key] = value
        
        print("## 网络架构参数")
        for key, value in network_params.items():
            print(f"  {key}: {value}")
        
        print("\n## 优化参数")
        for key, value in optimization_params.items():
            print(f"  {key}: {value}")
        
        print("\n## MoCo (Momentum Contrast) 参数")
        for key, value in moco_params.items():
            print(f"  {key}: {value}")
        
        print("\n## 注意力机制参数")
        for key, value in attention_params.items():
            print(f"  {key}: {value}")
        
        print("\n## 其他参数")
        for key, value in other_params.items():
            print(f"  {key}: {value}")
        
        print("\n## 完整参数字典格式")
        print("```python")
        print("iteration_5_params = {")
        for key, value in sorted(current_params.items()):
            if isinstance(value, str):
                print(f"    '{key}': '{value}',")
            else:
                print(f"    '{key}': {value},")
        print("}")
        print("```")
        
        # 保存到文件
        with open('iteration_5_complete_params.json', 'w', encoding='utf-8') as f:
            json.dump(current_params, f, indent=2, ensure_ascii=False)
        
        print(f"\n参数已保存到: iteration_5_complete_params.json")
        
    else:
        print("未找到第5次迭代的参数配置")

if __name__ == "__main__":
    extract_iteration_5_params()