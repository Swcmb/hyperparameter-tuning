#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正命令行参数类型问题
"""

import json
import math
from pathlib import Path

def fix_command_line():
    """生成修正后的命令行参数"""
    
    # 读取参数文件
    params_file = Path("iteration_5_complete_params.json")
    
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    print("=== 修正后的第5次迭代最佳参数命令行 ===\n")
    
    # 参数类型修正
    def fix_param_value(param_name, param_value):
        """根据参数名修正参数值的类型"""
        
        # 需要整数的参数
        int_params = [
            'dimensions', 'hidden1', 'hidden2', 'decoder1', 
            'batch', 'gat_heads', 'gt_heads', 'fusion_heads', 'moco_K'
        ]
        
        # 需要浮点数的参数
        float_params = [
            'lr', 'dropout', 'weight_decay', 'alpha', 'beta', 'gamma',
            'moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2'
        ]
        
        # 布尔值参数
        bool_params = ['enable_view_0']
        
        # 字符串参数
        str_params = ['fusion_strategy', 'feature_type', 'moco_type']
        
        if param_name in int_params:
            return str(int(float(param_value)))
        elif param_name in float_params:
            return str(float(param_value))
        elif param_name in bool_params:
            # 转换布尔值
            if str(param_value).lower() in ['true', '1', 'yes']:
                return 'true'
            else:
                return 'false'
        elif param_name in str_params:
            return str(param_value)
        else:
            return str(param_value)
    
    # 生成修正后的参数
    fixed_params = {}
    for param_name, param_value in params.items():
        fixed_params[param_name] = fix_param_value(param_name, param_value)
    
    # 参数映射
    param_mapping = {
        'dimensions': '--dimensions',
        'hidden1': '--hidden1', 
        'hidden2': '--hidden2',
        'decoder1': '--decoder1',
        'lr': '--lr',
        'dropout': '--dropout',
        'weight_decay': '--weight_decay',
        'batch': '--batch',
        'moco_momentum': '--moco_momentum',
        'moco_t': '--moco_t',
        'moco_tau1': '--moco_tau1',
        'moco_tau2': '--moco_tau2',
        'moco_K': '--moco_K',
        'moco_type': '--moco_type',
        'gat_heads': '--gat_heads',
        'gt_heads': '--gt_heads', 
        'fusion_heads': '--fusion_heads',
        'fusion_strategy': '--fusion_strategy',
        'alpha': '--alpha',
        'beta': '--beta',
        'gamma': '--gamma',
        'feature_type': '--feature_type',
        'enable_view_0': '--enable_view_0'
    }
    
    # 1. 修正后的完整命令
    print("## 1. 修正后的完整命令")
    print("```bash")
    
    base_command = "python main.py"
    all_params = []
    
    for param_name, param_value in fixed_params.items():
        if param_name in param_mapping:
            all_params.append(f"{param_mapping[param_name]} {param_value}")
    
    # 添加必要的额外参数
    extra_params = [
        "--task_type LDA",
        "--run_name BESTMODELLDA", 
        "--seed 42"
    ]
    
    full_command = f"{base_command} " + " ".join(all_params) + " " + " ".join(extra_params)
    print(full_command)
    print("```\n")
    
    # 2. 分类显示的修正命令
    print("## 2. 分类参数命令（推荐）")
    print("```bash")
    print(f"{base_command} \\")
    
    # 基础设置
    print("  # 基础设置")
    print("  --task_type LDA \\")
    print("  --run_name BESTMODELLDA \\")
    print("  --seed 42 \\")
    
    # 网络架构参数
    network_params = ['dimensions', 'hidden1', 'hidden2', 'decoder1']
    print("  # 网络架构参数")
    for param in network_params:
        if param in fixed_params:
            print(f"  {param_mapping[param]} {fixed_params[param]} \\")
    
    # 优化参数
    opt_params = ['lr', 'dropout', 'weight_decay', 'batch']
    print("  # 优化参数")
    for param in opt_params:
        if param in fixed_params:
            print(f"  {param_mapping[param]} {fixed_params[param]} \\")
    
    # MoCo参数
    moco_params = ['moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_K', 'moco_type']
    print("  # MoCo参数")
    for param in moco_params:
        if param in fixed_params:
            print(f"  {param_mapping[param]} {fixed_params[param]} \\")
    
    # 注意力机制参数
    attention_params = ['gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy']
    print("  # 注意力机制参数")
    for param in attention_params:
        if param in fixed_params:
            print(f"  {param_mapping[param]} {fixed_params[param]} \\")
    
    # 其他参数
    other_params = ['alpha', 'beta', 'gamma', 'feature_type', 'enable_view_0']
    print("  # 其他参数")
    for i, param in enumerate(other_params):
        if param in fixed_params:
            if i == len(other_params) - 1:
                print(f"  {param_mapping[param]} {fixed_params[param]}")  # 最后一个不加反斜杠
            else:
                print(f"  {param_mapping[param]} {fixed_params[param]} \\")
    
    print("```\n")
    
    # 3. 参数修正说明
    print("## 3. 参数修正说明")
    print("```")
    print("修正的参数类型:")
    
    for param_name, original_value in params.items():
        if param_name in fixed_params:
            fixed_value = fixed_params[param_name]
            if str(original_value) != str(fixed_value):
                print(f"  {param_name}: {original_value} -> {fixed_value} (类型修正)")
            else:
                print(f"  {param_name}: {fixed_value} (无需修正)")
    print("```\n")
    
    # 4. 关键修正点
    print("## 4. 关键修正点")
    print("```")
    print("主要修正:")
    print(f"  - decoder1: {params['decoder1']} -> {fixed_params['decoder1']} (浮点数转整数)")
    print(f"  - 所有整数参数确保为整数类型")
    print(f"  - 所有浮点数参数确保为浮点数类型")
    print(f"  - 布尔参数转换为字符串格式")
    print("```\n")
    
    # 保存修正后的命令
    with open('fixed_best_params_command.sh', 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# 修正后的第5次迭代最佳参数命令行\n")
        f.write("# AUROC: 0.956639\n")
        f.write("# 修正了参数类型问题\n\n")
        f.write(full_command + "\n")
    
    # 保存修正后的参数
    fixed_config = {
        "task_type": "LDA",
        "run_name": "BESTMODELLDA",
        "seed": 42,
        "parameters": fixed_params
    }
    
    with open('fixed_best_params_config.json', 'w', encoding='utf-8') as f:
        json.dump(fixed_config, f, indent=2, ensure_ascii=False)
    
    print("## 5. 输出文件")
    print("```")
    print("已生成修正后的文件:")
    print("  - fixed_best_params_command.sh: 修正后的可执行bash脚本")
    print("  - fixed_best_params_config.json: 修正后的JSON配置文件")
    print("```")

if __name__ == "__main__":
    fix_command_line()