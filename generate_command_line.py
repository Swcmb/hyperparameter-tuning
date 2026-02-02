#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将第5次迭代的最佳参数转换为命令行格式
"""

import json
from pathlib import Path

def generate_command_line():
    """生成命令行参数"""
    
    # 读取参数文件
    params_file = Path("iteration_5_complete_params.json")
    
    if not params_file.exists():
        print("错误：找不到参数文件 iteration_5_complete_params.json")
        return
    
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    print("=== 第5次迭代最佳参数的命令行格式 ===\n")
    
    # 基础命令
    base_command = "python autodl.py"
    
    # 构建完整命令行
    command_parts = [base_command]
    
    # 添加任务类型（从参数推断，默认LDA）
    command_parts.append("--task_type LDA")
    
    # 按类别组织参数
    network_params = []
    optimization_params = []
    moco_params = []
    attention_params = []
    other_params = []
    
    # 参数映射和分类
    param_mapping = {
        # 网络架构参数
        'dimensions': '--dimensions',
        'hidden1': '--hidden1', 
        'hidden2': '--hidden2',
        'decoder1': '--decoder1',
        
        # 优化参数
        'lr': '--lr',
        'dropout': '--dropout',
        'weight_decay': '--weight_decay',
        'batch': '--batch',
        
        # MoCo参数
        'moco_momentum': '--moco_momentum',
        'moco_t': '--moco_t',
        'moco_tau1': '--moco_tau1',
        'moco_tau2': '--moco_tau2',
        'moco_K': '--moco_K',
        'moco_type': '--moco_type',
        
        # 注意力机制参数
        'gat_heads': '--gat_heads',
        'gt_heads': '--gt_heads', 
        'fusion_heads': '--fusion_heads',
        'fusion_strategy': '--fusion_strategy',
        
        # 其他参数
        'alpha': '--alpha',
        'beta': '--beta',
        'gamma': '--gamma',
        'feature_type': '--feature_type',
        'enable_view_0': '--enable_view_0'
    }
    
    # 生成参数列表
    for param_name, param_value in params.items():
        if param_name in param_mapping:
            cmd_param = param_mapping[param_name]
            
            # 分类存储
            if param_name in ['dimensions', 'hidden1', 'hidden2', 'decoder1']:
                network_params.append(f"{cmd_param} {param_value}")
            elif param_name in ['lr', 'dropout', 'weight_decay', 'batch']:
                optimization_params.append(f"{cmd_param} {param_value}")
            elif param_name.startswith('moco_'):
                moco_params.append(f"{cmd_param} {param_value}")
            elif param_name in ['gat_heads', 'gt_heads', 'fusion_heads', 'fusion_strategy']:
                attention_params.append(f"{cmd_param} {param_value}")
            else:
                other_params.append(f"{cmd_param} {param_value}")
    
    # 1. 完整的单行命令
    print("## 1. 完整单行命令")
    print("```bash")
    all_params = []
    for param_name, param_value in params.items():
        if param_name in param_mapping:
            all_params.append(f"{param_mapping[param_name]} {param_value}")
    
    full_command = f"{base_command} --task_type LDA " + " ".join(all_params)
    print(full_command)
    print("```\n")
    
    # 2. 分类显示的命令
    print("## 2. 分类参数命令")
    print("```bash")
    print(f"{base_command} \\")
    print("  --task_type LDA \\")
    
    if network_params:
        print("  # 网络架构参数")
        for param in network_params:
            print(f"  {param} \\")
    
    if optimization_params:
        print("  # 优化参数")
        for param in optimization_params:
            print(f"  {param} \\")
    
    if moco_params:
        print("  # MoCo参数")
        for param in moco_params:
            print(f"  {param} \\")
    
    if attention_params:
        print("  # 注意力机制参数")
        for param in attention_params:
            print(f"  {param} \\")
    
    if other_params:
        print("  # 其他参数")
        for i, param in enumerate(other_params):
            if i == len(other_params) - 1:
                print(f"  {param}")  # 最后一个参数不加反斜杠
            else:
                print(f"  {param} \\")
    
    print("```\n")
    
    # 3. 配置文件格式
    print("## 3. JSON配置文件格式")
    print("```json")
    config_dict = {
        "task_type": "LDA",
        "max_iterations": 100,
        "parameters": params
    }
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    print("```\n")
    
    # 4. 关键参数说明
    print("## 4. 关键参数说明")
    print("```")
    print("网络架构:")
    print(f"  - 嵌入维度: {params.get('dimensions', 'N/A')}")
    print(f"  - 隐藏层1: {params.get('hidden1', 'N/A')}")
    print(f"  - 隐藏层2: {params.get('hidden2', 'N/A')}")
    print(f"  - 解码器维度: {params.get('decoder1', 'N/A')}")
    print()
    print("优化设置:")
    print(f"  - 学习率: {params.get('lr', 'N/A')}")
    print(f"  - Dropout率: {params.get('dropout', 'N/A')}")
    print(f"  - 权重衰减: {params.get('weight_decay', 'N/A')}")
    print(f"  - 批次大小: {params.get('batch', 'N/A')}")
    print()
    print("MoCo设置:")
    print(f"  - 动量系数: {params.get('moco_momentum', 'N/A')}")
    print(f"  - 温度参数: {params.get('moco_t', 'N/A')}")
    print(f"  - Tau1: {params.get('moco_tau1', 'N/A')}")
    print(f"  - Tau2: {params.get('moco_tau2', 'N/A')}")
    print(f"  - 队列大小: {params.get('moco_K', 'N/A')}")
    print(f"  - MoCo类型: {params.get('moco_type', 'N/A')}")
    print()
    print("注意力机制:")
    print(f"  - GAT头数: {params.get('gat_heads', 'N/A')}")
    print(f"  - GT头数: {params.get('gt_heads', 'N/A')}")
    print(f"  - 融合头数: {params.get('fusion_heads', 'N/A')}")
    print(f"  - 融合策略: {params.get('fusion_strategy', 'N/A')}")
    print()
    print("其他设置:")
    print(f"  - Alpha: {params.get('alpha', 'N/A')}")
    print(f"  - Beta: {params.get('beta', 'N/A')}")
    print(f"  - Gamma: {params.get('gamma', 'N/A')}")
    print(f"  - 特征类型: {params.get('feature_type', 'N/A')}")
    print(f"  - 启用视图0: {params.get('enable_view_0', 'N/A')}")
    print("```\n")
    
    # 5. 性能信息
    print("## 5. 预期性能")
    print("```")
    print("使用此参数配置的预期性能指标:")
    print("  - AUROC: 0.956639")
    print("  - AUPRC: 0.950112")
    print("  - F1 Score: 0.894600")
    print("  - Precision: 0.887206")
    print("  - Recall: 0.902254")
    print("  - Loss: 11.067706")
    print("  - 评估时间: ~1100秒 (约18分钟)")
    print("```\n")
    
    # 保存命令到文件
    with open('best_params_command.sh', 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# 第5次迭代最佳参数的命令行\n")
        f.write("# AUROC: 0.956639\n\n")
        f.write(full_command + "\n")
    
    with open('best_params_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print("## 6. 文件输出")
    print("```")
    print("已生成以下文件:")
    print("  - best_params_command.sh: 可执行的bash脚本")
    print("  - best_params_config.json: JSON配置文件")
    print("```")

if __name__ == "__main__":
    generate_command_line()