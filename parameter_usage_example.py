#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用新参数系统的示例

展示如何在autodl.py和其他模块中使用统一参数管理系统
"""

# 方法1: 使用原有的settings()函数（推荐，保持兼容性）
from parms_setting import settings

def main_with_settings():
    """使用settings()函数的方式"""
    args = settings()
    
    print(f"任务类型: {args.task_type}")
    print(f"最大迭代次数: {getattr(args, 'max_iterations', '未设置')}")
    print(f"随机种子: {getattr(args, 'random_seed', '未设置')}")
    print(f"训练轮数: {args.epochs}")


# 方法2: 直接使用参数管理器（高级用法）
from parameter_manager import get_parameter_manager
from unified_parameter_registry import initialize_unified_parameters

def main_with_manager():
    """使用参数管理器的方式"""
    # 初始化系统
    initialize_unified_parameters()
    
    # 获取管理器
    manager = get_parameter_manager()
    
    # 解析参数
    args = manager.parse_arguments()
    
    print(f"任务类型: {args.task_type}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"随机种子: {args.random_seed}")
    print(f"训练轮数: {args.epochs}")


# 方法3: 使用参数代理（延迟加载）
from parameter_manager import get_parameter_proxy

def main_with_proxy():
    """使用参数代理的方式"""
    proxy = get_parameter_proxy()
    
    # 延迟访问参数
    task_type = proxy.get('task_type', 'LDA')
    max_iterations = proxy.get('max_iterations', 50)
    
    print(f"任务类型: {task_type}")
    print(f"最大迭代次数: {max_iterations}")


if __name__ == "__main__":
    print("=== 方法1: 使用settings()函数 ===")
    main_with_settings()
    
    print("\n=== 方法2: 使用参数管理器 ===")
    # main_with_manager()  # 注释掉避免重复解析
    
    print("\n=== 方法3: 使用参数代理 ===")
    main_with_proxy()
