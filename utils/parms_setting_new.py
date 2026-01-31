#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
新的参数设置模块 - 兼容层

使用统一参数管理系统，保持与原有parms_setting.py的兼容性
"""

import sys
import logging
from parameter_manager import get_parameter_manager, get_parameter_proxy
from unified_parameter_registry import initialize_unified_parameters


class MoCoParameterError(Exception):
    """MoCo参数相关错误 - 保持兼容性"""
    pass


def _validate_moco_parameters(args):
    """
    验证MoCo参数 - 保持与原版本的兼容性
    
    Args:
        args: 解析后的参数命名空间
    """
    errors = []
    
    # 验证MoCo基础参数
    if hasattr(args, 'moco_momentum') and args.moco_momentum is not None:
        if not (0.9 <= args.moco_momentum <= 0.9999):
            errors.append(f"moco_momentum 应在 [0.9, 0.9999] 范围内，当前值: {args.moco_momentum}")
    
    if hasattr(args, 'moco_t') and args.moco_t is not None:
        if not (0.01 <= args.moco_t <= 1.0):
            errors.append(f"moco_t 应在 [0.01, 1.0] 范围内，当前值: {args.moco_t}")
    
    # 验证DoubleTau参数
    if hasattr(args, 'moco_type') and args.moco_type == 'double_tau':
        if hasattr(args, 'moco_tau1') and hasattr(args, 'moco_tau2'):
            if args.moco_tau1 is not None and args.moco_tau2 is not None:
                if args.moco_tau2 < args.moco_tau1:
                    errors.append(f"DoubleTau模式下，moco_tau2 ({args.moco_tau2}) 应大于等于 moco_tau1 ({args.moco_tau1})")
    
    # 验证队列大小
    if hasattr(args, 'moco_K') and hasattr(args, 'batch'):
        if args.moco_K is not None and args.batch is not None:
            if args.moco_K < args.batch * 4:
                errors.append(f"moco_K ({args.moco_K}) 建议至少为批大小 ({args.batch}) 的4倍")
    
    if errors:
        print("❌ MoCo参数验证失败:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("💡 常见MoCo参数问题：", file=sys.stderr)
        print("  - moco_tau1, moco_tau2, moco_t 应为浮点数", file=sys.stderr)
        print("  - moco_K, moco_queue 应为整数", file=sys.stderr)
        print("  - enable_view_0 应为 true 或 false", file=sys.stderr)
        print("  - moco_type 应为 'basic' 或 'double_tau'", file=sys.stderr)
        print("  - 队列大小建议至少为批大小的4倍", file=sys.stderr)
        raise MoCoParameterError(f"发现 {len(errors)} 个MoCo参数错误")


def settings():
    """
    构建并解析实验参数，返回包含所有设置的命名空间对象。
    
    这是新版本，使用统一参数管理系统，但保持与原版本的完全兼容性。
    
    Returns:
        argparse.Namespace: 包含所有解析参数的命名空间对象
    """
    try:
        # 初始化统一参数系统
        initialize_unified_parameters()
        
        # 获取参数管理器
        manager = get_parameter_manager()
        
        # 解析参数
        args = manager.parse_arguments()
        
        # MoCo参数验证
        _validate_moco_parameters(args)
        
        # 参数后处理和规范化
        _post_process_parameters(args)
        
        return args
        
    except Exception as e:
        print(f"❌ 参数解析失败: {e}", file=sys.stderr)
        raise MoCoParameterError(f"参数解析错误: {e}")


def _post_process_parameters(args):
    """
    参数后处理和规范化 - 保持与原版本的兼容性
    
    Args:
        args: 解析后的参数命名空间
    """
    try:
        # 规范化validation_type
        if hasattr(args, 'validation_type'):
            if args.validation_type == '5-cv1':
                args.validation_type = '5_cv1'
            elif args.validation_type == '5-cv2':
                args.validation_type = '5_cv2'
        
        # MoCo proj_dim 兜底：None 或非法值时跟随 hidden2
        if hasattr(args, 'proj_dim') and hasattr(args, 'hidden2'):
            if args.proj_dim is None or args.proj_dim <= 0:
                args.proj_dim = args.hidden2
        
        # 确保布尔类型参数正确处理
        bool_params = [
            'save_datasets', 'shutdown', 'use_co_attention', 'use_multihead',
            'transformer_style', 'enable_view_0', 'adv_rand_init', 'adv_project',
            'adv_use_amp', 'adv_on_moco', 'enable_threshold_scan', 'enable_temp_scaling',
            'kfold_recompute', 'kfold_cache', 'multi_objective'
        ]
        
        for param in bool_params:
            if hasattr(args, param):
                value = getattr(args, param)
                if isinstance(value, str):
                    setattr(args, param, str(value).lower() in ('true', '1', 'yes', 'on'))
        
    except Exception as e:
        print(f"❌ 参数后处理时发生错误: {e}", file=sys.stderr)
        raise MoCoParameterError(f"参数后处理错误: {e}")


# 为了保持完全兼容性，提供一些常用的便捷函数
def get_args():
    """获取解析后的参数 - 兼容layer.py"""
    proxy = get_parameter_proxy()
    return proxy.get_args()


def get_parameter(name, default=None):
    """获取单个参数值"""
    proxy = get_parameter_proxy()
    return proxy.get(name, default)


# 测试函数
def test_compatibility():
    """测试与原版本的兼容性"""
    print("测试新版parms_setting兼容性...")
    
    try:
        # 测试settings()函数
        args = settings()
        print(f"✓ settings()函数正常工作")
        print(f"  - task_type: {args.task_type}")
        print(f"  - epochs: {args.epochs}")
        print(f"  - lr: {args.lr}")
        print(f"  - batch: {args.batch}")
        
        # 测试get_args()函数
        args2 = get_args()
        assert args2 is args, "get_args()应该返回相同的对象"
        print(f"✓ get_args()函数正常工作")
        
        # 测试get_parameter()函数
        task_type = get_parameter('task_type')
        assert task_type == args.task_type, "get_parameter()应该返回正确的值"
        print(f"✓ get_parameter()函数正常工作")
        
        print("✅ 兼容性测试通过！")
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        raise


if __name__ == "__main__":
    # 运行兼容性测试
    test_compatibility()
    
    # 测试MoCo参数验证
    print("\n测试MoCo参数验证...")
    
    # 模拟一些参数来测试验证
    import argparse
    test_args = argparse.Namespace()
    test_args.moco_momentum = 0.999
    test_args.moco_t = 0.2
    test_args.moco_type = 'basic'
    test_args.moco_K = 4096
    test_args.batch = 25
    
    try:
        _validate_moco_parameters(test_args)
        print("✓ MoCo参数验证通过")
    except MoCoParameterError as e:
        print(f"❌ MoCo参数验证失败: {e}")
    
    print("\n🎉 所有测试完成！")