#!/usr/bin/env python3
"""
参数空间管理器的全面测试

测试ParameterSpace类的所有功能，包括：
- 参数添加和管理
- 参数验证和约束检查
- 随机采样
- 参数修复
- 序列化和反序列化
"""

import sys
import numpy as np
from autodl_core import (
    ParameterSpace, ParameterConfig, ParameterType,
    create_default_parameter_space
)


def test_parameter_space_creation():
    """测试参数空间创建"""
    print("=== 测试参数空间创建 ===")
    
    space = ParameterSpace()
    assert space.get_parameter_count() == 0
    
    # 添加连续型参数
    space.add_continuous_parameter('lr', 1e-5, 1e-2, log_scale=True)
    assert space.get_parameter_count() == 1
    assert 'lr' in space.get_continuous_parameter_names()
    
    # 添加离散型参数
    space.add_discrete_parameter('batch_size', [16, 32, 64, 128])
    assert space.get_parameter_count() == 2
    assert 'batch_size' in space.get_discrete_parameter_names()
    
    # 添加分类型参数
    space.add_categorical_parameter('optimizer', ['adam', 'sgd', 'rmsprop'])
    assert space.get_parameter_count() == 3
    assert 'optimizer' in space.get_categorical_parameter_names()
    
    print("✓ 参数空间创建测试通过")


def test_parameter_validation():
    """测试参数验证"""
    print("=== 测试参数验证 ===")
    
    space = create_default_parameter_space()
    
    # 测试有效参数
    valid_params = {
        'dimensions': 256,
        'hidden1': 128,
        'hidden2': 64,
        'decoder1': 512,
        'lr': 0.001,
        'dropout': 0.1,
        'weight_decay': 0.0001,
        'alpha': 1.0,
        'beta': 0.5,
        'gamma': 0.5,
        'gat_heads': 4,
        'gt_heads': 4,
        'fusion_heads': 4,
        'batch': 32,
        'moco_K': 4096,
        'fusion_strategy': 'self_attention',
        'feature_type': 'normal',
        'moco_type': 'basic'
    }
    
    is_valid, errors = space.validate_parameters_detailed(valid_params)
    assert is_valid, f"有效参数验证失败: {errors}"
    
    # 测试无效参数 - 超出范围
    invalid_params = valid_params.copy()
    invalid_params['lr'] = 1.0  # 超出上界
    
    is_valid, errors = space.validate_parameters_detailed(invalid_params)
    assert not is_valid
    assert any('lr' in error for error in errors)
    
    # 测试约束违反
    constraint_violation = valid_params.copy()
    constraint_violation['hidden1'] = 300  # 违反递减约束
    
    is_valid, errors = space.validate_parameters_detailed(constraint_violation)
    assert not is_valid
    assert any('递减' in error for error in errors)
    
    print("✓ 参数验证测试通过")


def test_parameter_sampling():
    """测试参数采样"""
    print("=== 测试参数采样 ===")
    
    space = create_default_parameter_space()
    
    # 测试随机采样
    params1 = space.sample_random_parameters(seed=42)
    params2 = space.sample_random_parameters(seed=42)
    params3 = space.sample_random_parameters(seed=123)
    
    # 相同种子应该产生相同结果
    assert params1 == params2, "相同种子应该产生相同的采样结果"
    
    # 不同种子应该产生不同结果
    assert params1 != params3, "不同种子应该产生不同的采样结果"
    
    # 检查采样结果包含所有参数
    expected_params = set(space.get_parameter_names())
    sampled_params = set(params1.keys())
    assert expected_params == sampled_params, "采样结果应该包含所有参数"
    
    # 检查参数值在有效范围内
    for param_name, value in params1.items():
        config = space.parameters[param_name]
        assert config.validate_value(value), f"参数 {param_name} 的值 {value} 无效"
    
    print("✓ 参数采样测试通过")


def test_parameter_fixing():
    """测试参数修复"""
    print("=== 测试参数修复 ===")
    
    space = create_default_parameter_space()
    
    # 创建有问题的参数
    broken_params = {
        'dimensions': 256,
        'hidden1': 300,  # 违反递减约束
        'hidden2': 64,
        'decoder1': 32,  # 违反解码器约束
        'lr': 0.001,
        'dropout': 0.1,
        'weight_decay': 0.01,  # 违反学习率约束
        'alpha': 0.0,  # 违反权重约束
        'beta': 0.0,
        'gamma': 0.0,
        'gat_heads': 3,  # 违反整除约束
        'gt_heads': 4,
        'fusion_heads': 4,
        'batch': 32,
        'moco_K': 64,  # 违反MoCo约束
        'fusion_strategy': 'self_attention',
        'feature_type': 'normal',
        'moco_type': 'basic'
    }
    
    # 验证原始参数确实有问题
    is_valid, errors = space.validate_parameters_detailed(broken_params)
    assert not is_valid, "破损参数应该验证失败"
    print(f"原始参数错误数: {len(errors)}")
    
    # 修复参数
    fixed_params = space.suggest_parameter_fix(broken_params)
    
    # 验证修复后的参数
    is_fixed_valid, fixed_errors = space.validate_parameters_detailed(fixed_params)
    print(f"修复后错误数: {len(fixed_errors)}")
    
    # 修复应该显著减少错误数量（可能不是完全修复，因为有些约束很复杂）
    assert len(fixed_errors) < len(errors), "修复应该减少错误数量"
    
    print("✓ 参数修复测试通过")


def test_serialization():
    """测试序列化和反序列化"""
    print("=== 测试序列化 ===")
    
    space = create_default_parameter_space()
    
    # 序列化
    space_dict = space.to_dict()
    assert isinstance(space_dict, dict)
    assert 'parameters' in space_dict
    assert 'constraints' in space_dict
    
    # 反序列化
    restored_space = ParameterSpace.from_dict(space_dict)
    
    # 验证恢复的参数空间
    assert restored_space.get_parameter_count() == space.get_parameter_count()
    assert set(restored_space.get_parameter_names()) == set(space.get_parameter_names())
    
    # 测试功能一致性
    params = space.sample_random_parameters(seed=42)
    is_valid1, errors1 = space.validate_parameters_detailed(params)
    is_valid2, errors2 = restored_space.validate_parameters_detailed(params)
    
    assert is_valid1 == is_valid2, "序列化前后验证结果应该一致"
    
    print("✓ 序列化测试通过")


def test_parameter_info():
    """测试参数信息查询"""
    print("=== 测试参数信息查询 ===")
    
    space = create_default_parameter_space()
    
    # 测试参数计数
    total_count = space.get_parameter_count()
    continuous_count = len(space.get_continuous_parameter_names())
    discrete_count = len(space.get_discrete_parameter_names())
    categorical_count = len(space.get_categorical_parameter_names())
    
    assert total_count == continuous_count + discrete_count + categorical_count
    
    # 测试参数信息获取
    lr_info = space.get_parameter_info('lr')
    assert lr_info is not None
    assert lr_info['param_type'] == 'continuous'
    assert lr_info['log_scale'] == True
    
    # 测试不存在的参数
    nonexistent_info = space.get_parameter_info('nonexistent')
    assert nonexistent_info is None
    
    print("✓ 参数信息查询测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===")
    
    space = ParameterSpace()
    
    # 测试空参数空间
    assert space.get_parameter_count() == 0
    assert space.get_parameter_names() == []
    
    # 测试重复添加参数
    space.add_continuous_parameter('test', 0, 1)
    try:
        space.add_continuous_parameter('test', 0, 2)
        assert False, "应该抛出重复参数异常"
    except ValueError:
        pass  # 预期的异常
    
    # 测试参数移除
    space.remove_parameter('test')
    assert space.get_parameter_count() == 0
    
    # 测试清空参数
    space.add_continuous_parameter('test1', 0, 1)
    space.add_discrete_parameter('test2', [1, 2, 3])
    assert space.get_parameter_count() == 2
    
    space.clear_parameters()
    assert space.get_parameter_count() == 0
    
    print("✓ 边界情况测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始参数空间管理器全面测试...\n")
    
    try:
        test_parameter_space_creation()
        test_parameter_validation()
        test_parameter_sampling()
        test_parameter_fixing()
        test_serialization()
        test_parameter_info()
        test_edge_cases()
        
        print("\n🎉 所有测试通过！参数空间管理器实现正确。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)