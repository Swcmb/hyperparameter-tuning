#!/usr/bin/env python3
"""
参数空间管理器使用示例

展示如何使用ParameterSpace类进行参数管理、验证和采样
"""

from autodl_core import create_default_parameter_space, ParameterSpace
import json


def main():
    print("=== 参数空间管理器使用示例 ===\n")
    
    # 1. 创建默认参数空间
    print("1. 创建参数空间")
    space = create_default_parameter_space("LDA")
    print(f"   创建了包含 {space.get_parameter_count()} 个参数的空间")
    print(f"   连续型参数: {len(space.get_continuous_parameter_names())} 个")
    print(f"   离散型参数: {len(space.get_discrete_parameter_names())} 个")
    print(f"   分类型参数: {len(space.get_categorical_parameter_names())} 个")
    
    # 2. 随机采样参数
    print("\n2. 随机采样参数")
    params = space.sample_random_parameters(seed=42)
    print("   采样结果:")
    for name, value in params.items():
        if isinstance(value, float):
            print(f"     {name}: {value:.6f}")
        else:
            print(f"     {name}: {value}")
    
    # 3. 验证参数
    print("\n3. 验证参数")
    is_valid, errors = space.validate_parameters_detailed(params)
    print(f"   验证结果: {'✓ 有效' if is_valid else '✗ 无效'}")
    if errors:
        print("   错误信息:")
        for error in errors:
            print(f"     - {error}")
    
    # 4. 创建有问题的参数并修复
    print("\n4. 参数修复示例")
    broken_params = params.copy()
    broken_params['hidden1'] = 600  # 超出范围
    broken_params['gat_heads'] = 3   # 不在允许值中
    broken_params['alpha'] = 0       # 违反约束
    broken_params['beta'] = 0
    broken_params['gamma'] = 0
    
    print("   原始有问题的参数:")
    print(f"     hidden1: {broken_params['hidden1']} (应该在64-256范围内)")
    print(f"     gat_heads: {broken_params['gat_heads']} (应该在[2,4,8,16]中)")
    print(f"     alpha/beta/gamma: {broken_params['alpha']}/{broken_params['beta']}/{broken_params['gamma']} (至少一个>0)")
    
    # 验证有问题的参数
    is_broken_valid, broken_errors = space.validate_parameters_detailed(broken_params)
    print(f"   验证结果: {'✓ 有效' if is_broken_valid else '✗ 无效'} ({len(broken_errors)} 个错误)")
    
    # 修复参数
    fixed_params = space.suggest_parameter_fix(broken_params)
    print("   修复后的参数:")
    print(f"     hidden1: {broken_params['hidden1']} -> {fixed_params['hidden1']}")
    print(f"     gat_heads: {broken_params['gat_heads']} -> {fixed_params['gat_heads']}")
    print(f"     alpha: {broken_params['alpha']} -> {fixed_params['alpha']}")
    
    # 验证修复后的参数
    is_fixed_valid, fixed_errors = space.validate_parameters_detailed(fixed_params)
    print(f"   修复后验证: {'✓ 有效' if is_fixed_valid else '✗ 无效'} ({len(fixed_errors)} 个错误)")
    
    # 5. 序列化和反序列化
    print("\n5. 序列化示例")
    space_dict = space.to_dict()
    print(f"   序列化后大小: {len(json.dumps(space_dict))} 字符")
    
    # 保存到文件
    with open('parameter_space_config.json', 'w', encoding='utf-8') as f:
        json.dump(space_dict, f, indent=2, ensure_ascii=False)
    print("   已保存到 parameter_space_config.json")
    
    # 从文件加载
    with open('parameter_space_config.json', 'r', encoding='utf-8') as f:
        loaded_dict = json.load(f)
    
    restored_space = ParameterSpace.from_dict(loaded_dict)
    print(f"   从文件恢复的参数空间包含 {restored_space.get_parameter_count()} 个参数")
    
    # 6. 参数信息查询
    print("\n6. 参数信息查询")
    lr_info = space.get_parameter_info('lr')
    print("   学习率参数信息:")
    print(f"     类型: {lr_info['param_type']}")
    print(f"     范围: {lr_info['bounds']}")
    print(f"     对数尺度: {lr_info['log_scale']}")
    
    fusion_info = space.get_parameter_info('fusion_strategy')
    print("   融合策略参数信息:")
    print(f"     类型: {fusion_info['param_type']}")
    print(f"     可选值: {fusion_info['values']}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()