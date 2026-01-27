"""
测试参数修复功能

演示如何自动修复违反约束的参数
"""

from autodl_core import create_default_parameter_space
from parameter_validator import ParameterValidator


def test_parameter_fix():
    """测试参数修复功能"""
    
    print("=== 参数修复功能测试 ===\n")
    
    # 创建参数空间和验证器
    space = create_default_parameter_space()
    validator = ParameterValidator(space)
    
    # 创建一个违反多个约束的参数组合
    broken_params = {
        'dimensions': 425.2,  # 浮点数，需要转换为整数
        'hidden1': 300.8,    # 违反递减约束，且为浮点数
        'hidden2': 150.3,    # 违反递减约束，且为浮点数
        'decoder1': 100.7,   # 违反解码器约束，且为浮点数
        'lr': 0.001,
        'dropout': 0.1,
        'weight_decay': 0.01,  # 违反学习率约束
        'alpha': 0.0,  # 违反损失权重约束
        'beta': 0.0,
        'gamma': 0.0,
        'gat_heads': 3,   # 违反整除约束
        'gt_heads': 5,    # 违反整除约束
        'fusion_heads': 7, # 违反整除约束
        'batch': 32,
        'moco_K': 64,     # 违反批大小约束
        'fusion_strategy': 'self_attention',
        'feature_type': 'normal',
        'moco_type': 'basic'
    }
    
    print("1. 原始参数（违反约束）:")
    print(f"   dimensions: {broken_params['dimensions']}")
    print(f"   hidden1: {broken_params['hidden1']}")
    print(f"   hidden2: {broken_params['hidden2']}")
    print(f"   decoder1: {broken_params['decoder1']}")
    print(f"   weight_decay: {broken_params['weight_decay']}")
    print(f"   alpha/beta/gamma: {broken_params['alpha']}/{broken_params['beta']}/{broken_params['gamma']}")
    print(f"   gat_heads: {broken_params['gat_heads']}")
    print(f"   gt_heads: {broken_params['gt_heads']}")
    print(f"   fusion_heads: {broken_params['fusion_heads']}")
    print(f"   moco_K: {broken_params['moco_K']}")
    print()
    
    # 验证原始参数
    is_valid, errors = validator.validate_parameters(broken_params)
    print(f"2. 原始参数验证: {'通过' if is_valid else '失败'}")
    if errors:
        print("   错误信息:")
        for error in errors:
            print(f"   - {error}")
    print()
    
    # 修复参数
    print("3. 自动修复参数...")
    fixed_params = validator.suggest_parameter_fix(broken_params)
    
    print("   修复后参数:")
    print(f"   dimensions: {fixed_params['dimensions']}")
    print(f"   hidden1: {fixed_params['hidden1']}")
    print(f"   hidden2: {fixed_params['hidden2']}")
    print(f"   decoder1: {fixed_params['decoder1']}")
    print(f"   weight_decay: {fixed_params['weight_decay']}")
    print(f"   alpha/beta/gamma: {fixed_params['alpha']}/{fixed_params['beta']}/{fixed_params['gamma']}")
    print(f"   gat_heads: {fixed_params['gat_heads']}")
    print(f"   gt_heads: {fixed_params['gt_heads']}")
    print(f"   fusion_heads: {fixed_params['fusion_heads']}")
    print(f"   moco_K: {fixed_params['moco_K']}")
    print()
    
    # 验证修复后的参数
    is_valid_fixed, errors_fixed = validator.validate_parameters(fixed_params)
    print(f"4. 修复后参数验证: {'通过' if is_valid_fixed else '失败'}")
    if errors_fixed:
        print("   剩余错误:")
        for error in errors_fixed:
            print(f"   - {error}")
    else:
        print("   所有约束都已满足！")
    print()
    
    # 显示修复摘要
    print("5. 修复摘要:")
    changes = []
    
    if fixed_params['hidden1'] != broken_params['hidden1']:
        changes.append(f"hidden1: {broken_params['hidden1']} → {fixed_params['hidden1']}")
    if fixed_params['hidden2'] != broken_params['hidden2']:
        changes.append(f"hidden2: {broken_params['hidden2']} → {fixed_params['hidden2']}")
    if fixed_params['weight_decay'] != broken_params['weight_decay']:
        changes.append(f"weight_decay: {broken_params['weight_decay']} → {fixed_params['weight_decay']}")
    if fixed_params['gat_heads'] != broken_params['gat_heads']:
        changes.append(f"gat_heads: {broken_params['gat_heads']} → {fixed_params['gat_heads']}")
    if fixed_params['gt_heads'] != broken_params['gt_heads']:
        changes.append(f"gt_heads: {broken_params['gt_heads']} → {fixed_params['gt_heads']}")
    if fixed_params['fusion_heads'] != broken_params['fusion_heads']:
        changes.append(f"fusion_heads: {broken_params['fusion_heads']} → {fixed_params['fusion_heads']}")
    
    if changes:
        print("   参数修改:")
        for change in changes:
            print(f"   - {change}")
    else:
        print("   没有参数被修改")
    
    print("\n=== 测试完成 ===")
    return is_valid_fixed


if __name__ == "__main__":
    success = test_parameter_fix()
    if success:
        print("✓ 参数修复功能正常工作")
    else:
        print("✗ 参数修复功能需要进一步改进")