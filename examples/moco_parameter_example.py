#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoCo参数优化使用示例

本示例展示如何使用贝叶斯优化系统优化MoCo（Momentum Contrast）对比学习相关的超参数。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodl_core import create_default_parameter_space
from bayesian_optimizer import create_bayesian_optimizer
from parameter_validator import ParameterValidator


def example_basic_moco_optimization():
    """基础MoCo参数优化示例"""
    print("=" * 60)
    print("基础MoCo参数优化示例")
    print("=" * 60)
    
    # 1. 创建参数空间
    parameter_space = create_default_parameter_space("LDA")
    print(f"参数空间包含 {len(parameter_space.parameters)} 个参数")
    
    # 2. 显示MoCo相关参数
    moco_params = ['moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_K', 'moco_type', 'enable_view_0']
    print("\nMoCo相关参数:")
    for param_name in moco_params:
        if param_name in parameter_space.parameters:
            param_config = parameter_space.parameters[param_name]
            print(f"  - {param_name}: {param_config.param_type.value}")
            if hasattr(param_config, 'bounds') and param_config.bounds:
                print(f"    范围: {param_config.bounds}")
            if hasattr(param_config, 'values') and param_config.values:
                print(f"    可选值: {param_config.values}")
    
    # 3. 采样MoCo参数组合
    print("\n采样MoCo参数组合:")
    for i in range(3):
        params = parameter_space.sample_random_parameters(seed=42+i)
        print(f"\n示例 {i+1}:")
        for param_name in moco_params:
            if param_name in params:
                print(f"  {param_name}: {params[param_name]}")
        
        # 验证参数有效性
        is_valid, errors = parameter_space.validate_parameters_detailed(params)
        if is_valid:
            print("  ✓ 参数组合有效")
        else:
            print(f"  ✗ 参数组合无效: {errors}")


def example_moco_constraint_validation():
    """MoCo参数约束验证示例"""
    print("\n" + "=" * 60)
    print("MoCo参数约束验证示例")
    print("=" * 60)
    
    parameter_space = create_default_parameter_space("LDA")
    validator = ParameterValidator(parameter_space)
    
    # 测试用例1: 有效的MoCo参数组合
    print("\n测试用例1: 有效的MoCo参数组合")
    valid_params = {
        'moco_momentum': 0.999,
        'moco_t': 0.2,
        'moco_tau1': 0.15,
        'moco_tau2': 0.25,  # tau2 >= tau1
        'moco_K': 4096,
        'moco_type': 'double_tau',
        'enable_view_0': 'true'
    }
    
    for param, value in valid_params.items():
        print(f"  {param}: {value}")
    
    is_valid, errors = validator.validate_parameters(valid_params)
    if is_valid:
        print("  ✓ 所有约束满足")
    else:
        print(f"  ✗ 约束违反: {errors}")
    
    # 测试用例2: 违反tau约束
    print("\n测试用例2: 违反tau约束 (tau1 > tau2)")
    invalid_tau_params = {
        'moco_tau1': 0.3,
        'moco_tau2': 0.2,  # 违反约束: tau2 < tau1
        'moco_momentum': 0.999,
        'moco_t': 0.2
    }
    
    for param, value in invalid_tau_params.items():
        print(f"  {param}: {value}")
    
    tau_constraint = validator.constraint_functions['moco_tau_ordering']
    if tau_constraint(invalid_tau_params):
        print("  ✓ tau约束满足")
    else:
        print("  ✗ tau约束违反: tau2必须大于等于tau1")
    
    # 测试参数修复
    print("\n参数修复:")
    fixed_params = parameter_space.suggest_parameter_fix(invalid_tau_params)
    print(f"  修复前: tau1={invalid_tau_params['moco_tau1']}, tau2={invalid_tau_params['moco_tau2']}")
    print(f"  修复后: tau1={fixed_params.get('moco_tau1', 'N/A')}, tau2={fixed_params.get('moco_tau2', 'N/A')}")
    
    # 测试用例3: 违反动量约束
    print("\n测试用例3: 违反动量约束")
    invalid_momentum_params = {
        'moco_momentum': 0.8,  # 违反约束: < 0.9
        'moco_tau1': 0.2,
        'moco_tau2': 0.3
    }
    
    for param, value in invalid_momentum_params.items():
        print(f"  {param}: {value}")
    
    momentum_constraint = validator.constraint_functions['moco_momentum_range']
    if momentum_constraint(invalid_momentum_params):
        print("  ✓ 动量约束满足")
    else:
        print("  ✗ 动量约束违反: 动量系数应在0.9-0.9999范围内")


def example_moco_optimization_workflow():
    """完整的MoCo参数优化工作流示例"""
    print("\n" + "=" * 60)
    print("完整的MoCo参数优化工作流示例")
    print("=" * 60)
    
    try:
        # 1. 创建贝叶斯优化器
        print("1. 创建贝叶斯优化器...")
        optimizer = create_bayesian_optimizer(
            task_type="LDA",
            acquisition_function_type="EI",
            n_initial_points=3,
            random_state=42
        )
        print("   ✓ 优化器创建成功")
        
        # 2. 初始化优化器
        print("2. 初始化优化器...")
        optimizer._initialize_optimization()
        print("   ✓ 优化器初始化完成")
        
        # 3. 进行几次参数建议和评估的模拟
        print("3. 模拟优化过程...")
        
        for iteration in range(1, 4):
            print(f"\n   迭代 {iteration}:")
            
            # 获取参数建议
            suggested_params = optimizer.suggest_next_parameters()
            
            # 显示MoCo相关参数
            moco_params = ['moco_momentum', 'moco_t', 'moco_tau1', 'moco_tau2', 'moco_K', 'moco_type', 'enable_view_0']
            print("   建议的MoCo参数:")
            for param in moco_params:
                if param in suggested_params:
                    value = suggested_params[param]
                    if isinstance(value, float):
                        print(f"     {param}: {value:.4f}")
                    else:
                        print(f"     {param}: {value}")
            
            # 模拟评估结果
            import random
            random.seed(42 + iteration)
            mock_auroc = 0.7 + random.random() * 0.2  # 0.7-0.9
            mock_metrics = {
                'AUROC': mock_auroc,
                'AUPRC': mock_auroc - 0.05,
                'F1': mock_auroc - 0.1
            }
            
            print(f"   模拟评估结果: AUROC={mock_auroc:.4f}")
            
            # 更新优化器
            optimizer.update_model(
                parameters=suggested_params,
                objective_value=mock_auroc,
                metrics=mock_metrics,
                evaluation_time=1.0
            )
            print("   ✓ 优化器已更新")
        
        print("\n   ✓ 模拟优化完成")
        
    except Exception as e:
        print(f"   ✗ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()


def example_moco_configuration_tips():
    """MoCo参数配置建议"""
    print("\n" + "=" * 60)
    print("MoCo参数配置建议")
    print("=" * 60)
    
    print("""
MoCo参数配置最佳实践:

1. 基础MoCo参数:
   - moco_momentum: 建议范围 0.995-0.999，较高的动量有助于稳定特征表示
   - moco_t: 建议范围 0.1-0.3，温度系数影响对比学习的难度
   - moco_K: 建议使用 4096 或 8192，更大的队列提供更多负样本

2. DoubleTau MoCo参数:
   - moco_tau1: 正样本温度，建议范围 0.1-0.3
   - moco_tau2: 负样本温度，建议范围 0.2-0.4
   - 约束: 必须满足 tau2 >= tau1
   - 用途: 允许对正负样本使用不同的温度系数

3. 视图控制参数:
   - enable_view_0: 控制是否启用第0视图
   - 'true': 启用所有视图，增加数据多样性
   - 'false': 禁用第0视图，可能减少计算开销

4. 参数组合建议:
   - 基础模式: moco_type='basic', 使用 moco_t 作为统一温度
   - 高级模式: moco_type='double_tau', 使用 moco_tau1 和 moco_tau2
   - 大数据集: 增大 moco_K，提高 moco_momentum
   - 小数据集: 适当降低 moco_K，调整温度参数

5. 调优策略:
   - 先优化基础参数 (moco_momentum, moco_t, moco_K)
   - 再尝试 DoubleTau 模式进行精细调优
   - 根据任务特点调整 enable_view_0
   - 监控验证集性能，避免过拟合

6. 常见问题:
   - 如果性能不稳定，尝试提高 moco_momentum
   - 如果收敛太慢，尝试降低温度参数
   - 如果内存不足，减小 moco_K
   - 如果训练时间过长，考虑禁用某些视图
    """)


def main():
    """主函数"""
    print("MoCo参数优化使用示例")
    print("本示例展示如何使用贝叶斯优化系统优化MoCo对比学习参数")
    
    try:
        # 运行所有示例
        example_basic_moco_optimization()
        example_moco_constraint_validation()
        example_moco_optimization_workflow()
        example_moco_configuration_tips()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())