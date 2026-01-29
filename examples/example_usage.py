"""
核心数据结构使用示例

演示如何使用贝叶斯优化系统的核心数据结构
"""

from datetime import datetime
import numpy as np
from autodl_core import (
    create_default_parameter_space, OptimizationResult, OptimizationHistory
)
from parameter_validator import ParameterValidator, ConfigurationConverter


def main():
    """主函数：演示核心功能的使用"""
    
    print("=== 贝叶斯超参数优化系统核心数据结构示例 ===\n")
    
    # 1. 创建参数空间
    print("1. 创建参数空间")
    space = create_default_parameter_space("LDA")
    print(f"   - 参数空间包含 {len(space.parameters)} 个参数")
    print(f"   - 连续型参数: {len(space.get_continuous_parameter_names())} 个")
    print(f"   - 离散型参数: {len(space.get_discrete_parameter_names())} 个")
    print(f"   - 分类型参数: {len(space.get_categorical_parameter_names())} 个")
    print()
    
    # 2. 参数采样和验证
    print("2. 参数采样和验证")
    validator = ParameterValidator(space)
    
    # 采样有效参数
    params = space.sample_random_parameters(seed=42)
    is_valid, errors = validator.validate_parameters(params)
    print(f"   - 随机采样参数验证: {'通过' if is_valid else '失败'}")
    if errors:
        print(f"   - 错误信息: {errors}")
    
    # 显示部分参数
    key_params = ['dimensions', 'lr', 'batch', 'fusion_strategy']
    print("   - 关键参数值:")
    for key in key_params:
        if key in params:
            print(f"     {key}: {params[key]}")
    print()
    
    # 3. 配置转换
    print("3. 配置转换")
    converter = ConfigurationConverter("LDA")
    exp_config = converter.convert_to_experiment_config(params)
    config_valid, config_errors = converter.validate_experiment_config(exp_config)
    print(f"   - 实验配置转换: {'成功' if config_valid else '失败'}")
    print(f"   - 配置项数量: {len(exp_config)}")
    if config_errors:
        print(f"   - 配置错误: {config_errors}")
    print()
    
    # 4. 优化结果管理
    print("4. 优化结果管理")
    history = OptimizationHistory()
    history.task_type = "LDA"
    history.acquisition_function = "EI"
    
    # 模拟多次优化迭代
    np.random.seed(42)
    for i in range(5):
        # 模拟参数评估
        iteration_params = space.sample_random_parameters(seed=42+i)
        
        # 模拟性能指标（随机生成，实际应该来自模型训练）
        auroc = 0.7 + np.random.random() * 0.25  # 0.7-0.95
        auprc = auroc - 0.05 + np.random.random() * 0.1  # 略低于AUROC
        f1 = auroc - 0.1 + np.random.random() * 0.15  # 通常低于AUROC
        
        result = OptimizationResult(
            parameters=iteration_params,
            objective_value=auroc,
            metrics={
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1
            },
            iteration=i+1,
            timestamp=datetime.now(),
            evaluation_time=100.0 + np.random.random() * 50  # 100-150秒
        )
        
        history.add_result(result)
        print(f"   - 迭代 {i+1}: AUROC={auroc:.4f}, 评估时间={result.evaluation_time:.1f}s")
    
    print(f"   - 总迭代次数: {history.total_iterations}")
    print(f"   - 最佳AUROC: {history.get_best_objective_value():.4f}")
    print()
    
    # 5. 收敛分析
    print("5. 收敛分析")
    convergence = history.get_convergence_curve()
    print("   - 收敛曲线 (历史最佳值):")
    for i, value in enumerate(convergence, 1):
        print(f"     迭代 {i}: {value:.4f}")
    
    improvement = convergence[-1] - convergence[0]
    print(f"   - 总体改进: {improvement:.4f}")
    print()
    
    # 6. 参数重要性分析（简化版）
    print("6. 参数历史分析")
    lr_history = history.get_parameter_history('lr')
    batch_history = history.get_parameter_history('batch')
    
    print(f"   - 学习率变化范围: {min(lr_history):.2e} - {max(lr_history):.2e}")
    print(f"   - 批大小变化范围: {min(batch_history)} - {max(batch_history)}")
    
    # 找到最佳参数组合
    best_params = history.get_best_parameters()
    print(f"   - 最佳学习率: {best_params['lr']:.2e}")
    print(f"   - 最佳批大小: {best_params['batch']}")
    print()
    
    # 7. 数据持久化示例
    print("7. 数据持久化")
    
    # 转换为字典格式（可保存为JSON）
    history_dict = history.to_dict()
    print(f"   - 历史数据序列化: {len(history_dict)} 个字段")
    
    # 从字典恢复
    restored_history = OptimizationHistory.from_dict(history_dict)
    print(f"   - 数据恢复验证: {'成功' if restored_history.total_iterations == history.total_iterations else '失败'}")
    print()
    
    print("=== 示例完成 ===")
    print("核心数据结构已成功实现，支持:")
    print("- 灵活的参数空间定义（连续型、离散型、分类型）")
    print("- 智能的参数约束检查和修复")
    print("- 完整的优化历史记录和分析")
    print("- 实验配置的自动转换")
    print("- 数据的序列化和持久化")


if __name__ == "__main__":
    main()