#!/usr/bin/env python3
"""
简单的多目标优化测试

验证多目标优化功能的基本工作原理，不依赖实际的模型训练。
"""

import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayesian_optimizer import create_multi_objective_optimizer
from autodl_core import OptimizationResult, OptimizationHistory


def mock_evaluate_parameters(parameters):
    """模拟参数评估函数"""
    # 使用参数的哈希值作为随机种子，确保一致性
    param_hash = hash(str(sorted(parameters.items())))
    np.random.seed(abs(param_hash) % 2**32)
    
    # 生成模拟的性能指标
    auroc = 0.6 + 0.3 * np.random.random()
    auprc = 0.5 + 0.4 * np.random.random()
    f1 = 0.4 + 0.5 * np.random.random()
    
    return {
        'AUROC': min(1.0, max(0.0, auroc)),
        'AUPRC': min(1.0, max(0.0, auprc)),
        'F1': min(1.0, max(0.0, f1))
    }


def test_multi_objective_basic():
    """测试多目标优化的基本功能"""
    print("=== 测试多目标优化基本功能 ===")
    
    # 模拟TaskEvaluator
    with patch('bayesian_optimizer.TaskEvaluator') as mock_task_evaluator:
        # 设置模拟对象
        mock_instance = Mock()
        mock_instance.evaluate_parameters.side_effect = mock_evaluate_parameters
        mock_instance.extract_multi_objective_values.side_effect = lambda params, objectives: {
            obj: mock_evaluate_parameters(params)[obj] for obj in objectives
        }
        mock_task_evaluator.return_value = mock_instance
        
        try:
            # 创建多目标优化器
            optimizer = create_multi_objective_optimizer(
                task_type='LDA',
                objectives=['AUROC', 'AUPRC'],
                objective_weights={'AUROC': 0.6, 'AUPRC': 0.4},
                n_initial_points=3,
                random_state=42
            )
            
            print(f"✓ 多目标优化器创建成功")
            print(f"  目标函数: {optimizer.objectives}")
            print(f"  目标权重: {optimizer.objective_weights}")
            print(f"  是否多目标: {optimizer.is_multi_objective}")
            
            # 运行优化
            history = optimizer.optimize(n_iterations=5, checkpoint_freq=3)
            
            print(f"✓ 优化完成")
            print(f"  总迭代次数: {history.total_iterations}")
            print(f"  结果数量: {len(history.results)}")
            
            # 检查结果
            if history.results:
                best_result = history.best_result
                if best_result and best_result.objective_values:
                    print(f"  最佳结果: AUROC={best_result.objective_values['AUROC']:.4f}, "
                          f"AUPRC={best_result.objective_values['AUPRC']:.4f}")
                
                # 检查帕累托前沿
                if hasattr(history, 'pareto_front') and history.pareto_front:
                    print(f"  帕累托前沿大小: {len(history.pareto_front)}")
                    for i, result in enumerate(history.pareto_front[:3]):
                        obj_vals = result.objective_values
                        print(f"    解 {i+1}: AUROC={obj_vals['AUROC']:.4f}, "
                              f"AUPRC={obj_vals['AUPRC']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_weighted_objective_calculation():
    """测试加权目标函数计算"""
    print("\n=== 测试加权目标函数计算 ===")
    
    with patch('bayesian_optimizer.TaskEvaluator') as mock_task_evaluator:
        mock_instance = Mock()
        mock_instance.evaluate_parameters.side_effect = mock_evaluate_parameters
        mock_instance.extract_multi_objective_values.side_effect = lambda params, objectives: {
            obj: mock_evaluate_parameters(params)[obj] for obj in objectives
        }
        mock_task_evaluator.return_value = mock_instance
        
        try:
            # 测试不同的权重配置
            weight_configs = [
                {'AUROC': 1.0, 'AUPRC': 0.0},  # 只关注AUROC
                {'AUROC': 0.0, 'AUPRC': 1.0},  # 只关注AUPRC
                {'AUROC': 0.5, 'AUPRC': 0.5},  # 均等权重
            ]
            
            for i, weights in enumerate(weight_configs):
                print(f"\n配置 {i+1}: {weights}")
                
                optimizer = create_multi_objective_optimizer(
                    task_type='LDA',
                    objectives=['AUROC', 'AUPRC'],
                    objective_weights=weights,
                    n_initial_points=2,
                    random_state=42 + i
                )
                
                # 验证权重标准化
                normalized_weights = optimizer.objective_weights
                weight_sum = sum(normalized_weights.values())
                print(f"  权重和: {weight_sum:.6f}")
                
                if abs(weight_sum - 1.0) < 1e-6:
                    print(f"  ✓ 权重标准化正确")
                else:
                    print(f"  ✗ 权重标准化错误")
                    return False
                
                # 运行短期优化
                history = optimizer.optimize(n_iterations=3, checkpoint_freq=2)
                
                # 验证加权目标函数计算
                if history.results:
                    result = history.results[0]
                    if result.objective_values:
                        # 手动计算加权值
                        expected_weighted = sum(
                            weights[obj] * result.objective_values[obj] 
                            for obj in weights.keys()
                        )
                        
                        # 系统计算的加权值
                        actual_weighted = history.get_weighted_objective_value(result)
                        
                        print(f"  期望加权值: {expected_weighted:.6f}")
                        print(f"  实际加权值: {actual_weighted:.6f}")
                        
                        if abs(actual_weighted - expected_weighted) < 1e-6:
                            print(f"  ✓ 加权目标函数计算正确")
                        else:
                            print(f"  ✗ 加权目标函数计算错误")
                            return False
            
            return True
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_pareto_front_computation():
    """测试帕累托前沿计算"""
    print("\n=== 测试帕累托前沿计算 ===")
    
    with patch('bayesian_optimizer.TaskEvaluator') as mock_task_evaluator:
        mock_instance = Mock()
        mock_instance.evaluate_parameters.side_effect = mock_evaluate_parameters
        mock_instance.extract_multi_objective_values.side_effect = lambda params, objectives: {
            obj: mock_evaluate_parameters(params)[obj] for obj in objectives
        }
        mock_task_evaluator.return_value = mock_instance
        
        try:
            optimizer = create_multi_objective_optimizer(
                task_type='LDA',
                objectives=['AUROC', 'AUPRC'],
                n_initial_points=3,
                random_state=42
            )
            
            history = optimizer.optimize(n_iterations=6, checkpoint_freq=3)
            
            if hasattr(history, 'pareto_front'):
                pareto_front = history.pareto_front
                print(f"帕累托前沿大小: {len(pareto_front)}")
                
                if pareto_front:
                    print("帕累托前沿解:")
                    for i, result in enumerate(pareto_front):
                        if result.objective_values:
                            obj_vals = result.objective_values
                            print(f"  解 {i+1}: AUROC={obj_vals['AUROC']:.4f}, "
                                  f"AUPRC={obj_vals['AUPRC']:.4f}, "
                                  f"帕累托最优: {result.is_pareto_optimal}")
                    
                    # 验证帕累托最优性
                    all_pareto_optimal = all(result.is_pareto_optimal for result in pareto_front)
                    if all_pareto_optimal:
                        print("✓ 所有帕累托前沿解都标记为帕累托最优")
                    else:
                        print("✗ 帕累托前沿中有非帕累托最优解")
                        return False
                
                return True
            else:
                print("✗ 历史记录中没有帕累托前沿信息")
                return False
                
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主测试函数"""
    print("多目标优化功能测试")
    print("=" * 50)
    
    tests = [
        test_multi_objective_basic,
        test_weighted_objective_calculation,
        test_pareto_front_computation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("测试失败，停止执行")
            break
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有多目标优化功能测试通过！")
        return True
    else:
        print("✗ 部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)