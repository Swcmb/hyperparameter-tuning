#!/usr/bin/env python3
"""
多目标贝叶斯优化示例

演示如何使用扩展的贝叶斯优化器进行多目标超参数优化。
支持帕累托前沿计算、加权目标函数和超体积指标。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayesian_optimizer import create_multi_objective_optimizer, create_bayesian_optimizer
from autodl_core import OptimizationResult


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('multi_objective_optimization.log')
        ]
    )


def run_single_objective_baseline():
    """运行单目标优化作为基线"""
    print("=== 单目标优化基线 ===")
    
    optimizer = create_bayesian_optimizer(
        task_type="LDA",
        acquisition_function_type="EI",
        n_initial_points=5,
        random_state=42
    )
    
    print(f"目标函数: {optimizer.objectives}")
    print(f"是否多目标: {optimizer.is_multi_objective}")
    
    # 运行优化
    history = optimizer.optimize(n_iterations=10, checkpoint_freq=5)
    
    print(f"单目标优化结果:")
    print(f"  最佳AUROC: {history.get_best_objective_value():.4f}")
    print(f"  总迭代次数: {history.total_iterations}")
    print(f"  总耗时: {history.total_time:.1f}秒")
    
    return history


def run_multi_objective_optimization():
    """运行多目标优化"""
    print("\n=== 多目标优化 ===")
    
    # 定义多个目标函数
    objectives = ['AUROC', 'AUPRC', 'F1']
    
    # 创建多目标优化器
    optimizer = create_multi_objective_optimizer(
        task_type="LDA",
        objectives=objectives,
        objective_weights={'AUROC': 0.5, 'AUPRC': 0.3, 'F1': 0.2},
        n_initial_points=5,
        random_state=42
    )
    
    print(f"目标函数: {optimizer.objectives}")
    print(f"目标权重: {optimizer.objective_weights}")
    print(f"是否多目标: {optimizer.is_multi_objective}")
    
    # 运行优化
    history = optimizer.optimize(n_iterations=15, checkpoint_freq=5)
    
    print(f"\n多目标优化结果:")
    print(f"  总迭代次数: {history.total_iterations}")
    print(f"  帕累托前沿大小: {len(history.pareto_front)}")
    print(f"  总耗时: {history.total_time:.1f}秒")
    
    # 显示帕累托前沿
    if history.pareto_front:
        print(f"\n帕累托前沿解 (前5个):")
        for i, result in enumerate(history.pareto_front[:5]):
            obj_vals = result.objective_values
            print(f"  解 {i+1}: AUROC={obj_vals['AUROC']:.4f}, "
                  f"AUPRC={obj_vals['AUPRC']:.4f}, F1={obj_vals['F1']:.4f}")
    
    # 计算帕累托前沿统计信息
    pareto_metrics = history.get_pareto_front_metrics()
    print(f"\n帕累托前沿统计:")
    for key, value in pareto_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 计算超体积
    try:
        hypervolume = optimizer.compute_hypervolume()
        print(f"  超体积: {hypervolume:.6f}")
    except Exception as e:
        print(f"  超体积计算失败: {e}")
    
    return history, optimizer


def test_objective_weights():
    """测试不同目标权重的影响"""
    print("\n=== 测试不同目标权重 ===")
    
    weight_configs = [
        {'AUROC': 1.0, 'AUPRC': 0.0, 'F1': 0.0},  # 只关注AUROC
        {'AUROC': 0.0, 'AUPRC': 1.0, 'F1': 0.0},  # 只关注AUPRC
        {'AUROC': 0.33, 'AUPRC': 0.33, 'F1': 0.34},  # 均等权重
        {'AUROC': 0.6, 'AUPRC': 0.3, 'F1': 0.1}   # AUROC优先
    ]
    
    results = []
    
    for i, weights in enumerate(weight_configs):
        print(f"\n配置 {i+1}: {weights}")
        
        optimizer = create_multi_objective_optimizer(
            task_type="LDA",
            objectives=['AUROC', 'AUPRC', 'F1'],
            objective_weights=weights,
            n_initial_points=3,
            random_state=42 + i
        )
        
        history = optimizer.optimize(n_iterations=8, checkpoint_freq=4)
        
        best_result = history.best_result
        if best_result and best_result.objective_values:
            obj_vals = best_result.objective_values
            print(f"  最佳解: AUROC={obj_vals['AUROC']:.4f}, "
                  f"AUPRC={obj_vals['AUPRC']:.4f}, F1={obj_vals['F1']:.4f}")
            print(f"  加权目标值: {history.get_weighted_objective_value(best_result):.4f}")
        
        results.append((weights, history))
    
    return results


def visualize_pareto_front(history, optimizer, save_path: str = "pareto_front.png"):
    """可视化帕累托前沿（2D或3D）"""
    if not history.pareto_front:
        print("没有帕累托前沿数据可视化")
        return
    
    objectives = optimizer.objectives
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 提取所有解的目标值
        all_auroc = [r.objective_values['AUROC'] for r in history.results if r.objective_values]
        all_auprc = [r.objective_values['AUPRC'] for r in history.results if r.objective_values]
        all_f1 = [r.objective_values['F1'] for r in history.results if r.objective_values]
        
        # 提取帕累托前沿的目标值
        pareto_auroc = [r.objective_values['AUROC'] for r in history.pareto_front]
        pareto_auprc = [r.objective_values['AUPRC'] for r in history.pareto_front]
        pareto_f1 = [r.objective_values['F1'] for r in history.pareto_front]
        
        if len(objectives) == 2:
            # 2D可视化
            plt.figure(figsize=(10, 6))
            plt.scatter(all_auroc, all_auprc, alpha=0.6, label='所有解', color='lightblue')
            plt.scatter(pareto_auroc, pareto_auprc, alpha=0.8, label='帕累托前沿', 
                       color='red', s=100, edgecolors='black')
            plt.xlabel('AUROC')
            plt.ylabel('AUPRC')
            plt.title('帕累托前沿可视化 (2D)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif len(objectives) >= 3:
            # 3D可视化
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(all_auroc, all_auprc, all_f1, alpha=0.6, 
                      label='所有解', color='lightblue', s=30)
            ax.scatter(pareto_auroc, pareto_auprc, pareto_f1, alpha=0.8, 
                      label='帕累托前沿', color='red', s=100, edgecolors='black')
            
            ax.set_xlabel('AUROC')
            ax.set_ylabel('AUPRC')
            ax.set_zlabel('F1')
            ax.set_title('帕累托前沿可视化 (3D)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"帕累托前沿图已保存到: {save_path}")
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化失败: {e}")


def analyze_convergence(history):
    """分析收敛性"""
    print("\n=== 收敛性分析 ===")
    
    if not history.results:
        print("没有优化历史数据")
        return
    
    # 分析单目标收敛
    convergence_curve = history.get_convergence_curve()
    print(f"收敛曲线长度: {len(convergence_curve)}")
    print(f"初始最佳值: {convergence_curve[0]:.4f}")
    print(f"最终最佳值: {convergence_curve[-1]:.4f}")
    print(f"总改进: {convergence_curve[-1] - convergence_curve[0]:.4f}")
    
    # 分析帕累托前沿演化
    if hasattr(history, 'pareto_front') and history.pareto_front:
        pareto_iterations = [r.iteration for r in history.pareto_front]
        print(f"帕累托前沿解的发现迭代: {sorted(pareto_iterations)}")
        
        # 计算帕累托前沿的多样性
        if len(history.pareto_front) > 1:
            auroc_range = max(r.objective_values['AUROC'] for r in history.pareto_front) - \
                         min(r.objective_values['AUROC'] for r in history.pareto_front)
            auprc_range = max(r.objective_values['AUPRC'] for r in history.pareto_front) - \
                         min(r.objective_values['AUPRC'] for r in history.pareto_front)
            f1_range = max(r.objective_values['F1'] for r in history.pareto_front) - \
                      min(r.objective_values['F1'] for r in history.pareto_front)
            
            print(f"帕累托前沿多样性:")
            print(f"  AUROC范围: {auroc_range:.4f}")
            print(f"  AUPRC范围: {auprc_range:.4f}")
            print(f"  F1范围: {f1_range:.4f}")


def main():
    """主函数"""
    print("多目标贝叶斯优化示例")
    print("=" * 50)
    
    # 设置日志
    setup_logging()
    
    try:
        # 1. 运行单目标基线
        single_history = run_single_objective_baseline()
        
        # 2. 运行多目标优化
        multi_history, multi_optimizer = run_multi_objective_optimization()
        
        # 3. 测试不同权重配置
        weight_results = test_objective_weights()
        
        # 4. 可视化帕累托前沿
        visualize_pareto_front(multi_history, multi_optimizer)
        
        # 5. 分析收敛性
        analyze_convergence(multi_history)
        
        # 6. 比较单目标和多目标结果
        print("\n=== 单目标 vs 多目标比较 ===")
        print(f"单目标最佳AUROC: {single_history.get_best_objective_value():.4f}")
        
        if multi_history.pareto_front:
            best_auroc_in_pareto = max(r.objective_values['AUROC'] for r in multi_history.pareto_front)
            print(f"多目标帕累托前沿最佳AUROC: {best_auroc_in_pareto:.4f}")
            
            # 找到平衡解（所有目标都较好的解）
            balanced_scores = []
            for result in multi_history.pareto_front:
                obj_vals = result.objective_values
                # 计算几何平均数作为平衡指标
                balanced_score = (obj_vals['AUROC'] * obj_vals['AUPRC'] * obj_vals['F1']) ** (1/3)
                balanced_scores.append((balanced_score, result))
            
            if balanced_scores:
                best_balanced = max(balanced_scores, key=lambda x: x[0])
                balanced_result = best_balanced[1]
                obj_vals = balanced_result.objective_values
                print(f"最平衡解: AUROC={obj_vals['AUROC']:.4f}, "
                      f"AUPRC={obj_vals['AUPRC']:.4f}, F1={obj_vals['F1']:.4f}")
                print(f"平衡分数: {best_balanced[0]:.4f}")
        
        print("\n多目标优化示例完成!")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()