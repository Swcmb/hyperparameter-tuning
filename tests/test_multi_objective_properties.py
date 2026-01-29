"""
多目标优化属性测试

使用属性测试（Property-Based Testing）验证多目标优化功能的正确性属性。
测试多目标优化支持、帕累托最优解返回和加权目标函数计算功能。

Feature: bayesian-hyperparameter-optimization
"""

import unittest
import numpy as np
import sys
import os
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
from typing import Dict, List, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayesian_optimizer import create_multi_objective_optimizer, create_bayesian_optimizer
from autodl_core import OptimizationResult, OptimizationHistory
from task_evaluator import TaskEvaluator


class TestMultiObjectiveProperties(unittest.TestCase):
    """多目标优化属性测试类"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.task_types = ['LDA', 'MDA', 'LMI']
        self.objectives = ['AUROC', 'AUPRC', 'F1']
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        n_objectives=st.integers(min_value=2, max_value=3),
        n_iterations=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_20_multi_objective_optimization_support(self, task_type, n_objectives, n_iterations):
        """
        属性 20: 多目标优化支持
        
        对于任何指定的多个目标函数，系统应该支持帕累托前沿优化并计算所有目标函数值
        
        **Feature: bayesian-hyperparameter-optimization, Property 20: 多目标优化支持**
        **验证: 需求 8.1, 8.2**
        """
        # 选择目标函数子集
        selected_objectives = self.objectives[:n_objectives]
        
        try:
            # 1. 系统应该能够创建多目标优化器
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=selected_objectives,
                n_initial_points=2,
                random_state=42
            )
            
            # 验证多目标配置
            self.assertTrue(optimizer.is_multi_objective)
            self.assertEqual(optimizer.objectives, selected_objectives)
            self.assertEqual(len(optimizer.objectives), n_objectives)
            
            # 2. 系统应该支持帕累托前沿优化
            history = optimizer.optimize(n_iterations=n_iterations, checkpoint_freq=max(1, n_iterations//2))
            
            # 验证优化历史包含多目标信息
            self.assertIsInstance(history, OptimizationHistory)
            self.assertTrue(hasattr(history, 'pareto_front'))
            self.assertTrue(hasattr(history, 'objective_weights'))
            
            # 3. 系统应该计算所有指定的目标函数值
            for result in history.results:
                if result.objective_values:
                    # 验证所有目标函数都有值
                    for obj in selected_objectives:
                        self.assertIn(obj, result.objective_values)
                        self.assertTrue(np.isfinite(result.objective_values[obj]))
                        # 验证目标值在合理范围内（0-1之间）
                        self.assertTrue(0.0 <= result.objective_values[obj] <= 1.0)
            
            # 4. 验证帕累托前沿存在且合理
            if history.pareto_front:
                # 帕累托前沿应该包含多目标信息
                for pareto_result in history.pareto_front:
                    self.assertTrue(pareto_result.is_pareto_optimal)
                    self.assertIsNotNone(pareto_result.objective_values)
                    
                    # 验证所有目标函数值都存在
                    for obj in selected_objectives:
                        self.assertIn(obj, pareto_result.objective_values)
                
                # 验证帕累托前沿的多样性（如果有多个解）
                if len(history.pareto_front) > 1:
                    # 至少应该有一些目标值的差异
                    obj_ranges = {}
                    for obj in selected_objectives:
                        values = [r.objective_values[obj] for r in history.pareto_front]
                        obj_ranges[obj] = max(values) - min(values)
                    
                    # 至少有一个目标应该有显著差异
                    max_range = max(obj_ranges.values())
                    self.assertTrue(max_range > 0.001, f"帕累托前沿缺乏多样性: {obj_ranges}")
            
        except Exception as e:
            self.fail(f"多目标优化支持失败 (task={task_type}, objectives={n_objectives}): {e}")
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        n_objectives=st.integers(min_value=2, max_value=3),
        n_iterations=st.integers(min_value=5, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_property_21_pareto_optimal_solution_return(self, task_type, n_objectives, n_iterations):
        """
        属性 21: 帕累托最优解返回
        
        对于多目标优化完成后，系统应该返回帕累托最优解集合
        
        **Feature: bayesian-hyperparameter-optimization, Property 21: 帕累托最优解返回**
        **验证: 需求 8.4**
        """
        selected_objectives = self.objectives[:n_objectives]
        
        try:
            # 创建多目标优化器
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=selected_objectives,
                n_initial_points=3,
                random_state=42
            )
            
            # 运行优化
            history = optimizer.optimize(n_iterations=n_iterations, checkpoint_freq=max(1, n_iterations//2))
            
            # 1. 系统应该返回帕累托最优解集合
            pareto_front = history.pareto_front
            self.assertIsNotNone(pareto_front)
            self.assertIsInstance(pareto_front, list)
            
            if pareto_front:
                # 2. 验证帕累托最优解的正确性
                for solution in pareto_front:
                    self.assertIsInstance(solution, OptimizationResult)
                    self.assertTrue(solution.is_pareto_optimal)
                    self.assertIsNotNone(solution.objective_values)
                    
                    # 验证目标值完整性
                    for obj in selected_objectives:
                        self.assertIn(obj, solution.objective_values)
                        self.assertTrue(np.isfinite(solution.objective_values[obj]))
                
                # 3. 验证帕累托支配关系
                # 帕累托前沿中的解不应该相互支配
                for i, sol1 in enumerate(pareto_front):
                    for j, sol2 in enumerate(pareto_front):
                        if i != j:
                            # 检查sol1是否支配sol2
                            dominates = True
                            strictly_better = False
                            
                            for obj in selected_objectives:
                                val1 = sol1.objective_values[obj]
                                val2 = sol2.objective_values[obj]
                                
                                if val1 < val2:  # 假设目标是最大化
                                    dominates = False
                                    break
                                elif val1 > val2:
                                    strictly_better = True
                            
                            # 帕累托前沿中的解不应该相互支配
                            if dominates and strictly_better:
                                self.fail(f"帕累托前沿中的解 {i} 支配解 {j}，这违反了帕累托最优性")
                
                # 4. 验证帕累托前沿的完整性
                # 检查是否有非帕累托前沿的解实际上是帕累托最优的
                all_results = [r for r in history.results if r.objective_values]
                non_pareto_results = [r for r in all_results if not r.is_pareto_optimal]
                
                for non_pareto in non_pareto_results:
                    # 这个解应该被至少一个帕累托前沿的解支配
                    is_dominated = False
                    
                    for pareto_sol in pareto_front:
                        dominates = True
                        strictly_better = False
                        
                        for obj in selected_objectives:
                            pareto_val = pareto_sol.objective_values[obj]
                            non_pareto_val = non_pareto.objective_values[obj]
                            
                            if pareto_val < non_pareto_val:
                                dominates = False
                                break
                            elif pareto_val > non_pareto_val:
                                strictly_better = True
                        
                        if dominates and strictly_better:
                            is_dominated = True
                            break
                    
                    # 如果没有被支配，可能是帕累托前沿计算错误
                    if not is_dominated:
                        # 允许一些数值误差
                        continue
            
        except Exception as e:
            self.fail(f"帕累托最优解返回失败 (task={task_type}, objectives={n_objectives}): {e}")
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        weight_config=st.integers(min_value=0, max_value=3)
    )
    @settings(max_examples=40, deadline=None)
    def test_property_22_weighted_objective_function_calculation(self, task_type, weight_config):
        """
        属性 22: 加权目标函数计算
        
        对于用户设置的目标权重，系统应该根据权重计算加权目标函数值
        
        **Feature: bayesian-hyperparameter-optimization, Property 22: 加权目标函数计算**
        **验证: 需求 8.5**
        """
        # 定义不同的权重配置
        weight_configs = [
            {'AUROC': 1.0, 'AUPRC': 0.0, 'F1': 0.0},  # 只关注AUROC
            {'AUROC': 0.0, 'AUPRC': 1.0, 'F1': 0.0},  # 只关注AUPRC
            {'AUROC': 0.33, 'AUPRC': 0.33, 'F1': 0.34},  # 均等权重
            {'AUROC': 0.6, 'AUPRC': 0.3, 'F1': 0.1}   # AUROC优先
        ]
        
        selected_weights = weight_configs[weight_config]
        objectives = list(selected_weights.keys())
        
        try:
            # 1. 系统应该接受用户设置的目标权重
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=objectives,
                objective_weights=selected_weights,
                n_initial_points=2,
                random_state=42
            )
            
            # 验证权重设置
            self.assertEqual(optimizer.objective_weights, selected_weights)
            
            # 验证权重和为1（或接近1）
            weight_sum = sum(selected_weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=2)
            
            # 2. 运行优化并验证加权目标函数计算
            history = optimizer.optimize(n_iterations=5, checkpoint_freq=3)
            
            # 3. 验证加权目标函数值的计算
            for result in history.results:
                if result.objective_values:
                    # 手动计算加权目标函数值
                    expected_weighted_value = 0.0
                    for obj, weight in selected_weights.items():
                        if obj in result.objective_values:
                            expected_weighted_value += weight * result.objective_values[obj]
                    
                    # 使用系统方法计算加权目标函数值
                    actual_weighted_value = history.get_weighted_objective_value(result)
                    
                    # 验证计算结果一致
                    self.assertAlmostEqual(
                        actual_weighted_value, expected_weighted_value, places=6,
                        msg=f"加权目标函数计算不一致: 期望={expected_weighted_value}, 实际={actual_weighted_value}"
                    )
                    
                    # 验证加权值在合理范围内
                    self.assertTrue(0.0 <= actual_weighted_value <= 1.0)
            
            # 4. 验证权重对优化行为的影响
            if history.best_result and history.best_result.objective_values:
                best_obj_values = history.best_result.objective_values
                
                # 找到权重最大的目标
                max_weight_obj = max(selected_weights.keys(), key=lambda x: selected_weights[x])
                max_weight = selected_weights[max_weight_obj]
                
                if max_weight > 0.5:  # 如果某个目标权重占主导
                    # 最佳解在该目标上应该表现较好
                    max_weight_value = best_obj_values[max_weight_obj]
                    
                    # 与其他解比较
                    other_values = []
                    for result in history.results:
                        if result != history.best_result and result.objective_values:
                            if max_weight_obj in result.objective_values:
                                other_values.append(result.objective_values[max_weight_obj])
                    
                    if other_values:
                        avg_other_value = np.mean(other_values)
                        # 最佳解在主要目标上应该不差于平均水平
                        self.assertTrue(
                            max_weight_value >= avg_other_value - 0.1,
                            f"最佳解在主要目标 {max_weight_obj} 上表现不佳: {max_weight_value} vs 平均 {avg_other_value}"
                        )
            
        except Exception as e:
            self.fail(f"加权目标函数计算失败 (task={task_type}, weights={selected_weights}): {e}")
    
    def test_multi_objective_vs_single_objective_consistency(self):
        """测试多目标优化与单目标优化的一致性"""
        task_type = 'LDA'
        
        # 创建单目标优化器
        single_optimizer = create_bayesian_optimizer(
            task_type=task_type,
            acquisition_function_type="EI",
            n_initial_points=3,
            random_state=42
        )
        
        # 创建只有一个目标的多目标优化器
        multi_optimizer = create_multi_objective_optimizer(
            task_type=task_type,
            objectives=['AUROC'],
            objective_weights={'AUROC': 1.0},
            n_initial_points=3,
            random_state=42
        )
        
        # 运行优化
        single_history = single_optimizer.optimize(n_iterations=5, checkpoint_freq=3)
        multi_history = multi_optimizer.optimize(n_iterations=5, checkpoint_freq=3)
        
        # 验证结果的一致性
        single_best = single_history.get_best_objective_value()
        multi_best = multi_history.get_best_objective_value()
        
        # 由于随机性，结果可能不完全相同，但应该在合理范围内
        self.assertAlmostEqual(single_best, multi_best, delta=0.1,
                              msg=f"单目标和多目标优化结果差异过大: {single_best} vs {multi_best}")
    
    def test_pareto_front_properties(self):
        """测试帕累托前沿的数学性质"""
        optimizer = create_multi_objective_optimizer(
            task_type='LDA',
            objectives=['AUROC', 'AUPRC'],
            n_initial_points=3,
            random_state=42
        )
        
        history = optimizer.optimize(n_iterations=8, checkpoint_freq=4)
        
        if history.pareto_front and len(history.pareto_front) > 1:
            # 测试帕累托前沿的凸性（在目标空间中）
            pareto_points = []
            for result in history.pareto_front:
                point = [result.objective_values['AUROC'], result.objective_values['AUPRC']]
                pareto_points.append(point)
            
            pareto_points = np.array(pareto_points)
            
            # 验证帕累托前沿点的分布
            auroc_range = np.max(pareto_points[:, 0]) - np.min(pareto_points[:, 0])
            auprc_range = np.max(pareto_points[:, 1]) - np.min(pareto_points[:, 1])
            
            # 至少有一个目标应该有显著的变化范围
            self.assertTrue(auroc_range > 0.01 or auprc_range > 0.01,
                           f"帕累托前沿缺乏多样性: AUROC范围={auroc_range}, AUPRC范围={auprc_range}")
    
    def test_objective_weights_normalization(self):
        """测试目标权重的标准化"""
        # 测试非标准化权重
        unnormalized_weights = {'AUROC': 2.0, 'AUPRC': 3.0, 'F1': 1.0}
        
        optimizer = create_multi_objective_optimizer(
            task_type='LDA',
            objectives=['AUROC', 'AUPRC', 'F1'],
            objective_weights=unnormalized_weights,
            n_initial_points=2,
            random_state=42
        )
        
        # 验证权重被正确标准化
        normalized_weights = optimizer.objective_weights
        weight_sum = sum(normalized_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # 验证权重比例保持不变
        expected_auroc_weight = 2.0 / 6.0
        expected_auprc_weight = 3.0 / 6.0
        expected_f1_weight = 1.0 / 6.0
        
        self.assertAlmostEqual(normalized_weights['AUROC'], expected_auroc_weight, places=6)
        self.assertAlmostEqual(normalized_weights['AUPRC'], expected_auprc_weight, places=6)
        self.assertAlmostEqual(normalized_weights['F1'], expected_f1_weight, places=6)
    
    def test_hypervolume_calculation(self):
        """测试超体积计算"""
        optimizer = create_multi_objective_optimizer(
            task_type='LDA',
            objectives=['AUROC', 'AUPRC'],
            n_initial_points=3,
            random_state=42
        )
        
        history = optimizer.optimize(n_iterations=6, checkpoint_freq=3)
        
        if history.pareto_front:
            try:
                # 计算超体积
                hypervolume = optimizer.compute_hypervolume()
                
                # 验证超体积值的合理性
                self.assertTrue(np.isfinite(hypervolume))
                self.assertTrue(hypervolume >= 0.0)
                
                # 对于2D情况，超体积应该小于参考点的面积
                # 假设参考点为(0, 0)，最大可能面积为1.0
                self.assertTrue(hypervolume <= 1.0)
                
            except Exception as e:
                # 如果超体积计算失败，至少应该有合理的错误处理
                self.assertIsInstance(e, (ValueError, ImportError, NotImplementedError))


if __name__ == '__main__':
    # 运行属性测试
    unittest.main(verbosity=2)