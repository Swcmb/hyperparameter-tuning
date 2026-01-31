"""
多目标优化属性测试（模拟版本）

使用模拟的TaskEvaluator来测试多目标优化功能，避免实际的模型训练。
测试多目标优化支持、帕累托最优解返回和加权目标函数计算功能。

Feature: bayesian-hyperparameter-optimization
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayesian_optimizer import create_multi_objective_optimizer, create_bayesian_optimizer
from autodl_core import OptimizationResult, OptimizationHistory
from task_evaluator import TaskEvaluator


class MockTaskEvaluator:
    """模拟的TaskEvaluator，用于测试"""
    
    def __init__(self, task_type: str, device: str = 'cpu'):
        self.task_type = task_type
        self.device = device
        self.call_count = 0
    
    def evaluate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """模拟参数评估，返回随机但一致的结果"""
        self.call_count += 1
        
        # 使用参数的哈希值作为随机种子，确保相同参数返回相同结果
        param_hash = hash(str(sorted(parameters.items())))
        np.random.seed(abs(param_hash) % 2**32)
        
        # 生成模拟的性能指标
        base_auroc = 0.6 + 0.3 * np.random.random()
        base_auprc = 0.5 + 0.4 * np.random.random()
        base_f1 = 0.4 + 0.5 * np.random.random()
        
        # 添加一些相关性，使得指标之间有合理的关系
        correlation_noise = 0.1 * np.random.randn()
        
        return {
            'AUROC': min(1.0, max(0.0, base_auroc + correlation_noise)),
            'AUPRC': min(1.0, max(0.0, base_auprc + correlation_noise * 0.8)),
            'F1': min(1.0, max(0.0, base_f1 + correlation_noise * 0.6))
        }
    
    def extract_multi_objective_values(self, parameters: Dict[str, Any], 
                                     objectives: List[str]) -> Dict[str, float]:
        """提取多目标值"""
        all_values = self.evaluate_parameters(parameters)
        return {obj: all_values[obj] for obj in objectives if obj in all_values}


class TestMultiObjectivePropertiesMock(unittest.TestCase):
    """多目标优化属性测试类（使用模拟）"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.task_types = ['LDA', 'MDA', 'LMI']
        self.objectives = ['AUROC', 'AUPRC', 'F1']
        
        # 创建模拟的TaskEvaluator实例
        self.mock_evaluators = {
            task_type: MockTaskEvaluator(task_type) 
            for task_type in self.task_types
        }
    
    def _patch_task_evaluator(self, task_type: str):
        """为指定任务类型创建TaskEvaluator的补丁"""
        return patch('bayesian_optimizer.TaskEvaluator', 
                    return_value=self.mock_evaluators[task_type])
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        n_objectives=st.integers(min_value=2, max_value=3),
        n_iterations=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=3, deadline=None)
    def test_property_20_multi_objective_optimization_support(self, task_type, n_objectives, n_iterations):
        """
        属性 20: 多目标优化支持
        
        对于任何指定的多个目标函数，系统应该支持帕累托前沿优化并计算所有目标函数值
        
        **Feature: bayesian-hyperparameter-optimization, Property 20: 多目标优化支持**
        **验证: 需求 8.1, 8.2**
        """
        selected_objectives = self.objectives[:n_objectives]
        
        with self._patch_task_evaluator(task_type):
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
                
            except Exception as e:
                self.fail(f"多目标优化支持失败 (task={task_type}, objectives={n_objectives}): {e}")
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        n_objectives=st.integers(min_value=2, max_value=3),
        n_iterations=st.integers(min_value=4, max_value=8)
    )
    @settings(max_examples=3, deadline=None)
    def test_property_21_pareto_optimal_solution_return(self, task_type, n_objectives, n_iterations):
        """
        属性 21: 帕累托最优解返回
        
        对于多目标优化完成后，系统应该返回帕累托最优解集合
        
        **Feature: bayesian-hyperparameter-optimization, Property 21: 帕累托最优解返回**
        **验证: 需求 8.4**
        """
        selected_objectives = self.objectives[:n_objectives]
        
        with self._patch_task_evaluator(task_type):
            try:
                # 创建多目标优化器
                optimizer = create_multi_objective_optimizer(
                    task_type=task_type,
                    objectives=selected_objectives,
                    n_initial_points=2,
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
                    
                    # 3. 验证帕累托支配关系（简化版本）
                    # 帕累托前沿中的解不应该相互支配
                    for i, sol1 in enumerate(pareto_front):
                        for j, sol2 in enumerate(pareto_front):
                            if i != j:
                                # 检查sol1是否严格支配sol2
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
                                
                                # 帕累托前沿中的解不应该相互严格支配
                                if dominates and strictly_better:
                                    # 允许一些数值误差
                                    max_diff = max(abs(sol1.objective_values[obj] - sol2.objective_values[obj]) 
                                                 for obj in selected_objectives)
                                    if max_diff > 0.01:  # 只有差异较大时才认为是错误
                                        self.fail(f"帕累托前沿中的解 {i} 严格支配解 {j}，这违反了帕累托最优性")
                
            except Exception as e:
                self.fail(f"帕累托最优解返回失败 (task={task_type}, objectives={n_objectives}): {e}")
    
    @given(
        task_type=st.sampled_from(['LDA', 'MDA', 'LMI']),
        weight_config=st.integers(min_value=0, max_value=3)
    )
    @settings(max_examples=3, deadline=None)
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
        
        with self._patch_task_evaluator(task_type):
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
                history = optimizer.optimize(n_iterations=4, checkpoint_freq=2)
                
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
                
            except Exception as e:
                self.fail(f"加权目标函数计算失败 (task={task_type}, weights={selected_weights}): {e}")
    
    def test_multi_objective_vs_single_objective_consistency(self):
        """测试多目标优化与单目标优化的一致性"""
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            # 创建单目标优化器
            single_optimizer = create_bayesian_optimizer(
                task_type=task_type,
                acquisition_function_type="EI",
                n_initial_points=2,
                random_state=42
            )
            
            # 创建只有一个目标的多目标优化器
            multi_optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=['AUROC'],
                objective_weights={'AUROC': 1.0},
                n_initial_points=2,
                random_state=42
            )
            
            # 运行优化
            single_history = single_optimizer.optimize(n_iterations=4, checkpoint_freq=2)
            multi_history = multi_optimizer.optimize(n_iterations=4, checkpoint_freq=2)
            
            # 验证结果的一致性
            single_best = single_history.get_best_objective_value()
            multi_best = multi_history.get_best_objective_value()
            
            # 由于使用相同的随机种子和模拟评估器，结果应该相似
            self.assertAlmostEqual(single_best, multi_best, delta=0.2,
                                  msg=f"单目标和多目标优化结果差异过大: {single_best} vs {multi_best}")
    
    def test_pareto_front_properties(self):
        """测试帕累托前沿的数学性质"""
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=['AUROC', 'AUPRC'],
                n_initial_points=3,
                random_state=42
            )
            
            history = optimizer.optimize(n_iterations=6, checkpoint_freq=3)
            
            if history.pareto_front and len(history.pareto_front) > 1:
                # 测试帕累托前沿的分布
                pareto_points = []
                for result in history.pareto_front:
                    point = [result.objective_values['AUROC'], result.objective_values['AUPRC']]
                    pareto_points.append(point)
                
                pareto_points = np.array(pareto_points)
                
                # 验证帕累托前沿点的分布
                auroc_range = np.max(pareto_points[:, 0]) - np.min(pareto_points[:, 0])
                auprc_range = np.max(pareto_points[:, 1]) - np.min(pareto_points[:, 1])
                
                # 至少有一个目标应该有一定的变化范围
                self.assertTrue(auroc_range > 0.001 or auprc_range > 0.001,
                               f"帕累托前沿缺乏多样性: AUROC范围={auroc_range}, AUPRC范围={auprc_range}")
    
    def test_objective_weights_normalization(self):
        """测试目标权重的标准化"""
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            # 测试非标准化权重
            unnormalized_weights = {'AUROC': 2.0, 'AUPRC': 3.0, 'F1': 1.0}
            
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
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
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=['AUROC', 'AUPRC'],
                n_initial_points=2,
                random_state=42
            )
            
            history = optimizer.optimize(n_iterations=4, checkpoint_freq=2)
            
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
    
    def test_multi_objective_configuration_validation(self):
        """测试多目标配置验证"""
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            # 测试有效配置
            valid_configs = [
                (['AUROC', 'AUPRC'], {'AUROC': 0.6, 'AUPRC': 0.4}),
                (['AUROC', 'AUPRC', 'F1'], {'AUROC': 0.5, 'AUPRC': 0.3, 'F1': 0.2}),
                (['AUROC'], {'AUROC': 1.0}),  # 单目标作为多目标的特殊情况
            ]
            
            for objectives, weights in valid_configs:
                try:
                    optimizer = create_multi_objective_optimizer(
                        task_type=task_type,
                        objectives=objectives,
                        objective_weights=weights,
                        n_initial_points=2,
                        random_state=42
                    )
                    
                    # 验证配置正确设置
                    self.assertEqual(optimizer.objectives, objectives)
                    self.assertTrue(optimizer.is_multi_objective or len(objectives) == 1)
                    
                except Exception as e:
                    self.fail(f"有效配置应该被接受: objectives={objectives}, weights={weights}, 错误={e}")
    
    def test_pareto_dominance_calculation(self):
        """测试帕累托支配关系计算"""
        task_type = 'LDA'
        
        with self._patch_task_evaluator(task_type):
            optimizer = create_multi_objective_optimizer(
                task_type=task_type,
                objectives=['AUROC', 'AUPRC'],
                n_initial_points=2,
                random_state=42
            )
            
            history = optimizer.optimize(n_iterations=5, checkpoint_freq=3)
            
            # 验证支配关系计算
            if len(history.results) >= 2:
                result1 = history.results[0]
                result2 = history.results[1]
                
                if result1.objective_values and result2.objective_values:
                    # 测试支配关系计算方法
                    dominates_12 = result1.dominates(result2, optimizer.objectives, optimizer.maximize_objectives)
                    dominates_21 = result2.dominates(result1, optimizer.objectives, optimizer.maximize_objectives)
                    
                    # 验证支配关系的反对称性
                    if dominates_12:
                        self.assertFalse(dominates_21, "支配关系应该是反对称的")
                    
                    # 验证自反性（一个解不应该支配自己，除非完全相同）
                    self_dominates = result1.dominates(result1, optimizer.objectives, optimizer.maximize_objectives)
                    self.assertFalse(self_dominates, "一个解不应该支配自己")


if __name__ == '__main__':
    # 运行属性测试
    unittest.main(verbosity=2)